"""雷虎練唱台：用顯示卡的工作（人聲分離、音高分析、歌詞辨識）在獨立的子程式裡執行（2.1.1）。

為什麼要分出來：
  PyTorch、ONNX Runtime、faster-whisper（CTranslate2）各有自己的顯示卡記憶體快取，
  torchcrepe 還會把模型存在全域變數裡，CUDA 執行環境本身也要 0.5～1 GB。
  這些在同一個程式裡就算清了也還不乾淨，只有程式結束才會全部還給系統。
  所以主程式（介面、播放、KEY 轉換）完全不載入 CUDA；要用顯示卡時才開子程式，
  子程式閒置 30 秒就結束，顯示卡記憶體回到開練唱台之前的水準，直播和 VTube Studio 不會被擠到。

做法：主程式用 subprocess 開 `python -c "import gpu_worker; gpu_worker.serve(...)"`，
  再用 multiprocessing.connection 互傳工作與結果。不用 multiprocessing.Process(spawn) 是因為
  spawn 會在子程式裡重新執行主程式的模組（介面、伺服器），又慢又容易出錯。
  按「停止」或「全部暫停」時直接結束子程式，顯示卡馬上釋放。
"""
from __future__ import annotations

import atexit
import gc
import os
import secrets
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

IDLE_EXIT = 30.0      # 子程式閒置幾秒後結束（把顯示卡記憶體全部還回去）
MAX_WORKERS = 3       # 最多同時幾個子程式（兩首同時分離＋一首分析旋律）
ROOT = Path(__file__).resolve().parent

BELOW_NORMAL, NORMAL = 0x00004000, 0x00000020   # Windows 程式優先順序


# ================================================================ 子程式

def _set_priority(low: bool) -> None:
    if sys.platform == "win32":
        try:
            import ctypes
            k = ctypes.windll.kernel32
            k.SetPriorityClass(k.GetCurrentProcess(), BELOW_NORMAL if low else NORMAL)
        except Exception:
            pass
    else:
        try:
            cur = os.getpriority(os.PRIO_PROCESS, 0)
            if low and cur < 10:
                os.setpriority(os.PRIO_PROCESS, 0, 10)
        except Exception:
            pass


def _parent_alive(pid: int) -> bool:
    if sys.platform == "win32":
        import ctypes
        k = ctypes.windll.kernel32
        h = k.OpenProcess(0x00100000, False, pid)   # SYNCHRONIZE
        if not h:
            return False
        try:
            return k.WaitForSingleObject(h, 0) != 0   # 0 = 已結束
        finally:
            k.CloseHandle(h)
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _watch_parent(pid: int) -> None:
    """主程式被強制關掉時，子程式也跟著結束（不要留著佔顯示卡）。"""
    while True:
        time.sleep(3)
        if not _parent_alive(pid):
            os._exit(0)


def _cleanup() -> None:
    """每個工作做完：清掉 torchcrepe 的全域模型、PyTorch 的暫存。分離器、ONNX session、Whisper 模型
    在各工作裡已經刪掉；CUDA 執行環境本身要等子程式結束才會還。"""
    tc = sys.modules.get("torchcrepe.infer") or sys.modules.get("torchcrepe")
    if tc is not None:
        try:
            import torchcrepe
            torchcrepe.infer.model = None
            torchcrepe.infer.capacity = None
        except Exception:
            pass
    gc.collect()
    if "torch" in sys.modules:
        try:
            import torch
            if torch.cuda.is_available() and torch.cuda.is_initialized():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()
        except Exception:
            pass


def t_probe(P, progress):
    ok = P.cuda_ok()
    info = {"ok": bool(ok), "note": P.CUDA_NOTE, "cc": None, "name": ""}
    if ok:
        try:
            import torch
            info["cc"] = torch.cuda.get_device_capability(0)[0]
            info["name"] = torch.cuda.get_device_name(0)
        except Exception:
            pass
    return info


def t_separate(P, progress, model: str, inp: str, out: str, overlap: int, autocast: bool,
               label: str, start: float, span: float) -> list[str]:
    try:
        import torch  # noqa: F401  先載入 PyTorch 的 CUDA 函式庫，ONNX 會沿用同一份
    except Exception:
        pass
    from audio_separator.separator import Separator
    sep = Separator(output_dir=out, model_file_dir=str(P.MODELS / "uvr"),
                    output_format="WAV", sample_rate=P.SR, use_autocast=autocast,
                    # 高品質模型（Roformer）每一段音訊預設重複算 4 次再平均；2 次的結果幾乎一樣，時間約省一半
                    mdxc_params={"segment_size": 256, "override_model_segment_size": False, "batch_size": None,
                                 "overlap": overlap, "pitch_shift": 0})
    try:
        sep.load_model(model_filename=model)
        with P._sep_progress(label, progress, start, span):
            outs = sep.separate(inp)
        return [str(p) for p in P._outputs(outs, Path(out))]
    finally:
        # 確實刪掉模型與 ONNX session（ONNX Runtime 的顯示卡記憶體池要 session 被刪掉才會還）
        mi = getattr(sep, "model_instance", None)
        if mi is not None:
            for a in ("model_run", "model", "model_instance"):
                try:
                    setattr(mi, a, None)
                except Exception:
                    pass
        sep.model_instance = None
        del sep, mi


def t_pitch(P, progress, y16, cfg: dict, strict: bool):
    return P._track_pitch_local(y16, cfg, strict)


def t_whisper(P, progress, y16, cfg: dict, want: str, per_song: bool) -> dict:
    progress("載入語音辨識模型（第一次會下載）", 0.7)
    model = P._whisper_model(cfg)
    try:
        progress("判斷歌詞語言", 0.74)
        lang, lang_note = P.pick_language(model, y16, want, per_song)
        progress("辨識歌詞", 0.75)
        stats: list = []
        try:
            segs, info = model.transcribe(
                y16, language=lang, vad_filter=True,
                vad_parameters={"min_silence_duration_ms": 500},
                word_timestamps=True, condition_on_previous_text=False, beam_size=5)
            words = P._collect_words(segs, len(y16) / P.PITCH_SR, progress, stats)
        except RuntimeError:
            # 例如有顯示卡但缺 CUDA 函式庫：改用 CPU 重跑
            traceback.print_exc()
            from faster_whisper import WhisperModel
            try:
                model.model.unload_model()
            except Exception:
                pass
            model = WhisperModel(cfg.get("whisper_model") or "large-v3-turbo", device="cpu",
                                 compute_type="int8", download_root=str(P.MODELS / "whisper"))
            segs, info = model.transcribe(y16, language=lang, vad_filter=True,
                                          word_timestamps=True, condition_on_previous_text=False)
            stats = []
            words = P._collect_words(segs, len(y16) / P.PITCH_SR, progress, stats)
        return {"words": [{"word": w.word, "start": w.start, "end": w.end,
                           "probability": getattr(w, "probability", None)} for w in words],
                "stats": stats, "language": getattr(info, "language", lang) or "",
                "lang": lang, "lang_note": lang_note}
    finally:
        try:
            model.model.unload_model()   # CTranslate2：模型與它的顯示卡記憶體一起放掉
        except Exception:
            pass
        del model


TASKS = {"probe": t_probe, "separate": t_separate, "pitch": t_pitch, "whisper": t_whisper}


def serve(address, authkey_hex: str, parent_pid: int) -> None:
    """子程式進入點。"""
    # CTranslate2 預設會把用過的顯示卡記憶體留著重複用；改成用完就還（要在載入 ctranslate2 之前設定）
    os.environ.setdefault("CT2_CUDA_ALLOCATOR", "cuda_malloc_async")
    from multiprocessing.connection import Client
    conn = Client(address, authkey=bytes.fromhex(authkey_hex))
    conn.send(("hello", os.getpid()))
    threading.Thread(target=_watch_parent, args=(parent_pid,), daemon=True).start()
    sys.path.insert(0, str(ROOT))
    import pipeline as P
    P.IN_WORKER = True
    print(f"[gpu] 子程式 {os.getpid()} 啟動", flush=True)
    while True:
        try:
            msg = conn.recv()
        except (EOFError, OSError):
            break
        if not msg or msg[0] == "exit":
            break
        _, name, args, kw, low = msg
        _set_priority(low)

        def progress(step, p):
            conn.send(("p", step, p))
        try:
            res = TASKS[name](P, progress, *args, **kw)
            out = ("ok", res)
        except BaseException as e:   # noqa: BLE001  任何錯誤都要回報給主程式
            traceback.print_exc()
            out = ("err", type(e).__name__, str(e))
        finally:
            _cleanup()
        try:
            conn.send(out)
        except Exception as e:
            conn.send(("err", type(e).__name__, f"結果傳不回主程式：{e}"))
    print(f"[gpu] 子程式 {os.getpid()} 閒置結束，顯示卡記憶體已釋放", flush=True)
    os._exit(0)


# ================================================================ 主程式這一端

class RemoteError(RuntimeError):
    def __init__(self, kind: str, msg: str):
        super().__init__(f"{kind}: {msg}" if kind not in ("RuntimeError",) else msg)
        self.kind, self.msg = kind, msg


class _Worker:
    def __init__(self, proc, conn, pid):
        self.proc, self.conn, self.pid = proc, conn, pid
        self.busy = True
        self.last = time.time()

    def alive(self) -> bool:
        return self.proc.poll() is None

    def kill(self) -> None:
        for fn in (lambda: os.kill(self.pid, 9) if sys.platform != "win32" else _win_kill(self.pid),
                   self.proc.kill):
            try:
                fn()
            except Exception:
                pass
        try:
            self.conn.close()
        except Exception:
            pass
        try:
            self.proc.wait(5)
        except Exception:
            pass


def _win_kill(pid: int) -> None:
    import ctypes
    k = ctypes.windll.kernel32
    h = k.OpenProcess(0x0001, False, pid)   # PROCESS_TERMINATE
    if h:
        k.TerminateProcess(h, 1)
        k.CloseHandle(h)


class Pool:
    def __init__(self):
        self.cv = threading.Condition()
        self.workers: list[_Worker] = []
        self.starting = 0
        self.listener = None
        self.hello: dict[int, object] = {}
        self.authkey = secrets.token_bytes(16)
        self.started = False

    # -- 子程式管理
    def _start(self) -> None:
        if self.started:
            return
        from multiprocessing.connection import Listener
        self.listener = Listener(authkey=self.authkey)
        threading.Thread(target=self._accept, daemon=True, name="gpu-accept").start()
        threading.Thread(target=self._reaper, daemon=True, name="gpu-reaper").start()
        atexit.register(self.shutdown)
        self.started = True

    def _accept(self) -> None:
        while True:
            try:
                c = self.listener.accept()
                msg = c.recv()
                if msg and msg[0] == "hello":
                    with self.cv:
                        self.hello[msg[1]] = c
                        self.cv.notify_all()
            except Exception:
                time.sleep(0.5)

    def _spawn(self) -> _Worker:
        cmd = [sys.executable, "-c",
               f"import gpu_worker; gpu_worker.serve({self.listener.address!r}, {self.authkey.hex()!r}, {os.getpid()})"]
        flags = 0x08000000 if sys.platform == "win32" else 0   # CREATE_NO_WINDOW
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
        proc = subprocess.Popen(cmd, cwd=str(ROOT), env=env, creationflags=flags)
        end = time.time() + 180
        with self.cv:
            while True:
                # Windows 的虛擬環境 python.exe 會再開一個真正的 Python，回報的 pid 是那一個
                got = None
                for pid, c in list(self.hello.items()):
                    if pid == proc.pid or _is_child(pid, proc.pid):
                        got = (pid, self.hello.pop(pid))
                        break
                if got is None and len(self.hello) == 1 and self.starting == 1 and sys.platform == "win32":
                    pid = next(iter(self.hello))
                    got = (pid, self.hello.pop(pid))
                if got:
                    return _Worker(proc, got[1], got[0])
                if proc.poll() is not None:
                    raise RuntimeError(f"顯示卡處理程式啟動失敗（代碼 {proc.returncode}），詳細原因在 logs/app.log")
                if time.time() > end:
                    proc.kill()
                    raise RuntimeError("顯示卡處理程式啟動逾時")
                self.cv.wait(0.5)

    def _acquire(self) -> _Worker:
        with self.cv:
            self._start()
            while True:
                for w in self.workers:
                    if not w.busy and w.alive():
                        w.busy = True
                        return w
                if len(self.workers) + self.starting < MAX_WORKERS:
                    self.starting += 1
                    break
                self.cv.wait(1)
        try:
            w = self._spawn()
        finally:
            with self.cv:
                self.starting -= 1
        with self.cv:
            self.workers.append(w)
        return w

    def _release(self, w: _Worker) -> None:
        with self.cv:
            w.busy = False
            w.last = time.time()
            self.cv.notify_all()

    def _drop(self, w: _Worker) -> None:
        w.kill()
        with self.cv:
            if w in self.workers:
                self.workers.remove(w)
            self.cv.notify_all()

    def _reaper(self) -> None:
        while True:
            time.sleep(3)
            old = []
            with self.cv:
                for w in list(self.workers):
                    if not w.alive():
                        self.workers.remove(w)
                    elif not w.busy and time.time() - w.last > IDLE_EXIT:
                        self.workers.remove(w)
                        old.append(w)
            for w in old:
                try:
                    w.conn.send(("exit",))
                    w.proc.wait(10)
                except Exception:
                    pass
                if w.alive():
                    w.kill()

    def shutdown(self) -> None:
        with self.cv:
            ws, self.workers = list(self.workers), []
        for w in ws:
            w.kill()

    def count(self) -> int:
        with self.cv:
            return len(self.workers)

    # -- 呼叫
    def call(self, name: str, *args, progress=None, low: bool = False, **kw):
        """在子程式裡執行 TASKS[name]。progress(step, p) 由主程式這邊呼叫；它丟出例外（停止、暫停）時
        直接結束子程式，顯示卡馬上釋放。"""
        w = self._acquire()
        ok = False
        try:
            w.conn.send(("task", name, args, kw, low))
            while True:
                if w.conn.poll(0.5):
                    msg = w.conn.recv()
                    if msg[0] == "p":
                        if progress:
                            progress(msg[1], msg[2])
                        continue
                    ok = True   # 子程式正常回覆（成功或一般錯誤）：可以繼續用
                    if msg[0] == "ok":
                        return msg[1]
                    if msg[1] == "UserError":
                        import pipeline as P
                        raise P.UserError(msg[2])
                    raise RemoteError(msg[1], msg[2])
                if not w.alive():
                    raise RuntimeError(f"顯示卡處理程式意外結束（代碼 {w.proc.returncode}），詳細原因在 logs/app.log")
        finally:
            if ok and w.alive():
                self._release(w)
            else:
                self._drop(w)


def _is_child(pid: int, parent: int) -> bool:
    """pid 是不是 parent 開出來的（Windows 虛擬環境的 python.exe 會再開一個真正的 Python）。"""
    if sys.platform != "win32":
        try:
            with open(f"/proc/{pid}/stat") as f:
                return int(f.read().rsplit(")", 1)[1].split()[1]) == parent
        except Exception:
            return False
    try:
        import ctypes
        from ctypes import wintypes

        class PE(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                        ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_void_p),
                        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                        ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
                        ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_char * 260)]
        k = ctypes.windll.kernel32
        snap = k.CreateToolhelp32Snapshot(0x2, 0)
        e = PE()
        e.dwSize = ctypes.sizeof(PE)
        try:
            ok = k.Process32First(snap, ctypes.byref(e))
            while ok:
                if e.th32ProcessID == pid:
                    return e.th32ParentProcessID == parent
                ok = k.Process32Next(snap, ctypes.byref(e))
        finally:
            k.CloseHandle(snap)
    except Exception:
        pass
    return False


POOL = Pool()


def call(name: str, *args, progress=None, **kw):
    """主程式呼叫：直播中（偵測到直播軟體）子程式用較低的優先順序跑。"""
    low = False
    try:
        import pipeline as P
        low = bool(P.live_status()["hold"])
    except Exception:
        pass
    return POOL.call(name, *args, progress=progress, low=low, **kw)
