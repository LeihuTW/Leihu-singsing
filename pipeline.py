"""練唱台：音訊處理流程。

每首歌存在 library/<id>/：
  original.*        匯入的原始檔（轉成 mix 後刪除）
  mix.flac          44.1kHz 立體聲（舊版是 .wav，照樣讀得到）
  lead.flac         主唱
  back.flac         合音
  inst.flac         伴奏
  pitch.json        主唱音高曲線與音符
  lyrics.lrc        歌詞（含時間）
  v2_k+2_s100/      移調／變速後的分軌快取
  takes/            你的錄音與評分
"""
from __future__ import annotations

import json
import os
import queue
import re
import shutil
import sys
import threading
import time
import traceback
import uuid
from pathlib import Path

import numpy as np
import soundfile as sf

APP_NAME = "雷虎練唱台"
ROOT = Path(__file__).resolve().parent
LIB = ROOT / "library"
MODELS = ROOT / "models"
CONFIG_PATH = ROOT / "config.json"
DEFAULT_LIB, DEFAULT_MODELS = LIB, MODELS


def _apply_storage() -> None:
    """2.3.0：歌曲庫、模型可以搬到別的位置（設定裡的 library_dir／models_dir；空的＝程式資料夾裡）。"""
    global LIB, MODELS
    try:
        c = json.loads(CONFIG_PATH.read_text("utf-8"))
    except Exception:
        c = {}
    LIB = Path(c["library_dir"]) if c.get("library_dir") else DEFAULT_LIB
    MODELS = Path(c["models_dir"]) if c.get("models_dir") else DEFAULT_MODELS


_apply_storage()

SR = 44100
PITCH_SR = 16000
HOP = 160            # 10 ms
HOP_S = HOP / PITCH_SR
STEMS = ("lead", "back", "inst")

DEFAULT_CONFIG = {
    "separation_model": "UVR-MDX-NET-Inst_HQ_3.onnx",
    "vocal_quality": "auto",   # 人聲／伴奏分離：auto：有顯示卡用高品質（BS-Roformer，乾淨很多但慢一倍）；standard；high
    "pause_while_streaming": True,   # 偵測到直播軟體（OBS、VTube Studio…）或直播模式開著時，暫停背景的分離、辨識
    "parallel_separation": False,
    "yt_enabled": False,     # 2.3.1：YouTube 下載與歌單批次匯入預設隱藏；點左上標題五下開啟後一直保持開啟（更新也不會關）
    "queue_paused": False,   # 2.1.1：使用者按了「全部暫停」（重開程式後仍維持暫停）   # 2.1.1：沒在直播時兩首同時分離（顯示卡記憶體 10 GB 以上建議）
    "streaming_apps": ["obs64.exe", "obs32.exe", "VTube Studio.exe", "Streamlabs OBS.exe", "XSplit.Core.exe", "vMix64.exe"],
    "karaoke_model": "UVR_MDXNET_KARA_2.onnx",
    "karaoke_quality": "auto",   # auto：有顯示卡用高品質（Roformer）；standard；high
    "karaoke_overlap": 2,
    "auto_fine": True,   # 合音多的歌，處理完自動排一次最精細的主唱分離（排在佇列最後）   # 高品質主唱分離的重疊次數：2＝快（建議），4＝模型預設、約慢一倍
    "whisper_model": "large-v3-turbo",
    "lyrics_language": "",   # 空字串＝自動判斷
    "convert_to_taiwan": True,
    "auto_recognize_lyrics": False,
    "global_hotkeys": True,   # 1.7：Ctrl+Alt 全域快捷鍵（重開練唱台後生效）   # 匯入時自動辨識歌詞（關閉時直接套用線上搜尋到的第一個版本）
    "audio_out": "",   # 音樂輸出裝置名稱；空字串＝系統預設
    "monitor_out": "",   # 「原唱只給我聽」時，自己聽的裝置（耳機）名稱
    "font": "huninn",   # 介面與歌詞字體：huninn（內建 jf open 粉圓）／kai（楷書）／sans（黑體）
    "pitch_model": "auto",
    "claude_model": "claude-sonnet-5-5",
    "anthropic_api_key": "",
    "balance_usd": None,
    "balance_set_at": None,
}


if sys.platform == "win32":
    # DLL 載入失敗時不要跳出需要按「確定」的視窗（會讓背景處理卡住），改成一般錯誤；子程序也會沿用
    try:
        import ctypes
        ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)
    except Exception:
        pass


class UserError(Exception):
    """可直接顯示給使用者的錯誤。"""


# ---------------------------------------------------------------- 設定

def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    if CONFIG_PATH.exists():
        try:
            cfg.update(json.loads(CONFIG_PATH.read_text("utf-8")))
        except Exception:
            pass
    return cfg


def save_config(cfg: dict) -> None:
    atomic_write_text(CONFIG_PATH, json.dumps(cfg, ensure_ascii=False, indent=2))


def module_status() -> dict:
    """哪些離線功能已安裝。"""
    import importlib.util as u
    has = lambda m: u.find_spec(m) is not None
    return {
        "uvr": has("audio_separator"),
        "crepe": has("torchcrepe"),
        "whisper": has("faster_whisper"),
        "pedalboard": has("pedalboard"),
        "opencc": has("opencc"),
        "anthropic": has("anthropic"),
        "kana": has("sudachipy") and has("sudachidict_small"),
        "cuda": cuda_ok() if IN_WORKER else (gpu_info(block=False) or {}).get("ok", False),
    }


_CUDA_OK = None
CUDA_NOTE = ""
IN_WORKER = False   # True：這裡是顯示卡子程式（gpu_worker.py），可以載入 PyTorch／CUDA
_GPU_INFO: dict | None = None
_GPU_INFO_LOCK = threading.Lock()
_GPU_PROBING = {"on": False}


def _torch_version() -> str:
    try:
        from importlib.metadata import version
        return version("torch")
    except Exception:
        return ""


def gpu_info(block: bool = True) -> dict | None:
    """顯示卡能不能用（主程式不載入 CUDA，所以由子程式檢查一次，結果存在 models/gpu_probe.json；
    換了 PyTorch 版本才重新檢查）。block=False：還不知道時回傳 None，並在背景開始檢查。"""
    global _GPU_INFO, CUDA_NOTE
    if _GPU_INFO is not None:
        return _GPU_INFO
    tv = _torch_version()
    if not tv:
        _GPU_INFO = {"ok": False, "note": "", "cc": None, "torch": ""}
        return _GPU_INFO
    cache = MODELS / "gpu_probe.json"
    try:
        c = json.loads(cache.read_text("utf-8"))
        if c.get("torch") == tv and (not c.get("ok") or "vram" in c):   # 舊的紀錄沒有記憶體大小：重新檢查一次
            _GPU_INFO = c
            CUDA_NOTE = c.get("note") or ""
            return c
    except Exception:
        pass
    if not block:
        with _GPU_INFO_LOCK:
            if not _GPU_PROBING["on"]:
                _GPU_PROBING["on"] = True
                threading.Thread(target=gpu_info, daemon=True, name="gpu-probe").start()
        return None
    with _GPU_INFO_LOCK:
        if _GPU_INFO is not None:
            return _GPU_INFO
        import gpu_worker as G
        try:
            info = G.POOL.call("probe")
        except Exception as e:
            log_error("gpu probe", e)
            info = {"ok": False, "note": "", "cc": None}
        info["torch"] = tv
        try:
            MODELS.mkdir(parents=True, exist_ok=True)
            atomic_write_text(cache, json.dumps(info, ensure_ascii=False))
        except Exception:
            pass
        CUDA_NOTE = info.get("note") or ""
        _GPU_INFO = info
        return info


def cuda_ok() -> bool:
    """顯示卡能不能真的跑 PyTorch。

    torch.cuda.is_available() 只代表「有卡、有驅動」；如果裝的 PyTorch 不支援這張卡的架構
    （例如 GTX 10 系列配上 CUDA 12.8 版），實際運算時才會出現「no kernel image is available」。
    所以實際跑一個小運算確認，失敗就改用 CPU。結果只算一次。"""
    global _CUDA_OK, CUDA_NOTE
    if not IN_WORKER:   # 主程式不載入 CUDA：問顯示卡子程式（結果會存起來）
        return bool((gpu_info() or {}).get("ok"))
    if _CUDA_OK is not None:
        return _CUDA_OK
    try:
        import torch
        if not torch.cuda.is_available():
            _CUDA_OK = False
            return False
        x = torch.ones(64, device="cuda:0")
        _CUDA_OK = float((x * 2).sum().item()) == 128.0
    except Exception as e:
        _CUDA_OK = False
        try:
            import torch
            name = torch.cuda.get_device_name(0)
        except Exception:
            name = "顯示卡"
        CUDA_NOTE = (f"{name} 和目前安裝的 PyTorch 版本不相容，已改用 CPU 處理（比較慢）。"
                     "執行開始選單的「雷虎練唱台 修復與更新」可改裝相容的版本。")
        print(f"[cuda] 顯示卡無法使用，改用 CPU：{type(e).__name__}: {str(e)[:300]}", flush=True)
    return _CUDA_OK


# ---------------------------------------------------------------- 檔案

def downloads_dir() -> Path:
    """使用者的「下載」資料夾（Windows 會讀系統設定，處理被搬到其他磁碟的情況）。"""
    if sys.platform == "win32":
        try:
            import ctypes
            guid = ctypes.c_byte * 16
            # FOLDERID_Downloads = {374DE290-123F-4565-9164-39C4925E467B}
            import uuid as _uuid
            raw = _uuid.UUID("374DE290-123F-4565-9164-39C4925E467B").bytes_le
            fid = guid(*raw)
            out = ctypes.c_wchar_p()
            if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(fid), 0, None, ctypes.byref(out)) == 0:
                path = Path(out.value)
                ctypes.windll.ole32.CoTaskMemFree(out)
                return path
        except Exception:
            pass
    d = Path.home() / "Downloads"
    return d if d.exists() else Path.home()

def song_dir(song_id: str) -> Path:
    if not re.fullmatch(r"[a-z0-9]{6,16}", song_id or ""):
        raise UserError("歌曲編號不正確")
    d = LIB / song_id
    if not d.exists():
        raise UserError("找不到這首歌")
    return d


# meta.json 會被好幾個執行緒同時讀寫（處理佇列、預先下載、背景歌詞搜尋、網頁請求）。
# 同時寫入同一個暫存檔會把兩份內容疊在一起，留下壞掉的 JSON，整個歌曲庫就讀不出來；
# 所以讀寫都排隊進行，每次寫入也用自己的暫存檔。
_META_LOCK = threading.RLock()


def _replace_retry(src: Path, dst: Path) -> None:
    """Windows 上目標檔剛好被別的程式（防毒、雲端同步、其他讀取）開著時，取代會失敗；稍等重試幾次。"""
    for i in range(6):
        try:
            src.replace(dst)
            return
        except PermissionError:
            if i == 5:
                raise
            time.sleep(0.05 * (i + 1))


def atomic_write_text(path: Path, text: str, encoding: str = "utf-8") -> None:
    """先寫到自己專用的暫存檔，再一次取代正式檔：寫到一半當機或同時寫入，都不會留下壞掉的檔案。"""
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex[:8]}.tmp")
    try:
        tmp.write_text(text, encoding)
        _replace_retry(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def read_meta(d: Path) -> dict:
    p = d / "meta.json"
    with _META_LOCK:
        if not p.exists():
            return {}
        text = p.read_text("utf-8", errors="replace")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            meta = _repair_meta(d, text, e)
            write_meta(d, meta)   # 修好後存回去，下次就正常了
            return meta


def _repair_meta(d: Path, text: str, err: Exception) -> dict:
    """1.6.1 以前同時寫入可能留下壞掉的 meta.json（兩份內容疊在一起）。
    先試著取出第一份完整的內容；完全讀不出來時，依資料夾裡的檔案重建基本資料，歌曲和錄音都不會不見。"""
    log_error(f"meta.json 壞掉，自動修復：{d.name}", err)
    try:
        meta, _ = json.JSONDecoder().raw_decode(text.lstrip())
        if isinstance(meta, dict) and meta.get("id"):
            return meta
    except Exception:
        pass
    m = re.search(r'"title"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
    title = json.loads(f'"{m.group(1)}"') if m else f"（資料損壞的歌曲 {d.name}）"
    has = lambda n: any((d / f"{n}{e}").exists() for e in (".flac", ".wav"))
    meta = {"id": d.name, "title": title[:120], "created": d.stat().st_mtime, "duration": None,
            "steps": {"import": has("mix"), "separate": has("lead") and has("back") and has("inst"),
                      "pitch": (d / "pitch.json").exists(), "lyrics": (d / "lyrics.lrc").exists()},
            "separation": "uvr" if has("lead") else None, "lyrics_source": None,
            "notes": ["這首歌的資料檔曾經損壞，已自動修復；如果歌名或設定不對，可以用「編輯資訊」修改。"]}
    for key in ("source_url", "artist"):   # 能找到的欄位盡量救回來（還沒下載的歌靠 source_url 才能接著下載）
        m = re.search(rf'"{key}"\s*:\s*"((?:[^"\\]|\\.)*)"', text)
        if m:
            meta[key] = json.loads(f'"{m.group(1)}"')
    m = re.search(r'"key"\s*:\s*(-?\d+)', text)
    if m and -6 <= int(m.group(1)) <= 6:
        meta["key"] = int(m.group(1))
    if meta.get("source_url") and not meta["steps"]["import"]:
        meta["dl_state"] = "failed"   # 還沒下載的歌：標成下載失敗，按「重新下載」就能接著下載
        meta["title"] = _DL_RE.sub("", meta["title"]) + "（下載失敗）"
    return meta


def write_meta(d: Path, meta: dict) -> None:
    with _META_LOCK:
        atomic_write_text(d / "meta.json", json.dumps(meta, ensure_ascii=False, indent=2))


def update_meta(d: Path, **kw) -> dict:
    with _META_LOCK:   # 讀出、修改、寫回之間不讓別的執行緒插進來，免得蓋掉對方的修改
        meta = read_meta(d)
        meta.update(kw)
        write_meta(d, meta)
        return meta


_SONG_CACHE: dict = {}   # 歌曲編號 → (檔案時間, 整理好的資料)


def _song_entry(d: Path) -> dict | None:
    """一首歌在歌曲庫清單裡的資料。資料檔、歌詞、錄音都沒變時直接用上次整理好的（幾百首歌時差很多）。"""
    def mt(p: Path) -> int:
        try:
            return p.stat().st_mtime_ns
        except OSError:
            return 0
    stamp = (mt(d / "meta.json"), mt(d / "lyrics.lrc"), mt(d / "takes"))
    hit = _SONG_CACHE.get(d.name)
    if hit and hit[0] == stamp:
        return dict(hit[1])
    try:
        m = read_meta(d)
    except Exception as e:   # 一首歌的資料讀不出來，不要讓整個歌曲庫打不開
        log_error(f"list_songs {d.name}", e)
        return None
    rc = m.get("lyrics_recheck")
    if isinstance(rc, dict):
        try:
            if (d / "lyrics.lrc").stat().st_mtime_ns != rc.get("mtime"):
                m["lyrics_recheck"] = None   # 歌詞改過了
        except OSError:
            m["lyrics_recheck"] = None
    takes = list_takes(d)
    m["takes"] = len(takes)
    scored = [t["acc"] for t in takes if t.get("acc") is not None]
    m["best"] = max(scored) if scored else None
    lp = d / "lyrics.lrc"
    if lp.exists():
        try:
            m["lyrics_text"] = " ".join(l["text"] for l in parse_lrc(lp.read_text("utf-8")))[:3000]
        except Exception:
            pass
    _SONG_CACHE[d.name] = (stamp, m)
    return dict(m)


def usage_score(m: dict, setlist_ids: set | None = None) -> float:
    """這首歌常不常唱（2.1.1）：大量重新處理時，常唱的排前面。
    看最近開過／直播唱過的時間、直播唱過幾次、錄音幾次、有沒有自己整理過歌詞或分類。"""
    now = time.time()
    score = 0.0
    t = max(m.get("last_opened") or 0, m.get("last_sung") or 0)
    if t:
        score += 1000 + max(0.0, 180 - (now - t) / 86400)   # 最近半年內開過，越近越前面
    if setlist_ids and m.get("id") in setlist_ids:
        score += 500
    score += 20 * (m.get("sung") or 0) + 10 * min(20, m.get("takes") or 0)
    if m.get("key"):
        score += 30   # 設過自己的 KEY：一定是會唱的歌
    if m.get("lyrics_source") in ("online", "import", "claude"):
        score += 5
    if m.get("category"):
        score += 2
    return score


def list_songs() -> list[dict]:
    LIB.mkdir(exist_ok=True)
    out = []
    for d in LIB.iterdir():
        if d.is_dir() and (d / "meta.json").exists():
            m = _song_entry(d)
            if m is not None:
                out.append(m)
    out.sort(key=lambda m: m.get("created", 0), reverse=True)
    return out


# ---------------------------------------------------------------- 歌曲庫搜尋（繁簡互通＋相近歌名）

_T2S = None
_SKEY_CACHE: dict = {}


def search_key(text: str) -> str:
    """搜尋用的正規化：全形半形統一、轉小寫、繁體一律轉成簡體（繁簡多對一，兩邊都會對到同一個字）。"""
    global _T2S
    import unicodedata
    t = unicodedata.normalize("NFKC", text or "").lower()
    if re.search(r"[\u3400-\u9fff]", t):
        if _T2S is None:
            try:
                from opencc import OpenCC
                _T2S = OpenCC("t2s")
            except Exception:
                _T2S = False
        if _T2S:
            t = _T2S.convert(t)
    return t


def _song_search_key(m: dict) -> str:
    hay = " ".join(str(x) for x in (m.get("title"), m.get("artist"), m.get("yt_title"), m.get("yt_artist"), m.get("yt_track"), m.get("yt_uploader"),
                                    m.get("category"), m.get("lyrics_text"),
                                    (m.get("lyrics_online") or {}).get("artist") if isinstance(m.get("lyrics_online"), dict) else None)
                   if x)
    k = _SKEY_CACHE.get(hay)
    if k is None:
        k = _SKEY_CACHE[hay] = search_key(hay)
        if len(_SKEY_CACHE) > 5000:
            _SKEY_CACHE.clear()
    return k


def _bigrams(s: str) -> set:
    return {s[i:i + 2] for i in range(len(s) - 1)} or ({s} if s else set())


def _skey(*parts) -> str:
    hay = " ".join(str(x) for x in parts if x)
    k = _SKEY_CACHE.get(hay)
    if k is None:
        k = _SKEY_CACHE[hay] = search_key(hay)
        if len(_SKEY_CACHE) > 5000:
            _SKEY_CACHE.clear()
    return k


def _lyric_snip(text: str, words: list[str]) -> str:
    """歌詞裡符合的那一句（給搜尋結果顯示）。"""
    if not text:
        return ""
    norm = search_key(text)
    pos = norm.find(words[0]) if words else -1
    if pos < 0:
        return ""
    src = text if len(text) == len(norm) else norm   # 繁簡轉換長度不同時，退回顯示轉換後的文字
    a = src.rfind(" ", 0, pos) + 1
    b = src.find(" ", pos + len(words[0]))
    b = len(src) if b < 0 else b
    a, b = max(a, pos - 14), min(b, pos + len(words[0]) + 14)
    return ("…" if a > 0 else "") + src[a:b].strip() + ("…" if b < len(src) else "")


def search_songs(q: str, scope: str = "all") -> dict:
    """回傳符合的歌曲（每個關鍵字都要出現，繁簡互通），依符合的地方排序：
    歌名 → 歌手 → 歌名＋歌手 → 其他資訊 → 只有歌詞裡有（另外附上符合的那句歌詞），以及歌名相近的推薦。
    scope：name＝只找歌名、歌手；lyrics＝只找歌詞；all＝都找（歌詞符合的排最後）。"""
    from difflib import SequenceMatcher
    nq = search_key(q).strip()
    if not nq:
        return {"match": [], "similar": [], "kind": {}, "snip": {}}
    words = nq.split()
    squash = lambda s: re.sub(r"[\W_]+", "", s)
    qq = squash(nq)

    def has(key: str) -> bool:
        # 也比對去掉空白、標點後的版本（「牛仔很忙cowboy」找得到「牛仔很忙 Cowboy on the Run」）
        return all(w in key for w in words) or (len(qq) >= 2 and qq in squash(key))
    ranked, sim, kind, snip = [], [], {}, {}
    for n, m in enumerate(list_songs()):
        lo = m.get("lyrics_online") if isinstance(m.get("lyrics_online"), dict) else {}
        tkey = _skey(m.get("title"), m.get("yt_track"))
        akey = _skey(m.get("artist"), m.get("yt_artist"), lo.get("artist"), m.get("yt_uploader"))
        r = None
        if scope == "lyrics":
            lyr = _skey(m.get("lyrics_text"))
            if has(lyr):
                ranked.append((5, n, m["id"]))
                kind[m["id"]] = "lyrics"
                snip[m["id"]] = _lyric_snip(m.get("lyrics_text") or "", words)
            continue
        if has(tkey):
            r = 0 if squash(tkey) == qq else 1
        elif has(akey):
            r = 2
        elif has(tkey + " " + akey):
            r = 3
        elif has(_skey(m.get("title"), m.get("yt_track"), m.get("artist"), m.get("yt_artist"), lo.get("artist"),
                       m.get("yt_uploader"), m.get("yt_title"), m.get("category"))):
            r = 4
        elif scope != "name" and has(_song_search_key(m)):
            r = 5
            kind[m["id"]] = "lyrics"
            snip[m["id"]] = _lyric_snip(m.get("lyrics_text") or "", words)
        if r is not None:
            ranked.append((r, n, m["id"]))
            continue
        if len(qq) < 2:
            continue
        best = 0.0
        for name in (m.get("title"), m.get("yt_track")):
            tk = squash(search_key(name or ""))
            if not tk:
                continue
            ratio = SequenceMatcher(None, qq, tk).ratio()
            cover = len(_bigrams(qq) & _bigrams(tk)) / len(_bigrams(qq))   # 輸入的字有多少出現在歌名裡
            best = max(best, ratio, cover * 0.9)
        if best >= 0.45:
            sim.append((best, m["id"]))
    ranked.sort()
    sim.sort(reverse=True)
    return {"match": [i for _, _, i in ranked], "similar": [i for _, i in sim[:6]], "kind": kind, "snip": snip}


def _blank_song(title: str, **extra) -> str:
    LIB.mkdir(exist_ok=True)
    sid = uuid.uuid4().hex[:10]
    d = LIB / sid
    d.mkdir()
    write_meta(d, {
        "id": sid, "title": title, "created": time.time(), "duration": None,
        "steps": {"import": False, "separate": False, "pitch": False, "lyrics": False},
        "separation": None, "lyrics_source": None, "notes": [], **extra,
    })
    return sid


def new_song(filename: str, data_path: Path) -> str:
    sid = _blank_song(Path(filename).stem)
    ext = Path(filename).suffix.lower() or ".wav"
    shutil.move(str(data_path), LIB / sid / f"original{ext}")
    return sid


# ---------------------------------------------------------------- ffmpeg 與 YouTube

def ffmpeg_path() -> str | None:
    """找 ffmpeg：系統安裝的，或 imageio-ffmpeg 附帶的。並讓 librosa/audioread 也找得到。"""
    found = shutil.which("ffmpeg")
    if found:
        return found
    try:
        import imageio_ffmpeg
        src = Path(imageio_ffmpeg.get_ffmpeg_exe())
    except Exception:
        return None
    bindir = ROOT / "bin"
    bindir.mkdir(exist_ok=True)
    dst = bindir / ("ffmpeg.exe" if sys.platform == "win32" else "ffmpeg")
    if not dst.exists() or dst.stat().st_size != src.stat().st_size:
        shutil.copy2(src, dst)
    os.environ["PATH"] = str(bindir) + os.pathsep + os.environ.get("PATH", "")
    return str(dst)


YOUTUBE_HOSTS = ("youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com", "youtu.be")


def clean_youtube_url(url: str) -> str:
    from urllib.parse import parse_qs, urlparse
    url = (url or "").strip()
    u = urlparse(url if "://" in url else "https://" + url)
    host = (u.hostname or "").lower()
    if host not in YOUTUBE_HOSTS:
        raise UserError("這不是 YouTube 網址")
    if host == "youtu.be":
        vid = u.path.strip("/").split("/")[0]
    elif u.path.startswith(("/shorts/", "/live/")):
        vid = u.path.split("/")[2]
    else:
        vid = (parse_qs(u.query).get("v") or [""])[0]
    if not re.fullmatch(r"[A-Za-z0-9_-]{11}", vid or ""):
        raise UserError("網址裡找不到影片編號，請貼上單一影片的網址")
    return f"https://www.youtube.com/watch?v={vid}"


# ---------------------------------------------------------------- 批次匯入（多個網址、歌單、歌名搜尋、Excel）

BATCH_MAX = 1000   # 一次最多幾首（表單可能有好幾百首）


def _ydl():
    import yt_dlp
    return yt_dlp.YoutubeDL({"quiet": True, "no_warnings": True, "skip_download": True,
                             "extract_flat": "in_playlist", "socket_timeout": 20})


def _entry(e: dict) -> dict:
    vid = e.get("id") or ""
    return {"id": vid, "url": f"https://www.youtube.com/watch?v={vid}", "title": e.get("title") or "",
            "duration": e.get("duration"), "channel": e.get("channel") or e.get("uploader") or ""}


def _playlist_id(line: str) -> str | None:
    from urllib.parse import parse_qs, urlparse
    u = urlparse(line if "://" in line else "https://" + line)
    if (u.hostname or "").lower() not in YOUTUBE_HOSTS:
        return None
    lid = (parse_qs(u.query).get("list") or [""])[0]
    # RD 開頭是 YouTube 自動產生的「合輯」，不是真的歌單：只取那一首
    if lid and not lid.startswith("RD") and (u.path.startswith("/playlist") or lid):
        return lid
    return None


def resolve_line(line: str) -> list[dict]:
    """把一行輸入變成要下載的影片清單：影片網址、歌單網址，或「歌手 歌名」（到 YouTube 搜尋第一筆）。"""
    line = re.sub(r"\s+", " ", line.replace("\t", " ")).strip()
    if not line:
        return []
    is_url = re.match(r"^(https?://)?([\w-]+\.)*(youtube\.com|youtu\.be)/", line, re.I)
    with _ydl() as ydl:
        if is_url and _playlist_id(line):
            info = ydl.extract_info(f"https://www.youtube.com/playlist?list={_playlist_id(line)}", download=False)
            items = [_entry(e) for e in (info.get("entries") or []) if e and e.get("id")]
            for it in items:
                it["from"] = f"歌單：{info.get('title') or ''}"
            return items[:BATCH_MAX]
        if is_url:
            url = clean_youtube_url(line)
            try:
                info = ydl.extract_info(url, download=False, process=False)
                it = _entry(info)
            except Exception:
                it = {"id": url[-11:], "url": url, "title": "", "duration": None, "channel": ""}
            return [it]
        info = ydl.extract_info(f"ytsearch1:{line}", download=False)
        items = [_entry(e) for e in (info.get("entries") or []) if e and e.get("id")]
        for it in items:
            it["from"] = f"搜尋：{line}"
        return items[:1]


def resolve_lines(lines: list, rows=None) -> list[dict]:
    """多行一起解析（同時查 4 行）。每一行的結果依原本順序排好；失敗的行也會列出來，附上原因。
    rows：從表單讀到的資料（以 line 對應）。表單有 YouTube 網址的直接用，不去搜尋；
    歌曲庫裡已經有同一個網址的歌，標成「更新資訊」（套用表單的歌手、曲名、KEY，不重新下載）。"""
    from concurrent.futures import ThreadPoolExecutor
    if isinstance(rows, list):   # 表單：每一列各自對應（同名的歌也不會混在一起）
        pairs = [(str(r.get("line") or "").strip(), r, r.get("row") or i + 2) for i, r in enumerate(rows) if isinstance(r, dict)]
        pairs = [p for p in pairs if p[0]][:BATCH_MAX]
    else:
        rows = rows or {}
        pairs = [(l, rows.get(l) or {}, None) for l in (str(x).strip() for x in lines) if l][:BATCH_MAX]
    by_url = {m.get("source_url"): m for m in list_songs() if m.get("source_url") and (m.get("steps") or {}).get("import")}
    # 已經下載好或正在排隊的算重複；之前下載失敗／取消的不算（勾選後會用原本那一項重新下載）
    existing = {m.get("source_url"): m.get("title") for m in list_songs()
                if m.get("source_url") and ((m.get("steps") or {}).get("import") or m.get("dl_state") not in ("failed", "cancelled"))}

    def one(pair):
        line, row, rowno = pair
        meta = {k: row.get(k) for k in ("artist", "song", "key", "lang", "category") if row.get(k) not in (None, "")}
        try:
            if row.get("url"):
                items = [{"id": row["url"][-11:], "url": row["url"], "title": row.get("song") or line,
                          "duration": None, "channel": row.get("artist") or "", "from": "表單網址"}]
            else:
                items = resolve_line(line)
            return [dict(it, input=line, sheet=meta or None, row=rowno) for it in items] \
                or [{"input": line, "error": "找不到符合的影片", "row": rowno}]
        except UserError as e:
            return [{"input": line, "error": str(e), "row": rowno}]
        except Exception as e:
            log_error(f"resolve({line})", e)
            return [{"input": line, "error": f"查詢失敗：{type(e).__name__}", "row": rowno}]
    with ThreadPoolExecutor(max_workers=4) as pool:
        groups = list(pool.map(one, pairs))
    out, seen = [], set()
    for g in groups:
        for it in g:
            if it.get("url"):
                it["dup"] = existing.get(it["url"]) or ("（清單裡重複）" if it["url"] in seen else None)
                old = by_url.get(it["url"])
                if old and it.get("sheet") and it["url"] not in seen:   # 舊歌：用表單資料更新
                    it["dup"] = None
                    it["update"] = {"id": old["id"], "title": old.get("title")}
                seen.add(it["url"])
            out.append(it)
    return out[:BATCH_MAX]


def read_sheet(data: bytes, name: str) -> list[dict]:
    """讀 Excel（.xlsx）或 CSV。認得「歌手」「曲名」「備註／KEY」「語言」和 YouTube 網址欄：
    回傳每一列 {line, artist, song, key, lang, url}。line 是給搜尋用的「歌手 曲名」（沒有網址時才會去搜尋）。"""
    import csv
    import io
    ext = Path(name).suffix.lower()
    rows: list[list[str]] = []
    nums: list[int] = []   # 每一列在表單裡是第幾列（給使用者對照）
    if ext == ".xlsx":
        import zipfile
        import xml.etree.ElementTree as ET
        ns = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
        try:
            z = zipfile.ZipFile(io.BytesIO(data))
        except zipfile.BadZipFile:
            raise UserError("讀不到這個 Excel 檔，請另存成 .xlsx 或 .csv 再試")
        shared = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")).findall("m:si", ns):
                shared.append("".join(t.text or "" for t in si.iter("{%s}t" % ns["m"])))
        sheets = sorted(n for n in z.namelist() if re.fullmatch(r"xl/worksheets/sheet\d+\.xml", n))
        if not sheets:
            raise UserError("Excel 檔裡沒有工作表")
        first = "xl/worksheets/sheet1.xml" if "xl/worksheets/sheet1.xml" in sheets else sheets[0]
        for r in ET.fromstring(z.read(first)).iter("{%s}row" % ns["m"]):
            cells = {}
            for c in r.findall("m:c", ns):
                ref = re.match(r"([A-Z]+)", c.get("r", "A"))
                col = 0
                for ch in ref.group(1):
                    col = col * 26 + ord(ch) - 64
                t, v = c.get("t"), c.find("m:v", ns)
                if t == "s" and v is not None:
                    val = shared[int(v.text)]
                elif t == "inlineStr":
                    val = "".join(x.text or "" for x in c.iter("{%s}t" % ns["m"]))
                else:
                    val = v.text if v is not None else ""
                cells[col] = (val or "").strip()
            if cells:
                rows.append([cells.get(i, "") for i in range(1, max(cells) + 1)])
                nums.append(int(r.get("r") or len(rows)))
    elif ext in (".csv", ".txt", ".tsv"):
        text = None
        for enc in ("utf-8-sig", "big5", "gbk"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise UserError("讀不到這個檔案的文字編碼")
        dialect = "excel-tab" if ext == ".tsv" or ("\t" in text and "," not in text) else "excel"
        rows = [[c.strip() for c in r] for r in csv.reader(io.StringIO(text), dialect=dialect)]
        nums = list(range(1, len(rows) + 1))
    elif ext == ".xls":
        raise UserError("舊版 Excel（.xls）請先另存成 .xlsx 或 .csv")
    else:
        raise UserError("只支援 .xlsx、.csv 或 .txt")
    keep = [i for i, r in enumerate(rows) if any(r)]
    rows, nums = [rows[i] for i in keep], [nums[i] for i in keep]
    if not rows:
        return []
    head = [c.lower() for c in rows[0]]
    find = lambda keys: next((i for i, c in enumerate(head) if any(k in c for k in keys)), None)
    ai = find(["歌手", "演唱", "藝人", "artist", "singer"])
    ti = find(["曲名", "歌名", "歌曲", "標題", "title", "song", "name"])
    ki = find(["key", "升降", "調"])
    if ki is None:
        ki = find(["備註", "note", "memo"])   # 雷虎的歌單：備註欄放的是升降 KEY
    li = find(["語言", "language", "lang"])
    ui = find(["ytlink", "youtube", "yt"])
    cell = lambda r, i: r[i].strip() if i is not None and i < len(r) else ""
    yt_re = re.compile(r"(https?://)?([\w-]+\.)*(youtube\.com|youtu\.be)/\S+", re.I)
    out = []
    if ai is not None or ti is not None:
        for r, rowno in zip(rows[1:], nums[1:]):
            artist, song = cell(r, ai), cell(r, ti)
            if not (artist or song):
                continue
            url = ""
            for c in ([cell(r, ui)] if ui is not None else []) + list(r):   # 先看網址欄，沒有再找整列
                m = yt_re.search(c or "")
                if m:
                    try:
                        url = clean_youtube_url(m.group(0))
                        break
                    except UserError:
                        continue
            lang = sheet_language(cell(r, li))
            out.append({"line": " ".join(x for x in (artist, song) if x), "artist": artist, "song": song,
                        "key": parse_key(cell(r, ki)), "lang": lang[0] if lang else None,
                        "category": lang[1] if lang else None, "url": url, "row": rowno})
    else:
        out = [{"line": " ".join(c for c in r if c)} for r in rows]
    return out[:BATCH_MAX]


# ---------------------------------------------------------------- 曲名／歌手／預設 KEY

LANG_FROM_SHEET = [   # 表單「語言」欄 → 歌詞辨識語言、分類名稱
    (("國語", "華語", "中文", "普通話", "mandarin", "chinese"), "zh", "中文歌"),
    (("台語", "閩南語", "台灣話", "taiwanese", "hokkien"), "zh", "台語歌"),
    (("粵語", "廣東話", "cantonese"), "yue", "粵語歌"),
    (("日語", "日文", "日本", "japanese"), "ja", "日文歌"),
    (("英語", "英文", "english"), "en", "英文歌"),
    (("韓語", "韓文", "韓國", "korean"), "ko", "韓文歌"),
]


def sheet_language(text: str) -> tuple[str, str] | None:
    t = (text or "").strip().lower()
    for words, code, cat in LANG_FROM_SHEET:
        if t and any(w in t for w in words):
            return code, cat
    return None


def parse_key(text) -> int | None:
    """表單裡的 KEY：-4、+2、4、＋３、=4（＝升 4）這類純數字才算，超過 ±6 的不採用。"""
    import unicodedata
    t = unicodedata.normalize("NFKC", str(text or "")).strip().replace("−", "-")
    if t.startswith("="):
        t = t[1:].strip()
    if not re.fullmatch(r"[+-]?\d{1,2}(?:\.0+)?", t):
        return None
    k = int(float(t))
    return k if -6 <= k <= 6 else None


_NOISE_RE = re.compile(r"\s*[\(（\[【「]?\s*(?:official\s*(?:music\s*)?(?:video|mv|audio|lyric(?:s)?\s*video)|"
                       r"music\s*video|lyric(?:s)?\s*video|官方\s*(?:完整版\s*)?(?:mv|音樂錄影帶|版)?|完整版|"
                       r"動態歌詞(?:版)?|歌詞版|高音質|hd|4k|mv|audio)\s*[\)）\]】」]?\s*", re.I)


def split_artist_title(yt_title: str, track: str = "", artist: str = "") -> tuple[str, str]:
    """從 YouTube 的資訊拆出（歌手, 曲名）。YouTube 有提供曲名就用；沒有就從標題常見寫法拆，拆不出來歌手留空。"""
    if track:
        return (artist or "").split(",")[0].strip(), track.strip()
    t = (yt_title or "").strip()
    m = re.match(r"^(.+?)\s*[【《「『〈]([^】》」』〉]+)[】》」』〉](.*)$", t)
    if m and not _NOISE_RE.fullmatch(m.group(2)):
        return m.group(1).strip(" -｜|/"), m.group(2).strip()
    clean = _NOISE_RE.sub(" ", t).strip()
    m = re.match(r"^(.+?)\s+[-–—｜|]\s+(.+)$", clean)
    if m:
        return m.group(1).strip(), m.group(2).strip()
    return "", re.sub(r"\s+", " ", clean).strip() or t


# YouTube 下載的狀態：排隊中／失敗／取消。寫在 meta 的 dl_state，歌名後面也標出來；下載成功就拿掉
DL_SUFFIX = {"queued": "（排隊下載）", "failed": "（下載失敗）", "cancelled": "（已取消下載）", "paused": "（暫停下載）"}
_DL_RE = re.compile("（(?:排隊下載|下載失敗|已取消下載|暫停下載)）$")

# YouTube 暫時限制這個 IP（429、要求登入確認不是機器人）時，暫停後面所有的下載，不再繼續送請求；
# 使用者確認後按「繼續下載」才會接著下載。只記在記憶體：程式重開時暫停的歌會留著「重新下載」按鈕。
YT_PAUSE: dict = {"on": False, "at": 0.0, "reason": ""}
_RATE_LIMIT_RE = re.compile(r"HTTP Error 429|Too Many Requests|Sign in to confirm you(?:'|’)re not a bot|"
                            r"confirm you(?:'|’)re not a bot|rate[- ]limit|try again later", re.I)


class YouTubeRateLimited(UserError):
    pass


def set_dl_state(d: Path, state: str | None) -> None:
    with _META_LOCK:
        meta = read_meta(d)
        base = _DL_RE.sub("", meta.get("title") or "").strip()
        if base.startswith("YouTube 下載"):
            base = ""
        if state == "cancelled":
            meta["dl_cancelled"] = True   # 預先下載剛好在進行中時，下載完也要維持「已取消」
        elif state == "queued":
            meta.pop("dl_cancelled", None)
        if state:
            meta["dl_state"] = state
            meta["title"] = (base or "YouTube 影片") + DL_SUFFIX[state]
        else:
            meta.pop("dl_state", None)
            meta["title"] = base or "YouTube 歌曲"
        write_meta(d, meta)


def pending_downloads() -> list[dict]:
    """還沒下載完成的 YouTube 歌曲（程式關掉時還在排隊的、失敗的、取消的）。
    舊版沒有記錄狀態：歌名結尾是（排隊下載）的就當作排隊中，有錯誤訊息的當作失敗。"""
    out = []
    for m in list_songs():
        if not m.get("source_url") or (m.get("steps") or {}).get("import"):
            continue
        st = m.get("dl_state")
        if not st:
            title = m.get("title") or ""
            if not (title.endswith("（排隊下載）") or title.startswith("YouTube 下載")):
                continue
            st = "failed" if m.get("notes") else "queued"
            set_dl_state(song_dir(m["id"]), st)
        out.append({**m, "dl_state": st})
    return out


def new_youtube_song(url: str, title: str = "") -> str:
    return _blank_song((title or "YouTube 影片") + DL_SUFFIX["queued"], source_url=url, dl_state="queued")


def _upgrade_ytdlp() -> None:
    import subprocess
    subprocess.run([sys.executable, "-m", "pip", "install", "-U", "yt-dlp", "--disable-pip-version-check"],
                   capture_output=True, timeout=300)


def step_youtube(d: Path, url: str, report) -> None:
    import importlib.util
    import subprocess
    if importlib.util.find_spec("yt_dlp") is None:
        raise UserError("沒有安裝 yt-dlp，無法從 YouTube 下載")
    ff = ffmpeg_path()
    if not ff:
        raise UserError("找不到 ffmpeg，無法轉換 YouTube 音訊")

    def attempt() -> tuple[int, str, str]:
        args = [sys.executable, "-m", "yt_dlp", url, "--no-playlist", "-f", "bestaudio/best",
                "-x", "--audio-format", "wav", "--ffmpeg-location", str(Path(ff).parent),
                "-o", str(d / "original.%(ext)s"), "--force-overwrites", "--no-simulate",
                "--newline", "--progress", "--progress-template", "download:PROG %(progress._percent_str)s",
                "--print", "before_dl:TITLE %(title)s",
                "--print", "before_dl:META %(track)s\t%(artist)s\t%(uploader)s", "--no-warnings"]
        flags = 0x08000000 if sys.platform == "win32" else 0   # 不要跳出黑色視窗
        p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
                             encoding="utf-8", errors="replace", creationflags=flags,
                             env=dict(os.environ, PYTHONIOENCODING="utf-8"))
        title = ""
        for line in p.stdout:
            line = line.strip()
            if line.startswith("TITLE "):
                title = line[6:]
                if not read_meta(d).get("title_locked"):
                    update_meta(d, title=title[:120])
            elif line.startswith("META "):
                track, artist, uploader = (line[5:].split("\t") + ["", "", ""])[:3]
                clean = lambda v: "" if v in ("NA", "None") else v.strip()
                update_meta(d, yt_track=clean(track), yt_artist=clean(artist), yt_uploader=clean(uploader))
            elif line.startswith("PROG "):
                m = re.search(r"([\d.]+)%", line)
                if m:
                    frac = float(m.group(1)) / 100
                    report(f"從 YouTube 下載音訊 {frac * 100:.0f}%", 0.01 + 0.05 * frac)
        err = p.stderr.read()
        p.wait()
        return p.returncode, title, err

    if YT_PAUSE["on"]:
        raise YouTubeRateLimited("YouTube 下載暫停中")
    report("取得影片資訊", 0.01)
    code, title, err = attempt()
    if code and _RATE_LIMIT_RE.search(err):   # 被限制時不要再更新 yt-dlp 重試，避免送更多請求
        raise YouTubeRateLimited(err.strip()[-200:])
    if code:
        if re.search(r"Private video|Video unavailable|This video is unavailable|members-only|"
                     r"confirm your age|age-restricted|age restricted", err):
            raise UserError(f"這部影片無法下載（可能是私人、會員限定或需要登入）：{err.strip()[-200:]}")
        report("下載失敗，更新 yt-dlp 後重試", 0.02)
        _upgrade_ytdlp()
        code, title, err = attempt()
        if code and _RATE_LIMIT_RE.search(err):
            raise YouTubeRateLimited(err.strip()[-200:])
        if code:
            raise UserError(f"YouTube 下載失敗：{err.strip()[-240:]}")
    for f in d.glob("original.*"):
        if f.suffix.lower() != ".wav":
            f.unlink(missing_ok=True)
    if not (d / "original.wav").exists():
        raise UserError("YouTube 下載完成，但找不到轉換後的音訊檔")
    meta = read_meta(d)
    yt = (title or "YouTube 歌曲")[:120]
    if meta.get("title_locked"):   # 從表單匯入：曲名、歌手用表單的寫法
        update_meta(d, yt_title=yt, source_url=url)
    else:
        artist, song = split_artist_title(yt, meta.get("yt_track", ""), meta.get("yt_artist", ""))
        update_meta(d, title=(song or yt)[:120], artist=(meta.get("artist") or artist)[:80], yt_title=yt,
                    source_url=url)
    set_dl_state(d, None)


# 預先下載：顯示卡在處理這首歌的時候，背景先把佇列裡接下來的 YouTube 歌下載好（網路和顯示卡同時工作）。
# 每首歌一把鎖，同一首不會同時被預先下載和正式處理。
_DL_LOCKS: dict = {}
_DL_LOCKS_GUARD = threading.Lock()


def _dl_lock(song_id: str) -> threading.Lock:
    with _DL_LOCKS_GUARD:
        return _DL_LOCKS.setdefault(song_id, threading.Lock())


def prefetch_youtube(song_id: str) -> bool:
    """先下載一首還在排隊的 YouTube 歌（只下載，不處理）。回傳有沒有真的下載。"""
    lock = _dl_lock(song_id)
    if not lock.acquire(blocking=False):
        return False
    try:
        d = song_dir(song_id)
        meta = read_meta(d)
        if YT_PAUSE["on"] or meta.get("steps", {}).get("import") or meta.get("prefetched") \
                or meta.get("prefetch_failed") or meta.get("dl_state") != "queued" or not meta.get("source_url"):
            return False
        try:
            step_youtube(d, meta["source_url"], lambda *a: None)
        except YouTubeRateLimited as e:
            if not YT_PAUSE["on"]:
                YT_PAUSE.update(on=True, at=time.time(), reason=str(e))
                log_error("youtube rate limited (prefetch)", e)
            set_dl_state(d, "queued")
            return False
        except Exception as e:   # 預先下載失敗沒關係，輪到這首時會再正式下載一次（失敗才標示）
            log_error(f"prefetch {song_id}", e)
            update_meta(d, prefetch_failed=True)
            set_dl_state(d, "queued")
            return False
        update_meta(d, prefetched=True)
        # 還沒處理，維持「排隊」；程式重開也會接著處理。下載途中被取消的，維持「已取消」
        set_dl_state(d, "cancelled" if read_meta(d).get("dl_cancelled") else "queued")
        return True
    finally:
        lock.release()


def youtube_job(song_id: str, url: str, after=None):
    """after(song_id)：整首處理完後要做的事（例如先把預設 KEY 的移調版本做好）。"""
    process = process_song(song_id, ["import", "separate", "pitch", "lyrics"])

    def run(report):
        d = song_dir(song_id)
        update_meta(d, notes=[])
        try:
            with _dl_lock(song_id):   # 正在預先下載這首時，等它下載完
                meta = read_meta(d)
                if meta.get("prefetched") and any(d.glob("original.*")):
                    report("已預先下載完成", 0.06)
                    update_meta(d, prefetched=False)
                    set_dl_state(d, None)
                else:
                    step_youtube(d, url, report)
        except JobDeferred:
            set_dl_state(d, "queued")   # 暫停：之後會重新下載，不算取消
            raise
        except JobCancelled:
            set_dl_state(d, "cancelled")
            raise
        except YouTubeRateLimited as e:
            if not YT_PAUSE["on"]:
                YT_PAUSE.update(on=True, at=time.time(), reason=str(e))
                log_error("youtube rate limited", e)
            set_dl_state(d, "paused")
            update_meta(d, notes=["YouTube 暫時限制下載，已暫停。過幾小時後按「繼續下載」或「重新下載」。"])
            return   # 不算錯誤：後面排隊的歌也會照樣停下來，不會再連到 YouTube
        except Exception as e:
            set_dl_state(d, "failed")
            update_meta(d, notes=[f"{e}　可以按「重新下載」再試一次，或刪除這一項。"])
            raise
        process(report)
        if after:
            try:
                after(song_id)
            except Exception as e:
                log_error("youtube_job after", e)
    return run


def read_audio(path: Path, sr: int = SR, mono: bool = False) -> np.ndarray:
    """回傳 (ch, n) 或 mono 時 (n,) 的 float32。"""
    import librosa
    if path.suffix.lower() not in (".wav", ".flac", ".ogg"):
        ffmpeg_path()  # m4a、mp3 等格式需要 ffmpeg
    y, _ = librosa.load(str(path), sr=sr, mono=mono)
    y = np.asarray(y, dtype=np.float32)
    if mono:
        return y
    if y.ndim == 1:
        y = np.stack([y, y])
    return y[:2]


def write_wav(path: Path, y: np.ndarray, sr: int = SR) -> None:
    """寫成 16 位元音訊；副檔名是 .flac 就存成 FLAC（無損壓縮，大小約 WAV 的一半，音質完全相同）。"""
    data = y.T if y.ndim == 2 else y
    sf.write(str(path), np.clip(data, -1.0, 1.0), sr, subtype="PCM_16")


# 分軌（mix／lead／back／inst，含移調版本）存成 FLAC；舊的 WAV 照樣讀得到，可以在設定裡一次轉換。
STEM_EXT = ".flac"


def stem_path(base: Path, name: str) -> Path:
    """某個分軌的檔案：有 FLAC 用 FLAC，沒有再找舊的 WAV；都沒有時回傳要寫入的 FLAC 路徑。"""
    for ext in (".flac", ".wav"):
        p = base / f"{name}{ext}"
        if p.exists():
            return p
    return base / f"{name}{STEM_EXT}"


def storage_report() -> dict:
    """歌曲庫佔用的空間，以及轉成 FLAC、刪掉原始檔後大約可以省多少。"""
    total = convertible = removable = 0
    songs = 0
    for d in LIB.iterdir() if LIB.exists() else []:
        if not d.is_dir():
            continue
        songs += 1
        for f in d.rglob("*"):
            if not f.is_file():
                continue
            sz = f.stat().st_size
            total += sz
            if f.suffix == ".wav" and f.stem in ("mix", *STEMS) and f.parent.name != "takes":
                convertible += sz
            elif f.name.startswith("original.") and (stem_path(d, "mix").exists()):
                removable += sz
            elif f.parent.name.startswith("v_k"):
                removable += sz
    # FLAC 通常是 WAV 的 40～60%，用 50% 估
    return {"songs": songs, "bytes": total, "convertible": convertible, "removable": removable,
            "saving": int(convertible * 0.5) + removable}


def compact_song(d: Path, busy=lambda: False) -> int:
    """把一首歌的舊 WAV 分軌（含移調版本）轉成 FLAC、刪掉已經不需要的原始檔。回傳省下的位元組。
    轉換是逐位元相同的無損壓縮，先寫暫存檔、確認長度一樣才取代原檔。"""
    saved = 0
    meta = read_meta(d)
    bases = [d] + [v for v in d.iterdir() if v.is_dir() and v.name.startswith("v2_")]
    for base in bases:
        if any(base.glob("*.tmp.*")) or busy():   # 正在產生的移調版本：這次先跳過
            continue
        for name in ("mix", *STEMS):
            w = base / f"{name}.wav"
            if not w.exists():
                continue
            info = sf.info(str(w))
            y, sr = sf.read(str(w), dtype="int16" if info.subtype == "PCM_16" else "float32", always_2d=True)
            tmp = base / f"{name}.tmp.flac"
            sf.write(str(tmp), y, sr, subtype="PCM_16")
            if sf.info(str(tmp)).frames != info.frames:
                tmp.unlink(missing_ok=True)
                raise IOError(f"{w.name} 轉換後長度不符，保留原檔")
            before = w.stat().st_size
            tmp.replace(base / f"{name}.flac")
            w.unlink()
            saved += before - (base / f"{name}.flac").stat().st_size
    for v in [v for v in d.iterdir() if v.is_dir() and v.name.startswith("v_k")]:   # 1.3 以前的移調版本，已經用不到
        saved += sum(f.stat().st_size for f in v.rglob("*") if f.is_file())
        shutil.rmtree(v, ignore_errors=True)
    if (meta.get("steps") or {}).get("import") and stem_path(d, "mix").exists():
        for f in d.glob("original.*"):
            saved += f.stat().st_size
            f.unlink(missing_ok=True)
    return saved


def compact_library(report, busy_song=lambda sid: False) -> None:
    dirs = [d for d in LIB.iterdir() if d.is_dir() and (d / "meta.json").exists()]
    saved, errors = 0, 0
    for i, d in enumerate(dirs):
        report(f"轉成 FLAC 節省空間 {i + 1}/{len(dirs)}（已省 {saved / 1e9:.1f} GB）", i / max(1, len(dirs)))
        try:
            saved += compact_song(d, lambda: busy_song(d.name))
        except Exception as e:
            errors += 1
            log_error(f"compact {d.name}", e)
    report(f"完成：省下 {saved / 1e9:.1f} GB" + (f"，{errors} 首沒有轉換（詳見紀錄）" if errors else ""), 1.0)


def save_stem(base: Path, name: str, y: np.ndarray, sr: int = SR) -> Path:
    """寫入分軌（先寫暫存檔再改名，中途中斷不會留下壞檔），並刪掉同名的舊 WAV。"""
    out = base / f"{name}{STEM_EXT}"
    tmp = base / f"{name}.tmp{STEM_EXT}"
    write_wav(tmp, y, sr)
    _replace_retry(tmp, out)
    if STEM_EXT != ".wav":
        (base / f"{name}.wav").unlink(missing_ok=True)
    return out


def fit_length(y: np.ndarray, n: int) -> np.ndarray:
    if y.ndim == 1:
        y = np.stack([y, y])
    if y.shape[1] >= n:
        return y[:, :n]
    return np.pad(y, ((0, 0), (0, n - y.shape[1])))


# ---------------------------------------------------------------- 匯入

# ---------------------------------------------------------------- 音量標準化
# 每首歌的響度差很多（YouTube 抓下來的歌常常一首很大聲、一首很小聲）。
# 匯入時用 ITU-R BS.1770（和 YouTube、Spotify 相同的響度標準）量出整首歌的響度，
# 播放時把主唱／合音／伴奏一起調到目標響度。不改動原始檔，關掉「音量標準化」就是原本的音量。
TARGET_LUFS = -16.0


def _biquad(b0, b1, b2, a0, a1, a2):
    return np.array([b0, b1, b2]) / a0, np.array([a0, a1, a2]) / a0


def _k_weight(sr: int):
    """BS.1770 的 K 加權濾波器（高架＋高通），依取樣率計算係數。"""
    import math
    # 第一段：高架濾波器（模擬頭部對高頻的影響）
    G, Q, fc = 3.999843853973347, 0.7071752369554196, 1681.974450955533
    A = 10 ** (G / 40)
    w0 = 2 * math.pi * fc / sr
    alpha = math.sin(w0) / (2 * Q)
    sh = _biquad(A * ((A + 1) + (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha),
                 -2 * A * ((A - 1) + (A + 1) * math.cos(w0)),
                 A * ((A + 1) + (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha),
                 (A + 1) - (A - 1) * math.cos(w0) + 2 * math.sqrt(A) * alpha,
                 2 * ((A - 1) - (A + 1) * math.cos(w0)),
                 (A + 1) - (A - 1) * math.cos(w0) - 2 * math.sqrt(A) * alpha)
    # 第二段：高通濾波器（去掉極低頻）
    Q, fc = 0.5003270373238773, 38.13547087602444
    w0 = 2 * math.pi * fc / sr
    alpha = math.sin(w0) / (2 * Q)
    hp = _biquad((1 + math.cos(w0)) / 2, -(1 + math.cos(w0)), (1 + math.cos(w0)) / 2,
                 1 + alpha, -2 * math.cos(w0), 1 - alpha)
    return sh, hp


def _block_power(y: np.ndarray, sr: int) -> np.ndarray | None:
    """K 加權後每個 400 ms 區塊（重疊 75%）的功率，左右聲道相加。"""
    from scipy.signal import lfilter
    x = np.atleast_2d(np.asarray(y, dtype=np.float64))
    (b1, a1), (b2, a2) = _k_weight(sr)
    x = lfilter(b2, a2, lfilter(b1, a1, x, axis=1), axis=1)
    block, step = int(0.4 * sr), int(0.1 * sr)
    if x.shape[1] < block:
        return None
    n = 1 + (x.shape[1] - block) // step
    p2 = np.cumsum(np.concatenate([np.zeros((x.shape[0], 1)), x ** 2], axis=1), axis=1)
    idx = np.arange(n) * step
    z = (p2[:, idx + block] - p2[:, idx]) / block          # 每個區塊、每個聲道的均方值
    return z.sum(axis=0)                                    # 左右聲道權重都是 1


def _gate(zsum: np.ndarray) -> np.ndarray | None:
    """BS.1770 的兩段門檻，回傳要計入的區塊。"""
    lk = -0.691 + 10 * np.log10(np.maximum(zsum, 1e-12))
    keep = lk > -70
    if not keep.any():
        return None
    rel = -0.691 + 10 * np.log10(zsum[keep].mean()) - 10
    keep &= lk > rel
    return keep if keep.any() else None


def integrated_loudness(y: np.ndarray, sr: int) -> float | None:
    """整首歌的響度（LUFS）。y 為 (聲道, 取樣) 或單聲道。太安靜量不到時回傳 None。"""
    zsum = _block_power(y, sr)
    if zsum is None:
        return None
    lk = -0.691 + 10 * np.log10(np.maximum(zsum, 1e-12))
    keep = lk > -70                                         # 絕對門檻：去掉靜音
    if not keep.any():
        return None
    rel = -0.691 + 10 * np.log10(zsum[keep].mean()) - 10    # 相對門檻：去掉比平均小 10 LU 的段落
    keep &= lk > rel
    if not keep.any():
        return None
    return round(float(-0.691 + 10 * np.log10(zsum[keep].mean())), 2)


def measure_loudness(d: Path) -> dict:
    """量 mix.wav 的響度與峰值，記在 meta（只算一次）。"""
    meta = read_meta(d)
    if meta.get("lufs") is not None or not stem_path(d, "mix").exists():
        return meta
    try:
        y = read_audio(stem_path(d, "mix"))
        lufs = integrated_loudness(y, SR)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        return update_meta(d, lufs=lufs, peak_db=round(20 * np.log10(max(peak, 1e-6)), 2))
    except Exception:
        traceback.print_exc()
        return meta


NORM_MIN_DB, NORM_MAX_DB = -15.0, 12.0


def _stems_sig(d: Path) -> str:
    try:
        return "|".join(str(stem_path(d, k).stat().st_mtime_ns) for k in ("mix", *STEMS))
    except OSError:
        return ""


def measure_stem_loudness(d: Path) -> dict:
    """1.6.5：量「主唱／合音／伴奏」每一種開關組合的響度（共 7 種）。
    只用整首歌的響度來調音量時，關掉原唱（卡拉OK）後的音量會差很多：
    人聲很突出的歌，拿掉人聲後伴奏會小聲很多。分軌之間有相關性，不能把各軌響度直接相加，
    所以每種組合實際混起來量。分軌重新處理過（例如精細處理）會自動重算。"""
    meta = read_meta(d)
    sig = _stems_sig(d)
    if not sig or (meta.get("combo_lufs") and meta.get("combo_lufs_sig") == sig):
        return meta
    try:
        y = {k: read_audio(stem_path(d, k)) for k in STEMS}
        n = min(v.shape[1] for v in y.values())
        out = {}
        for mask in range(1, 8):
            key = "".join("1" if mask >> (2 - i) & 1 else "0" for i in range(3))   # 順序：主唱、合音、伴奏
            mix = sum(y[k][:, :n] for k, b in zip(STEMS, key) if b == "1")
            v = integrated_loudness(mix, SR)
            out[key] = None if v is None else round(float(v), 2)
        return update_meta(d, combo_lufs=out, combo_lufs_sig=sig)
    except Exception:
        traceback.print_exc()
        return meta


def norm_gain_db(meta: dict, gains: dict | None = None) -> float:
    """播放時要加減幾 dB 才會到目標響度。小聲的歌最多加 12 dB（超過的部分由播放端的限幅器處理）。
    gains 是各音軌的音量：有組合響度時，用開著的音軌（音量 ≥ 0.5）那一種組合的響度計算，
    所以卡拉OK（關掉原唱）時伴奏也會調到一致的大小；拉桿的細部調整仍然照你設的比例。"""
    combo = meta.get("combo_lufs") or {}
    if gains is not None and combo:
        key = "".join("1" if (gains.get(k, 1) or 0) >= 0.5 else "0" for k in STEMS)
        if combo.get(key) is not None:
            return round(float(np.clip(TARGET_LUFS - combo[key], NORM_MIN_DB, NORM_MAX_DB)), 2)
    lufs = meta.get("lufs")
    if lufs is None:
        return 0.0
    return round(float(np.clip(TARGET_LUFS - lufs, NORM_MIN_DB, NORM_MAX_DB)), 2)


def step_import(d: Path, report) -> None:
    report("讀取音訊", 0.02)
    src = next(d.glob("original.*"))
    try:
        y = read_audio(src)
    except Exception as e:
        raise UserError(
            f"無法讀取這個音訊檔（{src.suffix}）。wav、mp3、flac、ogg 可直接讀；"
            "m4a/aac 需要先安裝 ffmpeg。") from e
    save_stem(d, "mix", y)
    for f in d.glob("original.*"):   # mix 已經是同一首歌（統一成 44.1 kHz），原始檔不用再留
        f.unlink(missing_ok=True)
    steps = read_meta(d).get("steps", {})
    steps["import"] = True
    peak = float(np.max(np.abs(y))) if y.size else 0.0
    update_meta(d, duration=round(y.shape[1] / SR, 3), steps=steps,
                lufs=integrated_loudness(y, SR), peak_db=round(20 * np.log10(max(peak, 1e-6)), 2))


# ---------------------------------------------------------------- 分離

def _stem_kind(name: str) -> str | None:
    marks = re.findall(r"\((vocals|instrumental|other|no vocals)\)", name.lower())
    return marks[-1] if marks else None


def _outputs(result, outdir: Path) -> list[Path]:
    paths = []
    for r in result or []:
        p = Path(r)
        if not p.is_absolute():
            p = outdir / p
        paths.append(p)
    if not paths:
        paths = list(outdir.glob("*.wav"))
    return paths


def _pick(paths: list[Path], kind: str) -> Path:
    for p in paths:
        if _stem_kind(p.name) == kind:
            return p
    alt = {"instrumental": ("other", "no vocals")}.get(kind, ())
    for p in paths:
        if _stem_kind(p.name) in alt:
            return p
    raise RuntimeError(f"分離結果裡找不到 {kind}：{[p.name for p in paths]}")


# 主唱／合音分離的高品質模型（Mel-Band Roformer，aufr33／viperx）。
# 合音多、或有和聲吟唱的歌，標準模型常把合音留在主唱軌，旋律會在兩個聲部之間亂跳；這個模型分得乾淨很多。
# 用 PyTorch 運算：有顯示卡約幾十秒，只有 CPU 會非常慢（一首歌可能要十幾分鐘），所以「自動」只在顯示卡可用時才用。
HQ_KARAOKE = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"
# 2.1.0：高品質的人聲／伴奏分離模型（BS-Roformer，viperx）。MVSEP 排行榜人聲 10.87／伴奏 17.17 dB，
# 比原本的 MDX-NET Inst HQ 3（9.38／15.68）乾淨很多；伴奏殘留的人聲和主唱軌殘留的鼓都明顯減少
HQ_VOCAL = "model_bs_roformer_ep_317_sdr_12.9755.ckpt"
_HQ_VOCAL_YAML = "model_bs_roformer_ep_317_sdr_12.9755.yaml"


# 模型下載來源：Hugging Face 的鏡像在台灣比 GitHub 快很多，放第一個；失敗或太慢再換 GitHub。
# 每個檔案都核對大小與 SHA-256（與 GitHub 原檔一致），鏡像就算被換掉也不會用到錯的檔案。
_HF_UVR = "https://huggingface.co/Politrees/UVR_resources/resolve/main/models/"
_RAW_CFG = "https://raw.githubusercontent.com/TRvlvr/application_data/main/mdx_model_data/mdx_c_configs/"
_GH_UVR = "https://github.com/TRvlvr/model_repo/releases/download/all_public_uvr_models/"
_GH_AS = "https://github.com/nomadkaraoke/python-audio-separator/releases/download/model-configs/"
_HQ_YAML = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956_config.yaml"
MODEL_FILES = {   # 檔名 → (大小, SHA-256, 下載網址…)
    "UVR-MDX-NET-Inst_HQ_3.onnx": (66759214, "317554b07fe1ea5279a77f2b1520a41ea4b93432560c4ffd08792c30fddf9adc", [
        _HF_UVR + "MDXNet/UVR-MDX-NET-Inst_HQ_3.onnx", _GH_UVR + "UVR-MDX-NET-Inst_HQ_3.onnx"]),
    "UVR_MDXNET_KARA_2.onnx": (52786726, "bf32e15105a09c0f7dddd2b67346146334d6f3ecb399ed7638eba2ab07cbf5f4", [
        _HF_UVR + "MDXNet/UVR_MDXNET_KARA_2.onnx", _GH_UVR + "UVR_MDXNET_KARA_2.onnx"]),
    HQ_KARAOKE: (913096801, "1de20d459332fe8869aeb01327a31df0032262706e1365114e852dc271779813", [
        _HF_UVR + "Roformer/MelBand/" + HQ_KARAOKE, _GH_UVR + HQ_KARAOKE, _GH_AS + HQ_KARAOKE]),
    _HQ_YAML: (None, None, [_GH_AS + _HQ_YAML]),
    # 2.1.1：Hugging Face 的資料夾是 BandSplit（2.1.0 寫成 BSRoformer → 404）；再多一個 GitHub 來源
    HQ_VOCAL: (639331213, "5b84f37e8d444c8cb30c79d77f613a41c05868ff9c9ac6c7049c00aefae115aa", [
        _HF_UVR + "Roformer/BandSplit/" + HQ_VOCAL, _GH_UVR + HQ_VOCAL, _GH_AS + HQ_VOCAL]),
    _HQ_VOCAL_YAML: (2273, "2bfdd16c656bd9519aba757cc4f8834b7ede675eb1e00ec4772d74ae1c41af7f", [
        _GH_AS + _HQ_VOCAL_YAML, _RAW_CFG + _HQ_VOCAL_YAML]),
}
# audio-separator 第一次載入模型時要的設定檔；raw.githubusercontent 不通時改走 jsDelivr
_RAW = "https://raw.githubusercontent.com/TRvlvr/application_data/main/"
_JSD = "https://cdn.jsdelivr.net/gh/TRvlvr/application_data@main/"
MODEL_DATA = {
    "download_checks.json": [_RAW + "filelists/download_checks.json", _JSD + "filelists/download_checks.json"],
    "vr_model_data.json": [_RAW + "vr_model_data/model_data_new.json", _JSD + "vr_model_data/model_data_new.json"],
    "mdx_model_data.json": [_RAW + "mdx_model_data/model_data_new.json", _JSD + "mdx_model_data/model_data_new.json"],
}
HQ_FILES = {k: MODEL_FILES[k] for k in (HQ_KARAOKE, _HQ_YAML)}
HQ_DL = {"state": "idle", "got": 0, "total": HQ_FILES[HQ_KARAOKE][0], "error": ""}
_hq_lock = threading.Lock()
HQV_FILES = {k: MODEL_FILES[k] for k in (HQ_VOCAL, _HQ_VOCAL_YAML)}
HQV_DL = {"state": "idle", "got": 0, "total": HQV_FILES[HQ_VOCAL][0], "error": ""}


def hqv_ready() -> bool:
    return _files_ready(HQV_FILES)


def hq_ready() -> bool:
    """高品質模型已經完整下載（大小完全正確才算，避免下載到一半的檔案被當成可用）。"""
    return _files_ready(HQ_FILES)


def _files_ready(files: dict) -> bool:
    uvr = MODELS / "uvr"
    for name, (size, _, _) in files.items():
        p = uvr / name
        if not p.exists() or (size and p.stat().st_size != size) or p.stat().st_size == 0:
            return False
    return True


class _SlowSource(Exception):
    pass


def _sha256(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def _open(url: str, start: int = 0, end: int | None = None, timeout: int = 30):
    import urllib.request
    hdr = {"User-Agent": USER_AGENT}
    if start or end is not None:
        hdr["Range"] = f"bytes={start}-{'' if end is None else end - 1}"
    return urllib.request.urlopen(urllib.request.Request(url, headers=hdr), timeout=timeout)


def fetch_file(urls: list[str], dst: Path, size: int | None = None, sha256: str | None = None,
               progress=None, conns: int = 4) -> None:
    """下載檔案：大檔分成幾段同時下載（單一連線被限速時快很多），每段都能續傳；
    一個來源連不上或中斷就換下一個來源。完成後核對大小與 SHA-256 才改成正式檔名。
    progress(已下載, 總大小, 速度) 每半秒呼叫一次；它丟出例外（例如按了停止）就中止下載。"""
    dst.parent.mkdir(parents=True, exist_ok=True)
    legacy = dst.with_name(dst.name + ".part")
    if dst.exists():
        if not size or dst.stat().st_size == size:
            return
        if not legacy.exists():
            dst.replace(legacy)            # 舊版直接寫正式檔名、下載到一半的檔案：接著續傳
        else:
            dst.unlink()

    if not size or size < (16 << 20):      # 小檔：一次下載
        last = None
        for url in urls:
            try:
                with _open(url) as r, open(legacy, "wb") as f:
                    shutil.copyfileobj(r, f, 1 << 20)
                if size and legacy.stat().st_size != size:
                    raise IOError(f"檔案大小不對（{legacy.stat().st_size} / {size}）")
                if sha256 and _sha256(legacy) != sha256:
                    raise IOError("檢查碼不符")
                legacy.replace(dst)
                return
            except Exception as e:
                last = e
                legacy.unlink(missing_ok=True)
                log_error(f"download {dst.name} {url}", e)
        raise last or IOError("下載失敗")

    seg = -(-size // conns)
    ranges = [(k * seg, min(size, (k + 1) * seg)) for k in range(conns)]
    parts = [dst.with_name(f"{dst.name}.part{k}") for k in range(conns)]
    if legacy.exists():                    # 1.4.5～1.4.9 的單一續傳檔：當作第一段的開頭
        if not parts[0].exists():
            with open(legacy, "r+b") as f:
                f.truncate(min(legacy.stat().st_size, seg))
            legacy.replace(parts[0])
        else:
            legacy.unlink()
    stop = threading.Event()
    errors: list[Exception] = []
    pref = [0]                             # 目前最好用的來源，其他段落也跟著用
    dead: set[str] = set()                 # 確定沒有這個檔案的來源（404 等）：之後所有段落都不再用
    lock = threading.Lock()

    def have(k: int) -> int:
        return parts[k].stat().st_size if parts[k].exists() else 0

    def alive() -> list[int]:
        with lock:
            return [i for i, u in enumerate(urls) if u not in dead]

    def next_src(ui: int) -> int:
        """從 ui 的下一個開始，找一個還能用的來源。"""
        ok = alive()
        if not ok:
            return ui
        for step in range(1, len(urls) + 1):
            if (ui + step) % len(urls) in ok:
                return (ui + step) % len(urls)
        return ui

    def worker(k: int) -> None:
        a, b = ranges[k]
        fails, ui = 0, pref[0]
        while not stop.is_set():
            got = have(k)
            if got >= b - a:
                if got > b - a:
                    with open(parts[k], "r+b") as f:
                        f.truncate(b - a)
                return
            if urls[ui % len(urls)] in dead:
                ui = next_src(ui)
            url = urls[ui % len(urls)]
            wrote = 0
            try:
                with _open(url, a + got, b) as r:
                    if r.status != 206:
                        raise IOError(f"來源不支援分段下載（HTTP {r.status}）")
                    with open(parts[k], "ab") as f:
                        w0, n0 = time.time(), 0
                        while not stop.is_set():
                            chunk = r.read(256 << 10)
                            if not chunk:
                                break
                            f.write(chunk)
                            n0 += len(chunk)
                            wrote += len(chunk)
                            if time.time() - w0 > 30:   # 這個來源慢到不行（每段 < 30 KB/s）：有別的來源就換
                                if n0 < 30e3 * 30 and len(alive()) > 1:
                                    raise _SlowSource()
                                w0, n0 = time.time(), 0
                fails = 0
            except _SlowSource:
                ui = next_src(ui)          # 太慢：換來源，不算失敗
                pref[0] = ui
            except Exception as e:
                code = getattr(e, "code", None)
                if code in (403, 404, 410):   # 這個來源沒有這個檔案：永久剔除，不要來回切換
                    with lock:
                        dead.add(url)
                    log_error(f"download {dst.name}：來源沒有這個檔案（HTTP {code}），不再使用 {url}", e)
                    if not alive():
                        errors.append(IOError(f"所有下載來源都找不到 {dst.name}"))
                        stop.set()
                        return
                    ui = next_src(ui)
                    pref[0] = ui
                    continue
                fails = 0 if wrote else fails + 1   # 有下載到東西（例如中途斷線）不算連續失敗
                ui = next_src(ui)          # 換下一個來源
                pref[0] = ui
                log_error(f"download {dst.name}[{k}] {url}", e)
                if fails >= 3 * max(1, len(alive())):
                    errors.append(e)
                    stop.set()
                    return
                stop.wait(2)

    threads = [threading.Thread(target=worker, args=(k,), daemon=True) for k in range(conns)]
    for t in threads:
        t.start()
    start_got, t0 = sum(have(k) for k in range(conns)), time.time()
    try:
        while any(t.is_alive() for t in threads):
            for t in threads:
                t.join(0.5 / conns)
            if progress:
                got = sum(have(k) for k in range(conns))
                progress(got, size, (got - start_got) / max(0.5, time.time() - t0))
    except BaseException:
        stop.set()
        for t in threads:
            t.join(5)
        raise
    if errors:
        raise errors[0]
    with open(parts[0], "ab") as f:        # 接起來
        for p in parts[1:]:
            with open(p, "rb") as g:
                shutil.copyfileobj(g, f, 1 << 22)
            p.unlink()
    if parts[0].stat().st_size != size or (sha256 and _sha256(parts[0]) != sha256):
        parts[0].unlink(missing_ok=True)
        raise IOError(f"{dst.name} 下載不完整（檢查碼不符），下次會重新下載")
    parts[0].replace(dst)


def ensure_model_data() -> None:
    """audio-separator 需要的模型設定檔；壞掉（例如下載到一半）就重新下載。"""
    uvr = MODELS / "uvr"
    for name, urls in MODEL_DATA.items():
        p = uvr / name
        if p.exists():
            try:
                json.loads(p.read_text("utf-8"))
                continue
            except Exception:
                p.unlink(missing_ok=True)
        try:
            fetch_file(urls, p)
        except Exception as e:
            log_error(f"model data {name}", e)   # 讓 audio-separator 自己再試一次


_model_locks: dict = {}


def ensure_model(name: str, progress=None) -> None:
    """確認分離模型已經完整下載；沒有或不完整就從最快的來源下載。"""
    ensure_model_data()
    spec = MODEL_FILES.get(name)
    if not spec:
        return                             # 不認得的模型：交給 audio-separator 自己下載
    size, sha, urls = spec
    p = MODELS / "uvr" / name
    with _model_locks.setdefault(name, threading.Lock()):   # 兩首同時處理時不要重複下載同一個模型
        if p.exists() and (not size or p.stat().st_size == size):
            return
        fetch_file(urls, p, size, sha, progress)


def start_hq_download() -> None:
    """在背景下載高品質主唱分離模型（約 900 MB），不佔用處理歌曲的佇列。"""
    _start_download(HQ_FILES, HQ_DL, hq_ready)


def start_hqv_download() -> None:
    """在背景下載高品質人聲／伴奏分離模型（約 640 MB）。"""
    _start_download(HQV_FILES, HQV_DL, hqv_ready)


def _start_download(files: dict, tracker: dict, ready) -> None:
    with _hq_lock:
        if tracker["state"] == "downloading" or ready():
            return
        tracker.update(state="downloading", error="")

    def run():
        try:
            for name in files:
                ensure_model(name, lambda got, total, spd: tracker.update(got=got))
            tracker.update(state="done", got=tracker["total"])
        except Exception as e:
            tracker.update(state="error", error=f"{type(e).__name__}: {e}")
    threading.Thread(target=run, daemon=True).start()


def hq_status() -> dict:
    if HQ_DL["state"] in ("idle", "done") and hq_ready():
        return {"state": "ready"}
    return dict(HQ_DL)


def hqv_status() -> dict:
    if HQV_DL["state"] in ("idle", "done") and hqv_ready():
        return {"state": "ready"}
    return dict(HQV_DL)


def vocal_model(cfg: dict) -> str:
    """要用的人聲／伴奏分離模型。高品質模型還沒下載好時，這首先用標準模型，模型在背景下載。"""
    q = cfg.get("vocal_quality", "auto")
    if q == "high" or (q == "auto" and cuda_ok()):
        if hqv_ready():
            return HQ_VOCAL
        start_hqv_download()
    return cfg["separation_model"]


def karaoke_model(cfg: dict) -> str:
    """要用的主唱／合音分離模型。高品質模型還沒下載好時，這首先用標準模型，模型在背景下載。"""
    q = cfg.get("karaoke_quality", "auto")
    if q == "high" or (q == "auto" and cuda_ok()):
        if hq_ready():
            return HQ_KARAOKE
        start_hq_download()
    return cfg["karaoke_model"]


def free_gpu() -> None:
    """把 PyTorch 暫存的顯示卡記憶體還回去。不還的話，後面的歌詞辨識（CTranslate2）可能記憶體不夠，
    Windows 會改用很慢的共用記憶體，看起來就像卡在「判斷歌詞語言」。"""
    import gc
    gc.collect()
    if "torch" in sys.modules:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
        except Exception:
            pass


_AUTOCAST = {"bad": False}


FINE_HARMONY = 0.5   # 合音能量達到主唱的一半以上，建議用最精細的方式再分一次


def harmony_ratio(d: Path) -> float | None:
    """主唱有聲音的段落裡，合音和主唱的音量比（0＝幾乎沒有合音，1＝和主唱一樣大聲）。"""
    lead = read_audio(stem_path(d, "lead"), sr=PITCH_SR, mono=True)
    back = read_audio(stem_path(d, "back"), sr=PITCH_SR, mono=True)
    n = min(len(lead), len(back)) // 1600 * 1600
    if n < 1600 * 10:
        return None
    lr = np.sqrt(np.mean(lead[:n].reshape(-1, 1600) ** 2, axis=1))
    br = np.sqrt(np.mean(back[:n].reshape(-1, 1600) ** 2, axis=1))
    voiced = lr > max(1e-4, 0.15 * float(np.percentile(lr, 95)))
    if voiced.sum() < 10:
        return None
    return round(float(np.sqrt(np.sum(br[voiced] ** 2) / np.sum(lr[voiced] ** 2))), 3)


LEAD_FIX_V = 2   # 主唱補回的判斷方式版本：改了判斷方式就加一，舊歌會重新檢查一次
SPLIT_NOTE = "主唱和合音可能沒分好"


def _lyric_spans(d: Path) -> list[tuple[float, float]]:
    """有時間的歌詞句子（開始、結束秒數）；整句括號的（和聲、旁白）不算。每句最多算 8 秒。"""
    p = d / "lyrics.lrc"
    if not p.exists():
        return []
    try:
        rows = [r for r in parse_lrc(p.read_text(encoding="utf-8", errors="ignore")) if r["t"] is not None]
    except Exception:
        return []
    out = []
    for i, r in enumerate(rows):
        tx = r["text"].strip()
        if not tx or re.fullmatch(r"[（(].*[)）]", tx) or ":" in tx[:8] or "：" in tx[:8]:
            continue   # 空白、整句括號、「作詞：」這類資訊行
        end = rows[i + 1]["t"] if i + 1 < len(rows) else r["t"] + 5
        end = min(end, r["t"] + 8)
        if end - r["t"] >= 1.0:
            out.append((r["t"], end))
    return out


def _lead_gap_gain(lead_m: np.ndarray, back_m: np.ndarray, sr: int,
                   spans: list[tuple[float, float]] | None = None) -> tuple[np.ndarray | None, float, dict]:
    """找出主唱被分到合音的地方，回傳（每個樣本要搬回主唱的比例 0～1、搬了幾秒、統計）。
    1. 「主唱幾乎沒聲音、合音卻很大聲」持續 8 秒以上（整段疊錄的副歌，例如〈牛仔很忙〉）。
    2. 有時間歌詞時，再逐句看：明明在唱歌詞，主唱卻幾乎整句沒聲音、合音很大聲 → 那句搬回主唱
       （對唱、多人輪流唱的歌，常常一兩句被分到合音）。
    短短幾秒、又不在歌詞句子裡的，通常是真的合音（間奏和聲、前奏哼唱），不動。
    統計裡 weak 是「主唱有聲音、但比合音小很多」的句數：這種沒辦法整句搬，只提醒。"""
    stats = {"lines": 0, "moved_lines": 0, "weak": 0}
    H = sr // 10
    n = min(len(lead_m), len(back_m)) // H * H
    if n < H * 50:
        return None, 0.0, stats
    lr = np.sqrt(np.mean(lead_m[:n].reshape(-1, H) ** 2, axis=1))
    br = np.sqrt(np.mean(back_m[:n].reshape(-1, H) ** 2, axis=1))
    k = np.ones(15) / 15   # 1.5 秒平滑：一個字、一口氣的空檔不算
    ls = np.sqrt(np.convolve(lr ** 2, k, "same"))
    bs = np.sqrt(np.convolve(br ** 2, k, "same"))
    ref = float(np.percentile(np.maximum(ls, bs), 95))
    if ref <= 1e-5:
        return None, 0.0, stats
    blk = np.zeros(len(ls), dtype=np.float32)
    cand = (bs > 0.12 * ref) & (ls < 0.1 * bs)
    i = 0
    while i < len(cand):
        if not cand[i]:
            i += 1
            continue
        j = i
        while j < len(cand) and cand[j]:
            j += 1
        if j - i >= 80:
            blk[max(0, i - 3):min(len(cand), j + 3)] = 1.0   # 前後各多 0.3 秒，接縫不會吃字
        i = j
    if spans:
        k2 = np.ones(5) / 5   # 逐句用 0.5 秒平滑
        ls2 = np.sqrt(np.convolve(lr ** 2, k2, "same"))
        bs2 = np.sqrt(np.convolve(br ** 2, k2, "same"))
        for st, en in spans:
            a, z = int(st * 10), int(en * 10)
            if z > len(ls2) or z - a < 10:
                continue
            act = np.maximum(ls2[a:z], bs2[a:z]) > 0.1 * ref
            if act.sum() < 10:   # 這句幾乎沒唱到（歌詞時間不準或間奏）
                continue
            stats["lines"] += 1
            miss = float(((ls2[a:z] < 0.1 * bs2[a:z]) & act).sum() / act.sum())
            weak = float(((ls2[a:z] < 0.5 * bs2[a:z]) & act).sum() / act.sum())
            if miss >= 0.8:
                stats["moved_lines"] += 1
                blk[max(0, a - 2):min(len(blk), z + 2)] = 1.0
            elif weak >= 0.6:
                stats["weak"] += 1
    sec = float(blk.sum()) / 10
    if sec < 1.0:
        return None, 0.0, stats
    fade = np.convolve(blk, np.ones(3) / 3, "same")   # 0.3 秒淡入淡出
    g = np.interp(np.arange(len(lead_m)), np.arange(len(fade)) * H + H / 2, fade).astype(np.float32)
    return g, sec, stats


def _split_note(d: Path, stats: dict) -> None:
    """很多句「主唱比合音小很多」：沒辦法自動整句搬，提醒使用者可以怎麼處理。"""
    with _META_LOCK:
        meta = read_meta(d)
        notes = [x for x in meta.get("notes", []) if SPLIT_NOTE not in x]
        if stats["lines"] >= 6 and stats["weak"] >= max(3, 0.25 * stats["lines"]):
            notes.append(f"{SPLIT_NOTE}：有 {stats['weak']} 句主唱聲音很小、合音很大（疊錄、對唱或多人合唱的歌常見）。"
                         "可以試「用最高品質人聲分離重新辨識」；多人輪流唱的歌可以開「合唱模式」")
        meta["notes"] = notes
        meta["split_suspect"] = stats["weak"]
        write_meta(d, meta)


def fix_lead_gaps(d: Path) -> float:
    """主唱被誤分到合音的段落、句子搬回主唱（合音那段同時拿掉）。回傳搬了幾秒；沒有就不改檔。"""
    lp, bp = stem_path(d, "lead"), stem_path(d, "back")
    if not lp.exists() or not bp.exists():
        return 0.0
    lead, back = read_audio(lp), read_audio(bp)
    n = min(lead.shape[-1], back.shape[-1])
    lead, back = lead[..., :n], back[..., :n]
    g, sec, stats = _lead_gap_gain(lead.mean(axis=0), back.mean(axis=0), SR, _lyric_spans(d))
    if g is not None:
        save_stem(d, "lead", lead + back * g)
        save_stem(d, "back", back * (1 - g))
        print(f"[separate] 主唱有 {sec:.0f} 秒（{stats['moved_lines']} 句歌詞）被分到合音，已搬回主唱", flush=True)
    _split_note(d, stats)
    update_meta(d, lead_fix=LEAD_FIX_V, lead_fix_sec=round(sec, 1), lead_fix_lrc=_lrc_stamp(d))
    return sec


def lead_fix_job(song_id: str, after=None):
    """舊歌：主唱被分到合音的段落搬回主唱，再重新分析旋律（不用重新分離）。"""
    def run(report):
        d = song_dir(song_id)
        report("檢查主唱有沒有被分到合音", 0.05)
        if fix_lead_gaps(d) > 0:
            step_pitch(d, load_config(), report)
            if after:
                after(song_id)
    return run


def _lrc_stamp(d: Path) -> float:
    p = d / "lyrics.lrc"
    return round(p.stat().st_mtime, 1) if p.exists() else 0.0


def lead_fix_needed(meta: dict) -> bool:
    """還沒用目前的判斷方式檢查過，或檢查之後歌詞換過（逐句檢查要用新的歌詞時間）。"""
    stp = meta.get("steps") or {}
    if not (stp.get("separate") and meta.get("separation") == "uvr" and meta.get("id")):
        return False
    if meta.get("lead_fix") != LEAD_FIX_V:
        return True
    return meta.get("lead_fix_lrc", 0.0) != _lrc_stamp(song_dir(meta["id"]))


def fine_job(song_id: str, after=None):
    """精細處理：用最精細的設定（重疊 4 次）重新分主唱與合音，再重新分析旋律。歌詞、錄音不變。"""
    def run(report):
        d = song_dir(song_id)
        cfg = dict(load_config())
        cfg["karaoke_overlap"] = 4
        if cfg.get("karaoke_quality") == "standard":
            cfg["karaoke_quality"] = "auto"
        full = hq_models_ready(cfg)   # 高品質模型都在：這次就等於直播後的補做
        lrc = (d / "lyrics.lrc").exists()
        fit_before = lyrics_timing_fit(d) if lrc else None
        step_separate(d, cfg, report)
        getattr(report, "gpu_done", lambda: None)()
        step_pitch(d, cfg, report)
        update_meta(d, fine_suggest=False, fine_done=True, fine_queued=False,
                    **({"live_light": False} if full else {}))
        if lrc:
            try:
                check_lyrics_after_reseparate(d, fit_before)
            except Exception as e:
                log_error("check_lyrics_after_reseparate", e)
        if after:
            try:
                after(song_id)
            except Exception as e:
                log_error("fine_job after", e)
    return run


def _autocast_ok() -> bool:
    """顯示卡支援快速的半精度運算（RTX 20 系列以後）時，高品質分離用混合精度，速度約快一倍、結果幾乎相同。
    GTX 10 系列以前的顯示卡半精度反而比較慢，維持原本的方式。"""
    if _AUTOCAST["bad"] or not cuda_ok():
        return False
    if not IN_WORKER:
        return ((gpu_info() or {}).get("cc") or 0) >= 7
    try:
        import torch
        return torch.cuda.get_device_capability(0)[0] >= 7
    except Exception:
        return False


_PROG_TLS = threading.local()   # 每個執行緒目前的分離進度（2.1.1：兩首同時分離時各回報各的）
_PROG_PATCHED = {"done": False}
_prog_lock = threading.Lock()


def _patch_tqdm() -> None:
    """把 audio-separator 內部的 tqdm 換成會依執行緒回報進度的版本（只換一次，之後不再還原）。"""
    with _prog_lock:
        if _PROG_PATCHED["done"]:
            return
        import importlib

        def tq(it=None, *a, **k):
            cur = getattr(_PROG_TLS, "cur", None)
            if cur is None or it is None:
                return it if it is not None else iter(())
            return cur.wrap(it)
        for name in _sep_progress.MODS:
            try:
                m = importlib.import_module("audio_separator.separator.architectures." + name)
                m.tqdm = tq
            except Exception:
                pass
        _PROG_PATCHED["done"] = True


class _sep_progress:
    """audio-separator 內部用 tqdm 跑每一段，換成會回報進度的版本，畫面上就看得到分離跑到哪裡；
    按停止時 report 會丟出例外，也能在分離到一半時停下來。"""
    MODS = ("mdx_separator", "mdxc_separator", "vr_separator")

    def __init__(self, label: str, report, start: float, span: float):
        self.best, self.t0 = 0.0, time.time()
        self.cb = lambda p, eta: report(f"{label} {p * 100:.0f}%{eta}", start + span * p)

    def eta(self, p: float) -> str:
        el = time.time() - self.t0
        if p < 0.05 or el < 5:
            return ""
        left = el * (1 - p) / p
        return f"（約剩 {left / 60:.0f} 分鐘）" if left >= 90 else f"（約剩 {max(5, round(left / 5) * 5):.0f} 秒）"

    def wrap(self, it):
        try:
            n = len(it)
        except Exception:
            n = 0

        def gen():
            for i, x in enumerate(it):
                if n:
                    p = i / n
                    if p >= self.best:
                        self.best = p
                        self.cb(p, self.eta(p))
                yield x
        return gen()

    def __enter__(self):
        _patch_tqdm()
        self.prev = getattr(_PROG_TLS, "cur", None)
        _PROG_TLS.cur = self
        return self

    def __exit__(self, *exc):
        _PROG_TLS.cur = self.prev
        return False


def _dl_report(report, label: str, frac: float):
    last = [0.0]

    def cb(got, total, spd):
        if time.time() - last[0] >= 0.5:
            last[0] = time.time()
            report(f"下載{label} {got / 1e6:.0f}／{total / 1e6:.0f} MB（{spd / 1e6:.1f} MB/s）", frac)
    return cb


def _separate(model: str, inp: Path, tmp: Path, overlap: int, autocast: bool,
              label: str, report, start: float, span: float, cpu: bool = False) -> list[Path]:
    """用一個分離模型分一次（2.1.1：在顯示卡子程式裡跑，主程式不載入 CUDA）。"""
    import gpu_worker as G
    outs = G.call("separate", model, str(inp), str(tmp), overlap, autocast, label, start, span, progress=report, cpu=cpu)
    return [Path(p) for p in outs]


def separate_uvr(d: Path, cfg: dict, report) -> None:
    tmp = d / "_sep"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    (MODELS / "uvr").mkdir(parents=True, exist_ok=True)
    overlap = 4 if cfg.get("karaoke_overlap") == 4 else 2
    cpu = bool(cfg.get("_cpu"))   # 直播中：只用 CPU
    autocast = False if cpu else _autocast_ok()

    report("準備人聲分離模型", 0.07)
    vmodel = vocal_model(cfg)
    vhq = vmodel == HQ_VOCAL
    ensure_model(vmodel, _dl_report(report, "人聲分離模型", 0.07))
    vlabel = "分離人聲與伴奏" + ("（直播中：CPU 輕量處理，下播後自動補做高品質）" if cpu else "") + ("（高品質）" if vhq else "（高品質模型下載中，這首先用標準模型）" if HQV_DL["state"] == "downloading" else "")
    report(vlabel, 0.08)
    outs = _separate(vmodel, stem_path(d, "mix"), tmp, overlap, autocast, vlabel, report, 0.08, 0.26, cpu)
    update_meta(d, vocal_model="bs_roformer" if vhq else "mdx")   # 之後可以找出還沒用新模型處理的歌
    voc = tmp / "vocals_only.wav"
    _pick(outs, "vocals").replace(voc)
    inst_p = _pick(outs, "instrumental")

    kara = karaoke_model(cfg)
    hq = kara == HQ_KARAOKE
    waiting = not hq and HQ_DL["state"] == "downloading"
    label = "把人聲分成主唱與合音" + ("（高品質）" if hq else "（高品質模型下載中，這首先用標準模型）" if waiting else "")
    if not hq:
        ensure_model(kara, _dl_report(report, "主唱分離模型", 0.35))
    report(label, 0.35)

    def run_kara(kara, label, ac):
        outs = _separate(kara, voc, tmp, overlap, ac, label, report, 0.35, 0.18, cpu)
        return [p for p in outs if p != voc and p != inst_p]
    try:
        try:
            outs2 = run_kara(kara, label, autocast)
        except JobCancelled:
            raise
        except Exception as e:
            if not (hq and autocast) or "out of memory" in str(e).lower():
                raise
            # 混合精度在這張顯示卡上出問題：之後都改用一般精度，這首重新分一次
            _AUTOCAST["bad"] = True
            log_error("autocast separation", e)
            outs2 = run_kara(kara, label, False)
    except JobCancelled:
        raise
    except Exception as e:
        if not hq:
            raise
        # 高品質模型下載失敗或顯示卡記憶體不夠：改用標準模型，不要整首失敗
        print(f"[separate] 高品質主唱分離失敗，改用標準模型：{type(e).__name__}: {e}", flush=True)
        kara = cfg["karaoke_model"]
        report("把人聲分成主唱與合音（標準）", 0.35)
        ensure_model(kara, _dl_report(report, "主唱分離模型", 0.35))
        outs2 = run_kara(kara, "把人聲分成主唱與合音（標準）", autocast)
    with _META_LOCK:   # 讀出到寫回之間不讓別的執行緒改這首歌的資料
        meta = read_meta(d)
        meta["karaoke"] = kara
        write_meta(d, meta)
    lead_p = _pick(outs2, "vocals")
    back_p = _pick(outs2, "instrumental")

    n = sf.info(str(stem_path(d, "mix"))).frames
    for name, p in (("lead", lead_p), ("back", back_p), ("inst", inst_p)):
        save_stem(d, name, fit_length(read_audio(p), n))
    shutil.rmtree(tmp, ignore_errors=True)
    try:
        fix_lead_gaps(d)   # 2.3.2：疊錄的副歌被整段分到合音 → 搬回主唱
    except Exception as e:
        log_error("fix_lead_gaps", e)
    # 合音比重：合音越多，主唱和合音越難分乾淨，更精細的分離越有幫助
    try:
        h = harmony_ratio(d)
        used_hq = kara == HQ_KARAOKE
        update_meta(d, harmony=h, sep_overlap=overlap if used_hq else None,
                    fine_suggest=bool(used_hq and overlap < 4 and h is not None and h >= FINE_HARMONY))
    except Exception as e:
        log_error("harmony ratio", e)


def separate_dsp(d: Path, report) -> None:
    """沒有 AI 模型時的備援：依聲像位置分離（中央=主唱）。"""
    import librosa
    from scipy.ndimage import uniform_filter1d

    report("以聲像位置分離（備援模式）", 0.1)
    mix = read_audio(stem_path(d, "mix"))
    n = mix.shape[1]
    N, H = 2048, 512
    L = librosa.stft(mix[0], n_fft=N, hop_length=H)
    R = librosa.stft(mix[1], n_fft=N, hop_length=H)
    P = np.abs(L) ** 2 + np.abs(R) ** 2 + 1e-12
    cross = L * np.conj(R)
    center = 2 * cross.real / P
    sim = 2 * np.abs(cross) / P
    f = librosa.fft_frequencies(sr=SR, n_fft=N)[:, None]
    band = np.where(f < 110, 0, np.where(f < 220, (f - 110) / 110,
                    np.where(f > 9000, np.maximum(0.25, 1 - (f - 9000) / 6000), 1)))
    x = np.clip((center - 0.67) / 0.25, 0, 1)
    x = x * x * (3 - 2 * x)
    ml = uniform_filter1d((x * band).astype(np.float32), 3, axis=1)
    vb = ((f > 180) & (f < 7000)).astype(np.float32)
    yy = np.clip((sim - 0.45) / 0.3, 0, 1)
    mb = uniform_filter1d(((1 - x) * yy * vb * (center > 0)).astype(np.float32), 3, axis=1)
    report("合成分軌", 0.3)
    lead = librosa.istft(ml * (L + R) / 2, hop_length=H, length=n)
    bl = librosa.istft(mb * L, hop_length=H, length=n)
    br = librosa.istft(mb * R, hop_length=H, length=n)
    back = np.stack([bl, br])
    inst = mix - lead[None, :] - back
    save_stem(d, "lead", np.stack([lead, lead]))
    save_stem(d, "back", back)
    save_stem(d, "inst", inst)


def is_acapella(d: Path) -> bool:
    """伴奏軌幾乎沒有聲音 → 原曲本來就是清唱／純人聲。"""
    try:
        def rms(name):
            y, _ = sf.read(str(stem_path(d, name)), dtype="float32", always_2d=True)
            return float(np.sqrt(np.mean(y ** 2)) + 1e-9)
        lead, inst = rms("lead"), rms("inst")
        return inst / lead < 0.12
    except Exception:
        return False


def step_separate(d: Path, cfg: dict, report) -> None:
    notes = []
    method = "dsp"
    if module_status()["uvr"]:
        try:
            try:
                separate_uvr(d, cfg, report)
            except Exception as e:
                if type(e).__name__ != "GpuStalled" or cfg.get("_cpu"):
                    raise
                # 2.3.2：顯示卡被其他程式佔滿、分離沒有進度 → 改用 CPU＋標準模型重做，之後自動補做高品質
                print("[gpu] 顯示卡太忙，這首改用 CPU＋標準模型分離（之後自動補做高品質）", flush=True)
                cfg = dict(cfg, **LIGHT_CFG)
                update_meta(d, live_light=True)
                report("顯示卡太忙：改用 CPU 輕量處理（之後自動補做高品質）", 0.08)
                separate_uvr(d, cfg, report)
            method = "uvr"
        except Exception as e:
            traceback.print_exc()
            notes.append(f"AI 分離失敗，改用聲像分離：{e}")
            separate_dsp(d, report)
    else:
        notes.append("沒有安裝 audio-separator，使用聲像分離（品質較差）")
        separate_dsp(d, report)
    for old in [*d.glob("v_*"), *d.glob("v2_*")]:
        shutil.rmtree(old, ignore_errors=True)
    acap = is_acapella(d)   # 要讀音訊，先在鎖外面算好
    with _META_LOCK:   # 讀出到寫回之間不讓別的執行緒改這首歌的資料
        meta = read_meta(d)
        meta["steps"]["separate"] = True
        meta["separation"] = method
        meta["acapella"] = acap
        meta["notes"] = [x for x in meta.get("notes", []) if "分離" not in x] + notes
        write_meta(d, meta)


# ---------------------------------------------------------------- 音高

def track_pitch(y16: np.ndarray, cfg: dict | None = None, strict: bool = False) -> np.ndarray:
    """16 kHz 單聲道 → 每 10 ms 一個 MIDI 音高（無聲為 NaN）。
    2.1.1：用 torchcrepe（顯示卡）時在子程式裡跑，主程式不載入 CUDA。"""
    cfg = cfg or load_config()
    if not IN_WORKER and module_status()["crepe"]:
        import gpu_worker as G
        # 直播中（含錄音評分）：在只用 CPU 的處理程式裡分析，不跟直播軟體、VTube Studio 搶顯示卡
        cpu = bool(cfg.get("_cpu")) or bool(live_status(cfg)["hold"])
        if strict and not cpu and not parallel_ok() and G._GPU_GATE.locked():
            cpu = True   # 錄音評分不要等顯示卡上的分離做完：先用 CPU 分析
        return G.call("pitch", np.ascontiguousarray(y16, dtype=np.float32), cfg, strict, cpu=cpu)
    return _track_pitch_local(y16, cfg, strict)


def _track_pitch_local(y16: np.ndarray, cfg: dict | None = None, strict: bool = False) -> np.ndarray:
    """16 kHz 單聲道 → 每 10 ms 一個 MIDI 音高（無聲為 NaN）。

    strict=True 用於麥克風錄音：門檻較嚴、依背景噪音決定靜音門檻，並去掉每個音頭尾的滑落／滑入
    （聲音變弱時音高分析會往下掉，畫出來像垂直往下的線，也會拉低評分）。"""
    import librosa
    cfg = cfg or load_config()
    n_frames = 1 + len(y16) // HOP
    f0 = per = None
    if module_status()["crepe"]:
        import torch
        import torchcrepe
        dev = "cuda:0" if cuda_ok() else "cpu"
        model = cfg.get("pitch_model", "auto")
        if model not in ("full", "tiny"):
            model = "full" if dev != "cpu" else "tiny"   # CPU 跑 full 太慢
        x = torch.from_numpy(np.ascontiguousarray(y16)).unsqueeze(0)
        with torch.no_grad():
            f, p = torchcrepe.predict(
                x, PITCH_SR, HOP, 65.0, 1100.0, model,
                decoder=torchcrepe.decode.viterbi, return_periodicity=True,
                batch_size=512, device=dev)
            p = torchcrepe.filter.median(p, 3)
        f0 = f[0].cpu().numpy()
        per = p[0].cpu().numpy()
        voiced = per > (0.6 if strict else 0.45)
        if not strict:
            # 原唱：把「確定有聲音」的段落往前後延伸到中等把握的影格（氣音、咬字、尾音），減少旋律斷斷續續
            weak = per > 0.22
            for _ in range(25):   # 最多延伸 0.25 秒
                grow = weak & ~voiced & (np.roll(voiced, 1) | np.roll(voiced, -1))
                if not grow.any():
                    break
                voiced = voiced | grow
    else:
        f0, voiced, _ = librosa.pyin(y16, fmin=65, fmax=1100, sr=PITCH_SR,
                                     frame_length=1024, hop_length=HOP)
    rms = librosa.feature.rms(y=y16, frame_length=1024, hop_length=HOP)[0]
    m = min(len(f0), len(rms), n_frames)
    f0, voiced, rms = f0[:m], voiced[:m], rms[:m]
    gate = rms < max(float(rms.max()) * 0.03, 1e-4)
    if strict and m:
        floor = float(np.percentile(rms, 10))   # 背景噪音
        gate |= rms < max(floor * 4, float(np.percentile(rms, 95)) * 0.06)
    midi = np.full(m, np.nan, dtype=np.float32)
    ok = voiced & ~gate & np.isfinite(f0) & (f0 > 0)
    midi[ok] = 69 + 12 * np.log2(f0[ok] / 440.0)
    if strict:
        midi[midi < 36] = np.nan   # 低於 C2 不可能是歌聲
        midi = trim_glides(midi)
        return clean_pitch(midi)
    return fix_octaves(bridge_gaps(clean_pitch(midi)), rms, y16)


def bridge_gaps(midi: np.ndarray, max_gap: int = 12, tol: float = 1.0) -> np.ndarray:
    """前後音高接近（1 個半音內）而且只斷了很短（0.12 秒內）的地方直接連起來。"""
    out = midi.copy()
    n, i = len(out), 0
    while i < n:
        if not np.isnan(out[i]):
            i += 1
            continue
        j = i
        while j < n and np.isnan(out[j]):
            j += 1
        if 0 < i and j < n and j - i <= max_gap and abs(out[i - 1] - out[j]) <= tol:
            out[i:j] = np.linspace(out[i - 1], out[j], j - i + 2)[1:-1]
        i = j
    return out


def octave_low_score(y16: np.ndarray, a: int, b: int, low_midi: float) -> float | None:
    """看這段聲音的泛音：低八度那個音是不是真的存在。

    回傳（第 1＋3 泛音）÷（第 2＋4 泛音）。真的是低音時第 1、3 泛音明顯（約 0.4 以上）；
    其實是高八度時，低音位置幾乎沒能量（約 0.1 以下）。"""
    s0, s1 = a * HOP_S * PITCH_SR, b * HOP_S * PITCH_SR
    seg = y16[int(s0):int(s1)]
    if len(seg) < 400:
        return None
    S = np.abs(np.fft.rfft(seg * np.hanning(len(seg)), n=1 << 15))
    fr = np.fft.rfftfreq(1 << 15, 1 / PITCH_SR)
    f = 440.0 * 2 ** ((low_midi - 69) / 12)
    h = [float(S[(fr > k * f * 0.97) & (fr < k * f * 1.03)].max(initial=0.0)) for k in (1, 2, 3, 4)]
    if h[1] + h[3] <= 0:
        return None
    return (h[0] + h[2]) / (h[1] + h[3])


def fix_octaves(midi: np.ndarray, rms: np.ndarray | None = None, y16: np.ndarray | None = None,
                ctx_s: float = 2.0) -> np.ndarray:
    """修正整段被判成高／低一個八度的音（常見在句尾漸弱的尾音）。

    找出和周圍（前後 2 秒，或緊接在前面的長音）差了將近一個八度的短音或弱音，
    再用泛音確認：低八度的音確實存在才往下移，確定是高音就保留；
    泛音看不出來時，只有「比周圍明顯小聲」或「句尾跟在長音後面的短尾音」才移。"""
    out = midi.copy()
    n = len(out)
    level = rms[:n] if rms is not None and len(rms) >= n else None
    runs, i = [], 0
    while i < n:
        if np.isnan(out[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isnan(out[j]):
            j += 1
        runs.append([i, j])
        i = j
    if len(runs) < 3:
        return out
    win = int(ctx_s / HOP_S)
    done = set()          # 每段最多移一次，避免第二輪又被移回去
    for _ in range(2):    # 修一次後周圍的參考也會變準，再看一次
        meds = [float(np.median(out[a:b])) for a, b in runs]
        for k, (a, b) in enumerate(runs):
            if k in done or (b - a) > 60:
                continue
            near = [q for q, (c, d) in enumerate(runs) if q != k and d > a - win and c < b + win]
            wts = [runs[q][1] - runs[q][0] for q in near]
            if sum(wts) < 30:
                continue
            vals = [meds[q] for q in near]
            order = np.argsort(vals)
            cum = np.cumsum(np.array(wts)[order])
            ctx = float(np.array(vals)[order][np.searchsorted(cum, cum[-1] / 2)])   # 依長度加權的中位數
            diff = meds[k] - ctx
            tail = False
            if k > 0:
                # 句尾的短尾音：跟緊接在前面（0.5 秒內）、比較長的那個音比
                pa, pb = runs[k - 1]
                pd = meds[k] - meds[k - 1]
                if a - pb <= 50 and (pb - pa) > 2 * (b - a) and (b - a) <= 30 and 10 <= abs(pd) <= 14:
                    diff, tail = pd, True
            if not 10 <= abs(diff) <= 14:
                continue
            weak = False
            if level is not None and near:
                ctx_lvl = float(np.median(np.concatenate([level[runs[q][0]:runs[q][1]] for q in near])))
                weak = float(np.median(level[a:b])) < 0.5 * ctx_lvl
            low = meds[k] - 12 if diff > 0 else meds[k]   # 兩個候選音裡比較低的那個
            score = octave_low_score(y16, a, b, low) if y16 is not None else None
            if score is not None and diff > 0 and score <= 0.12:
                continue      # 低音位置沒有能量：真的是高音
            if score is not None and diff < 0 and score >= 0.35:
                continue      # 低音確實存在：真的是低音
            sure = score is not None and ((diff > 0 and score >= 0.35) or (diff < 0 and score <= 0.12))
            if sure or weak or tail:
                shift = 12 * np.sign(diff)
                out[a:b] -= shift
                meds[k] -= shift
                done.add(k)
    return out


def trim_glides(midi: np.ndarray, slope: float = 0.9, k: int = 3) -> np.ndarray:
    """去掉每段聲音開頭與結尾「快速滑動」的部分（30 ms 內變化超過 slope 個半音）。

    只從兩端往內修，句子中間正常的換音、滑音不受影響。"""
    out = midi.copy()
    n = len(out)
    i = 0
    while i < n:
        if np.isnan(out[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isnan(out[j]):
            j += 1
        s, e = i, j
        while s + k < e and abs(out[s] - out[s + k]) > slope:
            s += 1
        while e - k - 1 >= s and abs(out[e - 1] - out[e - 1 - k]) > slope:
            e -= 1
        # 第二步：從兩端找最近一段穩定的音（8 格內起伏小於 0.6 半音），
        # 比它偏離超過 1.5 半音的頭尾視為滑入／滑落（慢慢掉下去的尾音也能去掉）
        w = 8
        for q in range(e - w, s - 1, -1):
            seg = out[q:q + w]
            if seg.max() - seg.min() < 0.6:
                med = float(np.median(seg))
                t = q + w
                while t < e and abs(out[t] - med) <= 1.5:
                    t += 1
                e = t
                break
        for q in range(s, e - w + 1):
            seg = out[q:q + w]
            if seg.max() - seg.min() < 0.6:
                med = float(np.median(seg))
                t = q
                while t > s and abs(out[t - 1] - med) <= 1.5:
                    t -= 1
                s = t
                break
        out[i:s] = np.nan
        out[e:j] = np.nan
        if e - s < 6:
            out[s:e] = np.nan
        i = j
    return out


def clean_pitch(f0: np.ndarray) -> np.ndarray:
    n = len(f0)
    out = np.full(n, np.nan, dtype=np.float32)
    for i in range(n):
        v = f0[i]
        if np.isnan(v):
            continue
        w = f0[max(0, i - 3): i + 4]
        w = w[~np.isnan(w)]
        med = float(np.median(w))
        if abs(v - med - 12) < 1.5:
            v -= 12
        elif abs(v - med + 12) < 1.5:
            v += 12
        out[i] = med if abs(v - med) > 2.5 else v
    i = 0
    while i < n:
        if np.isnan(out[i]):
            i += 1
            continue
        j = i
        while j < n and not np.isnan(out[j]):
            j += 1
        if j - i < 6:
            out[i:j] = np.nan
        i = j
    return out


def segment_notes(f0: np.ndarray) -> list[list[float]]:
    notes = []

    def close(cur, end):
        if cur and end - cur["s"] >= 0.08 and float(np.median(cur["v"])) >= 38:
            notes.append([round(cur["s"], 3), round(end, 3), int(round(float(np.median(cur["v"]))))])

    cur = None
    dev = 0
    for i, v in enumerate(f0):
        t = i * HOP_S
        if np.isnan(v):
            if cur:
                close(cur, t)
                cur = None
            continue
        if cur is None:
            cur = {"s": t, "v": [float(v)]}
            dev = 0
            continue
        mean = sum(cur["v"]) / len(cur["v"])
        if abs(v - mean) > 0.7:
            dev += 1
            if dev >= 5:
                split = t - 4 * HOP_S
                tail = cur["v"][-4:]
                cur["v"] = cur["v"][:-4]
                close(cur, split)
                cur = {"s": split, "v": tail + [float(v)]}
                dev = 0
                continue
        else:
            dev = 0
        cur["v"].append(float(v))
    if cur:
        close(cur, len(f0) * HOP_S)
    return notes


PITCH_ALGO = 2   # 1.4.2：旋律分析改良（補斷點、修八度）；舊的歌可以重新分析旋律


def analysis_vocal(d: Path) -> np.ndarray:
    """旋律分析與歌詞辨識用的人聲（16 kHz 單聲道）。
    1.6.5：多人輪流唱的歌（例如 S.H.E），分離模型常把某位成員唱的整句歸到「合音」，主唱軌就斷斷續續；
    這種歌設定成「合唱模式」後，改用主唱＋合音一起分析，音高追蹤會跟著當下最大聲（通常就是旋律）的那個人。"""
    lead = read_audio(stem_path(d, "lead"), sr=PITCH_SR, mono=True)
    if read_meta(d).get("vocal_mode") == "all" and stem_path(d, "back").exists():
        back = read_audio(stem_path(d, "back"), sr=PITCH_SR, mono=True)
        n = min(len(lead), len(back))
        lead = lead[:n] + back[:n]
    return lead


_PITCH_LOCK = threading.Lock()


def step_pitch(d: Path, cfg: dict, report) -> None:
    report("分析主唱旋律" if read_meta(d).get("vocal_mode") != "all" else "分析旋律（合唱模式：主唱＋合音）", 0.55)
    y16 = analysis_vocal(d)
    with _PITCH_LOCK:   # 重疊處理時一次只分析一首的旋律（模型共用、也比較省顯示卡記憶體）
        midi = track_pitch(y16, cfg)
    notes = segment_notes(midi)
    data = {
        "hop": HOP_S,
        "midi": [None if np.isnan(v) else round(float(v), 2) for v in midi],
        "notes": notes,
    }
    atomic_write_text(d / "pitch.json", json.dumps(data, separators=(",", ":")))
    free_gpu()
    with _META_LOCK:   # 讀出到寫回之間不讓別的執行緒改這首歌的資料
        meta = read_meta(d)
        meta["steps"]["pitch"] = True
        meta["pitch_algo"] = PITCH_ALGO
        write_meta(d, meta)


def load_pitch(d: Path) -> tuple[np.ndarray, list]:
    data = json.loads((d / "pitch.json").read_text("utf-8"))
    midi = np.array([np.nan if v is None else v for v in data["midi"]], dtype=np.float32)
    return midi, data["notes"]


# ---------------------------------------------------------------- 歌詞

# Whisper 在間奏、沒有人聲的地方常會「聽到」訓練資料裡影片字幕的片頭片尾字樣，這些句子直接丟掉
HALLUCINATION = re.compile(
    r"字幕|訂閱|订阅|點贊|点赞|感謝觀看|感谢观看|Amara|請不吝|请不吝|明鏡|明镜"
    r"|優優獨播|优优独播|獨播劇場|独播剧场|YoYo\s*Television|Television\s*Series|Series\s*Exclusive"
    r"|ご視聴|チャンネル登録|시청해\s*주셔서|구독"
    r"|thanks?\s+(you\s+)?for\s+watching|please\s+subscribe|subtitles?\s+by", re.I)


def strip_hallucinations(lrc: str) -> str:
    """把辨識結果裡的幻覺句子拿掉（1.6.5 以前辨識的歌，開啟時順便清理）。"""
    out = [l for l in lrc.splitlines() if not HALLUCINATION.search(re.sub(r"^(\[[^\]]*\])+", "", l))]
    return "\n".join(out) + ("\n" if lrc.endswith("\n") and out else "")
PUNCT_END = "，。！？、,.!?;；：:…"


def is_japanese(text: str) -> bool:
    """文字裡有一定比例的平假名／片假名 → 日文（日文的漢字不能轉成台灣繁體，例如「来」會被改成「來」）。"""
    kana = len(re.findall(r"[\u3040-\u30ff]", text or ""))
    han = len(re.findall(r"[\u3400-\u9fff]", text or ""))
    return kana >= 3 and kana >= 0.15 * (kana + han)


def to_taiwan(text: str) -> str:
    if is_japanese(text):
        return text
    try:
        from opencc import OpenCC
    except ImportError:
        return text
    # 只做簡體→繁體的字形轉換（s2tw）。以前用的 s2twp 還會把詞換成台灣慣用語
    # （例如「信息→資訊」「出租車→計程車」），會改到歌詞原本的用字，所以不用
    for conf in ("s2tw", "s2tw.json"):
        try:
            return OpenCC(conf).convert(text)
        except Exception:
            continue
    return text


def _whisper_model(cfg):
    try:
        import torch  # noqa: F401  Whisper 需要的 cublas64_12.dll 由 PyTorch 提供
    except Exception:
        pass
    from faster_whisper import WhisperModel
    root = str(MODELS / "whisper")
    name = cfg.get("whisper_model") or "large-v3-turbo"
    try:
        model = WhisperModel(name, device="auto", compute_type="default", download_root=root)
        if model.model.device != "cuda":
            return model
        try:
            # 有 NVIDIA 顯示卡時 device="auto" 一定選顯示卡，但裝的是 CPU 版 PyTorch 時沒有 cublas64_12.dll。
            # 這時第一次運算會出錯，之後的 transcribe 卻會一直卡住、按停止也沒用，所以先試算一小段確認
            model.detect_language(np.zeros(16000, dtype=np.float32), vad_filter=False)
            return model
        except RuntimeError as e:
            print(f"[lyrics] 顯示卡無法執行語音辨識，改用 CPU（比較慢）：{e}", flush=True)
            del model
    except Exception:
        traceback.print_exc()
    try:
        return WhisperModel(name, device="cpu", compute_type="int8", download_root=root)
    except Exception as e:
        raise UserError(f"載入語音辨識模型 {name} 失敗（第一次使用需要連網下載）：{e}") from e


def fmt_lrc_time(t: float) -> str:
    t = max(0.0, t)
    m = int(t // 60)
    return f"{m:02d}:{t - m * 60:05.2f}"


def parse_lrc(text: str) -> list[dict]:
    lines = []
    for raw in text.splitlines():
        tags = re.findall(r"\[(\d+):(\d+(?:\.\d+)?)\]", raw)
        body = re.sub(r"\[[^\]]*\]", "", raw)
        body = re.sub(r"<\d+:\d+(?:\.\d+)?>", "", body).strip()   # 逐字時間標記（Enhanced LRC）
        if not tags:
            if body:
                lines.append({"t": None, "text": body})
            continue
        for mm, ss in tags:
            lines.append({"t": int(mm) * 60 + float(ss), "text": body})
    timed = sorted([l for l in lines if l["t"] is not None], key=lambda l: l["t"])
    return timed + [l for l in lines if l["t"] is None]


def to_lrc(lines: list[dict]) -> str:
    out = []
    for l in lines:
        out.append((f"[{fmt_lrc_time(l['t'])}]" if l.get("t") is not None else "") + l["text"])
    return "\n".join(out) + "\n"


def _collect_words(segs, dur: float, report, stats: list | None = None) -> list:
    """逐段取出辨識結果，同時更新進度；使用者按「停止」時在這裡停下。stats 會收到每段的可信度資料。"""
    words = []
    for s in segs:
        done = min(1.0, (s.end or 0) / max(dur, 1))
        report(f"辨識歌詞 {int(s.end // 60)}:{int(s.end % 60):02d} / {int(dur // 60)}:{int(dur % 60):02d}",
               0.75 + 0.19 * done)
        words.extend(s.words or [])
        if stats is not None:
            stats.append({"start": s.start, "end": s.end, "lp": getattr(s, "avg_logprob", 0.0),
                          "cr": getattr(s, "compression_ratio", 1.0), "ns": getattr(s, "no_speech_prob", 0.0),
                          "text": s.text or ""})
    return words


LANG_NAMES = {"zh": "中文", "ja": "日文", "en": "英文", "ko": "韓文", "yue": "粵語"}


def _sung_windows(y16: np.ndarray, n: int = 3, win: float = 30.0) -> np.ndarray:
    """主唱聲音最多的幾段（每段 30 秒）接起來，用來判斷語言。
    以前先對整首歌跑一次語音偵測（VAD）再判斷，比較慢；主唱軌本來就只有人聲，直接挑最大聲的段落即可。"""
    sr = PITCH_SR
    W = int(win * sr)
    if len(y16) <= W:
        return y16.astype(np.float32)
    hop = sr // 2
    e = np.array([float(np.sqrt(np.mean(y16[i:i + hop] ** 2))) for i in range(0, len(y16) - hop + 1, hop)])
    k = W // hop
    sums = np.convolve((e > max(1e-4, np.median(e) * 0.5)).astype(np.float32) * e, np.ones(k), "valid")
    picks: list[int] = []
    for i in np.argsort(-sums):
        if all(abs(int(i) - j) >= k for j in picks):
            picks.append(int(i))
        if len(picks) >= n:
            break
    return np.concatenate([y16[j * hop:j * hop + W] for j in sorted(picks)]).astype(np.float32)


def pick_language(model, y16: np.ndarray, want: str, per_song: bool) -> tuple[str | None, str]:
    """決定辨識語言。回傳 (語言, 說明)。

    每首歌個別指定的語言一定照用。全域設定是某個語言（例如中文）時，如果偵測到明顯是別的語言
    （例如日文歌），改用偵測到的語言，避免把日文硬辨識成中文。設定是自動時直接用偵測結果。"""
    if per_song and want:
        return want, ""
    t0 = time.time()
    try:
        det, prob, _ = model.detect_language(_sung_windows(y16), vad_filter=False, language_detection_segments=3)
    except Exception:
        traceback.print_exc()
        return (want or None), ""
    print(f"[lyrics] 偵測語言 {det} {prob:.2f}（設定 {want or '自動'}，{time.time() - t0:.1f} 秒）", flush=True)
    if want and det != want and prob >= 0.7:
        return det, f"偵測到這首是{LANG_NAMES.get(det, det)}歌，改用{LANG_NAMES.get(det, det)}辨識"
    if want:
        return want, ""
    return (det if prob >= 0.5 else None), ""


def lyric_quality(stats: list, words: list, d: Path, lines: list) -> dict:
    """判斷辨識結果可不可靠（Whisper 的可信度指標＋和主唱有聲音的時間對不對得上）。"""
    total = sum(max(0.0, s["end"] - s["start"]) for s in stats) or 0.0
    bad = sum(max(0.0, s["end"] - s["start"]) for s in stats if s["lp"] < -1.0 or s["cr"] > 2.4)
    lp = (sum(s["lp"] * max(0.0, s["end"] - s["start"]) for s in stats) / total) if total else -9.0
    chars = sum(len(re.sub(r"\s", "", l["text"])) for l in lines)
    texts = [_norm(l["text"]) for l in lines if _norm(l["text"])]
    loop, run = 0, 1   # 同一句「連續」重複很多次：Whisper 卡住的典型症狀（副歌隔一段重複不算）
    for a, b in zip(texts, texts[1:]):
        run = run + 1 if a == b else 1
        loop = max(loop, run)
    sung = 0.0
    try:   # 主唱有聲音（有旋律）的總時間
        pj = json.loads((d / "pitch.json").read_text("utf-8"))
        sung = sum(1 for v in pj["midi"] if v is not None) * pj.get("hop", HOP_S)
    except Exception:
        pass
    spoken = sum(max(0.0, (w.end or 0) - (w.start or 0)) for w in words)
    cover = min(1.0, spoken / sung) if sung > 5 else None
    q = {"lp": round(lp, 3), "bad": round(bad / total, 3) if total else 1.0, "chars": chars,
         "loop": loop, "cover": None if cover is None else round(cover, 3)}
    reasons = []
    if chars < 15:
        reasons.append("幾乎沒有辨識出文字")
    if q["bad"] > 0.5:
        reasons.append("大部分段落的辨識把握很低")
    if lp < -1.1:
        reasons.append("整體辨識把握很低")
    if loop >= 5:
        reasons.append("同一句連續重複太多次")
    if cover is not None and cover < 0.25:
        reasons.append("有唱歌的地方大多沒有辨識出字")
    q["ok"] = not reasons
    q["reasons"] = reasons
    return q


def step_lyrics(d: Path, cfg: dict, report) -> None:
    if not module_status()["whisper"]:
        raise UserError("沒有安裝 faster-whisper，無法辨識歌詞")
    from types import SimpleNamespace
    import gpu_worker as G
    report("載入語音辨識模型（第一次會下載）", 0.7)
    y16 = analysis_vocal(d)
    meta0 = read_meta(d)
    song_lang = meta0.get("lyrics_language")          # 這首歌個別指定的語言（None＝依設定）
    want = song_lang if song_lang is not None else (cfg.get("lyrics_language") or "")
    # 2.1.1：載入模型、判斷語言、辨識都在顯示卡子程式裡跑（CTranslate2 的顯示卡記憶體才放得乾淨）
    r = G.call("whisper", np.ascontiguousarray(y16, dtype=np.float32), cfg, want, song_lang is not None,
               progress=report)
    words = [SimpleNamespace(**w) for w in r["words"]]
    stats = r["stats"]
    lang, lang_note = r["lang"], r["lang_note"]
    info = SimpleNamespace(language=r["language"])

    lines, cur = [], []

    def flush():
        if not cur:
            return
        text = "".join(w.word for w in cur).strip()
        text = re.sub(f"[{re.escape(PUNCT_END)}]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        if text and not HALLUCINATION.search(text):
            lines.append({"t": round(cur[0].start, 2), "end": round(cur[-1].end, 2), "text": text})
        cur.clear()

    for w in words:
        if not w.word.strip():
            continue
        if cur:
            gap = w.start - cur[-1].end
            length = len("".join(x.word for x in cur).strip())
            if gap > 0.5 or length >= 18:
                flush()
        cur.append(w)
        if w.word.strip()[-1] in PUNCT_END:
            flush()
    flush()

    language = getattr(info, "language", lang) or ""
    if cfg.get("convert_to_taiwan", True) and language.startswith("zh"):
        for l in lines:
            l["text"] = to_taiwan(l["text"])

    lrc = to_lrc(lines)
    q = lyric_quality(stats, words, d, lines)
    print(f"[lyrics] 辨識可信度 {q}", flush=True)
    atomic_write_text(d / "lyrics_auto.lrc", lrc)      # 辨識結果一律保留，之後可以叫出來看
    with _META_LOCK:   # 讀出到寫回之間不讓別的執行緒改這首歌的資料
        meta = read_meta(d)
        keep_user = meta.get("lyrics_source") in ("online", "import", "claude") and (d / "lyrics.lrc").exists() \
            and (d / "lyrics.lrc").read_text("utf-8").strip()
        notes = [x for x in meta.get("notes", []) if "歌詞" not in x and "辨識" not in x]
        if lang_note:
            notes.append(lang_note + "。")
        if q["ok"]:
            if not keep_user:   # 使用者已經換成線上／匯入的歌詞時，重新辨識不覆蓋
                atomic_write_text(d / "lyrics.lrc", lrc)
                meta["lyrics_source"] = "whisper"
            meta["lyrics_unreliable"] = False
        else:
            # 辨識不可靠：不要把錯的歌詞放上去，留白並建議找線上歌詞
            if not keep_user:
                atomic_write_text(d / "lyrics.lrc", "")
                meta["lyrics_source"] = None
            meta["lyrics_unreliable"] = True
            notes.append("自動辨識的歌詞不夠準確（" + "、".join(q["reasons"]) + "），已先留白。建議用練唱畫面的「搜尋線上歌詞」或匯入歌詞檔。")
        meta["notes"] = notes
        meta["steps"]["lyrics"] = True
        meta["language"] = language
        meta["lyrics_quality"] = q
        write_meta(d, meta)


# ---------------------------------------------------------------- 移調／變速

def variant_dir(d: Path, key: int, speed: float) -> Path:
    if key == 0 and abs(speed - 1) < 1e-3:
        return d
    # v2：1.3 起分段處理（時間不漂移）；舊版的 v_k… 資料夾會在產生新版本時清掉
    return d / f"v2_k{key:+d}_s{int(round(speed * 100))}"


def shift_audio(y: np.ndarray, sr: int, semis: int, speed: float) -> np.ndarray:
    try:
        from pedalboard import time_stretch
        return time_stretch(np.ascontiguousarray(y, dtype=np.float32), sr,
                            stretch_factor=float(speed),
                            pitch_shift_in_semitones=float(semis))
    except ImportError:
        import librosa
        out = []
        for ch in y:
            z = ch
            if abs(speed - 1) > 1e-3:
                z = librosa.effects.time_stretch(z, rate=speed)
            if semis:
                z = librosa.effects.pitch_shift(z, sr=sr, n_steps=semis)
            out.append(z)
        return np.stack(out)


def shift_parts(n: int, sr: int, cpus: int = 0) -> list[tuple[int, int]]:
    """把一軌切成約 6 秒一段。

    好處有兩個：所有 CPU 核心可以一起算；而且移調演算法處理長音檔時時間會慢慢漂移
    （實測 40 秒累積約 45 毫秒），切短後每段重新對齊，伴奏和旋律線、歌詞不會越來越不同步。"""
    parts = max(1, -(-n // (6 * sr)))
    step = -(-n // parts)
    return [(a, min(n, a + step)) for a in range(0, n, step)]


def shift_piece(y: np.ndarray, sr: int, semis: int, speed: float, a: int, b: int) -> np.ndarray:
    """處理 [a, b) 這一段，前後多帶 1 秒讓接縫處的聲音和整段處理時一樣。"""
    pad = sr
    sa, sb = max(0, a - pad), min(y.shape[1], b + pad)
    return shift_audio(y[:, sa:sb], sr, semis, speed), sa


def join_pieces(pieces: list, bounds: list[tuple[int, int]], n: int, sr: int, speed: float) -> np.ndarray:
    """把各段接回去：交界處用 40 ms 交叉淡化，聽不出接縫。"""
    total = int(round(n / speed))
    ch = pieces[0][0].shape[0]
    out = np.zeros((ch, total), dtype=np.float32)
    xf = int(0.04 * sr)
    for i, ((seg, sa), (a, b)) in enumerate(zip(pieces, bounds)):
        g0 = int(round(a / speed)) - (xf // 2 if i else 0)
        g1 = int(round(b / speed)) + (xf - xf // 2 if i < len(bounds) - 1 else 0)
        g1 = min(g1, total)
        off = int(round(sa / speed))
        l0, l1 = g0 - off, g1 - off
        l1 = min(l1, seg.shape[1])
        g1 = g0 + (l1 - l0)
        w = np.ones(g1 - g0, dtype=np.float32)
        if i:
            w[:xf] = np.linspace(0, 1, xf, dtype=np.float32)
        if i < len(bounds) - 1:
            w[-xf:] = np.linspace(1, 0, xf, dtype=np.float32)
        out[:, g0:g1] += seg[:, l0:l1] * w
    return out


def step_variant(d: Path, key: int, speed: float, report) -> None:
    """產生移調／變速版本。三軌各切成多段，用所有 CPU 核心同時處理。"""
    from concurrent.futures import ThreadPoolExecutor, as_completed
    vd = variant_dir(d, key, speed)
    if vd == d:
        return
    vd.mkdir(exist_ok=True)
    for old in d.glob("v_k*"):   # 1.3 以前產生的版本（時間會漂移），用不到了
        shutil.rmtree(old, ignore_errors=True)
    todo = [s for s in STEMS if not stem_path(vd, s).exists()]
    if not todo:
        return
    label = f"移調 {key:+d}、速度 {int(speed * 100)}%"
    cpus = max(1, os.cpu_count() or 1)
    try:
        if live_status()["hold"]:   # 2.1.1 直播中：只用一半的 CPU 核心，留給直播軟體和 VTube Studio
            cpus = max(1, cpus // 2)
    except Exception:
        pass
    report(f"{label}：讀取音軌", 0.02)
    data = {}
    for stem in todo:
        y, sr = sf.read(str(stem_path(d, stem)), dtype="float32", always_2d=True)
        y = np.ascontiguousarray(y.T)
        data[stem] = (y, sr, shift_parts(y.shape[1], sr, cpus))
    total = sum(len(v[2]) for v in data.values())
    report(f"{label}：用 {min(cpus, total)} 個 CPU 核心同時處理", 0.05)
    results = {stem: [None] * len(v[2]) for stem, v in data.items()}
    done = 0
    ex = ThreadPoolExecutor(max_workers=cpus)
    try:
        futs = {ex.submit(shift_piece, y, sr, key, speed, a, b): (stem, i)
                for stem, (y, sr, parts) in data.items() for i, (a, b) in enumerate(parts)}
        for f in as_completed(futs):
            stem, i = futs[f]
            results[stem][i] = f.result()
            done += 1
            report(f"{label}：{done}/{total} 段（{min(cpus, total)} 核心同時處理）", 0.05 + 0.9 * done / total)
    finally:
        ex.shutdown(wait=True, cancel_futures=True)   # 取消時，還沒開始的段落直接丟掉
    for stem, (y, sr, parts) in data.items():
        out = join_pieces(results[stem], parts, y.shape[1], sr, speed)
        save_stem(vd, stem, out, sr)
    report(f"{label}：完成", 1.0)


def variant_ready(d: Path, key: int, speed: float) -> bool:
    vd = variant_dir(d, key, speed)
    return all(stem_path(vd, s).exists() for s in STEMS)


# ---------------------------------------------------------------- 錄音評分

def list_takes(d: Path) -> list[dict]:
    td = d / "takes"
    if not td.exists():
        return []
    out = []
    for p in td.glob("*.json"):
        try:
            t = json.loads(p.read_text("utf-8"))
            t.pop("pitch", None)
            # 評分中途程式被關掉：超過 5 分鐘還在「評分中」就標成失敗，讓使用者可以重新評分或刪除
            if t.get("status") == "scoring" and time.time() - t.get("created", 0) > 300 and not t.get("_job"):
                t["status"], t["error"] = "error", "評分沒有完成（可能是程式被關掉），可以按「重新評分」"
            out.append(t)
        except Exception:
            pass
    out.sort(key=lambda t: t.get("created", 0), reverse=True)
    return out


# ---------------------------------------------------------------- 日文假名標音（ふりがな）

# 用 SudachiPy（Apache-2.0）的小型字典判斷漢字讀音。第一次用到時才在背景安裝（約 45 MB），安裝檔也會先裝好。
KANA_PKGS = ["sudachipy==0.7.0", "sudachidict_small==20261015"]
KANA = {"state": "idle", "error": ""}
_kana_lock = threading.Lock()
_KANA_TK = None
_KANJI = "\u3400-\u4dbf\u4e00-\u9fff\uf900-\ufaff々〆ヶ"
_KANJI_RE = re.compile(f"[{_KANJI}]")
_PARTS_RE = re.compile(f"[{_KANJI}]+|[^{_KANJI}]+")


def kana_ready() -> bool:
    import importlib.util as u
    return u.find_spec("sudachipy") is not None and u.find_spec("sudachidict_small") is not None


def kana_status() -> dict:
    return {"state": "ready"} if kana_ready() else dict(KANA)


def start_kana_install() -> None:
    """在背景用 pip 安裝日文讀音字典（舊版升級上來的使用者沒有，不用重跑安裝）。"""
    with _kana_lock:
        if KANA["state"] == "installing" or kana_ready():
            return
        KANA.update(state="installing", error="")

    def run():
        import importlib
        import subprocess
        exe = Path(sys.executable)
        if exe.name.lower() == "pythonw.exe" and (exe.parent / "python.exe").exists():
            exe = exe.parent / "python.exe"
        try:
            r = subprocess.run([str(exe), "-m", "pip", "install", "--disable-pip-version-check", *KANA_PKGS],
                               capture_output=True, text=True, timeout=1800,
                               creationflags=0x08000000 if sys.platform == "win32" else 0)
            importlib.invalidate_caches()
            if r.returncode != 0 or not kana_ready():
                raise RuntimeError((r.stderr or r.stdout or "").strip().splitlines()[-1:] or ["安裝失敗"])
            KANA.update(state="done")
        except Exception as e:
            log_error("kana install", e)
            KANA.update(state="error", error=f"{type(e).__name__}: {e}")
    threading.Thread(target=run, daemon=True).start()


def _kata(s: str) -> str:
    return "".join(chr(ord(c) + 0x60) if "ぁ" <= c <= "ゖ" else c for c in s)


def _hira(s: str) -> str:
    return "".join(chr(ord(c) - 0x60) if "ァ" <= c <= "ヶ" else c for c in s)


def _ruby_word(surface: str, reading: str) -> list[list[str]]:
    """一個詞拆成「漢字＋讀音」與「假名」：沈む／シズム → [["沈","しず"],["む",""]]"""
    if not _KANJI_RE.search(surface) or not reading or reading == "*" or _KANJI_RE.search(reading):
        return [[surface, ""]]   # 字典查不到讀音（讀音還是漢字）：不標
    parts = _PARTS_RE.findall(surface)
    pat = "".join("(.+?)" if _KANJI_RE.match(p) else re.escape(_kata(p)) for p in parts)
    m = re.fullmatch(pat, _kata(reading))
    if not m:
        return [[surface, _hira(reading)]]
    out, g = [], iter(m.groups())
    for p in parts:
        out.append([p, _hira(next(g))] if _KANJI_RE.match(p) else [p, ""])
    return out


_JA_RE = re.compile(f"[\u3040-\u30ff{_KANJI}]")


_TW2JA: dict | None = None


def tw_to_ja(text: str) -> str:
    """把被轉成台灣繁體的日文字改回日文字形（1.5 以前的線上歌詞／匯入歌詞會把「来」改成「來」、「会」改成「會」）。
    對照表：日本常用漢字（JIS 第一水準）經過簡轉繁後變成的字 → 原本的日文字；只換逐字對應、不影響日文本來就用的字。"""
    global _TW2JA
    if _TW2JA is None:
        table: dict = {}
        try:
            from opencc import OpenCC
            cc = OpenCC("s2twp")
            level1 = []
            for hi in range(0x88, 0x99):
                for lo in list(range(0x40, 0x7F)) + list(range(0x80, 0xFD)):
                    if (hi, lo) < (0x88, 0x9F) or (hi, lo) > (0x98, 0x72):
                        continue
                    try:
                        level1.append(bytes([hi, lo]).decode("shift_jis"))
                    except UnicodeDecodeError:
                        pass
            jp = set(level1)
            for j in level1:
                c = cc.convert(j)
                if len(c) == 1 and c != j and c not in jp:
                    table.setdefault(c, j)
        except Exception as e:
            log_error("tw_to_ja", e)
        _TW2JA = table
    return "".join(_TW2JA.get(c, c) for c in text) if _TW2JA else text


def furigana(text: str) -> list[dict] | None:
    """一句日文歌詞拆成詞：[{"p": [[字, 讀音], …], "pt": 是否助詞}, …]。
    讀音只標在漢字上（空字串＝不用標）；羅馬拼音由前端依每個詞的讀音換算（助詞は／へ／を 念 wa／e／o）。
    沒有假名也沒有漢字的句子回傳 None。"""
    global _KANA_TK
    if not text or not _JA_RE.search(text):
        return None
    with _kana_lock:
        if _KANA_TK is None:
            from sudachipy import dictionary
            _KANA_TK = dictionary.Dictionary(dict="small").create()
        from sudachipy import tokenizer
        norm = tw_to_ja(text)
        toks = _KANA_TK.tokenize(norm, tokenizer.Tokenizer.SplitMode.C)
    out: list[dict] = []
    for t in toks:
        parts: list[list[str]] = []
        pos = t.begin()
        for b, r in _ruby_word(t.surface(), t.reading_form()):
            b, pos = text[pos:pos + len(b)], pos + len(b)   # 顯示原本的字
            if not r and parts and not parts[-1][1]:
                parts[-1][0] += b
            else:
                parts.append([b, r])
        tok = {"p": parts}
        if t.part_of_speech()[0] == "助詞":
            tok["pt"] = 1
        out.append(tok)
    if "".join(b for t in out for b, _ in t["p"]) != text:
        return None
    return out


def lyric_lines(d: Path) -> list[dict]:
    p = d / "lyrics.lrc"
    if not p.exists():
        return []
    return [l for l in parse_lrc(p.read_text("utf-8")) if l["t"] is not None]


def suggest_key(dd: np.ndarray, key: int) -> dict | None:
    """依你和原唱的音高差（已去掉八度差，-6～+6 半音）找出最常見的偏移，推算適合你的 KEY。

    大多數音都固定差同樣幾個半音，代表你習慣的音域和目前的 KEY 不合，而不是單純唱不準。"""
    v = dd[~np.isnan(dd)]
    if v.size < 150:
        return None
    b = np.round(v).astype(int)
    b[b == -6] = 6   # -6 和 +6 是同一個音
    vals, counts = np.unique(b, return_counts=True)
    i = int(np.argmax(counts))
    shift, share = int(vals[i]), float(counts[i] / b.size)
    # 只差 0～1 個半音多半是音準問題；差 2 個以上而且有一定比例的音都這樣，才建議換 KEY
    if abs(shift) < 2 or share < 0.3:
        return {"shift": 0, "share": round(share, 2), "suggest": key}
    return {"shift": shift, "share": round(share, 2), "suggest": (key + shift + 6) % 12 - 6}   # 保持在 ±6


TAKE_ALGO = 4   # 錄音分析方式的版本；舊版分析的錄音會自動重新評分
TOL_FRAMES = 10   # 評分的時間容錯：±0.1 秒


def score_take(d: Path, take_id: str, report) -> None:
    td = d / "takes"
    info = json.loads((td / f"{take_id}.json").read_text("utf-8"))
    report("分析你的錄音", 0.2)
    y16 = read_audio(td / f"{take_id}.wav", sr=PITCH_SR, mono=True)
    mine = track_pitch(y16, strict=True)
    ref, _ = load_pitch(d)
    key = int(info.get("key", 0))
    speed = float(info.get("speed", 1.0))
    start = float(info.get("start", 0.0))
    ref_shifted = ref + key

    # 每一個錄音影格對應的歌曲時間 = start + τ*speed + offset
    tau = np.arange(len(mine)) * HOP_S
    voiced = ~np.isnan(mine)

    def diffs(offset):
        ts = start + tau * speed + offset
        idx = np.round(ts / HOP_S).astype(int)
        ok = voiced & (idx >= 0) & (idx < len(ref_shifted))
        r = np.full(len(mine), np.nan, dtype=np.float32)
        r[ok] = ref_shifted[idx[ok]]
        dd = mine - r
        dd = dd - 12 * np.round(dd / 12)
        return ts, dd, r

    report("對齊時間", 0.7)
    best, best_off = -1, 0.0
    for off in np.arange(-0.5, 1.0001, 0.01):
        _, dd, _ = diffs(off)
        m = int(np.sum(np.abs(dd[~np.isnan(dd)]) < 0.5))
        if m > best:
            best, best_off = m, float(off)
    # 建議 KEY 用另一種對齊：找「最多音都差同樣幾個半音」的位置（整首唱在別的 KEY 時，
    # 上面「最準」的對齊會找錯位置）
    kbest, koff = -1, best_off
    for off in np.arange(-0.5, 1.0001, 0.02):
        _, dk, _ = diffs(off)
        b = np.round(dk[~np.isnan(dk)]).astype(int) % 12
        c = int(np.bincount(b, minlength=12).max()) if b.size else 0
        if c > kbest:
            kbest, koff = c, float(off)
    key_hint = suggest_key(diffs(koff)[1], key)
    ts, dd, r = diffs(best_off)
    # 時間容錯：每一格取前後 0.1 秒內最接近的原唱音高（換音稍早或稍晚不算走音）
    for k in list(range(-TOL_FRAMES, 0)) + list(range(1, TOL_FRAMES + 1)):
        _, d2, _ = diffs(best_off + k * HOP_S)
        better = np.isnan(dd) & ~np.isnan(d2) | (~np.isnan(d2) & (np.abs(d2) < np.abs(dd)))
        dd = np.where(better, d2, dd)
    valid = ~np.isnan(dd)
    cents = dd[valid] * 100
    octave = 0
    raw = (mine - r)[~np.isnan(mine - r)]
    if raw.size:
        octave = int(np.median(np.round(raw / 12)))

    per_line = []
    lines = lyric_lines(d)
    for i, l in enumerate(lines):
        end = lines[i + 1]["t"] if i + 1 < len(lines) else l["t"] + 8
        sel = valid & (ts >= l["t"]) & (ts < end)
        c = dd[sel] * 100
        if c.size < 10:
            continue
        per_line.append({
            "t": l["t"], "text": l["text"],
            "acc": round(float(np.mean(np.abs(c) <= 50)), 3),
            "mean": round(float(np.mean(c)), 1),
            "n": int(c.size),
        })

    if not cents.size:
        # 沒有偵測到歌聲：通常是麥克風沒收到音（音量太小、選錯裝置）或這段沒有唱
        peak = float(np.max(np.abs(y16))) if y16.size else 0.0
        info.update({"status": "empty", "acc": None, "mean": None, "frames": 0, "algo": TAKE_ALGO,
                     "error": "沒有偵測到歌聲" + ("：錄音幾乎沒有聲音，請確認麥克風選對、音量表有動" if peak < 0.02
                                                else "：這段時間原唱沒有旋律，或沒有唱到")})
        info.pop("pitch", None)
        atomic_write_text(td / f"{take_id}.json", json.dumps(info, ensure_ascii=False))
        return
    info.update({
        "offset": round(best_off, 2),
        "acc": round(float(np.mean(np.abs(cents) <= 50)), 3) if cents.size else None,
        "mean": round(float(np.mean(cents)), 1) if cents.size else None,
        "spread": round(float(np.mean(np.abs(cents))), 1) if cents.size else None,
        "frames": int(cents.size),
        "octave": octave,
        "lines": per_line,
        "pitch": {
            "t0": round(start + best_off, 3), "hop": round(HOP_S * speed, 5),
            "midi": [None if np.isnan(v) else round(float(v), 2) for v in mine],
        },
        "status": "done",
        "algo": TAKE_ALGO,
        "key_hint": key_hint,
    })
    atomic_write_text(td / f"{take_id}.json", json.dumps(info, ensure_ascii=False))


# ---------------------------------------------------------------- 背景工作

class JobCancelled(Exception):
    """工作被新的要求取代（例如又換了 KEY），中途停止。"""


# ---------------------------------------------------------------- 直播中暫停背景處理（2.1.0）
# 分離、辨識會把顯示卡吃滿，直播畫面和 VTube Studio 的模型追蹤會掉格。
# 偵測到直播軟體在跑、或練唱台的直播模式開著時，背景的重工作（升級、細修、最高品質、壓縮）先停住；結束後自動繼續。
# 直播中自己匯入的歌照常處理，但改用輕量模型（標準分離），處理快、佔顯示卡時間短，
# 直播結束後再自動用高品質模型補做一次（live_light 標記）。KEY 轉換、錄音評分照常。
LIVE_HOLD = {"apps": [], "live_seen": 0.0, "override_until": 0.0, "checked": 0.0}
_live_lock = threading.Lock()


def _process_names() -> set[str] | None:
    """目前所有程式的執行檔名稱（小寫）。2.3.2：直接問 Windows（不到 0.1 秒），
    不再用 tasklist：電腦很忙時 tasklist 會超過 10 秒逾時，結果「沒偵測到直播」。"""
    try:
        import ctypes
        from ctypes import wintypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)

        class PE(ctypes.Structure):
            _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD),
                        ("th32ProcessID", wintypes.DWORD), ("th32DefaultHeapID", ctypes.c_size_t),
                        ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                        ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long),
                        ("dwFlags", wintypes.DWORD), ("szExeFile", ctypes.c_wchar * 260)]
        k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
        snap = k32.CreateToolhelp32Snapshot(0x2, 0)
        if not snap or snap == wintypes.HANDLE(-1).value:
            return None
        try:
            e = PE()
            e.dwSize = ctypes.sizeof(PE)
            out = set()
            ok = k32.Process32FirstW(snap, ctypes.byref(e))
            while ok:
                out.add(e.szExeFile.lower())
                ok = k32.Process32NextW(snap, ctypes.byref(e))
            return out
        finally:
            k32.CloseHandle(snap)
    except Exception:
        return None


def _running_apps(names: list[str]) -> list[str] | None:
    """有在跑的直播軟體；查不到（不是「沒有」）時回傳 None，沿用上一次的結果。"""
    if sys.platform != "win32" or not names:
        return []
    procs = _process_names()
    how = "Windows API" if procs is not None else "tasklist"
    if _LIVE_LAST.get("how") != how:
        _LIVE_LAST["how"] = how
        print(f"[live] 直播軟體偵測方式：{how}", flush=True)
    if procs is None:
        try:
            flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            out = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True,
                                 timeout=20, creationflags=flags, encoding="utf-8", errors="ignore").stdout
        except Exception:
            return None
        low = out.lower()
        return [n for n in names if f'"{n.lower()}"' in low]
    return [n for n in names if n.lower() in procs]


# 2.3.2：顯示卡被其他程式（直播軟體、VTube Studio、遊戲、錄影）佔滿時，也不要再把重工作丟給顯示卡：
# 顯示卡記憶體不夠時 Windows 會把它搬到系統記憶體，分離慢幾十倍、整台電腦跟著卡。
GPU_LOAD = {"checked": 0.0, "free": None, "total": None, "util": None, "hot": 0, "stalled_until": 0.0}
GPU_MIN_FREE = 3.5e9    # 剩下的顯示卡記憶體少於這個就不用顯示卡分離
GPU_BUSY_UTIL = 85      # 其他程式已經把顯示卡用到這麼滿（連續兩次）也不用


def gpu_stalled_mark(seconds: float = 300) -> None:
    """顯示卡處理程式剛剛卡住：接下來一段時間先當作顯示卡很忙。"""
    GPU_LOAD["stalled_until"] = time.time() + seconds


def _sample_gpu_load() -> None:
    """用 nvidia-smi 看整張顯示卡（含其他程式）的用量。練唱台自己的顯示卡處理程式開著時不看（會算到自己）。"""
    try:
        import gpu_worker as G
        if G.POOL.count():
            return
    except Exception:
        pass
    try:
        flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu",
                              "--format=csv,noheader,nounits"], capture_output=True, text=True,
                             timeout=8, creationflags=flags).stdout.strip().splitlines()
        used, total, util = (float(x) for x in out[0].split(","))
    except Exception:
        return
    GPU_LOAD.update(free=(total - used) * 2**20, total=total * 2**20, util=util,
                    hot=GPU_LOAD["hot"] + 1 if util >= GPU_BUSY_UTIL else 0)


def gpu_busy_reason() -> str:
    now = time.time()
    if now < GPU_LOAD["stalled_until"]:
        return "顯示卡剛剛處理沒有進度（可能被其他程式佔滿）"
    if now - GPU_LOAD["checked"] > 20:
        GPU_LOAD["checked"] = now
        _sample_gpu_load()
    free = GPU_LOAD["free"]
    if free is not None and free < GPU_MIN_FREE:
        return f"顯示卡記憶體只剩 {free / 1e9:.1f} GB（被其他程式佔用）"
    if GPU_LOAD["hot"] >= 2:
        return f"顯示卡已被其他程式用到 {GPU_LOAD['util']:.0f}%"
    return ""


_LIVE_LAST = {"reason": None}


def _live_log(st: dict) -> dict:
    r = st.get("reason") or ""
    key = ("gpu" if st.get("gpu") else r)   # 顯示卡用量的數字每次都不一樣，只在狀態改變時記
    if key != _LIVE_LAST["reason"]:
        _LIVE_LAST["reason"] = key
        print(f"[live] {r}：新的分離、音高分析改用 CPU，背景工作暫停" if r else "[live] 沒有偵測到直播或顯示卡忙碌：恢復正常處理", flush=True)
    return st


def live_touch() -> None:
    """練唱台畫面在直播模式時每隔幾秒呼叫一次。"""
    LIVE_HOLD["live_seen"] = time.time()


def live_override(seconds: float = 3600) -> None:
    """使用者按「直播中也照常處理」：暫時不管偵測結果。"""
    LIVE_HOLD["override_until"] = time.time() + seconds


def live_status(cfg: dict | None = None) -> dict:
    """現在該不該暫停重工作，以及原因（給狀態列顯示）。每 15 秒才真的去查一次程式清單。
    2.3.2：顯示卡被其他程式佔滿時也算（不管有沒有開「直播中暫停背景處理」）。"""
    cfg = cfg or load_config()
    now = time.time()
    if now < LIVE_HOLD["override_until"]:
        return _live_log({"hold": False, "reason": "", "apps": [], "override": True})
    if cfg.get("pause_while_streaming", True):
        with _live_lock:
            if now - LIVE_HOLD["checked"] > 15:
                LIVE_HOLD["checked"] = now
                got = _running_apps(cfg.get("streaming_apps") or [])
                if got is not None:
                    LIVE_HOLD["apps"] = got
        apps = list(LIVE_HOLD["apps"])
        if apps:
            return _live_log({"hold": True, "reason": "偵測到 " + "、".join(a.rsplit(".", 1)[0] for a in apps), "apps": apps})
        if now - LIVE_HOLD["live_seen"] < 20:
            return _live_log({"hold": True, "reason": "直播模式開著", "apps": []})
    busy = gpu_busy_reason()
    if busy:
        return _live_log({"hold": True, "reason": busy, "apps": [], "gpu": True})
    return _live_log({"hold": False, "reason": "", "apps": []})


# 直播中要先停住的背景工作（使用者沒有在等結果的）
LIVE_BACKGROUND = {"upgrade", "fine", "compact", "max", "leadfix"}


class JobDeferred(JobCancelled):
    """背景工作做到一半開播：放回佇列，直播結束後從頭再做（不佔著佇列擋住直播中的匯入）。"""


def wants_hq(cfg: dict) -> bool:
    """這台電腦平常會不會用高品質（較慢、較吃顯示卡）的分離模型。"""
    def hq(key):
        q = cfg.get(key, "auto")
        return q == "high" or (q == "auto" and cuda_ok())
    return hq("vocal_quality") or hq("karaoke_quality")


def hq_models_ready(cfg: dict) -> bool:
    """平常要用的高品質模型都下載好了（直播後補做才有意義）。"""
    def hq(key):
        q = cfg.get(key, "auto")
        return q == "high" or (q == "auto" and cuda_ok())
    return (not hq("vocal_quality") or hqv_ready()) and (not hq("karaoke_quality") or hq_ready())


LIGHT_CFG = {"vocal_quality": "standard", "karaoke_quality": "standard", "auto_recognize_lyrics": False, "_cpu": True}


def live_light_cfg(cfg: dict) -> dict | None:
    """直播中（或顯示卡被其他程式佔滿）：回傳要改用的輕量設定（CPU＋標準模型）；否則 None。"""
    if not live_status(cfg)["hold"]:
        return None
    return dict(LIGHT_CFG)


# 2.1.1：可以和下一首重疊處理的工作（分離完、還在分析旋律時，下一首就先開始分離）
PIPELINE_LABELS = {"all", "youtube", "upgrade", "fine", "max"}


PARALLEL_MIN_VRAM = 10e9   # 2.3.1：顯示卡記憶體 10 GB 以上才同時分離兩首


def parallel_ok() -> bool:
    """這張顯示卡適不適合兩首同時分離。8 GB 的顯示卡實測：兩首互搶運算，每首反而慢一倍以上，
    所以記憶體不到 10 GB 就不啟用（設定勾了也一樣）。還不知道顯示卡資訊時先當作不適合。"""
    info = gpu_info(block=False) or {}
    return bool(info.get("ok")) and (info.get("vram") or 0) >= PARALLEL_MIN_VRAM


class Jobs:
    """重工作的佇列。預設一次分離一首；分離完的那首在分析旋律、存檔時，下一首可以先開始分離（pipeline）。
    設定開啟「閒置時兩首同時分離」且沒在直播時，最多兩首同時分離。"""

    def __init__(self, live_safe: bool = False, pipeline: bool = False):
        self.pending: list[tuple[str, object]] = []   # 排隊中的工作（可以調整順序）
        self.state: dict[str, dict] = {}
        self.lock = threading.Lock()
        self.cv = threading.Condition(self.lock)
        self.live_safe = live_safe   # 直播中要不要暫停（只有重工作的佇列開）
        self.pipeline = pipeline     # 可不可以重疊處理
        self.holding = ""            # 目前因為直播暫停中的原因（空字串＝沒有暫停）
        self.running: dict[str, str] = {}   # 執行中的工作 → "gpu"（分離中）／"post"（分析、存檔）
        self.oom_until = 0.0                # 顯示卡記憶體不夠過：這個時間之前只同時分離一首
        self.paused = False                 # 「全部暫停」：處理歌曲的工作都先停（錄音評分、匯出照常）
        if live_safe:
            try:
                self.paused = bool(load_config().get("queue_paused"))
            except Exception:
                pass
        threading.Thread(target=self._run, daemon=True).start()

    def _held(self, jid: str) -> bool:
        st = self.state.get(jid) or {}
        return self.live_safe and st.get("label") in LIVE_BACKGROUND

    def _pausable(self, st: dict) -> bool:
        return st.get("kind") == "process"

    def set_paused(self, on: bool) -> None:
        """全部暫停／繼續處理。暫停時處理中的歌在下一個進度點停下、讓出顯示卡，繼續後從頭重做這首。"""
        with self.cv:
            self.paused = on
            if on:
                for j in self.running:
                    if self._pausable(self.state.get(j) or {}):
                        self.state[j]["pause"] = True
            self.cv.notify_all()
        try:
            cfg = load_config()
            cfg["queue_paused"] = on
            save_config(cfg)
        except Exception as e:
            log_error("save queue_paused", e)

    def _slots(self, hold: bool) -> int:
        """同時可以分離幾首。"""
        if not self.pipeline or hold or time.time() < self.oom_until:
            return 1
        try:
            return 2 if load_config().get("parallel_separation") and parallel_ok() else 1
        except Exception:
            return 1

    def _can_start(self, st: dict, slots: int) -> bool:
        if not self.running:
            return True
        if not self.pipeline or st.get("kind") != "process" or st.get("label") not in PIPELINE_LABELS:
            return False   # 其他工作（錄音評分、匯出、壓縮）等前面的都做完才開始
        others = [self.state.get(j) or {} for j in self.running]
        if any(o.get("kind") != "process" or o.get("label") not in PIPELINE_LABELS for o in others):
            return False
        if st.get("song") and any(o.get("song") == st.get("song") for o in others):
            return False   # 同一首歌不同時處理
        gpu = sum(1 for v in self.running.values() if v == "gpu")
        return gpu < slots and len(self.running) < slots + 1

    def _next(self, hold: bool, slots: int):
        """挑下一個要做的工作。直播中跳過背景工作，先做使用者在等的（例如直播中匯入的歌）。
        回傳 (jid, fn, 只剩直播中暫停的工作)。"""
        for n, (jid, fn) in enumerate(self.pending):
            st = self.state.get(jid) or {}
            if st.get("status") == "cancelled":
                self.pending.pop(n)
                return jid, None, False
            if hold and st.get("label") in LIVE_BACKGROUND:
                continue
            if self.paused and self._pausable(st):
                continue
            if not self._can_start(st, slots):
                return None, None, False   # 依順序等前面的，不插隊
            self.pending.pop(n)
            return jid, fn, False
        return None, None, bool(self.pending) and not self.paused

    def submit(self, kind: str, song_id: str, fn, label: str = "") -> str:
        jid = uuid.uuid4().hex[:10]
        with self.cv:
            old = time.time() - 3600   # 結束超過一小時的工作紀錄不用再留（大量匯入時會累積上千筆）
            for k in [k for k, j in self.state.items() if j.get("finished", old + 1) < old]:
                del self.state[k]
            self.state[jid] = {"id": jid, "kind": kind, "song": song_id, "label": label,
                               "status": "queued", "step": "排隊中", "progress": 0.0,
                               "error": None, "created": time.time()}
            self.pending.append((jid, fn))
            self.cv.notify_all()
        return jid

    def move(self, jid: str, delta: int) -> bool:
        """調整排隊順序：delta = -1 往前、+1 往後（只在還沒開始的工作之間移動）。"""
        with self.lock:
            ids = [j for j, _ in self.pending if self.state[j]["status"] == "queued"]
            if jid not in ids:
                return False
            i = ids.index(jid)
            k = i + (1 if delta > 0 else -1)
            if not 0 <= k < len(ids):
                return False
            a = next(n for n, (j, _) in enumerate(self.pending) if j == jid)
            b = next(n for n, (j, _) in enumerate(self.pending) if j == ids[k])
            self.pending[a], self.pending[b] = self.pending[b], self.pending[a]
            return True

    def to_front(self, jid: str) -> bool:
        """移到佇列最前面（下一個開始處理）。"""
        with self.cv:
            for n, (j, fn) in enumerate(self.pending):
                if j == jid and self.state[j]["status"] == "queued":
                    self.pending.insert(0, self.pending.pop(n))
                    self.state[j]["priority"] = True
                    self.cv.notify_all()
                    return True
        return False

    def upcoming(self, label: str, n: int) -> list[str]:
        """排在最前面、還沒開始的 n 個工作的歌曲編號（只看指定種類）。"""
        with self.lock:
            out = []
            for j, _ in self.pending:
                st = self.state[j]
                if st["status"] == "queued" and st["label"] == label and not st.get("cancel"):
                    out.append(st["song"])
                    if len(out) >= n:
                        break
            return out

    def order(self) -> dict[str, int]:
        with self.lock:
            ids = [j for j, _ in self.pending if self.state[j]["status"] == "queued"]
        return {j: n for n, j in enumerate(ids)}

    def update(self, jid: str, **kw) -> None:
        with self.lock:
            j = self.state.get(jid)
            if j is None:
                return
            if kw.get("status") in ("done", "error", "cancelled"):
                kw.setdefault("finished", time.time())
            j.update(kw)

    def get(self, jid: str) -> dict | None:
        with self.lock:
            j = self.state.get(jid)
            return dict(j) if j else None

    def active(self) -> list[dict]:
        cutoff = time.time() - 600
        order = self.order()
        with self.lock:
            out = [dict(j, order=order.get(j["id"])) for j in self.state.values() if j["status"] != "cancelled"
                   and (j["status"] in ("queued", "running") or j.get("finished", j["created"]) > cutoff)]
        # 執行中的在前，排隊中的依目前順序
        return sorted(out, key=lambda j: (j["status"] != "running", j["order"] if j["order"] is not None else -1,
                                          j["created"]))

    def cancel_queued(self, kind: str, song_id: str, keep_label: str) -> int:
        """取消同一首歌還在排隊、但已經不需要的工作（例如連按好幾次 KEY 時中間的那些）。"""
        n = 0
        with self.lock:
            for j in self.state.values():
                if j["kind"] == kind and j["song"] == song_id and j["label"] != keep_label:
                    if j["status"] == "queued":
                        j.update(status="cancelled", step="已取消")
                        n += 1
                    elif j["status"] == "running":
                        j["cancel"] = True   # 執行中的工作會在下一段處理完時停下
                        n += 1
        return n

    def cancel(self, jid: str) -> bool:
        """使用者按「停止」：排隊中的直接取消；執行中的在下一個檢查點停下。"""
        with self.lock:
            j = self.state.get(jid)
            if not j:
                return False
            if j["status"] == "queued":
                j.update(status="cancelled", step="已取消")
                return True
            if j["status"] == "running":
                j["cancel"] = True
                j["step"] = "停止中…"
                return True
        return False

    def position(self, jid: str) -> int:
        """前面還有幾個工作。"""
        with self.lock:
            me = self.state.get(jid)
            if not me or me["status"] != "queued":
                return 0
            ahead = 0
            for j, _ in self.pending:
                if j == jid:
                    break
                if self.state[j]["status"] == "queued":
                    ahead += 1
            return ahead + sum(1 for j in self.state.values() if j["status"] == "running")

    def find(self, kind: str, song_id: str, label: str = "") -> str | None:
        with self.lock:
            for j in self.state.values():
                if (j["kind"], j["song"], j["label"]) == (kind, song_id, label) \
                        and j["status"] in ("queued", "running") and not j.get("cancel"):
                    return j["id"]
        return None

    def _run(self):
        while True:
            st = live_status() if self.live_safe else {"hold": False}   # 查程式清單可能要幾秒，不要鎖著佇列查
            slots = self._slots(st["hold"])
            with self.cv:
                jid, fn, held = self._next(st["hold"], slots)
                if jid is None:
                    if held and not self.running:   # 只剩直播中暫停的背景工作
                        reason = st.get("reason") or "直播中"
                        self.holding = reason
                        for j, _ in self.pending:
                            js = self.state.get(j)
                            if js and js["status"] == "queued":
                                js["step"] = f"直播中暫停（{reason}），結束後自動繼續"
                    elif not held:
                        self.holding = ""
                    if self.paused:
                        for j, _ in self.pending:
                            js = self.state.get(j)
                            if js and js["status"] == "queued" and self._pausable(js):
                                js["step"] = "已全部暫停"
                    self.cv.wait(5 if (self.pending or self.running) else None)
                    continue
                self.holding = ""
                if fn is None or self.state[jid]["status"] == "cancelled":
                    continue
                self.running[jid] = "gpu"
                self.state[jid].update(status="running", step="開始")
            threading.Thread(target=self._work, args=(jid, fn), daemon=True, name=f"job-{jid}").start()

    def _work(self, jid: str, fn) -> None:
        def report(step, p, jid=jid):
            if (self.state.get(jid) or {}).get("cancel"):
                raise JobCancelled()
            self.update(jid, step=step, progress=round(p, 3))
            if (self.state.get(jid) or {}).get("pause"):
                raise JobDeferred("paused")   # 按了全部暫停：讓出顯示卡，繼續後重做這首
            if self._held(jid) and live_status()["hold"]:
                raise JobDeferred()   # 做到一半開播：讓出顯示卡，直播後重做

        def gpu_done():
            """分離做完了（接下來是分析旋律、存檔）：讓下一首先開始分離。"""
            with self.cv:
                if jid in self.running:
                    self.running[jid] = "post"
                    self.cv.notify_all()
        report.gpu_done = gpu_done

        def check(jid=jid):
            """顯示卡子程式在跑、沒有進度回報時，也要能馬上停止／暫停。"""
            st = self.state.get(jid) or {}
            if st.get("cancel"):
                raise JobCancelled()
            if st.get("pause"):
                raise JobDeferred("paused")
        import gpu_worker as G
        G.CURRENT.check = check
        try:
            fn(report)
            self.update(jid, status="done", step="完成", progress=1.0)
        except JobDeferred as e:
            paused = bool(e.args) and e.args[0] == "paused"
            with self.cv:
                self.state[jid].pop("pause", None)
                self.state[jid].update(status="queued", progress=0.0,
                                       step="已全部暫停，繼續後重新開始這首" if paused else "直播中暫停，結束後重新開始")
                if paused:
                    self.pending.insert(0, (jid, fn))   # 繼續處理時先做這首
                else:
                    self.pending.append((jid, fn))
            free_gpu()
        except JobCancelled:
            self.update(jid, status="cancelled", step="已取消")
        except UserError as e:
            self.update(jid, status="error", error=str(e))
        except Exception as e:
            traceback.print_exc()
            if "out of memory" in str(e).lower() and not self.state[jid].get("oom_retry") and len(self.running) > 1:
                # 兩首同時處理時顯示卡記憶體不夠：接下來十分鐘只處理一首，這首排回最前面重做
                self.oom_until = time.time() + 600
                free_gpu()
                with self.cv:
                    self.state[jid].update(status="queued", step="顯示卡記憶體不夠，稍後重新處理", progress=0.0, oom_retry=True)
                    self.pending.insert(0, (jid, fn))
                log_error("parallel separation OOM", e)
            else:
                self.update(jid, status="error", error=f"{type(e).__name__}: {e}")
        finally:
            G.CURRENT.check = None
            with self.cv:
                self.running.pop(jid, None)
                (self.state.get(jid) or {}).pop("pause", None)
                self.cv.notify_all()


def process_song(song_id: str, steps: list[str], cfg_override: dict | None = None):
    def run(report):
        d = song_dir(song_id)
        cfg = dict(load_config(), **(cfg_override or {}))
        meta = read_meta(d)
        recheck = "import" not in steps and ("separate" in steps or "pitch" in steps) and (d / "lyrics.lrc").exists()
        fit_before = lyrics_timing_fit(d) if recheck else None   # 重新分離前，歌詞和舊的歌聲位置吻合度
        if "import" in steps or not meta["steps"].get("import"):
            step_import(d, report)
        # 2.3.2：匯入時（不自動辨識）先上網抓歌詞，和分離同時進行；分離完先套用歌詞，
        # 再用歌詞逐句檢查主唱有沒有被分到合音、搬回來，最後才分析旋律——一次做好，不用事後再補一輪。
        early = None
        if ("lyrics" in steps and "import" in steps and "separate" in steps
                and not cfg.get("auto_recognize_lyrics", False)
                and not (d / "lyrics.lrc").exists() and not meta.get("lyrics_source")):
            early = _early_lyrics(read_meta(d))
        if "separate" in steps:
            # 2.3.2：直播中不管是匯入、優先處理或重新處理，都改用 CPU＋標準模型（不碰顯示卡），下播後自動補做
            light = live_light_cfg(cfg)
            if light:   # 直播中匯入：先用輕量模型，直播結束後自動補做高品質
                cfg.update(light)
                update_meta(d, live_light=True)
                report("直播中：改用輕量模型分離（下播後自動補做高品質）", 0.1)
            step_separate(d, cfg, report)
            if not light and meta.get("live_light"):
                update_meta(d, live_light=False)   # 高品質補做完成
            recognize = "lyrics" in steps and not ("import" in steps and not cfg.get("auto_recognize_lyrics", False))
            if not recognize:   # 接下來只剩分析旋律、存檔：下一首可以先開始分離
                getattr(report, "gpu_done", lambda: None)()
            measure_stem_loudness(d)   # 各音軌響度，播放時依開著的音軌調整音量
        if early is not None:
            early = _apply_early_lyrics(d, cfg, early, report)
        if "pitch" in steps:
            step_pitch(d, cfg, report)
        if recheck:
            try:
                check_lyrics_after_reseparate(d, fit_before)
            except Exception as e:
                log_error("check_lyrics_after_reseparate", e)
        if early:
            pass   # 歌詞已經在分離後套用（或確定找不到），不用再排背景搜尋
        elif "lyrics" in steps and "import" in steps and not cfg.get("auto_recognize_lyrics", False):
            # 1.6.5：匯入時預設不自動辨識（準度不穩、又佔顯示卡），直接套用線上搜尋到的第一個版本
            report("排入線上歌詞搜尋（背景進行）", 0.97)
            with _META_LOCK:   # 讀出到寫回之間不讓別的執行緒改這首歌的資料
                meta = read_meta(d)
                meta["steps"]["lyrics"] = True
                meta["lyrics_recognized"] = False
                meta["notes"] = [x for x in meta.get("notes", []) if "歌詞" not in x and "辨識" not in x] + [
                    "正在背景搜尋線上歌詞，找到後會自動套用。"]
                write_meta(d, meta)
            search_lyrics_later(d, cfg, apply_first=True)
        elif "lyrics" in steps:
            try:
                step_lyrics(d, cfg, report)   # 歌詞相關的提示由 step_lyrics 自己更新
            except JobDeferred:
                raise
            except JobCancelled:
                with _META_LOCK:   # 讀出到寫回之間不讓別的執行緒改這首歌的資料
                    meta = read_meta(d)
                    meta["notes"] = [x for x in meta.get("notes", []) if "歌詞" not in x] + [
                        "已停止辨識歌詞。可以在練唱畫面貼上或搜尋歌詞，或用「重新處理 → 只重新辨識歌詞」再試一次。"]
                    write_meta(d, meta)
                raise
            except Exception as e:
                traceback.print_exc()
                with _META_LOCK:   # 讀出到寫回之間不讓別的執行緒改這首歌的資料
                    meta = read_meta(d)
                    meta["notes"] = [x for x in meta.get("notes", []) if "歌詞" not in x] + [
                        f"歌詞辨識沒有完成：{e}。可以在練唱畫面貼上歌詞。"]
                    write_meta(d, meta)
            # 線上歌詞搜尋要等網站回應（網站忙的時候可能要幾十秒），改在背景做，不卡住下一首歌的處理
            report("排入線上歌詞搜尋（背景進行）", 0.97)
            search_lyrics_later(d, cfg)
    return run


_LYRICS_POOL = None
_EARLY_POOL = None


def _early_lyrics(meta: dict):
    """在背景先上網搜尋歌詞（只抓資料，不需要分離結果）。回傳 Future。"""
    global _EARLY_POOL
    from concurrent.futures import ThreadPoolExecutor
    if _EARLY_POOL is None:
        _EARLY_POOL = ThreadPoolExecutor(max_workers=2, thread_name_prefix="lyrics-early")
    return _EARLY_POOL.submit(fetch_candidates, meta, "", None, 45)


def _apply_early_lyrics(d: Path, cfg: dict, fut, report) -> bool:
    """分離完：等先前的歌詞搜尋（最多再等 45 秒），套用最相符的版本，再逐句檢查主唱。
    回傳 True＝這一步處理完了（套用、或確定沒有相符的）；False＝搜尋失敗，交回原本的背景搜尋。"""
    report("套用線上歌詞、檢查主唱有沒有被分到合音", 0.62)
    import gpu_worker as G
    check = getattr(G.CURRENT, "check", None)
    end = time.time() + 45
    while not fut.done() and time.time() < end:
        if check:
            check()
        time.sleep(0.3)
    if not fut.done():
        return False
    try:
        pre = fut.result()
        find_and_store_candidates(d, cfg, apply_first=True, pre=pre)
    except Exception as e:
        log_error("early lyrics", e)
        return False
    with _META_LOCK:
        meta = read_meta(d)
        meta["steps"]["lyrics"] = True
        meta.setdefault("lyrics_recognized", False)
        write_meta(d, meta)
    if (d / "lyrics.lrc").exists():
        try:
            fix_lead_gaps(d)   # 有歌詞了：逐句檢查一次（分離時只看得到整段的）
        except Exception as e:
            log_error("fix_lead_gaps(early)", e)
    return True


def search_lyrics_later(d: Path, cfg: dict, apply_first: bool = False) -> None:
    """在背景依序搜尋線上歌詞（一次一首），結果出來後歌曲庫會標「有線上歌詞可選」。
    apply_first：沒有自動辨識的歌，直接套用最符合的第一個版本。"""
    global _LYRICS_POOL
    from concurrent.futures import ThreadPoolExecutor
    if _LYRICS_POOL is None:
        _LYRICS_POOL = ThreadPoolExecutor(max_workers=1, thread_name_prefix="lyrics-search")

    def run():
        try:
            if d.exists():
                find_and_store_candidates(d, cfg, apply_first=apply_first)
        except Exception as e:
            traceback.print_exc()   # 沒有網路或網站不通時，維持辨識結果
            if apply_first and d.exists():
                _lyrics_note(d, f"線上歌詞搜尋失敗（{e}）。可以在練唱畫面按「搜尋線上歌詞」再試一次，或用「重新處理 → 只重新辨識歌詞」自動辨識。")
    _LYRICS_POOL.submit(run)


# 啟動時就把 ffmpeg 放進 PATH：AI 分離（audio-separator）與 m4a/mp3 讀取都需要
try:
    ffmpeg_path()
except Exception:
    traceback.print_exc()


# ---------------------------------------------------------------- 線上歌詞搜尋（LRCLIB）

LRCLIB = "https://lrclib.net/api/search"
_NOISE = re.compile(r"(official|music\s*video|lyric[s]?|audio|mv|m/v|4k|hd|hq|live|cover|翻唱|清唱|純人聲|人聲|"
                    r"a\s*cappella|acapella|vocal[s]?\s*only|伴奏|動態歌詞|歌詞版?|完整版|高音質|官方|字幕)", re.I)


def lyric_queries(meta: dict) -> list[str]:
    """從歌名與 YouTube 資訊猜幾個搜尋字串。"""
    out = []
    track, artist = meta.get("yt_track") or "", meta.get("yt_artist") or ""
    if track:
        out.append(f"{track} {artist}".strip())
    title = meta.get("title") or ""
    inner = re.findall(r"[【「《『\[]([^】」》』\]]+)[】」》』\]]", title)
    for x in inner:
        if not _NOISE.search(x):
            out.append(x.strip())
    t = re.sub(r"[【「《『\[(（][^】」》』\])）]*[】」》』\])）]", " ", title)
    t = _NOISE.sub(" ", t)
    t = re.sub(r"[|｜/\\_~\-–—:：#]+", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    good_inner = [x.strip() for x in inner if not _NOISE.search(x)]
    if good_inner and t:
        out.insert(len(out) - len(good_inner), f"{good_inner[0]} {t}")   # 歌名＋歌手 優先
    if t:
        out.append(t)
    seen, uniq = set(), []
    for q in out:
        k = q.lower()
        if q and k not in seen:
            seen.add(k)
            uniq.append(q)
    return uniq[:3]


def to_simplified(text: str) -> str:
    try:
        from opencc import OpenCC
    except ImportError:
        return text
    for conf in ("t2s", "t2s.json"):
        try:
            return OpenCC(conf).convert(text)
        except Exception:
            continue
    return text


_CREDIT = re.compile(r"^\s*(作词|作曲|编曲|作詞|編曲|制作人|製作人|制作|製作|监制|監製|演唱|原唱|词|詞|曲|混音|录音|錄音|和声|和聲|"
                     r"吉他|贝斯|貝斯|鼓|弦乐|弦樂|出品|发行|發行|OP|SP|Lyrics|Composer|Arranger|Producer)\s*[:：]", re.I)


def clean_lrc(text: str, track: str = "", artist: str = "") -> str:
    """去掉作詞作曲等資訊行、「歌手 - 歌名」標題行與空白行。"""
    out = []
    title_lines = {f"{artist} - {track}", f"{track} - {artist}"} if track and artist else set()
    for line in (text or "").splitlines():
        body = re.sub(r"\[[^\]]*\]", "", line).strip()
        if not body or _CREDIT.match(body) or body in title_lines or \
                re.match(r"^\[(ti|ar|al|by|offset|length|re|id|hash|sign|qq|total):", line, re.I):
            continue
        out.append(line.strip())
    return "\n".join(out)


def _norm(text: str) -> str:
    """比對用：轉簡體、只留文字。"""
    text = re.sub(r"\[[^\]]*\]|<[^>]*>", "", text or "")
    return "".join(ch for ch in to_simplified(text).lower() if ch.isalnum())


def similarity(recognized: str, candidate: str) -> float | None:
    """辨識結果中，有多少比例的字依序出現在候選歌詞裡（0–1）。"""
    import difflib
    a, b = _norm(recognized), _norm(candidate)
    if len(a) < 6 or not b:
        return None
    m = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return round(sum(x.size for x in m.get_matching_blocks()) / len(a), 3)


USER_AGENT = "LeihuLiantai/2.3.2 (+https://github.com/LeihuTW/Leihu-singsing)"   # HTTP 標頭只能用英文字


def _http_json(url: str, headers: dict | None = None, timeout: int = 8):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT, **(headers or {})})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode("utf-8", "replace"))


def log_error(where: str, err: Exception) -> None:
    """把連線等錯誤寫進 logs/lyrics.log，方便之後查原因。"""
    try:
        (ROOT / "logs").mkdir(exist_ok=True)
        with open(ROOT / "logs" / "lyrics.log", "a", encoding="utf-8") as f:
            f.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {where}: {type(err).__name__}: {err}\n")
    except Exception:
        pass


def search_lrclib(query: str) -> list[dict]:
    import urllib.parse
    data = _http_json("https://lrclib.net/api/search?" + urllib.parse.urlencode({"q": query}), timeout=12)
    out = []
    for it in data if isinstance(data, list) else []:
        if it.get("instrumental") or not (it.get("plainLyrics") or it.get("syncedLyrics")):
            continue
        out.append({"source": "LRCLIB", "id": f"lrclib:{it.get('id')}",
                    "track": it.get("trackName") or it.get("name") or "", "artist": it.get("artistName") or "",
                    "album": it.get("albumName") or "", "duration": it.get("duration"),
                    "synced": clean_lrc(it.get("syncedLyrics") or ""), "plain": it.get("plainLyrics") or ""})
    return out


def search_netease(query: str, limit: int = 6) -> list[dict]:
    """網易雲音樂（中國最大的歌詞來源之一）：先搜尋歌曲，再取每首的同步歌詞。"""
    import urllib.parse
    h = {"Referer": "https://music.163.com/"}
    data = _http_json("https://music.163.com/api/search/get/web?" +
                      urllib.parse.urlencode({"s": query, "type": 1, "offset": 0, "limit": limit}), h)
    out = []
    for song in ((data or {}).get("result") or {}).get("songs") or []:
        try:
            lyr = _http_json(f"https://music.163.com/api/song/lyric?id={song['id']}&lv=1&kv=1&tv=-1", h)
        except Exception:
            continue
        synced = clean_lrc(((lyr or {}).get("lrc") or {}).get("lyric") or "")
        if not synced or "纯音乐" in synced:
            continue
        has_time = bool(re.search(r"\[\d+:\d+", synced))
        plain = "\n".join(re.sub(r"\[[^\]]*\]", "", l).strip() for l in synced.splitlines()).strip()
        out.append({"source": "網易雲音樂", "id": f"netease:{song['id']}", "track": song.get("name") or "",
                    "artist": " / ".join(a.get("name", "") for a in song.get("artists") or []),
                    "album": (song.get("album") or {}).get("name") or "",
                    "duration": round((song.get("duration") or 0) / 1000, 1) or None,
                    "synced": synced if has_time else "", "plain": plain})
    return out


def search_kugou(query: str, limit: int = 5) -> list[dict]:
    """酷狗音樂：先搜尋歌曲，再用歌曲 hash 找歌詞並下載 LRC。"""
    import base64
    import urllib.parse
    data = _http_json("http://mobilecdn.kugou.com/api/v3/search/song?" +
                      urllib.parse.urlencode({"format": "json", "keyword": query, "page": 1, "pagesize": limit}))
    out = []
    for song in (((data or {}).get("data") or {}).get("info") or [])[:limit]:
        try:
            dur = int(song.get("duration") or 0)
            cands = _http_json("http://krcs.kugou.com/search?" + urllib.parse.urlencode(
                {"ver": 1, "man": "yes", "client": "mobi", "keyword": "", "duration": dur * 1000,
                 "hash": song.get("hash", "")})).get("candidates") or []
            if not cands:
                continue
            c = cands[0]
            lyr = _http_json("http://lyrics.kugou.com/download?" + urllib.parse.urlencode(
                {"ver": 1, "client": "pc", "id": c["id"], "accesskey": c["accesskey"], "fmt": "lrc", "charset": "utf8"}))
            raw = base64.b64decode(lyr.get("content") or "").decode("utf-8", "replace")
        except Exception as e:
            log_error("kugou lyric", e)
            continue
        track, artist = song.get("songname") or "", song.get("singername") or ""
        synced = clean_lrc(raw, track, artist)
        if not synced:
            continue
        plain = "\n".join(re.sub(r"\[[^\]]*\]", "", l).strip() for l in synced.splitlines()).strip()
        out.append({"source": "酷狗音樂", "id": f"kugou:{song.get('hash')}", "track": track, "artist": artist,
                    "album": song.get("album_name") or "", "duration": dur or None,
                    "synced": synced if re.search(r"\[\d+:\d+", synced) else "", "plain": plain})
    return out


def recognized_text(d: Path) -> str:
    p = d / "lyrics_auto.lrc"
    return "\n".join(l["text"] for l in parse_lrc(p.read_text("utf-8"))) if p.exists() else ""


def find_candidates(d: Path, cfg: dict, query: str = "", check=None,
                    budget: float = 45, pre: tuple | None = None) -> tuple[list[str], list[dict]]:
    """用歌名（原文與簡體）到 LRCLIB、網易雲和酷狗搜尋，與辨識結果比對後排序。
    pre：先前已經抓好的（搜尋字串, 原始結果）——2.3.2 匯入時在分離的同時先上網抓，分離完再排序、套用。"""
    tried, found = pre if pre else fetch_candidates(read_meta(d), query, check, budget)
    return rank_candidates(d, cfg, tried, found)


def fetch_candidates(meta: dict, query: str = "", check=None, budget: float = 45) -> tuple[list[str], dict]:
    """只做上網搜尋（不需要分離結果）。"""
    queries = [query] if query else lyric_queries(meta)
    from concurrent.futures import ThreadPoolExecutor, wait as fwait
    tried, found, errors = [], {}, []
    sites = [search_lrclib, search_netease, search_kugou]
    dead = set()                      # 連不上的網站，後面的搜尋就跳過，不要每次都等逾時
    deadline = time.time() + budget   # 整體時間上限：網站很慢時也不會一直卡在這一步
    pool = ThreadPoolExecutor(max_workers=3)
    try:
        for q in queries:
            for qq in dict.fromkeys([q, to_simplified(q)]):
                if time.time() >= deadline:
                    break
                tried.append(qq)
                futs = {pool.submit(fn, qq): fn for fn in sites if fn not in dead}
                pending = set(futs)
                while pending and time.time() < deadline:
                    if check:
                        check()       # 使用者按「停止」時在這裡停下
                    _, pending = fwait(pending, timeout=0.5)
                for f, fn in futs.items():
                    if f in pending:
                        errors.append(f"{fn.__name__}: 逾時")
                        dead.add(fn)
                        continue
                    try:
                        for c in f.result():
                            found.setdefault(c["id"], c)
                    except Exception as e:
                        errors.append(f"{fn.__name__}: {type(e).__name__}")
                        log_error(f"{fn.__name__}({qq})", e)
                        dead.add(fn)
            if len(found) >= 8 or time.time() >= deadline or len(dead) == len(sites):
                break
    finally:
        pool.shutdown(wait=False, cancel_futures=True)
    if not found and errors and len(dead) == len(sites):
        raise UserError("無法連到歌詞網站，請確認網路後再試。")
    return tried, found


def rank_candidates(d: Path, cfg: dict, tried: list[str], found: dict) -> tuple[list[str], list[dict]]:
    meta = read_meta(d)
    found = {k: dict(v) for k, v in found.items()}   # 不改到呼叫的人手上的原始結果
    reliable = recognition_reliable(meta)
    rec = recognized_text(d) if reliable else ""
    dur = meta.get("duration")
    to_tw = bool(cfg.get("convert_to_taiwan", True))
    out = []
    for c in found.values():
        c["similarity"] = similarity(rec, c["plain"] or c["synced"])
        c["diff"] = round(abs(c["duration"] - dur), 1) if (c.get("duration") and dur) else None
        if to_tw and re.search(r"[\u4e00-\u9fff]", c["plain"] + c["synced"] + c["track"] + c["artist"]) \
                and not is_japanese(c["plain"] + c["synced"]):
            for k in ("plain", "synced", "track", "artist", "album"):
                c[k] = to_taiwan(c[k])
        lines = [l for l in c["plain"].splitlines() if l.strip()]
        c["preview"] = "\n".join(lines[:3])
        c["lines"] = len(lines)
        out.append(c)

    title = _norm(" ".join(filter(None, [meta.get("title"), meta.get("yt_track"), meta.get("yt_artist")])))
    # 2.1.0：有時間軸的候選，拿它的時間軸和這首歌主唱開口的位置比對，算出「吻合度」0～1。
    # 歌名和長度都一樣、但其實是翻唱版、不同剪輯版的，吻合度會明顯比較低，自動套用時就不會挑到
    phrases = voice_phrases(d)
    for c in out:
        c["voice_fit"] = timeline_fit(c, phrases, dur) if (c["synced"] and phrases) else None

    def score(c):
        # 歌名吻合度：候選的歌名（與歌手）有多少字出現在這首歌的標題裡；和辨識結果無關
        tr = _norm(c.get("track") or "")
        c["title_match"] = round(sum(1 for ch in tr if ch in title) / len(tr), 2) if tr and title else 0.0
        pen = min((c["diff"] or 0) / 30, 0.3) if c["diff"] is not None else 0.1
        base = 0.5 * c["title_match"] - pen + (0.08 if c["synced"] else 0)
        if reliable and c["similarity"] is not None:   # 辨識可靠時，相似度才拿來參考
            base += c["similarity"]
        if c.get("voice_fit") is not None:
            base += 0.6 * (c["voice_fit"] - 0.5)   # 吻合的加分、不吻合的扣分（±0.3）
        return -base
    out.sort(key=score)
    return list(dict.fromkeys(tried)), out[:12]


def timeline_fit(c: dict, phrases: list[tuple[float, float]], dur: float | None) -> float | None:
    """候選歌詞的時間軸和這首歌主唱開口位置的吻合度（0～1）。
    依網站的時間軸（長度不同時先依比例與位移對齊，和套用時的做法相同）算每一句開頭離最近一次開口多遠：
    0.3 秒內算吻合，愈遠分數愈低。句子太少時不算。"""
    times = [l["t"] for l in parse_lrc(c["synced"]) if l["t"] is not None and l["text"].strip()]
    if len(times) < 4 or len(phrases) < 2:
        return None
    ts = np.array(times, dtype=float)
    r = None
    try:
        r = voice_fit(times, _PhraseStub(phrases), c.get("duration"), dur)   # 需要整體對齊時先對齊
    except Exception:
        r = None
    if r:
        ts = ts * r[0] + r[1]
    onsets = np.array([a for a, _ in phrases])
    k = np.clip(np.searchsorted(onsets, ts), 1, len(onsets) - 1)
    dist = np.minimum(np.abs(ts - onsets[k - 1]), np.abs(ts - onsets[k]))
    return round(float(np.mean(np.exp(-(dist ** 2) / (2 * 0.35 ** 2)))), 2)


def lyrics_timing_fit(d: Path) -> float | None:
    """目前歌詞的時間軸（不做整體對齊，就是現在顯示的位置）和主唱開口位置的吻合度（0～1）。"""
    try:
        times = [l["t"] for l in lyric_lines(d) if l["text"].strip()]
    except Exception:
        return None
    phrases = voice_phrases(d)
    if len(times) < 4 or len(phrases) < 2:
        return None
    ts = np.array(times, dtype=float)
    onsets = np.array([a for a, _ in phrases])
    k = np.clip(np.searchsorted(onsets, ts), 1, len(onsets) - 1)
    dist = np.minimum(np.abs(ts - onsets[k - 1]), np.abs(ts - onsets[k]))
    return round(float(np.mean(np.exp(-(dist ** 2) / (2 * 0.35 ** 2)))), 2)


def recheck_bad(after: float | None, before: float | None) -> bool:
    """2.1.1：只看「重新分離前後」差多少。歌詞句子常從一段歌聲中間開始，吻合度本來就不高，
    單看分數會把大部分歌都標起來；分離前後明顯變差（掉 0.15 以上）才代表歌聲位置真的變了。"""
    return after is not None and before is not None and before - after >= 0.15


def clear_false_recheck() -> int:
    """2.1.0 的判斷太嚴，把很多歌詞沒問題的歌也標成「建議重新匯入歌詞」：依新的標準重新判斷一次。"""
    n = 0
    for d in LIB.iterdir() if LIB.exists() else []:
        mp = d / "meta.json"
        if not mp.exists():
            continue
        try:
            m = read_meta(d)
        except Exception:
            continue
        rc = m.get("lyrics_recheck")
        if isinstance(rc, dict) and not recheck_bad(rc.get("fit"), rc.get("before")):
            update_meta(d, lyrics_recheck=None)
            n += 1
    return n


def check_lyrics_after_reseparate(d: Path, before: float | None) -> None:
    """2.1.0：重新分離／分析旋律後，目前的歌詞和新的歌聲位置對不太上 → 標「建議重新匯入歌詞」。
    標記記下當時歌詞檔的時間，之後歌詞有改過（重新匯入、套用線上歌詞、手動調整）就自動消失。"""
    after = lyrics_timing_fit(d)
    bad = recheck_bad(after, before)
    rec = None
    if bad:
        try:
            mt = (d / "lyrics.lrc").stat().st_mtime_ns
        except OSError:
            mt = 0
        rec = {"fit": after, "before": before, "mtime": mt}
    update_meta(d, lyrics_recheck=rec)


class _PhraseStub:
    """讓 voice_fit 直接用算好的開口段落（不用每個候選都重讀 pitch.json）。"""
    def __init__(self, phrases):
        self.phrases = phrases


def recognition_reliable(meta: dict) -> bool:
    """自動辨識的歌詞可不可靠。1.4.7 之前辨識的歌沒有這項資料，視為可靠（維持原本的行為）。"""
    if meta.get("lyrics_recognized") is False and not meta.get("lyrics_quality"):
        return False   # 匯入時沒有自動辨識
    q = meta.get("lyrics_quality")
    return bool(q.get("ok", True)) if isinstance(q, dict) else not meta.get("lyrics_unreliable")


def guess_language(text: str) -> str:
    """從歌詞文字猜語言（沒有自動辨識時用）。"""
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\uac00-\ud7af]", text):
        return "ko"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh"
    return "en" if re.search(r"[A-Za-z]", text) else ""


def _lyrics_note(d: Path, note: str | None) -> None:
    with _META_LOCK:   # 讀出到寫回之間不讓別的執行緒改這首歌的資料
        meta = read_meta(d)
        meta["notes"] = [x for x in meta.get("notes", []) if "線上歌詞" not in x] + ([note] if note else [])
        write_meta(d, meta)


def find_and_store_candidates(d: Path, cfg: dict, check=None, apply_first: bool = False, pre=None) -> None:
    _, cands = find_candidates(d, cfg, check=check, pre=pre)
    meta = read_meta(d)
    # 候選一律保留給使用者看，不再用辨識結果過濾。是否主動提示「有線上歌詞可選」：
    # 辨識可靠時看和辨識結果像不像；辨識不可靠（或沒有辨識）時看歌名和長度。
    if recognition_reliable(meta) and (d / "lyrics_auto.lrc").exists():
        likely = [c for c in cands if c["similarity"] is None or c["similarity"] >= 0.25]
    else:
        likely = [c for c in cands if c.get("title_match", 0) >= 0.5 or (c["diff"] is not None and c["diff"] <= 5)]
    # 和歌聲對不上的（吻合度低於 0.5：一半以上的句子離開口超過 0.5 秒）不自動套用，留給使用者自己挑
    likely = [c for c in likely if c.get("voice_fit") is None or c["voice_fit"] >= 0.5]
    atomic_write_text(d / "lyrics_candidates.json", json.dumps(cands, ensure_ascii=False))
    if meta.get("lyrics_source") in (None, "whisper"):
        update_meta(d, lyrics_pending=bool(likely))
    if not apply_first:
        return
    lrc = d / "lyrics.lrc"
    if meta.get("lyrics_source") or (lrc.exists() and lrc.read_text("utf-8").strip()):
        _lyrics_note(d, None)   # 等搜尋的時候使用者已經自己選好歌詞，不覆蓋
        return
    if not likely:
        _lyrics_note(d, "沒有找到相符的線上歌詞。" + ("可以在練唱畫面的「搜尋線上歌詞」看看其他候選、" if cands else "可以在練唱畫面")
                     + "貼上歌詞，或用「重新處理 → 只重新辨識歌詞」自動辨識。")
        return
    c = likely[0]
    _, how = apply_candidate(d, c)
    lang = guess_language((c.get("plain") or "") + (c.get("synced") or ""))
    if lang and not read_meta(d).get("language"):
        update_meta(d, language=lang)
    _lyrics_note(d, f"已自動套用線上歌詞（{c.get('track') or ''}／{c.get('artist') or ''}，{how}）。"
                    "不對的話可以在練唱畫面的「搜尋線上歌詞」換其他版本。")


# ---------------------------------------------------------------- 依主唱聲音對時間（不需要辨識結果，任何語言都可以）

def vocal_energy_phrases(d: Path, gap: float = 0.35, min_len: float = 0.3) -> list[tuple[float, float]]:
    """2.3.2：還沒分析旋律時，用「主唱＋合音」的音量找有在唱的段落。
    主唱被分到合音的副歌也算得到，所以拿來對歌詞時間不會因為主唱分錯而對歪。"""
    try:
        y = read_audio(stem_path(d, "lead"), sr=PITCH_SR, mono=True)
        b = stem_path(d, "back")
        if b.exists():
            yb = read_audio(b, sr=PITCH_SR, mono=True)
            n = min(len(y), len(yb))
            y = y[:n] + yb[:n]
    except Exception:
        return []
    H = int(PITCH_SR * 0.02)
    n = len(y) // H * H
    if n < H * 50:
        return []
    r = np.sqrt(np.mean(y[:n].reshape(-1, H) ** 2, axis=1))
    r = np.convolve(r, np.ones(3) / 3, "same")
    ref = float(np.percentile(r, 95))
    if ref <= 1e-5:
        return []
    on = r > 0.3 * ref   # 實測 6 首：和旋律分析結果對歌詞時間的準度相當
    out, i = [], 0
    while i < len(on):
        if not on[i]:
            i += 1
            continue
        j = i
        while j < len(on) and on[j]:
            j += 1
        a, e = i * 0.02, j * 0.02
        if out and a - out[-1][1] < gap:
            out[-1] = (out[-1][0], e)
        else:
            out.append((a, e))
        i = j
    return [(a, e) for a, e in out if e - a >= min_len]


def voice_phrases(d, gap: float = 0.35, min_len: float = 0.3) -> list[tuple[float, float]]:
    """從旋律分析結果找出「有在唱」的段落 (開始, 結束)。還沒分析旋律時改用主唱＋合音的音量。"""
    if isinstance(d, _PhraseStub):
        return d.phrases
    try:
        pj = json.loads((d / "pitch.json").read_text("utf-8"))
    except Exception:
        return vocal_energy_phrases(d, gap, min_len) if isinstance(d, Path) else []
    hop = pj.get("hop", HOP_S)
    on = np.array([v is not None for v in pj["midi"]])
    out, i, n = [], 0, len(on)
    while i < n:
        if not on[i]:
            i += 1
            continue
        j = i
        while j < n and on[j]:
            j += 1
        a, b = i * hop, j * hop
        if out and a - out[-1][1] < gap:
            out[-1] = (out[-1][0], b)
        else:
            out.append((a, b))
        i = j
    return [(a, b) for a, b in out if b - a >= min_len]


def align_synced_to_voice(lines: list[dict], d: Path, cand_dur: float | None, dur: float | None) -> tuple[list[dict], str]:
    """網站的時間軸和這首歌對不上（版本長度不同、前奏不同）時，找最能對上主唱開口位置的整體位移（與速度比例）。"""
    r = voice_fit([l["t"] for l in lines], d, cand_dur, dur) if lines else None
    if not r:
        return lines, ""
    sc, off = r
    out = [{**l, "t": round(max(0.0, l["t"] * sc + off), 2)} for l in lines]
    return out, f"依主唱的聲音位置調整了網站的時間軸（{off:+.1f} 秒{'、速度比例 %.3f' % sc if sc != 1.0 else ''}）"


def voice_fit(times: list[float], d: Path, cand_dur: float | None = None, dur: float | None = None):
    """回傳 (速度比例, 位移秒數)；和原本的時間差不多就回傳 None。"""
    ph = voice_phrases(d)
    if not ph or not times:
        return None
    onsets = np.array([a for a, _ in ph])
    ts = np.array(times, dtype=float)
    scales = [1.0]
    if cand_dur and dur and 0.8 < dur / cand_dur < 1.25:
        scales.append(dur / cand_dur)

    if len(onsets) < 2:
        return None

    def fit(t):
        k = np.clip(np.searchsorted(onsets, t), 1, len(onsets) - 1)
        dist = np.minimum(np.abs(t - onsets[k - 1]), np.abs(t - onsets[k]))
        return float(np.exp(-(dist ** 2) / (2 * 0.3 ** 2)).sum())
    base = fit(ts)
    best = (base, 1.0, 0.0)
    # 1.6.8：MV 版前奏常常長很多，範圍放寬到 ±60 秒；先每 0.25 秒粗找，再在最佳位置附近每 0.02 秒細找
    for sc in scales:
        coarse = max(((fit(ts * sc + off), float(off)) for off in np.arange(-60, 60.01, 0.25)))
        for off in np.arange(coarse[1] - 0.5, coarse[1] + 0.501, 0.02):
            v = fit(ts * sc + off)
            if v > best[0] + 1e-9:
                best = (v, sc, round(float(off), 2))
    v, sc, off = best
    # 明顯比原本的時間軸更貼合主唱才調整，避免把本來就對的時間軸亂移
    if v < base + max(1.0, 0.08 * len(ts)) or (abs(off) < 0.15 and sc == 1.0):
        return None
    return sc, off


def lyrics_shift(d: Path, times: list[float]) -> tuple[float, float]:
    """目前的歌詞整體和歌聲差了好幾秒時，依主唱開口位置找出整體位移與速度比例（畫面套用，逐字時間一起移動）。
    不需要調整時回傳 (1, 0)。"""
    ts = [float(t) for t in times if t is not None]
    if len(ts) < 3:
        raise UserError("含時間的歌詞太少，沒辦法自動校正")
    if not (d / "pitch.json").exists():
        raise UserError("這首歌還沒分析旋律，沒辦法依歌聲校正")
    meta = read_meta(d)
    return voice_fit(ts, d, None, meta.get("duration")) or (1.0, 0.0)


def align_by_voice(lines: list[str], d: Path, bonus: float = 0.6) -> list[dict]:
    """只有文字的歌詞：依每句的字數，找出最符合「有唱歌的時間長度」的分句位置（動態規劃）。

    候選的句子開頭是主唱開口的位置（段落開頭、音符開頭）；每句預期唱的時間和字數成正比。
    和語言無關，但只是估計，通常還需要用「對準目前位置」和逐句調整微調。"""
    try:
        pj = json.loads((d / "pitch.json").read_text("utf-8"))
    except Exception:
        pj = None
    if not pj or not lines:
        return [{"t": round(3.5 * i, 2), "text": t} for i, t in enumerate(lines)]
    hop = pj.get("hop", HOP_S)
    on = np.array([v is not None for v in pj["midi"]], dtype=float)
    cumv = np.concatenate([[0.0], np.cumsum(on)]) * hop          # 累積「有唱歌」的秒數
    strong = {round(a, 2) for a, _ in voice_phrases(d)}
    cands = sorted(strong | {round(n[0], 2) for n in pj.get("notes", [])})
    if len(cands) < len(lines):
        return [{"t": round(3.5 * i, 2), "text": t} for i, t in enumerate(lines)]
    C = np.array(cands)
    is_strong = np.array([c in strong for c in cands])
    V = np.array([cumv[min(len(cumv) - 1, int(t / hop))] for t in C])
    end_v = cumv[-1]
    w = np.array([max(1, len(_norm(t))) for t in lines], float)
    e = w * (end_v / w.sum())                                     # 每句預期的有聲秒數
    N, M = len(lines), len(C)
    cost = np.full((N, M), 1e18)
    back = np.zeros((N, M), int)
    cost[0] = 0.02 * V - bonus * is_strong
    for i in range(1, N):
        prev = cost[i - 1]
        for k in range(i, M):
            js = np.arange(max(i - 1, k - 80), k)
            c = prev[js] + (V[k] - V[js] - e[i - 1]) ** 2 / (e[i - 1] + 0.5)
            j = int(np.argmin(c))
            cost[i, k] = c[j] - bonus * is_strong[k]
            back[i, k] = js[j]
    last = cost[N - 1] + 0.3 * (end_v - V - e[N - 1]).clip(min=0) ** 2 / (e[N - 1] + 0.5)
    k = int(np.argmin(last))
    idx = [k]
    for i in range(N - 1, 0, -1):
        k = back[i, k]
        idx.append(k)
    idx.reverse()
    return [{"t": round(max(0.0, float(C[k]) - 0.1), 2), "text": t} for k, t in zip(idx, lines)]


def align_to_recognized(lines: list[str], d: Path) -> list[dict]:
    """把只有文字的歌詞，依辨識結果的時間對上：逐字比對後，每句取第一個對到的字的時間。"""
    import difflib
    auto = d / "lyrics_auto.lrc"
    rec = [l for l in parse_lrc(auto.read_text("utf-8")) if l["t"] is not None] if auto.exists() else []
    stream, times = [], []
    for i, l in enumerate(rec):
        end = rec[i + 1]["t"] if i + 1 < len(rec) else l["t"] + 4
        chars = _norm(l["text"])
        for k, ch in enumerate(chars):
            stream.append(ch)
            times.append(l["t"] + (end - l["t"]) * 0.85 * k / max(1, len(chars)))
    cand, owner = [], []
    for i, l in enumerate(lines):
        for ch in _norm(l):
            cand.append(ch)
            owner.append(i)
    starts = [None] * len(lines)
    if stream and cand:
        m = difflib.SequenceMatcher(None, "".join(cand), "".join(stream), autojunk=False)
        for blk in m.get_matching_blocks():
            for k in range(blk.size):
                li = owner[blk.a + k]
                if starts[li] is None:
                    starts[li] = times[blk.b + k]   # 該句第一個對到的字
    # 沒對到的句子：前後內插
    known = [(i, t) for i, t in enumerate(starts) if t is not None]
    for i in range(len(lines)):
        if starts[i] is not None:
            continue
        prev = max((k for k in known if k[0] < i), default=None, key=lambda k: k[0])
        nxt = min((k for k in known if k[0] > i), default=None, key=lambda k: k[0])
        if prev and nxt:
            starts[i] = prev[1] + (nxt[1] - prev[1]) * (i - prev[0]) / (nxt[0] - prev[0])
        elif prev:
            starts[i] = prev[1] + 3.5 * (i - prev[0])
        elif nxt:
            starts[i] = max(0.0, nxt[1] - 3.5 * (nxt[0] - i))
        else:
            starts[i] = 3.5 * i
    out, last = [], -1.0
    for t, text in zip(starts, lines):   # 保持遞增
        t = max(t, last + 0.3)
        out.append({"t": round(t, 2), "text": text})
        last = t
    return out


def srt_to_lrc(text: str) -> str:
    """SRT 字幕 → LRC（每段字幕取開始時間，多行合成一句）。"""
    out = []
    for block in re.split(r"\r?\n\s*\r?\n", text.strip()):
        m = re.search(r"(\d+):(\d+):(\d+)[,.](\d+)\s*-->", block)
        if not m:
            continue
        t = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + int(m.group(3)) + int(m.group(4)[:3].ljust(3, "0")) / 1000
        body = " ".join(l.strip() for l in block.splitlines()[block.splitlines().index(
            next(l for l in block.splitlines() if "-->" in l)) + 1:] if l.strip())
        body = re.sub(r"<[^>]+>|\{[^}]*\}", "", body).strip()
        if body:
            out.append({"t": t, "text": body})
    return to_lrc(out)


def import_lyrics(d: Path, text: str, cfg: dict, filename: str = "") -> tuple[str, str]:
    """匯入歌詞檔（LRC、SRT 或純文字）。純文字依辨識結果的時間自動對上每一句。回傳 (lrc, 說明)。"""
    text = (text or "").replace("\ufeff", "").strip()
    if not text:
        raise UserError("歌詞檔是空的")
    ext = filename.lower().rsplit(".", 1)[-1] if "." in filename else ""
    if ext == "srt" or re.search(r"\d+:\d+:\d+[,.]\d+\s*-->", text):
        lrc, how = srt_to_lrc(text), "已從 SRT 字幕轉成 LRC，使用字幕的時間"
    elif re.search(r"\[\d+:\d+(?:\.\d+)?\]", text):
        lrc, how = clean_lrc(text) + "\n", "使用歌詞檔裡的時間"
    else:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        auto = d / "lyrics_auto.lrc"
        if auto.exists() and recognition_reliable(read_meta(d)):
            lrc = to_lrc(align_to_recognized(lines, d))
            how = "歌詞檔沒有時間，已依辨識結果自動對上每一句"
        elif (d / "pitch.json").exists():
            lrc = to_lrc(align_by_voice(lines, d))
            how = "歌詞檔沒有時間，已依主唱有聲音的段落分配每一句，可以用「對準目前位置」和逐句調整微調"
        else:
            lrc = to_lrc(align_to_recognized(lines, d))
            how = "歌詞檔沒有時間，暫時每句間隔 3.5 秒，可以用「跟拍打點」或 Claude 對齊調整"
    if not lrc.strip():
        raise UserError("讀不到歌詞內容，請確認檔案格式（LRC、SRT 或 TXT）")
    if cfg.get("convert_to_taiwan", True) and re.search(r"[\u4e00-\u9fff]", lrc):
        lrc = to_taiwan(lrc)
    atomic_write_text(d / "lyrics.lrc", lrc)
    with _META_LOCK:   # 讀出到寫回之間不讓別的執行緒改這首歌的資料
        meta = read_meta(d)
        meta["steps"]["lyrics"] = True
        meta["lyrics_source"] = "import"
        meta["lyrics_pending"] = False
        meta["notes"] = [x for x in meta.get("notes", []) if "歌詞辨識沒有完成" not in x]
        write_meta(d, meta)
    return lrc, how


def apply_candidate(d: Path, cand: dict) -> tuple[str, str]:
    """套用一筆線上歌詞。長度相符的同步歌詞直接用；否則用辨識結果的時間對齊。回傳 (lrc, 說明)。"""
    meta = read_meta(d)
    dur = meta.get("duration")
    synced = cand.get("synced") or ""
    plain_lines = [l.strip() for l in (cand.get("plain") or "").splitlines() if l.strip()]
    diff = abs(cand["duration"] - dur) if (cand.get("duration") and dur) else None
    reliable = recognition_reliable(meta) and (d / "lyrics_auto.lrc").exists()
    if synced and (diff is None or diff <= 3):
        lines = [l for l in parse_lrc(synced) if l["t"] is not None and l["text"]]
        # 1.6.8：長度相同，前奏卻可能不同（MV 版、剪輯版），整體差好幾秒；依主唱開口位置檢查並校正
        lines, how = align_synced_to_voice(lines, d, cand.get("duration"), dur)
        how = how or "使用網站提供的時間軸"
    elif synced:
        # 網站有時間軸、但版本長度不同：不靠辨識結果，改依主唱的聲音位置整體平移
        lines = [l for l in parse_lrc(synced) if l["t"] is not None and l["text"]]
        lines, how = align_synced_to_voice(lines, d, cand.get("duration"), dur)
        how = how or "使用網站提供的時間軸（版本長度不同，可能需要用「對準目前位置」微調）"
    else:
        if reliable:
            lines = align_to_recognized(plain_lines, d)
            how = "依辨識結果的時間自動對齊"
        else:
            lines = align_by_voice(plain_lines, d)
            how = "依主唱有聲音的段落分配時間（建議用「對準目前位置」和逐句調整微調）"
    lrc = to_lrc(lines)
    atomic_write_text(d / "lyrics.lrc", lrc)
    atomic_write_text(d / "lyrics_online.lrc", lrc)   # 線上歌詞另外保存，和辨識結果互不影響
    with _META_LOCK:   # 讀出到寫回之間不讓別的執行緒改這首歌的資料
        meta = read_meta(d)
        meta["steps"]["lyrics"] = True
        meta["lyrics_source"] = "online"
        meta["lyrics_pending"] = False
        meta["lyrics_online"] = {"source": cand.get("source"), "track": cand.get("track"), "artist": cand.get("artist")}
        meta["notes"] = [x for x in meta.get("notes", []) if "歌詞辨識沒有完成" not in x]
        write_meta(d, meta)
    return lrc, how


# ---------------------------------------------------------------- 分類

def _cats_path() -> Path:
    return LIB / "categories.json"   # 歌曲庫可以搬家：每次用目前的位置


def load_categories() -> list[str]:
    try:
        return [c for c in json.loads(_cats_path().read_text("utf-8")) if isinstance(c, str)]
    except Exception:
        return []


def save_categories(cats: list[str]) -> None:
    LIB.mkdir(exist_ok=True)
    atomic_write_text(_cats_path(), json.dumps(cats, ensure_ascii=False))


# ---------------------------------------------------------------- 歌曲庫／模型搬移（2.3.0）
STORAGE = {"moving": False}   # 搬移中：擋住會寫進歌曲庫的操作


def dir_size(p: Path) -> tuple[int, int]:
    """資料夾大小（位元組、檔案數）。"""
    total = n = 0
    for root, _dirs, files in os.walk(p):
        for f in files:
            try:
                total += os.path.getsize(os.path.join(root, f))
                n += 1
            except OSError:
                pass
    return total, n


def drives() -> list[dict]:
    """可以搬過去的磁碟（Windows：每個磁碟機代號的剩餘空間）。"""
    out = []
    if sys.platform == "win32":
        import string
        for L in string.ascii_uppercase:
            r = f"{L}:\\"
            if os.path.exists(r):
                try:
                    u = shutil.disk_usage(r)
                    out.append({"path": r, "free": u.free, "total": u.total})
                except OSError:
                    pass
    else:
        u = shutil.disk_usage("/")
        out.append({"path": "/", "free": u.free, "total": u.total})
    return out


def storage_info() -> dict:
    cfg = load_config()
    info = {"library": str(LIB), "models": str(MODELS),
            "library_default": LIB == DEFAULT_LIB, "models_default": MODELS == DEFAULT_MODELS,
            "library_missing": not LIB.exists(), "drives": drives(),
            "old": [p for p in cfg.get("storage_old", []) if Path(p).exists()],
            "moving": STORAGE["moving"]}
    for k, p in (("library", LIB), ("models", MODELS)):
        info[k + "_size"] = dir_size(p)[0] if p.exists() else 0
        try:
            info[k + "_free"] = shutil.disk_usage(p if p.exists() else p.anchor).free
        except OSError:
            info[k + "_free"] = None
    return info


def _inside(a: Path, b: Path) -> bool:
    """a 是不是 b 本身或在 b 裡面。"""
    try:
        a.resolve().relative_to(b.resolve())
        return True
    except ValueError:
        return False


def move_storage(target: str, what: list[str], report) -> dict:
    """把歌曲庫（library）和／或模型（models）複製到 target 底下，逐檔核對大小後才改用新位置。
    舊的那份不刪，記在設定的 storage_old，使用者確認新位置正常後再刪。中途停止：設定不變，照常用舊的。"""
    global LIB, MODELS
    tgt = Path(target.strip().strip('"'))
    if not tgt.is_absolute():
        raise UserError("請輸入完整的資料夾位置，例如 D:\\雷虎練唱台資料")
    if not Path(tgt.anchor).exists():
        raise UserError(f"找不到磁碟 {tgt.anchor}")
    srcs = [(k, LIB if k == "library" else MODELS) for k in ("library", "models") if k in what]
    if not srcs:
        raise UserError("請至少選一項要搬移的東西")
    plan = []
    for k, src in srcs:
        dst = tgt / k
        if _inside(dst, src) or _inside(src, dst):
            raise UserError("新位置不能在原本的資料夾裡面（或包含原本的資料夾）")
        if dst.exists() and any(dst.iterdir()):
            raise UserError(f"{dst} 已經有檔案，請換一個空的資料夾")
        if not src.exists():
            raise UserError(f"找不到目前的{'歌曲庫' if k == 'library' else '模型'}資料夾：{src}")
        plan.append((k, src, dst))
    report("計算大小", 0.01)
    need = 0
    files: list[tuple[Path, Path, int]] = []
    for k, src, dst in plan:
        for root, _dirs, fs in os.walk(src):
            for f in fs:
                s = Path(root) / f
                if f.endswith((".part", ".tmp")) or ".part" in f:
                    continue   # 下載到一半的暫存檔不搬
                try:
                    size = s.stat().st_size
                except OSError:
                    continue
                files.append((s, dst / s.relative_to(src), size))
                need += size
    free = shutil.disk_usage(tgt.anchor).free
    if need > free - (1 << 30):   # 至少多留 1 GB
        raise UserError(f"空間不夠：需要 {need / 1e9:.1f} GB，{tgt.anchor} 只剩 {free / 1e9:.1f} GB")
    done, t0, last = 0, time.time(), 0.0
    try:
        for s, d, size in files:
            d.parent.mkdir(parents=True, exist_ok=True)
            with open(s, "rb") as fi, open(d, "wb") as fo:
                while True:
                    b = fi.read(8 << 20)
                    if not b:
                        break
                    fo.write(b)
                    done += len(b)
                    if time.time() - last > 0.5:
                        last = time.time()
                        sp = done / max(1e-3, last - t0)
                        left = (need - done) / sp if sp > 0 else 0
                        eta = f"（約剩 {left / 60:.0f} 分鐘）" if left >= 90 else ""
                        report(f"搬移中 {done / 1e9:.1f}／{need / 1e9:.1f} GB{eta}", 0.02 + 0.93 * done / max(1, need))
            shutil.copystat(s, d)
            if d.stat().st_size != size:
                raise UserError(f"複製後大小不符：{s.name}，已停止搬移（仍使用原本的位置）")
    except BaseException:
        for _k, _src, dst in plan:   # 沒搬完：清掉新位置的半成品，繼續用舊的
            shutil.rmtree(dst, ignore_errors=True)
        raise
    report("核對檔案", 0.96)
    for k, src, dst in plan:
        a, b = dir_size(dst)
        exp = sum(sz for s, d, sz in files if _inside(d, dst))
        if a != exp:
            shutil.rmtree(dst, ignore_errors=True)
            raise UserError("核對失敗：新位置的檔案總大小不符，已停止搬移（仍使用原本的位置）")
    cfg = load_config()
    old = list(cfg.get("storage_old", []))
    for k, src, dst in plan:
        cfg[k + "_dir"] = str(dst)
        if str(src) not in old:
            old.append(str(src))
    cfg["storage_old"] = old
    save_config(cfg)
    _apply_storage()
    try:
        _SONG_CACHE.clear()
    except Exception:
        pass
    report("完成", 1.0)
    return {k: str(dst) for k, _src, dst in plan}


def delete_old_storage(path: str) -> None:
    """刪除搬移前的舊位置（只能刪設定裡記下來的舊位置，而且不能是目前正在用的）。"""
    cfg = load_config()
    old = cfg.get("storage_old", [])
    if path not in old:
        raise UserError("只能刪除搬移前的舊位置")
    p = Path(path)
    if _inside(LIB, p) or _inside(MODELS, p) or _inside(p, LIB) or _inside(p, MODELS):
        raise UserError("這是目前正在使用的位置，不能刪除")
    shutil.rmtree(p, ignore_errors=True)
    cfg["storage_old"] = [x for x in old if x != path]
    save_config(cfg)


def adopt_storage(path: str) -> dict:
    """換電腦、外接硬碟：直接改用一個已經有歌曲庫的資料夾（例如從舊電腦整個複製過來的），不複製檔案。
    path 可以是含有 library／models 的資料夾，或 library 資料夾本身。"""
    global LIB, MODELS
    p = Path(path.strip().strip('"'))
    if not p.is_absolute() or not p.exists():
        raise UserError("找不到這個資料夾")
    lib = p / "library" if (p / "library").is_dir() else p
    n = sum(1 for d in lib.iterdir() if (d / "meta.json").exists()) if lib.is_dir() else 0
    if not n:
        raise UserError("這個資料夾裡沒有練唱台的歌曲庫（找不到任何歌曲）")
    cfg = load_config()
    cfg["library_dir"] = str(lib)
    out = {"library": str(lib), "songs": n}
    models = lib.parent / "models"
    if lib.name == "library" and (models / "uvr").is_dir():
        cfg["models_dir"] = str(models)
        out["models"] = str(models)
    save_config(cfg)
    _apply_storage()
    try:
        _SONG_CACHE.clear()
    except Exception:
        pass
    return out
