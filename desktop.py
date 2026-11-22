"""雷虎練唱台：桌面視窗版。

視窗（WebView2）與處理伺服器分成兩個程序：AI 模型的 CUDA 函式庫和 WebView2/.NET
放在同一個程序裡會互相卡死，所以伺服器（app.py）在背景另開一個程序執行，
視窗關閉時一起結束。用 pythonw 執行時不會出現黑色主控台視窗。
"""
from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import traceback
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
os.chdir(ROOT)
(ROOT / "logs").mkdir(exist_ok=True)
_log = open(ROOT / "logs" / "desktop.log", "a", encoding="utf-8", buffering=1)
sys.stdout = sys.stderr = _log


def pick_port(preferred: int) -> int:
    for port in (preferred, 0):
        s = socket.socket()
        try:
            s.bind(("127.0.0.1", port))
            return s.getsockname()[1]
        except OSError:
            continue
        finally:
            s.close()
    raise RuntimeError("找不到可用的連接埠")


def start_server(port: int) -> subprocess.Popen:
    lp = ROOT / "logs" / "app.log"
    try:   # 記錄檔超過 10 MB：留一份 app.log.old，重新開始（不會無限長大）
        if lp.exists() and lp.stat().st_size > 10 * 1024 * 1024:
            os.replace(lp, lp.with_name("app.log.old"))
    except OSError:
        pass
    log = open(lp, "a", encoding="utf-8", buffering=1)
    env = dict(os.environ, LIANTAI_PORT=str(port), PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    flags = 0x08000000 if sys.platform == "win32" else 0   # CREATE_NO_WINDOW
    # 伺服器要用 python.exe（不是 pythonw.exe）執行：pythonw 沒有主控台，之後伺服器每次呼叫 ffmpeg、pip 等程式，
    # Windows 都會替它們各開一個新的黑色視窗、閃一下就不見。用 python.exe 加上 CREATE_NO_WINDOW，
    # 伺服器有一個看不見的主控台，這些程式會共用它，就不會再跳出視窗
    py = Path(sys.executable).with_name("python.exe")
    return subprocess.Popen(
        [str(py if py.exists() else sys.executable), str(ROOT / "app.py"), "--no-browser", "--parent-pid", str(os.getpid())],
        cwd=str(ROOT), env=env, stdout=log, stderr=subprocess.STDOUT, creationflags=flags)


def wait_ready(base: str, proc: subprocess.Popen, timeout: float = 90) -> None:
    end = time.time() + timeout
    while time.time() < end:
        if proc.poll() is not None:
            raise RuntimeError("處理伺服器啟動失敗")
        try:
            urllib.request.urlopen(base + "/api/songs", timeout=2).read()
            return
        except Exception:
            time.sleep(0.4)
    raise RuntimeError("處理伺服器啟動逾時")


def allow_microphone(window, base_url: str, done: list) -> None:
    """自動允許本程式頁面使用麥克風；失敗時退回 WebView2 預設的詢問視窗。"""
    if done:
        return
    done.append(True)
    try:
        from Microsoft.Web.WebView2.Core import CoreWebView2PermissionKind, CoreWebView2PermissionState
        from System import Action

        form = window.native
        wv = form.webview

        def on_permission(sender, args):
            if str(args.Uri).startswith(base_url) and args.PermissionKind == CoreWebView2PermissionKind.Microphone:
                args.State = CoreWebView2PermissionState.Allow

        def attach():
            wv.CoreWebView2.PermissionRequested += on_permission

        form.Invoke(Action(attach))
    except Exception:
        traceback.print_exc()


def running_instance(port: int) -> bool:
    """同一台電腦上已經有練唱台在跑，就共用它的伺服器，避免兩份同時搶顯示卡。"""
    try:
        import json
        data = json.loads(urllib.request.urlopen(f"http://127.0.0.1:{port}/api/status", timeout=2).read())
        return os.path.normcase(data.get("root", "")) == os.path.normcase(str(ROOT))   # 必須是同一份安裝
    except Exception:
        return False


class Api:
    """給網頁呼叫的功能（window.pywebview.api.*）。"""

    def __init__(self, base: str):
        self._base = base
        self._timing = None

    def open_timing(self, owner: str) -> bool:
        """1.6.5：逐句調整的獨立視窗，用練唱台自己的視窗開（不經過 Edge，不會帶出瀏覽器的擴充功能或還原舊視窗）。"""
        import re
        import webview
        if not re.fullmatch(r"[a-z0-9]{4,20}", str(owner)):
            return False
        url = f"{self._base}/timing?owner={owner}"
        w = self._timing
        if w is not None:
            try:
                w.load_url(url)
                w.restore()
                w.show()
                return True
            except Exception:
                self._timing = None
        w = webview.create_window("逐句調整 · 雷虎練唱台", url, width=560, height=860, min_size=(320, 360),
                                  background_color="#0E1318", text_select=True)
        set_window_icon("逐句調整 · 雷虎練唱台")

        def closed():
            if self._timing is w:
                self._timing = None
        w.events.closed += closed
        self._timing = w
        return True

    def pick_folder(self, start: str = "") -> str:
        """2.3.0：搬移歌曲庫時選資料夾（網頁本身打不開資料夾選擇視窗）。取消時回傳空字串。"""
        import webview
        try:
            kind = webview.FileDialog.FOLDER
        except AttributeError:
            kind = webview.FOLDER_DIALOG
        win = webview.windows[0] if webview.windows else None
        if win is None:
            return ""
        r = win.create_file_dialog(kind, directory=start or "")
        if not r:
            return ""
        return str(r[0] if isinstance(r, (list, tuple)) else r)


# 全域快捷鍵（練唱台不在最前面也能用，Stream Deck／巨集鍵盤可以直接綁）。
# 按鍵定義在 hotkeys.py；設定改了（config.json）或設定畫面正在錄新按鍵時，會自動重新註冊，不用重開
HOTKEY_STATUS = ROOT / "hotkeys_status.json"
HOTKEY_PAUSE = ROOT / ".hotkeys_pause"


def global_hotkeys(window) -> None:
    """用 Windows 的 RegisterHotKey 註冊全域快捷鍵，按下時呼叫網頁的 liveHotkey()。
    某個組合被其他程式佔用時只略過那一個，結果寫在 hotkeys_status.json 給設定畫面顯示。"""
    import ctypes
    from ctypes import wintypes
    import hotkeys as HK
    user32 = ctypes.windll.user32
    reg: dict[int, str] = {}

    def unregister():
        for i in list(reg):
            user32.UnregisterHotKey(None, i)
        reg.clear()

    def register():
        try:
            cfg = json.loads((ROOT / "config.json").read_text("utf-8"))
        except (OSError, ValueError):
            cfg = {}
        ok, failed = [], []
        if cfg.get("global_hotkeys") is not False:
            for n, (act, combo) in enumerate(HK.current(cfg).items(), 1):
                p = HK.parse(combo)
                if not p:
                    continue
                if user32.RegisterHotKey(None, n, p[0] | 0x4000, p[1]):   # 0x4000：按住不放不重複觸發
                    reg[n] = act
                    ok.append(act)
                else:
                    failed.append(act)
                    print("全域快捷鍵註冊失敗（可能被其他程式佔用）：", act, combo, flush=True)
        try:
            HOTKEY_STATUS.write_text(json.dumps({"ok": ok, "failed": failed, "at": time.time()}), "utf-8")
        except OSError:
            pass

    msg = wintypes.MSG()
    sig, last = None, 0.0
    while True:
        while user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):   # PM_REMOVE
            if msg.message == 0x0312 and msg.wParam in reg:   # WM_HOTKEY
                try:
                    window.evaluate_js(f"window.liveHotkey && window.liveHotkey('{reg[msg.wParam]}')")
                except Exception:
                    traceback.print_exc()
        now = time.time()
        if now - last > 0.4:
            last = now
            try:
                cfg_m = (ROOT / "config.json").stat().st_mtime_ns
            except OSError:
                cfg_m = 0
            try:   # 設定畫面正在錄新按鍵：暫時放開全部快捷鍵，按下的組合才傳得到畫面（兩分鐘後自動恢復）
                paused = now - HOTKEY_PAUSE.stat().st_mtime < 120
            except OSError:
                paused = False
            if (cfg_m, paused) != sig:
                sig = (cfg_m, paused)
                unregister()
                if not paused:
                    register()
        time.sleep(0.03)


class _Existing:
    def terminate(self): pass
    def wait(self, *_): pass
    def kill(self): pass


APP_ID = "LeihuTW.LeihuLiantai"


def set_app_id() -> None:
    """2.3.4：讓工作列把練唱台當成自己的程式（用練唱台的圖示），不要併到 Python 底下顯示 Python 的圖示。"""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(APP_ID)
    except Exception:
        pass


def set_window_icon(title: str, tries: int = 40) -> None:
    """2.3.4：把視窗左上角、工作列、Alt+Tab 的小圖示換成練唱台的圖示（pywebview 在 Windows 只會用 python 的圖示）。
    視窗建好之前找不到，所以在背景每 0.25 秒試一次。"""
    if sys.platform != "win32":
        return
    ico = ROOT / "static" / "icon.ico"
    if not ico.exists():
        return

    def run():
        try:
            import ctypes
            from ctypes import wintypes
            user32 = ctypes.windll.user32
            user32.FindWindowW.restype = wintypes.HWND
            user32.LoadImageW.restype = wintypes.HANDLE
            user32.SendMessageW.argtypes = [wintypes.HWND, wintypes.UINT, wintypes.WPARAM, wintypes.LPARAM]
            for _ in range(tries):
                hwnd = user32.FindWindowW(None, title)
                if hwnd:
                    for which, size in ((1, 32), (0, 16)):   # ICON_BIG、ICON_SMALL
                        sx = user32.GetSystemMetrics(11 if which else 49) or size   # SM_CXICON／SM_CXSMICON（高 DPI 時更大）
                        h = user32.LoadImageW(None, str(ico), 1, sx, sx, 0x10)      # IMAGE_ICON, LR_LOADFROMFILE
                        if h:
                            user32.SendMessageW(hwnd, 0x80, which, h)              # WM_SETICON
                    return
                time.sleep(0.25)
        except Exception:
            traceback.print_exc()
    threading.Thread(target=run, daemon=True, name="window-icon").start()


def main():
    set_app_id()
    try:
        import webview
    except Exception:
        traceback.print_exc()
        webview = None

    preferred = int(os.environ.get("LIANTAI_PORT", "8765"))
    # 已有同一份練唱台在跑就共用；別的程式（或別份安裝）占用時改用其他連接埠
    for p in range(preferred, preferred + 10):
        if running_instance(p):
            preferred = p
            break
    if running_instance(preferred):
        port, proc = preferred, _Existing()
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} 使用已在執行的練唱台伺服器")
    else:
        port = pick_port(preferred)
        proc = start_server(port)
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} 練唱台桌面版：伺服器 pid {proc.pid}，http://127.0.0.1:{port}")
    base = f"http://127.0.0.1:{port}"
    try:
        if not isinstance(proc, _Existing):
            wait_ready(base, proc)
        if webview is None:   # 視窗元件無法載入時改用瀏覽器，關閉伺服器前一直等待
            import webbrowser
            webbrowser.open(base + "/")
            proc.wait()
            return
        version = "1.4.9"
        try:
            version = (ROOT / "VERSION").read_text("utf-8").strip() or version
        except OSError:
            pass
        title = f"雷虎練唱台 {version}"
        set_window_icon(title)
        window = webview.create_window(
            title, base + "/", width=1380, height=920, min_size=(960, 640),
            background_color="#0E1318", text_select=True, js_api=Api(base),
        )
        done: list = []
        window.events.loaded += lambda: allow_microphone(window, base, done)
        if sys.platform == "win32":
            threading.Thread(target=global_hotkeys, args=(window,), daemon=True, name="hotkeys").start()
        webview.start(private_mode=False, storage_path=str(ROOT / ".webview"))
    finally:
        proc.terminate()
        try:
            proc.wait(5)
        except Exception:
            proc.kill()
    os._exit(0)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        try:
            import ctypes
            ctypes.windll.user32.MessageBoxW(
                0, "雷虎練唱台無法啟動，詳細原因記錄在安裝資料夾 logs 裡的 desktop.log 與 app.log。", "雷虎練唱台", 0x10)
        except Exception:
            pass
        os._exit(1)
