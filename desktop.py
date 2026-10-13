"""雷虎練唱台：桌面視窗版。

視窗（WebView2）與處理伺服器分成兩個程序：AI 模型的 CUDA 函式庫和 WebView2/.NET
放在同一個程序裡會互相卡死，所以伺服器（app.py）在背景另開一個程序執行，
視窗關閉時一起結束。用 pythonw 執行時不會出現黑色主控台視窗。
"""
from __future__ import annotations

import os
import socket
import subprocess
import sys
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
    log = open(ROOT / "logs" / "app.log", "a", encoding="utf-8", buffering=1)
    env = dict(os.environ, LIANTAI_PORT=str(port), PYTHONIOENCODING="utf-8", PYTHONUNBUFFERED="1")
    flags = 0x08000000 if sys.platform == "win32" else 0   # CREATE_NO_WINDOW
    return subprocess.Popen(
        [sys.executable, str(ROOT / "app.py"), "--no-browser", "--parent-pid", str(os.getpid())],
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


class _Existing:
    def terminate(self): pass
    def wait(self, *_): pass
    def kill(self): pass


def main():
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
        version = "1.4.4"
        try:
            version = (ROOT / "VERSION").read_text("utf-8").strip() or version
        except OSError:
            pass
        window = webview.create_window(
            f"雷虎練唱台 {version}", base + "/", width=1380, height=920, min_size=(960, 640),
            background_color="#0E1318", text_select=True,
        )
        done: list = []
        window.events.loaded += lambda: allow_microphone(window, base, done)
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
