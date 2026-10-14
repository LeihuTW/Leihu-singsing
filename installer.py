"""雷虎練唱台 安裝／修復程式（Windows）。

由安裝程式（install.ps1）在 Python 準備好之後執行，也可以從開始選單的「修復與更新」執行：
    python\\python.exe installer.py [--repair] [--launch]

做的事：
  1. 關閉正在執行的雷虎練唱台
  2. 偵測 NVIDIA 顯示卡，安裝相符的 PyTorch（CUDA 12.8／12.6 或 CPU 版）
  3. 安裝其餘套件，並固定 ONNX Runtime 1.22（CUDA 12）
  4. 清掉舊版殘留檔案，檢查每個元件都能正常載入
  5. 預先下載 AI 模型
紀錄寫在 logs\\install.log。
"""
from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PY = Path(sys.executable).with_name("python.exe")
PYW = Path(sys.executable).with_name("pythonw.exe")
SP = Path(sys.prefix) / "Lib" / "site-packages"
(ROOT / "logs").mkdir(exist_ok=True)
LOG = open(ROOT / "logs" / "install.log", "a", encoding="utf-8", buffering=1)

TORCH = {"torch": "2.11.0", "torchaudio": "2.11.0", "torchvision": "0.26.0"}
ORT_VERSION = "1.22.0"
SEP_MODELS = ["UVR-MDX-NET-Inst_HQ_3.onnx", "UVR_MDXNET_KARA_2.onnx"]
HQ_KARAOKE = "mel_band_roformer_karaoke_aufr33_viperx_sdr_10.1956.ckpt"   # 高品質主唱／合音分離（有顯示卡時使用）
WHISPER = "large-v3-turbo"

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)   # DLL 錯誤不要跳視窗卡住


_bar = {"on": False, "base": 0.0, "span": 0.0, "pct": -1, "detail": ""}


def _clear_bar() -> None:
    if _bar["on"]:
        print("\r" + " " * 100 + "\r", end="", flush=True)
        _bar["on"] = False


def say(msg: str = "") -> None:
    _clear_bar()
    print(msg, flush=True)
    LOG.write(msg + "\n")


def _title(text: str) -> None:
    if sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetConsoleTitleW(text)
        except Exception:
            pass


def stage(n: int, total: int, title: str, start: float, end: float) -> None:
    """開始一個步驟；start/end 是這個步驟在總進度中的範圍（0–100）。"""
    say(f"\n[{n}/{total}] {title}")
    _bar.update(base=start, span=end - start)
    progress(0, title)


def progress(frac: float, detail: str = "") -> None:
    """在同一行更新總進度條，並把百分比顯示在視窗標題。"""
    frac = max(0.0, min(1.0, frac))
    pct = int(_bar["base"] + _bar["span"] * frac)
    detail = (detail or _bar["detail"])[:46]
    if pct == _bar["pct"] and detail == _bar["detail"] and _bar["on"]:
        return
    _bar.update(pct=pct, detail=detail)
    filled = pct * 30 // 100
    line = f"  總進度 [{'█' * filled}{'░' * (30 - filled)}] {pct:3d}%  {detail}"
    print("\r" + line.ljust(98), end="", flush=True)
    _bar["on"] = True
    _title(f"雷虎練唱台 安裝中 {pct}%（請勿關閉）")
    try:
        (ROOT / "logs" / "progress.json").write_text(json.dumps({"percent": pct, "detail": detail}, ensure_ascii=False), "utf-8")
    except OSError:
        pass


def step(n: int, total: int, title: str) -> None:
    say(f"\n[{n}/{total}] {title}")


def run(args: list[str], quiet: bool = True, on_line=None) -> int:
    LOG.write("> " + " ".join(args) + "\n")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PIP_DISABLE_PIP_VERSION_CHECK="1",
               PIP_NO_WARN_SCRIPT_LOCATION="1")
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env=env)
    for line in p.stdout:
        line = line.rstrip()
        LOG.write("  " + line + "\n")
        if on_line:
            on_line(line)
        if not quiet or line.startswith("ERROR"):
            say("  " + line[:150])
    return p.wait()


def pip(*args: str, on_line=None) -> int:
    return run([str(PY), "-m", "pip", *args], on_line=on_line)


def powershell(script: str) -> str:
    enc = base64.b64encode(script.encode("utf-16-le")).decode()
    r = subprocess.run(["powershell", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass",
                        "-EncodedCommand", enc], capture_output=True, text=True, encoding="utf-8", errors="replace")
    return r.stdout.strip()


def probe(code: str, timeout: int = 90) -> tuple[bool, str]:
    """另開程序執行一小段檢查；逾時就連同子程序一起結束。"""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    p = subprocess.Popen([str(PY), "-c", f"import sys; sys.path.insert(0, r'{ROOT}')\n" + code],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env=env)
    try:
        out, _ = p.communicate(timeout=timeout)
        LOG.write(f"probe exit {p.returncode}: {out[-1500:]}\n")
        last = (out or "").strip().splitlines()
        return p.returncode == 0, last[-1] if last else ""
    except subprocess.TimeoutExpired:
        subprocess.run(["taskkill", "/PID", str(p.pid), "/T", "/F"], capture_output=True)
        p.communicate()
        return False, "逾時（卡住）"


# ---------------------------------------------------------------- 步驟

def stop_app() -> None:
    out = powershell(
        "$ErrorActionPreference='SilentlyContinue'; $ProgressPreference='SilentlyContinue';"
        "$all = Get-CimInstance Win32_Process;"
        "$keep = @(); $p = %d; while ($p) { $keep += $p; $p = ($all | Where-Object ProcessId -eq $p).ParentProcessId; if ($keep.Count -gt 8) { break } };"
        "$all | Where-Object { ($_.Name -like 'python*.exe') -and ($keep -notcontains $_.ProcessId) -and "
        "($_.ExecutablePath -like '%s*') } | ForEach-Object { Stop-Process -Id $_.ProcessId -Force; $_.ProcessId }"
        % (os.getpid(), str(Path(sys.prefix))))
    if out:
        say(f"  已關閉正在執行的雷虎練唱台（{out.count(chr(10)) + 1} 個程序）")
        time.sleep(2)


def _smi_candidates() -> list[str]:
    win = os.environ.get("SystemRoot", r"C:\Windows")
    pf = os.environ.get("ProgramW6432") or os.environ.get("ProgramFiles", r"C:\Program Files")
    c = [shutil.which("nvidia-smi") or "",
         rf"{win}\System32\nvidia-smi.exe", rf"{win}\Sysnative\nvidia-smi.exe",
         rf"{pf}\NVIDIA Corporation\NVSMI\nvidia-smi.exe"]
    # 新版驅動也會放一份在 DriverStore
    c += [str(x) for x in Path(rf"{win}\System32\DriverStore\FileRepository").glob("nv*/nvidia-smi.exe")]
    return [x for x in dict.fromkeys(c) if x and Path(x).exists()]


def _driver_to_cuda(driver: float) -> tuple[int, int]:
    """NVIDIA 驅動版本 → 最高支援的 CUDA 版本（Windows）。"""
    for need, cuda in ((580.0, (13, 0)), (570.0, (12, 8)), (560.7, (12, 6)), (550.0, (12, 4)), (527.4, (12, 0))):
        if driver >= need:
            return cuda
    return (11, 0)


def _wmi_nvidia() -> tuple[str, float | None]:
    """用 Windows 的顯示卡資訊（不需要 nvidia-smi）找 NVIDIA 顯示卡和驅動版本。"""
    out = powershell("Get-CimInstance Win32_VideoController | ForEach-Object { $_.Name + '|' + $_.DriverVersion }")
    LOG.write("WMI video: " + out.replace("\n", " ; ") + "\n")
    for line in out.splitlines():
        name, _, ver = line.partition("|")
        if "nvidia" in name.lower():
            digits = "".join(ver.split(".")[-2:]) if ver.count(".") >= 3 else ""
            drv = float(f"{digits[-5:-2]}.{digits[-2:]}") if len(digits) >= 5 and digits.isdigit() else None
            return name.strip(), drv
    return "", None


def detect_gpu() -> tuple[str, str]:
    """回傳 (PyTorch 套件來源標籤, 說明)。偵測過程全部寫進 install.log。"""
    forced = next((a.split("=", 1)[1] for a in sys.argv if a.startswith("--gpu=")), "")
    if forced in ("cu128", "cu126", "cpu"):
        return forced, f"依指定安裝 {forced} 版"
    gpu, cuda, driver, cap = "", None, None, None
    for exe in _smi_candidates():
        try:
            r = subprocess.run([exe, "--query-gpu=name,driver_version", "--format=csv,noheader"],
                               capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=30)
            full = subprocess.run([exe], capture_output=True, text=True, encoding="utf-8",
                                  errors="replace", timeout=30).stdout
            LOG.write(f"nvidia-smi {exe}: rc={r.returncode} {r.stdout.strip()} {r.stderr.strip()[:200]}\n")
        except Exception as e:
            LOG.write(f"nvidia-smi {exe}: {type(e).__name__}: {e}\n")
            continue
        first = r.stdout.strip().splitlines()[0] if r.stdout.strip() else ""
        if "," in first:
            gpu, _, dv = first.partition(",")
            gpu = gpu.strip()
            try:
                driver = float(".".join(dv.strip().split(".")[:2]))
            except ValueError:
                pass
        m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", full)
        if m:
            cuda = (int(m.group(1)), int(m.group(2)))
        if gpu and cap is None:
            try:   # 顯示卡架構（例如 6.1＝GTX 10 系列、8.6＝RTX 30 系列、12.0＝RTX 50 系列）；舊驅動不支援這個欄位
                rc = subprocess.run([exe, "--query-gpu=compute_cap", "--format=csv,noheader"], capture_output=True,
                                    text=True, encoding="utf-8", errors="replace", timeout=30)
                mc = re.match(r"\s*(\d+)\.(\d+)", rc.stdout or "")
                if rc.returncode == 0 and mc:
                    cap = (int(mc.group(1)), int(mc.group(2)))
                LOG.write(f"compute_cap: rc={rc.returncode} {rc.stdout.strip()}\n")
            except Exception as e:
                LOG.write(f"compute_cap: {type(e).__name__}: {e}\n")
        if gpu or cuda:
            break
    if not gpu:   # 找不到 nvidia-smi 或它執行失敗：改看 Windows 的顯示卡清單
        gpu, driver = _wmi_nvidia()
    if not gpu:
        return "cpu", "沒有偵測到 NVIDIA 顯示卡，安裝 CPU 版（處理會比較慢）"
    if cuda is None and driver:
        cuda = _driver_to_cuda(driver)
    say(f"  偵測到 {gpu}（驅動 {driver or '版本不明'}，支援 CUDA {'%d.%d' % cuda if cuda else '不明'}"
        f"{'，架構 %d.%d' % cap if cap else ''}）")
    if cap is not None and cap < (5, 0):
        return "cpu", f"{gpu} 太舊，PyTorch 已不支援，安裝 CPU 版（處理會比較慢）"
    if cap is not None and cap < (7, 5):
        # GTX 10 系列以前（Pascal、Maxwell、Volta）：CUDA 12.8 版的 PyTorch 已不支援，只能用 12.6 版
        if cuda is None or cuda >= (12, 6):
            return "cu126", f"{gpu}，安裝顯示卡加速版（CUDA 12.6，支援較舊的顯示卡）"
        return "cpu", (f"{gpu} 的驅動程式太舊（支援到 CUDA {cuda[0]}.{cuda[1]}），先安裝 CPU 版。"
                       "更新顯示卡驅動後執行「修復與更新」即可改用顯示卡")
    if cuda is None:   # 有 NVIDIA 卡但讀不到版本：裝相容性較好的 12.6 版，安裝後再確認能不能用
        return "cu126", f"{gpu}：讀不到驅動版本，先安裝 CUDA 12.6 版"
    if cuda >= (12, 8):
        return "cu128", f"{gpu}，安裝顯示卡加速版（CUDA 12.8）"
    if cuda >= (12, 6):
        return "cu126", f"{gpu}，安裝顯示卡加速版（CUDA 12.6）"
    return "cpu", (f"{gpu} 的驅動程式太舊（支援到 CUDA {cuda[0]}.{cuda[1]}），先安裝 CPU 版。"
                   "更新顯示卡驅動後執行「修復與更新」即可改用顯示卡")


def installed_versions() -> dict:
    import importlib.metadata as md
    out = {}
    for name in list(TORCH) + ["onnxruntime", "onnxruntime-gpu"]:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    return out


DL = ROOT / "downloads"   # 安裝檔暫存（中斷後重裝可以接著下載）


def _fmt_mb(n: float) -> str:
    return f"{n / 1e9:.2f} GB" if n >= 1e9 else f"{n / 1e6:.0f} MB"


def _sha(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, sha256: str = "", label: str = "", part: tuple[float, float] = (0, 1)) -> Path:
    """下載檔案並更新進度條；中斷過的檔案會從斷點續傳，下載完核對 SHA-256。"""
    import urllib.request
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    if dest.exists() and (not sha256 or _sha(dest) == sha256):
        return dest
    dest.unlink(missing_ok=True)
    for attempt in range(4):
        have = tmp.stat().st_size if tmp.exists() else 0
        req = urllib.request.Request(url, headers={"User-Agent": "LeihuLiantai-installer",
                                                   **({"Range": f"bytes={have}-"} if have else {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                if have and r.status != 206:   # 伺服器不支援續傳：從頭來
                    have = 0
                total = have + int(r.headers.get("Content-Length") or 0)
                with open(tmp, "ab" if have else "wb") as f:
                    done, t0 = have, time.time()
                    while True:
                        chunk = r.read(1 << 20)
                        if not chunk:
                            break
                        f.write(chunk)
                        done += len(chunk)
                        speed = (done - have) / max(0.5, time.time() - t0)
                        frac = done / total if total else 0
                        progress(part[0] + (part[1] - part[0]) * frac,
                                 f"{label} {_fmt_mb(done)}/{_fmt_mb(total)}  {speed / 1e6:.1f} MB/s")
            break
        except Exception as e:
            LOG.write(f"download {url} attempt {attempt + 1}: {e}\n")
            if attempt == 3:
                raise
            time.sleep(3)
    if sha256 and _sha(tmp) != sha256:
        tmp.unlink(missing_ok=True)
        raise RuntimeError(f"{dest.name} 下載不完整（檢查碼不符），請重新執行安裝")
    tmp.replace(dest)
    return dest


def _wheel_links(tag: str) -> list[tuple[str, str, str]]:
    """從 PyTorch 官方套件清單找出三個 wheel 的網址與檢查碼。"""
    import urllib.parse
    import urllib.request
    out = []
    for name, ver in TORCH.items():
        index = f"https://download.pytorch.org/whl/{tag}/{name}/"
        html = urllib.request.urlopen(urllib.request.Request(index, headers={"User-Agent": "LeihuLiantai-installer"}),
                                      timeout=60).read().decode("utf-8", "replace")
        fname = f"{name}-{ver}%2B{tag}-cp311-cp311-win_amd64.whl"
        m = re.search(r'href="([^"]*' + re.escape(fname) + r')(?:#sha256=([0-9a-f]{64}))?"', html)
        if not m:
            raise RuntimeError(f"清單裡找不到 {fname}")
        out.append((urllib.parse.urljoin(index, m.group(1)), urllib.parse.unquote(fname), m.group(2) or ""))
    return out


def torch_ok() -> bool:
    ok, out = probe("import torch, torchvision, torchaudio; print(torch.__version__)", timeout=180)
    return ok


GPU_TEST = ("import torch; assert torch.cuda.is_available(), 'CUDA 無法使用'; "
            "x = torch.ones(64, device='cuda'); print(torch.cuda.get_device_name(0), float((x * 2).sum()))")


def check_gpu_kernels(tag: str) -> str:
    """實際在顯示卡上跑一個小運算。這張卡不被這個版本支援（no kernel image）時，改裝 12.6 版，再不行就改 CPU 版。"""
    while tag != "cpu":
        ok, out = probe(GPU_TEST)
        if ok:
            say(f"  ✓ 顯示卡運算測試通過（{out}）")
            return tag
        nxt = "cu126" if tag == "cu128" else "cpu"
        say(f"  ! 顯示卡運算測試失敗（{out[:160]}），改裝{'CUDA 12.6 版' if nxt == 'cu126' else ' CPU 版'}…")
        pip("uninstall", "-y", *TORCH)
        if not install_torch(nxt):
            return tag
        tag = nxt
    return tag


def install_torch(tag: str) -> bool:
    want = {k: f"{v}+{tag}" for k, v in TORCH.items()}
    have = installed_versions()
    if all(have.get(k) == v for k, v in want.items()):
        if torch_ok():
            say(f"  PyTorch 已是正確版本（{want['torch']}）")
            progress(1, "PyTorch 已安裝")
            return True
        say("  PyTorch 檔案不完整（可能是上次安裝中斷），重新安裝…")
    say(f"  下載 PyTorch {TORCH['torch']}（{tag}）{'，約 2.8 GB' if tag != 'cpu' else '，約 200 MB'}…")
    files = []
    try:
        links = _wheel_links(tag)
        for i, (url, fname, sha) in enumerate(links):
            # torch 本體佔了絕大部分，給它進度條的 85%
            part = (0, 0.85) if i == 0 else (0.85 + 0.05 * (i - 1), 0.85 + 0.05 * i)
            files.append(download(url, DL / fname, sha, fname.split("-")[0], part))
    except Exception as e:
        say(f"  直接下載失敗（{e}），改由 pip 下載…")
        files = []
    progress(0.95, "安裝 PyTorch 套件")
    if files:
        code = pip("install", "--force-reinstall", "--no-deps", *map(str, files))
    else:
        code = pip("install", "--force-reinstall", "--no-deps", *[f"{k}=={v}" for k, v in TORCH.items()],
                   "--index-url", f"https://download.pytorch.org/whl/{tag}",
                   on_line=lambda l: progress(0.5, l.strip()[:46]) if "Downloading" in l else None)
    if code and tag == "cu128":
        say("  CUDA 12.8 版安裝失敗，改試 CUDA 12.6 版…")
        return install_torch("cu126")
    if code == 0:
        for f in files:   # 裝好了就刪掉暫存的安裝檔，省下約 3 GB
            f.unlink(missing_ok=True)
        progress(1, "PyTorch 安裝完成")
    return code == 0


def install_packages(tag: str) -> bool:
    # 鎖住 PyTorch 版本，避免其他套件把它換掉
    cons = ROOT / "logs" / "_constraints.txt"
    have = installed_versions()
    cons.write_text("\n".join(f"{k}=={have[k]}" for k in TORCH if k in have) + "\n", "utf-8")
    say("  安裝其他套件（分離、音高、歌詞辨識、YouTube、視窗介面）…")
    # 內嵌版 Python 會忽略 PYTHONPATH，pip 的隔離建置環境找不到 setuptools；
    # 先裝好 setuptools，再用 --no-build-isolation 建置唯一沒有預編譯檔的 proxy_tools（純 Python）
    pip("install", "setuptools", "wheel")
    pip("install", "--no-build-isolation", "proxy_tools==0.1.0")
    seen = {"n": 0}

    def on_line(line: str) -> None:
        m = re.match(r"\s*(Collecting|Requirement already satisfied:)\s+([A-Za-z0-9_.\-]+)", line)
        if m:
            seen["n"] += 1
            progress(min(0.92, seen["n"] / 95), f"套件：{m.group(2)}")
        elif "Downloading" in line:
            mm = re.search(r"Downloading\s+(\S+)\s*\(([^)]+)\)", line)
            if mm:
                progress(min(0.92, seen["n"] / 95), f"下載 {mm.group(1).split('/')[-1][:28]} {mm.group(2)}")
        elif line.startswith("Installing collected packages"):
            progress(0.93, "安裝中…")

    code = pip("install", "-r", str(ROOT / "requirements-win.txt"), "-c", str(cons), "--upgrade-strategy",
               "only-if-needed", on_line=on_line)
    if code:
        say("  第一次安裝沒有成功，改用不隔離的建置方式重試…")
        code = pip("install", "--no-build-isolation", "-r", str(ROOT / "requirements-win.txt"), "-c", str(cons),
                   "--upgrade-strategy", "only-if-needed")
    if code:
        return False
    pip("install", "-U", "yt-dlp")   # YouTube 常改版，下載工具保持最新
    # faster-whisper 會帶進 CPU 版 onnxruntime；換成固定版本（顯示卡用 CUDA 12 版）
    pkg = "onnxruntime-gpu" if tag != "cpu" else "onnxruntime"
    pip("uninstall", "-y", "onnxruntime", "onnxruntime-gpu")
    return pip("install", "--force-reinstall", "--no-deps", f"{pkg}=={ORT_VERSION}") == 0


def clean_stale() -> None:
    """刪掉套件資料夾裡不屬於目前版本的殘留檔案（之前更新時被鎖住沒刪掉的舊 DLL）。"""
    import importlib.metadata as md
    owned = set()
    for dist in md.distributions():
        for f in dist.files or []:
            owned.add((SP / f).resolve())
    removed = 0
    for top in ("torch", "torchvision", "torchaudio", "onnxruntime", "nvidia"):
        d = SP / top
        if not d.exists():
            continue
        root = d.resolve()
        if top != "nvidia" and not any(root in o.parents for o in owned):
            continue
        for f in d.rglob("*"):
            if f.is_file() and "__pycache__" not in f.parts and f.resolve() not in owned:
                try:
                    f.unlink()
                    removed += 1
                except OSError:
                    pass
        for sub in sorted((x for x in d.rglob("*") if x.is_dir()), key=lambda x: -len(x.parts)):
            try:
                sub.rmdir()
            except OSError:
                pass
    for junk in SP.glob("~*"):
        shutil.rmtree(junk, ignore_errors=True)
    if removed:
        say(f"  清除舊版殘留檔案 {removed} 個")


CHECKS = [   # (名稱, 檢查程式, 壞掉時要重裝的套件)
    ("PyTorch", "import torch; print(torch.__version__, 'GPU' if torch.cuda.is_available() else 'CPU')", "torch"),
    ("torchvision", "import torch, torchvision; print(torchvision.__version__)", "torch"),
    ("人聲分離", "import torch, audio_separator.separator.architectures.mdx_separator as m, onnxruntime as o; "
                 "print(o.__version__, o.get_available_providers()[0])", ["audio-separator", "onnx", "onnx2torch-py313"]),
    ("歌詞辨識", "import torch, ctranslate2; print('GPU' if ctranslate2.get_cuda_device_count() else 'CPU')",
     ["faster-whisper", "ctranslate2"]),
    ("音高分析", "import torchcrepe; print('ok')", ["torchcrepe"]),
    ("視窗介面", "import webview; print('ok')", ["pywebview", "pythonnet"]),
    ("程式本體", "import app; print(app.VERSION)", ["flask", "numpy", "scipy", "soundfile", "librosa", "pedalboard",
                                                     "opencc-python-reimplemented", "anthropic", "imageio-ffmpeg"]),
]


def _pinned(pkgs: list[str]) -> list[str]:
    req = {}
    for line in (ROOT / "requirements-win.txt").read_text("utf-8").splitlines():
        m = re.match(r"\s*([A-Za-z0-9_.\-]+)\s*==\s*([^\s;#]+)", line)
        if m:
            req[m.group(1).lower()] = m.group(2)
    return [f"{p}=={req[p.lower()]}" if p.lower() in req else p for p in pkgs]


def verify(tag: str) -> bool:
    """檢查每個元件；壞掉的自動重裝一次再檢查（處理安裝中斷造成的半套檔案）。"""
    ok_all = True
    for i, (name, code, fix) in enumerate(CHECKS):
        progress(i / len(CHECKS), f"檢查 {name}")
        ok, out = probe(code)
        if not ok:
            say(f"  ! {name} 無法載入（{out}），自動修復…")
            if fix == "torch":
                pip("uninstall", "-y", *TORCH)
                install_torch(tag)
            else:
                pip("install", "--force-reinstall", "--no-deps", *_pinned(fix))
                if name == "人聲分離":
                    pkg = "onnxruntime-gpu" if tag != "cpu" else "onnxruntime"
                    pip("install", "--force-reinstall", "--no-deps", f"{pkg}=={ORT_VERSION}")
            ok, out = probe(code)
        say(f"  {'✓' if ok else '✗'} {name}：{out}")
        ok_all &= ok or name == "視窗介面"
    model = ROOT / "models" / "uvr" / SEP_MODELS[0]
    if tag != "cpu" and model.exists():
        ok, out = probe("import torch, onnxruntime as o; s=o.InferenceSession(r'%s', providers=['CUDAExecutionProvider','CPUExecutionProvider']); print(s.get_providers()[0])" % model)
        say(f"  {'✓' if ok and 'CUDA' in out else '!'} 人聲分離使用：{out}")
    progress(1, "檢查完成")
    return ok_all


def _dir_size(d: Path) -> int:
    return sum(f.stat().st_size for f in d.rglob("*") if f.is_file()) if d.exists() else 0


def _watch(d: Path, expect: float, label: str, part: tuple[float, float]):
    """背景監看資料夾大小，換算成下載進度。回傳停止用的函式。"""
    import threading
    stop = threading.Event()
    base = _dir_size(d)

    def loop():
        while not stop.wait(1.0):
            got = _dir_size(d) - base
            progress(part[0] + (part[1] - part[0]) * min(0.98, got / expect), f"{label} {_fmt_mb(got)}/{_fmt_mb(expect)}")
    threading.Thread(target=loop, daemon=True).start()
    return stop.set


def model_ok(path: Path) -> bool:
    """模型檔存在、大小合理，而且 ONNX 能讀得起來（排除下載到一半的壞檔）。"""
    if not path.exists() or path.stat().st_size < 10_000_000:
        return False
    ok, _ = probe(f"import onnxruntime as o; o.InferenceSession(r'{path}', providers=['CPUExecutionProvider']); print('ok')",
                  timeout=120)
    return ok


def prefetch_models(tag: str = "cpu") -> None:
    uvr = ROOT / "models" / "uvr"
    uvr.mkdir(parents=True, exist_ok=True)
    missing = []
    for m in SEP_MODELS:
        if not model_ok(uvr / m):
            if (uvr / m).exists():
                say(f"  模型 {m} 不完整，刪除後重新下載")
                (uvr / m).unlink(missing_ok=True)
            missing.append(m)
    if missing:
        say("  下載人聲分離模型（約 120 MB，來源伺服器較慢，可能要 5–10 分鐘）…")
        stop = _watch(uvr, 60e6 * len(missing), "人聲分離模型", (0, 0.2))
        code = ("import pipeline\nfrom audio_separator.separator import Separator\n"
                f"s = Separator(model_file_dir=r'{uvr}', output_dir=r'{ROOT / 'logs'}')\n"
                + "".join(f"s.download_model_and_data(r'{m}')\n" for m in missing) + "print('ok')")
        ok, out = probe(code, timeout=3600)
        if not ok:   # 舊版 audio-separator 沒有 download_model_and_data，改用 load_model 觸發下載
            code = ("import pipeline\nfrom audio_separator.separator import Separator\n"
                    f"s = Separator(model_file_dir=r'{uvr}', output_dir=r'{ROOT / 'logs'}')\n"
                    + "".join(f"s.load_model(model_filename=r'{m}')\n" for m in missing) + "print('ok')")
            ok, out = probe(code, timeout=3600)
        stop()
        ok = ok and all(model_ok(uvr / m) for m in missing)
        say(f"  {'✓' if ok else '✗'} 人聲分離模型{'' if ok else '下載失敗，第一次匯入歌曲時會再試：' + out}")
    else:
        say("  ✓ 人聲分離模型已存在")
    if tag != "cpu":   # 有顯示卡：預先下載高品質主唱分離模型（約 900 MB），第一次匯入歌曲就不用等
        hq = uvr / HQ_KARAOKE
        if hq.exists() and hq.stat().st_size > 900_000_000:
            say("  ✓ 高品質主唱分離模型已存在")
        else:
            say("  下載高品質主唱分離模型（約 900 MB）…")
            stop = _watch(uvr, 913e6, "高品質主唱分離模型", (0.1, 0.2))
            code = ("import pipeline\nfrom audio_separator.separator import Separator\n"
                    f"s = Separator(model_file_dir=r'{uvr}', output_dir=r'{ROOT / 'logs'}')\n"
                    f"s.download_model_and_data(r'{HQ_KARAOKE}')\nprint('ok')")
            ok, out = probe(code, timeout=3600)
            stop()
            ok = ok and hq.exists() and hq.stat().st_size > 900_000_000
            say(f"  {'✓' if ok else '!'} 高品質主唱分離模型{'' if ok else '下載失敗，第一次用到時會再試：' + out}")
    progress(0.2, "歌詞辨識模型")
    wdir = ROOT / "models" / "whisper"
    say(f"  下載歌詞辨識模型 {WHISPER}（約 1.6 GB，已下載的部分會接著下載）…")
    stop = _watch(wdir, max(1e8, 1.62e9 - _dir_size(wdir)), "歌詞辨識模型", (0.2, 1))
    ok, out = probe("from faster_whisper.utils import download_model; "
                    f"download_model('{WHISPER}', cache_dir=r'{wdir}'); print('ok')", timeout=3600)
    stop()
    progress(1, "模型準備完成")
    say(f"  {'✓' if ok else '✗'} 歌詞辨識模型{'' if ok else '下載失敗，第一次匯入歌曲時會再試：' + out}")


def launch() -> None:
    subprocess.Popen([str(PYW if PYW.exists() else PY), str(ROOT / "desktop.py")], cwd=str(ROOT),
                     creationflags=0x00000008 | 0x00000200)


def main() -> int:
    repair = "--repair" in sys.argv
    version = (ROOT / "VERSION").read_text("utf-8").strip() if (ROOT / "VERSION").exists() else "?"
    say(f"\n===== 雷虎練唱台 {version} {'修復與更新' if repair else '安裝'}　{time.strftime('%Y-%m-%d %H:%M')} =====")
    done_file = ROOT / "installed.json"
    if done_file.exists():
        try:
            prev = json.loads(done_file.read_text("utf-8"))
            say(f"  偵測到已安裝的版本：{prev.get('version')}（{prev.get('torch')}），這次會更新並檢查")
        except ValueError:
            pass
    elif any(SP.glob("torch*")) or any(SP.glob("flask*")):
        say("  偵測到上次安裝沒有完成，這次會清除不完整的檔案並接著安裝（已下載好的部分不會重來）")
    # 安裝中先拿掉完成標記；中途被關掉時，下次就知道要修復
    done_file.unlink(missing_ok=True)
    total = 6
    stage(1, total, "準備", 0, 2)
    stop_app()
    clean_stale()
    tag, desc = detect_gpu()
    say("  " + desc)
    stage(2, total, "安裝 PyTorch", 2, 45)
    if not install_torch(tag):
        say("\n安裝 PyTorch 失敗。請確認網路連線後重新執行，已下載的部分會接著下載。")
        return 1
    if tag == "cu128" and "+cu126" in installed_versions().get("torch", ""):
        tag = "cu126"
    tag = check_gpu_kernels(tag)
    stage(3, total, "安裝其他套件", 45, 70)
    if not install_packages(tag):
        say("\n套件安裝失敗。請確認網路連線後重新執行。")
        return 1
    clean_stale()
    stage(4, total, "下載 AI 模型", 70, 94)
    prefetch_models(tag)
    stage(5, total, "檢查", 94, 99)
    ok = verify(tag)
    stage(6, total, "完成", 99, 100)
    done_file.write_text(json.dumps(
        {"version": version, "torch": tag, "time": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False), "utf-8")
    shutil.rmtree(DL, ignore_errors=True)
    progress(1, "完成")
    if not ok:
        say("有元件檢查沒有通過，程式仍可開啟，但部分功能可能無法使用。紀錄在 logs\\install.log。")
    say("雷虎練唱台已準備好。")
    if "--launch" in sys.argv:
        launch()
    return 0


if __name__ == "__main__":
    try:
        code = main()
    except Exception:
        import traceback
        say(traceback.format_exc())
        code = 1
    if "--pause" in sys.argv:
        input("\n按 Enter 關閉這個視窗…")
    sys.exit(code)
