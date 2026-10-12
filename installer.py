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
WHISPER = "large-v3-turbo"

if sys.platform == "win32":
    import ctypes
    ctypes.windll.kernel32.SetErrorMode(0x0001 | 0x0002 | 0x8000)   # DLL 錯誤不要跳視窗卡住


def say(msg: str = "") -> None:
    print(msg, flush=True)
    LOG.write(msg + "\n")


def step(n: int, total: int, title: str) -> None:
    say(f"\n[{n}/{total}] {title}")


def run(args: list[str], quiet: bool = True) -> int:
    LOG.write("> " + " ".join(args) + "\n")
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PIP_DISABLE_PIP_VERSION_CHECK="1",
               PIP_NO_WARN_SCRIPT_LOCATION="1")
    p = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                         encoding="utf-8", errors="replace", env=env)
    for line in p.stdout:
        line = line.rstrip()
        LOG.write("  " + line + "\n")
        if not quiet or line.startswith(("ERROR", "Successfully installed")) or "Downloading" in line and "MB)" in line:
            print("  " + line[:150], flush=True)
    return p.wait()


def pip(*args: str) -> int:
    return run([str(PY), "-m", "pip", *args])


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


def detect_gpu() -> tuple[str, str]:
    """回傳 (PyTorch 套件來源標籤, 說明)。"""
    try:
        smi = subprocess.run(["nvidia-smi"], capture_output=True, text=True, encoding="utf-8",
                             errors="replace", timeout=20).stdout
    except Exception:
        smi = ""
    m = re.search(r"CUDA Version:\s*(\d+)\.(\d+)", smi)
    name = re.search(r"\|\s+\d+\s+(NVIDIA[^|]+?)\s{2,}", smi)
    gpu = name.group(1).strip() if name else "NVIDIA 顯示卡"
    if not m:
        return "cpu", "沒有偵測到 NVIDIA 顯示卡，安裝 CPU 版（處理會比較慢）"
    cuda = (int(m.group(1)), int(m.group(2)))
    if cuda >= (12, 8):
        return "cu128", f"{gpu}（驅動支援 CUDA {cuda[0]}.{cuda[1]}），安裝顯示卡加速版"
    if cuda >= (12, 6):
        return "cu126", f"{gpu}（驅動支援 CUDA {cuda[0]}.{cuda[1]}），安裝顯示卡加速版"
    return "cpu", f"{gpu} 的驅動程式太舊（CUDA {cuda[0]}.{cuda[1]}），安裝 CPU 版。更新顯示卡驅動後執行「修復與更新」即可改用顯示卡"


def installed_versions() -> dict:
    import importlib.metadata as md
    out = {}
    for name in list(TORCH) + ["onnxruntime", "onnxruntime-gpu"]:
        try:
            out[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    return out


def install_torch(tag: str) -> bool:
    want = {k: f"{v}+{tag}" for k, v in TORCH.items()}
    have = installed_versions()
    if all(have.get(k) == v for k, v in want.items()):
        say(f"  PyTorch 已是正確版本（{want['torch']}）")
        return True
    say(f"  下載 PyTorch {TORCH['torch']}（{tag}）{'，約 2.8 GB' if tag != 'cpu' else '，約 200 MB'}…")
    code = pip("install", "--force-reinstall", "--no-deps",
               *[f"{k}=={v}" for k, v in TORCH.items()],
               "--index-url", f"https://download.pytorch.org/whl/{tag}")
    if code and tag == "cu128":
        say("  CUDA 12.8 版安裝失敗，改試 CUDA 12.6 版…")
        return install_torch("cu126")
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
    code = pip("install", "-r", str(ROOT / "requirements-win.txt"), "-c", str(cons), "--upgrade-strategy", "only-if-needed")
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


def verify(tag: str) -> bool:
    model = ROOT / "models" / "uvr" / SEP_MODELS[0]
    checks = [
        ("PyTorch", "import torch; print(torch.__version__, 'GPU' if torch.cuda.is_available() else 'CPU')"),
        ("torchvision", "import torch, torchvision; print(torchvision.__version__)"),
        ("人聲分離", "import torch, audio_separator.separator.architectures.mdx_separator as m, onnxruntime as o; "
                     "print(o.__version__, o.get_available_providers()[0])"),
        ("歌詞辨識", "import torch, ctranslate2; print('GPU' if ctranslate2.get_cuda_device_count() else 'CPU')"),
        ("音高分析", "import torchcrepe; print('ok')"),
        ("視窗介面", "import webview; print('ok')"),
        ("程式本體", "import app; print(app.VERSION)"),
    ]
    ok_all = True
    for name, code in checks:
        ok, out = probe(code)
        say(f"  {'✓' if ok else '✗'} {name}：{out}")
        ok_all &= ok or name == "視窗介面"
    if tag != "cpu" and model.exists():
        ok, out = probe("import torch, onnxruntime as o; s=o.InferenceSession(r'%s', providers=['CUDAExecutionProvider','CPUExecutionProvider']); print(s.get_providers()[0])" % model)
        say(f"  {'✓' if ok and 'CUDA' in out else '!'} 人聲分離使用：{out}")
    return ok_all


def prefetch_models() -> None:
    (ROOT / "models" / "uvr").mkdir(parents=True, exist_ok=True)
    have_sep = all((ROOT / "models" / "uvr" / m).exists() for m in SEP_MODELS)
    if not have_sep:
        say("  下載人聲分離模型（約 120 MB，來源伺服器較慢，可能要 5–10 分鐘）…")
        code = ("import pipeline\nfrom audio_separator.separator import Separator\n"
                f"s = Separator(model_file_dir=r'{ROOT / 'models' / 'uvr'}', output_dir=r'{ROOT / 'logs'}')\n"
                + "".join(f"s.download_model_and_data(r'{m}')\n" for m in SEP_MODELS) + "print('ok')")
        ok, out = probe(code, timeout=3600)
        if not ok:   # 舊版 audio-separator 沒有 download_model_and_data，改用 load_model 觸發下載
            code = ("import pipeline\nfrom audio_separator.separator import Separator\n"
                    f"s = Separator(model_file_dir=r'{ROOT / 'models' / 'uvr'}', output_dir=r'{ROOT / 'logs'}')\n"
                    + "".join(f"s.load_model(model_filename=r'{m}')\n" for m in SEP_MODELS) + "print('ok')")
            ok, out = probe(code, timeout=3600)
        say(f"  {'✓' if ok else '✗'} 人聲分離模型{'' if ok else '下載失敗，第一次匯入歌曲時會再試：' + out}")
    else:
        say("  ✓ 人聲分離模型已存在")
    say(f"  下載歌詞辨識模型 {WHISPER}（約 1.6 GB）…")
    ok, out = probe("from faster_whisper.utils import download_model; "
                    f"download_model('{WHISPER}', cache_dir=r'{ROOT / 'models' / 'whisper'}'); print('ok')", timeout=3600)
    say(f"  {'✓' if ok else '✗'} 歌詞辨識模型{'' if ok else '下載失敗，第一次匯入歌曲時會再試：' + out}")


def launch() -> None:
    subprocess.Popen([str(PYW if PYW.exists() else PY), str(ROOT / "desktop.py")], cwd=str(ROOT),
                     creationflags=0x00000008 | 0x00000200)


def main() -> int:
    repair = "--repair" in sys.argv
    version = (ROOT / "VERSION").read_text("utf-8").strip() if (ROOT / "VERSION").exists() else "?"
    say(f"\n===== 雷虎練唱台 {version} {'修復與更新' if repair else '安裝'}　{time.strftime('%Y-%m-%d %H:%M')} =====")
    total = 6
    step(1, total, "準備")
    stop_app()
    tag, desc = detect_gpu()
    say("  " + desc)
    step(2, total, "安裝 PyTorch")
    if not install_torch(tag):
        say("\n安裝 PyTorch 失敗。請確認網路連線後重新執行。")
        return 1
    if tag == "cu128" and "+cu126" in installed_versions().get("torch", ""):
        tag = "cu126"
    step(3, total, "安裝其他套件")
    if not install_packages(tag):
        say("\n套件安裝失敗。請確認網路連線後重新執行。")
        return 1
    clean_stale()
    step(4, total, "下載 AI 模型")
    prefetch_models()
    step(5, total, "檢查")
    ok = verify(tag)
    step(6, total, "完成")
    (ROOT / "installed.json").write_text(json.dumps(
        {"version": version, "torch": tag, "time": time.strftime("%Y-%m-%d %H:%M")}, ensure_ascii=False), "utf-8")
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
