"""練唱台：音訊處理流程。

每首歌存在 library/<id>/：
  original.*        匯入的原始檔
  mix.wav           44.1kHz 立體聲
  lead.wav          主唱
  back.wav          合音
  inst.wav          伴奏
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

SR = 44100
PITCH_SR = 16000
HOP = 160            # 10 ms
HOP_S = HOP / PITCH_SR
STEMS = ("lead", "back", "inst")

DEFAULT_CONFIG = {
    "separation_model": "UVR-MDX-NET-Inst_HQ_3.onnx",
    "karaoke_model": "UVR_MDXNET_KARA_2.onnx",
    "whisper_model": "large-v3-turbo",
    "lyrics_language": "zh",
    "convert_to_taiwan": True,
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
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), "utf-8")


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
        "cuda": _cuda(),
    }


def _cuda() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available())
    except Exception:
        return False


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


def read_meta(d: Path) -> dict:
    p = d / "meta.json"
    return json.loads(p.read_text("utf-8")) if p.exists() else {}


def write_meta(d: Path, meta: dict) -> None:
    tmp = d / "meta.json.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(d / "meta.json")


def update_meta(d: Path, **kw) -> dict:
    meta = read_meta(d)
    meta.update(kw)
    write_meta(d, meta)
    return meta


def list_songs() -> list[dict]:
    LIB.mkdir(exist_ok=True)
    out = []
    for d in LIB.iterdir():
        if d.is_dir() and (d / "meta.json").exists():
            m = read_meta(d)
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
            out.append(m)
    out.sort(key=lambda m: m.get("created", 0), reverse=True)
    return out


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


def new_youtube_song(url: str) -> str:
    return _blank_song("YouTube 下載中…", source_url=url)


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

    report("取得影片資訊", 0.01)
    code, title, err = attempt()
    if code:
        if re.search(r"Private video|unavailable|Sign in|members-only|age", err):
            raise UserError(f"這部影片無法下載（可能是私人、會員限定或需要登入）：{err.strip()[-200:]}")
        report("下載失敗，更新 yt-dlp 後重試", 0.02)
        _upgrade_ytdlp()
        code, title, err = attempt()
        if code:
            raise UserError(f"YouTube 下載失敗：{err.strip()[-240:]}")
    for f in d.glob("original.*"):
        if f.suffix.lower() != ".wav":
            f.unlink(missing_ok=True)
    if not (d / "original.wav").exists():
        raise UserError("YouTube 下載完成，但找不到轉換後的音訊檔")
    update_meta(d, title=(title or "YouTube 歌曲")[:120], source_url=url)


def youtube_job(song_id: str, url: str):
    process = process_song(song_id, ["import", "separate", "pitch", "lyrics"])

    def run(report):
        d = song_dir(song_id)
        try:
            step_youtube(d, url, report)
        except Exception as e:
            meta = read_meta(d)
            if meta.get("title", "").startswith("YouTube 下載中"):
                meta["title"] = "YouTube 下載失敗"
            meta["notes"] = [f"{e}　可以刪除這一項後重試。"]
            write_meta(d, meta)
            raise
        process(report)
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
    data = y.T if y.ndim == 2 else y
    sf.write(str(path), np.clip(data, -1.0, 1.0), sr, subtype="PCM_16")


def fit_length(y: np.ndarray, n: int) -> np.ndarray:
    if y.ndim == 1:
        y = np.stack([y, y])
    if y.shape[1] >= n:
        return y[:, :n]
    return np.pad(y, ((0, 0), (0, n - y.shape[1])))


# ---------------------------------------------------------------- 匯入

def step_import(d: Path, report) -> None:
    report("讀取音訊", 0.02)
    src = next(d.glob("original.*"))
    try:
        y = read_audio(src)
    except Exception as e:
        raise UserError(
            f"無法讀取這個音訊檔（{src.suffix}）。wav、mp3、flac、ogg 可直接讀；"
            "m4a/aac 需要先安裝 ffmpeg。") from e
    write_wav(d / "mix.wav", y)
    steps = read_meta(d).get("steps", {})
    steps["import"] = True
    update_meta(d, duration=round(y.shape[1] / SR, 3), steps=steps)


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


def separate_uvr(d: Path, cfg: dict, report) -> None:
    try:
        import torch  # noqa: F401  先載入 PyTorch 的 CUDA 函式庫，ONNX 會沿用同一份
    except Exception:
        pass
    from audio_separator.separator import Separator

    tmp = d / "_sep"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    (MODELS / "uvr").mkdir(parents=True, exist_ok=True)
    sep = Separator(output_dir=str(tmp), model_file_dir=str(MODELS / "uvr"),
                    output_format="WAV", sample_rate=SR)

    report("分離人聲與伴奏（第一次會下載模型）", 0.08)
    sep.load_model(model_filename=cfg["separation_model"])
    outs = _outputs(sep.separate(str(d / "mix.wav")), tmp)
    voc = tmp / "vocals_only.wav"
    _pick(outs, "vocals").replace(voc)
    inst_p = _pick(outs, "instrumental")

    report("把人聲分成主唱與合音", 0.35)
    sep.load_model(model_filename=cfg["karaoke_model"])
    outs2 = [p for p in _outputs(sep.separate(str(voc)), tmp) if p != voc and p != inst_p]
    lead_p = _pick(outs2, "vocals")
    back_p = _pick(outs2, "instrumental")

    n = sf.info(str(d / "mix.wav")).frames
    for name, p in (("lead", lead_p), ("back", back_p), ("inst", inst_p)):
        write_wav(d / f"{name}.wav", fit_length(read_audio(p), n))
    shutil.rmtree(tmp, ignore_errors=True)


def separate_dsp(d: Path, report) -> None:
    """沒有 AI 模型時的備援：依聲像位置分離（中央=主唱）。"""
    import librosa
    from scipy.ndimage import uniform_filter1d

    report("以聲像位置分離（備援模式）", 0.1)
    mix = read_audio(d / "mix.wav")
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
    write_wav(d / "lead.wav", np.stack([lead, lead]))
    write_wav(d / "back.wav", back)
    write_wav(d / "inst.wav", inst)


def is_acapella(d: Path) -> bool:
    """伴奏軌幾乎沒有聲音 → 原曲本來就是清唱／純人聲。"""
    try:
        def rms(name):
            y, _ = sf.read(str(d / f"{name}.wav"), dtype="float32", always_2d=True)
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
    meta = read_meta(d)
    meta["steps"]["separate"] = True
    meta["separation"] = method
    meta["acapella"] = is_acapella(d)
    meta["notes"] = [x for x in meta.get("notes", []) if "分離" not in x] + notes
    write_meta(d, meta)


# ---------------------------------------------------------------- 音高

def track_pitch(y16: np.ndarray, cfg: dict | None = None, strict: bool = False) -> np.ndarray:
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
        dev = "cuda:0" if torch.cuda.is_available() else "cpu"
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


def step_pitch(d: Path, cfg: dict, report) -> None:
    report("分析主唱旋律", 0.55)
    y16 = read_audio(d / "lead.wav", sr=PITCH_SR, mono=True)
    midi = track_pitch(y16, cfg)
    notes = segment_notes(midi)
    data = {
        "hop": HOP_S,
        "midi": [None if np.isnan(v) else round(float(v), 2) for v in midi],
        "notes": notes,
    }
    (d / "pitch.json").write_text(json.dumps(data, separators=(",", ":")), "utf-8")
    meta = read_meta(d)
    meta["steps"]["pitch"] = True
    write_meta(d, meta)


def load_pitch(d: Path) -> tuple[np.ndarray, list]:
    data = json.loads((d / "pitch.json").read_text("utf-8"))
    midi = np.array([np.nan if v is None else v for v in data["midi"]], dtype=np.float32)
    return midi, data["notes"]


# ---------------------------------------------------------------- 歌詞

HALLUCINATION = re.compile(r"字幕|訂閱|订阅|點贊|点赞|感謝觀看|感谢观看|Amara|請不吝|请不吝|明鏡|明镜")
PUNCT_END = "，。！？、,.!?;；：:…"


def to_taiwan(text: str) -> str:
    try:
        from opencc import OpenCC
    except ImportError:
        return text
    for conf in ("s2twp", "s2twp.json"):
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
        return WhisperModel(name, device="auto", compute_type="default", download_root=root)
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


def step_lyrics(d: Path, cfg: dict, report) -> None:
    if not module_status()["whisper"]:
        raise UserError("沒有安裝 faster-whisper，無法辨識歌詞")
    report("載入語音辨識模型（第一次會下載）", 0.7)
    model = _whisper_model(cfg)
    y16 = read_audio(d / "lead.wav", sr=PITCH_SR, mono=True)
    report("辨識歌詞", 0.75)
    lang = cfg.get("lyrics_language") or None
    try:
        segs, info = model.transcribe(
            y16, language=lang, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500},
            word_timestamps=True, condition_on_previous_text=False, beam_size=5)
        words = [w for s in segs for w in (s.words or [])]
    except RuntimeError:
        # 例如有顯示卡但缺 CUDA 函式庫：改用 CPU 重跑
        traceback.print_exc()
        from faster_whisper import WhisperModel
        model = WhisperModel(cfg.get("whisper_model") or "large-v3-turbo", device="cpu",
                             compute_type="int8", download_root=str(MODELS / "whisper"))
        segs, info = model.transcribe(y16, language=lang, vad_filter=True,
                                      word_timestamps=True, condition_on_previous_text=False)
        words = [w for s in segs for w in (s.words or [])]

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
    (d / "lyrics_auto.lrc").write_text(lrc, "utf-8")
    (d / "lyrics.lrc").write_text(lrc, "utf-8")
    meta = read_meta(d)
    meta["steps"]["lyrics"] = True
    meta["lyrics_source"] = "whisper"
    meta["language"] = language
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
    todo = [s for s in STEMS if not (vd / f"{s}.wav").exists()]
    if not todo:
        return
    label = f"移調 {key:+d}、速度 {int(speed * 100)}%"
    cpus = max(1, os.cpu_count() or 1)
    report(f"{label}：讀取音軌", 0.02)
    data = {}
    for stem in todo:
        y, sr = sf.read(str(d / f"{stem}.wav"), dtype="float32", always_2d=True)
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
        write_wav(vd / f"{stem}.tmp.wav", out, sr)
        (vd / f"{stem}.tmp.wav").replace(vd / f"{stem}.wav")
    report(f"{label}：完成", 1.0)


def variant_ready(d: Path, key: int, speed: float) -> bool:
    vd = variant_dir(d, key, speed)
    return all((vd / f"{s}.wav").exists() for s in STEMS)


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
        (td / f"{take_id}.json").write_text(json.dumps(info, ensure_ascii=False), "utf-8")
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
    (td / f"{take_id}.json").write_text(json.dumps(info, ensure_ascii=False), "utf-8")


# ---------------------------------------------------------------- 背景工作

class JobCancelled(Exception):
    """工作被新的要求取代（例如又換了 KEY），中途停止。"""


class Jobs:
    """一次只跑一個重工作，避免記憶體爆掉。"""

    def __init__(self):
        self.q: queue.Queue = queue.Queue()
        self.state: dict[str, dict] = {}
        self.lock = threading.Lock()
        threading.Thread(target=self._run, daemon=True).start()

    def submit(self, kind: str, song_id: str, fn, label: str = "") -> str:
        jid = uuid.uuid4().hex[:10]
        with self.lock:
            self.state[jid] = {"id": jid, "kind": kind, "song": song_id, "label": label,
                               "status": "queued", "step": "排隊中", "progress": 0.0,
                               "error": None, "created": time.time()}
        self.q.put((jid, fn))
        return jid

    def update(self, jid: str, **kw) -> None:
        with self.lock:
            self.state[jid].update(kw)

    def get(self, jid: str) -> dict | None:
        with self.lock:
            j = self.state.get(jid)
            return dict(j) if j else None

    def active(self) -> list[dict]:
        cutoff = time.time() - 600
        with self.lock:
            return [dict(j) for j in self.state.values() if j["status"] != "cancelled"
                    and (j["status"] in ("queued", "running") or j["created"] > cutoff)]

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

    def position(self, jid: str) -> int:
        """前面還有幾個工作。"""
        with self.lock:
            me = self.state.get(jid)
            if not me or me["status"] != "queued":
                return 0
            return sum(1 for j in self.state.values()
                       if j["status"] in ("queued", "running") and j["created"] < me["created"])

    def find(self, kind: str, song_id: str, label: str = "") -> str | None:
        with self.lock:
            for j in self.state.values():
                if (j["kind"], j["song"], j["label"]) == (kind, song_id, label) \
                        and j["status"] in ("queued", "running") and not j.get("cancel"):
                    return j["id"]
        return None

    def _run(self):
        while True:
            jid, fn = self.q.get()
            if self.get(jid)["status"] == "cancelled":
                continue
            self.update(jid, status="running", step="開始")

            def report(step, p, jid=jid):
                if self.state[jid].get("cancel"):
                    raise JobCancelled()
                self.update(jid, step=step, progress=round(p, 3))
            try:
                fn(report)
                self.update(jid, status="done", step="完成", progress=1.0)
            except JobCancelled:
                self.update(jid, status="cancelled", step="已取消")
            except UserError as e:
                self.update(jid, status="error", error=str(e))
            except Exception as e:
                traceback.print_exc()
                self.update(jid, status="error", error=f"{type(e).__name__}: {e}")


def process_song(song_id: str, steps: list[str]):
    def run(report):
        d = song_dir(song_id)
        cfg = load_config()
        meta = read_meta(d)
        if "import" in steps or not meta["steps"].get("import"):
            step_import(d, report)
        if "separate" in steps:
            step_separate(d, cfg, report)
        if "pitch" in steps:
            step_pitch(d, cfg, report)
        if "lyrics" in steps:
            try:
                step_lyrics(d, cfg, report)
                meta = read_meta(d)
                meta["notes"] = [x for x in meta.get("notes", []) if "歌詞" not in x]
                write_meta(d, meta)
            except Exception as e:
                traceback.print_exc()
                meta = read_meta(d)
                meta["notes"] = [x for x in meta.get("notes", []) if "歌詞" not in x] + [
                    f"歌詞辨識沒有完成：{e}。可以在練唱畫面貼上歌詞。"]
                write_meta(d, meta)
            report("搜尋線上歌詞並與辨識結果比對", 0.95)
            try:
                find_and_store_candidates(d, cfg)
            except Exception:
                traceback.print_exc()   # 沒有網路或網站不通時，維持辨識結果
    return run


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


USER_AGENT = "LeihuLiantai/1.3.1 (+https://github.com/LeihuTW/Leihu-singsing)"   # HTTP 標頭只能用英文字


def _http_json(url: str, headers: dict | None = None, timeout: int = 10):
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


def find_candidates(d: Path, cfg: dict, query: str = "") -> tuple[list[str], list[dict]]:
    """用歌名（原文與簡體）到 LRCLIB、網易雲和酷狗搜尋，與辨識結果比對後排序。"""
    meta = read_meta(d)
    queries = [query] if query else lyric_queries(meta)
    tried, found, errors = [], {}, []
    for q in queries:
        for qq in dict.fromkeys([q, to_simplified(q)]):
            tried.append(qq)
            for fn in (search_lrclib, search_netease, search_kugou):
                try:
                    for c in fn(qq):
                        found.setdefault(c["id"], c)
                except Exception as e:
                    errors.append(f"{fn.__name__}: {type(e).__name__}")
                    log_error(f"{fn.__name__}({qq})", e)
        if len(found) >= 8:
            break
    if not found and errors and len(errors) >= len(tried) * 3:
        raise UserError("無法連到歌詞網站，請確認網路後再試。")
    rec = recognized_text(d)
    dur = meta.get("duration")
    to_tw = bool(cfg.get("convert_to_taiwan", True))
    out = []
    for c in found.values():
        c["similarity"] = similarity(rec, c["plain"] or c["synced"])
        c["diff"] = round(abs(c["duration"] - dur), 1) if (c.get("duration") and dur) else None
        if to_tw and re.search(r"[\u4e00-\u9fff]", c["plain"] + c["synced"] + c["track"] + c["artist"]):
            for k in ("plain", "synced", "track", "artist", "album"):
                c[k] = to_taiwan(c[k])
        lines = [l for l in c["plain"].splitlines() if l.strip()]
        c["preview"] = "\n".join(lines[:3])
        c["lines"] = len(lines)
        out.append(c)

    def score(c):
        sim = c["similarity"] if c["similarity"] is not None else 0.3
        pen = min((c["diff"] or 0) / 30, 0.3) if c["diff"] is not None else 0.1
        return -(sim - pen + (0.05 if c["synced"] else 0))
    out.sort(key=score)
    return list(dict.fromkeys(tried)), out[:10]


def find_and_store_candidates(d: Path, cfg: dict) -> None:
    _, cands = find_candidates(d, cfg)
    # 至少要有一點像，才請使用者挑選；完全不像的多半是同名的其他歌
    good = [c for c in cands if c["similarity"] is None or c["similarity"] >= 0.25]
    (d / "lyrics_candidates.json").write_text(json.dumps(good, ensure_ascii=False), "utf-8")
    meta = read_meta(d)
    if meta.get("lyrics_source") in (None, "whisper"):
        update_meta(d, lyrics_pending=bool(good))


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
        lrc = to_lrc(align_to_recognized(lines, d))
        how = "歌詞檔沒有時間，已依辨識結果自動對上每一句" if auto.exists() else \
              "歌詞檔沒有時間，暫時每句間隔 3.5 秒，可以用「跟拍打點」或 Claude 對齊調整"
    if not lrc.strip():
        raise UserError("讀不到歌詞內容，請確認檔案格式（LRC、SRT 或 TXT）")
    if cfg.get("convert_to_taiwan", True) and re.search(r"[\u4e00-\u9fff]", lrc):
        lrc = to_taiwan(lrc)
    (d / "lyrics.lrc").write_text(lrc, "utf-8")
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
    if synced and (diff is None or diff <= 3):
        lines = [l for l in parse_lrc(synced) if l["t"] is not None and l["text"]]
        how = "使用網站提供的時間軸"
    else:
        if not plain_lines:
            plain_lines = [l["text"] for l in parse_lrc(synced) if l["text"]]
        lines = align_to_recognized(plain_lines, d)
        how = "依辨識結果的時間自動對齊" + ("（網站的時間軸和這首歌長度不同）" if synced else "")
    lrc = to_lrc(lines)
    (d / "lyrics.lrc").write_text(lrc, "utf-8")
    meta = read_meta(d)
    meta["steps"]["lyrics"] = True
    meta["lyrics_source"] = "online"
    meta["lyrics_pending"] = False
    meta["lyrics_online"] = {"source": cand.get("source"), "track": cand.get("track"), "artist": cand.get("artist")}
    meta["notes"] = [x for x in meta.get("notes", []) if "歌詞辨識沒有完成" not in x]
    write_meta(d, meta)
    return lrc, how


# ---------------------------------------------------------------- 分類

CATS_PATH = LIB / "categories.json"


def load_categories() -> list[str]:
    try:
        return [c for c in json.loads(CATS_PATH.read_text("utf-8")) if isinstance(c, str)]
    except Exception:
        return []


def save_categories(cats: list[str]) -> None:
    LIB.mkdir(exist_ok=True)
    CATS_PATH.write_text(json.dumps(cats, ensure_ascii=False), "utf-8")
