"""雷虎練唱台：把錄音和合音、伴奏混成完整作品，匯出成音訊或直式影片（含旋律線條與歌詞）。

對齊方式和評分相同：錄音第 τ 秒對應到歌曲時間 start + offset + τ × speed；
用的合音、伴奏是錄音當時的 KEY／速度版本，所以聽起來和你錄的時候一樣。
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf

import pipeline as P

W, H, FPS = 1080, 1920, 30          # 直式 9:16
BRAND = "雷虎練唱工具"                # 分享出去的影片、成績卡上顯示的名稱
REVERB = {"none": 0.0, "light": 0.12, "medium": 0.22, "hall": 0.32}


def _db(v) -> float:
    """dB → 倍率；-40 dB 以下視為靜音。"""
    v = float(v)
    return 0.0 if v <= -40 else 10 ** (v / 20)


def _resample(y: np.ndarray, sr: int, to: int) -> np.ndarray:
    if sr == to:
        return y
    from math import gcd
    from scipy.signal import resample_poly
    g = gcd(sr, to)
    return resample_poly(y, to // g, sr // g, axis=-1).astype(np.float32)


def load_take(d: Path, tid: str) -> dict:
    info = json.loads((d / "takes" / f"{tid}.json").read_text("utf-8"))
    if info.get("status") != "done" or info.get("acc") is None:
        raise P.UserError("這次錄音還沒有評分完成，或沒有偵測到歌聲，無法匯出")
    return info


def variant_needed(d: Path, info: dict) -> tuple[int, float]:
    return int(info.get("key", 0)), float(info.get("speed", 1.0))


def polish_vocal(v: np.ndarray, sr: int, reverb: str) -> np.ndarray:
    """人聲美化：去掉低頻雜音、輕壓縮讓音量平均，再加一點殘響。"""
    try:
        from pedalboard import Compressor, HighpassFilter, Pedalboard, Reverb
    except ImportError:
        return v
    # 先把音量對齊（峰值約 -6 dB），壓縮器才不會因為錄音音量大小不同而壓太多或沒作用；
    # 輕度壓縮（2.5:1、10 ms 起音）保留歌聲的動態和咬字，不會聽起來扁扁的
    pk = float(np.percentile(np.abs(v), 99.9)) or 1.0
    v = v * (0.5 / pk)
    fx = [HighpassFilter(cutoff_frequency_hz=80), Compressor(threshold_db=-16, ratio=2.5, attack_ms=10, release_ms=150)]
    wet = REVERB.get(reverb, 0.12)
    if wet > 0:
        fx.append(Reverb(room_size=0.45 if reverb != "hall" else 0.7, damping=0.5, wet_level=wet, dry_level=1.0, width=0.9))
    return Pedalboard(fx)(v.astype(np.float32), sr)


def _stems(d: Path, info: dict):
    key, speed = variant_needed(d, info)
    if not P.variant_ready(d, key, speed):
        raise P.UserError("需要先準備錄音時的 KEY／速度版本")
    vd = P.variant_dir(d, key, speed)
    back, sr = sf.read(str(vd / "back.wav"), dtype="float32", always_2d=True)
    inst, _ = sf.read(str(vd / "inst.wav"), dtype="float32", always_2d=True)
    back, inst = back.T, inst.T
    if back.shape[0] == 1:
        back, inst = np.repeat(back, 2, 0), np.repeat(inst, 2, 0)
    return back, inst, sr, key, speed


def vocal_track(d: Path, tid: str, opts: dict, sr: int, length: int) -> np.ndarray:
    """把錄音處理好（美化、殘響、標準化音量），放到伴奏時間軸上錄音的位置，回傳和伴奏一樣長的單聲道人聲。"""
    info = load_take(d, tid)
    speed = float(info.get("speed", 1.0))
    voc, vsr = sf.read(str(d / "takes" / f"{tid}.wav"), dtype="float32", always_2d=True)
    voc = _resample(voc.mean(axis=1), vsr, sr)
    v = polish_vocal(voc[None, :], sr, opts.get("reverb", "light"))[0] if opts.get("polish", True) else voc
    peak = float(np.percentile(np.abs(v), 99.9)) or 1.0
    v = v * (0.5 / peak)
    i0 = int(round((float(info.get("start", 0)) + float(info.get("offset", 0))) / speed * sr))
    out = np.zeros(length, dtype=np.float32)
    a, b = max(0, i0), min(length, i0 + len(v))
    if b > a:
        out[a:b] = v[a - i0:b - i0]
    return out


def suggest_balance(d: Path, tid: str, opts: dict | None = None) -> dict:
    """自動平衡：讓歌聲比伴奏大約大 3 dB（流行歌常見的比例），合音比伴奏小一點。

    只看你有唱的段落來比較響度，回傳建議的 vocal_db／back_db／inst_db。"""
    info = load_take(d, tid)
    back, inst, sr, _, _ = _stems(d, info)
    vocal = vocal_track(d, tid, opts or {"polish": True, "reverb": "light"}, sr, back.shape[1])
    hop = sr // 20
    n = len(vocal) // hop * hop

    def rms(x):   # 每 50 ms 的響度
        return np.sqrt(np.mean(x[:n].reshape(-1, hop) ** 2, axis=1) + 1e-12)
    rv = rms(vocal)
    sing = rv > max(rv.max() * 0.05, 1e-4)
    if sing.sum() < 20:
        return {"vocal_db": 0.0, "back_db": -3.0, "inst_db": 0.0}
    back_db = -3.0
    bed = back.mean(axis=0) * 10 ** (back_db / 20) + inst.mean(axis=0)
    lv = 20 * np.log10(np.median(rv[sing]))
    lb = 20 * np.log10(np.median(rms(bed)[sing]))
    vocal_db = (lb + 3.0) - lv
    inst_db = 0.0
    if vocal_db > 12:            # 歌聲已經拉到最大還不夠：改成把伴奏調小
        inst_db, back_db = max(-12.0, 12 - vocal_db), max(-15.0, back_db + 12 - vocal_db)
        vocal_db = 12.0
    elif vocal_db < -12:
        vocal_db = -12.0
    r = lambda x: round(float(x) * 2) / 2   # 對齊拉桿的 0.5 dB 刻度
    return {"vocal_db": r(vocal_db), "back_db": r(back_db), "inst_db": r(inst_db)}


EXPORT_LUFS = -14.0       # 匯出成品的響度
BED_TARGET_LUFS = -17.0   # 合音＋伴奏調到這個響度，歌聲再大約 3 dB，整體約 -14 LUFS（和串流平台差不多）


def master_db(d: Path, key: int, speed: float) -> float:
    """整體音量：依這個 KEY／速度版本的合音＋伴奏響度，算出要加減幾 dB。

    原曲很大聲（例如 -7 LUFS）時先整體調小，再加歌聲就不會被限幅器硬壓（聲音扁、會抽動）；
    原曲很小聲時整體調大，匯出來才不會比練唱時小聲、單薄。結果存在版本資料夾，只算一次。"""
    vd = P.variant_dir(d, key, speed)
    cache = vd / "loudness.json"
    try:
        return float(json.loads(cache.read_text("utf-8"))["master_db"])
    except Exception:
        pass
    back, sr = sf.read(str(vd / "back.wav"), dtype="float32", always_2d=True)
    inst, _ = sf.read(str(vd / "inst.wav"), dtype="float32", always_2d=True)
    n = min(len(back), len(inst))
    bed = (back[:n] * _db(-3) + inst[:n]).T
    lufs = P.integrated_loudness(bed, sr)
    m = 0.0 if lufs is None else float(np.clip(BED_TARGET_LUFS - lufs, -20.0, 12.0))
    try:
        cache.write_text(json.dumps({"bed_lufs": lufs, "master_db": round(m, 2)}), "utf-8")
    except OSError:
        pass
    return round(m, 2)


def peak_limit(x: np.ndarray, sr: int, ceiling_db: float = -1.0) -> np.ndarray:
    """透明的峰值限幅：只在峰值超過上限的瞬間把音量壓下來，其他地方完全不動。

    1.4.9 以前用的 pedalboard Limiter（JUCE）其實是兩段壓縮器：-10 dB 以上就 4:1 壓縮、
    還會自動加大音量，整首歌被壓得扁扁的，這是匯出音質變差的主要原因。"""
    from scipy.ndimage import minimum_filter1d, uniform_filter1d
    x = np.atleast_2d(np.asarray(x, dtype=np.float32))
    ceil = 10 ** (ceiling_db / 20)
    amp = np.abs(x).max(axis=0)
    need = np.minimum(1.0, ceil / np.maximum(amp, 1e-9))          # 每個取樣需要的音量倍率
    if need.min() >= 1.0:
        return x
    g = need
    for w in (int(0.005 * sr), int(0.06 * sr)):                     # 5 ms 前瞻、60 ms 釋放，平滑不會失真
        w = max(1, w)
        g = minimum_filter1d(g, size=2 * w + 1, mode="nearest")
        g = uniform_filter1d(g, size=w, mode="nearest")
    g = np.minimum(g, need)                                          # 保證不超過上限
    return (x * g[None, :]).astype(np.float32)


def mix_take(d: Path, tid: str, opts: dict) -> tuple[np.ndarray, int, dict]:
    """回傳 (立體聲混音, 取樣率, 時間資訊)。opts["range"]："full" 整首歌（預設）／"take" 只有錄音的段落。"""
    info = load_take(d, tid)
    back, inst, sr, key, speed = _stems(d, info)
    total = back.shape[1]
    vocal = vocal_track(d, tid, opts, sr, total) * _db(opts.get("vocal_db", 0))
    out = back * _db(opts.get("back_db", 0)) + inst * _db(opts.get("inst_db", 0)) + vocal[None, :]

    if opts.get("range") == "take":
        voc_len = sf.info(str(d / "takes" / f"{tid}.wav")).duration
        v0 = (float(info.get("start", 0)) + float(info.get("offset", 0))) / speed
        a, b = max(0, int(v0 * sr)), min(total, int((v0 + voc_len) * sr))
        out, start = out[:, a:b], a / sr
    else:
        start = 0.0
    # 整體音量：把混好的成品調到 -14 LUFS（和串流平台差不多），再確認峰值，
    # 讓限幅器只需要壓住少數瞬間的峰值，不會整首被壓扁（原曲很大聲時以前會出現聲音扁、抽動）
    lufs = P.integrated_loudness(out, sr)
    g = 10 ** ((EXPORT_LUFS - lufs) / 20) if lufs is not None else 1.0
    pk = float(np.percentile(np.abs(out), 99.99)) * g if out.size else 0.0
    if pk > 1.41:   # 峰值會超過 +3 dB 太多：整體少加一點
        g *= 1.41 / pk
    out = out * g
    n = out.shape[1]
    fi, fo = int(0.05 * sr), int(min(1.5, n / sr / 4) * sr)
    out[:, :fi] *= np.linspace(0, 1, fi, dtype=np.float32)
    out[:, n - fo:] *= np.linspace(1, 0, fo, dtype=np.float32)
    out = peak_limit(out, sr)
    timing = {"song_t0": start * speed, "speed": speed, "key": key, "dur": n / sr}
    return out.astype(np.float32), sr, timing


def _ffmpeg() -> str:
    exe = P.ffmpeg_path()
    if not exe:
        raise P.UserError("找不到 ffmpeg，無法匯出 MP3／影片。請執行「修復與更新」")
    return exe


def _run(args: list[str], **kw) -> None:
    flags = 0x08000000 if sys.platform == "win32" else 0   # 不要跳出黑色視窗
    r = subprocess.run(args, capture_output=True, creationflags=flags, **kw)
    if r.returncode:
        raise P.UserError("ffmpeg 轉檔失敗：" + r.stderr.decode("utf-8", "replace")[-300:])


def export_audio(y: np.ndarray, sr: int, path: Path) -> Path:
    if path.suffix == ".wav":
        P.write_wav(path, y, sr)
        return path
    tmp = path.with_suffix(".tmp.wav")
    P.write_wav(tmp, y, sr)
    try:
        _run([_ffmpeg(), "-y", "-loglevel", "error", "-i", str(tmp), "-c:a", "libmp3lame", "-b:a", "256k", str(path)])
    finally:
        tmp.unlink(missing_ok=True)
    return path


# ---------------------------------------------------------------- 影片

def _font(size: int, bold: bool = False):
    from PIL import ImageFont
    win = Path(r"C:\Windows\Fonts")
    names = (["msjhbd.ttc", "msjh.ttc"] if bold else ["msjh.ttc", "msjhbd.ttc"]) + [
        "/usr/share/fonts/opentype/noto/NotoSansCJK-%s.ttc" % ("Bold" if bold else "Regular"),
        "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc", "/System/Library/Fonts/PingFang.ttc"]
    for n in names:
        p = Path(n) if "/" in n else win / n
        if p.exists():
            try:
                return ImageFont.truetype(str(p), size, index=0)
            except OSError:
                continue
    return ImageFont.load_default()


def _hex(c: str, a: int = 255) -> tuple:
    c = c.lstrip("#")
    return tuple(int(c[i:i + 2], 16) for i in (0, 2, 4)) + (a,)


class VideoPainter:
    """畫直式影片的每一格：上方標題、中間旋律卷軸（原唱音符＋你的音高線）、下方歌詞。"""

    BG, GRID, MEL, VOICE, BAD, INK, MUTED = "#0f1216", "#232a33", "#f2a33a", "#4da3ff", "#ef5b5b", "#f2f4f7", "#8b95a3"

    def __init__(self, d: Path, info: dict, timing: dict, title: str):
        from PIL import Image, ImageDraw
        self.Image, self.ImageDraw = Image, ImageDraw
        self.key = timing["key"]
        self.t0, self.speed, self.dur = timing["song_t0"], timing["speed"], timing["dur"]
        ref, notes = P.load_pitch(d)
        self.ref, self.hop = ref + self.key, P.HOP_S
        self.notes = [(s, e, m + self.key) for s, e, m in notes]
        p = info["pitch"]
        self.mine = np.array([np.nan if v is None else v for v in p["midi"]], dtype=np.float32)
        self.mine_t0, self.mine_hop = float(p["t0"]), float(p["hop"])
        self.octave = int(info.get("octave") or 0)
        self.lines = P.lyric_lines(d)
        span = [n for n in self.notes if n[1] > self.t0 - 2 and n[0] < self.t0 + self.dur * self.speed + 2]
        ms = [n[2] for n in span] or [60]
        self.lo, self.hi = min(ms) - 3, max(ms) + 3
        if self.hi - self.lo < 14:
            mid = (self.hi + self.lo) / 2
            self.lo, self.hi = mid - 7, mid + 7
        self.f_title, self.f_sub = _font(56, True), _font(34)
        self.f_now, self.f_next, self.f_note = _font(66, True), _font(42), _font(24)
        self.title = title
        self.acc = info.get("acc")
        self.roll = (90, 360, W - 90, 1380)      # 卷軸範圍
        self.window = 6.0                         # 卷軸一次顯示幾秒（歌曲時間）
        self.bg = self._background()

    def _background(self):
        img = self.Image.new("RGB", (W, H), _hex(self.BG)[:3])   # RGB 底圖，半透明的筆畫才會混色
        dr = self.ImageDraw.Draw(img, "RGBA")
        # 上方淡淡的琥珀色漸層
        for y in range(0, 640, 4):
            a = int(70 * (1 - y / 640) ** 2)
            dr.rectangle((0, y, W, y + 4), fill=_hex("#f2a33a", a // 3))
        x0, y0, x1, y1 = self.roll
        dr.rounded_rectangle((x0 - 20, y0 - 20, x1 + 20, y1 + 20), 28, fill=_hex("#151a20"), outline=_hex(self.GRID))
        for m in range(int(np.ceil(self.lo)), int(self.hi) + 1):
            y = self._y(m)
            dr.line((x0, y, x1, y), fill=_hex(self.GRID, 255 if m % 12 == 0 else 120), width=1)
            if m % 12 == 0:
                dr.text((x0 - 10, y - 14), f"C{m // 12 - 1}", font=_font(22), fill=_hex(self.MUTED), anchor="rt")
        tw = self.f_title.getlength(self.title)
        title = self.title
        while tw > W - 160 and len(title) > 4:
            title = title[:-2]
            tw = self.f_title.getlength(title + "…")
        if title != self.title:
            title += "…"
        dr.text((W / 2, 120), title, font=self.f_title, fill=_hex(self.INK), anchor="mm")
        sub = f"KEY {self.key:+d}" if self.key else "原調"
        if abs(self.speed - 1) > 1e-3:
            sub += f"・{int(round(self.speed * 100))}% 速度"
        if self.acc is not None:
            sub += f"・音準命中 {round(self.acc * 100)}%"
        dr.text((W / 2, 200), sub, font=self.f_sub, fill=_hex(self.MUTED), anchor="mm")
        dr.text((W / 2, H - 70), BRAND, font=_font(28), fill=_hex(self.MUTED, 200), anchor="mm")
        return img

    def _y(self, m: float) -> float:
        x0, y0, x1, y1 = self.roll
        return y1 - (m - self.lo) / (self.hi - self.lo) * (y1 - y0)

    def _x(self, t: float, now: float) -> float:
        x0, _, x1, _ = self.roll
        return x0 + (x1 - x0) * (0.35 + (t - now) / self.window)

    def _ref_at(self, t: float) -> float:
        i = int(round(t / self.hop))
        return float(self.ref[i]) if 0 <= i < len(self.ref) else np.nan

    def frame(self, vt: float):
        """vt：影片第幾秒。"""
        now = self.t0 + vt * self.speed          # 歌曲時間
        img = self.bg.copy()
        dr = self.ImageDraw.Draw(img, "RGBA")
        x0, y0, x1, y1 = self.roll
        lo_t, hi_t = now - self.window * 0.35, now + self.window * 0.65
        rh = (y1 - y0) / (self.hi - self.lo)
        for s, e, m in self.notes:
            if e < lo_t or s > hi_t:
                continue
            xa, xb = max(x0, self._x(s, now)), min(x1, self._x(e, now))
            if xb <= xa:
                continue
            y = self._y(m)
            active = s <= now < e
            if active:   # 正在唱的音：亮一點，外面加一圈光
                dr.rounded_rectangle((xa - 6, y - rh * 0.42 - 6, xb + 6, y + rh * 0.42 + 6), 10, fill=_hex(self.MEL, 60))
            dr.rounded_rectangle((xa, y - rh * 0.42, xb, y + rh * 0.42), 6,
                                 fill=_hex(self.MEL, 245 if active else 95 if s > now else 150))
        # 你的音高線（只畫到現在；八度差換算到原唱的高度）
        a = max(0, int((lo_t - self.mine_t0) / self.mine_hop))
        b = min(len(self.mine), int((now - self.mine_t0) / self.mine_hop) + 1)
        prev = None
        for i in range(a, b):
            m = self.mine[i]
            if np.isnan(m):
                prev = None
                continue
            t = self.mine_t0 + i * self.mine_hop
            shown = m - 12 * self.octave
            r = self._ref_at(t)
            if not np.isnan(r):
                shown = r + ((m - r + 6) % 12 - 6)
            pt = (self._x(t, now), self._y(shown))
            if prev and abs(shown - prev[2]) <= 3:
                ok = np.isnan(r) or abs((m - r + 6) % 12 - 6) <= 0.5
                dr.line((prev[0], prev[1], pt[0], pt[1]), fill=_hex(self.VOICE if ok else self.BAD), width=7)
            prev = (pt[0], pt[1], shown)
        if prev:
            dr.ellipse((prev[0] - 11, prev[1] - 11, prev[0] + 11, prev[1] + 11), fill=_hex(self.VOICE))
        px = self._x(now, now)
        dr.line((px, y0 - 10, px, y1 + 10), fill=_hex(self.INK, 150), width=3)
        # 歌詞：正在唱的（依時間上色）＋下一句
        cur, nxt = None, None
        for i, l in enumerate(self.lines):
            if l["t"] <= now + 0.05:
                cur, nxt = l, (self.lines[i + 1] if i + 1 < len(self.lines) else None)
        if cur is None and self.lines and self.lines[0]["t"] - now < 4:
            nxt = self.lines[0]
        if cur:
            end = nxt["t"] if nxt else cur["t"] + 5
            frac = min(1.0, max(0.0, (now - cur["t"]) / max(0.5, end - cur["t"]) * 1.15))
            self._lyric(img, cur["text"], 1560, frac)
        if nxt:
            dr.text((W / 2, 1680), self._fit(nxt["text"], self.f_next), font=self.f_next, fill=_hex(self.MUTED), anchor="mm")
        return img

    def _fit(self, text: str, font) -> str:
        while font.getlength(text) > W - 140 and len(text) > 2:
            text = text[:-2] + "…"
        return text

    def _lyric(self, img, text: str, y: int, frac: float) -> None:
        text = self._fit(text, self.f_now)
        dr = self.ImageDraw.Draw(img, "RGBA")
        dr.text((W / 2, y), text, font=self.f_now, fill=_hex(self.INK), anchor="mm")
        if frac > 0:   # 唱過的部分換成琥珀色
            tw = self.f_now.getlength(text)
            x0 = int(W / 2 - tw / 2)
            layer = self.Image.new("RGBA", (W, 120), (0, 0, 0, 0))
            self.ImageDraw.Draw(layer).text((W / 2, 60), text, font=self.f_now, fill=_hex(self.MEL), anchor="mm")
            cut = x0 + int(tw * frac)
            layer = layer.crop((0, 0, max(1, cut), 120))
            img.paste(layer, (0, y - 60), layer)


def export_video(d: Path, info: dict, timing: dict, title: str, audio: Path, out: Path, report) -> Path:
    painter = VideoPainter(d, info, timing, title)
    total = int(timing["dur"] * FPS)
    flags = 0x08000000 if sys.platform == "win32" else 0
    cmd = [_ffmpeg(), "-y", "-loglevel", "error",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
           "-i", str(audio), "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-pix_fmt", "yuv420p",
           "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", str(out)]
    proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE, creationflags=flags)
    t_last = 0.0
    try:
        for k in range(total):
            proc.stdin.write(painter.frame(k / FPS).tobytes())
            if time.time() - t_last > 0.5:
                report(f"製作影片：{k / FPS:.0f}/{timing['dur']:.0f} 秒", 0.2 + 0.78 * k / max(1, total))
                t_last = time.time()
        proc.stdin.close()
    except BrokenPipeError:
        pass
    err = proc.stderr.read().decode("utf-8", "replace")
    if proc.wait():
        raise P.UserError("影片轉檔失敗：" + err[-300:])
    return out



# ---------------------------------------------------------------- 成績卡圖片

def export_scorecard(d: Path, info: dict, title: str, out: Path) -> Path:
    """成績卡（1080×1350，IG／FB／X／Threads 貼文都適合的 4:5）：分數、旋律對照、最需要練的句子。"""
    from PIL import Image, ImageDraw
    CW, CH = 1080, 1350
    BG, GRID, MEL, VOICE, BAD, INK, MUTED = (VideoPainter.BG, VideoPainter.GRID, VideoPainter.MEL, VideoPainter.VOICE,
                                             VideoPainter.BAD, VideoPainter.INK, VideoPainter.MUTED)
    img = Image.new("RGB", (CW, CH), _hex(BG)[:3])
    dr = ImageDraw.Draw(img, "RGBA")
    for y in range(0, 560, 4):
        dr.rectangle((0, y, CW, y + 4), fill=_hex(MEL, int(70 * (1 - y / 560) ** 2) // 3))
    key, speed = int(info.get("key", 0)), float(info.get("speed", 1))

    def fit(text, font, width):
        if font.getlength(text) <= width:
            return text
        while len(text) > 2 and font.getlength(text + "…") > width:
            text = text[:-1]
        return text + "…"

    f_title, f_sub = _font(52, True), _font(30)
    dr.text((CW / 2, 96), fit(title, f_title, CW - 140), font=f_title, fill=_hex(INK), anchor="mm")
    when = time.strftime("%Y/%m/%d", time.localtime(info.get("created", time.time())))
    sub = ("原調" if not key else f"KEY {key:+d}") + (f"・{int(round(speed * 100))}% 速度" if abs(speed - 1) > 1e-3 else "")
    sub += f"・{when}"
    dr.text((CW / 2, 160), sub, font=f_sub, fill=_hex(MUTED), anchor="mm")

    # 大分數：圓環＋百分比
    acc = float(info.get("acc") or 0)
    cx, cy, r = CW / 2, 410, 190
    dr.ellipse((cx - r, cy - r, cx + r, cy + r), outline=_hex(GRID), width=24)
    dr.arc((cx - r, cy - r, cx + r, cy + r), -90, -90 + 360 * acc, fill=_hex(MEL), width=24)
    big, pct, num = _font(140, True), _font(52, True), f"{round(acc * 100)}"
    wn, wp = big.getlength(num), pct.getlength("%")
    left = cx - (wn + 6 + wp) / 2            # 數字和 % 整組置中
    dr.text((left, cy + 30), num, font=big, fill=_hex(INK), anchor="ls")
    dr.text((left + wn + 6, cy + 30), "%", font=pct, fill=_hex(INK), anchor="ls")
    dr.text((cx, cy + 84), "音準命中率", font=_font(30), fill=_hex(MUTED), anchor="mm")

    # 三個小數字
    mean = info.get("mean") or 0
    stats = [(f"{'+' if mean > 0 else ''}{round(mean)}¢", "平均偏差" + ("（偏高）" if mean > 15 else "（偏低）" if mean < -15 else "")),
             (f"{int(info.get('duration') or 0) // 60}:{int(info.get('duration') or 0) % 60:02d}", "演唱長度"),
             (("低八度" if info.get("octave", 0) < 0 else "高八度") if info.get("octave") else "同八度", "音域")]
    for i, (v, label) in enumerate(stats):
        x = CW / 2 + (i - 1) * 300
        dr.text((x, 650), v, font=_font(48, True), fill=_hex(INK), anchor="mm")
        dr.text((x, 702), label, font=_font(26), fill=_hex(MUTED), anchor="mm")

    # 旋律對照：整段錄音的原唱音符與你的音高
    x0, y0, x1, y1 = 90, 770, CW - 90, 1040
    dr.rounded_rectangle((x0 - 20, y0 - 20, x1 + 20, y1 + 20), 24, fill=_hex("#151a20"), outline=_hex(GRID))
    ref, notes = P.load_pitch(d)
    p = info.get("pitch") or {}
    mine = np.array([np.nan if v is None else v for v in p.get("midi", [])], dtype=np.float32)
    t0, hop = float(p.get("t0", 0)), float(p.get("hop", P.HOP_S))
    ta, tb = t0, t0 + len(mine) * hop
    span = [(s, e, m + key) for s, e, m in notes if e > ta and s < tb]
    ms = [n[2] for n in span] or [60]
    lo, hi = min(ms) - 2, max(ms) + 2
    X = lambda t: x0 + (t - ta) / max(0.1, tb - ta) * (x1 - x0)
    Y = lambda m: y1 - (m - lo) / max(1, hi - lo) * (y1 - y0)
    for s, e, m in span:
        dr.rounded_rectangle((X(max(s, ta)), Y(m) - 5, max(X(max(s, ta)) + 2, X(min(e, tb))), Y(m) + 5), 3, fill=_hex(MEL, 170))
    octave = int(info.get("octave") or 0)
    prev = None
    for i, m in enumerate(mine):
        if np.isnan(m):
            prev = None
            continue
        t = t0 + i * hop
        j = int(round(t / P.HOP_S))
        rr = float(ref[j]) + key if 0 <= j < len(ref) and not np.isnan(ref[j]) else np.nan
        shown = rr + ((m - rr + 6) % 12 - 6) if not np.isnan(rr) else m - 12 * octave
        pt = (X(t), Y(shown))
        if prev and abs(shown - prev[2]) <= 3:
            ok = np.isnan(rr) or abs((m - rr + 6) % 12 - 6) <= 0.5
            dr.line((prev[0], prev[1], pt[0], pt[1]), fill=_hex(VOICE if ok else BAD), width=3)
        prev = (pt[0], pt[1], shown)
    dr.text((x0, y1 + 34), "■ 原唱旋律", font=_font(24), fill=_hex(MEL), anchor="lm")
    dr.text((x0 + 170, y1 + 34), "■ 我的音高", font=_font(24), fill=_hex(VOICE), anchor="lm")
    dr.text((x0 + 340, y1 + 34), "■ 偏離半音以上", font=_font(24), fill=_hex(BAD), anchor="lm")

    # 最需要練的句子
    worst = sorted(info.get("lines") or [], key=lambda l: l["acc"])[:2]
    y = 1130
    if worst:
        dr.text((x0, y), "還可以更好的句子", font=_font(28, True), fill=_hex(INK), anchor="lm")
        for l in worst:
            y += 50
            dr.text((x0, y), fit(l["text"], _font(28), CW - 360), font=_font(28), fill=_hex(MUTED), anchor="lm")
            dr.text((x1, y), f"{round(l['acc'] * 100)}%", font=_font(28, True), fill=_hex(MEL), anchor="rm")
    dr.text((CW / 2, CH - 50), BRAND, font=_font(28), fill=_hex(MUTED, 220), anchor="mm")
    img.save(out, "PNG", optimize=True)
    return out


def safe_name(s: str) -> str:
    s = re.sub(r'[\\/:*?"<>|\r\n]+', " ", s or "").strip()
    return (s[:60] or "我的演唱").rstrip(". ")


def export_take(d: Path, tid: str, opts: dict, out_base: Path, report) -> list[str]:
    """依 opts["formats"]（mp3／wav／mp4）匯出，回傳檔案路徑。"""
    info = load_take(d, tid)
    title = P.read_meta(d).get("title", "我的演唱")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    fmts = [f for f in opts.get("formats", ["mp3"]) if f in ("mp3", "wav", "mp4", "png")] or ["mp3"]
    paths = []
    if "png" in fmts:
        report("製作成績卡", 0.03)
        paths.append(str(export_scorecard(d, info, title, out_base.with_name(out_base.name + "_成績卡.png"))))
        if fmts == ["png"]:
            report("完成", 1.0)
            return paths
    report("混音中", 0.05)
    y, sr, timing = mix_take(d, tid, opts)
    for f in ("wav", "mp3"):
        if f in fmts:
            paths.append(str(export_audio(y, sr, out_base.with_suffix("." + f))))
    if "mp4" in fmts:
        report("準備影片", 0.15)
        tmp = out_base.with_suffix(".mix.wav")
        P.write_wav(tmp, y, sr)
        try:
            paths.append(str(export_video(d, info, timing, title, tmp, out_base.with_suffix(".mp4"), report)))
        finally:
            tmp.unlink(missing_ok=True)
    report("完成", 1.0)
    return paths
