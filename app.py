"""雷虎練唱台：本機伺服器。執行後用瀏覽器開 http://127.0.0.1:8765"""
from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import threading
import time
import uuid
import webbrowser
from pathlib import Path

from flask import Flask, abort, jsonify, request, send_file, send_from_directory

import claude_api as C
import pipeline as P
import share as SH

VERSION = "1.6.1"
HOST, PORT = "127.0.0.1", int(os.environ.get("LIANTAI_PORT", "8765"))
app = Flask(__name__, static_folder=str(P.ROOT / "static"), static_url_path="/static")
app.config["MAX_CONTENT_LENGTH"] = 1024 * 1024 * 1024
jobs = P.Jobs()
vjobs = P.Jobs()   # 移調／變速另外排隊：不會被分軌、評分卡住，也不會卡住它們


@app.errorhandler(P.UserError)
def user_error(e):
    return jsonify(error=str(e)), 400


@app.get("/")
def index():
    return send_from_directory(app.static_folder, "index.html")


# ---------------------------------------------------------------- 狀態與設定

@app.get("/api/status")
def status():
    cfg = P.load_config()
    mods = P.module_status()
    q = cfg.get("karaoke_quality", "auto")
    hq = P.hq_status() if (q == "high" or (q == "auto" and mods.get("cuda"))) else None
    return jsonify(version=VERSION, root=str(P.ROOT), modules=mods, cuda_note=P.CUDA_NOTE, hq=hq,
                   hq_model=P.HQ_KARAOKE,
                   claude_ready=bool(cfg.get("anthropic_api_key") or os.environ.get("ANTHROPIC_API_KEY")),
                   session_cost=C.summary()["session"]["cost_usd"],
                   jobs=jobs.active() + vjobs.active(), remote=dict(REMOTE), yt_pause=dict(P.YT_PAUSE))


@app.post("/api/youtube/resume")
def youtube_resume():
    """YouTube 限制解除後，繼續下載所有被暫停的歌。"""
    P.YT_PAUSE.update(on=False, reason="")
    n = 0
    for m in sorted(P.pending_downloads(), key=lambda m: m.get("created") or 0):
        if m["dl_state"] == "paused" and not _downloading(m["id"]):
            P.set_dl_state(P.song_dir(m["id"]), "queued")
            P.update_meta(P.song_dir(m["id"]), notes=[])
            yt_submit(m["id"], m["source_url"])
            n += 1
    return jsonify(resumed=n)


# 遠端開歌：本機的其他程式（例如點歌系統）請練唱畫面打開某首歌並切到指定 KEY。
# 練唱畫面輪詢 /api/status 時看到 seq 變了就照做。
REMOTE = {"seq": 0, "id": None, "key": 0}
_remote_lock = threading.Lock()


@app.post("/api/remote/open")
def remote_open():
    # 只給本機程式呼叫：瀏覽器裡的網頁送出的請求一定帶 Origin，一律拒絕，
    # 避免任何網站趁你開著練唱台時切換歌曲
    if request.headers.get("Origin"):
        abort(403)
    body = request.get_json(force=True) or {}
    sid = body.get("id") or ""
    P.song_dir(sid)
    try:
        key = int(body.get("key", 0))
    except (TypeError, ValueError):
        raise P.UserError("KEY 格式不正確")
    if not -6 <= key <= 6:
        raise P.UserError("KEY 範圍是 ±6")
    with _remote_lock:
        REMOTE.update(seq=REMOTE["seq"] + 1, id=sid, key=key)
        return jsonify(seq=REMOTE["seq"])


@app.get("/api/config")
def get_config():
    cfg = P.load_config()
    k = cfg.get("anthropic_api_key") or ""
    cfg["anthropic_api_key"] = (k[:7] + "…" + k[-4:]) if len(k) > 12 else ""
    cfg["has_key"] = bool(k or os.environ.get("ANTHROPIC_API_KEY"))
    return jsonify(cfg)


@app.post("/api/config")
def set_config():
    body = request.get_json(force=True) or {}
    cfg = P.load_config()
    for k in P.DEFAULT_CONFIG:
        if k not in body:
            continue
        if k == "balance_usd":
            v = body[k]
            v = None if v in (None, "") else round(float(v), 2)
            if v != cfg.get("balance_usd"):
                cfg["balance_usd"], cfg["balance_set_at"] = v, (time.time() if v is not None else None)
            continue
        if k == "balance_set_at":
            continue
        if k == "anthropic_api_key":
            v = (body[k] or "").strip()
            if v == "" or "…" in v:
                if body.get("clear_key"):
                    cfg[k] = ""
                continue
            cfg[k] = v
        else:
            cfg[k] = body[k]
    P.save_config(cfg)
    return get_config()


# ---------------------------------------------------------------- 歌曲庫

@app.get("/api/songs")
def songs():
    return jsonify(P.list_songs())


@app.get("/api/songs/search")
def songs_search():
    """歌曲庫搜尋：繁體、簡體互相找得到，另外列出歌名相近的歌。"""
    return jsonify(P.search_songs(request.args.get("q", "")[:200]))


@app.post("/api/songs")
def add_song():
    f = request.files.get("file")
    if not f or not f.filename:
        raise P.UserError("沒有收到檔案")
    P.LIB.mkdir(exist_ok=True)
    fd, name = tempfile.mkstemp(suffix=Path(f.filename).suffix, dir=P.LIB)
    os.close(fd)  # Windows 上檔案開著就無法搬移
    tmp = Path(name)
    f.save(tmp)
    sid = P.new_song(f.filename, tmp)
    cat = request.form.get("category") or None
    if cat in P.load_categories():
        P.update_meta(P.song_dir(sid), category=cat)
    steps = ["import", "separate", "pitch", "lyrics"]
    proc = P.process_song(sid, steps)

    def run(report, sid=sid):
        proc(report)
        after_process(sid)
    jid = jobs.submit("process", sid, run, label="all")
    return jsonify(id=sid, job=jid)


@app.post("/api/youtube")
def add_youtube():
    url = P.clean_youtube_url((request.get_json(force=True) or {}).get("url", ""))
    for m in P.list_songs():
        if m.get("source_url") != url:
            continue
        if m.get("steps", {}).get("import"):
            raise P.UserError(f"這首歌已經在歌曲庫裡：{m.get('title')}")
        if _downloading(m["id"]) or m.get("dl_state") == "queued":
            raise P.UserError(f"這首歌已經在下載佇列裡：{m.get('title')}")
        # 之前下載失敗或取消的同一首：直接用原本那一項重新下載，不另外多一項
        P.set_dl_state(P.song_dir(m["id"]), "queued")
        jid = yt_submit(m["id"], url)
        return jsonify(id=m["id"], job=jid, retried=True)
    sid = P.new_youtube_song(url)
    cat = (request.get_json(force=True) or {}).get("category") or None
    if cat in P.load_categories():
        P.update_meta(P.song_dir(sid), category=cat)
    jid = yt_submit(sid, url)
    return jsonify(id=sid, job=jid)


def prerender_key(sid: str) -> str | None:
    """這首歌有預設 KEY（例如表單的 -4）時，先把那個 KEY 的移調版本做好，打開就能直接唱。"""
    d = P.song_dir(sid)
    meta = P.read_meta(d)
    key = meta.get("key") or 0
    if not key or not meta.get("steps", {}).get("separate") or P.variant_ready(d, key, 1.0):
        return None
    label = f"{key}:1.0"
    return vjobs.find("variant", sid, label) or vjobs.submit(
        "variant", sid, lambda report: P.step_variant(d, key, 1.0, report), label=label)


def after_process(sid: str) -> None:
    """一首歌處理完：先做好預設 KEY 的移調版本；合音多的歌，依設定自動排一次精細處理（排在佇列最後）。"""
    meta = P.read_meta(P.song_dir(sid))
    if meta.get("fine_suggest") and not meta.get("fine_done") and P.load_config().get("auto_fine", True):
        queue_fine(sid)   # 精細處理完才做移調版本（重新分軌會清掉舊的移調版本）
    else:
        prerender_key(sid)


def queue_fine(sid: str) -> str:
    jid = jobs.find("process", sid, "fine")
    if jid:
        return jid
    P.update_meta(P.song_dir(sid), fine_queued=True)
    return jobs.submit("process", sid, P.fine_job(sid, after=prerender_key), label="fine")


def yt_submit(sid: str, url: str) -> str:
    return jobs.submit("process", sid, P.youtube_job(sid, url, after=after_process), label="youtube")


def prefetcher() -> None:
    """背景預先下載：佇列裡接下來的兩首 YouTube 歌，趁顯示卡在處理別首時先下載好。"""
    while True:
        time.sleep(3)
        try:
            if P.YT_PAUSE["on"]:
                continue
            for sid in jobs.upcoming("youtube", 2):
                P.prefetch_youtube(sid)
        except Exception as e:
            P.log_error("prefetcher", e)


def apply_sheet(sid: str, info: dict, batch_cat: str | None = None) -> None:
    """把表單的歌手、曲名、KEY、語言（＋分類）寫進歌曲資料。曲名有填就鎖住，下載完不會被 YouTube 標題蓋掉。"""
    d = P.song_dir(sid)
    meta = P.read_meta(d)
    ch: dict = {}
    if info.get("song"):
        ch.update(title=str(info["song"])[:120], title_locked=True)
    if info.get("artist"):
        ch["artist"] = str(info["artist"])[:80]
    if info.get("key") is not None:
        k = P.parse_key(info["key"])
        if k is not None and k != (meta.get("key") or 0):
            ch.update(key=k, key_set_at=time.time())
    if info.get("lang") in ("zh", "ja", "en", "ko", "yue"):
        ch["lyrics_language"] = info["lang"]
    cat = batch_cat or (info.get("category") if not meta.get("category") else None)
    if cat:
        cats = P.load_categories()
        if cat not in cats:
            P.save_categories(cats + [cat])
        ch["category"] = cat
    if ch:
        P.update_meta(d, **ch)
        if meta.get("dl_state"):   # 還沒下載完：歌名後面繼續標（排隊下載）等狀態
            P.set_dl_state(d, meta["dl_state"])


def _downloading(sid: str) -> bool:
    return any(j.get("song") == sid and j.get("status") in ("queued", "running") for j in jobs.active())


@app.post("/api/sheet-template")
def sheet_template():
    """批次加入的「下載範本」：把建議格式的歌單 .csv 存到「下載\雷虎練唱台」並在檔案總管裡顯示
    （視窗程式裡的網頁不能直接下載檔案）。Excel 開啟不會亂碼。"""
    out = P.downloads_dir() / P.APP_NAME / "歌單範本.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(P.ROOT / "static" / "song-list-template.csv", out)
    if sys.platform == "win32":
        import subprocess
        subprocess.Popen(["explorer", "/select,", str(out)])
    return jsonify(path=str(out))


@app.post("/api/youtube/resolve")
def youtube_resolve():
    """批次匯入：把多行輸入（影片網址、歌單網址、歌手＋歌名）解析成影片清單，讓使用者確認。"""
    body = request.get_json(force=True) or {}
    lines = body.get("lines") or []
    if not isinstance(lines, list):
        raise P.UserError("格式不正確")
    if isinstance(body.get("sheet"), list):   # 匯入的表單：每一列直接對應，不經過文字框
        return jsonify(items=P.resolve_lines([], body["sheet"]))
    return jsonify(items=P.resolve_lines([str(x) for x in lines]))


@app.post("/api/youtube/batch")
def youtube_batch():
    """把確認好的影片依順序加入下載佇列。"""
    body = request.get_json(force=True) or {}
    cat = body.get("category") or None
    if cat not in P.load_categories():
        cat = None
    songs = [m for m in P.list_songs() if m.get("source_url")]
    existing = {m["source_url"] for m in songs
                if m.get("steps", {}).get("import") or m.get("dl_state") == "queued" or _downloading(m["id"])}
    retry = {m["source_url"]: m["id"] for m in songs if m["source_url"] not in existing}   # 失敗／取消過的
    added, skipped = [], 0
    updated = 0
    for it in (body.get("items") or [])[:P.BATCH_MAX]:
        sheet = it.get("sheet") if isinstance(it.get("sheet"), dict) else None
        if it.get("update") and sheet:   # 歌曲庫已經有的歌：只更新資訊，不重新下載
            try:
                apply_sheet(str(it["update"]), sheet, cat)
                prerender_key(str(it["update"]))
                updated += 1
            except Exception as e:
                P.log_error("batch update", e)
            continue
        try:
            url = P.clean_youtube_url(it.get("url", ""))
        except P.UserError:
            skipped += 1
            continue
        if url in existing:
            skipped += 1
            continue
        existing.add(url)
        if url in retry:   # 之前下載失敗或取消的同一首：用原本那一項重新下載
            sid = retry[url]
            if sheet:
                apply_sheet(sid, sheet, cat)
            P.set_dl_state(P.song_dir(sid), "queued")
            yt_submit(sid, url)
            added.append(sid)
            continue
        sid = P.new_youtube_song(url, str((sheet or {}).get("song") or it.get("title") or "")[:100])
        if sheet:
            apply_sheet(sid, sheet, cat)
        elif cat:
            P.update_meta(P.song_dir(sid), category=cat)
        yt_submit(sid, url)
        added.append(sid)
    return jsonify(added=len(added), skipped=skipped, updated=updated)


@app.post("/api/youtube/sheet")
def youtube_sheet():
    f = request.files.get("file")
    if not f:
        raise P.UserError("沒有收到檔案")
    data = f.read()
    if len(data) > 10_000_000:
        raise P.UserError("檔案太大")
    rows = P.read_sheet(data, f.filename or "")
    return jsonify(lines=[r["line"] for r in rows if r.get("line")], rows=rows)


@app.post("/api/songs/<sid>/mixdown")
def mixdown(sid):
    """隱藏功能：依目前的 KEY、速度、各軌音量（含音量標準化）匯出混音，例如去掉主唱的伴奏。"""
    d = P.song_dir(sid)
    body = request.get_json(force=True) or {}
    key, speed = _key_speed()
    if not P.variant_ready(d, key, speed):
        raise P.UserError("這個 KEY／速度的版本還沒準備好，請先播放一次再匯出")
    import numpy as np
    import soundfile as sf
    vol = body.get("vol") or {}
    gains = {k: min(2.0, max(0.0, float(vol.get(k, 1.0) or 0))) for k in P.STEMS}
    if not any(gains.values()):
        raise P.UserError("所有音軌都是靜音，沒有東西可以匯出")
    fmt = "wav" if body.get("fmt") == "wav" else "mp3"
    meta = P.read_meta(d)
    norm = P.norm_gain_db(meta) if body.get("norm") else 0.0
    kname = f"KEY{key:+d}" if key else "原調"
    sname = f"_{int(round(speed * 100))}%" if abs(speed - 1) > 1e-3 else ""
    vname = "伴奏" if gains["lead"] == 0 else "混音"
    out = P.downloads_dir() / P.APP_NAME / f"{_safe_name(meta.get('title', '歌曲'))[:50]}_{vname}_{kname}{sname}.{fmt}"

    def run(report):
        report("混音中", 0.2)
        vd = P.variant_dir(d, key, speed)
        parts, sr = [], P.SR
        for k, g in gains.items():
            if g <= 0:
                continue
            y, sr = sf.read(str(P.stem_path(vd, k)), dtype="float32", always_2d=True)
            y = y.T if y.shape[1] == 2 else np.repeat(y.T, 2, 0)
            parts.append(y * g)
        n = min(p.shape[1] for p in parts)
        mix = sum(p[:, :n] for p in parts) * (10 ** (norm / 20))
        mix = SH.peak_limit(mix, sr)
        report("輸出檔案", 0.7)
        out.parent.mkdir(parents=True, exist_ok=True)
        SH.export_audio(mix.astype(np.float32), sr, out)
    jid = jobs.submit("export", sid, run, label="mixdown")
    return jsonify(job=jid, file=str(out))


@app.post("/api/jobs/<jid>/move")
def job_move(jid):
    """調整排隊順序（往前／往後一個）。"""
    delta = int((request.get_json(force=True) or {}).get("delta") or 0)
    if not jobs.move(jid, delta):
        raise P.UserError("無法移動（可能已經開始處理）")
    return jsonify(ok=True)


@app.get("/api/songs/<sid>")
def song(sid):
    d = P.song_dir(sid)
    meta = P.read_meta(d)
    if "acapella" not in meta and P.stem_path(d, "inst").exists() and P.stem_path(d, "lead").exists():
        meta = P.update_meta(d, acapella=P.is_acapella(d))   # 舊歌曲補上清唱判斷
    if meta.get("lufs") is None and meta.get("steps", {}).get("import"):
        meta = P.measure_loudness(d)   # 舊歌曲第一次開啟時補量響度（約 1 秒）
    meta["norm_gain_db"] = P.norm_gain_db(meta)
    meta["takes"] = P.list_takes(d)
    return jsonify(meta)


@app.patch("/api/songs/<sid>")
def rename(sid):
    d = P.song_dir(sid)
    body = request.get_json(force=True) or {}
    changes = {}
    if "title" in body:
        title = (body.get("title") or "").strip()
        if not title:
            raise P.UserError("歌名不能是空的")
        changes["title"] = title[:120]
    if "category" in body:
        cat = body.get("category") or None
        if cat and cat not in P.load_categories():
            raise P.UserError("找不到這個分類")
        changes["category"] = cat
    if "artist" in body:
        changes["artist"] = (body.get("artist") or "").strip()[:80]
    if "title" in body:
        changes["title_locked"] = True
    if "key" in body:   # 預設 KEY：打開這首歌時自動切到這個 KEY，並先做好移調版本
        k = P.parse_key(body.get("key"))
        changes.update(key=k or 0, key_set_at=time.time())
    if "lyrics_language" in body:   # 這首歌的辨識語言；null＝依設定
        v = body.get("lyrics_language")
        changes["lyrics_language"] = v if v in ("", "zh", "ja", "en", "ko", "yue") else None
    meta = P.update_meta(d, **changes)
    if "key" in changes:
        prerender_key(sid)
    return jsonify(meta)


@app.delete("/api/songs/<sid>")
def delete_song(sid):
    d = P.song_dir(sid)
    shutil.rmtree(d)
    return jsonify(ok=True)


@app.post("/api/songs/<sid>/process")
def reprocess(sid):
    P.song_dir(sid)
    steps = (request.get_json(force=True) or {}).get("steps") or ["separate", "pitch", "lyrics"]
    steps = [s for s in steps if s in ("import", "separate", "pitch", "lyrics")]
    label = ",".join(steps)
    jid = jobs.find("process", sid, label) or jobs.submit("process", sid, P.process_song(sid, steps), label=label)
    return jsonify(job=jid)


@app.get("/api/songs/<sid>/pitch")
def pitch(sid):
    d = P.song_dir(sid)
    p = d / "pitch.json"
    if not p.exists():
        raise P.UserError("這首歌還沒分析旋律")
    return send_file(p, mimetype="application/json", max_age=0)


@app.get("/api/songs/<sid>/lyrics")
def get_lyrics(sid):
    d = P.song_dir(sid)

    def rd(n):
        if not (d / n).exists():
            return ""
        t = (d / n).read_text("utf-8")
        if P.is_japanese(t):   # 1.5 以前把日文歌詞轉成了台灣繁體（来→來）：改回日文字形並存回去
            j = P.tw_to_ja(t)
            if j != t:
                (d / n).write_text(j, "utf-8")
                t = j
        return t
    meta = P.read_meta(d)
    return jsonify(lrc=rd("lyrics.lrc"), auto=rd("lyrics_auto.lrc"), online=rd("lyrics_online.lrc"),
                   source=meta.get("lyrics_source"), online_info=meta.get("lyrics_online"),
                   unreliable=not P.recognition_reliable(meta),
                   reasons=(meta.get("lyrics_quality") or {}).get("reasons", []),
                   language=meta.get("language"), song_language=meta.get("lyrics_language"),
                   kana_fix=meta.get("kana_fix") or {})


@app.post("/api/furigana")
def furigana():
    """日文歌詞的假名標音。字典還沒裝好時先在背景安裝，回傳安裝狀態。"""
    lines = [str(x)[:300] for x in ((request.get_json(force=True) or {}).get("lines") or [])][:500]
    if not P.kana_ready():
        P.start_kana_install()
        return jsonify(state=P.kana_status())
    out = {}
    for t in dict.fromkeys(lines):
        try:
            out[t] = P.furigana(t)
        except Exception as e:
            P.log_error("furigana", e)
            out[t] = None
    return jsonify(state={"state": "ready"}, ruby=out)


@app.post("/api/songs/<sid>/kana")
def set_kana(sid):
    """修改這首歌某個漢字詞的讀音（每首歌各自記住）；讀音空白＝恢復自動判斷。"""
    b = request.get_json(force=True) or {}
    base, reading = str(b.get("base") or "").strip()[:40], str(b.get("reading") or "").strip()[:80]
    if not base:
        raise P.UserError("缺少要修改的字")
    d = P.song_dir(sid)
    meta = P.read_meta(d)
    fix = meta.get("kana_fix") or {}
    if reading:
        fix[base] = reading
    else:
        fix.pop(base, None)
    meta["kana_fix"] = fix
    P.write_meta(d, meta)
    return jsonify(kana_fix=fix)


@app.post("/api/songs/<sid>/lyrics/use")
def use_lyrics(sid):
    """切換目前使用的歌詞：辨識結果或線上歌詞（兩份各自保存，切換不會互相覆蓋）。"""
    d = P.song_dir(sid)
    which = (request.get_json(force=True) or {}).get("which")
    src = {"auto": "lyrics_auto.lrc", "online": "lyrics_online.lrc"}.get(which)
    if not src or not (d / src).exists():
        raise P.UserError("沒有這份歌詞")
    lrc = (d / src).read_text("utf-8")
    (d / "lyrics.lrc").write_text(lrc, "utf-8")
    meta = P.read_meta(d)
    meta["lyrics_source"] = "online" if which == "online" else "whisper"
    meta["lyrics_pending"] = False
    meta["steps"]["lyrics"] = True
    P.write_meta(d, meta)
    return jsonify(lrc=lrc, source=meta["lyrics_source"])


@app.put("/api/songs/<sid>/lyrics")
def put_lyrics(sid):
    d = P.song_dir(sid)
    lrc = (request.get_json(force=True) or {}).get("lrc", "")
    (d / "lyrics.lrc").write_text(lrc, "utf-8")
    meta = P.read_meta(d)
    meta["steps"]["lyrics"] = bool(lrc.strip())
    if meta.get("lyrics_source") == "online" and lrc.strip():
        (d / "lyrics_online.lrc").write_text(lrc, "utf-8")   # 調整線上歌詞的時間：存回線上歌詞那一份
    else:
        meta["lyrics_source"] = "manual"
    if lrc.strip():
        meta["notes"] = [x for x in meta.get("notes", []) if "歌詞辨識沒有完成" not in x]
    P.write_meta(d, meta)
    return jsonify(ok=True)


# ---------------------------------------------------------------- 音訊與移調

def _key_speed():
    try:
        key = int(request.args.get("key", request.json.get("key", 0) if request.is_json else 0))
        speed = float(request.args.get("speed", request.json.get("speed", 1) if request.is_json else 1))
    except (TypeError, ValueError):
        raise P.UserError("KEY 或速度格式不正確")
    if not -6 <= key <= 6 or not 0.5 <= speed <= 1.5:
        raise P.UserError("KEY 範圍是 ±6，速度範圍是 50%–150%")
    return key, round(speed, 2)


@app.post("/api/songs/<sid>/variant")
def variant(sid):
    d = P.song_dir(sid)
    key, speed = _key_speed()
    if P.variant_ready(d, key, speed):
        return jsonify(ready=True)
    label = f"{key}:{speed}"
    vjobs.cancel_queued("variant", sid, label)   # 連按時，中間不要的 KEY 不用再算
    jid = vjobs.find("variant", sid, label) or vjobs.submit(
        "variant", sid, lambda report: P.step_variant(d, key, speed, report), label=label)
    return jsonify(ready=False, job=jid)


@app.post("/api/songs/<sid>/fine")
def song_fine(sid):
    """手動排一次精細處理（最精細的主唱分離＋重新分析旋律）。"""
    P.song_dir(sid)
    return jsonify(job=queue_fine(sid))


@app.get("/api/storage")
def storage():
    return jsonify(P.storage_report())


@app.post("/api/storage/compact")
def storage_compact():
    """把舊歌曲的 WAV 分軌轉成 FLAC（無損）、刪掉不需要的原始檔。排進處理佇列，和處理歌曲不會同時進行。"""
    j = jobs.find("process", "", "compact")
    if j:
        return jsonify(job=j)
    busy = lambda sid: any(x.get("song") == sid and x.get("status") in ("queued", "running") for x in vjobs.active())
    jid = jobs.submit("process", "", lambda report: P.compact_library(report, busy), label="compact")
    return jsonify(job=jid)


@app.get("/api/songs/<sid>/audio/<stem>")
def audio(sid, stem):
    d = P.song_dir(sid)
    if stem not in P.STEMS + ("mix",):
        abort(404)
    key, speed = _key_speed()
    base = d if stem == "mix" else P.variant_dir(d, key, speed)
    p = P.stem_path(base, stem)
    if not p.exists():
        abort(404)
    return send_file(p, mimetype="audio/flac" if p.suffix == ".flac" else "audio/wav", conditional=True, max_age=0)


@app.get("/api/debug/stacks")
def debug_stacks():
    """疑難排解用：列出每個執行緒目前停在哪一行。"""
    import traceback as tb
    names = {t.ident: t.name for t in threading.enumerate()}
    out = []
    for tid, frame in sys._current_frames().items():
        out.append(f"--- {names.get(tid, tid)}\n" + "".join(tb.format_stack(frame)[-12:]))
    return app.response_class("\n".join(out), mimetype="text/plain; charset=utf-8")


@app.post("/api/jobs/<jid>/cancel")
def job_cancel(jid):
    """停止處理（例如歌詞辨識卡很久）。已經完成的步驟會保留。"""
    q = jobs if jobs.get(jid) else vjobs
    if not q.cancel(jid):
        raise P.UserError("這個工作已經結束了")
    j = q.get(jid) or {}
    if j.get("label") == "youtube":   # 還在排隊的 YouTube 下載被取消：歌名標成「已取消下載」，之後可以重新下載
        try:
            d = P.song_dir(j["song"])
            if not P.read_meta(d).get("steps", {}).get("import"):
                P.set_dl_state(d, "cancelled")
        except Exception:
            pass
    return jsonify(ok=True)


@app.post("/api/songs/<sid>/redownload")
def redownload(sid):
    """下載失敗或取消的 YouTube 歌曲，重新排隊下載。"""
    d = P.song_dir(sid)
    meta = P.read_meta(d)
    url = meta.get("source_url")
    if not url or meta.get("steps", {}).get("import"):
        raise P.UserError("這首歌不需要重新下載")
    P.YT_PAUSE.update(on=False, reason="")   # 使用者手動重試：解除暫停
    if any(j.get("song") == sid and j.get("status") in ("queued", "running") for j in jobs.active()):
        raise P.UserError("這首歌已經在下載佇列裡")
    P.set_dl_state(d, "queued")
    jid = yt_submit(sid, url)
    return jsonify(job=jid)


def resume_downloads() -> None:
    """程式關掉時還在排隊的 YouTube 下載，重新開啟後自動接著排隊（失敗、取消的不會自動重試）。"""
    try:
        for m in sorted(P.pending_downloads(), key=lambda m: m.get("created") or 0):
            if m["dl_state"] == "queued":
                yt_submit(m["id"], m["source_url"])
    except Exception as e:
        P.log_error("resume_downloads", e)


@app.get("/api/jobs/<jid>")
def job(jid):
    q = jobs if jobs.get(jid) else vjobs
    j = q.get(jid)
    if not j:
        abort(404)
    j["ahead"] = q.position(jid)
    return jsonify(j)


# ---------------------------------------------------------------- 錄音

@app.post("/api/songs/<sid>/takes")
def add_take(sid):
    d = P.song_dir(sid)
    f = request.files.get("file")
    if not f:
        raise P.UserError("沒有收到錄音")
    td = d / "takes"
    td.mkdir(exist_ok=True)
    tid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    f.save(td / f"{tid}.wav")
    info = {
        "id": tid, "created": time.time(), "status": "scoring",
        "start": float(request.form.get("start", 0)),
        "key": int(request.form.get("key", 0)),
        "speed": float(request.form.get("speed", 1)),
        "duration": float(request.form.get("duration", 0)),
    }
    (td / f"{tid}.json").write_text(json.dumps(info, ensure_ascii=False), "utf-8")
    jid = jobs.submit("take", sid, lambda report: _score(d, tid, report), label=tid)
    return jsonify(id=tid, job=jid)


def _score(d, tid, report):
    """評分；失敗時把原因寫回錄音資料，畫面才不會一直停在「評分中」。"""
    try:
        P.score_take(d, tid, report)
    except Exception as e:
        j = d / "takes" / f"{tid}.json"
        try:
            info = json.loads(j.read_text("utf-8"))
            info.update(status="error", error=f"評分失敗：{e}")
            j.write_text(json.dumps(info, ensure_ascii=False), "utf-8")
        except Exception:
            pass
        raise


def _take_paths(sid, tid):
    d = P.song_dir(sid)
    if not all(c.isalnum() or c == "-" for c in tid):
        abort(404)
    return d, d / "takes" / f"{tid}.json", d / "takes" / f"{tid}.wav"


@app.get("/api/songs/<sid>/takes/<tid>")
def get_take(sid, tid):
    _, j, _ = _take_paths(sid, tid)
    if not j.exists():
        abort(404)
    return send_file(j, mimetype="application/json", max_age=0)


@app.post("/api/songs/<sid>/takes/<tid>/rescore")
def rescore_take(sid, tid):
    d, j, w = _take_paths(sid, tid)
    if not (j.exists() and w.exists()):
        abort(404)
    jid = jobs.find("take", sid, tid) or jobs.submit("take", sid, lambda report: _score(d, tid, report), label=tid)
    return jsonify(job=jid)


# ---------------------------------------------------------------- 匯出／分享錄音（1.4）

def _share_opts():
    b = request.get_json(force=True) or {}
    def num(k, lo, hi):
        try:
            return max(lo, min(hi, float(b.get(k, 0))))
        except (TypeError, ValueError):
            return 0.0
    return {"vocal_db": num("vocal_db", -40, 12), "back_db": num("back_db", -40, 6), "inst_db": num("inst_db", -40, 6),
            "reverb": b.get("reverb") if b.get("reverb") in SH.REVERB else "light", "polish": bool(b.get("polish", True)),
            "formats": [f for f in (b.get("formats") or ["mp3"]) if f in ("mp3", "wav", "mp4", "png")] or ["mp3"],
            "range": "take" if b.get("range") == "take" else "full"}


def _need_variant(d, sid, tid):
    """匯出要用錄音當時的 KEY／速度版本；還沒有就先排隊產生。"""
    info = SH.load_take(d, tid)
    key, speed = SH.variant_needed(d, info)
    if P.variant_ready(d, key, speed):
        return None
    label = f"{key}:{speed}"
    return vjobs.find("variant", sid, label) or vjobs.submit(
        "variant", sid, lambda report: P.step_variant(d, key, speed, report), label=label)


@app.post("/api/songs/<sid>/takes/<tid>/mixprep")
def take_mixprep(sid, tid):
    """即時試聽用：回傳處理好的人聲（和伴奏一樣長）、合音／伴奏的網址，以及自動平衡的建議音量。"""
    d, _, _ = _take_paths(sid, tid)
    need = _need_variant(d, sid, tid)
    if need:
        return jsonify(need_job=need)
    opts = _share_opts()
    info = SH.load_take(d, tid)
    key, speed = SH.variant_needed(d, info)
    back = P.stem_path(P.variant_dir(d, key, speed), "back")
    sr, n = sf_info(back)
    tag = f"v2_{int(opts['polish'])}_{opts['reverb']}"   # v2：1.4.9 改了人聲處理，舊的暫存要重做
    vp = d / "takes" / f"{tid}.vocal_{tag}.wav"
    if not vp.exists():
        for old in (d / "takes").glob(f"{tid}.vocal_*.wav"):
            old.unlink(missing_ok=True)
        P.write_wav(vp, SH.vocal_track(d, tid, opts, sr, n), sr)
    q = f"key={key}&speed={speed}"
    start = (float(info.get("start", 0)) + float(info.get("offset", 0))) / speed
    return jsonify(vocal=f"/api/songs/{sid}/takes/{tid}/vocal/{tag}?t={int(vp.stat().st_mtime)}",
                   back=f"/api/songs/{sid}/audio/back?{q}", inst=f"/api/songs/{sid}/audio/inst?{q}",
                   start=round(start, 3), key=key, speed=speed, master_db=SH.master_db(d, key, speed),
                   balance=SH.suggest_balance(d, tid, opts) if request.args.get("balance") else None)


def sf_info(path):
    import soundfile as sf
    i = sf.info(str(path))
    return i.samplerate, i.frames


@app.get("/api/songs/<sid>/takes/<tid>/vocal/<tag>")
def take_vocal(sid, tid, tag):
    d, _, _ = _take_paths(sid, tid)
    p = d / "takes" / f"{tid}.vocal_{tag}.wav"
    if not all(c.isalnum() or c == "_" for c in tag) or not p.exists():
        abort(404)
    return send_file(p, mimetype="audio/wav", max_age=0)


@app.post("/api/songs/<sid>/takes/delete")
def delete_takes(sid):
    """批次刪除錄音。"""
    ids = (request.get_json(force=True) or {}).get("ids") or []
    n = 0
    for tid in ids:
        _, j, w = _take_paths(sid, str(tid))
        for f in [j, w, *w.parent.glob(f"{tid}.*.wav")]:
            f.unlink(missing_ok=True)
        n += 1
    return jsonify(deleted=n)


@app.post("/api/songs/<sid>/takes/<tid>/preview")
def take_preview(sid, tid):
    d, j, w = _take_paths(sid, tid)
    need = _need_variant(d, sid, tid)
    if need:
        return jsonify(need_job=need)
    y, sr, _ = SH.mix_take(d, tid, _share_opts())
    P.write_wav(d / "takes" / f"{tid}.preview.wav", y, sr)
    return jsonify(url=f"/api/songs/{sid}/takes/{tid}/preview.wav?t={int(time.time() * 1000)}")


@app.get("/api/songs/<sid>/takes/<tid>/preview.wav")
def take_preview_file(sid, tid):
    d, _, _ = _take_paths(sid, tid)
    p = d / "takes" / f"{tid}.preview.wav"
    if not p.exists():
        abort(404)
    return send_file(p, mimetype="audio/wav", max_age=0)


@app.post("/api/songs/<sid>/takes/<tid>/export")
def take_export(sid, tid):
    d, _, _ = _take_paths(sid, tid)
    opts = _share_opts()
    if opts["formats"] != ["png"]:   # 只匯出成績卡不需要伴奏
        need = _need_variant(d, sid, tid)
        if need:
            return jsonify(need_job=need)
    SH.load_take(d, tid)
    title = P.read_meta(d).get("title", "我的演唱")
    base = P.downloads_dir() / P.APP_NAME / f"{_safe_name(title)[:50]}_我的演唱_{tid[:13]}"
    files = ([str(base.with_name(base.name + "_成績卡.png"))] if "png" in opts["formats"] else []) + \
        [str(base.with_suffix("." + f)) for f in ("wav", "mp3", "mp4") if f in opts["formats"]]
    jid = jobs.submit("export", sid, lambda report: SH.export_take(d, tid, opts, base, report), label=tid)
    return jsonify(job=jid, files=files)


@app.get("/api/songs/<sid>/takes/<tid>/audio")
def take_audio(sid, tid):
    _, _, w = _take_paths(sid, tid)
    if not w.exists():
        abort(404)
    return send_file(w, mimetype="audio/wav", conditional=True, max_age=0)


@app.delete("/api/songs/<sid>/takes/<tid>")
def delete_take(sid, tid):
    _, j, w = _take_paths(sid, tid)
    j.unlink(missing_ok=True)
    w.unlink(missing_ok=True)
    for f in w.parent.glob(f"{tid}.*.wav"):   # 試聽用的暫存人聲
        f.unlink(missing_ok=True)
    return jsonify(ok=True)


# ---------------------------------------------------------------- Claude（需要連線）

ALIGN_TOOL = {
    "name": "set_line_times",
    "description": "回傳每一行歌詞的開始秒數，順序和行數都必須與使用者提供的歌詞相同。",
    "input_schema": {
        "type": "object",
        "properties": {
            "times": {"type": "array", "items": {"type": "number"},
                      "description": "每行歌詞的開始秒數（遞增，保留兩位小數）"},
        },
        "required": ["times"],
    },
}


@app.post("/api/songs/<sid>/claude/align")
def claude_align(sid):
    """把使用者貼的正確歌詞，對到辨識出的時間與旋律段落上。"""
    d = P.song_dir(sid)
    ref = ((request.get_json(force=True) or {}).get("text") or "").strip()
    ref_lines = [l.strip() for l in ref.splitlines() if l.strip()]
    if not ref_lines:
        raise P.UserError("先貼上正確的歌詞")
    meta = P.read_meta(d)
    auto = d / "lyrics_auto.lrc"
    heard = P.parse_lrc(auto.read_text("utf-8")) if auto.exists() else []
    phrases = []
    if (d / "pitch.json").exists():
        _, notes = P.load_pitch(d)
        for s, e, m in notes:
            if phrases and s - phrases[-1][1] < 0.35:
                phrases[-1][1] = e
                phrases[-1][2] += 1
            else:
                phrases.append([s, e, 1])
    n = len(ref_lines)
    tool = json.loads(json.dumps(ALIGN_TOOL))
    tool["input_schema"]["properties"]["times"]["minItems"] = n
    tool["input_schema"]["properties"]["times"]["maxItems"] = n
    prompt = (
        "你在幫歌曲練唱工具對齊歌詞時間。\n"
        f"歌曲長度 {meta.get('duration') or 0:.1f} 秒。\n"
        "資料一：語音辨識從人聲音軌聽到的句子與開始秒數（可能有錯字、斷句不準、漏句）。\n"
        "資料二：從人聲旋律偵測到的樂句（開始秒、結束秒、音符數），一個音符大約對應一個中文字。\n"
        "資料三：使用者提供的正確歌詞，每行一句，順序就是演唱順序。\n\n"
        "請為資料三的每一行決定開始秒數，依據是內容與辨識結果的相似度，以及字數與樂句音符數的對應。"
        "秒數必須遞增。重複的段落（副歌）要對到各自出現的位置。"
        f"用 set_line_times 工具回傳，times 必須剛好 {n} 個數字。\n\n"
        "資料一：\n" + "\n".join(f"{l['t']:.2f}\t{l['text']}" for l in heard if l["t"] is not None)
        + "\n\n資料二：\n" + "\n".join(f"{s:.2f}\t{e:.2f}\t{k}" for s, e, k in phrases)
        + f"\n\n資料三（共 {n} 行）：\n" + "\n".join(f"{i + 1}. {t}" for i, t in enumerate(ref_lines))
    )
    data, usage = C.ask_tool(prompt, tool, 8000, "align", meta.get("title", ""))
    times = data.get("times") or []
    if len(times) != n:
        raise P.UserError(f"Claude 回傳了 {len(times)} 個時間，但歌詞有 {n} 行。再試一次。")
    dur = float(meta.get("duration") or 1e9)
    clean, prev = [], -1.0
    for t in times:   # 確保遞增、不超出歌曲長度
        t = min(max(float(t), prev + 0.3, 0.0), dur)
        clean.append(round(t, 2))
        prev = t
    lrc = P.to_lrc([{"t": t, "text": txt} for t, txt in zip(clean, ref_lines)])
    (d / "lyrics.lrc").write_text(lrc, "utf-8")
    meta["steps"]["lyrics"] = True
    meta["lyrics_source"] = "claude"
    P.write_meta(d, meta)
    return jsonify(lrc=lrc, usage=usage)


@app.post("/api/songs/<sid>/claude/coach")
def claude_coach(sid):
    d = P.song_dir(sid)
    tid = (request.get_json(force=True) or {}).get("take")
    _, j, _ = _take_paths(sid, tid or "")
    if not j.exists():
        raise P.UserError("找不到這次錄音")
    t = json.loads(j.read_text("utf-8"))
    if t.get("acc") is None:
        raise P.UserError("這次錄音還沒評分完成")
    title = P.read_meta(d).get("title", "")
    rows = "\n".join(f"{l['t']:.1f}s\t{l['text']}\t命中 {l['acc'] * 100:.0f}%\t平均偏差 {l['mean']:+.0f} 音分"
                     for l in t.get("lines", []))
    prompt = (
        f"你是聲樂老師。學生用練唱工具唱了〈{title}〉，以下是音準分析結果。\n"
        f"移調 {t['key']:+d} 半音，速度 {t['speed'] * 100:.0f}%，"
        f"{'比原唱低一個八度' if t.get('octave', 0) < 0 else '比原唱高一個八度' if t.get('octave', 0) > 0 else '與原唱同八度'}。\n"
        f"整體：音準命中率 {t['acc'] * 100:.0f}%（±50 音分內），平均偏差 {t['mean']:+.0f} 音分（正值偏高、負值偏低），"
        f"平均絕對偏差 {t['spread']:.0f} 音分。\n"
        "逐句：\n" + (rows or "（沒有歌詞時間，無逐句資料）") + "\n\n"
        "請用台灣繁體中文，寫 200 字以內的練習建議：先講整體傾向，再挑最需要練的 2–3 句，"
        "說明可能原因（例如起音偏低、高音撐不住、尾音下墜）和具體練法。不要用條列以外的格式，不要客套。"
    )
    text, usage = C.ask_text(prompt, 3000, "coach", title)
    t["coach"] = text
    j.write_text(json.dumps(t, ensure_ascii=False), "utf-8")
    return jsonify(text=text, usage=usage)


@app.get("/api/songs/<sid>/lyrics/search")
def lyrics_search(sid):
    """手動搜尋：LRCLIB＋網易雲＋酷狗（原文與簡體各搜一次），依和辨識結果的相似度排序。"""
    d = P.song_dir(sid)
    q = (request.args.get("q") or "").strip()
    if not q and not P.lyric_queries(P.read_meta(d)):
        raise P.UserError("請輸入歌名或歌手再搜尋")
    tried, results = P.find_candidates(d, P.load_config(), q)
    return jsonify(query=q or (tried[0] if tried else ""), queries=tried, results=results)


@app.get("/api/songs/<sid>/lyrics/candidates")
def lyrics_candidates(sid):
    d = P.song_dir(sid)
    p = d / "lyrics_candidates.json"
    meta = P.read_meta(d)
    if not p.exists() and meta.get("lyrics_source") in (None, "whisper") and (d / "lyrics_auto.lrc").exists():
        try:   # 1.1 之前匯入的歌：第一次開啟時補做線上歌詞比對
            P.find_and_store_candidates(d, P.load_config())
            meta = P.read_meta(d)
        except Exception:
            pass
    cands = json.loads(p.read_text("utf-8")) if p.exists() else []
    return jsonify(pending=bool(meta.get("lyrics_pending")), results=cands,
                   recognized=P.recognized_text(d)[:300])


@app.post("/api/songs/<sid>/lyrics/apply")
def lyrics_apply(sid):
    d = P.song_dir(sid)
    cand = (request.get_json(force=True) or {}).get("candidate") or {}
    if not (cand.get("synced") or cand.get("plain")):
        raise P.UserError("這筆歌詞沒有內容")
    lrc, how = P.apply_candidate(d, cand)
    return jsonify(lrc=lrc, how=how)


@app.post("/api/songs/<sid>/lyrics/import")
def lyrics_import(sid):
    d = P.song_dir(sid)
    body = request.get_json(force=True) or {}
    lrc, how = P.import_lyrics(d, body.get("text", ""), P.load_config(), body.get("name", ""))
    return jsonify(lrc=lrc, how=how)


@app.post("/api/songs/<sid>/lyrics/keep")
def lyrics_keep(sid):
    d = P.song_dir(sid)
    P.update_meta(d, lyrics_pending=False)
    return jsonify(ok=True)


# ---------------------------------------------------------------- 分類

@app.get("/api/categories")
def categories():
    return jsonify(P.load_categories())


@app.post("/api/categories")
def edit_categories():
    body = request.get_json(force=True) or {}
    act, name, new = body.get("action"), (body.get("name") or "").strip(), (body.get("new") or "").strip()
    cats = P.load_categories()
    if act == "add":
        if not name:
            raise P.UserError("分類名稱不能是空的")
        if name in cats:
            raise P.UserError("已經有這個分類")
        cats.append(name[:30])
    elif act == "rename":
        if name not in cats or not new:
            raise P.UserError("找不到這個分類")
        if new in cats:
            raise P.UserError("已經有這個分類")
        cats[cats.index(name)] = new[:30]
        for m in P.list_songs():
            if m.get("category") == name:
                P.update_meta(P.song_dir(m["id"]), category=new[:30])
    elif act == "delete":
        if name in cats:
            cats.remove(name)
        for m in P.list_songs():   # 歌曲不會刪掉，只是變成未分類
            if m.get("category") == name:
                P.update_meta(P.song_dir(m["id"]), category=None)
    elif act == "order":
        order = [c for c in body.get("order") or [] if c in cats]
        cats = order + [c for c in cats if c not in order]
    else:
        raise P.UserError("不支援的操作")
    P.save_categories(cats)
    return jsonify(cats)


@app.get("/api/usage")
def usage():
    return jsonify(C.summary())


# ---------------------------------------------------------------- 匯出歌詞

def _srt_time(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    return f"{ms // 3600000:02d}:{ms // 60000 % 60:02d}:{ms // 1000 % 60:02d},{ms % 1000:03d}"


def _safe_name(name: str) -> str:
    import re
    name = re.sub(r'[\\/:*?"<>|\r\n\t]+', " ", name).strip(" .") or "歌詞"
    return name[:80]


@app.post("/api/songs/<sid>/lyrics/export")
def export_lyrics(sid):
    d = P.song_dir(sid)
    body = request.get_json(force=True) or {}
    fmt = (body.get("fmt") or "lrc").lower()
    if fmt not in ("lrc", "txt", "srt"):
        raise P.UserError("格式只支援 LRC、TXT、SRT")
    text = body.get("lrc")
    if text is None:
        p = d / "lyrics.lrc"
        text = p.read_text("utf-8") if p.exists() else ""
    lines = P.parse_lrc(text)
    if not any(l["text"] for l in lines):
        raise P.UserError("這首歌還沒有歌詞可以匯出")
    meta = P.read_meta(d)
    if fmt == "lrc":
        head = f"[ti:{meta.get('title', '')}]\n[re:{P.APP_NAME} {VERSION}]\n"
        body = "\n".join(l for l in text.splitlines() if not __import__("re").match(r"^\[(ti|ar|al|re|by|offset|length):", l))
        out = head + body.strip() + "\n"   # 保留逐字時間標記
    elif fmt == "txt":
        out = "\n".join(l["text"] for l in lines) + "\n"
    else:
        timed = [l for l in lines if l["t"] is not None]
        if not timed:
            raise P.UserError("SRT 需要有時間的歌詞")
        dur = float(meta.get("duration") or 0)
        rows = []
        for i, l in enumerate(timed):
            end = timed[i + 1]["t"] if i + 1 < len(timed) else (min(l["t"] + 5, dur) if dur else l["t"] + 5)
            end = max(end - 0.05, l["t"] + 0.5)
            rows.append(f"{i + 1}\n{_srt_time(l['t'])} --> {_srt_time(end)}\n{l['text']}\n")
        out = "\n".join(rows)
    folder = P.downloads_dir() / P.APP_NAME
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{_safe_name(meta.get('title', '歌詞'))}.{fmt}"
    path.write_text(out, encoding="utf-8-sig" if fmt in ("txt", "srt") else "utf-8")
    return jsonify(path=str(path))


EXTERNAL_URLS = ("https://www.twitch.tv/leihustore", "https://github.com/LeihuTW/Leihu-singsing",
                 # 分享錄音：各平台的發文／上傳頁
                 "https://www.facebook.com", "https://www.instagram.com", "https://x.com/compose/post",
                 "https://www.threads.com")


# 1.4.3：直播歌詞（去背）。練唱畫面把目前狀態送來，/overlay 頁面再來讀
OVERLAY = {"sid": None, "title": "", "t": 0.0, "playing": False, "rate": 1.0, "color": True,
           "lines": [], "ver": 0, "at": 0.0}
_ov_lock = threading.Lock()


@app.get("/overlay")
def overlay_page():
    return send_from_directory(app.static_folder, "overlay.html", max_age=0)


@app.post("/api/overlay")
def overlay_set():
    d = request.get_json(force=True) or {}
    with _ov_lock:
        if d.get("sid") != OVERLAY["sid"] and "lines" not in d:
            OVERLAY["lines"] = []
        if "lines" in d:
            OVERLAY["lines"] = d["lines"] if isinstance(d["lines"], list) else []
        if "lines" in d or d.get("sid") != OVERLAY["sid"]:
            OVERLAY["ver"] += 1
        OVERLAY.update(sid=d.get("sid"), title=str(d.get("title") or ""), t=float(d.get("t") or 0),
                       playing=bool(d.get("playing")), rate=float(d.get("rate") or 1),
                       color=bool(d.get("color", True)), at=time.time())
    return jsonify(ok=True)


@app.get("/api/overlay")
def overlay_get():
    with _ov_lock:
        o = dict(OVERLAY)
    t = o["t"] + ((time.time() - o["at"]) * o["rate"] if o["playing"] else 0)
    out = {"sid": o["sid"], "title": o["title"], "t": t, "playing": o["playing"], "rate": o["rate"],
           "color": o["color"], "ver": o["ver"]}
    if request.args.get("ver") != str(o["ver"]):
        out["lines"] = o["lines"]
    return jsonify(out)


@app.post("/api/open-overlay")
def open_overlay():
    """用預設瀏覽器開啟直播歌詞預覽（只允許本機的 /overlay）。"""
    q = (request.get_json(force=True) or {}).get("query", "")
    import re
    import webbrowser
    if not re.fullmatch(r"[A-Za-z0-9=&%._-]*", q):
        raise P.UserError("網址格式不正確")
    webbrowser.open(f"http://{HOST}:{PORT}/overlay?{q}")
    return jsonify(ok=True)


@app.post("/api/open-url")
def open_url():
    """用系統預設瀏覽器開啟開發者頁面（只允許固定的幾個網址）。"""
    url = (request.get_json(force=True) or {}).get("url", "").rstrip("/")
    if url not in EXTERNAL_URLS:
        raise P.UserError("不允許開啟這個網址")
    import webbrowser
    webbrowser.open(url)
    return jsonify(ok=True)


@app.post("/api/reveal")
def reveal():
    """在檔案總管裡顯示匯出的檔案（只允許下載資料夾裡的練唱台資料夾）。"""
    path = Path((request.get_json(force=True) or {}).get("path", "")).resolve()
    base = (P.downloads_dir() / P.APP_NAME).resolve()
    if base not in path.parents or not path.exists():
        raise P.UserError("找不到這個檔案")
    import subprocess
    if sys.platform == "win32":
        subprocess.Popen(["explorer", "/select,", str(path)])
    elif sys.platform == "darwin":
        subprocess.Popen(["open", "-R", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path.parent)])
    return jsonify(ok=True)


# ---------------------------------------------------------------- 啟動

def watch_parent() -> None:
    """桌面版啟動時帶 --parent-pid：視窗程式結束（包含被強制關閉）時，伺服器跟著結束。"""
    if "--parent-pid" not in sys.argv:
        return
    pid = int(sys.argv[sys.argv.index("--parent-pid") + 1])

    def wait():
        if sys.platform == "win32":
            import ctypes
            k = ctypes.windll.kernel32
            h = k.OpenProcess(0x00100000, False, pid)   # SYNCHRONIZE
            if h:
                k.WaitForSingleObject(h, 0xFFFFFFFF)
        else:
            while True:
                try:
                    os.kill(pid, 0)
                except OSError:
                    break
                time.sleep(2)
        os._exit(0)

    threading.Thread(target=wait, daemon=True).start()


def main():
    P.LIB.mkdir(exist_ok=True)
    watch_parent()
    resume_downloads()
    threading.Thread(target=prefetcher, daemon=True, name="prefetch").start()
    url = f"http://{HOST}:{PORT}"
    print(f"\n  {P.APP_NAME} {VERSION} 已啟動：{url}\n  關閉這個視窗就會結束程式。\n")
    if "--no-browser" not in sys.argv:
        threading.Timer(1.2, lambda: webbrowser.open(url)).start()
    app.run(host=HOST, port=PORT, debug=False, threaded=True)


if __name__ == "__main__":
    main()
