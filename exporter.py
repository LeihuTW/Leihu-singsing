"""雷虎練唱台：匯出歌單（給觀眾看的網頁／Excel、可以再匯入的 CSV）與直播歌單回顧（2.3.0）。"""
from __future__ import annotations

import html
import io
import json
import time
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pipeline as P

# 分類名稱 → 批次加入歌單範本的「語言」欄（匯出的 CSV 再匯入時會放回同一個分類）
_CAT_LANG = {cat: names[0] for names, _lang, cat in P.LANG_FROM_SHEET}


def key_text(k) -> str:
    try:
        k = int(k or 0)
    except (TypeError, ValueError):
        return ""
    return f"{'+' if k > 0 else ''}{k}" if k else ""


def song_rows(scope_cat: str | None = None, ready_only: bool = True) -> list[dict]:
    """要匯出的歌：依分類（分類清單的順序，未分類放最後）、歌手、曲名排好。"""
    cats = P.load_categories()
    order = {c: i for i, c in enumerate(cats)}
    rows = []
    for m in P.list_songs():
        st = m.get("steps") or {}
        if ready_only and not st.get("pitch"):
            continue
        cat = m.get("category") or ""
        if scope_cat is not None and cat != scope_cat:
            continue
        rows.append({"title": m.get("title") or "", "artist": m.get("artist") or "", "category": cat,
                     "key": key_text(m.get("key")), "url": m.get("source_url") or ""})
    rows.sort(key=lambda r: (order.get(r["category"], 1e9 if r["category"] else 2e9), r["artist"] == "",
                             r["artist"], r["title"]))
    return rows


# ---------------------------------------------------------------- 網頁
_PAGE = """<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>__TITLE__</title>
<style>
:root { --bg:#f6f7f9; --card:#fff; --fg:#1b1f24; --muted:#66707c; --line:#e3e6ea; --acc:#13807a; }
@media (prefers-color-scheme: dark) { :root { --bg:#0e1318; --card:#161d24; --fg:#e8edf2; --muted:#93a0ad; --line:#26313b; --acc:#3cc6b6; } }
* { box-sizing: border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:16px/1.5 "Noto Sans TC","Microsoft JhengHei",system-ui,sans-serif; }
main { max-width: 860px; margin: 0 auto; padding: 24px 16px 48px; }
h1 { margin: 0 0 4px; font-size: 28px; }
.meta { color: var(--muted); font-size: 14px; margin-bottom: 16px; }
.tools { position: sticky; top: 0; background: var(--bg); padding: 8px 0 10px; z-index: 1; }
input { width: 100%; padding: 10px 12px; font-size: 16px; border: 1px solid var(--line); border-radius: 10px; background: var(--card); color: var(--fg); }
.chips { display: flex; flex-wrap: wrap; gap: 6px; margin-top: 8px; }
.chips button { border: 1px solid var(--line); background: var(--card); color: var(--fg); border-radius: 999px; padding: 4px 12px; font-size: 14px; cursor: pointer; }
.chips button.on { background: var(--acc); border-color: var(--acc); color: #fff; }
h2 { font-size: 17px; margin: 22px 0 6px; color: var(--acc); }
ol { list-style: none; margin: 0; padding: 0; background: var(--card); border: 1px solid var(--line); border-radius: 12px; overflow: hidden; }
li { display: flex; gap: 10px; align-items: baseline; padding: 9px 14px; border-top: 1px solid var(--line); }
li:first-child { border-top: 0; }
.t { font-weight: 600; }
.a { color: var(--muted); font-size: 14px; }
.k { margin-left: auto; color: var(--muted); font-size: 13px; white-space: nowrap; }
.none { color: var(--muted); padding: 20px 4px; }
</style></head><body><main>
<h1>__TITLE__</h1>
<div class="meta">共 <b id="n"></b> 首 · 更新於 __DATE__</div>
<div class="tools"><input id="q" type="search" placeholder="搜尋歌名或歌手" aria-label="搜尋歌名或歌手">
<div class="chips" id="cats"></div></div>
<div id="list"></div>
</main>
<script>
const SONGS = __DATA__, SHOW_KEY = __KEY__;
const $ = s => document.querySelector(s);
let cat = null;   // null＝全部；""＝未分類
const cats = [...new Set(SONGS.map(s => s.category))];
function chips() {
  const box = $("#cats"); box.innerHTML = "";
  if (cats.length < 2) return;
  for (const c of [null, ...cats]) {
    const b = document.createElement("button");
    b.textContent = c === null ? "全部" : (c || "其他");
    b.className = c === cat ? "on" : "";
    b.onclick = () => { cat = c; chips(); render(); };
    box.append(b);
  }
}
function render() {
  const q = $("#q").value.trim().toLowerCase();
  const list = SONGS.filter(s => (cat === null || s.category === cat) && (!q || (s.title + " " + s.artist).toLowerCase().includes(q)));
  $("#n").textContent = list.length;
  const box = $("#list"); box.innerHTML = "";
  if (!list.length) { box.innerHTML = '<div class="none">找不到符合的歌</div>'; return; }
  const groups = new Map();
  for (const s of list) { if (!groups.has(s.category)) groups.set(s.category, []); groups.get(s.category).push(s); }
  for (const [c, arr] of groups) {
    if (cats.length > 1) { const h = document.createElement("h2"); h.textContent = (c || "其他") + "（" + arr.length + "）"; box.append(h); }
    const ol = document.createElement("ol");
    for (const s of arr) {
      const li = document.createElement("li");
      const t = document.createElement("span"); t.className = "t"; t.textContent = s.title;
      const a = document.createElement("span"); a.className = "a"; a.textContent = s.artist;
      li.append(t, a);
      if (SHOW_KEY && s.key) { const k = document.createElement("span"); k.className = "k"; k.textContent = "KEY " + s.key; li.append(k); }
      ol.append(li);
    }
    box.append(ol);
  }
}
$("#q").oninput = render;
chips(); render();
</script></body></html>
"""


def songlist_html(rows: list[dict], title: str, show_key: bool) -> str:
    data = [{"title": r["title"], "artist": r["artist"], "category": r["category"], "key": r["key"]} for r in rows]
    js = json.dumps(data, ensure_ascii=False).replace("</", "<\\/")
    return (_PAGE.replace("__TITLE__", html.escape(title))
            .replace("__DATE__", time.strftime("%Y/%m/%d"))
            .replace("__KEY__", "true" if show_key else "false")
            .replace("__DATA__", js))


# ---------------------------------------------------------------- Excel（.xlsx，不需要額外套件）
def _col(i: int) -> str:
    s = ""
    i += 1
    while i:
        i, r = divmod(i - 1, 26)
        s = chr(65 + r) + s
    return s


def xlsx(sheet: str, header: list[str], rows: list[list], widths: list[int]) -> bytes:
    """產生一個工作表的 .xlsx：第一列粗體、凍結、可篩選。數字欄位寫成數字，其他寫成文字。"""
    def cell(ref, v, style=0):
        st = f' s="{style}"' if style else ""
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            return f'<c r="{ref}"{st}><v>{v}</v></c>'
        return f'<c r="{ref}" t="inlineStr"{st}><is><t xml:space="preserve">{escape(str(v or ""))}</t></is></c>'
    out = []
    for ri, r in enumerate([header] + rows, start=1):
        cs = "".join(cell(f"{_col(ci)}{ri}", v, 1 if ri == 1 else 0) for ci, v in enumerate(r))
        out.append(f'<row r="{ri}">{cs}</row>')
    last = f"{_col(len(header) - 1)}{len(rows) + 1}"
    cols = "".join(f'<col min="{i + 1}" max="{i + 1}" width="{w}" customWidth="1"/>' for i, w in enumerate(widths))
    ws = ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
          '<worksheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
          '<sheetViews><sheetView workbookViewId="0"><pane ySplit="1" topLeftCell="A2" activePane="bottomLeft" state="frozen"/></sheetView></sheetViews>'
          f'<cols>{cols}</cols><sheetData>{"".join(out)}</sheetData>'
          f'<autoFilter ref="A1:{last}"/></worksheet>')
    files = {
        "[Content_Types].xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
            '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
            '<Default Extension="xml" ContentType="application/xml"/>'
            '<Override PartName="/xl/workbook.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet.main+xml"/>'
            '<Override PartName="/xl/worksheets/sheet1.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.worksheet+xml"/>'
            '<Override PartName="/xl/styles.xml" ContentType="application/vnd.openxmlformats-officedocument.spreadsheetml.styles+xml"/>'
            '</Types>',
        "_rels/.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" Target="xl/workbook.xml"/>'
            '</Relationships>',
        "xl/workbook.xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<workbook xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            f'<sheets><sheet name="{escape(sheet[:31])}" sheetId="1" r:id="rId1"/></sheets>'
            '<definedNames><definedName name="_xlnm._FilterDatabase" localSheetId="0" hidden="1">'
            f"'{escape(sheet[:31])}'!$A$1:${_col(len(header) - 1)}${len(rows) + 1}</definedName></definedNames></workbook>",
        "xl/_rels/workbook.xml.rels": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            '<Relationship Id="rId1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet" Target="worksheets/sheet1.xml"/>'
            '<Relationship Id="rId2" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/styles" Target="styles.xml"/>'
            '</Relationships>',
        "xl/styles.xml": '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
            '<styleSheet xmlns="http://schemas.openxmlformats.org/spreadsheetml/2006/main">'
            '<fonts count="2"><font><sz val="11"/><name val="Microsoft JhengHei"/></font>'
            '<font><b/><sz val="11"/><name val="Microsoft JhengHei"/></font></fonts>'
            '<fills count="2"><fill><patternFill patternType="none"/></fill><fill><patternFill patternType="gray125"/></fill></fills>'
            '<borders count="1"><border><left/><right/><top/><bottom/><diagonal/></border></borders>'
            '<cellStyleXfs count="1"><xf numFmtId="0" fontId="0" fillId="0" borderId="0"/></cellStyleXfs>'
            '<cellXfs count="2"><xf numFmtId="0" fontId="0" fillId="0" borderId="0" xfId="0"/>'
            '<xf numFmtId="0" fontId="1" fillId="0" borderId="0" xfId="0" applyFont="1"/></cellXfs>'
            '<cellStyles count="1"><cellStyle name="Normal" xfId="0" builtinId="0"/></cellStyles></styleSheet>',
        "xl/worksheets/sheet1.xml": ws,
    }
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for n, t in files.items():
            z.writestr(n, t)
    return buf.getvalue()


def songlist_xlsx(rows: list[dict], show_key: bool) -> bytes:
    head = ["分類", "歌手", "曲名"] + (["KEY"] if show_key else [])
    data = [[r["category"] or "未分類", r["artist"], r["title"]] + ([r["key"]] if show_key else []) for r in rows]
    return xlsx("歌單", head, data, [12, 22, 34] + ([8] if show_key else []))


def songlist_csv(rows: list[dict]) -> bytes:
    """和「批次加入」的歌單範本同一個格式：換電腦時用批次加入一次匯回（會照 YouTube 網址重新下載）。"""
    import csv
    s = io.StringIO()
    w = csv.writer(s)
    w.writerow(["歌手", "曲名", "KEY", "語言", "YouTube"])
    for r in rows:
        w.writerow([r["artist"], r["title"], r["key"], _CAT_LANG.get(r["category"], ""), r["url"]])
    return ("﻿" + s.getvalue()).encode("utf-8")


# ---------------------------------------------------------------- 直播歌單回顧
def hms(sec: float) -> str:
    sec = max(0, int(sec))
    h, m, s = sec // 3600, sec // 60 % 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def song_start(it: dict) -> float:
    """這首開始唱的時間：2.3.0 起有記開始時間；舊紀錄是播放滿 30 秒時記的，往前推 30 秒。"""
    return float(it.get("s") or (float(it["t"]) - 30))


def chapters(items: list[dict], start: float, intro: str = "開場") -> str:
    """YouTube 章節：第一行一定要 00:00；每段至少 10 秒，太近的併掉。"""
    lines = [f"00:00 {intro}"] if intro else []
    last = 0.0
    for it in items:
        off = song_start(it) - start
        if off < 10 or off - last < 10:
            continue
        name = it.get("title", "") + (f" － {it['artist']}" if it.get("artist") else "")
        lines.append(f"{hms(off)} {name}")
        last = off
    return "\n".join(lines)


def setlist_xlsx(items: list[dict], start: float | None) -> bytes:
    head = ["#", "時間", "直播時間", "曲名", "歌手", "KEY"]
    data = []
    for i, it in enumerate(items, 1):
        t = song_start(it)
        data.append([i, time.strftime("%H:%M", time.localtime(t)), hms(t - start) if start else "",
                     it.get("title", ""), it.get("artist", ""), key_text(it.get("key"))])
    return xlsx("本場歌單", head, data, [5, 8, 10, 34, 22, 8])


def save_download(name: str, data: bytes | str) -> Path:
    """存到「下載\\雷虎練唱台」，同名的話加上編號。"""
    base = P.downloads_dir() / P.APP_NAME
    base.mkdir(parents=True, exist_ok=True)
    p = base / name
    n = 2
    while p.exists():
        p = base / f"{Path(name).stem} ({n}){Path(name).suffix}"
        n += 1
    if isinstance(data, str):
        p.write_text(data, "utf-8-sig")
    else:
        p.write_bytes(data)
    return p
