"use strict";
/* 練唱台 前端 */
const $ = (s, r = document) => r.querySelector(s);
const $$ = (s, r = document) => [...r.querySelectorAll(s)];
const NAMES = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"];
const noteName = m => NAMES[((Math.round(m) % 12) + 12) % 12] + (Math.floor(Math.round(m) / 12) - 1);
const fmt = (t, dec = 1) => { t = Math.max(0, t); const m = Math.floor(t / 60); const s = t - m * 60; return `${m}:${s.toFixed(dec).padStart(dec ? dec + 3 : 2, "0")}`; };
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const store = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : JSON.parse(v); } catch { return d; } },
  set(k, v) { try { localStorage.setItem(k, JSON.stringify(v)); } catch {} },
};

async function api(path, opts = {}) {
  const o = { ...opts };
  if (o.json !== undefined) { o.body = JSON.stringify(o.json); o.headers = { "Content-Type": "application/json" }; o.method = o.method || "POST"; delete o.json; }
  const r = await fetch(path, o);
  const ct = r.headers.get("content-type") || "";
  const data = ct.includes("json") ? await r.json() : await r.text();
  if (!r.ok) throw new Error((data && data.error) || `請求失敗（${r.status}）`);
  return data;
}
let toastTimer;
function toast(msg, ms = 3500) {
  const t = $("#toast"); t.textContent = msg; t.hidden = false;
  clearTimeout(toastTimer); toastTimer = setTimeout(() => (t.hidden = true), ms);
}

/* ------------------------------------------------------------ 狀態 */
const S = {
  song: null, pitch: null, notes: [], lines: [], range: [55, 79],
  key: 0, speed: 1, bufs: {}, guide: null,
  vol: { lead: 1, back: 1, inst: 1, guide: 0, take: 0.9 },
  playing: false, t: 0, startCtx: 0, nodes: {}, loopA: null, loopB: null, zoom: 8,
  mic: null, live: [], liveNow: null, rec: null,
  takes: [], take: null, takeBuf: null,
  tap: null, dirty: true, loadingAudio: false,
};
let ctx = null;
const ac = () => (ctx ||= new (window.AudioContext || window.webkitAudioContext)());

/* ------------------------------------------------------------ 導覽 */
function show(view) {
  $("#view-lib").hidden = view !== "lib";
  $("#view-practice").hidden = view !== "practice";
  $("#nav-lib").classList.toggle("cur", view === "lib");
  $("#nav-practice").classList.toggle("cur", view === "practice");
  if (view === "lib") { pause(); loadSongs(); }
  S.dirty = true;
}
$("#nav-lib").onclick = () => show("lib");
$("#back").onclick = () => show("lib");
$("#nav-practice").onclick = () => S.song && show("practice");

/* ------------------------------------------------------------ 歌曲庫 */
/* 歌曲庫：分類（自訂）＋關鍵字搜尋 */
const LIB = { songs: [], cats: [], cur: store.get("libCat", "__all__"), q: "" };
async function loadSongs() {
  const [songs, cats] = await Promise.all([api("/api/songs"), api("/api/categories")]);
  LIB.songs = songs; LIB.cats = cats;
  if (LIB.cur !== "__all__" && LIB.cur !== "__none__" && !cats.includes(LIB.cur)) LIB.cur = "__all__";
  songs.forEach(s => (titles[s.id] = s.title));
  renderCats(); renderSongs();
}
function renderCats() {
  const box = $("#cats"); box.innerHTML = "";
  const count = c => LIB.songs.filter(s => c === "__all__" || (c === "__none__" ? !s.category : s.category === c)).length;
  const items = [["__all__", "全部"], ...LIB.cats.map(c => [c, c]), ["__none__", "未分類"]];
  for (const [key, label] of items) {
    if (key === "__none__" && !LIB.cats.length) continue;
    const b = document.createElement("button");
    b.className = "cat" + (LIB.cur === key ? " cur" : "");
    b.innerHTML = `<span></span><span class="n">${count(key)}</span>`;
    $("span", b).textContent = label;
    b.onclick = () => { LIB.cur = key; store.set("libCat", key); renderCats(); renderSongs(); };
    box.appendChild(b);
  }
  const add = document.createElement("button");
  add.className = "cat add"; add.textContent = "＋ 新增分類";
  add.onclick = () => {
    const inp = document.createElement("input");
    inp.className = "cat-input"; inp.placeholder = "例如：日文歌";
    add.replaceWith(inp); inp.focus();
    const done = async ok => {
      const v = inp.value.trim();
      if (ok && v) {
        try { LIB.cats = await api("/api/categories", { json: { action: "add", name: v } }); LIB.cur = v; store.set("libCat", v); }
        catch (e) { toast(e.message); }
      }
      renderCats(); renderSongs();
    };
    inp.onkeydown = e => { if (e.key === "Enter") done(true); if (e.key === "Escape") done(false); };
    inp.onblur = () => done(true);
  };
  box.appendChild(add);
  const real = LIB.cats.includes(LIB.cur);
  $("#cat-tools").hidden = !real;
  if (real) $("#cat-hint").textContent = `匯入的歌會放進「${LIB.cur}」`;
}
$("#lib-q").addEventListener("input", e => { LIB.q = e.target.value.trim().toLowerCase(); renderSongs(); });
$("#cat-rename").onclick = async () => {
  const name = LIB.cur;
  const btn = [...$$(".cat", $("#cats"))].find(b => $("span", b)?.textContent === name);
  if (!btn) return;
  const inp = document.createElement("input");
  inp.className = "cat-input"; inp.value = name;
  btn.replaceWith(inp); inp.focus(); inp.select();
  const done = async ok => {
    const nv = inp.value.trim();
    if (ok && nv && nv !== name) {
      try { LIB.cats = await api("/api/categories", { json: { action: "rename", name, new: nv } }); LIB.cur = nv; store.set("libCat", nv); }
      catch (e) { toast(e.message); }
    }
    loadSongs();
  };
  inp.onkeydown = e => { if (e.key === "Enter") done(true); if (e.key === "Escape") done(false); };
  inp.onblur = () => done(true);
};
$("#cat-del").onclick = async () => {
  const b = $("#cat-del");
  if (!b.dataset.armed) { b.dataset.armed = "1"; b.textContent = "確定刪除？歌曲會移到未分類"; setTimeout(() => { delete b.dataset.armed; b.textContent = "刪除分類"; }, 3500); return; }
  delete b.dataset.armed; b.textContent = "刪除分類";
  try { await api("/api/categories", { json: { action: "delete", name: LIB.cur } }); LIB.cur = "__all__"; store.set("libCat", "__all__"); }
  catch (e) { toast(e.message); }
  loadSongs();
};
function songMatches(s) {
  if (LIB.cur === "__none__" && s.category) return false;
  if (LIB.cur !== "__all__" && LIB.cur !== "__none__" && s.category !== LIB.cur) return false;
  if (!LIB.q) return true;
  const hay = [s.title, s.yt_artist, s.yt_track, s.yt_uploader, s.category, s.lyrics_text,
    s.lyrics_online && s.lyrics_online.artist].filter(Boolean).join(" ").toLowerCase();
  return LIB.q.split(/\s+/).every(w => hay.includes(w));
}
function renderSongs() {
  const box = $("#songs");
  box.innerHTML = "";
  if (!LIB.songs.length) {
    box.innerHTML = `<div class="empty">歌曲庫是空的。匯入第一首歌，處理完就能開始練。</div>`;
    return;
  }
  const list = LIB.songs.filter(songMatches);
  if (!list.length) {
    box.innerHTML = `<div class="empty">${LIB.q ? "找不到符合的歌曲。" : "這個分類還沒有歌。匯入歌曲，或在其他歌的「分類」選單把歌放進來。"}</div>`;
    return;
  }
  for (const s of list) {
    const el = document.createElement("div");
    el.className = "song";
    const st = s.steps || {};
    const chips = [];
    if (s.duration) chips.push(`<span class="num">${fmt(s.duration, 0)}</span>`);
    if (st.separate) chips.push(`<span class="chip ${s.separation === "uvr" ? "ok" : "warn"}">${s.separation === "uvr" ? "AI 分軌" : "聲像分軌"}</span>`);
    if (s.acapella) chips.push(`<span class="chip acap">清唱</span>`);
    chips.push(`<span class="chip ${st.pitch ? "ok" : ""}">${st.pitch ? "旋律" : "旋律未完成"}</span>`);
    chips.push(`<span class="chip ${st.lyrics ? "ok" : ""}">${st.lyrics ? "歌詞" : "無歌詞"}</span>`);
    if (s.lyrics_pending) chips.push(`<span class="chip warn">有線上歌詞可選</span>`);
    if (s.takes) chips.push(`<span>${s.takes} 次錄音</span>`);
    if (s.best != null) chips.push(`<span>最佳 <b class="num">${Math.round(s.best * 100)}%</b></span>`);
    el.innerHTML = `
      <div><div class="t"></div><div class="m">${chips.join("")}</div></div>
      <div class="acts">
        <button class="primary" data-a="open" ${st.pitch ? "" : "disabled"}>練唱</button>
        <select class="catsel" data-a="cat" aria-label="分類"></select>
        <button data-a="rename">改名</button>
        <select data-a="redo" aria-label="重新處理">
          <option value="">重新處理…</option>
          <option value="separate,pitch,lyrics">全部重做</option>
          <option value="lyrics">只重新辨識歌詞</option>
          <option value="pitch">只重新分析旋律</option>
        </select>
        <button class="ghost danger" data-a="del">刪除</button>
      </div>
      ${(s.notes || []).map(n => `<div class="note"></div>`).join("")}`;
    $(".t", el).textContent = s.title;
    $$(".note", el).forEach((n, i) => (n.textContent = s.notes[i]));
    const sel = $('[data-a="cat"]', el);
    sel.innerHTML = `<option value="">未分類</option>` + LIB.cats.map(() => `<option></option>`).join("");
    LIB.cats.forEach((c, i) => { sel.options[i + 1].value = c; sel.options[i + 1].textContent = c; });
    sel.value = s.category || "";
    sel.hidden = !LIB.cats.length;
    sel.onchange = async () => {
      try { await api(`/api/songs/${s.id}`, { method: "PATCH", json: { category: sel.value || null } }); s.category = sel.value || null; renderCats(); renderSongs(); }
      catch (e) { toast(e.message); }
    };
    $('[data-a="open"]', el).onclick = () => openSong(s.id);
    $('[data-a="rename"]', el).onclick = async () => {
      const t = $(".t", el);
      const inp = document.createElement("input");
      inp.type = "text"; inp.value = s.title; inp.style.width = "100%";
      t.replaceWith(inp); inp.focus(); inp.select();
      const done = async ok => {
        if (ok && inp.value.trim() && inp.value.trim() !== s.title) {
          try { await api(`/api/songs/${s.id}`, { method: "PATCH", json: { title: inp.value.trim() } }); } catch (e) { toast(e.message); }
        }
        loadSongs();
      };
      inp.onkeydown = e => { if (e.key === "Enter") done(true); if (e.key === "Escape") done(false); };
      inp.onblur = () => done(true);
    };
    $('[data-a="redo"]', el).onchange = async e => {
      const v = e.target.value; e.target.value = "";
      if (!v) return;
      try { await api(`/api/songs/${s.id}/process`, { json: { steps: v.split(",") } }); toast("已加入處理佇列"); pollJobs(); }
      catch (err) { toast(err.message); }
    };
    const del = $('[data-a="del"]', el);
    del.onclick = async () => {
      if (!del.dataset.armed) {
        del.dataset.armed = "1"; del.textContent = "確定刪除？";
        setTimeout(() => { delete del.dataset.armed; del.textContent = "刪除"; }, 3000);
        return;
      }
      await api(`/api/songs/${s.id}`, { method: "DELETE" });
      if (S.song && S.song.id === s.id) { S.song = null; $("#nav-practice").disabled = true; }
      loadSongs();
    };
    box.appendChild(el);
  }
}
const importCategory = () => (LIB.cats.includes(LIB.cur) ? LIB.cur : "");

async function importFiles(files) {
  for (const f of files) {
    const fd = new FormData();
    fd.append("file", f);
    fd.append("category", importCategory());
    toast(`上傳 ${f.name}…`);
    try { await api("/api/songs", { method: "POST", body: fd }); }
    catch (e) { toast(`${f.name}：${e.message}`); }
  }
  loadSongs(); pollJobs();
}
$("#yt-form").addEventListener("submit", async e => {
  e.preventDefault();
  const inp = $("#yt-url"), msg = $("#yt-msg"), btn = $("#yt-go");
  const url = inp.value.trim();
  msg.classList.remove("err");
  if (!url) { msg.textContent = "先貼上 YouTube 影片網址"; msg.classList.add("err"); return; }
  btn.disabled = true; msg.textContent = "加入佇列…";
  try {
    await api("/api/youtube", { json: { url, category: importCategory() } });
    inp.value = ""; msg.textContent = "已開始下載，完成後會自動分離、分析旋律並辨識歌詞。";
    loadSongs(); pollJobs();
  } catch (err) { msg.textContent = err.message; msg.classList.add("err"); }
  finally { btn.disabled = false; }
});
$("#yt-url").addEventListener("input", () => { $("#yt-msg").textContent = ""; $("#yt-msg").classList.remove("err"); });
$("#import").onchange = e => { importFiles([...e.target.files]); e.target.value = ""; };
const drop = $("#drop");
["dragenter", "dragover"].forEach(ev => document.addEventListener(ev, e => {
  if (!$("#view-lib").hidden && e.dataTransfer?.types?.includes("Files")) { e.preventDefault(); drop.classList.add("over"); }
}));
["dragleave", "drop"].forEach(ev => document.addEventListener(ev, () => drop.classList.remove("over")));
document.addEventListener("drop", e => {
  if ($("#view-lib").hidden || !e.dataTransfer?.files?.length) return;
  e.preventDefault(); importFiles([...e.dataTransfer.files]);
});

/* ------------------------------------------------------------ 背景工作 */
const jobSeen = {};
let jobTimer = null;
async function pollJobs() {
  clearTimeout(jobTimer);
  let st;
  try { st = await api("/api/status"); } catch { jobTimer = setTimeout(pollJobs, 3000); return; }
  updateClaudeChip(st.claude_ready, st.session_cost);
  if (st.version) { $("#ver").textContent = st.version; document.title = `雷虎練唱台 ${st.version}`; }
  const box = $("#jobs");
  box.innerHTML = "";
  let busy = false, finished = false;
  for (const j of st.jobs) {
    if (j.status === "queued" || j.status === "running") busy = true;
    if (j.status === "done" && jobSeen[j.id] && jobSeen[j.id] !== "done") {
      finished = true;
      if (j.kind === "take" && S.song && j.song === S.song.id) refreshTakes(j.label);
    }
    jobSeen[j.id] = j.status;
    if (j.kind !== "process" || j.status === "done") continue;
    const el = document.createElement("div");
    el.className = "job" + (j.status === "error" ? " error" : "");
    el.innerHTML = `<span class="lbl"></span><span class="num">${Math.round(j.progress * 100)}%</span>
      <div class="meter"><i style="width:${Math.round(j.progress * 100)}%"></i></div>${j.error ? `<div class="err"></div>` : ""}`;
    $(".lbl", el).textContent = `${songTitle(j.song)}：${j.status === "queued" ? "排隊中" : j.step}`;
    if (j.error) $(".err", el).textContent = j.error;
    box.appendChild(el);
  }
  if (finished && !$("#view-lib").hidden) loadSongs();
  jobTimer = setTimeout(pollJobs, busy ? 1000 : 5000);
}
const titles = {};
let titleFetch = 0;
function songTitle(id) {
  if (!titles[id] || /下載中/.test(titles[id])) {
    if (Date.now() - titleFetch > 1500) { titleFetch = Date.now(); api("/api/songs").then(l => l.forEach(s => (titles[s.id] = s.title))); }
  }
  return titles[id] || "歌曲";
}
async function waitJob(jid, onStep) {
  for (;;) {
    const j = await api(`/api/jobs/${jid}`);
    onStep && onStep(j);
    if (j.status === "done") return j;
    if (j.status === "error") throw new Error(j.error || "處理失敗");
    await new Promise(r => setTimeout(r, 700));
  }
}
const usd = v => v == null ? "—" : "US$" + (v < 0.01 ? v.toFixed(4) : v.toFixed(2));
const num = v => (v || 0).toLocaleString("en-US");
const usageLine = u => u ? `這次用了 ${num(u.input_tokens + u.output_tokens)} tokens（輸入 ${num(u.input_tokens)}、輸出 ${num(u.output_tokens)}），約 ${usd(u.cost_usd)}` : "";
function updateClaudeChip(ready, cost) {
  const c = $("#claude-chip");
  c.textContent = !ready ? "Claude：未設定" : cost ? `Claude：本次 ${usd(cost)}` : "Claude：可用（需連網）";
  c.className = "chip" + (ready ? " ok" : "");
}
$("#claude-chip").onclick = () => openSettings(true);
async function renderUsage() {
  let u;
  try { u = await api("/api/usage"); } catch { return; }
  const purpose = { align: "歌詞對齊", coach: "練習建議" };
  const cells = [
    ["本次開啟", u.session.cost_usd, `${num(u.session.input_tokens + u.session.output_tokens)} tokens・${u.session.calls} 次`],
    ["累計", u.total.cost_usd, `${num(u.total.input_tokens + u.total.output_tokens)} tokens・${u.total.calls} 次`],
  ];
  if (u.last) cells.unshift(["最近一次", u.last.cost_usd, `${num(u.last.input_tokens + u.last.output_tokens)} tokens・${purpose[u.last.purpose] || ""}`]);
  if (u.balance) cells.push(["估計剩餘餘額", u.balance.estimated_usd, `填入 ${usd(u.balance.entered_usd)} 後用了 ${usd(u.balance.spent_since_usd)}`]);
  const grid = $("#usage-grid"); grid.innerHTML = "";
  for (const [label, v, sub] of cells) {
    const el = document.createElement("div");
    el.innerHTML = `<span></span><b></b><span></span>`;
    const sp = $$("span", el); sp[0].textContent = label; sp[1].textContent = sub; $("b", el).textContent = usd(v);
    grid.appendChild(el);
  }
  const rec = $("#usage-recent"); rec.innerHTML = "";
  for (const c of u.recent || []) {
    const d = new Date(c.time * 1000), el = document.createElement("div");
    el.innerHTML = "<span></span><span></span>";
    const sp = $$("span", el);
    sp[0].textContent = `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}　${purpose[c.purpose] || c.purpose}　${c.song || ""}`;
    sp[1].textContent = `${num(c.input_tokens + c.output_tokens)} tok・${usd(c.cost_usd)}`;
    rec.appendChild(el);
  }
  if (!u.total.calls) rec.textContent = "還沒有使用過 Claude。";
}

/* ------------------------------------------------------------ 開啟歌曲 */
async function openSong(id) {
  pause();
  stopTap();
  const [song, pitch, lyr] = await Promise.all([
    api(`/api/songs/${id}`), api(`/api/songs/${id}/pitch`), api(`/api/songs/${id}/lyrics`),
  ]);
  S.song = song; titles[id] = song.title;
  S.pitch = { hop: pitch.hop, midi: Float32Array.from(pitch.midi, v => (v == null ? NaN : v)) };
  S.notes = pitch.notes.map(([s, e, m]) => ({ s, e, m }));
  S.autoLrc = lyr.auto;
  $("#lrc").value = lyr.lrc;
  setLines(lyr.lrc);
  const saved = store.get(`song:${id}`, {});
  S.key = saved.key || 0; S.speed = saved.speed || 1;
  if (saved.vol) Object.assign(S.vol, saved.vol);
  S.t = 0; S.loopA = S.loopB = null; S.live = []; S.take = null; S.takeBuf = null;
  computeRange();
  $("#p-title").textContent = song.title;
  $("#p-chips").innerHTML = `<span class="chip ${song.separation === "uvr" ? "ok" : "warn"}">${song.separation === "uvr" ? "AI 分軌" : "聲像分軌"}</span>`
    + (song.acapella ? `<span class="chip acap">清唱／純人聲</span>` : "");
  $("#ls-results").innerHTML = ""; $("#ls-msg").textContent = ""; $("#ls-q").value = "";
  S.editLyr = false; syncLyrButtons();
  checkCandidates(id);

  $("#speed").value = String(S.speed);
  $("#key-val").textContent = (S.key > 0 ? "+" : "") + S.key;
  $("#nav-practice").disabled = false;
  renderFaders();
  renderTakes(song.takes);
  $("#take-detail").innerHTML = "";
  $("#exp-msg").textContent = ""; $("#exp-show").hidden = true; $("#align-cost").textContent = ""; $("#align-state").textContent = "";
  show("practice");
  await loadAudio();
}

/* 整首歌的音域：用音高曲線的分布（不只音符），避免短促的高音被排除在外 */
function computeRange() {
  const vals = [];
  // 低於 D2 的多半是貝斯殘留，不算人聲；高音完整保留，低音去掉最低 1% 的雜訊
  if (S.pitch) for (const v of S.pitch.midi) if (!isNaN(v) && v >= 38) vals.push(v);
  for (const n of S.notes) if (n.m >= 38) vals.push(n.m);
  if (!vals.length) { S.range = [55 + S.key, 79 + S.key]; return; }
  vals.sort((a, b) => a - b);
  const q = p => vals[Math.min(vals.length - 1, Math.max(0, Math.floor(p * (vals.length - 1))))];
  S.range = fitSpan(q(0.01) + S.key - 2, q(0.998) + S.key + 2, 18);
}
function fitSpan(lo, hi, min) {
  lo = Math.floor(lo); hi = Math.ceil(hi);
  if (hi - lo < min) { const c = (hi + lo) / 2; lo = Math.floor(c - min / 2); hi = lo + min; }
  return [lo, hi];
}
async function loadAudio() {
  const id = S.song.id;
  const q = `key=${S.key}&speed=${S.speed}`;
  const busy = $("#loading");
  S.loadingAudio = true;
  busy.hidden = false; busy.textContent = "載入分軌…";
  try {
    const v = await api(`/api/songs/${id}/variant`, { json: { key: S.key, speed: S.speed } });
    if (!v.ready) {
      await waitJob(v.job, j => (busy.textContent = `${j.step}（${Math.round(j.progress * 100)}%）`));
    }
    busy.textContent = "解碼音訊…";
    const ctx = ac();
    const stems = await Promise.all(["lead", "back", "inst"].map(async s => {
      const r = await fetch(`/api/songs/${id}/audio/${s}?${q}`);
      if (!r.ok) throw new Error("讀不到分軌檔");
      return ctx.decodeAudioData(await r.arrayBuffer());
    }));
    if (!S.song || S.song.id !== id) return;
    S.bufs = { lead: stems[0], back: stems[1], inst: stems[2] };
    S.dur = S.song.duration || stems[2].duration * S.speed;
    S.guide = await renderGuide();
    $("#seek").max = String(S.dur);
    busy.hidden = true;
  } catch (e) {
    busy.textContent = e.message;
  } finally {
    S.loadingAudio = false; S.dirty = true;
  }
}

/* 旋律導引音：依音符合成，跟著 KEY 與速度 */
async function renderGuide() {
  if (!S.notes.length) return null;
  const sr = 22050;
  const len = Math.ceil((S.dur / S.speed + 1) * sr);
  const oc = new OfflineAudioContext(1, len, sr);
  const out = oc.createGain(); out.gain.value = 0.5; out.connect(oc.destination);
  for (const n of S.notes) {
    const s = n.s / S.speed, e = n.e / S.speed;
    if (e - s < 0.06) continue;
    const o = oc.createOscillator();
    o.type = "triangle";
    o.frequency.value = 440 * Math.pow(2, (n.m + S.key - 69) / 12);
    const g = oc.createGain();
    g.gain.setValueAtTime(0, s);
    g.gain.linearRampToValueAtTime(0.35, s + 0.02);
    g.gain.setValueAtTime(0.35, Math.max(s + 0.03, e - 0.04));
    g.gain.linearRampToValueAtTime(0, e);
    o.connect(g).connect(out); o.start(s); o.stop(e + 0.01);
  }
  return oc.startRendering();
}

/* ------------------------------------------------------------ 播放 */
const TRACKS = [
  { k: "lead", label: "主唱", color: "--melody" },
  { k: "back", label: "合音", color: "--muted" },
  { k: "inst", label: "伴奏", color: "--accent" },
  { k: "guide", label: "導引音", color: "--melody" },
  { k: "take", label: "我的錄音", color: "--voice" },
];
function cur() {
  if (!S.playing) return S.t;
  return S.t + Math.max(0, ac().currentTime - S.startCtx) * S.speed;
}
function stopNodes() {
  for (const n of Object.values(S.nodes)) { try { n.src.stop(); } catch {} n.src.disconnect(); }
  S.nodes = {};
}
function play() {
  if (S.loadingAudio || !S.bufs.inst) return;
  const c = ac(); c.resume();
  stopNodes();
  if (S.t >= S.dur - 0.05) S.t = 0;
  const now = c.currentTime + 0.06;
  const pos = S.t / S.speed;
  const srcs = { lead: S.bufs.lead, back: S.bufs.back, inst: S.bufs.inst, guide: S.guide };
  for (const [k, buf] of Object.entries(srcs)) {
    if (!buf || pos >= buf.duration) continue;
    startNode(k, buf, now, pos);
  }
  if (S.take && S.takeBuf && S.take.pitch) {
    if (Math.abs(S.take.speed - S.speed) < 1e-3) {
      const tau = (S.t - S.take.pitch.t0) / S.speed;
      if (tau < S.takeBuf.duration) startNode("take", S.takeBuf, now + Math.max(0, -tau), Math.max(0, tau));
    }
  }
  S.startCtx = now; S.playing = true;
  $("#play").textContent = "暫停";
}
function startNode(k, buf, when, pos) {
  const c = ac();
  const src = c.createBufferSource(); src.buffer = buf;
  const g = c.createGain(); g.gain.value = S.vol[k] ?? 1;
  src.connect(g).connect(c.destination);
  src.start(when, pos);
  S.nodes[k] = { src, g };
}
function pause() {
  if (S.playing) S.t = cur();
  stopNodes(); S.playing = false;
  $("#play").textContent = "播放";
  if (S.rec) stopRecording();
  S.dirty = true;
}
function seek(t) {
  S.t = clamp(t, 0, S.dur || 0);
  S.live = [];
  if (S.playing) play();
  S.dirty = true;
}
$("#play").onclick = () => (S.playing ? pause() : play());
$("#seek").oninput = e => seek(+e.target.value);
$("#zoom").oninput = e => { S.zoom = +e.target.value; S.dirty = true; };
$("#set-a").onclick = () => { S.loopA = cur(); if (S.loopB != null && S.loopB <= S.loopA) S.loopB = null; S.dirty = true; };
$("#set-b").onclick = () => { const t = cur(); if (S.loopA != null && t > S.loopA + 0.5) S.loopB = t; else toast("先在較早的位置按 A"); S.dirty = true; };
$("#clear-ab").onclick = () => { S.loopA = S.loopB = null; S.dirty = true; };

async function changeVariant(key, speed) {
  const wasPlaying = S.playing;
  const songT = cur();
  pause();
  S.key = clamp(key, -12, 12); S.speed = speed;
  $("#key-val").textContent = (S.key > 0 ? "+" : "") + S.key;
  $("#speed").value = String(S.speed);
  saveSongPrefs();
  computeRange();
  await loadAudio();
  S.t = songT;
  if (wasPlaying) play();
}
$("#key-up").onclick = () => changeVariant(S.key + 1, S.speed);
$("#key-down").onclick = () => changeVariant(S.key - 1, S.speed);
$("#speed").onchange = e => changeVariant(S.key, +e.target.value);
function saveSongPrefs() { if (S.song) store.set(`song:${S.song.id}`, { key: S.key, speed: S.speed, vol: S.vol }); }

/* 混音 */
const PRESETS = {
  orig: { lead: 1, back: 1, inst: 1, guide: 0 },
  karaoke: { lead: 0, back: 1, inst: 1, guide: 0 },
  practice: { lead: 0.25, back: 0.9, inst: 1, guide: 0.45 },
  solo: { lead: 1, back: 0, inst: 0, guide: 0 },
};
function renderFaders() {
  const box = $("#faders"); box.innerHTML = "";
  for (const tr of TRACKS) {
    const row = document.createElement("label");
    row.className = "fader";
    row.innerHTML = `<span><i class="sw" style="background:var(${tr.color})"></i>${tr.label}</span>
      <input type="range" min="0" max="1.5" step="0.05" id="vol-${tr.k}"><span class="num"></span>`;
    const inp = $("input", row), out = $("span.num", row);
    inp.value = S.vol[tr.k]; out.textContent = Math.round(S.vol[tr.k] * 100);
    inp.oninput = () => {
      S.vol[tr.k] = +inp.value; out.textContent = Math.round(S.vol[tr.k] * 100);
      if (S.nodes[tr.k]) S.nodes[tr.k].g.gain.value = S.vol[tr.k];
      saveSongPrefs();
    };
    box.appendChild(row);
  }
}
$$("[data-preset]").forEach(b => (b.onclick = () => {
  Object.assign(S.vol, PRESETS[b.dataset.preset]);
  for (const k of Object.keys(S.vol)) if (S.nodes[k]) S.nodes[k].g.gain.value = S.vol[k];
  renderFaders(); saveSongPrefs();
}));

/* ------------------------------------------------------------ 歌詞 */
/* 歌詞格式：LRC，每句 [分:秒]；逐字時間用 Enhanced LRC 的 <分:秒> 標記（例如 [00:12.00]<00:12.00>是<00:12.40>我） */
const CJK = "぀-ヿ㐀-鿿豈-﫿가-힯";
const UNIT_RE = new RegExp(`[${CJK}]|[^\\s${CJK}]+`, "g");
const isPunct = c => /^[\p{P}\p{S}]+$/u.test(c);
function splitUnits(text) {
  const out = []; let m; UNIT_RE.lastIndex = 0;
  while ((m = UNIT_RE.exec(text))) {
    const after = text[m.index + m[0].length];
    out.push({ c: m[0], sp: after !== undefined && /\s/.test(after), t: null });
  }
  return out;
}
const unitsText = units => units.map(u => u.c + (u.sp ? " " : "")).join("").trim();
function parseLrc(text) {
  const out = [];
  for (const raw of text.split(/\r?\n/)) {
    if (/^\[(ti|ar|al|re|by|offset|length):/i.test(raw)) continue;
    const tags = [...raw.matchAll(/\[(\d+):(\d+(?:\.\d+)?)\]/g)];
    const body = raw.replace(/\[[^\]]*\]/g, "");
    let chars = null;
    if (/<\d+:\d+(?:\.\d+)?>/.test(body)) {
      chars = []; let curT = null;
      for (const part of body.split(/(<\d+:\d+(?:\.\d+)?>)/)) {
        const tm = part.match(/^<(\d+):(\d+(?:\.\d+)?)>$/);
        if (tm) { curT = +tm[1] * 60 + +tm[2]; continue; }
        const us = splitUnits(part);
        us.forEach((u, i) => { u.t = i === 0 ? curT : null; });
        if (us.length && /\s$/.test(part)) us[us.length - 1].sp = true;
        chars.push(...us);
      }
      if (!chars.length) chars = null;
    }
    const plain = body.replace(/<\d+:\d+(?:\.\d+)?>/g, "").trim();
    if (!tags.length) { if (plain) out.push({ t: null, text: plain, chars: null }); continue; }
    for (const m of tags) out.push({ t: +m[1] * 60 + +m[2], text: chars ? unitsText(chars) : plain, chars: chars ? chars.map(u => ({ ...u })) : null });
  }
  return out;
}
const lrcTime = t => { t = Math.max(0, t); const m = Math.floor(t / 60); return `${String(m).padStart(2, "0")}:${(t - m * 60).toFixed(2).padStart(5, "0")}`; };
function lineToLrc(l) {
  const head = l.t != null ? `[${lrcTime(l.t)}]` : "";
  if (l.chars && !l.autoChars && l.chars.every(u => u.t != null || isPunct(u.c)))
    return head + l.chars.map(u => (isPunct(u.c) ? "" : `<${lrcTime(u.t)}>`) + u.c + (u.sp ? " " : "")).join("").trimEnd();
  return head + l.text;
}
const toLrc = lines => lines.map(lineToLrc).join("\n") + "\n";

/* 依旋律音符把一句的每個字放到音上（動態規劃：字的分界盡量落在音符之間的空隙） */
function distributeChars(line, limit) {
  const units = line.chars ? line.chars.map(u => ({ ...u })) : splitUnits(line.text);
  const idx = units.map((u, i) => (isPunct(u.c) ? -1 : i)).filter(i => i >= 0);
  const U = idx.length;
  if (!U) return units;
  const notes = S.notes.filter(n => n.e > line.t - 0.15 && n.s < limit);
  const N = notes.length;
  const assign = [];
  if (N >= U) {
    const gap = k => (k === 0 ? 1 : Math.min(0.4, Math.max(0, notes[k].s - notes[k - 1].e)));
    const INF = 1e9, dp = [], from = [];
    for (let i = 0; i <= U; i++) { dp.push(new Array(N + 1).fill(INF)); from.push(new Array(N + 1).fill(-1)); }
    dp[0][0] = 0;
    for (let i = 1; i <= U; i++)
      for (let j = i; j <= N - (U - i); j++)
        for (let k = i - 1; k < j; k++) {
          if (dp[i - 1][k] >= INF) continue;
          const c = dp[i - 1][k] + 0.3 * (j - k - 1) - 2 * gap(k);
          if (c < dp[i][j]) { dp[i][j] = c; from[i][j] = k; }
        }
    let j = N;
    for (let i = U; i >= 1; i--) { const k = from[i][j]; assign[i - 1] = notes[Math.max(0, k)].s; j = k; }
  } else {
    // 音符比字少：沿著有聲音的時間平均分配
    const segs = N ? notes.map(n => [Math.max(n.s, line.t - 0.15), n.e]) : [[line.t, Math.max(line.t + 0.5, Math.min(limit, line.end || line.t + 3))]];
    const total = segs.reduce((a, [s0, e0]) => a + (e0 - s0), 0);
    for (let i = 0; i < U; i++) {
      let pos = (i / U) * total;
      for (const [s0, e0] of segs) { if (pos <= e0 - s0) { assign[i] = s0 + pos; break; } pos -= e0 - s0; }
      if (assign[i] == null) assign[i] = segs[segs.length - 1][1];
    }
  }
  idx.forEach((ui, k) => (units[ui].t = Math.round(assign[k] * 100) / 100));
  let last = assign[0];
  units.forEach(u => { if (u.t == null) u.t = last; else last = u.t; });
  return units;
}
function fillCharTimes(l) {
  const u = l.chars;
  if (u[0].t == null) u[0].t = l.t;
  for (let i = 1; i < u.length; i++) {
    if (u[i].t != null) continue;
    let j = i; while (j < u.length && u[j].t == null) j++;
    const a = u[i - 1].t, b = j < u.length ? u[j].t : a + 0.25 * (j - i + 1);
    for (let k = i; k < j; k++) u[k].t = Math.round((a + ((b - a) * (k - i + 1)) / (j - i + 1)) * 100) / 100;
  }
}
function lineLimit(i) {
  const l = S.lines[i], next = S.lines[i + 1];
  return next ? next.t : (l.end || l.t + 6) + 0.5;
}
function setLines(text) {
  const timed = parseLrc(text).filter(l => l.t != null && l.text).sort((a, b) => a.t - b.t);
  for (let i = 0; i < timed.length; i++) {
    const next = i + 1 < timed.length ? timed[i + 1].t : timed[i].t + 6;
    let end = timed[i].t;
    for (const n of S.notes) if (n.s >= timed[i].t - 0.1 && n.s < next) end = Math.max(end, n.e);
    timed[i].end = end > timed[i].t + 0.3 ? end : Math.min(next, timed[i].t + 4);
  }
  for (const l of timed) if (l.chars) fillCharTimes(l);
  S.lines = timed;
  // 沒有逐字時間的句子先自動分配（只用於顯示，不寫回歌詞，除非使用者調整或按「自動分配」）
  S.lines.forEach((l, i) => { if (!l.chars) { l.chars = distributeChars(l, lineLimit(i)); l.autoChars = true; } });
  S.dirty = true;
  renderTiming();
}
/* 把畫面上的歌詞（含逐字時間）寫回文字框；沒有時間的句子保留在最後 */
function commitLines(save) {
  const untimed = parseLrc($("#lrc").value).filter(l => l.t == null);
  S.lines.sort((a, b) => a.t - b.t);
  $("#lrc").value = toLrc(S.lines.concat(untimed));
  renderTiming(); S.dirty = true;
  if (save) saveLyricsSoon();
}
let saveTimer = null;
function saveLyricsSoon() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(async () => {
    try { await api(`/api/songs/${S.song.id}/lyrics`, { method: "PUT", json: { lrc: $("#lrc").value } }); toast("歌詞已儲存", 1500); }
    catch (e) { toast(e.message); }
  }, 700);
}
$("#lrc").oninput = e => setLines(e.target.value);
$("#timing-box").addEventListener("toggle", () => { if ($("#timing-box").open) renderTiming(); });
$("#lrc-save").onclick = async () => {
  try { await api(`/api/songs/${S.song.id}/lyrics`, { method: "PUT", json: { lrc: $("#lrc").value } }); toast("歌詞已儲存"); }
  catch (e) { toast(e.message); }
};
$("#lrc-auto").onclick = () => {
  if (!S.autoLrc) return toast("這首歌沒有自動辨識的歌詞");
  $("#lrc").value = S.autoLrc; setLines(S.autoLrc); toast("已還原，按「儲存歌詞」才會保存");
};

/* 匯入歌詞檔（LRC／SRT／TXT），自動判斷 UTF-8、Big5、GBK 編碼 */
$("#lrc-import").onclick = () => { if (!S.song) return; $("#lrc-file").value = ""; $("#lrc-file").click(); };
$("#lrc-file").onchange = async e => {
  const f = e.target.files[0];
  if (!f) return;
  if (f.size > 2e6) return toast("檔案太大，不像是歌詞檔");
  const buf = await f.arrayBuffer();
  let text = new TextDecoder("utf-8").decode(buf);
  if (text.includes("\ufffd")) {
    // 不是 UTF-8：試繁體（Big5）和簡體（GBK），取亂碼最少的
    const tries = ["big5", "gbk"].map(enc => { try { return new TextDecoder(enc).decode(buf); } catch { return ""; } });
    const bad = t => (t ? (t.match(/\ufffd/g) || []).length + (t.match(/[\u4e00-\u9fff]/g) ? 0 : 50) : 1e9);
    text = [text, ...tries].sort((a, b) => bad(a) - bad(b))[0];
  }
  try {
    const r = await api(`/api/songs/${S.song.id}/lyrics/import`, { json: { text, name: f.name } });
    $("#lrc").value = r.lrc; setLines(r.lrc); S.dirty = true;
    $("#cand-box").hidden = true;
    toast(`已匯入「${f.name}」：${r.how}`, 5000);
  } catch (err) { toast(err.message); }
};

/* 旋律與歌詞區的高度比例（拖曳分隔條，記在這台電腦） */
(() => {
  const wrap = $(".roll-wrap"), bar = $("#split");
  const apply = h => { if (h) wrap.style.height = clamp(h, 160, window.innerHeight * 0.8) + "px"; else wrap.style.height = ""; S.dirty = true; };
  apply(store.get("rollH", 0));
  let drag = null;
  bar.addEventListener("pointerdown", e => { drag = { y: e.clientY, h: wrap.getBoundingClientRect().height }; bar.setPointerCapture(e.pointerId); bar.classList.add("drag"); });
  bar.addEventListener("pointermove", e => { if (drag) apply(drag.h + e.clientY - drag.y); });
  const end = () => { if (!drag) return; drag = null; bar.classList.remove("drag"); store.set("rollH", Math.round(wrap.getBoundingClientRect().height)); };
  bar.addEventListener("pointerup", end); bar.addEventListener("pointercancel", end);
  bar.addEventListener("dblclick", () => { apply(0); store.set("rollH", 0); });
  bar.addEventListener("keydown", e => {
    const d = e.key === "ArrowUp" ? -20 : e.key === "ArrowDown" ? 20 : 0;
    if (!d) return;
    e.preventDefault(); apply(wrap.getBoundingClientRect().height + d); store.set("rollH", Math.round(wrap.getBoundingClientRect().height));
  });
})();

/* 旋律顯示方式、歌詞字級（記在這台電腦） */
S.melMode = store.get("melMode", "both");
S.lyrSize = store.get("lyrSize", 18);
S.lyrScale = store.get("lyrScale", 1);
$("#mel-mode").value = S.melMode;
$("#mel-mode").onchange = e => { S.melMode = e.target.value; store.set("melMode", S.melMode); S.dirty = true; };
$("#lyr-size").value = S.lyrSize;
$("#lyr-size").oninput = e => { S.lyrSize = +e.target.value; store.set("lyrSize", S.lyrSize); if (!S.showLyr) { S.showLyr = true; store.set("showLyr", true); syncLyrButtons(); } S.dirty = true; };
function applyLyrScale() { $("#lyr").style.setProperty("--ls", S.lyrScale); store.set("lyrScale", S.lyrScale); }
$("#lyr-smaller").onclick = () => { S.lyrScale = Math.max(0.7, +(S.lyrScale - 0.1).toFixed(1)); applyLyrScale(); };
$("#lyr-bigger").onclick = () => { S.lyrScale = Math.min(2, +(S.lyrScale + 0.1).toFixed(1)); applyLyrScale(); };
applyLyrScale();

/* 歌詞顯示／調整開關（記在這台電腦） */
S.showLyr = store.get("showLyr", true);
S.editLyr = false;
function syncLyrButtons() {
  $("#lyr-show").classList.toggle("on", S.showLyr);
  $("#lyr-edit").classList.toggle("on", S.editLyr);
  $("#edit-hint").hidden = !S.editLyr;
  $("#roll").classList.toggle("editing", S.editLyr);
  S.dirty = true;
}
$("#lyr-show").onclick = () => { S.showLyr = !S.showLyr; if (!S.showLyr) S.editLyr = false; store.set("showLyr", S.showLyr); syncLyrButtons(); };
$("#lyr-edit").onclick = () => { S.editLyr = !S.editLyr; if (S.editLyr) { S.showLyr = true; store.set("showLyr", true); if (S.playing) pause(); } syncLyrButtons(); };
syncLyrButtons();

/* 逐句調整開始時間 */
function parseTimeInput(v) {
  const m = String(v).trim().match(/^(?:(\d+):)?(\d+(?:\.\d+)?)$/);
  return m ? (+(m[1] || 0)) * 60 + +m[2] : null;
}
function shiftLine(l, nt) {
  const i = S.lines.indexOf(l);
  nt = Math.max(0, Math.round(nt * 100) / 100);
  const d = nt - l.t;
  l.t = nt;
  if (l.chars) l.chars.forEach(u => (u.t = Math.round((u.t + d) * 100) / 100));
  setLineEnds(); commitLines(false); $("#timing-box").dataset.dirty = "1";
}
function renderTiming() {
  const box = $("#timing");
  if (!box || $("#timing-box").open === false && box.childElementCount) { /* 收合時不必重繪 */ }
  box.innerHTML = "";
  if (!S.lines.length) { box.innerHTML = `<span class="hint">還沒有含時間的歌詞。</span>`; return; }
  S.lines.forEach((l, i) => {
    const row = document.createElement("div");
    row.className = "trow"; row.dataset.i = i;
    row.innerHTML = `<input type="text" aria-label="開始時間"><span class="btns">
      <button data-a="play" title="從這句前 1 秒播放">▶</button>
      <button data-a="now" title="把這句的開始對準目前播放位置">對準</button>
      <button data-a="m" title="提早 0.1 秒">−</button><button data-a="p" title="延後 0.1 秒">＋</button></span><span class="tx"></span>`;
    const inp = $("input", row); inp.value = lrcTime(l.t);
    $(".tx", row).textContent = l.text;
    inp.onchange = () => { const v = parseTimeInput(inp.value); if (v == null) { inp.value = lrcTime(l.t); return toast("時間格式是 分:秒，例如 01:23.45"); } shiftLine(l, v); };
    row.onclick = ev => {
      const a = ev.target.dataset?.a; if (!a) return;
      if (a === "play") { seek(Math.max(0, l.t - 1)); if (!S.playing) play(); }
      if (a === "now") shiftLine(l, cur() - 0.08 * S.speed);
      if (a === "m") shiftLine(l, l.t - 0.1);
      if (a === "p") shiftLine(l, l.t + 0.1);
    };
    box.appendChild(row);
  });
}
function highlightTiming(t) {
  const box = $("#timing"); if (!box || !$("#timing-box").open) return;
  let k = -1; for (let i = 0; i < S.lines.length; i++) if (S.lines[i].t <= t + 0.05) k = i;
  if (box._cur === k) return; box._cur = k;
  $$(".trow", box).forEach(r => r.classList.toggle("cur", +r.dataset.i === k));
}
$("#lyr-dist").onclick = () => {
  if (!S.lines.length) return toast("先要有含時間的歌詞");
  S.lines.forEach((l, i) => { l.chars = distributeChars({ ...l, chars: null }, lineLimit(i)); l.autoChars = false; });
  setLineEnds(); commitLines(true);
  if (!S.showLyr) { S.showLyr = true; store.set("showLyr", true); syncLyrButtons(); }
  toast("已依旋律分配每個字的時間，可以按「調整」拖曳微調");
};

/* 線上歌詞：候選清單（自動比對的結果與手動搜尋共用） */
function renderCandidates(box, results) {
  box.innerHTML = "";
  for (const it of results) {
    const el = document.createElement("div");
    el.className = "ls-item";
    const sim = it.similarity;
    const simCls = sim == null ? "lo" : sim >= 0.6 ? "hi" : sim >= 0.35 ? "mid" : "lo";
    el.innerHTML = `<div class="h"><b></b><span class="hint"></span>
        <span class="sim ${simCls}">${sim == null ? "無法比對" : "相似度 " + Math.round(sim * 100) + "%"}</span>
        ${it.synced ? '<span class="chip ok">有時間軸</span>' : '<span class="chip">純文字</span>'}</div>
      <div class="pv"></div><div class="row"><span class="hint src"></span><button class="primary">使用這版</button></div>`;
    $("b", el).textContent = it.track || "（未命名）";
    const dur = it.duration ? fmt(it.duration, 0) : "?";
    $(".h .hint", el).textContent = `${it.artist || "未知歌手"}・${dur}${it.diff != null ? `（長度差 ${it.diff} 秒）` : ""}`;
    $(".pv", el).textContent = it.preview || (it.plain || "").split(/\n/).filter(Boolean).slice(0, 3).join("\n");
    $(".src", el).textContent = `來源：${it.source}・共 ${it.lines || "?"} 行`;
    $("button.primary", el).onclick = () => applyCandidate(it);
    box.appendChild(el);
  }
}
async function applyCandidate(it) {
  try {
    const r = await api(`/api/songs/${S.song.id}/lyrics/apply`, { json: { candidate: it } });
    $("#lrc").value = r.lrc; setLines(r.lrc);
    $("#cand-box").hidden = true;
    $("#ref-lyrics").value = (it.plain || "").split(/\r?\n/).map(x => x.trim()).filter(Boolean).join("\n");
    toast(`已套用「${it.track}」，${r.how}。時間不準可以用「逐句調整」或 Claude 對齊`, 5000);
    $("#timing-box").open = true;
  } catch (e) { toast(e.message); }
}
async function checkCandidates(id) {
  $("#cand-box").hidden = true;
  try {
    const r = await api(`/api/songs/${id}/lyrics/candidates`);
    if (!S.song || S.song.id !== id || !r.pending || !r.results.length) return;
    renderCandidates($("#cand-list"), r.results);
    $("#cand-box").hidden = false;
    toast("找到線上歌詞，在右下的歌詞面板選一版使用，或保留辨識結果", 5000);
  } catch {}
}
$("#cand-keep").onclick = async () => {
  try { await api(`/api/songs/${S.song.id}/lyrics/keep`, { json: {} }); } catch {}
  $("#cand-box").hidden = true; toast("保留辨識結果");
};
async function searchLyrics(q) {
  const msg = $("#ls-msg"), box = $("#ls-results");
  msg.textContent = "搜尋中（LRCLIB、網易雲、酷狗，原文與簡體各搜一次）…"; box.innerHTML = "";
  try {
    const r = await api(`/api/songs/${S.song.id}/lyrics/search?q=${encodeURIComponent(q || "")}`);
    if (!q) $("#ls-q").value = r.query;
    if (!r.results.length) { msg.textContent = `找不到「${r.queries.join("」、「")}」的歌詞，可以改用歌名＋歌手再搜尋。`; return; }
    msg.textContent = `找到 ${r.results.length} 筆，依和辨識結果的相似度排序：`;
    renderCandidates(box, r.results);
  } catch (e) { msg.textContent = e.message; }
}
$("#ls-go").onclick = () => searchLyrics($("#ls-q").value.trim());
$("#ls-q").addEventListener("keydown", e => { if (e.key === "Enter") { e.preventDefault(); searchLyrics($("#ls-q").value.trim()); } });

/* 跟拍打點 */
$("#tap-start").onclick = () => {
  const lines = parseLrc($("#lrc").value).map(l => l.text).filter(Boolean);
  if (!lines.length) return toast("先在上面貼上歌詞，每行一句");
  S.tap = { lines, i: 0, out: [] };
  $("#tapbox").hidden = false;
  $("#tap-next").textContent = lines[0];
  seek(S.loopA ?? 0); if (!S.playing) play();
};
function tapHit() {
  if (!S.tap) return;
  const t = Math.max(0, cur() - 0.12 * S.speed);
  S.tap.out.push({ t, text: S.tap.lines[S.tap.i] });
  S.tap.i++;
  const merged = S.tap.out.concat(S.tap.lines.slice(S.tap.i).map(text => ({ t: null, text })));
  $("#lrc").value = toLrc(merged); setLines($("#lrc").value);
  if (S.tap.i >= S.tap.lines.length) { stopTap(); $("#lrc-save").click(); }
  else $("#tap-next").textContent = S.tap.lines[S.tap.i];
}
function stopTap() { S.tap = null; $("#tapbox").hidden = true; }
$("#tap-hit").onclick = tapHit;
$("#tap-stop").onclick = stopTap;

$("#align").onclick = async () => {
  const text = $("#ref-lyrics").value.trim();
  if (!text) return toast("先貼上正確的歌詞");
  const st = $("#align-state"); const b = $("#align");
  b.disabled = true; st.textContent = "Claude 對齊中…";
  try {
    const r = await api(`/api/songs/${S.song.id}/claude/align`, { json: { text } });
    $("#lrc").value = r.lrc; setLines(r.lrc);
    S.lines.forEach((l, i) => { l.chars = distributeChars({ ...l, chars: null }, lineLimit(i)); l.autoChars = false; });
    setLineEnds(); commitLines(true);
    st.textContent = "對齊完成，已儲存。每個字也依旋律放好了，可以按上方「調整」微調";
    if (!S.showLyr) { S.showLyr = true; store.set("showLyr", true); syncLyrButtons(); }
    $("#align-cost").textContent = usageLine(r.usage); pollJobs();
  } catch (e) { st.textContent = e.message; }
  finally { b.disabled = false; }
};

/* 匯出歌詞：存到「下載\雷虎練唱台」 */
let exportedPath = "";
$("#exp-go").onclick = async () => {
  const msg = $("#exp-msg");
  try {
    const r = await api(`/api/songs/${S.song.id}/lyrics/export`, { json: { fmt: $("#exp-fmt").value, lrc: $("#lrc").value } });
    exportedPath = r.path; msg.textContent = `已存到 ${r.path}`; $("#exp-show").hidden = false;
  } catch (e) { msg.textContent = e.message; $("#exp-show").hidden = true; }
};
$("#exp-show").onclick = () => exportedPath && api("/api/reveal", { json: { path: exportedPath } }).catch(e => toast(e.message));

/* ------------------------------------------------------------ 麥克風與即時音高 */
const nsdf = new Float32Array(4096);
function detect(buf, sr) {
  const W = buf.length;
  let e = 0; for (let i = 0; i < W; i++) e += buf[i] * buf[i];
  if (Math.sqrt(e / W) < 0.008) return null;
  const minLag = Math.floor(sr / 1100), maxLag = Math.min(W - 2, Math.ceil(sr / 65));
  for (let t = 0; t <= maxLag + 1; t++) {
    let r = 0, m = 0;
    for (let j = 0, n = W - t; j < n; j++) { const a = buf[j], b = buf[j + t]; r += a * b; m += a * a + b * b; }
    nsdf[t] = m > 0 ? (2 * r) / m : 0;
  }
  let t = 1; while (t < maxLag && nsdf[t] > 0) t++;
  const peaks = []; let best = -1, bt = 0, pos = false;
  for (; t <= maxLag; t++) {
    if (nsdf[t] > 0) { pos = true; if (nsdf[t] > best) { best = nsdf[t]; bt = t; } }
    else if (pos) { peaks.push([bt, best]); pos = false; best = -1; }
  }
  if (!peaks.length) return null;
  const gmax = Math.max(...peaks.map(p => p[1]));
  if (gmax < 0.7) return null;
  const p = peaks.find(p => p[0] >= minLag && p[1] >= 0.88 * gmax);
  if (!p) return null;
  const k = p[0], a = nsdf[k - 1], b = nsdf[k], c = nsdf[k + 1];
  const den = a - 2 * b + c; const lag = k + (den ? (0.5 * (a - c)) / den : 0);
  const hz = sr / lag;
  return hz < 65 || hz > 1100 ? null : 69 + 12 * Math.log2(hz / 440);
}

/* 麥克風：可以選擇輸入裝置，記住上次的選擇 */
async function listMics() {
  const sel = $("#mic-dev");
  let devs = [];
  try { devs = (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === "audioinput"); } catch {}
  const saved = store.get("micId", "");
  sel.innerHTML = `<option value="">系統預設</option>`;
  devs.filter(d => d.deviceId && d.deviceId !== "default" && d.deviceId !== "communications").forEach((d, i) => {
    const o = document.createElement("option");
    o.value = d.deviceId; o.textContent = d.label || `麥克風 ${i + 1}（開啟一次麥克風後會顯示名稱）`;
    sel.appendChild(o);
  });
  sel.value = [...sel.options].some(o => o.value === saved) ? saved : "";
}
navigator.mediaDevices?.addEventListener?.("devicechange", listMics);
listMics();
$("#mic-dev").onchange = async () => {
  store.set("micId", $("#mic-dev").value);
  if (S.mic) { stopMic(); await startMic(); }
};
$("#mic").onclick = () => (S.mic ? stopMic() : startMic());
async function startMic() {
  const id = $("#mic-dev").value;
  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false,
               ...(id ? { deviceId: { exact: id } } : {}) },
    });
    const c = ac(); c.resume();
    const src = c.createMediaStreamSource(stream);
    // 麥克風 → (原始音量表) → 增益 → (音量表、音高) → 限幅器 → 錄音
    const raw = c.createAnalyser(); raw.fftSize = 2048;
    const gain = c.createGain(); gain.gain.value = S.micGain;
    const an = c.createAnalyser(); an.fftSize = 2048;
    const lim = c.createDynamicsCompressor();   // 防爆音：超過 -3 dB 才壓，平常不影響聲音
    lim.threshold.value = -3; lim.knee.value = 0; lim.ratio.value = 20; lim.attack.value = 0.002; lim.release.value = 0.1;
    const dest = c.createMediaStreamDestination();
    src.connect(raw); src.connect(gain); gain.connect(an); gain.connect(lim); lim.connect(dest);
    S.mic = { stream, src, raw, gain, an, lim, dest, buf: new Float32Array(an.fftSize),
              rbuf: new Float32Array(raw.fftSize), tick: 0, peak: 0, peakT: 0, clipRaw: 0, clipGain: 0, auto: null };
    $("#mic-lv").hidden = false;
    $("#mic").textContent = "關閉麥克風"; $("#mic").classList.add("on");
    $("#rec").disabled = false;
    const label = stream.getAudioTracks()[0]?.label || "";
    $("#mic-state").textContent = `麥克風已開啟${label ? "（" + label + "）" : ""}。藍線是你現在的音高。`;
    listMics();   // 取得權限後才看得到裝置名稱
  } catch (e) {
    if (id && e && (e.name === "OverconstrainedError" || e.name === "NotFoundError")) {
      toast("找不到上次選的麥克風，改用系統預設");
      store.set("micId", ""); $("#mic-dev").value = ""; return startMic();
    }
    toast("無法使用麥克風：請確認 Windows 設定裡允許應用程式使用麥克風");
  }
}
function stopMic() {
  if (S.rec) stopRecording();
  S.mic.stream.getTracks().forEach(t => t.stop());
  S.mic.src.disconnect(); S.mic.gain.disconnect(); S.mic.lim.disconnect(); S.mic = null; S.liveNow = null;
  $("#mic-lv").hidden = true;
  $("#mic").textContent = "開啟麥克風"; $("#mic").classList.remove("on");
  $("#rec").disabled = true;
  $("#mic-state").textContent = "戴耳機唱，避免伴奏被麥克風收進去。";
}
function micFrame() {
  if (!S.mic) return;
  if (++S.mic.tick % 2) return;
  S.mic.an.getFloatTimeDomainData(S.mic.buf);
  micLevel();
  const m = detect(S.mic.buf, ac().sampleRate);
  S.liveNow = m;
  if (S.playing) {
    const t = cur() - (S.mic.buf.length / 2 / ac().sampleRate) * S.speed;
    S.live.push({ t, m });
    if (S.live.length > 4000) S.live.splice(0, 1000);
  }
  S.dirty = true;
}

/* 麥克風音量表、增益、自動調整（1.2） */
S.micGain = store.get("micGain", 1);
const toDb = v => (v > 1e-5 ? 20 * Math.log10(v) : -120);
function syncGainUi() {
  $("#mic-gain").value = Math.round(S.micGain * 100);
  $("#mic-gain-v").textContent = Math.round(S.micGain * 100) + "%";
  if (S.mic) S.mic.gain.gain.setTargetAtTime(S.micGain, ac().currentTime, 0.02);
}
function setMicGain(g) { S.micGain = clamp(g, 0.1, 3); store.set("micGain", S.micGain); syncGainUi(); }
$("#mic-gain").oninput = e => setMicGain(e.target.value / 100);
syncGainUi();
function micLevel() {
  const M = S.mic, now = performance.now();
  let pk = 0, sum = 0;
  for (const v of M.buf) { const a = Math.abs(v); if (a > pk) pk = a; sum += v * v; }
  M.raw.getFloatTimeDomainData(M.rbuf);
  let rpk = 0; for (const v of M.rbuf) { const a = Math.abs(v); if (a > rpk) rpk = a; }
  const db = toDb(pk), rms = toDb(Math.sqrt(sum / M.buf.length));
  if (pk >= M.peak || now - M.peakT > 1500) { M.peak = pk; M.peakT = now; }
  const pos = d => clamp((d + 60) / 60, 0, 1) * 100;
  $("#mic-bar").style.setProperty("--v", pos(db).toFixed(1) + "%");
  $("#mic-pk").style.left = `calc(${pos(toDb(M.peak)).toFixed(1)}% - 1px)`;
  $("#mic-db").textContent = db < -59 ? "-∞ dB" : `${db.toFixed(0)} dB`;
  if (rpk >= 0.985) M.clipRaw = now;
  if (pk >= 0.985) M.clipGain = now;
  if (M.auto) autoStep(pk, rms);
  const w = $("#mic-warn");
  let msg = "", cls = "";
  if (now - M.clipRaw < 2500) { msg = "麥克風本身收音過大（爆音）：請到 Windows 的音效設定把麥克風音量調低，或離麥克風遠一點"; cls = "bad"; }
  else if (now - M.clipGain < 2500) { msg = "音量太大，已自動壓住避免爆音；建議把增益調低一點，評分會更準"; cls = "bad"; }
  else if (M.auto) { msg = "自動調整中：請用平常練唱的音量唱幾句…"; cls = "ok"; }
  if (w._m !== msg) { w.textContent = msg; w.className = "hint " + cls; w._m = msg; }
}
// 自動調整：聽約 4 秒有聲音的片段，讓最大音量落在 -6 dB 左右
$("#mic-auto").onclick = () => {
  if (!S.mic) return toast("先開啟麥克風");
  S.mic.auto = { peaks: [], t0: performance.now() };
};
function autoStep(pk) {
  const A = S.mic.auto;
  if (pk > 0.02) A.peaks.push(pk / S.micGain);   // 換算回增益前的音量
  if (A.peaks.length >= 70 || performance.now() - A.t0 > 12000) {
    S.mic.auto = null;
    if (A.peaks.length < 15) return toast("沒聽到聲音，請靠近麥克風再試一次");
    A.peaks.sort((a, b) => a - b);
    const ref = A.peaks[Math.floor(A.peaks.length * 0.95)];
    setMicGain(0.5 / ref);
    toast(`已把增益調成 ${Math.round(S.micGain * 100)}%`);
  }
}

/* 錄音 */
$("#rec").onclick = () => (S.rec ? stopRecording() : startRecording());
function startRecording() {
  if (!S.mic) return;
  if (!S.playing) { if (S.loopA != null) S.t = S.loopA; S.live = []; play(); }
  const mr = new MediaRecorder(S.mic.dest.stream);   // 錄的是調過增益、防爆音後的聲音
  const chunks = [];
  mr.ondataavailable = e => e.data.size && chunks.push(e.data);
  S.rec = { mr, chunks, start: cur() - 0.06 * S.speed, key: S.key, speed: S.speed };
  mr.onstop = () => uploadTake(S.recDone);
  mr.start();
  $("#rec").textContent = "■ 停止"; $("#rec").classList.add("live");
}
function stopRecording() {
  if (!S.rec) return;
  S.recDone = S.rec; S.rec = null;
  S.recDone.mr.stop();
  $("#rec").textContent = "● 錄一次"; $("#rec").classList.remove("live");
  if (S.playing) pause();
}
async function uploadTake(r) {
  try {
    const blob = new Blob(r.chunks, { type: r.mr.mimeType });
    const buf = await ac().decodeAudioData(await blob.arrayBuffer());
    if (buf.duration < 2) return toast("錄音太短，沒有存");
    const wav = encodeWav(buf.getChannelData(0), buf.sampleRate);
    const fd = new FormData();
    fd.append("file", new Blob([wav], { type: "audio/wav" }), "take.wav");
    fd.append("start", r.start.toFixed(3)); fd.append("key", r.key);
    fd.append("speed", r.speed); fd.append("duration", buf.duration.toFixed(2));
    toast("錄音已上傳，正在評分…");
    const res = await api(`/api/songs/${S.song.id}/takes`, { method: "POST", body: fd });
    await refreshTakes();
    await waitJob(res.job);
    await refreshTakes(res.id);
  } catch (e) { toast(e.message); }
}
function encodeWav(x, sr) {
  const n = x.length, buf = new ArrayBuffer(44 + n * 2), v = new DataView(buf);
  const w = (o, s) => [...s].forEach((ch, i) => v.setUint8(o + i, ch.charCodeAt(0)));
  w(0, "RIFF"); v.setUint32(4, 36 + n * 2, true); w(8, "WAVE"); w(12, "fmt ");
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  w(36, "data"); v.setUint32(40, n * 2, true);
  for (let i = 0; i < n; i++) v.setInt16(44 + i * 2, clamp(x[i], -1, 1) * 32767, true);
  return buf;
}

/* 錄音列表與評分 */
async function refreshTakes(selectId) {
  if (!S.song) return;
  const song = await api(`/api/songs/${S.song.id}`);
  renderTakes(song.takes);
  if (selectId) selectTake(selectId);
}
function renderTakes(takes) {
  S.takes = takes || [];
  const box = $("#takes"); box.innerHTML = "";
  if (!S.takes.length) { box.innerHTML = `<p class="hint">還沒有錄音。開啟麥克風後按「錄一次」，唱完按停止就會自動評分。</p>`; return; }
  for (const t of S.takes) {
    const el = document.createElement("div");
    el.className = "take" + (S.take && S.take.id === t.id ? " sel" : "");
    const d = new Date(t.created * 1000);
    const when = `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, "0")}:${String(d.getMinutes()).padStart(2, "0")}`;
    el.innerHTML = `<div><div class="sub">${when}・KEY ${t.key > 0 ? "+" : ""}${t.key}・${Math.round(t.speed * 100)}%・${fmt(t.duration || 0, 0)}</div>
      <div class="sub">${t.acc == null ? "評分中…" : `平均偏差 ${t.mean > 0 ? "+" : ""}${Math.round(t.mean)}¢${t.octave ? `・${t.octave < 0 ? "低" : "高"}八度` : ""}`}</div></div>
      <div class="score">${t.acc == null ? "—" : Math.round(t.acc * 100) + "%"}</div>`;
    el.onclick = () => selectTake(t.id);
    box.appendChild(el);
  }
}
async function rescoreTake(id, quiet) {
  try {
    if (!quiet) toast("重新分析這次錄音…");
    const r = await api(`/api/songs/${S.song.id}/takes/${id}/rescore`, { json: {} });
    await waitJob(r.job);
    await refreshTakes(id);
  } catch (e) { toast(e.message); }
}
async function selectTake(id) {
  const t = await api(`/api/songs/${S.song.id}/takes/${id}`);
  if (t.status === "done" && (t.algo || 1) < 2 && !S.rescored?.[id]) {
    // 1.2 以前的錄音：用新的分析方式（去掉音頭音尾往下掉的雜訊）自動重新評分一次
    (S.rescored ||= {})[id] = true;
    toast("用新的分析方式重新評分這次錄音…");
    return rescoreTake(id, true);
  }
  S.take = t; S.takeBuf = null;
  if (t.pitch) t.pitch.midiArr = Float32Array.from(t.pitch.midi, v => (v == null ? NaN : v));
  renderTakes(S.takes);
  renderTakeDetail();
  if (S.vol.take === 0) { S.vol.take = 0.9; renderFaders(); }
  try {
    const r = await fetch(`/api/songs/${S.song.id}/takes/${id}/audio`);
    S.takeBuf = await ac().decodeAudioData(await r.arrayBuffer());
  } catch {}
  if (t.status === "done" && Math.abs(t.speed - S.speed) > 1e-3) toast(`這次錄音用 ${Math.round(t.speed * 100)}% 速度，切換到相同速度才能一起播放`);
  if (t.pitch && !S.playing) seek(Math.max(0, t.pitch.t0 - 1));
  S.dirty = true;
}
function renderTakeDetail() {
  const t = S.take, box = $("#take-detail");
  if (!t) { box.innerHTML = ""; return; }
  if (t.acc == null) { box.innerHTML = `<p class="hint">評分中…</p>`; return; }
  const worst = (t.lines || []).slice().sort((a, b) => a.acc - b.acc).slice(0, 3);
  box.innerHTML = `<div class="hint">命中率是落在原唱 ±50 音分內的比例。已自動對齊 ${t.offset > 0 ? "+" : ""}${t.offset}s 的錄音延遲${t.octave ? "，並換算八度" : ""}。</div>
    ${worst.length ? `<b>最需要練的句子</b>` : ""}
    <div class="wl"></div>
    <div class="row"><button id="coach">請 Claude 給練習建議</button><button id="rescore" class="ghost">重新評分</button><button id="del-take" class="ghost danger">刪除這次錄音</button></div>
    <div class="coach" ${t.coach ? "" : "hidden"}></div>`;
  const wl = $(".wl", box);
  for (const l of worst) {
    const row = document.createElement("div"); row.className = "line";
    row.innerHTML = `<button class="ghost" style="justify-content:flex-start"></button><span class="num">${Math.round(l.acc * 100)}%・${l.mean > 0 ? "+" : ""}${Math.round(l.mean)}¢</span>`;
    $("button", row).textContent = `${fmt(l.t, 0)}　${l.text}`;
    $("button", row).onclick = () => seek(Math.max(0, l.t - 1));
    wl.appendChild(row);
  }
  $(".coach", box).textContent = t.coach || "";
  $("#coach").onclick = async () => {
    const b = $("#coach"), out = $(".coach", box);
    b.disabled = true; out.hidden = false; out.textContent = "Claude 分析中…";
    try { const r = await api(`/api/songs/${S.song.id}/claude/coach`, { json: { take: t.id } }); t.coach = r.text; out.textContent = r.text + (r.usage ? "\n\n" + usageLine(r.usage) : ""); pollJobs(); }
    catch (e) { out.textContent = e.message; }
    finally { b.disabled = false; }
  };
  $("#rescore").onclick = () => rescoreTake(t.id);
  const del = $("#del-take");
  del.onclick = async () => {
    if (!del.dataset.armed) { del.dataset.armed = "1"; del.textContent = "確定刪除？"; setTimeout(() => { delete del.dataset.armed; del.textContent = "刪除這次錄音"; }, 3000); return; }
    await api(`/api/songs/${S.song.id}/takes/${t.id}`, { method: "DELETE" });
    S.take = null; S.takeBuf = null; box.innerHTML = ""; refreshTakes();
  };
}

/* ------------------------------------------------------------ 繪圖 */
const cv = $("#roll");
let colors = {};
function readColors() {
  const cs = getComputedStyle(document.documentElement);
  for (const k of ["bg", "panel", "ink", "muted", "line", "lane", "lane-k", "melody", "voice", "good", "bad", "accent", "shade"]) colors[k] = cs.getPropertyValue(`--${k}`).trim();
}
readColors();
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", () => { readColors(); S.dirty = true; });

const KEYW = 40, TOP = 22;
function geom() {
  const W = cv.clientWidth, H = cv.clientHeight;
  const span = S.zoom, t = cur();
  const left = t - span * 0.3;
  const [lo, hi] = S.range;   // 整首歌固定的音域，不隨播放上下移動
  const rowH = (H - TOP) / (hi - lo + 1);
  return {
    W, H, t, left, lo, hi, rowH,
    x: tt => KEYW + ((tt - left) / span) * (W - KEYW),
    tx: x => left + ((x - KEYW) / (W - KEYW)) * span,
    y: m => TOP + (hi - m + 0.5) * rowH,
  };
}
function refAt(t) {
  if (!S.pitch) return NaN;
  const i = Math.round(t / S.pitch.hop);
  return i >= 0 && i < S.pitch.midi.length ? S.pitch.midi[i] + S.key : NaN;
}
function fold(d) { return d - 12 * Math.round(d / 12); }

function draw() {
  const dpr = window.devicePixelRatio || 1;
  const W = cv.clientWidth, H = cv.clientHeight;
  if (cv.width !== Math.round(W * dpr) || cv.height !== Math.round(H * dpr)) { cv.width = Math.round(W * dpr); cv.height = Math.round(H * dpr); }
  const g = cv.getContext("2d");
  g.setTransform(dpr, 0, 0, dpr, 0, 0);
  const G = geom(), C = colors;
  g.fillStyle = C.panel; g.fillRect(0, 0, W, H);

  // 琴鍵列
  for (let m = Math.floor(G.lo) - 1; m <= Math.ceil(G.hi) + 1; m++) {
    const y = G.y(m) - G.rowH / 2;
    const pc = ((m % 12) + 12) % 12;
    const black = [1, 3, 6, 8, 10].includes(pc);
    g.fillStyle = black ? C["lane-k"] : C.lane;
    g.fillRect(KEYW, y, W - KEYW, G.rowH + 0.5);
    if (pc !== 0 && !black && G.rowH >= 15) {
      g.fillStyle = C.muted; g.globalAlpha = 0.55; g.font = "10px ui-monospace, Consolas, monospace"; g.textBaseline = "middle";
      g.fillText(noteName(m), 8, G.y(m)); g.globalAlpha = 1;
    }
    if (pc === 0) {
      g.fillStyle = C.line; g.fillRect(KEYW, y + G.rowH - 1, W - KEYW, 1);
      g.fillStyle = C.muted; g.font = "11px ui-monospace, Consolas, monospace"; g.textBaseline = "middle";
      g.fillText(noteName(m), 6, G.y(m));
    }
  }
  g.fillStyle = C.panel; g.fillRect(0, 0, KEYW, TOP);
  g.fillStyle = C.line; g.fillRect(KEYW - 1, TOP, 1, H - TOP);

  // 卷軸區域裁切，避免音符蓋到琴鍵欄
  g.save(); g.beginPath(); g.rect(KEYW, TOP, W - KEYW, H - TOP); g.clip();

  // 時間格線
  const t0 = Math.ceil(G.left), t1 = G.left + S.zoom;
  g.font = "10px ui-monospace, Consolas, monospace"; g.textBaseline = "alphabetic";
  for (let s = Math.max(0, t0); s <= t1; s++) {
    const x = G.x(s);
    g.fillStyle = C.shade; g.fillRect(x, TOP, 1, H - TOP);
    if (s % 5 === 0) { g.fillStyle = C.muted; g.fillText(fmt(s, 0), x + 3, H - 4); }
  }

  // A–B 循環
  if (S.loopA != null) {
    const xa = G.x(S.loopA), xb = S.loopB != null ? G.x(S.loopB) : G.x(cur());
    g.fillStyle = C.melody; g.globalAlpha = 0.08; g.fillRect(xa, TOP, xb - xa, H - TOP); g.globalAlpha = 1;
    g.fillStyle = C.melody; g.fillRect(xa, TOP, 1.5, H - TOP);
    if (S.loopB != null) g.fillRect(xb, TOP, 1.5, H - TOP);
  }

  // 原唱音符（區塊）
  const blocks = S.melMode !== "curve", curveOnly = S.melMode === "curve", blocksOnly = S.melMode === "blocks";
  if (blocks) for (const n of S.notes) {
    if (n.e < G.left || n.s > t1) continue;
    const m = n.m + S.key, x0 = G.x(n.s), x1 = G.x(n.e), y = G.y(m);
    const active = G.t >= n.s && G.t < n.e;
    g.fillStyle = C.melody; g.globalAlpha = blocksOnly ? (active ? 0.95 : 0.6) : (active ? 0.5 : 0.22);
    roundRect(g, x0, y - G.rowH * 0.42, Math.max(2, x1 - x0 - (blocksOnly ? 2 : 0)), G.rowH * 0.84, 3); g.fill();
    g.globalAlpha = 1;
    if (x1 - x0 > (blocksOnly ? 20 : 30) && G.rowH > 11) {
      g.fillStyle = blocksOnly ? C.panel : C.ink; g.font = `${blocksOnly ? "600 " : ""}10px ui-monospace, Consolas, monospace`; g.textBaseline = "middle";
      g.fillText(noteName(m), x0 + 4, y);
    }
  }

  // 原唱音高曲線（原聲）
  if (S.pitch && !blocksOnly) {
    const hop = S.pitch.hop, a = Math.max(0, Math.floor(G.left / hop)), b = Math.min(S.pitch.midi.length - 1, Math.ceil(t1 / hop));
    g.strokeStyle = C.melody; g.lineWidth = curveOnly ? 2.6 : 1.6; g.beginPath();
    let pen = false;
    for (let i = a; i <= b; i++) {
      const v = S.pitch.midi[i];
      if (isNaN(v)) { pen = false; continue; }
      const x = G.x(i * hop), y = G.y(v + S.key);
      pen ? g.lineTo(x, y) : g.moveTo(x, y); pen = true;
    }
    g.stroke();
  }

  // 選取的錄音
  if (S.take && S.take.pitch && S.take.pitch.midiArr) {
    const p = S.take.pitch, arr = p.midiArr;
    const a = Math.max(0, Math.floor((G.left - p.t0) / p.hop)), b = Math.min(arr.length - 1, Math.ceil((t1 - p.t0) / p.hop));
    drawVoice(g, G, i => p.t0 + i * p.hop, i => arr[i], a, b, 1.8, 0.75);
  }
  // 即時麥克風
  if (S.live.length) drawVoice(g, G, i => S.live[i].t, i => S.live[i].m ?? NaN, 0, S.live.length - 1, 2.6, 1, true);

  // 旋律上的逐字歌詞
  S.charHits = [];
  if (S.showLyr) drawCharLyrics(g, G, t1);

  g.restore();

  // 歌詞標記
  g.fillStyle = C.panel; g.fillRect(KEYW, 0, W - KEYW, TOP);
  g.fillStyle = C.line; g.fillRect(KEYW, TOP - 1, W - KEYW, 1);
  g.font = "12px " + getComputedStyle(document.body).fontFamily; g.textBaseline = "middle";
  S.lineHits = [];
  for (const l of S.lines) {
    if (l.end < G.left || l.t > t1) continue;
    const x = G.x(l.t);
    S.lineHits.push({ x, line: l });
    g.fillStyle = S.editLyr ? C.melody : C.muted;
    g.fillRect(x - (S.editLyr ? 1 : 0), 2, S.editLyr ? 3 : 1, TOP - 4);
    g.fillStyle = C.muted;
    g.save(); g.beginPath(); g.rect(x, 0, Math.max(0, G.x(l.end) - x + 40), TOP); g.clip();
    g.fillText(l.text, x + 4, TOP / 2); g.restore();
  }

  // 播放線
  const px = G.x(G.t);
  g.fillStyle = C.ink; g.fillRect(px - 0.75, 0, 1.5, H);

  updateHud(G.t);
}
function noteAt(t, near = 0) {
  let best = null, bd = Infinity;
  for (const n of S.notes) {
    if (t >= n.s - 0.04 && t < n.e) return n;
    const d = t < n.s ? n.s - t : t - n.e;
    if (d < bd) { bd = d; best = n; }
  }
  return bd <= near ? best : null;   // 字不在任何音符上時，用附近的音符決定高度
}
function drawCharLyrics(g, G, t1) {
  const C = colors;
  const fs = Math.round(S.lyrSize || 18);
  g.font = `${fs}px ${getComputedStyle(document.documentElement).getPropertyValue("--lyric")}`;
  g.textBaseline = "alphabetic"; g.textAlign = "center"; g.lineJoin = "round";
  const placed = [];
  for (const l of S.lines) {
    if (!l.chars || l.end < G.left - 1 || l.t > t1 + 1) continue;
    l.chars.forEach((u, i) => {
      if (isPunct(u.c) || u.t < G.left - 1 || u.t > t1 + 1) return;
      const n = noteAt(u.t + 0.02, 0.8);
      const m = n ? n.m + S.key : refAt(u.t + 0.05);
      const x = G.x(u.t) + (u.c.length === 1 ? fs * 0.5 : 0);
      let y = isNaN(m) ? TOP + fs + 4 : Math.max(TOP + fs + 2, G.y(m) - G.rowH * 0.42 - 4);
      const half = g.measureText(u.c).width / 2 + 1;
      // 兩個字靠太近會疊在一起：往上錯開，最多疊三層
      for (let k = 0; k < 3 && placed.some(b => x - half < b.x1 && x + half > b.x0 && Math.abs(b.y - y) < fs * 0.9); k++) y -= fs + 2;
      placed.push({ x0: x - half, x1: x + half, y });
      const sung = G.t >= u.t;
      g.lineWidth = 3.5; g.strokeStyle = C.panel; g.strokeText(u.c, x, y);
      g.fillStyle = sung ? C.melody : C.ink; g.fillText(u.c, x, y);
      const w = g.measureText(u.c).width;
      S.charHits.push({ x0: x - w / 2 - 3, x1: x + w / 2 + 3, y0: y - fs, y1: y + 4, line: l, i });
      if (S.editLyr) { g.fillStyle = C.melody; g.fillRect(G.x(u.t) - 0.5, y + 2, 1, 5); }
    });
  }
  g.textAlign = "left";
}
function drawVoice(g, G, tAt, mAt, a, b, width, alpha, liveMode) {
  g.lineWidth = width; g.globalAlpha = alpha; g.lineCap = "round";
  let prev = null;
  for (let i = a; i <= b; i++) {
    const t = tAt(i), m = mAt(i);
    // 沒聲音、中斷太久，或一瞬間跳超過 3 個半音（通常是音頭音尾分析不準）：斷開線，不畫垂直的線
    if (isNaN(m) || (prev && (Math.abs(m - prev.m) > 3 || (liveMode && t - prev.t > 0.12)))) { prev = isNaN(m) ? null : { t, m }; continue; }
    if (prev) {
      const ref = refAt(t);
      let shown = m, ok = true;
      if (!isNaN(ref)) { const d = fold(m - ref); shown = ref + d; ok = Math.abs(d) <= 0.5; }
      const pm = isNaN(refAt(prev.t)) ? prev.m : refAt(prev.t) + fold(prev.m - refAt(prev.t));
      g.strokeStyle = ok ? colors.voice : colors.bad;
      g.beginPath(); g.moveTo(G.x(prev.t), G.y(pm)); g.lineTo(G.x(t), G.y(shown)); g.stroke();
    }
    prev = { t, m };
  }
  g.globalAlpha = 1;
}
function roundRect(g, x, y, w, h, r) {
  g.beginPath(); g.moveTo(x + r, y); g.arcTo(x + w, y, x + w, y + h, r); g.arcTo(x + w, y + h, x, y + h, r);
  g.arcTo(x, y + h, x, y, r); g.arcTo(x, y, x + w, y, r); g.closePath();
}
function updateHud(t) {
  const ref = refAt(t);
  const parts = [`<span>目標</span><b>${isNaN(ref) ? "—" : noteName(ref)}</b>`];
  if (S.mic) {
    const m = S.liveNow;
    let txt = "—";
    if (m != null) {
      if (!isNaN(ref)) { const d = fold(m - ref) * 100; txt = `${noteName(ref + d / 100)} ${d >= 0 ? "+" : ""}${Math.round(d)}¢`; }
      else txt = noteName(m);
    }
    parts.push(`<span>你</span><b>${txt}</b>`);
  }
  const html = parts.join("");
  const hud = $("#hud");
  if (hud._last !== html) { hud.innerHTML = html; hud._last = html; }
}

function updateLyrics(t) {
  // 三行：上一句、正在唱的、下一句。正在唱的那句會一直留到下一句開始，時間不準時也不會跑掉看不到
  let i = -1;
  for (let k = 0; k < S.lines.length; k++) { if (S.lines[k].t <= t + 0.05) i = k; else break; }
  const now = i >= 0 ? S.lines[i] : null;
  const prev = i > 0 ? S.lines[i - 1] : null;
  const next = S.lines[i + 1] || null;
  const el = $("#lyr-now");
  const text = now ? now.text : "";
  if (el._t !== text) { $(".base", el).textContent = text; $(".fill", el).textContent = text; el._t = text; }
  let p = now ? clamp((t - now.t) / Math.max(0.3, now.end - now.t), 0, 1) : 0;
  if (now && now.chars && now.chars.length) {   // 有逐字時間時，跟著每個字變色
    const u = now.chars, n = u.length;
    let k = 0; while (k < n && u[k].t <= t) k++;
    const nextT = k < n ? u[k].t : Math.max(now.end, (u[n - 1].t || now.t) + 0.3);
    const prevT = k > 0 ? u[k - 1].t : now.t;
    p = k === 0 ? 0 : clamp((k - 1 + clamp((t - prevT) / Math.max(0.05, nextT - prevT), 0, 1)) / n, 0, 1);
  }
  el.style.setProperty("--p", (p * 100).toFixed(1) + "%");
  for (const [id, l] of [["#lyr-prev", prev], ["#lyr-next", next]]) {
    const node = $(id), tx = l ? l.text : (id === "#lyr-next" && !now && S.lines[0] ? S.lines[0].text : "");
    if (node._t !== tx) { node.textContent = tx; node._t = tx; }
  }
}

let lastTimeText = "";
function frame() {
  if (S.playing) {
    const t = cur();
    if (S.loopA != null && S.loopB != null && t >= S.loopB) { S.t = S.loopA; S.live = []; play(); }
    else if (t >= S.dur) { pause(); S.t = S.dur; }
    S.dirty = true;
  }
  micFrame();
  if (S.dirty && !$("#view-practice").hidden) {
    S.dirty = false;
    const t = cur();
    draw(); updateLyrics(t); highlightTiming(t);
    const tt = `${fmt(t)} / ${fmt(S.dur || 0, 0)}`;
    if (tt !== lastTimeText) { $("#time").textContent = tt; lastTimeText = tt; }
    if (document.activeElement !== $("#seek")) $("#seek").value = String(t);
  }
  requestAnimationFrame(frame);
}
new ResizeObserver(() => (S.dirty = true)).observe(cv);

/* 卷軸操作：一般模式點一下跳到該時間；調整模式可拖曳句子標記與每個字 */
let drag = null;
function hitTest(x, y) {
  if (!S.editLyr) return null;
  if (y <= TOP + 2) {
    let best = null;
    for (const h of S.lineHits || []) if (Math.abs(h.x - x) < 8 && (!best || Math.abs(h.x - x) < Math.abs(best.x - x))) best = h;
    if (best) return { kind: "line", line: best.line };
  }
  for (const h of S.charHits || []) if (x >= h.x0 && x <= h.x1 && y >= h.y0 && y <= h.y1) return { kind: "char", line: h.line, i: h.i };
  return null;
}
cv.addEventListener("pointerdown", e => {
  const hit = hitTest(e.offsetX, e.offsetY);
  cv.setPointerCapture(e.pointerId);
  if (hit) {
    const G = geom();
    drag = { ...hit, startX: e.offsetX, t0: hit.kind === "line" ? hit.line.t : hit.line.chars[hit.i].t, moved: false, G };
    if (hit.line.autoChars) { hit.line.autoChars = false; }
    return;
  }
  drag = { kind: "seek" };
  seek(geom().tx(e.offsetX));
});
cv.addEventListener("pointermove", e => {
  if (!drag) {
    const h = hitTest(e.offsetX, e.offsetY);
    cv.style.cursor = h ? "ew-resize" : "";
    return;
  }
  if (drag.kind === "seek") { if (!S.playing) seek(geom().tx(e.offsetX)); return; }
  const G = drag.G, dt = ((e.offsetX - drag.startX) / (G.W - KEYW)) * S.zoom;
  if (Math.abs(e.offsetX - drag.startX) > 2) drag.moved = true;
  const l = drag.line, idx = S.lines.indexOf(l);
  if (drag.kind === "line") {
    const prevEnd = idx > 0 ? S.lines[idx - 1].t + 0.1 : 0;
    const nt = Math.max(prevEnd, Math.min(drag.t0 + dt, (S.lines[idx + 1]?.t ?? 1e9) - 0.1));
    const d = nt - l.t;
    l.t = Math.round(nt * 100) / 100;
    if (l.chars) l.chars.forEach(u => (u.t = Math.round((u.t + d) * 100) / 100));
  } else {
    const u = l.chars, i = drag.i;
    let lo = i > 0 ? u[i - 1].t + 0.03 : (idx > 0 ? S.lines[idx - 1].t + 0.1 : 0);
    let hi = i + 1 < u.length ? u[i + 1].t - 0.03 : (S.lines[idx + 1]?.t ?? 1e9) - 0.03;
    u[i].t = Math.round(clamp(drag.t0 + dt, lo, hi) * 100) / 100;
    if (i === 0) l.t = u[0].t;   // 第一個字就是這句的開始
  }
  S.dirty = true;
});
cv.addEventListener("pointerup", () => {
  if (drag && drag.kind !== "seek" && drag.moved) { setLineEnds(); commitLines(true); }
  drag = null;
});
cv.addEventListener("dblclick", e => {
  const hit = hitTest(e.offsetX, e.offsetY);
  if (!hit || hit.kind !== "char") return;
  const box = (S.charHits || []).find(h => h.line === hit.line && h.i === hit.i);
  const inp = $("#char-edit");
  inp.value = hit.line.chars[hit.i].c;
  inp.style.left = Math.max(44, box.x0 - 10) + "px"; inp.style.top = Math.max(24, box.y0 - 6) + "px";
  inp.hidden = false; inp.focus(); inp.select();
  const done = ok => {
    inp.hidden = true; inp.onkeydown = inp.onblur = null;
    const v = inp.value.trim();
    if (!ok || !v || v === hit.line.chars[hit.i].c) return;
    const us = splitUnits(v);
    us.forEach((u, k) => (u.t = hit.line.chars[hit.i].t + k * 0.12));
    us[us.length - 1].sp = hit.line.chars[hit.i].sp;
    hit.line.chars.splice(hit.i, 1, ...us);
    hit.line.text = unitsText(hit.line.chars); hit.line.autoChars = false;
    commitLines(true);
  };
  inp.onkeydown = ev => { if (ev.key === "Enter") done(true); if (ev.key === "Escape") done(false); ev.stopPropagation(); };
  inp.onblur = () => done(true);
});
function setLineEnds() {
  S.lines.sort((a, b) => a.t - b.t);
  S.lines.forEach((l, i) => {
    const next = S.lines[i + 1]?.t ?? l.t + 6;
    let end = l.chars ? l.chars[l.chars.length - 1].t + 0.3 : l.t;
    for (const n of S.notes) if (n.s >= l.t - 0.1 && n.s < next) end = Math.max(end, n.e);
    l.end = Math.min(Math.max(end, l.t + 0.3), next + 0.5);
  });
}

document.addEventListener("keydown", e => {
  if ($("#view-practice").hidden || /INPUT|TEXTAREA|SELECT/.test(document.activeElement?.tagName || "")) return;
  if (e.code === "Space") { e.preventDefault(); S.tap ? tapHit() : (S.playing ? pause() : play()); }
  else if (e.code === "ArrowLeft") seek(cur() - 3);
  else if (e.code === "ArrowRight") seek(cur() + 3);
});

/* ------------------------------------------------------------ 設定 */
async function openSettings(focusUsage) {
  const [cfg, st] = await Promise.all([api("/api/config"), api("/api/status")]);
  $("#cfg-balance").value = cfg.balance_usd == null ? "" : cfg.balance_usd;
  renderUsage().then(() => { if (focusUsage === true) $("#usage").scrollIntoView({ block: "center" }); });
  $("#cfg-key").value = ""; $("#cfg-key").placeholder = cfg.has_key ? `已設定 ${cfg.anthropic_api_key || "（環境變數）"}` : "sk-ant-…";
  $("#cfg-model").value = cfg.claude_model;
  $("#cfg-whisper").value = cfg.whisper_model;
  $("#cfg-lang").value = cfg.lyrics_language || "";
  $("#cfg-tw").checked = !!cfg.convert_to_taiwan;
  $("#cfg-sep").value = cfg.separation_model;
  $("#cfg-kara").value = cfg.karaoke_model;
  $("#cfg-pitch").value = cfg.pitch_model;
  const m = st.modules;
  const items = [["uvr", "AI 人聲分離"], ["crepe", "CREPE 音高"], ["whisper", "Whisper 歌詞辨識"], ["pedalboard", "高品質移調"], ["opencc", "台灣繁體轉換"], ["anthropic", "Claude 連線"], ["cuda", "顯示卡加速"]];
  $("#mods").innerHTML = items.map(([k, l]) => `<span class="chip ${m[k] ? "ok" : ""}">${m[k] ? "✓" : "✕"} ${l}</span>`).join("");
  $("#settings").hidden = false;
}
async function saveSettings(extra = {}) {
  try {
    await api("/api/config", { json: {
      anthropic_api_key: $("#cfg-key").value, claude_model: $("#cfg-model").value.trim(),
      whisper_model: $("#cfg-whisper").value, lyrics_language: $("#cfg-lang").value,
      convert_to_taiwan: $("#cfg-tw").checked, separation_model: $("#cfg-sep").value.trim(),
      karaoke_model: $("#cfg-kara").value.trim(), pitch_model: $("#cfg-pitch").value,
      balance_usd: $("#cfg-balance").value.trim().replace(/^US\$|^\$/, ""), ...extra,
    } });
    $("#settings").hidden = true; toast("設定已儲存"); pollJobs();
  } catch (e) { toast(e.message); }
}
$("#btn-settings").onclick = openSettings;
$("#cfg-close").onclick = () => ($("#settings").hidden = true);
$("#cfg-save").onclick = () => saveSettings();
$("#cfg-clear-key").onclick = () => { $("#cfg-key").value = ""; saveSettings({ anthropic_api_key: "", clear_key: true }); };
$("#settings").addEventListener("click", e => { if (e.target.id === "settings") $("#settings").hidden = true; });

/* ------------------------------------------------------------ 啟動 */
loadSongs().catch(e => toast(e.message));
pollJobs();
requestAnimationFrame(frame);
