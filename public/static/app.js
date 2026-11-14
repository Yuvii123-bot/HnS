/* संस्कृतम् · संवादसेवकः
   /api/chat इत्यस्मात् प्रवाहरूपेण (NDJSON) उत्तरं गृह्णाति, अक्षरखण्डान् च दर्शयति।
   सर्वं पाठः देवनागर्यामेव — न कोऽपि अनुवादः। */

const $ = (id) => document.getElementById(id);

const DEV_DIGITS = ["०", "१", "२", "३", "४", "५", "६", "७", "८", "९"];

/* अङ्कानां देवनागरीरूपम् */
function dev(n, decimals = 0) {
  if (n === null || n === undefined || Number.isNaN(Number(n))) return "—";
  const s = Number(n).toFixed(decimals);
  return s.replace(/\d/g, (d) => DEV_DIGITS[Number(d)]).replace(",", "·");
}

function devGrouped(n) {
  if (!n && n !== 0) return "—";
  return String(Math.round(n))
    .replace(/\B(?=(\d{3})+(?!\d))/g, "·")
    .replace(/\d/g, (d) => DEV_DIGITS[Number(d)]);
}

const KIND_SA = {
  conjunct: "संयुक्ताक्षरम्",
  conjunct_: "संयुक्ताक्षरम्",
  syllable: "अक्षरम्",
  consonant: "व्यञ्जनम्",
  vowel: "स्वरः",
  punct: "दण्डः",
  space: "अवकाशः",
  juncture: "सन्धिः",
  digit: "अङ्कः",
  latin: "अविदितम्",
};

const state = {
  streaming: false,
  abort: null,
  settings: {
    mode: "chat",
    temperature: 0.85,
    top_k: 40,
    max_new_tokens: 90,
    show_tokens: true,
  },
  ready: false,
  turns: [],            // [{user, assistant}] — context sent with the next message
};

/* ───────────────────────── विन्यासः ───────────────────────── */

const SETTINGS_KEY = "sanskrit-llm:settings";

function loadSettings() {
  try {
    Object.assign(state.settings, JSON.parse(localStorage.getItem(SETTINGS_KEY) || "{}"));
  } catch (_) {}
}

function saveSettings() {
  try { localStorage.setItem(SETTINGS_KEY, JSON.stringify(state.settings)); } catch (_) {}
}

function syncControls() {
  $("mode").value = state.settings.mode;
  $("temperature").value = state.settings.temperature;
  $("temperature-val").textContent = dev(state.settings.temperature, 2);
  $("topk").value = state.settings.top_k;
  $("topk-val").textContent = dev(state.settings.top_k);
  $("maxtok").value = state.settings.max_new_tokens;
  $("maxtok-val").textContent = dev(state.settings.max_new_tokens);
  $("show-tokens").checked = state.settings.show_tokens;
}

function bindControls() {
  $("mode").addEventListener("change", (e) => {
    state.settings.mode = e.target.value; saveSettings();
  });
  $("temperature").addEventListener("input", (e) => {
    state.settings.temperature = Number(e.target.value);
    $("temperature-val").textContent = dev(state.settings.temperature, 2);
    saveSettings();
  });
  $("topk").addEventListener("input", (e) => {
    state.settings.top_k = Number(e.target.value);
    $("topk-val").textContent = dev(state.settings.top_k);
    saveSettings();
  });
  $("maxtok").addEventListener("input", (e) => {
    state.settings.max_new_tokens = Number(e.target.value);
    $("maxtok-val").textContent = dev(state.settings.max_new_tokens);
    saveSettings();
  });
  $("show-tokens").addEventListener("change", (e) => {
    state.settings.show_tokens = e.target.checked; saveSettings();
    document.querySelectorAll(".tokens-inline").forEach((el) => {
      el.style.display = e.target.checked ? "flex" : "none";
    });
  });
  $("clear").addEventListener("click", () => {
    const t = $("transcript");
    [...t.querySelectorAll(".msg")].forEach((m, i) => { if (i > 0) m.remove(); });
  });
}

/* ───────────────────────── मूल्यस्य स्थितिः ───────────────────────── */

async function refreshInfo() {
  try {
    const info = await (await fetch("/api/info")).json();
    state.ready = !!info.ready;
    const dot = $("status-dot");
    const text = $("status-text");
    if (info.ready) {
      dot.className = "dot live";
      text.textContent = "सज्जम्";
    } else {
      dot.className = "dot wait";
      text.textContent = "प्रशिक्षणं प्रचलति…";
    }
    $("fact-params").textContent = devGrouped(info.model && info.model.params);
    $("fact-vocab").textContent = devGrouped(info.tokenizer && info.tokenizer.vocab_size);
    const summary = (info.training && info.training.summary) || {};
    const seenTotal = info.tokens_seen_total || summary.tokens_seen || 0;
    $("fact-tokens").textContent = devGrouped(seenTotal);
    const last = (info.training && info.training.last_point) || {};
    $("fact-loss").textContent =
      last.val_loss !== undefined ? dev(last.val_loss, 3)
      : (summary.final_loss !== undefined ? dev(summary.final_loss, 3) : "—");
    const steps = summary.total_steps
      || ((summary.pretrain && summary.pretrain.steps) || 0)
      || summary.steps || 0;
    $("foot-build").textContent = steps ? "०.१ · " + dev(steps) + " पदानि" : "०.१";
    return info.ready;
  } catch (err) {
    $("status-dot").className = "dot";
    $("status-text").textContent = "सेवकः न लभ्यते";
    return false;
  }
}

async function pollUntilReady() {
  const ready = await refreshInfo();
  if (!ready) {
    setTimeout(async () => {
      try { await fetch("/api/reload", { method: "POST" }); } catch (_) {}
      pollUntilReady();
    }, 20000);
  }
}

/* ───────────────────────── अक्षरविभाजनम् ───────────────────────── */

let tokenTimer = null;
let tokenSeq = 0;

function hasLatin(text) {
  return /[A-Za-z]/.test(text);
}

async function updateLiveTokens() {
  const raw = $("input").value;
  const text = raw.trim();
  const box = $("live-tokens");

  if (!text) {
    box.innerHTML = '<span class="placeholder">अत्र लिखतु…</span>';
    $("live-stats").textContent = "—";
    $("char-count").textContent = "० अक्षरम्";
    return;
  }
  if (hasLatin(text)) {
    box.innerHTML = '<span class="placeholder">केवलं देवनागरी स्वीक्रियते।</span>';
    $("live-stats").textContent = "—";
    $("char-count").textContent = "—";
    return;
  }

  const seq = ++tokenSeq;
  try {
    const res = await fetch("/api/tokenize", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text }),
    });
    const data = await res.json();
    if (seq !== tokenSeq) return;
    renderTokens(box, data.tokens);
    $("live-stats").textContent =
      `${dev(data.n_tokens)} खण्डाः · ${dev(data.n_aksharas)} अक्षराणि · ` +
      `${dev(data.chars_per_token, 2)} वर्णाः/खण्डः`;
    $("char-count").textContent = `${dev(data.n_aksharas)} अक्षरम्`;
  } catch (_) {}
}

function renderTokens(box, tokens) {
  box.innerHTML = "";
  tokens.forEach((t) => {
    const span = document.createElement("span");
    span.className = "tok " + t.kind;
    span.textContent = t.kind === "space" ? "␣" : t.text;
    span.title = `${KIND_SA[t.kind] || t.kind} · ${dev(t.id)}`;
    box.appendChild(span);
  });
}

/* ───────────────────────── सन्देशाः ───────────────────────── */

function el(tag, cls, text) {
  const node = document.createElement(tag);
  if (cls) node.className = cls;
  if (text !== undefined) node.textContent = text;
  return node;
}

function addMessage(role, opts = {}) {
  const article = el("article", `msg ${role}`);
  const avatar = el("div", "avatar", role === "user" ? "त्व" : "सू");
  const bubble = el("div", "bubble");
  const body = el("p", "deva");
  if (opts.text) body.textContent = opts.text;
  bubble.appendChild(body);

  const inline = el("div", "tokens-inline");
  inline.style.display = state.settings.show_tokens ? "flex" : "none";
  article.appendChild(avatar);
  article.appendChild(bubble);
  $("transcript").appendChild(article);
  $("transcript").scrollTop = $("transcript").scrollHeight;

  return { article, bubble, body, inline };
}

/* ───────────────────────── संवादप्रवाहः ───────────────────────── */

async function sendMessage(text) {
  if (!text || state.streaming) return;

  $("input").value = "";
  autoresize();
  updateLiveTokens();
  addMessage("user", { text });

  const { bubble, body, inline } = addMessage("assistant");
  body.classList.add("cursor");
  const controller = new AbortController();
  state.abort = controller;
  state.streaming = true;
  setSending(true);

  let tokenCount = 0;
  const decoder = new TextDecoder();
  let buffer = "";

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        history: state.settings.mode === "chat" ? state.turns.slice(-4) : [],
        ...state.settings,
      }),
      signal: controller.signal,
    });
    if (!res.ok || !res.body) throw new Error(dev(res.status));

    const reader = res.body.getReader();
    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split("\n");
      buffer = lines.pop();
      for (const line of lines) {
        if (!line.trim()) continue;
        try { handleEvent(JSON.parse(line)); } catch (_) {}
      }
    }
    if (buffer.trim()) { try { handleEvent(JSON.parse(buffer)); } catch (_) {} }
  } catch (err) {
    if (err.name !== "AbortError") {
      body.textContent = "क्षम्यताम् — " + err.message;
      body.style.color = "var(--rose)";
    }
  } finally {
    body.classList.remove("cursor");
    state.streaming = false;
    state.abort = null;
    setSending(false);
    $("transcript").scrollTop = $("transcript").scrollHeight;
  }

  function handleEvent(evt) {
    if (evt.event === "token") {
      body.textContent += evt.text;
      tokenCount += 1;
      if (state.settings.show_tokens) {
        const glyph = evt.text === " " ? "␣" : evt.text === "\n" ? "↵" : evt.text;
        const chip = el("span", "tok " + (evt.kind || ""), glyph);
        chip.title = `${KIND_SA[evt.kind] || ""} · ${dev(evt.token_id)}`;
        inline.appendChild(chip);
      }
      $("transcript").scrollTop = $("transcript").scrollHeight;
    } else if (evt.event === "done") {
      finalize(evt);
    } else if (evt.event === "error") {
      body.textContent = "क्षम्यताम् — " + evt.message;
      body.style.color = "var(--rose)";
    }
  }

  function finalize(evt) {
    if (inline.childNodes.length) bubble.appendChild(inline);
    if (evt.text && evt.text.trim()) {
      state.turns.push({ user: text, assistant: evt.text.trim() });
      if (state.turns.length > 8) state.turns = state.turns.slice(-8);
    }
    const tools = el("div", "tools");

    const copy = el("button", "", "प्रतिलिपिः");
    copy.addEventListener("click", () => {
      navigator.clipboard.writeText(body.textContent || "");
      copy.textContent = "जातम् ✓";
      setTimeout(() => (copy.textContent = "प्रतिलिपिः"), 1200);
    });
    tools.appendChild(copy);

    tools.appendChild(el("p", "meta",
      `${dev(evt.n_tokens || tokenCount)} खण्डाः · उष्णता ${dev(state.settings.temperature, 2)}`));
    bubble.appendChild(tools);
  }
}

function setSending(on) {
  const btn = $("send");
  btn.disabled = false;
  btn.querySelector("span").textContent = on ? "रुध्" : "प्रेषय";
  btn.classList.toggle("stop", on);
}

/* ───────────────────────── लेखनम् ───────────────────────── */

function autoresize() {
  const ta = $("input");
  ta.style.height = "auto";
  ta.style.height = Math.min(ta.scrollHeight, 180) + "px";
}

function bindComposer() {
  const form = $("composer");
  const ta = $("input");

  ta.addEventListener("input", () => {
    autoresize();
    clearTimeout(tokenTimer);
    tokenTimer = setTimeout(updateLiveTokens, 220);
  });

  ta.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      submit();
    }
  });

  form.addEventListener("submit", (e) => {
    e.preventDefault();
    submit();
  });

  $("send").addEventListener("click", (e) => {
    if (state.streaming && state.abort) {
      e.preventDefault();
      state.abort.abort();
      setSending(false);
    }
  });

  function submit() {
    if (state.streaming) return;
    const text = ta.value.trim();
    if (!text) return;
    if (hasLatin(text)) {
      // केवलं देवनागरी — अनुवादः न भवति
      const box = $("live-tokens");
      box.innerHTML = '<span class="placeholder">केवलं देवनागर्यां लिखतु।</span>';
      return;
    }
    sendMessage(text);
  }

  document.querySelectorAll("#suggestions button").forEach((b) => {
    b.addEventListener("click", () => {
      ta.value = b.dataset.text || b.textContent;
      autoresize();
      updateLiveTokens();
      ta.focus();
    });
  });
}

/* ───────────────────────── आरम्भः ───────────────────────── */

loadSettings();
syncControls();
bindControls();
bindComposer();
autoresize();
updateLiveTokens();
pollUntilReady();
