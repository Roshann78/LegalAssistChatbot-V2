#!/usr/bin/env node
/*
 * Builds INTERVIEW_PREP.html from the five INTERVIEW_*.md files.
 * No dependencies: a small markdown converter, inline CSS/JS, dark theme,
 * sidebar navigation, per-section table of contents and full-text search.
 *
 * Usage: node build_interview_html.js
 */
const fs = require("fs");
const path = require("path");

const ROOT = __dirname;
const SECTIONS = [
  { file: "INTERVIEW_GUIDE.md", title: "Master Guide" },
  { file: "INTERVIEW_QA.md", title: "Q&A" },
  { file: "INTERVIEW_TECH_DEEP_DIVE.md", title: "Tech Deep Dive" },
  { file: "INTERVIEW_SYSTEM_DESIGN.md", title: "System Design" },
  { file: "INTERVIEW_WHITEBOARD.md", title: "Whiteboard" },
];

// ───────────────────────── markdown → HTML ─────────────────────────
function esc(s) {
  return s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
}

function inline(raw) {
  let s = esc(raw);
  const codes = [];
  s = s.replace(/`([^`]+)`/g, function (m, c) {
    codes.push(c);
    return "\u0000" + (codes.length - 1) + "\u0000";
  });
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/(^|[^*\w])\*([^*\s][^*]*?)\*(?![*\w])/g, "$1<em>$2</em>");
  s = s.replace(/\[([^\]]+)\]\((https?:[^)\s]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
  s = s.replace(/\u0000(\d+)\u0000/g, function (m, i) {
    return "<code>" + codes[Number(i)] + "</code>";
  });
  return s;
}

function slugify(text) {
  return text
    .toLowerCase()
    .replace(/`/g, "")
    .replace(/[^a-z0-9]+/g, "-")
    .replace(/^-+|-+$/g, "")
    .slice(0, 60) || "section";
}

function splitRow(line) {
  let t = line.trim();
  if (t.startsWith("|")) t = t.slice(1);
  if (t.endsWith("|")) t = t.slice(0, -1);
  // split on pipes that are not inside backticks
  const cells = [];
  let cur = "";
  let inCode = false;
  for (let i = 0; i < t.length; i++) {
    const ch = t[i];
    if (ch === "`") inCode = !inCode;
    if (ch === "\\" && t[i + 1] === "|") {
      cur += "|";
      i++;
      continue;
    }
    if (ch === "|" && !inCode) {
      cells.push(cur.trim());
      cur = "";
    } else cur += ch;
  }
  cells.push(cur.trim());
  return cells;
}

function isTableSep(line) {
  return /^\s*\|?\s*:?-{2,}:?\s*(\|\s*:?-{2,}:?\s*)*\|?\s*$/.test(line);
}

const LIST_RE = /^(\s*)([-*]|\d+\.)\s+(.*)$/;

function renderBlocks(lines, ctx) {
  const out = [];
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (/^\s*$/.test(line)) {
      i++;
      continue;
    }
    // fenced code
    const fence = line.match(/^```(\w*)\s*$/);
    if (fence) {
      const buf = [];
      i++;
      while (i < lines.length && !/^```\s*$/.test(lines[i])) buf.push(lines[i++]);
      i++;
      out.push('<pre><code class="lang-' + fence[1] + '">' + esc(buf.join("\n")) + "</code></pre>");
      continue;
    }
    // heading
    const h = line.match(/^(#{1,4})\s+(.*)$/);
    if (h) {
      const level = h[1].length;
      const text = h[2].trim();
      let id = ctx.prefix + "-" + slugify(text);
      let n = 2;
      while (ctx.ids.has(id)) id = ctx.prefix + "-" + slugify(text) + "-" + n++;
      ctx.ids.add(id);
      ctx.headings.push({ level: level, text: text.replace(/[`*]/g, ""), id: id });
      out.push("<h" + level + ' id="' + id + '">' + inline(text) + "</h" + level + ">");
      i++;
      continue;
    }
    // hr
    if (/^\s*(-{3,}|\*{3,})\s*$/.test(line)) {
      out.push("<hr>");
      i++;
      continue;
    }
    // table
    if (line.trim().startsWith("|") && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      const head = splitRow(line);
      i += 2;
      const rows = [];
      while (i < lines.length && lines[i].trim().startsWith("|")) rows.push(splitRow(lines[i++]));
      let html = '<div class="tablewrap"><table><thead><tr>' + head.map(function (c) { return "<th>" + inline(c) + "</th>"; }).join("") + "</tr></thead><tbody>";
      rows.forEach(function (r) {
        html += "<tr>" + r.map(function (c) { return "<td>" + inline(c) + "</td>"; }).join("") + "</tr>";
      });
      out.push(html + "</tbody></table></div>");
      continue;
    }
    // blockquote
    if (/^\s*>/.test(line)) {
      const buf = [];
      while (i < lines.length && /^\s*>/.test(lines[i])) buf.push(lines[i++].replace(/^\s*> ?/, ""));
      out.push("<blockquote>" + renderBlocks(buf, ctx) + "</blockquote>");
      continue;
    }
    // list
    if (LIST_RE.test(line)) {
      const items = [];
      while (i < lines.length && (LIST_RE.test(lines[i]) || (/^\s{2,}\S/.test(lines[i]) && items.length))) {
        const m = lines[i].match(LIST_RE);
        if (m) items.push({ indent: m[1].length, ordered: /\d/.test(m[2]), text: m[3] });
        else items[items.length - 1].text += " " + lines[i].trim();
        i++;
      }
      out.push(renderList(items));
      continue;
    }
    // paragraph
    const buf = [line];
    i++;
    while (
      i < lines.length &&
      !/^\s*$/.test(lines[i]) &&
      !/^(#{1,4}\s|```|\s*>|\s*\|)/.test(lines[i]) &&
      !LIST_RE.test(lines[i]) &&
      !/^\s*(-{3,}|\*{3,})\s*$/.test(lines[i])
    )
      buf.push(lines[i++]);
    out.push("<p>" + inline(buf.join(" ")) + "</p>");
  }
  return out.join("\n");
}

function renderList(items) {
  let html = "";
  const stack = [];
  items.forEach(function (it) {
    while (stack.length && it.indent < stack[stack.length - 1].indent) {
      html += "</li></" + stack.pop().tag + ">";
    }
    const tag = it.ordered ? "ol" : "ul";
    if (!stack.length || it.indent > stack[stack.length - 1].indent) {
      html += "<" + tag + ">";
      stack.push({ indent: it.indent, tag: tag });
    } else {
      html += "</li>";
    }
    html += "<li>" + inline(it.text);
  });
  while (stack.length) html += "</li></" + stack.pop().tag + ">";
  return html;
}

function plain(md) {
  return md
    .replace(/```[\s\S]*?```/g, function (m) { return m.replace(/```\w*/g, ""); })
    .replace(/[`*#>|]/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function buildSection(sec, idx) {
  const md = fs.readFileSync(path.join(ROOT, sec.file), "utf8").replace(/\r\n/g, "\n");
  const ctx = { prefix: "s" + idx, ids: new Set(), headings: [] };
  const html = renderBlocks(md.split("\n"), ctx);
  // search entries: split markdown at headings
  const entries = [];
  let cur = { id: ctx.prefix + "-top", heading: sec.title, text: [] };
  const hs = ctx.headings.slice();
  md.split("\n").forEach(function (line) {
    const m = line.match(/^(#{1,4})\s+(.*)$/);
    if (m) {
      if (cur.text.length) entries.push({ id: cur.id, heading: cur.heading, text: plain(cur.text.join("\n")) });
      const hd = hs.shift();
      cur = { id: hd ? hd.id : cur.id, heading: m[2].replace(/[`*]/g, ""), text: [] };
    } else cur.text.push(line);
  });
  if (cur.text.length) entries.push({ id: cur.id, heading: cur.heading, text: plain(cur.text.join("\n")) });
  return {
    title: sec.title,
    file: sec.file,
    html: html,
    toc: ctx.headings.filter(function (h) { return h.level === 2 || h.level === 3; }),
    entries: entries.filter(function (e) { return e.text.length > 0; }),
  };
}

// ───────────────────────── client script (inlined via toString) ─────────────────────────
function clientMain(DATA) {
  var current = 0;
  var side = document.getElementById("side");
  var content = document.getElementById("content");
  var results = document.getElementById("results");
  var nav = document.getElementById("nav");
  var subtoc = document.getElementById("subtoc");
  var q = document.getElementById("q");

  function el(tag, cls, html) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (html !== undefined) e.innerHTML = html;
    return e;
  }
  function esc(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }

  DATA.forEach(function (s, i) {
    var b = el("button", "navbtn", esc(s.title));
    b.addEventListener("click", function () { show(i); side.classList.remove("open"); });
    b.id = "nav" + i;
    nav.appendChild(b);
    var sec = el("section", "sec");
    sec.id = "sec" + i;
    var toc = el("details", "toc");
    toc.innerHTML = "<summary>Table of contents (" + s.toc.length + ")</summary><ul>" + s.toc.map(function (h) {
      return '<li class="l' + h.level + '"><a href="#' + h.id + '" data-id="' + h.id + '">' + esc(h.text) + "</a></li>";
    }).join("") + "</ul>";
    sec.appendChild(toc);
    var body = el("div", "md", s.html);
    sec.appendChild(body);
    content.appendChild(sec);
  });

  function show(i, anchor) {
    current = i;
    DATA.forEach(function (_, j) {
      document.getElementById("sec" + j).style.display = j === i ? "block" : "none";
      document.getElementById("nav" + j).classList.toggle("active", j === i);
    });
    subtoc.innerHTML = "";
    DATA[i].toc.filter(function (h) { return h.level === 2; }).forEach(function (h) {
      var a = el("a", "sub", esc(h.text));
      a.href = "#" + h.id;
      a.addEventListener("click", function (ev) { ev.preventDefault(); goto(h.id); side.classList.remove("open"); });
      subtoc.appendChild(a);
    });
    results.style.display = "none";
    content.style.display = "block";
    if (anchor) goto(anchor);
    else window.scrollTo(0, 0);
    history.replaceState(null, "", "#sec" + i);
  }

  function goto(id) {
    var t = document.getElementById(id);
    if (!t) return;
    t.scrollIntoView({ behavior: "smooth", block: "start" });
    t.classList.remove("flash");
    void t.offsetWidth;
    t.classList.add("flash");
  }

  content.addEventListener("click", function (ev) {
    var a = ev.target.closest && ev.target.closest("a[data-id]");
    if (a) { ev.preventDefault(); goto(a.getAttribute("data-id")); }
  });

  // full-text search: every word must appear in a section entry
  var index = [];
  DATA.forEach(function (s, i) {
    s.entries.forEach(function (e) {
      index.push({ sec: i, id: e.id, heading: e.heading, text: e.text, low: (e.heading + " " + e.text).toLowerCase() });
    });
  });

  function snippet(text, words) {
    var low = text.toLowerCase();
    var pos = -1;
    for (var k = 0; k < words.length; k++) {
      var p = low.indexOf(words[k]);
      if (p >= 0 && (pos < 0 || p < pos)) pos = p;
    }
    var start = Math.max(0, pos - 60);
    var s = text.slice(start, start + 220);
    s = esc(s);
    words.forEach(function (w) {
      if (!w) return;
      var re = new RegExp("(" + w.replace(/[.*+?^${}()|[\]\\]/g, "\\$&") + ")", "ig");
      s = s.replace(re, "<mark>$1</mark>");
    });
    return (start > 0 ? "… " : "") + s + (start + 220 < text.length ? " …" : "");
  }

  function search(term) {
    var words = term.toLowerCase().split(/\s+/).filter(Boolean);
    if (!words.length) { show(current); return; }
    var hits = index.filter(function (e) {
      return words.every(function (w) { return e.low.indexOf(w) >= 0; });
    });
    content.style.display = "none";
    results.style.display = "block";
    results.innerHTML = "";
    results.appendChild(el("h2", "", hits.length + " result" + (hits.length === 1 ? "" : "s") + " for “" + esc(term) + "”"));
    hits.slice(0, 60).forEach(function (h) {
      var r = el("div", "hit");
      r.innerHTML = '<div class="hs">' + esc(DATA[h.sec].title) + '</div><div class="hh">' + esc(h.heading) + '</div><div class="ht">' + snippet(h.text, words) + "</div>";
      r.addEventListener("click", function () { q.value = ""; show(h.sec, h.id); });
      results.appendChild(r);
    });
    if (hits.length > 60) results.appendChild(el("p", "muted", "Showing the first 60. Refine the search."));
  }

  var timer = null;
  q.addEventListener("input", function () {
    clearTimeout(timer);
    timer = setTimeout(function () { search(q.value.trim()); }, 120);
  });
  document.addEventListener("keydown", function (ev) {
    var tag = (ev.target.tagName || "").toLowerCase();
    if (ev.key === "/" && tag !== "input") { ev.preventDefault(); q.focus(); }
    if (ev.key === "Escape") { q.value = ""; show(current); q.blur(); }
  });
  document.getElementById("burger").addEventListener("click", function () { side.classList.toggle("open"); });

  var m = location.hash.match(/^#sec(\d+)$/);
  show(m && DATA[Number(m[1])] ? Number(m[1]) : 0);
}

// ───────────────────────── page template ─────────────────────────
const CSS = [
  ":root{--bg:#0f1115;--panel:#171a21;--panel2:#1d2129;--text:#e6e8ee;--muted:#9aa3b2;--accent:#6aa9ff;--accent2:#7ee0b5;--border:#2a2f3a;--code:#11141a;--warn:#f5c36b}",
  "*{box-sizing:border-box}html{scroll-behavior:smooth}",
  "body{margin:0;background:var(--bg);color:var(--text);font:16px/1.65 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}",
  ".app{display:flex;min-height:100vh}",
  "#side{position:sticky;top:0;align-self:flex-start;height:100vh;width:290px;flex:0 0 290px;overflow:auto;background:var(--panel);border-right:1px solid var(--border);padding:18px 14px}",
  "#side h1{font-size:17px;margin:0 0 12px;letter-spacing:.2px}",
  "#q{width:100%;padding:9px 11px;border-radius:8px;border:1px solid var(--border);background:var(--bg);color:var(--text);font-size:14px;outline:none}",
  "#q:focus{border-color:var(--accent)}",
  ".hint{font-size:12px;color:var(--muted);margin:6px 2px 14px}",
  ".navbtn{display:block;width:100%;text-align:left;margin:3px 0;padding:9px 11px;border:0;border-radius:8px;background:transparent;color:var(--text);font-size:14.5px;cursor:pointer}",
  ".navbtn:hover{background:var(--panel2)}.navbtn.active{background:#22304a;color:#fff;font-weight:600}",
  "#subtoc{margin-top:10px;padding-top:10px;border-top:1px solid var(--border)}",
  "#subtoc a.sub{display:block;padding:4px 8px;color:var(--muted);text-decoration:none;font-size:13px;border-radius:6px}",
  "#subtoc a.sub:hover{color:var(--text);background:var(--panel2)}",
  "main{flex:1;min-width:0;padding:28px clamp(16px,4vw,56px) 80px;max-width:1100px}",
  "#top{display:none;position:sticky;top:0;z-index:5;background:var(--panel);border-bottom:1px solid var(--border);padding:10px 14px}",
  "#burger{background:var(--panel2);border:1px solid var(--border);color:var(--text);border-radius:8px;padding:6px 12px;font-size:14px}",
  ".md h1{font-size:30px;margin:0 0 14px;line-height:1.25}",
  ".md h2{font-size:23px;margin:38px 0 10px;padding-top:14px;border-top:1px solid var(--border);color:#fff}",
  ".md h3{font-size:18.5px;margin:26px 0 8px;color:var(--accent)}",
  ".md h4{font-size:16px;margin:18px 0 6px;color:var(--accent2)}",
  ".md p{margin:10px 0}.md a{color:var(--accent)}",
  ".md code{background:var(--code);border:1px solid var(--border);border-radius:5px;padding:1px 5px;font:13.5px ui-monospace,SFMono-Regular,Consolas,monospace;color:#f0c987}",
  ".md pre{background:var(--code);border:1px solid var(--border);border-radius:10px;padding:14px 16px;overflow:auto;line-height:1.35}",
  ".md pre code{background:none;border:0;padding:0;color:#d7dce6;font-size:13px;white-space:pre}",
  ".md blockquote{margin:10px 0 18px;padding:10px 16px;border-left:4px solid var(--accent2);background:var(--panel);border-radius:0 8px 8px 0}",
  ".md blockquote p{margin:6px 0}",
  ".md hr{border:0;border-top:1px solid var(--border);margin:28px 0}",
  ".md ul,.md ol{padding-left:24px}.md li{margin:4px 0}",
  ".tablewrap{overflow:auto;margin:12px 0;border:1px solid var(--border);border-radius:10px}",
  "table{border-collapse:collapse;width:100%;font-size:14.5px}",
  "th,td{padding:8px 12px;border-bottom:1px solid var(--border);text-align:left;vertical-align:top}",
  "th{background:var(--panel2);position:sticky;top:0}tr:last-child td{border-bottom:0}",
  "details.toc{background:var(--panel);border:1px solid var(--border);border-radius:10px;padding:8px 14px;margin:0 0 22px}",
  "details.toc summary{cursor:pointer;color:var(--muted);font-size:14px}",
  "details.toc ul{list-style:none;padding-left:0;columns:2;column-gap:28px;margin:10px 0 4px}",
  "details.toc li{break-inside:avoid;margin:3px 0;font-size:14px}details.toc li.l3{padding-left:14px;font-size:13px}",
  "details.toc a{color:var(--text);text-decoration:none}details.toc a:hover{color:var(--accent)}",
  ".hit{padding:12px 14px;margin:10px 0;background:var(--panel);border:1px solid var(--border);border-radius:10px;cursor:pointer}",
  ".hit:hover{border-color:var(--accent)}.hs{font-size:12px;color:var(--accent2);text-transform:uppercase;letter-spacing:.6px}",
  ".hh{font-weight:600;margin:2px 0}.ht{color:var(--muted);font-size:14px}",
  "mark{background:#5a4a12;color:#ffe9a8;border-radius:3px;padding:0 2px}.muted{color:var(--muted)}",
  ".flash{animation:flash 1.6s ease}@keyframes flash{0%{background:#2b3f66}100%{background:transparent}}",
  "@media(max-width:860px){#side{position:fixed;left:-310px;top:0;z-index:10;transition:left .2s;box-shadow:0 0 30px #000a}#side.open{left:0}#top{display:block}.app{display:block}main{padding-top:18px}details.toc ul{columns:1}}",
  "@media print{#side,#top{display:none}body{background:#fff;color:#000}}",
].join("\n");

function build() {
  const data = SECTIONS.map(buildSection);
  const json = JSON.stringify(data)
    .split("<").join(String.fromCharCode(92) + "u003c")
    .split(String.fromCharCode(0x2028)).join(String.fromCharCode(92) + "u2028")
    .split(String.fromCharCode(0x2029)).join(String.fromCharCode(92) + "u2029");
  const html =
    '<!doctype html><html lang="en"><head><meta charset="utf-8">' +
    '<meta name="viewport" content="width=device-width,initial-scale=1">' +
    "<title>Legal Assist RAG: Interview Prep</title><style>" + CSS + "</style></head><body>" +
    '<div id="top"><button id="burger" type="button">Menu</button> <span class="muted">Interview Prep</span></div>' +
    '<div class="app"><aside id="side"><h1>Legal Assist RAG<br><span class="muted" style="font-weight:400;font-size:13px">Interview prep</span></h1>' +
    '<input id="q" type="search" placeholder="Search everything…" autocomplete="off">' +
    '<div class="hint">Press / to search, Esc to clear</div><nav id="nav"></nav><div id="subtoc"></div></aside>' +
    '<main id="main"><div id="results" style="display:none"></div><article id="content"></article></main></div>' +
    "<script>var DATA = " + json + ";\n(" + clientMain.toString() + ")(DATA);</script></body></html>";
  const out = path.join(ROOT, "INTERVIEW_PREP.html");
  fs.writeFileSync(out, html, "utf8");
  const kb = (Buffer.byteLength(html) / 1024).toFixed(0);
  const words = data.reduce(function (n, s) { return n + s.entries.reduce(function (m, e) { return m + e.text.split(" ").length; }, 0); }, 0);
  console.log("Wrote " + out + " (" + kb + " KB)");
  data.forEach(function (s) { console.log("  " + s.title + ": " + s.toc.length + " TOC entries, " + s.entries.length + " searchable blocks"); });
  console.log("  ~" + words + " words indexed for search");
}

build();
