# Training Book Visual Audit — Pending Work

**Session:** 2026-10-31 10:39–10:48 CET  
**Commit Audited:** `099a43c` (feat: Training Book visual overhaul — SVG diagrams, screenshots, Neon Manifold reader SPA)  
**Status:** ⏳ Audit complete, fixes NOT applied

---

## 1. Context & Objective

Augusto requested a chapter-by-chapter headless-browser audit of the Training Book SPA to fix:
1. Screenshots being much wider than the text window
2. Original ASCII diagrams either displayed properly OR hidden behind expandable panels with a visually superior alternative shown by default
3. Readability on both PC (Desktop 1280×900) and phone (Mobile 390×844)

The audit ran successfully using Puppeteer in a background process (`fast-seaslug`) but the session was interrupted before fixes could be written or committed.

---

## 2. Findings Summary

### 2.1 Critical CSS Layout Bug: Flex Container Blowout

**File:** `docs/training-book/triantica-book.css` (lines 1–217)

**Root Cause:**
```css
.content {
  margin-left: var(--sidebar-w);
  min-height: 100vh;
  display: flex;
  flex-direction: column;  /* ← flex container */
}
.article {
  max-width: var(--content-max);  /* 860px */
  margin: 0 auto;
  padding: 48px 40px 24px;
  flex: 1;
  /* MISSING: width: 100%; min-width: 0; */
}
```

**Symptom:**
- On narrow viewports (Mobile 390px), wide children (`<pre>` blocks, `<table>`, `<img>`) with natural widths > 860px cause `.article` to expand beyond the viewport, creating horizontal page scroll.
- Screenshots (1440px–2880px natural width) overflow the text column.
- ASCII diagrams with 80–105 char line lengths overflow.

**Fix Required:**
```css
.article {
  max-width: var(--content-max);
  width: 100%;
  min-width: 0;           /* ← critical: allows flex item to shrink below content size */
  margin: 0 auto;
  padding: 48px 40px 24px;
  flex: 1;
  overflow-wrap: break-word;
}
.article pre, .article table, .article img {
  max-width: 100%;
  overflow-x: auto;
}
```

---

### 2.2 Image Classification Bug in reader.js

**File:** `docs/training-book/reader.js` (lines 75–85)

**Current Code:**
```javascript
renderer.image = function (token) {
  const href = token.href || '';
  const text = token.text || '';
  if (href.includes('assets/') || /\.(png|jpg|jpeg|webp|gif)$/i.test(href)) {
    return `<div class="screenshot-container">...</div>`;
  }
  if (/\.svg$/i.test(href)) {
    return `<div class="diagram-container">...</div>`;
  }
  return origImage(token);
};
```

**Bug:**
- `href.includes('assets/')` matches `assets/diagrams/*.svg` BEFORE the `.svg` test, wrapping SVGs as screenshots.

**Fix:**
```javascript
if (/\.svg$/i.test(href)) {
  return `<div class="diagram-container">...</div>`;  // SVG check FIRST
}
if (href.includes('assets/') || /\.(png|jpg|jpeg|webp|gif)$/i.test(href)) {
  return `<div class="screenshot-container">...</div>`;
}
```

---

### 2.3 Screenshots Lack Responsive Framing

**Current:** `triantica-book.css`
```css
.screenshot-container img {
  max-width: 100%;
  border-radius: var(--radius);
  border: 1px solid var(--border-dim);
  box-shadow: 0 4px 30px rgba(0,0,0,.5);
}
```

**Problem:**
- No bounded height; tall screenshots (dashboard views) can fill the entire viewport.
- High-DPI captures stay at native resolution, appearing sharp but enormous.

**Fix:**
```css
.screenshot-container {
  margin: 28px 0;
  text-align: center;
}
.screenshot-container img {
  max-width: 100%;
  max-height: 480px;          /* cap vertical space */
  width: auto;
  height: auto;
  object-fit: contain;        /* preserve aspect ratio */
  border-radius: var(--radius);
  border: 1px solid var(--border-dim);
  box-shadow: 0 4px 30px rgba(0,0,0,.5);
}
```

---

### 2.4 Bare ASCII Diagrams: 44 Need Visual Replacement or Wrapping

**Total:** 49 ASCII box-drawing diagrams across all chapter files  
**Wrapped in `<details>`:** 5  
**Bare (need work):** 44

#### Inventory by Chapter:

| Chapter | Bare ASCII Blocks | Wrapped Blocks | Notes |
|---------|-------------------|----------------|-------|
| `01-the-game.md` | 5 | 2 | Includes game rules diagram, cell geometry |
| `02-rl-fundamentals.md` | 6 | 1 | RL loop, policy network, credit assignment |
| `03-architecture-journey.md` | 3 | 0 | Migration bugs, forensic audit |
| `04-hardware-m2-ultra.md` | 3 | 0 | Hardware pipeline, batch sizing |
| `05-parameters-explored.md` | 2 | 0 | Parameter categories |
| `06-training-strategies.md` | 4 | 0 | Self-play, opponent mix, cluster rewards |
| `07-critical-bugs-and-audits.md` | 8 | 0 | Systemic bugs, browser illusion |
| `08-the-heuristic-wall.md` | 3 | 1 | Heuristic strategy, blind spots |
| `10-key-takeaways.md` | 1 | 0 | Key lessons |
| `11a-replay-and-conv-encoding.md` | 5 | 0 | Replay buffer, convolution detection |
| `11b-milestone-runs-tf-b5-to-b13.md` | 2 | 0 | Phase 2, sawtooth loss |
| `11c-actor-critic-and-batchnorm.md` | 2 | 0 | Actor-critic, residual network |
| `11d-mcts-theory-and-implementation.md` | 5 | 1 | MCTS pipeline, UCT formula |
| `11e-quiescence-mcts-and-torus-mastery.md` | 3 | 0 | Quiescence comparison, toroidal topology |
| `12-toroidal-3d-and-telemetry.md` | 1 | 0 | Engine/view separation |
| `appendices.md` | 1 | 0 | Project structure |

**Total: 44 bare ASCII blocks require one of:**
1. **Visual replacement:** Create Neon Manifold-styled SVG equivalent, show by default, wrap original ASCII in `<details>`
2. **Styled HTML card:** For tables/structured data, render as styled `<div>` with CSS grid/flex
3. **Responsive scaling:** For diagrams that are acceptable as-is, add `overflow-x: auto` and ensure they scale

**Priority Diagram Candidates (most visible/important):**
- `01-the-game.md` L11-L24: Game rules summary (12 lines, 60 chars) → styled info card
- `02-rl-fundamentals.md` L53-L73: Policy Network (19 lines, 64 chars) → SVG flowchart
- `02-rl-fundamentals.md` L122-L148: RL Paradigms (25 lines, 80 chars) → comparison table
- `07-critical-bugs-and-audits.md` L177-L201: 6 Systemic Bugs (23 lines, 80 chars) → bug catalog cards
- `11b-milestone-runs-tf-b5-to-b13.md` L37-L67: Phase 2 No Breakthrough (29 lines) → timeline infographic
- `11e-quiescence-mcts-and-torus-mastery.md` L211-L277: Toroidal Topology T² (65 lines, 94 chars) → SVG topology diagram

---

### 2.5 Raw `{#anchor-id}` Heading IDs Rendering

**Files Affected:** Multiple chapters (e.g., `01-the-game.md`, `09-experiment-reference.md`)

**Example:**
```markdown
## 1. The Game {#1-the-game}
```

Renders as:
```
1. The Game {#1-the-game}
```

**Root Cause:** Marked.js doesn't strip Pandoc-style `{#...}` custom heading IDs.

**Fix in `reader.js`:**
```javascript
renderer.heading = function (token) {
  const text = (token.text || '').replace(/\s*\{#[^}]+\}\s*$/g, '').trim();
  const depth = token.depth || 1;
  const id = slugify(text);
  return `<h${depth} id="${id}">${text}</h${depth}>`;
};
```

---

### 2.6 UTF-8 Mojibake in 11d

**File:** `docs/training-book/11d-mcts-theory-and-implementation.md`

**Issues:**
- Line 21: `â€"` should be `—` (em-dash)
- Line 21: `ðŸ“–` should be `📖` (book emoji)

**Cause:** UTF-8 file edited or saved with wrong encoding at some point.

**Fix:** Re-save file as UTF-8, replace mojibake sequences with correct characters.

---

### 2.7 SPA Anchor Link Navigation

**Symptom:** Clicking `#section-id` links inside a chapter triggers `#/chXX` hash change, resetting to Book Index instead of scrolling to the heading.

**Root Cause:** Hash router intercepts all `#...` clicks.

**Fix in `reader.js`:**
```javascript
document.addEventListener('click', e => {
  const a = e.target.closest('a[href^="#"]');
  if (!a) return;
  const href = a.getAttribute('href');
  if (href.startsWith('#/') ) return;  // chapter nav
  e.preventDefault();
  const target = document.querySelector(href);
  if (target) target.scrollIntoView({ behavior: 'smooth' });
});
```

---

## 3. Tables Inventory

**Total Tables:** 35 across all files

**Largest Tables (may need horizontal scroll):**
- `09-experiment-reference.md` L11-L32: 22 rows × 12 cols (experiment reference)
- `11e-quiescence-mcts-and-torus-mastery.md` L179-L191: 13 rows × 11 cols (TF-B15 results)
- `11b-milestone-runs-tf-b5-to-b13.md` L215-L227: 13 rows × 9 cols (TF-B9 telemetry)

**Current CSS:**
```css
.article table {
  width: 100%;
  display: block;
  overflow-x: auto;
}
```

**Status:** Tables have horizontal scroll capability. Should verify on Mobile 390px during fix pass.

---

## 4. Images Inventory

**SVG Diagrams:** 14 (all in `assets/diagrams/`)  
**PNG Screenshots:** 5 (all in `assets/`)

**All Images:**

| Chapter | Image | Type | Location |
|---------|-------|------|----------|
| 01 | `triantica-menu-d20.png` | screenshot | L7 |
| 01 | `cell-geometry-rotations.svg` | diagram | L30 |
| 01 | `enclosure-cascade-snowball.svg` | diagram | L87 |
| 01 | `triantica-gameplay-25d.png` | screenshot | L125 |
| 01 | `toroidal-wrap.svg` | diagram | L209 |
| 02 | `rl-training-loop.svg` | diagram | L11 |
| 03 | `architecture-climb.svg` | diagram | L40 |
| 08 | `triantica-dashboard-overview.png` | screenshot | L9 |
| 08 | `architecture-climb.svg` | diagram | L125 |
| 09 | `rl-training-loop.svg` | diagram | L7 |
| 11a | `convnet-architecture.svg` | diagram | L93 |
| 11d | `mcts-search-tree.svg` | diagram | L21 |
| 11e | `triantica-dashboard-tf-b15.png` | screenshot | L300 |
| 12 | `triantica-toroidal-pan-seam.png` | screenshot | L7 |
| 12 | `toroidal-wrap.svg` | diagram | L9 |

---

## 5. Pending Work Checklist

### Immediate Fixes (Est. 30 min)

- [ ] **Fix CSS flex blowout** — Add `width: 100%; min-width: 0;` to `.article`
- [ ] **Fix image classification** — Move SVG check before assets check in `reader.js`
- [ ] **Add screenshot max-height** — Cap at 480px with `object-fit: contain`
- [ ] **Strip `{#id}` from headings** — Add regex replacement in `reader.js`
- [ ] **Fix mojibake in 11d** — Re-save as UTF-8, replace bad sequences

### ASCII Diagram Work (Est. 2–4 hours)

- [ ] **Audit each of 44 bare ASCII blocks**
- [ ] **Create visual replacements for top 10 priority diagrams**
- [ ] **Wrap remaining ASCII in `<details>` panels**
- [ ] **Test rendering on Desktop + Mobile viewports**

### SPA Navigation Fix (Est. 15 min)

- [ ] **Implement anchor scroll handler** — Intercept non-chapter `#` clicks

### Verification (Est. 20 min)

- [ ] **Run headless browser audit again** — Test all 19 chapters on Desktop + Mobile
- [ ] **Manual spot-check** in browser at multiple widths
- [ ] **Commit and push** with descriptive message

---

## 6. Test Commands

To re-run the headless browser audit:

```bash
cd ~/.openclaw/workspace/triantica

# Full audit (Desktop + Mobile)
node -e '
const http = require("http");
const fs = require("fs");
const path = require("path");
const puppeteer = require("puppeteer");

const root = path.join(process.cwd(), "docs/training-book");
const mime = { ".html": "text/html", ".css": "text/css", ".js": "application/javascript", ".md": "text/markdown", ".svg": "image/svg+xml", ".png": "image/png" };

const server = http.createServer((req, res) => {
  let reqPath = req.url.split("?")[0];
  if (reqPath === "/") reqPath = "/index.html";
  const fp = path.join(root, decodeURIComponent(reqPath));
  if (!fs.existsSync(fp) || fs.statSync(fp).isDirectory()) { res.writeHead(404); res.end("Not found"); return; }
  res.writeHead(200, { "Content-Type": mime[path.extname(fp)] || "application/octet-stream" });
  fs.createReadStream(fp).pipe(res);
});

(async () => {
  await new Promise(r => server.listen(0, "127.0.0.1", r));
  const port = server.address().port;
  const browser = await puppeteer.launch({ headless: "new" });
  const chapters = ["index","ch01","ch02","ch03","ch04","ch05","ch06","ch07","ch08","ch09","ch10","ch11","ch11a","ch11b","ch11c","ch11d","ch11e","ch12","appendices"];

  for (const vp of [{ name: "Desktop", w: 1280, h: 900 }, { name: "Mobile", w: 390, h: 844 }]) {
    console.log(`\n========== ${vp.name} (${vp.w}x${vp.h}) ==========`);
    const page = await browser.newPage();
    await page.setViewport({ width: vp.w, height: vp.h });
    
    for (const ch of chapters) {
      await page.goto(`http://127.0.0.1:${port}/#/${ch}`, { waitUntil: "networkidle0" });
      await new Promise(r => setTimeout(r, 300));
      
      const report = await page.evaluate(() => {
        const docW = document.documentElement.clientWidth;
        const docScrollW = document.documentElement.scrollWidth;
        const art = document.getElementById("article");
        if (!art) return { error: "no article" };
        const artRect = art.getBoundingClientRect();
        const artStyle = getComputedStyle(art);
        const artInnerW = artRect.width - parseFloat(artStyle.paddingLeft) - parseFloat(artStyle.paddingRight);
        
        const overflows = [];
        art.querySelectorAll("*").forEach(el => {
          const r = el.getBoundingClientRect();
          if (r.width === 0 && r.height === 0) return;
          if (r.right > docW + 5 || r.width > artInnerW + 5) {
            overflows.push({
              tag: el.tagName.toLowerCase() + (el.className ? "." + String(el.className).trim().replace(/\s+/g, ".") : ""),
              w: Math.round(r.width),
              right: Math.round(r.right),
              scrollW: el.scrollWidth,
              src: (el.getAttribute("src") || el.textContent || "").trim().slice(0, 50)
            });
          }
        });
        
        return { docW, docScrollW, artInnerW: Math.round(artInnerW), overflowCount: overflows.length, overflows: overflows.slice(0, 5) };
      });
      
      if (report.overflowCount > 0 || report.docScrollW > report.docW + 10) {
        console.log(`[${ch}] docW=${report.docW} scrollW=${report.docScrollW} artInner=${report.artInnerW} OVERFLOWS=${report.overflowCount}`);
        report.overflows.forEach(o => console.log(`  - ${