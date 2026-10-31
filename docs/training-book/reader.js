/* ══════════════════════════════════════════════════════════════
   TRIANTICA TRAINING BOOK — Zero-Build SPA Reader
   Loads Markdown chapters, renders with Marked.js + KaTeX,
   builds navigation, search, deep-linking, progress tracking.
   ══════════════════════════════════════════════════════════════ */

(function () {
  'use strict';

  // ── Chapter Manifest ──
  const CHAPTERS = [
    { id: 'index',      file: 'README.md',                               num: '',   title: 'Book Index',                    sub: false },
    { id: 'ch01',       file: '01-the-game.md',                          num: '1',  title: 'The Game',                      sub: false },
    { id: 'ch02',       file: '02-rl-fundamentals.md',                   num: '2',  title: 'RL Fundamentals',               sub: false },
    { id: 'ch03',       file: '03-architecture-journey.md',              num: '3',  title: 'The Architecture Journey',      sub: false },
    { id: 'ch04',       file: '04-hardware-m2-ultra.md',                 num: '4',  title: 'Hardware: M2 Ultra',            sub: false },
    { id: 'ch05',       file: '05-parameters-explored.md',               num: '5',  title: 'Parameters Explored',           sub: false },
    { id: 'ch06',       file: '06-training-strategies.md',               num: '6',  title: 'Training Strategies',           sub: false },
    { id: 'ch07',       file: '07-critical-bugs-and-audits.md',          num: '7',  title: 'Critical Bugs & Audits',        sub: false },
    { id: 'ch08',       file: '08-the-heuristic-wall.md',               num: '8',  title: 'The Heuristic Wall',            sub: false },
    { id: 'ch09',       file: '09-experiment-reference.md',             num: '9',  title: 'Experiment Reference',          sub: false },
    { id: 'ch10',       file: '10-key-takeaways.md',                    num: '10', title: 'Key Takeaways',                sub: false },
    { id: 'ch11',       file: '11-deep-dive-convolutions-mcts.md',      num: '11', title: 'Deep Dive Overview',           sub: false },
    { id: 'ch11a',      file: '11a-replay-and-conv-encoding.md',        num: 'A',  title: 'Replay Buffer & ConvNet Encoding', sub: true },
    { id: 'ch11b',      file: '11b-milestone-runs-tf-b5-to-b13.md',    num: 'B',  title: 'Milestone Runs (TF-B5→B13)',  sub: true },
    { id: 'ch11c',      file: '11c-actor-critic-and-batchnorm.md',      num: 'C',  title: 'Actor-Critic & BatchNorm',    sub: true },
    { id: 'ch11d',      file: '11d-mcts-theory-and-implementation.md',  num: 'D',  title: 'MCTS Theory & Implementation', sub: true },
    { id: 'ch11e',      file: '11e-quiescence-mcts-and-torus-mastery.md', num: 'E', title: 'Quiescence MCTS & Torus Mastery', sub: true },
    { id: 'ch12',       file: '12-toroidal-3d-and-telemetry.md',        num: '12', title: 'Toroidal 3D & Telemetry',     sub: false },
    { id: 'appendices', file: 'appendices.md',                          num: '★',  title: 'Appendices',                   sub: false },
  ];

  // ── State ──
  let currentChapter = null;
  let searchIndex = [];
  let searchDebounce = null;
  let chapterCache = {};

  // ── DOM Refs ──
  const $ = id => document.getElementById(id);
  const $article       = $('article');
  const $splash        = $('splash');
  const $navTree       = $('nav-tree');
  const $searchInput   = $('search-input');
  const $searchResults = $('search-results');
  const $sidebar       = $('sidebar');
  const $overlay       = $('sidebar-overlay');
  const $progressFill  = document.querySelector('.progress-fill');
  const $chapterNav    = $('chapter-nav');
  const $prevBtn       = $('prev-chapter');
  const $prevTitle     = $('prev-chapter-title');
  const $nextBtn       = $('next-chapter');
  const $nextTitle     = $('next-chapter-title');

  // ── Initialize ──
  async function init() {
    buildNav();
    setupEvents();
    configureMarked();
    configureMermaid();
    await buildSearchIndex();
    route();
    window.addEventListener('hashchange', route);
  }

  // ── Configure Marked ──
  function configureMarked() {
    if (typeof marked === 'undefined') return;

    const renderer = new marked.Renderer();

    // Images → screenshot/diagram containers
    const origImage = renderer.image.bind(renderer);
    renderer.image = function (token) {
      const href = token.href || '';
      const text = token.text || '';
      if (href.includes('assets/') || /\.(png|jpg|jpeg|webp|gif)$/i.test(href)) {
        return `<div class="screenshot-container"><img src="${href}" alt="${escAttr(text)}" loading="lazy"><div class="screenshot-caption">${escHtml(text)}</div></div>`;
      }
      if (/\.svg$/i.test(href)) {
        return `<div class="diagram-container"><img src="${href}" alt="${escAttr(text)}" loading="lazy"><div class="diagram-caption">${escHtml(text)}</div></div>`;
      }
      return origImage(token);
    };

    // Code blocks → detect ASCII diagrams
    const origCode = renderer.code.bind(renderer);
    renderer.code = function (token) {
      const code = token.text || '';
      const lang = token.lang || '';
      if (lang === 'ascii' || lang === 'diagram') {
        return `<pre class="ascii-diagram"><code>${escHtml(code)}</code></pre>`;
      }
      // Auto-detect box-drawing characters in large blocks
      if (code.length > 80 && /[┌┐└┘├┤┬┴┼═║╔╗╚╝╠╣╦╩╬│┤├─┼┬┴┏┓┗┛►◄▼▲]/.test(code.substring(0, 200))) {
        return `<pre class="ascii-diagram"><code>${escHtml(code)}</code></pre>`;
      }
      return origCode(token);
    };

    // Headings → add IDs for TOC
    const origHeading = renderer.heading.bind(renderer);
    renderer.heading = function (token) {
      const text = token.text || '';
      const depth = token.depth || 1;
      const id = slugify(text);
      return `<h${depth} id="${id}">${text}</h${depth}>`;
    };

    marked.setOptions({
      renderer,
      gfm: true,
      breaks: false,
    });
  }

  // ── Configure Mermaid ──
  function configureMermaid() {
    if (typeof mermaid === 'undefined') return;
    mermaid.initialize({
      startOnLoad: false,
      theme: 'dark',
      themeVariables: {
        primaryColor: '#071120',
        primaryTextColor: '#E6F9FF',
        primaryBorderColor: '#00F0FF',
        lineColor: '#0D4054',
        secondaryColor: '#0A1A30',
        tertiaryColor: '#060B14',
        fontFamily: 'Share Tech Mono, monospace',
      },
    });
  }

  // ── Routing ──
  function route() {
    const hash = location.hash.replace(/^#\/?/, '');
    const chapter = CHAPTERS.find(c => c.id === hash) || CHAPTERS[0];
    loadChapter(chapter);
  }

  // ── Load Chapter ──
  async function loadChapter(chapter) {
    // Splash while loading
    $article.style.display = 'none';
    $chapterNav.style.display = 'none';
    $splash.style.display = 'flex';

    try {
      let md;
      if (chapterCache[chapter.id]) {
        md = chapterCache[chapter.id];
      } else {
        const resp = await fetch(chapter.file);
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        md = await resp.text();
        chapterCache[chapter.id] = md;
      }

      // Render
      let html;
      if (typeof marked !== 'undefined') {
        html = marked.parse(md);
      } else {
        html = `<pre>${escHtml(md)}</pre>`;
      }

      $article.innerHTML = html;
      $article.style.display = 'block';
      $splash.style.display = 'none';
      $chapterNav.style.display = 'flex';

      // Fade in
      $article.classList.remove('fade-in');
      void $article.offsetWidth;
      $article.classList.add('fade-in');

      // Post-processing
      postProcess(chapter);
      currentChapter = chapter;

      // Save progress
      saveProgress(chapter.id);
      window.scrollTo({ top: 0, behavior: 'instant' });
    } catch (err) {
      $article.innerHTML = renderError(chapter, err);
      $article.style.display = 'block';
      $splash.style.display = 'none';
    }
  }

  // ── Post-Processing Pipeline ──
  function postProcess(chapter) {
    renderMath();
    renderMermaidDiagrams();
    wrapSVGDiagrams();
    addHeadingAnchors();
    buildChapterNav(chapter);
    updateActiveNav(chapter.id);
    generateInPageTOC();
    convertIndexLinks();
    updateProgressBarFromProgress();
  }

  // ── KaTeX Math Rendering ──
  function renderMath() {
    if (typeof renderMathInElement === 'undefined') return;
    renderMathInElement($article, {
      delimiters: [
        { left: '$$', right: '$$', display: true },
        { left: '$', right: '$', display: false },
        { left: '\\(', right: '\\)', display: false },
        { left: '\\[', right: '\\]', display: true },
      ],
      throwOnError: false,
    });
  }

  // ── Mermaid Diagrams ──
  async function renderMermaidDiagrams() {
    if (typeof mermaid === 'undefined') return;
    const blocks = $article.querySelectorAll('code.language-mermaid');
    for (const block of blocks) {
      const pre = block.parentElement;
      const id = 'mermaid-' + Math.random().toString(36).substr(2, 9);
      try {
        const { svg } = await mermaid.render(id, block.textContent);
        const div = document.createElement('div');
        div.className = 'diagram-container';
        div.innerHTML = svg;
        pre.replaceWith(div);
      } catch { /* leave as-is */ }
    }
  }

  // ── Wrap SVG image references in diagram containers ──
  function wrapSVGDiagrams() {
    $article.querySelectorAll('img').forEach(img => {
      if (img.src.endsWith('.svg')) {
        if (img.closest('.diagram-container')) return; // already wrapped
        const wrap = document.createElement('div');
        wrap.className = 'diagram-container';
        const cap = document.createElement('div');
        cap.className = 'diagram-caption';
        cap.textContent = img.alt || '';
        img.parentNode.insertBefore(wrap, img);
        wrap.appendChild(img);
        wrap.appendChild(cap);
      }
    });
  }

  // ── Heading anchors (clickable § links) ──
  function addHeadingAnchors() {
    $article.querySelectorAll('h2, h3, h4').forEach(h => {
      if (h.querySelector('.heading-anchor')) return;
      const a = document.createElement('a');
      a.className = 'heading-anchor';
      a.href = '#' + h.id;
      a.title = 'Link to this section';
      a.innerHTML = '§';
      h.appendChild(a);
    });
  }

  // ── In-page TOC (floating for long chapters) ──
  function generateInPageTOC() {
    const existing = $('in-page-toc');
    if (existing) existing.remove();

    const headings = $article.querySelectorAll('h2, h3');
    if (headings.length < 4) return;

    const toc = document.createElement('div');
    toc.id = 'in-page-toc';
    toc.className = 'in-page-toc';
    toc.innerHTML = '<div class="toc-title">On This Page</div>' +
      Array.from(headings).map(h => {
        const indent = h.tagName === 'H3' ? 'toc-h3' : h.tagName === 'H4' ? 'toc-h4' : '';
        return `<a class="toc-link ${indent}" href="#${h.id}">${h.textContent.replace(/§$/, '').trim()}</a>`;
      }).join('');

    $article.insertBefore(toc, $article.firstChild);
  }

  // ── Build Navigation ──
  function buildNav() {
    let html = '<div class="nav-section-label">Chapters</div>';
    let inSubGroup = false;

    CHAPTERS.forEach(ch => {
      if (ch.sub && !inSubGroup) {
        html += '<div class="nav-section-label" style="padding-left:34px">Deep Dive</div>';
        inSubGroup = true;
      }
      if (!ch.sub && inSubGroup) inSubGroup = false;

      const cls = ch.sub ? 'nav-item sub' : 'nav-item';
      const prefix = ch.sub ? `<span class="nav-item-num">↳</span>` :
                     ch.num ? `<span class="nav-item-num">${ch.num}</span>` : '';
      html += `<a class="${cls}" href="#/${ch.id}" data-chapter="${ch.id}">${prefix}<span>${ch.title}</span></a>`;
    });
    $navTree.innerHTML = html;
  }

  function updateActiveNav(id) {
    $navTree.querySelectorAll('.nav-item').forEach(el =>
      el.classList.toggle('active', el.dataset.chapter === id));
    // Scroll active item into view in sidebar
    const active = $navTree.querySelector('.nav-item.active');
    if (active) active.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
  }

  // ── Chapter Prev/Next ──
  function buildChapterNav(chapter) {
    const idx = CHAPTERS.indexOf(chapter);
    const prev = idx > 0 ? CHAPTERS[idx - 1] : null;
    const next = idx < CHAPTERS.length - 1 ? CHAPTERS[idx + 1] : null;

    if (prev) {
      $prevBtn.href = '#/' + prev.id;
      $prevTitle.textContent = prev.title;
      $prevBtn.style.visibility = 'visible';
    } else {
      $prevBtn.style.visibility = 'hidden';
    }
    if (next) {
      $nextBtn.href = '#/' + next.id;
      $nextTitle.textContent = next.title;
      $nextBtn.style.visibility = 'visible';
    } else {
      $nextBtn.style.visibility = 'hidden';
    }
  }

  // ── Search ──
  async function buildSearchIndex() {
    searchIndex = [];
    const promises = CHAPTERS.map(async ch => {
      try {
        let text;
        if (chapterCache[ch.id]) {
          text = chapterCache[ch.id];
        } else {
          const resp = await fetch(ch.file);
          if (!resp.ok) return;
          text = await resp.text();
          chapterCache[ch.id] = text;
        }
        const clean = text
          .replace(/^>\s*.+$/gm, '')
          .replace(/^```[\s\S]*?```/gm, '')
          .replace(/[#*_`~]/g, '')
          .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
          .replace(/\$[^$]+\$/g, '')
          .replace(/!\[.*?\]\(.*?\)/g, '')
          .slice(0, 8000);
        searchIndex.push({ id: ch.id, title: ch.title, text: clean });
      } catch { /* skip */ }
    });
    await Promise.all(promises);
  }

  function doSearch(query) {
    if (!query || query.length < 2) {
      $searchResults.classList.remove('visible');
      return;
    }
    const q = query.toLowerCase();
    const results = searchIndex
      .map(ch => {
        const idx = ch.text.toLowerCase().indexOf(q);
        if (idx === -1) return null;
        const start = Math.max(0, idx - 50);
        const end = Math.min(ch.text.length, idx + query.length + 80);
        const snippet = (start > 0 ? '…' : '') +
          ch.text.substring(start, end)
            .replace(new RegExp(`(${escRx(query)})`, 'gi'), '<mark>$1</mark>') +
          (end < ch.text.length ? '…' : '');
        return { id: ch.id, title: ch.title, snippet };
      })
      .filter(Boolean)
      .slice(0, 8);

    $searchResults.innerHTML = results.length === 0
      ? '<div class="search-result-item"><div class="search-result-title">No results found</div></div>'
      : results.map(r =>
          `<div class="search-result-item" data-chapter="${r.id}">
            <div class="search-result-title">§ ${r.title}</div>
            <div class="search-result-snippet">${r.snippet}</div>
          </div>`
        ).join('');
    $searchResults.classList.add('visible');
  }

  // ── Progress Tracking ──
  function saveProgress(chapterId) {
    try {
      const visited = JSON.parse(localStorage.getItem('triantica-progress') || '[]');
      if (!visited.includes(chapterId)) {
        visited.push(chapterId);
        localStorage.setItem('triantica-progress', JSON.stringify(visited));
      }
    } catch { /* no localStorage */ }
  }

  function getProgress() {
    try { return JSON.parse(localStorage.getItem('triantica-progress') || '[]'); }
    catch { return []; }
  }

  // ── Progress Bar ──
  function updateProgress() {
    const scrollH = document.documentElement.scrollHeight - window.innerHeight;
    const pct = scrollH > 0 ? (window.scrollY / scrollH) * 100 : 0;
    $progressFill.style.width = Math.min(100, pct) + '%';
  }

  // ── Sidebar ──
  function openSidebar() {
    $sidebar.classList.add('open');
    $overlay.classList.add('visible');
    document.body.style.overflow = 'hidden';
  }
  function closeSidebar() {
    $sidebar.classList.remove('open');
    $overlay.classList.remove('visible');
    document.body.style.overflow = '';
  }

  // ── Error Rendering ──
  function renderError(chapter, err) {
    return `<div class="hero">
      <div class="hero-badge">Error</div>
      <h1>Chapter Not Found</h1>
      <p class="lead">Could not load "${chapter.file}": ${err.message}</p>
      <a href="#/" class="chapter-nav-btn">← Back to Index</a>
    </div>`;
  }

  // ── Convert .md links to hash routes ──
  function convertIndexLinks() {
    $article.querySelectorAll('a[href$=".md"]').forEach(a => {
      const href = a.getAttribute('href');
      const ch = CHAPTERS.find(c => c.file === href || href.endsWith('/' + c.file));
      if (ch) {
        a.setAttribute('href', '#/' + ch.id);
        a.removeAttribute('target');
      }
    });
  }

  // ── Update progress bar from visited chapters ──
  function updateProgressBarFromProgress() {
    const visited = getProgress();
    const pct = (visited.length / CHAPTERS.length) * 100;
    if ($progressFill) $progressFill.style.width = pct + '%';
  }

  // ── Events ──
  function setupEvents() {
    // Scroll progress
    window.addEventListener('scroll', updateProgress, { passive: true });

    // Sidebar toggle
    $('sidebar-toggle')?.addEventListener('click', openSidebar);
    $('sidebar-close')?.addEventListener('click', closeSidebar);
    $('search-toggle-mobile')?.addEventListener('click', () => {
      openSidebar();
      setTimeout(() => $searchInput.focus(), 300);
    });
    $overlay.addEventListener('click', closeSidebar);

    // Search
    $searchInput.addEventListener('input', () => {
      clearTimeout(searchDebounce);
      searchDebounce = setTimeout(() => doSearch($searchInput.value.trim()), 200);
    });
    $searchInput.addEventListener('focus', () => {
      if ($searchInput.value.trim().length >= 2) $searchResults.classList.add('visible');
    });
    $searchResults.addEventListener('click', e => {
      const item = e.target.closest('.search-result-item');
      if (!item?.dataset.chapter) return;
      location.hash = '#/' + item.dataset.chapter;
      $searchInput.value = '';
      $searchResults.classList.remove('visible');
      closeSidebar();
    });
    document.addEventListener('click', e => {
      if (!e.target.closest('.sidebar-search')) $searchResults.classList.remove('visible');
    });

    // Keyboard shortcuts
    document.addEventListener('keydown', e => {
      const isInput = ['INPUT', 'TEXTAREA'].includes(document.activeElement.tagName);

      if (e.key === '/' && !e.ctrlKey && !e.metaKey && !isInput) {
        e.preventDefault();
        $searchInput.focus();
      }
      if (e.key === 'Escape') {
        $searchInput.blur();
        $searchResults.classList.remove('visible');
        closeSidebar();
      }
      // Arrow key navigation between chapters
      if (e.key === 'ArrowLeft' && !isInput && currentChapter) {
        const idx = CHAPTERS.indexOf(currentChapter);
        if (idx > 0) location.hash = '#/' + CHAPTERS[idx - 1].id;
      }
      if (e.key === 'ArrowRight' && !isInput && currentChapter) {
        const idx = CHAPTERS.indexOf(currentChapter);
        if (idx < CHAPTERS.length - 1) location.hash = '#/' + CHAPTERS[idx + 1].id;
      }
    });

    // Close sidebar on nav click (mobile)
    $navTree.addEventListener('click', e => {
      if (e.target.closest('.nav-item')) closeSidebar();
    });
  }

  // ── Helpers ──
  function escHtml(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
  }
  function escAttr(s) {
    return s.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
  function escRx(s) {
    return s.replace(/[.*+?^${}()|[\]\\]/g, '\\$&');
  }
  function slugify(text) {
    return text.toLowerCase()
      .replace(/[^\w\s-]/g, '')
      .replace(/\s+/g, '-')
      .replace(/-+/g, '-')
      .substring(0, 64);
  }

  // ── Boot ──
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }
})();
