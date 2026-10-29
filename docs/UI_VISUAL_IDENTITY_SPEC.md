# Triantica UI & Visual Identity Specification: *"NEON MANIFOLD"*

**Document Status:** Canonical Visual Design System (v1.0)  
**Target Surface:** `triantica/index.html` & `triantica/game.js` (`MenuScene`, `GameScene`, and Step 2 3D Companion View)  
**Architectural Constraint:** 100% View-Projection & Rendering Layer only. Zero changes to `GridManager`, `AIPlayer`, or core game logic.

---

## 1. Creative Direction & Core Decisions

**Theme:** *Neon Manifold* — a Tron-inspired high-contrast cyber-grid where Player 1 (**Red / Ares**) and Player 2 (**Blue / Tron**) battle to close contiguous light-circuit territories across a finite toroidal manifold ($T^2$).

### Locked Design Decisions (2026-10-29)
1. **Title Screen 3D Centerpiece (Choice `1-2`):** **Geodesic Icosahedron (D20 Core)** — a 20-face equilateral triangle wireframe polyhedron rotating in 3D space on the main menu, with live territory color-swapping between Red and Blue faces and stochastic cyberpunk wireframe/RGB-split glitch bursts.
2. **Board & Matrix Grid Accent (Choice `2-A`):** **Deep Cyber-Cyan (`#0D4054` grid mesh + `#00F0FF` slab border)** — pure *Tron: Legacy* obsidian glass aesthetic with crisp cyan vertex crosshairs and glowing laser-cyan slab boundaries.
3. **Triangle Interior Style (Choice `3-B`):** **High-Opacity Neon Solid (`82%` saturated fill + bright neon rim)** — maximum legibility and punch at both `1.0x` ($10 \times 10$) and `2.5x` ($4 \times 4$) zoom levels, with a subtle screen-space inset (`~1.2 px`) so individual power cells and circuit seams remain crisp.

---

## 2. Color Palette & Token Specification

| Token Name | Hex / Value | Usage |
| :--- | :--- | :--- |
| **`BG_VOID`** | `#03060C` | Primary canvas & HTML body background |
| **`SLAB_SURFACE`** | `#060B14` | $10 \times 10$ tabletop board surface (Obsidian Cyber-Glass) |
| **`SLAB_BEVEL_LEFT`** | `#091526` | 2.5D tabletop slab front-left thickness bevel |
| **`SLAB_BEVEL_RIGHT`** | `#050D1A` | 2.5D tabletop slab bottom-right thickness bevel |
| **`GRID_LINE_CYAN`** | `#0D4054` (`0.88` $\alpha$) | Primary $10 \times 10$ internal matrix grid lines |
| **`GRID_NODE_CYAN`** | `#00F0FF` (`0.38` $\alpha$) | Micro-nodes at grid intersections $(i, j)$ |
| **`SLAB_BORDER_NEON`** | `#00F0FF` (`1.0` $\alpha$) | Crisp outer frame of the $10 \times 10$ fundamental domain + outer halo (`0.22` $\alpha$) |
| **`TORUS_SEAM_LASER`** | `#00FFCC` (`0.65` $\alpha$) | Canonical $(0,0)$ wrap-around seam indicator when panned |
| **`P1_RED_FILL`** | `#E6193C` (`0.82` $\alpha$) | Player 1 (Red) high-opacity neon solid interior |
| **`P1_RED_RIM`** | `#FF4D6D` (`1.0` $\alpha$) | Player 1 (Red) glowing perimeter stroke |
| **`P2_BLUE_FILL`** | `#0066FF` (`0.82` $\alpha$) | Player 2 (Blue) high-opacity neon solid interior |
| **`P2_BLUE_RIM`** | `#00F0FF` (`1.0` $\alpha$) | Player 2 (Blue) glowing perimeter stroke |
| **`HUD_PANEL_BG`** | `#071120` (`0.92` $\alpha$) | Chamfered HUD panels, scoreboard, and cyber-buttons |
| **`TEXT_PRIMARY`** | `#E6F9FF` | Primary headers and HUD labels |
| **`TEXT_MUTED`** | `#6495ED` / `#5CE1E6` | Subtitles, telemetry hints, and secondary readouts |

---

## 3. Typography System

Loaded via Google Fonts in `triantica/index.html` (with clean fallback to `Arial` / `monospace`):
- **Display / Headers (`Orbitron`):** Used for `TRIANTICA` title, menu headers (`SELECT AI DIFFICULTY`, `CHOOSE YOUR SIDE`), primary action button (`PLACE TRIANGLE`), and game-over banners.
- **Technical Telemetry (`Share Tech Mono`):** Used for the live scoreboard, HUD pill toggles (`◈ 2.5D TILT`, `👻 GHOST`, `⌖ 2.5×`), and debug telemetry.

---

## 4. Title Screen Centerpiece: Glitching D20 Icosahedron (`MenuScene`)

1. **Geometry:**
   - Canonical regular icosahedron with **12 vertices** (golden ratio $\phi = \frac{1+\sqrt{5}}{2}$) and **20 equilateral triangular faces**.
   - Positioned at $(300, 215)$ on the main menu between the `TRIANTICA` header and the action buttons.
2. **Continuous 3D Rotation & Backface Layering:**
   - Rotates smoothly around $Y$ and $X$ axes every frame (`update(time, delta)`).
   - Back-facing triangles are rendered as faint translucent wireframe (`alpha = 0.16`) to create true 3D depth; front-facing triangles are painter-sorted by average $Z$-depth.
3. **Territory Color-Swapping Automaton:**
   - Each of the 20 faces holds a state $s_f \in \{0\text{ (Neutral Cyan)}, 1\text{ (P1 Red)}, 2\text{ (P2 Blue)}\}$.
   - Every `260 ms`, a contagious wave propagates across adjacent faces, flipping clusters between Red (`#FF2A54`) and Blue (`#00F0FF`) with high-opacity glowing fills (`0.78` $\alpha$) and bright neon edges.
4. **Wireframe Glitch & Chromatic Aberration:**
   - Every `2.2–4.0 s`, a short glitch window (`90–160 ms`) triggers:
     - **Horizontal Slice Jitter:** Random vertices snap horizontally by $\pm 6\text{ to }14\text{ px}$.
     - **RGB Split Wireframe:** Offset red (`#FF2A54`) and cyan (`#00F0FF`) ghost wireframes render alongside the main mesh.
     - **Telemetry Scanlines:** Brief horizontal laser ticks flicker across the stage.

---

## 5. Playfield & HUD Components (`GameScene`)

1. **Obsidian Tabletop Slab & Neon Frame:**
   - The $10 \times 10$ stencil slab uses `SLAB_SURFACE` (`#060B14`) with a dual-pass `SLAB_BORDER_NEON` (`#00F0FF`) outer halo (`width = 7, alpha = 0.22`) and crisp core stroke (`width = 2.5, alpha = 0.95`).
2. **High-Opacity Neon Solid Triangles (Choice `3-B`):**
   - Each placed triangle is slightly inset (`~3%` toward its centroid) so adjacent triangles display a razor-sharp dark circuit channel between them, filled at `0.82` alpha (`#E6193C` for P1, `#0066FF` for P2) and bordered by a `1.6 px` neon rim (`#FF4D6D` / `#00F0FF`).
3. **Chamfered Cyber-Buttons:**
   - All menu buttons, HUD pills, scoreboard frame, and bottom control bar (`↻`, `PLACE TRIANGLE`, `⚑`) use 45° chamfered octagon geometry with neon rim strokes and hover/press glow states.
