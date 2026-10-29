# Toroidal 3D Visualization Concept: The Dual-View Gameplay Experience

**Goal:** Bridge the gap between the mathematically symmetric $T^2$ toroidal ruleset and an intuitive, tactile human gameplay experience—**without modifying a single line of the core game engine (`GridManager`), AI player (`AIPlayer`), or neural network (`ConvNetwork`)**.

**Status:** Architectural & UX Blueprint (Phase 3.5 / Phase 4 Exploration)  
**Last Updated:** 2026-10-29  
**Dependencies:** Purely View/Projection Layer inside `GameScene` (`game.js`) + Optional Lightweight 3D Companion Canvas.

---

## 1. Architectural Principle: 100% Core Engine Isolation

A strict design requirement is that **all visual tilting, infinite panning, and 3D surface folding happen purely in the View-Projection Layer**, leaving the core game state and AI untouched:

- **Unchanged Canonical State (`GridManager`):**  
  `GridManager` continues to operate exclusively on canonical integer grid coordinates:
  $$(g_x, g_y) \in \{0, \dots, 9\} \times \{0, \dots, 9\}, \quad r \in \{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$$
  `isValidMove`, `placeTriangle`, `claimEnclosedTriangles`, `getLargestContiguous`, `AIPlayer`, and `ConvNetwork` remain 100% unchanged.

- **Forward Projection (Logical Grid $\to$ Screen Rendering):**  
  `GameScene` maintains a continuous camera pan offset $(\Delta u, \Delta v) \in \mathbb{R}^2$ (in cell units, wrapped modulo $10$) and an optional isometric/tilt transform matrix $\mathbf{M}_{\text{tilt}}$. Every frame (or on state/pan change), the renderer reads `gridManager.grid[gy][gx]`, shifts by $(\Delta u, \Delta v)$, applies $\mathbf{M}_{\text{tilt}}$, and draws the triangles.

- **Inverse Projection (Screen Pointer $\to$ Logical Grid):**  
  When the human player moves the cursor or taps at screen coordinates $(s_x, s_y)$:
  1. Apply $\mathbf{M}_{\text{tilt}}^{-1}$ to convert $(s_x, s_y)$ into continuous board coordinates $(u, v)$.
  2. Subtract the pan offset: $(u', v') = (u - \Delta u, v - \Delta v)$.
  3. Wrap modulo $10$ into canonical logical cell coordinates:
     $$g_x = \bigl(\lfloor u' \rfloor \bmod 10 + 10\bigr) \bmod 10, \quad g_y = \bigl(\lfloor v' \rfloor \bmod 10 + 10\bigr) \bmod 10$$
  4. Compute the sub-cell fractional offset $\bigl(u' - \lfloor u' \rfloor, v' - \lfloor v' \rfloor\bigr)$ to determine the triangle quadrant $r \in \{0^\circ, 90^\circ, 180^\circ, 270^\circ\}$.
  5. Pass $(g_x, g_y, r)$ directly to `gridManager.isValidMove(gx, gy, r)` and `gridManager.placeTriangle(gx, gy, r, player)`.

---

## 2. Primary Play Surface: The Draggable Tilted Plane & Field-of-View Modes

To solve the two classic flaws of pure 3D board games—**back-side occlusion** (hiding 50%+ of the board) and **grazing-angle touch errors**—the primary interactive board is rendered as an **angled / isometric 2.5D tabletop plane** that the player can smoothly drag in any direction to center any border battle.

We define two Field-of-View (FoV) modes governing whether pieces ever duplicate on screen:

```
┌──────────────────────────────────────────────────────────────────────────────┐
│  MODE A: STRICT ZERO DUPLICATION (10×10 Isometric Stencil Slab)              │
│                                                                              │
│         ╱───────────────────────────────╲   ◄── Exact 10×10 Clipping Mask    │
│        ╱  ◀── As a piece slides off the  ╲      • All 100 cells visible      │
│       ╱       right edge by Δx px, its    ╲     • ZERO duplicate pieces      │
│      ╱        wrapped part slides in from  ╲    • Smooth conveyor-belt pan   │
│     ╱         the left edge by Δx px  ──▶   ╲                                │
│    ╱─────────────────────────────────────────╲                               │
├──────────────────────────────────────────────────────────────────────────────┤
│  MODE B: 10×10 ACTIVE CORE + 1-CELL GHOST RING (12×12 Window)                │
│                                                                              │
│       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░ ◄── 1-Cell Ghost Ring (30% α)    │
│      ░░┌─────────────────────────────────┐░░    • Shows cross-border holes   │
│     ░░ │   10×10 Primary Domain (100% α) │ ░░     even without dragging      │
│      ░░└─────────────────────────────────┘░░    • Fades smoothly into 100%   │
│       ░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░░       opacity when dragged in    │
└──────────────────────────────────────────────────────────────────────────────┘
```

### Mode A: Strict Zero Duplication — The $10 \times 10$ Conveyor-Belt Slab (Recommended Default)
- **How It Works:**  
  The visible board is clipped by a crisp **$10 \times 10$ Isometric Geometry / Stencil Mask** (`width = 10 × cellSize`, `height = 10 × cellSize` in board space).
- **Why Nothing Ever Duplicates:**  
  Because the clipping window's dimensions exactly match the fundamental period $(10, 10)$ of the torus:
  - Every interior cell appears **strictly once** on screen.
  - A cell currently straddling the boundary during a smooth pixel drag has fraction $p \in (0, 1)$ visible on one edge and fraction $1 - p$ visible on the opposite edge, summing to **1.0 copy** at all times.
  - All 100 cells (400 triangle slots) remain simultaneously visible—zero blind spots and zero visual clutter.

### Mode B: Controlled Duplication — The 1-Cell Translucent "Ghost Ring" ($12 \times 12$ Window)
- **How It Works:**  
  Extends the visible window by 1 cell outside each of the 4 borders ($u, v \in [-1, 11)$), rendering the outer 1-cell border at **25–30% opacity** outside a subtle glowing frame around the $10 \times 10$ core.
- **Why It's Useful (Optional Toggle):**  
  Even when stationary, players can immediately spot 3-sided enclosure holes that wrap across the seam. During dragging, ghost cells glide smoothly in unison with the board and ramp up to 100% opacity as they cross inside the $10 \times 10$ frame.

---

## 3. Input Disambiguation: Drag-to-Pan vs. Tap-to-Place

To ensure dragging the infinite plane feels effortless on both mobile touchscreens and desktop mice without ever triggering accidental moves:

1. **`pointerdown`:** Record initial screen coordinates $(s_{x0}, s_{y0})$ and initial pan offset $(\Delta u_0, \Delta v_0)$. Set `isDragging = false`.
2. **`pointermove`:** Compute Euclidean screen displacement $d = \hypot(s_x - s_{x0}, s_y - s_{y0})$.
   - If $d < 8\text{ px}$: Treat as hover/aiming—update the triangle cursor preview at the wrapped logical slot $(g_x, g_y, r)$.
   - If $d \ge 8\text{ px}$: Set `isDragging = true`, hide the placement cursor, and update $(\Delta u, \Delta v)$ continuously so the board glides under the finger/mouse at 60 FPS.
3. **`pointerup`:**
   - If `!isDragging`: Execute triangle placement at $(g_x, g_y, r)$.
   - If `isDragging`: Finish pan gesture without placing a piece.
4. **"Center Origin" Quick Action:** A small HUD button (or double-tap outside the grid) smoothly tweens $(\Delta u, \Delta v) \to (0, 0)$.

---

## 4. Secondary Companion View: The Hinged 3D Closed Surface

Alongside the primary tilted plane (docked to the side on desktop or as a compact, collapsible Picture-in-Picture / top-corner holo-widget on mobile portrait), a **3D Companion View** shows how the flat grid closes into a 3D surface in real time.

### Geometric Insight: Rigid Hinged Triangles vs. Smooth Torus
If all 400 triangles remain strictly rigid flat polygons that **rotate along their shared edges like origami hinges**, they form a **Faceted Polyhedral Torus** (crystalline low-poly ring) rather than a stretched smooth donut.

We can parameterize the 3D hinge-folding with two continuous folding angles $\alpha, \beta \in [0, 1]$:
1. **$\alpha = 0, \beta = 0$ (Flat Sheet):** The 400 triangles lie flat in the $10 \times 10$ plane, matching the primary play surface.
2. **$\alpha \in (0 \to 1), \beta = 0$ (First Hinge Fold — Cylinder):** Each row hinges by $36^\circ$ ($360^\circ / 10$) along its horizontal shared edges, rolling the flat sheet into a closed 10-sided **3D Cylinder** where the top and bottom edges snap together.
3. **$\alpha = 1, \beta \in (0 \to 1)$ (Second Hinge Fold — Closed Faceted Torus):** The cylinder hinges by $36^\circ$ per column around the major ring axis until the left and right ends meet and lock into a closed **10×10 Faceted Torus**.

### Synchronized Visual Feedback Between Views
- **Live Camera Sync:** As the player drags the primary tilted plane by $(\Delta u, \Delta v)$, the 3D torus smoothly rotates by $(\Delta \theta = \frac{2\pi}{10}\Delta u, \, \Delta \phi = \frac{2\pi}{10}\Delta v)$ so the region currently centered on the flat board always faces the player on the 3D model.
- **Cross-Border Enclosure Highlight:** When a move claims enclosed triangles across the border seam, the 3D companion view pulses the closed loop on the 3D torus (and optionally plays a quick unfold/refold hinge animation on request).
- **Future Phase 4 Compatibility:** Because `GridManager` already supports arbitrary triangulated 3D manifold graphs $G = (V_T, E_{\text{edge}}, E_{\text{vertex}})$, this exact same Dual-View paradigm (Unfolded 2D Net + Hinged 3D Polyhedron) will naturally extend to Octahedra, Icosahedra, and higher-genus surfaces in Phase 4.

---

## 5. Phased Implementation Plan (Zero-Risk Rollout)

1. **Step 1 — Draggable Tilted Plane (`GameScene` in `game.js`):**
   - Add continuous toroidal pan offset $(\Delta u, \Delta v)$ + drag-vs-tap disambiguation ($8\text{ px}$ threshold).
   - Add the **$10 \times 10$ Non-Duplicating Stencil Slab** (Mode A) + optional **1-Cell Ghost Ring** toggle (Mode B) and **2D Flat vs. Isometric Tilt** toggle.
2. **Step 2 — 3D Hinged Companion Viewport:**
   - Add a lightweight, self-contained 3D renderer (Three.js or pure Canvas3D projection of the 400 triangle faces) reading directly from `gridManager.grid`.
   - Include an interactive **"Fold / Unfold" slider** ($\alpha, \beta \in [0, 1]$) and synchronized toroidal rotation when the player pans the main board.
