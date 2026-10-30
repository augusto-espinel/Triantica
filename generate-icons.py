#!/usr/bin/env python3
"""
Generate Triantica PWA icons featuring the 'T' built from 7 contiguous
right-angle triangles (45-45-90 half-squares), matching the exact in-game
neon solid fill + laser rim + circuit grid aesthetic:

 /|/|/   <- Row 0: (col 0, rot 2 BR) | (col 1, rot 0 TL + rot 2 BR) | (col 2, rot 0 TL)
  ☑      <- Row 1: (col 1, rot 0 TL + rot 2 BR)
  |/     <- Row 2: (col 1, rot 0 TL)
"""

import os
from PIL import Image, ImageDraw, ImageFilter

SIZES = [72, 96, 128, 144, 152, 180, 192, 384, 512]

BG_COLOR = (3, 6, 12, 255)          # #03060C
GRID_SUB_COLOR = (8, 36, 51, 190)   # #082433
GRID_MAIN_COLOR = (13, 64, 84, 240) # #0D4054
NODE_COLOR = (0, 240, 255, 125)     # #00F0FF micro-nodes

# Exact in-game colors from drawProjectedTriangleOnGraphics (game.js):
# P1 (Ares Red): fill 0xE6193C (82% alpha), rim 0xFF4D6D (96% alpha)
# P2 (Tron Blue): fill 0x0066FF (82% alpha), rim 0x00F0FF (96% alpha)
RED_FILL = (230, 25, 60, 218)
RED_RIM = (255, 77, 109, 255)
RED_GLOW = (255, 42, 84, 130)

BLUE_FILL = (0, 102, 255, 218)
BLUE_RIM = (0, 240, 255, 255)
BLUE_GLOW = (0, 240, 255, 130)


def get_triangle_vertices(x0, y0, cell, col, row, rot, inset=0.042):
    u0 = x0 + col * cell
    v0 = y0 + row * cell
    u1 = u0 + cell
    v1 = v0 + cell

    p00 = (u0, v0)  # Top-Left
    p10 = (u1, v0)  # Top-Right
    p11 = (u1, v1)  # Bottom-Right
    p01 = (u0, v1)  # Bottom-Left

    if rot == 0:    # Top-Left right-angle triangle (|/) -> right angle at TL (p00)
        raw = [p01, p10, p00]
    elif rot == 1:  # Bottom-Left right-angle triangle (|\) -> right angle at BL (p01)
        raw = [p00, p01, p11]
    elif rot == 2:  # Bottom-Right right-angle triangle (/|) -> right angle at BR (p11)
        raw = [p11, p01, p10]
    elif rot == 3:  # Top-Right right-angle triangle (\|) -> right angle at TR (p10)
        raw = [p10, p00, p11]

    cx = sum(p[0] for p in raw) / 3.0
    cy = sum(p[1] for p in raw) / 3.0

    return [
        (p[0] + (cx - p[0]) * inset, p[1] + (cy - p[1]) * inset)
        for p in raw
    ]


def draw_closed_polygon_line(draw, pts, fill, width):
    seq = list(pts) + [pts[0]]
    draw.line(seq, fill=fill, width=width, joint="curve")
    r = width / 2.0
    for px, py in pts:
        draw.ellipse([px - r, py - r, px + r, py + r], fill=fill)


def render_icon(target_size):
    # Supersample 4x for crisp antialiasing
    scale = 4
    S = target_size * scale

    base = Image.new("RGBA", (S, S), BG_COLOR)
    grid_layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    gdraw = ImageDraw.Draw(grid_layer)

    # 3x3 board grid centered in the icon
    # 3 cells occupy 68% of icon width, leaving 16% margin on all sides (safe for maskable icons)
    cell = S * 0.228
    board_span = 3 * cell
    x0 = (S - board_span) / 2.0
    y0 = (S - board_span) / 2.0

    sub_w = max(2, int(round(S * 0.0045)))
    main_w = max(3, int(round(S * 0.0075)))

    # 1. Subtle diagonal sub-grid circuit traces inside each cell
    for r in range(3):
        for c in range(3):
            cx0 = x0 + c * cell
            cy0 = y0 + r * cell
            cx1 = cx0 + cell
            cy1 = cy0 + cell
            gdraw.line([cx0, cy0, cx1, cy1], fill=GRID_SUB_COLOR, width=sub_w)
            gdraw.line([cx0, cy1, cx1, cy0], fill=GRID_SUB_COLOR, width=sub_w)

    # 2. Primary 3x3 matrix grid lines
    for i in range(4):
        gx = x0 + i * cell
        gy = y0 + i * cell
        gdraw.line([gx, y0, gx, y0 + board_span], fill=GRID_MAIN_COLOR, width=main_w)
        gdraw.line([x0, gy, x0 + board_span, gy], fill=GRID_MAIN_COLOR, width=main_w)

    # 3. Glowing micro-nodes at grid intersections
    node_r = max(2, int(round(S * 0.009)))
    for i in range(4):
        for j in range(4):
            gx = x0 + i * cell
            gy = y0 + j * cell
            gdraw.ellipse([gx - node_r, gy - node_r, gx + node_r, gy + node_r], fill=NODE_COLOR)

    base = Image.alpha_composite(base, grid_layer)

    # 7 Contiguous Right-Angle Triangles forming the T:
    #   /|/|/
    #    ☑
    #    |/
    # Colors: Top bar Blue & Red, Stem Red & Blue
    triangles = [
        # Top bar (Row 0): /| |/| |/
        (0, 0, 2, 'blue'),  # Col 0, Bottom-Right (/|)
        (1, 0, 0, 'blue'),  # Col 1, Top-Left (|/)
        (1, 0, 2, 'red'),   # Col 1, Bottom-Right (/|) - center junction
        (2, 0, 0, 'blue'),  # Col 2, Top-Left (|/)
        # Stem middle (Row 1): ☑ (full square = 2 right-angle triangles)
        (1, 1, 0, 'red'),   # Col 1, Top-Left (|/)
        (1, 1, 2, 'blue'),  # Col 1, Bottom-Right (/|)
        # Stem bottom (Row 2): |/
        (1, 2, 0, 'blue'),  # Col 1, Top-Left (|/)
    ]

    glow_layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow_layer)

    tri_layer = Image.new("RGBA", (S, S), (0, 0, 0, 0))
    tri_draw = ImageDraw.Draw(tri_layer)

    rim_w = max(3, int(round(S * 0.011)))
    glow_w = max(6, int(round(S * 0.026)))

    for col, row, rot, color in triangles:
        verts = get_triangle_vertices(x0, y0, cell, col, row, rot, inset=0.042)
        if color == 'red':
            fill_c, rim_c, glow_c = RED_FILL, RED_RIM, RED_GLOW
        else:
            fill_c, rim_c, glow_c = BLUE_FILL, BLUE_RIM, BLUE_GLOW

        # Outer neon halo
        draw_closed_polygon_line(glow_draw, verts, glow_c, glow_w)

        # Solid neon fill + crisp laser rim
        tri_draw.polygon(verts, fill=fill_c)
        draw_closed_polygon_line(tri_draw, verts, rim_c, rim_w)

    blur_radius = max(1.5, S * 0.012)
    glow_layer = glow_layer.filter(ImageFilter.GaussianBlur(radius=blur_radius))

    base = Image.alpha_composite(base, glow_layer)
    base = Image.alpha_composite(base, tri_layer)

    return base.resize((target_size, target_size), Image.Resampling.LANCZOS)


def main():
    out_dirs = [
        "/Users/aespinel/.openclaw/workspace/triantica/icons",
        "/Users/aespinel/.openclaw/workspace/triantica/dist/icons"
    ]
    for d in out_dirs:
        os.makedirs(d, exist_ok=True)

    for sz in SIZES:
        img = render_icon(sz)
        filename = "apple-touch-icon.png" if sz == 180 else f"icon-{sz}.png"
        for d in out_dirs:
            path = os.path.join(d, filename)
            img.save(path, "PNG")
        print(f"Generated {filename} ({sz}x{sz})")


if __name__ == "__main__":
    main()
