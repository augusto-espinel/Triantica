#!/bin/bash
# Generate PWA icons with 7 contiguous triangles forming a T
# Matches in-game triangle appearance with edge-to-edge connections

mkdir -p icons

# Colors matching game (TRON BLU and ARES RED)
BLUE="#00F0FF"
RED="#FF2A54"
BG="#03060C"
STROKE_WIDTH=1

for size in 72 96 128 144 152 192 384 512; do
  s=$size
  
  # T-shape with 7 contiguous triangles:
  # Top bar: 5 triangles (alternating rotations for proper connection)
  # Stem: 2 triangles below center
  
  # Triangle geometry (equilateral-like, matching game visuals)
  # Base width and height scale with icon size
  tw=$((s*22/100))   # Triangle width (~22% of icon)
  th=$((s*19/100))   # Triangle height (~19% of icon)
  
  # Center point for the T
  cx=$((s*50/100))
  cy=$((s*48/100))
  
  # Top bar triangles (5 across)
  # y position for top bar
  ty=$((s*30/100))
  
  # Stem triangles (2 down)
  sy=$((s*48/100))
  
  convert -size ${s}x${s} xc:"$BG" \
    `# Top bar - 5 triangles (Blue, Blue, Red, Blue, Blue)` \
    -fill "$BLUE" -stroke "$BLUE" -strokewidth $STROKE_WIDTH \
    -draw "polygon $((cx-2*tw)),$ty $((cx-tw)),$ty $((cx-3*tw/2)),$((ty+th))" \
    -fill "$BLUE" -stroke "$BLUE" -strokewidth $STROKE_WIDTH \
    -draw "polygon $((cx-tw)),$ty $((cx)),$ty $((cx-tw/2)),$((ty+th))" \
    -fill "$RED" -stroke "$RED" -strokewidth $STROKE_WIDTH \
    -draw "polygon $((cx)),$ty $((cx+tw)),$ty $((cx+tw/2)),$((ty+th))" \
    -fill "$BLUE" -stroke "$BLUE" -strokewidth $STROKE_WIDTH \
    -draw "polygon $((cx+tw)),$ty $((cx+2*tw)),$ty $((cx+3*tw/2)),$((ty+th))" \
    -fill "$BLUE" -stroke "$BLUE" -strokewidth $STROKE_WIDTH \
    -draw "polygon $((cx+2*tw)),$ty $((cx+3*tw)),$ty $((cx+5*tw/2)),$((ty+th))" \
    `# Stem - 2 triangles below center` \
    -fill "$RED" -stroke "$RED" -strokewidth $STROKE_WIDTH \
    -draw "polygon $((cx+tw/4)),$((sy+th/2)) $((cx+5*tw/4)),$((sy+th/2)) $((cx+3*tw/4)),$((sy+2*th))" \
    -fill "$BLUE" -stroke "$BLUE" -strokewidth $STROKE_WIDTH \
    -draw "polygon $((cx+tw/4)),$((sy+2*th)) $((cx+5*tw/4)),$((sy+2*th)) $((cx+3*tw/4)),$((sy+3*th))" \
    icons/icon-${size}.png 2>/dev/null
  
  if [ $? -eq 0 ]; then
    echo "Created icons/icon-${size}.png"
  else
    echo "Failed: icons/icon-${size}.png"
  fi
done

# Apple touch icon (180x180)
s=180
tw=$((s*20/100))
th=$((s*17/100))
cx=$((s*50/100))
ty=$((s*28/100))
sy=$((s*46/100))

convert -size ${s}x${s} xc:"$BG" \
  -fill "$BLUE" -stroke "$BLUE" -strokewidth 1 \
  -draw "polygon $((cx-2*tw)),$ty $((cx-tw)),$ty $((cx-3*tw/2)),$((ty+th))" \
  -fill "$BLUE" -stroke "$BLUE" -strokewidth 1 \
  -draw "polygon $((cx-tw)),$ty $((cx)),$ty $((cx-tw/2)),$((ty+th))" \
  -fill "$RED" -stroke "$RED" -strokewidth 1 \
  -draw "polygon $((cx)),$ty $((cx+tw)),$ty $((cx+tw/2)),$((ty+th))" \
  -fill "$BLUE" -stroke "$BLUE" -strokewidth 1 \
  -draw "polygon $((cx+tw)),$ty $((cx+2*tw)),$ty $((cx+3*tw/2)),$((ty+th))" \
  -fill "$BLUE" -stroke "$BLUE" -strokewidth 1 \
  -draw "polygon $((cx+2*tw)),$ty $((cx+3*tw)),$ty $((cx+5*tw/2)),$((ty+th))" \
  -fill "$RED" -stroke "$RED" -strokewidth 1 \
  -draw "polygon $((cx+tw/4)),$((sy+th/2)) $((cx+5*tw/4)),$((sy+th/2)) $((cx+3*tw/4)),$((sy+2*th))" \
  -fill "$BLUE" -stroke "$BLUE" -strokewidth 1 \
  -draw "polygon $((cx+tw/4)),$((sy+2*th)) $((cx+5*tw/4)),$((sy+2*th)) $((cx+3*tw/4)),$((sy+3*th))" \
  icons/apple-touch-icon.png 2>/dev/null

if [ $? -eq 0 ]; then
  echo "Created icons/apple-touch-icon.png"
fi

echo "Done! Generated 7-triangle T icons."
