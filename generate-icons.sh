#!/bin/bash
# Generate PWA icons with a T composed of game triangles
# Uses ImageMagick to draw triangles matching the game aesthetic

mkdir -p icons

# Colors matching game (TRON BLU and ARES RED)
BLUE="#00F0FF"
RED="#FF2A54"
BG="#03060C"
BORDER="#00F0FF"

for size in 72 96 128 144 152 192 384 512; do
  # Scale factors
  s=$((size))
  
  # Create background
  convert -size ${s}x${s} xc:"$BG" \
    -fill "$BLUE" \
    -stroke "$BLUE" -strokewidth 1 \
    -draw "polygon $((s*15/100)),$((s*30/100)) $((s*50/100)),$((s*30/100)) $((s*32/100)),$((s*50/100))" \
    -draw "polygon $((s*50/100)),$((s*30/100)) $((s*85/100)),$((s*30/100)) $((s*68/100)),$((s*50/100))" \
    -fill "$RED" \
    -stroke "$RED" -strokewidth 1 \
    -draw "polygon $((s*32/100)),$((s*50/100)) $((s*68/100)),$((s*50/100)) $((s*50/100)),$((s*70/100))" \
    -fill "$BLUE" \
    -stroke "$BLUE" -strokewidth 1 \
    -draw "polygon $((s*38/100)),$((s*70/100)) $((s*62/100)),$((s*70/100)) $((s*50/100)),$((s*90/100))" \
    icons/icon-${size}.png 2>/dev/null
  
  if [ $? -eq 0 ]; then
    echo "Created icons/icon-${size}.png"
  else
    echo "Skipped icons/icon-${size}.png"
  fi
done

# Apple touch icon with cleaner design
convert -size 180x180 xc:"$BG" \
  -fill "$BLUE" -stroke "$BLUE" -strokewidth 2 \
  -draw "polygon 27,54 90,54 58,90" \
  -draw "polygon 90,54 153,54 122,90" \
  -fill "$RED" -stroke "$RED" -strokewidth 2 \
  -draw "polygon 58,90 122,90 90,126" \
  -fill "$BLUE" -stroke "$BLUE" -strokewidth 2 \
  -draw "polygon 68,126 112,126 90,162" \
  icons/apple-touch-icon.png 2>/dev/null

if [ $? -eq 0 ]; then
  echo "Created icons/apple-touch-icon.png"
fi

echo "Icon generation complete!"
