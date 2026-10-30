#!/bin/bash
# Generate PWA icons using ImageMagick
# Run: ./generate-icons.sh

# Create icons directory
mkdir -p icons

# Generate icons at various sizes
for size in 72 96 128 144 152 192 384 512; do
  convert -size ${size}x${size} xc:'#03060C' \
    -fill '#00F0FF' \
    -draw "polygon $((size*15/100)),$((size*85/100)) $((size*50/100)),$((size*15/100)) $((size*85/100)),$((size*85/100))" \
    -stroke '#00F0FF' -strokewidth $((size*3/100)) \
    -fill none \
    -draw "polygon $((size*25/100)),$((size*75/100)) $((size*50/100)),$((size*25/100)) $((size*75/100)),$((size*75/100))" \
    icons/icon-${size}.png 2>/dev/null
  
  if [ $? -eq 0 ]; then
    echo "Created icons/icon-${size}.png"
  else
    echo "Skipped icons/icon-${size}.png (ImageMagick not available)"
    break
  fi
done

# Create apple-touch-icon (180x180)
if command -v convert &> /dev/null; then
  convert -size 180x180 xc:'#03060C' \
    -fill '#00F0FF' \
    -draw "polygon 27,153 90,27 153,153" \
    -stroke '#00F0FF' -strokewidth 5 \
    -fill none \
    -draw "polygon 45,135 90,45 135,135" \
    icons/apple-touch-icon.png 2>/dev/null
  echo "Created icons/apple-touch-icon.png"
fi

echo "Icon generation complete!"
