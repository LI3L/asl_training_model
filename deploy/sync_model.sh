#!/usr/bin/env bash
# Copies the freshly converted model.h into the Arduino sketch folder.
# Run this after every `python src/convert_model.py`.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SRC="$SCRIPT_DIR/../output/model.h"
DST="$SCRIPT_DIR/esp32/ASL_Detector/model.h"

if [ ! -f "$SRC" ]; then
    echo "Error: $SRC not found. Run src/convert_model.py first." >&2
    exit 1
fi

cp "$SRC" "$DST"

# xxd -i emits a plain (non-const) byte array, which the ESP32 linker places
# in writable DRAM (~400KB total) instead of leaving it memory-mapped in
# flash -- a 400KB+ model blows that budget instantly. Mark it const and
# 16-byte aligned (required by the flatbuffer parser) so it stays in flash.
sed -i \
    -e 's/^unsigned char \(_[a-zA-Z0-9_]*\)\[\] = {/alignas(16) const unsigned char \1[] = {/' \
    -e 's/^unsigned int \(_[a-zA-Z0-9_]*_len\) = /const unsigned int \1 = /' \
    "$DST"

echo "Copied $SRC -> $DST (patched array to const/aligned for flash placement)"
