#!/usr/bin/env bash
# Copia N vistas de un objeto por cada combinación resolución × iluminación y las empaqueta.
# Uso: bash ~/muestras_render.sh [object_id] [n_imagenes] [carpeta_tipo]
set -euo pipefail

OBJ="${1:-10433e5bd8fa2a337b00c7b93209c459}"
N="${2:-5}"
SYM_DIR="${3:-$HOME/data/renders/axis_sym}"

BASE="$SYM_DIR/$OBJ"
[ -d "$BASE" ] || { echo "No existe $BASE"; exit 1; }

OUT="$HOME/muestras_${OBJ:0:8}"
rm -rf "$OUT" && mkdir -p "$OUT"

# Elige las vistas una sola vez (repartidas uniformemente) y usa las mismas en todas las combinaciones
REF_DIR=$(find "$BASE" -mindepth 2 -maxdepth 2 -type d | sort -V | head -n 1)
mapfile -t TODAS < <(ls "$REF_DIR" | grep -E '^IND_.*\.png$' | sort -V)
TOTAL=${#TODAS[@]}
[ "$TOTAL" -gt 0 ] || { echo "Sin imágenes en $REF_DIR"; exit 1; }
VISTAS=()
for ((k = 0; k < N; k++)); do
  VISTAS+=("${TODAS[$(( k * TOTAL / N ))]}")
done
echo "Vistas elegidas (de $TOTAL): ${VISTAS[*]}"

for SIZE_DIR in $(find "$BASE" -mindepth 1 -maxdepth 1 -type d | sort -V); do
  SIZE=$(basename "$SIZE_DIR")
  for LIGHT in flat darker brighter; do
    DIR="$SIZE_DIR/$LIGHT"
    if [ ! -d "$DIR" ]; then
      echo "  [falta] ${SIZE}px / $LIGHT"
      continue
    fi
    DEST="$OUT/${SIZE}px_${LIGHT}"
    mkdir -p "$DEST"
    n=0
    for V in "${VISTAS[@]}"; do
      if [ -f "$DIR/$V" ]; then cp "$DIR/$V" "$DEST/"; n=$((n + 1)); fi
    done
    echo "  ${SIZE}px / $LIGHT: $n imágenes"
  done
done

tar -czf "$OUT.tar.gz" -C "$HOME" "$(basename "$OUT")"
echo "Listo: $OUT.tar.gz"
