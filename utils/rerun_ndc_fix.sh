#!/usr/bin/env bash
# Re-corre la triangulación sin malla y la evaluación con la conversión NDC
# corregida (pipeline_common/camera.py, 2026-10-05) para los prompts indicados,
# SIN sobrescribir los resultados anteriores: cada prompt se procesa con un
# identificador nuevo <PROMPT>_ndcfix. Molmo2 no se vuelve a correr: se crea un
# enlace molmo_multiview_<PROMPT>_ndcfix.json -> molmo_multiview_<PROMPT>.json
# en cada carpeta de render.
#
# Uso (en el servidor, desde la raíz del repo, con el código ya actualizado):
#   bash utils/rerun_ndc_fix.sh axis_sym  axis_v08
#   bash utils/rerun_ndc_fix.sh plane_sym plane_v04_1
# Variables opcionales: DATA=~/data  SIZE=224  LIGHT=flat  SUFIJO=_ndcfix
set -euo pipefail

TIPO="${1:?tipo: axis_sym o plane_sym}"
PROMPT="${2:?prompt, p. ej. axis_v08}"
DATA="${DATA:-$HOME/data}"
SIZE="${SIZE:-224}"
LIGHT="${LIGHT:-flat}"
SUFIJO="${SUFIJO:-_ndcfix}"
NUEVO="${PROMPT}${SUFIJO}"
RENDERS="$DATA/renders"
OBJECTS="$DATA/objects"

# 0) comprobar que el código tiene la corrección
python3 -c "
import sys; sys.path.insert(0, '.')
from pipeline_common.camera import molmo_to_ndc
assert molmo_to_ndc(0, 0)[0] == 1.0, 'camera.py NO tiene la corrección NDC'
print('[ok] camera.py con la convención corregida')"

# 1) enlaces al JSON de Molmo2 con el identificador nuevo
n=0; faltan=0
for d in "$RENDERS/$TIPO"/*/"$SIZE"/"$LIGHT"; do
  src="$d/molmo_multiview_${PROMPT}.json"
  if [[ -f "$src" ]]; then
    ln -sf "molmo_multiview_${PROMPT}.json" "$d/molmo_multiview_${NUEVO}.json"; n=$((n+1))
  else
    faltan=$((faltan+1))
  fi
done
echo "[ok] $n enlaces creados ($faltan carpetas sin molmo_multiview_${PROMPT}.json)"
[[ $n -gt 0 ]] || { echo "[error] no se encontró ningún molmo_multiview_${PROMPT}.json"; exit 1; }

# 2) triangulación sin malla y 3) evaluación
if [[ "$TIPO" == "axis_sym" ]]; then
  python3 Mapping/estimate_symmetry_no_mesh.py --renders-root "$RENDERS" \
      --symmetry-type axis_sym --sizes "$SIZE" --lightings "$LIGHT" \
      --experiment-id "$NUEVO" --overwrite
  python3 Mapping/evaluate.py --renders-root "$RENDERS" --objects-root "$OBJECTS" \
      --symmetry-type axis_sym --sizes "$SIZE" --lightings "$LIGHT" \
      --experiment-id "$NUEVO" --method triangulation
else
  python3 Mapping/estimate_symmetry_no_mesh.py --renders-root "$RENDERS" \
      --symmetry-type plane_sym --sizes "$SIZE" --lightings "$LIGHT" \
      --experiment-id "$NUEVO" --max-planes 3 --overwrite
  python3 Mapping/evaluate.py --renders-root "$RENDERS" --objects-root "$OBJECTS" \
      --symmetry-type plane_sym --sizes "$SIZE" --lightings "$LIGHT" \
      --experiment-id "$NUEVO" --method triangulation_multiplane --with-reference-metrics
fi
echo "[ok] listo: resultados con identificador $NUEVO (los de $PROMPT no se tocaron)"
