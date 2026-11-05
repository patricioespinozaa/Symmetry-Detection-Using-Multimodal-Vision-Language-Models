#!/usr/bin/env bash
# Recalcula los resultados finales con los mismos puntos ya inferidos: EXP-E en eje y S_dat en
# plano (un plano por objeto). No ejecuta ningún modelo ni sobrescribe las líneas base: cada
# configuración recibe un alias y un id propios, con resolución e iluminación en el nombre.
#
#   bash PlaneScoreData/run_resultados_finales.sh
#   LISTAR=1 bash PlaneScoreData/run_resultados_finales.sh    # lista las configuraciones
#   DRY_RUN=1 bash PlaneScoreData/run_resultados_finales.sh   # muestra los comandos
#   SOLO=ndcfix_448 bash PlaneScoreData/run_resultados_finales.sh   # solo esa carpeta
set -uo pipefail

R="${R:-/data/pespinoz/renders}"
O="${O:-/data/pespinoz/objects}"
R_MODELOS="${R_MODELOS:-/home/pespinoz/data/renders}"
O_MODELOS="${O_MODELOS:-/home/pespinoz/data/objects}"
OUT="${OUT:-resultados_finales}"
JOBS="${JOBS:-12}"
PY="${PY:-python3}"
DRY_RUN="${DRY_RUN:-0}"
LISTAR="${LISTAR:-0}"
SOLO="${SOLO:-}"

# tipo | carpeta | resolución | iluminación | id de la línea base
CONFIGS="
axis_sym|ndcfix_flujoA|224|flat|axis_v00_ndcfix_nomesh
axis_sym|ndcfix_flujoA|224|flat|axis_v00_1_ndcfix_nomesh
axis_sym|ndcfix_flujoA|224|flat|axis_v01_ndcfix_nomesh
axis_sym|ndcfix_flujoA|224|flat|axis_v01_1_ndcfix_nomesh
axis_sym|ndcfix_flujoA|224|flat|axis_v02_ndcfix_nomesh
axis_sym|ndcfix_flujoA|224|flat|axis_v02_1_ndcfix_nomesh
axis_sym|ndcfix_flujoA|224|flat|axis_v03_ndcfix_nomesh
axis_sym|ndcfix_flujoA|224|flat|axis_v03_1_ndcfix_nomesh
axis_sym|ndcfix_flujoA|224|flat|axis_v04_ndcfix_nomesh
axis_sym|ndcfix_flujoA|224|flat|axis_v04_1_ndcfix_nomesh
axis_sym|ndcfix_flujoA|224|flat|axis_v05_ndcfix_nomesh
axis_sym|ndcfix_flujoA|224|flat|axis_v05_1_ndcfix_nomesh
axis_sym|ndcfix_flujoA|224|flat|axis_v06_ndcfix_nomesh
axis_sym|ndcfix_flujoA|224|flat|axis_v07_ndcfix_nomesh
axis_sym|ndcfix_v08|224|flat|axis_v08_ndcfix_nomesh
axis_sym|ndcfix_448|448|flat|axis_v04_ndcfix_nomesh
axis_sym|ndcfix_448|448|flat|axis_v06_ndcfix_nomesh
axis_sym|ndcfix_448|448|flat|axis_v08_ndcfix_nomesh
axis_sym|ndcfix_1136|1136|flat|axis_v04_ndcfix_nomesh
axis_sym|ndcfix_224_darker|224|darker|axis_v04_ndcfix_nomesh
axis_sym|ndcfix_224_brighter|224|brighter|axis_v04_ndcfix_nomesh
axis_sym|ndcfix_single_oct26|224|flat|axis_v04_single_oct26_ndcfix_nomesh
axis_sym|ndcfix_flujoB|224|flat|axis_v04_flowB_ndcfix_nomesh
axis_sym|ndcfix_flujoB|224|flat|axis_v05_1_flowB_ndcfix_nomesh
axis_sym|ndcfix_flujoC|224|flat|axis_v05_1_flowC_ndcfix_nomesh
axis_sym|molmopoint|224|flat|molmopoint
axis_sym|qwen3vl|224|flat|qwen3vl
plane_sym|ndcfix_flujoA|224|flat|plane_v00_ndcfix_mp1_nomesh
plane_sym|ndcfix_flujoA|224|flat|plane_v00_1_ndcfix_mp1_nomesh
plane_sym|ndcfix_flujoA|224|flat|plane_v01_ndcfix_mp1_nomesh
plane_sym|ndcfix_flujoA|224|flat|plane_v01_1_ndcfix_mp1_nomesh
plane_sym|ndcfix_flujoA|224|flat|plane_v02_ndcfix_mp1_nomesh
plane_sym|ndcfix_flujoA|224|flat|plane_v02_1_ndcfix_mp1_nomesh
plane_sym|ndcfix_flujoA|224|flat|plane_v03_ndcfix_mp1_nomesh
plane_sym|ndcfix_flujoA|224|flat|plane_v03_1_ndcfix_mp1_nomesh
plane_sym|ndcfix_flujoA|224|flat|plane_v04_ndcfix_mp1_nomesh
plane_sym|ndcfix_flujoA|224|flat|plane_v04_1_ndcfix_mp1_nomesh
plane_sym|ndcfix_flujoA|224|flat|plane_v05_ndcfix_mp1_nomesh
plane_sym|ndcfix_flujoA|224|flat|plane_v05_1_ndcfix_mp1_nomesh
plane_sym|ndcfix_flujoA|224|flat|plane_v06_ndcfix_mp1_nomesh
plane_sym|ndcfix_flujoA|224|flat|plane_v07_ndcfix_mp1_nomesh
plane_sym|ndcfix_v08|224|flat|plane_v08_ndcfix_mp1_nomesh
plane_sym|ndcfix_448|448|flat|plane_v02_ndcfix_mp1_nomesh
plane_sym|ndcfix_448|448|flat|plane_v05_1_ndcfix_mp1_nomesh
plane_sym|ndcfix_448|448|flat|plane_v06_ndcfix_mp1_nomesh
plane_sym|ndcfix_1136|1136|flat|plane_v05_1_ndcfix_mp1_nomesh
plane_sym|ndcfix_224_darker|224|darker|plane_v05_1_ndcfix_mp1_nomesh
plane_sym|ndcfix_224_brighter|224|brighter|plane_v05_1_ndcfix_mp1_nomesh
plane_sym|ndcfix_single_oct26|224|flat|plane_v05_1_single_oct26_ndcfix_mp1_nomesh
plane_sym|ndcfix_flujoB|224|flat|plane_v04_1_flowB_ndcfix_mp1_nomesh
plane_sym|ndcfix_flujoB|224|flat|plane_v05_1_flowB_ndcfix_mp1_nomesh
plane_sym|ndcfix_flujoC|224|flat|plane_v04_1_flowC_ndcfix_mp1_nomesh
plane_sym|molmopoint|224|flat|molmopoint
plane_sym|qwen3vl|224|flat|qwen3vl
"

ERRORES="$OUT/logs/errores.txt"

registrar_error() {
    echo "ERROR $1" | tee -a "$ERRORES"
}

correr() {
    # uso: correr <log> <comando...>
    local log="$1"
    shift
    if [ "$DRY_RUN" = "1" ]; then
        echo "  [dry] $*"
        return 0
    fi
    "$@" 2>&1 | tee "$log"
}

prefijo() {
    # uso: prefijo <tipo> <id base>. Los alias deben empezar por axis_v o plane_v.
    case "$2" in
        molmopoint|qwen3vl)
            if [ "$1" = "axis_sym" ]; then echo "axis_v04_$2"; else echo "plane_v05_1_$2"; fi ;;
        *)  local p="${2%_ndcfix_mp1_nomesh}"; echo "${p%_ndcfix_nomesh}" ;;
    esac
}

id_alias() {
    # uso: id_alias <tipo> <id base> <resolución> <iluminación>
    local p
    p="$(prefijo "$1" "$2")_s${3}_${4}_ndcfix"
    if [ "$1" = "axis_sym" ]; then echo "${p}_nomesh"; else echo "${p}_mp1_nomesh"; fi
}

id_final() {
    # uso: id_final <tipo> <id base> <resolución> <iluminación>
    local p
    p="$(prefijo "$1" "$2")_s${3}_${4}_ndcfix"
    if [ "$1" = "axis_sym" ]; then echo "${p}_expE_nomesh"; else echo "${p}_sdat_mp1_nomesh"; fi
}

archivo_entrada() {
    # uso: archivo_entrada <id base>
    case "$1" in
        molmopoint) echo "molmopoint_multiview.json" ;;
        qwen3vl)    echo "qwen3vl_multiview.json" ;;
        *)          echo "molmo_multiview_$1.json" ;;
    esac
}

raiz_renders() {
    case "$1" in molmopoint|qwen3vl) echo "$R_MODELOS" ;; *) echo "$R" ;; esac
}

raiz_objetos() {
    case "$1" in molmopoint|qwen3vl) echo "$O_MODELOS" ;; *) echo "$O" ;; esac
}

crear_alias() {
    # Enlaza molmo_multiview_<alias>.json con los puntos originales; devuelve cuántos encontró.
    # uso: crear_alias <raíz> <tipo> <resolución> <iluminación> <archivo> <alias>
    local n=0 d
    for d in "$1/$2"/*/"$3"/"$4"; do
        if [ -e "$d/$5" ]; then
            ln -sfn "$5" "$d/molmo_multiview_$6.json"
            n=$((n + 1))
        fi
    done
    echo "$n"
}

renombrar_csv() {
    # compare_results_no_mesh.py nombra el CSV con los ids; se acorta para Windows.
    # uso: renombrar_csv <carpeta> <tipo>
    [ "$DRY_RUN" = "1" ] && return 0
    local origen destino
    origen=$(ls -t "$1"/experiments_*/"$2"_*comparison.csv 2>/dev/null | head -1)
    [ -n "$origen" ] || return 1
    destino="$(dirname "$origen")/$2_final_comparison.csv"
    [ "$origen" = "$destino" ] || mv "$origen" "$destino"
}

csv_base() {
    # uso: csv_base <tipo> <carpeta>
    case "$2" in
        molmopoint) ls MolmoPointPointing/experiments_*/"$1"_molmopoint_comparison.csv ;;
        qwen3vl)    ls QwenPointing/experiments_*/"$1"_qwen3vl_comparison.csv ;;
        *)          ls ndc_fix/"$2"/experiments_*/"$1"_*comparison.csv ;;
    esac 2>/dev/null
}

filas_activas=$(echo "$CONFIGS" | grep '|')
if [ -n "$SOLO" ]; then
    filas_activas=$(echo "$filas_activas" | awk -F'|' -v c="$SOLO" '$2 == c')
fi

if [ "$LISTAR" = "1" ]; then
    for fila in $filas_activas; do
        IFS='|' read -r tipo carpeta size luz base <<< "$fila"
        printf '%-9s %-20s %5s %-8s %s -> %s\n' "$tipo" "$carpeta" "$size" "$luz" "$base" \
            "$(id_final "$tipo" "$base" "$size" "$luz")"
    done
    echo "Total: $(echo "$filas_activas" | wc -l)"
    exit 0
fi

if [ "${AUDITAR:-0}" = "1" ]; then
    # Solo lee archivos. Opcional: DESDE="2026-11-05 00:00" y MUESTRA=30.
    fallas=0
    for fila in $filas_activas; do
        IFS='|' read -r tipo _ size luz base <<< "$fila"
        if [ "$tipo" = "axis_sym" ]; then estimador=expE; else estimador=sdat; fi
        "$PY" PlaneScoreData/audit_run.py --renders-root "$(raiz_renders "$base")" \
            --symmetry-type "$tipo" --input-id "$(id_alias "$tipo" "$base" "$size" "$luz")" \
            --expected-input "$(archivo_entrada "$base")" \
            --output-id "$(id_final "$tipo" "$base" "$size" "$luz")" --estimador "$estimador" \
            --size "$size" --lighting "$luz" --muestra "${MUESTRA:-30}" \
            ${DESDE:+--desde "$DESDE"} || fallas=$((fallas + 1))
    done
    echo "Corridas con problemas: $fallas de $(echo "$filas_activas" | wc -l)"
    [ "$fallas" -eq 0 ]
    exit $?
fi

mkdir -p "$OUT/logs"
rm -f "$ERRORES"
echo "Renders Molmo2: $(readlink -f "$R" 2>/dev/null || echo "$R")"
echo "Renders modelos: $(readlink -f "$R_MODELOS" 2>/dev/null || echo "$R_MODELOS")"

total=$(echo "$filas_activas" | wc -l)
n=0
for fila in $filas_activas; do
    IFS='|' read -r tipo carpeta size luz base <<< "$fila"
    alias_id=$(id_alias "$tipo" "$base" "$size" "$luz")
    final=$(id_final "$tipo" "$base" "$size" "$luz")
    raiz=$(raiz_renders "$base")
    objetos=$(raiz_objetos "$base")
    logs="$OUT/$carpeta/logs"
    n=$((n + 1))
    mkdir -p "$logs"
    echo "=== [$n/$total] $(date '+%H:%M:%S') $tipo $carpeta ${size}px $luz: $base -> $final"

    if [ "$DRY_RUN" != "1" ]; then
        if [ ! -d "$objetos" ]; then
            registrar_error "no existe la carpeta de objetos $objetos ($final)"
            continue
        fi
        entradas=$(crear_alias "$raiz" "$tipo" "$size" "$luz" "$(archivo_entrada "$base")" "$alias_id")
        echo "  entradas enlazadas: $entradas"
        if [ "$entradas" -eq 0 ]; then
            registrar_error "sin entradas: $tipo $carpeta $size $luz $base"
            continue
        fi
    fi

    if [ "$tipo" = "axis_sym" ]; then
        correr "$logs/est_$final.log" \
            "$PY" Pipeline_Experiments/estimate_symmetry_variants.py --renders-root "$raiz" \
            --symmetry-type axis_sym --sizes "$size" --lightings "$luz" \
            --experiment-id "$alias_id" --variant expE --overwrite \
            || { registrar_error "estimación: $final"; continue; }
    else
        correr "$logs/est_$final.log" \
            "$PY" PlaneScoreData/plane_score_data.py --renders-root "$raiz" \
            --input-id "$alias_id" --output-id "$final" --criterio datos \
            --size "$size" --lighting "$luz" --jobs "$JOBS" --overwrite \
            || { registrar_error "estimación: $final"; continue; }
    fi

    correr "$logs/eval_$final.log" \
        "$PY" Mapping/evaluate.py --renders-root "$raiz" --objects-root "$objetos" \
        --symmetry-type "$tipo" --sizes "$size" --lightings "$luz" \
        --experiment-id "$final" --method triangulation --with-reference-metrics \
        || registrar_error "evaluación: $final"
done

echo "=== Comparaciones"
for grupo in $(echo "$filas_activas" | cut -d'|' -f1,2 | sort -u); do
    tipo=${grupo%%|*}
    carpeta=${grupo##*|}
    filas=$(echo "$filas_activas" | awk -F'|' -v t="$tipo" -v c="$carpeta" '$1 == t && $2 == c')
    IFS='|' read -r _ _ size luz base <<< "$(echo "$filas" | head -1)"
    raiz=$(raiz_renders "$base")
    dir="$OUT/$carpeta"
    finales=""
    for fila in $filas; do
        IFS='|' read -r _ _ _ _ base <<< "$fila"
        finales="$finales $(id_final "$tipo" "$base" "$size" "$luz")"
    done
    echo "--- $tipo $carpeta"

    # shellcheck disable=SC2086
    correr "$dir/logs/compare_$tipo.log" \
        "$PY" Mapping/compare_results_no_mesh.py --renders-root "$raiz" \
        --symmetry-type "$tipo" --sizes "$size" --lightings "$luz" \
        --experiment-id $finales --csv-dir "$dir" --no-plots --total-objects 850 \
        && renombrar_csv "$dir" "$tipo" \
        || { registrar_error "comparación: $tipo $carpeta"; continue; }

    if [ "$DRY_RUN" = "1" ]; then
        csv_nuevo="$dir/experiments_FECHA/${tipo}_final_comparison.csv"
    else
        csv_nuevo=$(ls -t "$dir"/experiments_*/"${tipo}"_final_comparison.csv | head -1)
    fi
    mapfile -t csvs_base < <(csv_base "$tipo" "$carpeta")
    if [ "${#csvs_base[@]}" -eq 0 ]; then
        registrar_error "sin CSV de línea base: $tipo $carpeta"
        continue
    fi
    for fila in $filas; do
        IFS='|' read -r _ _ _ _ base <<< "$fila"
        final=$(id_final "$tipo" "$base" "$size" "$luz")
        mkdir -p "$dir/pareado"
        correr "$dir/logs/pareado_$final.log" \
            "$PY" PlaneScoreData/compare_plane_scores.py --renders-root "$raiz" \
            --symmetry-type "$tipo" --comparison-csv "${csvs_base[@]}" "$csv_nuevo" \
            --base-id "$base" --new-id "$final" --size "$size" --lighting "$luz" \
            --salida "$dir/pareado/$final.csv" \
            || registrar_error "comparación pareada: $final"
    done
done

correr "$OUT/logs/triviales.log" \
    "$PY" Pipeline_Experiments/diagnostics/trivial_baselines.py --objects-root "$O" \
    || registrar_error "predictores triviales"

echo "=== Fin $(date '+%H:%M:%S')"
if [ -s "$ERRORES" ]; then
    echo "Terminó CON errores; revisa $ERRORES:"
    cat "$ERRORES"
    exit 1
fi
echo "Terminó sin errores."
