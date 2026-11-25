# Plan: reproducir las rotaciones de Aguirre y Sipiran en las mallas planares

Documento de trabajo para retomar esta tarea en otra conversación. Resume lo
que ya se averiguó, qué hay que construir y cómo verificarlo.

## 1. Objetivo

Comparar el pipeline sin malla (Molmo2 `plane_v02`, S_dat, un plano por objeto)
contra Aguirre y Sipiran (WACV 2026) **sobre la misma pose de los objetos**.
Hoy la Tabla `tab:res_aguirre` (cap4, `sec:res-aguirre`) compara poses
distintas:

- Los valores de Aguirre y Sipiran (F1 0.906/0.912, SDE 0.29/0.28 ×10⁻⁴,
  Tabla 1 del paper, p. 73) se midieron, según todo indica, con los planos
  **rotados**.
- La corrida canónica de este trabajo (F1 0.645/0.542/0.614) usa planos
  **alineados con los ejes cartesianos**.
- La corrida `rot_so3` de este trabajo (F1 ≈ 0.01) usa rotaciones uniformes
  **propias**, no las de ellos.

La idea es construir una versión **limpia** (sin ruido) de las mallas
rotadas *exactamente* como las de ellos, correr el pipeline y agregar esa fila
a la tabla.

## 2. Lo que ya se sabe

### El repositorio de referencia no rota objetos

`C:\Users\HP\Desktop\Seminario de Tesis I\EnhancedBackProjection`:

- `implementation/rotation/rotation.py` solo rota **imágenes** 0/90/180/270°
  (parámetro `rot_aug`, el "4R" del paper). Es un aumento sobre los renders y
  no toca el objeto 3D.
- `datasets/` solo trae listas de ids (`curated_plane_sym_obj.txt`, 850 ids,
  los mismos de este trabajo). Las mallas se distribuyen aparte ("in the
  release").
- El paper (Sec. 4) dice: *"arbitrary rotations were applied to remove
  orientation bias"*. La versión del conjunto que usa esta tesis tiene los
  planos **canónicos** (normales ≈ (1,0,0)); los ejes sí vienen con
  direcciones aleatorias.

### El zip descargado

`C:\Users\HP\Desktop\Seminario de Tesis I\data\curated_plane_sym_obj_rot_robustness.zip`
(516 MB comprimido, 1.8 GB descomprimido). Es descarga externa: extraer en una
carpeta nueva y vacía, y no ejecutar nada desde ahí.

| Carpeta | Objetos | Contenido |
|---|---|---|
| `curated_plane_sym_obj_rot_1_noise/` | 850 | malla rotada + ruido gaussiano 1 % de R_max, `.txt` rotado |
| `curated_plane_sym_obj_rot_3_noise/` | 850 | ídem con ruido 3 % |
| `curated_plane_sym_obj_rot_5_noise/` | 850 | ídem con ruido 5 % |
| `curated_plane_sym_obj_rot_20_extracted/` | 850 | malla rotada con 20 % de partes eliminadas |

Son los conjuntos de robustez de su Tabla 2 (N1 %, N3 %, N5 %, PR20 %). **No
hay versión rotada limpia** (la "S" de su evaluación estándar).

Verificado (análisis leyendo el zip sin extraerlo):

- Los 850 ids coinciden exactamente con su lista y con los de esta tesis.
- Las cuatro carpetas comparten la **misma** referencia rotada por objeto, con
  la misma distribución de planos que la tesis: 692 objetos con 1 plano, 152
  con 2 y 6 con 3 (1014 planos).
- Solo 40 de 1014 normales quedan a menos de ~8° de un eje cartesiano. Ángulo
  al eje más cercano: mediana 29°, p90 44° (uniforme en la esfera: 34° y 46°).
- Con las 30 mallas canónicas que hay localmente
  (`Experiments/objects/curated_plane_sym_obj/`), Kabsch contra
  `rot_1_noise` dio rotaciones de 15° a 175° (mediana 111°): arbitrarias y
  grandes, comparables a SO(3) uniforme.
- El **orden y número de vértices se conserva** (30/30 con el mismo número),
  por eso Kabsch funciona.
- Coherencia malla–referencia: aplicando la R estimada a la normal canónica
  propia, 25/30 coinciden con el `.txt` del zip con error < 2°. Los otros 5
  (3°–60°) comparaban solo el **primer** plano del archivo; probablemente son
  objetos con varios planos en distinto orden. Hay que compararlos como
  **conjunto** (ver §3, paso 2).

Conclusión: sus planos rotados son la pose en la que se midieron sus
resultados, y el texto de cap3 (`sec:sm_aguirre`) que dice que la comparación
más equivalente en plano es con una corrida rotada es correcto.

## 3. Qué hay que construir

Un script nuevo, p. ej. `ExploratoryDataAnalysis/rotaciones_aguirre.py`, solo
con `numpy` y la biblioteca estándar, que se ejecuta **en el servidor** (ahí
están las 850 mallas canónicas). Debe producir la misma salida que
`ExploratoryDataAnalysis/rotar_planares.py`, para reutilizar sin cambios
`ExploratoryDataAnalysis/rotar_mallas.py` y los runners de rotación.

Formato que espera `rotar_mallas.py --rot-dir`:

- `<rot-dir>/<id>.txt`: etiqueta rotada, mismo formato que las originales
  (primera línea = número de planos; luego `plane nx ny nz px py pz`).
- `<rot-dir>/rotaciones.csv`: columnas `object_id`, `r00` … `r22` (y las que
  agregue `rotar_planares.py`; revisarlo antes de escribir el CSV).

### Paso 1. Estimar R por objeto (Kabsch)

Para cada id:

1. Leer los vértices `v` de la malla canónica limpia (servidor:
   `~/data/objects/curated_plane_sym_obj/<id>.obj`) y de
   `curated_plane_sym_obj_rot_1_noise/<id>.obj`. Se usa la de 1 % porque
   tiene el menor ruido y no le faltan vértices.
2. Comprobar el mismo número de vértices. Si no, el objeto se marca como
   rechazado.
3. Kabsch: centrar ambos conjuntos, `H = A_c^T B_c`, SVD, `R = V U^T`, y
   corregir el signo si `det R < 0`. Registrar también la traslación
   (debería ser ≈ 0: los objetos están centrados en el origen).
4. Residuo: RMS de `R a_i − b_i` después de alinear. Debe ser del orden del
   ruido (≈ 0.01·R_max, con R_max = max ‖v‖). Si es mucho mayor, se rechaza.

### Paso 2. Referencia: la propia rotada, verificada con la de ellos

- **Referencia que se usa:** los `.txt` originales propios rotados con la
  misma R (n' = R n, p' = R p). Así coincide exactamente con la malla que se
  renderiza.
- **Verificación:** comparar con el `.txt` de `rot_1_noise` **como conjunto**:
  - mismo número de planos;
  - cada plano propio rotado se empareja con el de ellos de menor ángulo,
    invariante al signo (`arccos |n1·n2|`), sin reutilizar planos;
  - error angular máximo del objeto menor a un umbral (p. ej. 2°; fijarlo
    antes de mirar la distribución completa).
- Los objetos que fallan se listan para revisarlos a mano; no se descartan en
  silencio.

### Paso 3. CSV de control

`validacion_rotacion_aguirre.csv`, una fila por objeto: `object_id`,
`n_vertices`, `angulo_R_grados`, `traslacion_norma`, `rms_kabsch`,
`rms_relativo` (÷ R_max), `n_planos_propios`, `n_planos_aguirre`,
`error_max_grados`, `aceptado`, `motivo`. Al final, imprimir un resumen
(aceptados/total, distribución del ángulo de R, peor error).

Este CSV es la evidencia para escribir en la tesis que se reprodujeron las
rotaciones de Aguirre y Sipiran.

### Paso 4. Rotar las mallas limpias

```bash
python3 ExploratoryDataAnalysis/rotar_mallas.py \
    --obj-dir ~/data/objects/curated_plane_sym_obj \
    --rot-dir <salida del paso 1-3> \
    --out-root /data/pespinoz/objects_rot_aguirre
```

`rotar_mallas.py` ya valida que la etiqueta sea R aplicada a la original, que
R^T v' recupere los vértices y que el error de simetría no cambie al rotar.

## 4. Corrida y evaluación

Igual que la corrida `rot_so3`, cambiando las raíces y el nombre:

- Referencia: `PlaneScoreData/run_plane_v02_224_rot_so3.py` (Molmo2 multiimagen)
  y `PlaneScoreData/run_plane_v02_224_rot_so3_vlms.py` (una vista por llamada).
  Copiar como `run_plane_v02_224_rot_aguirre.py` con
  `--objects-root /data/pespinoz/objects_rot_aguirre`,
  `--renders-root /data/pespinoz/renders_rot_aguirre` y el experimento
  `rot_aguirre` (id final
  `plane_v02_rot_aguirre_s224_flat_ndcfix_sdat_mp1_nomesh`).
- Renders: las 36 vistas de 6/14/26 de las 114 (como en `rot_so3`).
- Configuración base: 224 px, *flat*, Flujo A, envío multiimagen, `plane_v02`,
  S_dat con un plano.
- `evaluate.py` con métricas de referencia (SDE_ref, F1_ref voraz y húngaro).
- Pareado contra la corrida canónica (mismos 850 ids).
- Resultados en `resultados_finales/plane_v02_224_rot_aguirre/experiments_<fecha>/`.

Opcional: correr también sobre `rot_1_noise` tal cual (mallas de ellos, con
ruido) para compararse con su columna N1 % de la Tabla 2 (F1 0.906, SDE 0.343
×10⁻⁴) con exactamente las mismas mallas.

## 5. Integración en la tesis

- `utils/generar_tablas_resultados_cap4.py`, función `tabla_aguirre()`:
  agregar el grupo "Este trabajo, rotación de Aguirre y Sipiran" (6/14/26) en
  el bloque planar, con su `\cmidrule`. Después regenerar y correr
  `utils/verificar_tablas_resultados_cap4.py`.
- `Tesis_DCC_MDS/cap4.tex`, `sec:res-aguirre`: actualizar el texto y eliminar el
  `\verificar` sobre la rotación.
- `Tesis_DCC_MDS/cap3.tex`, `sec:sm_aguirre`: mencionar que las rotaciones se
  recuperaron de sus conjuntos de robustez (Kabsch) y se aplicaron a las mallas
  limpias.
- Si se corre la opción sobre `rot_1_noise`, agregar los valores N1 % de su
  Tabla 2 a `resultados_finales/referencias/aguirre2026_publicado.csv`.

## 6. Convenciones del repo a respetar

- Los códigos se ejecutan en el servidor (no hay GPU local). Hay que activar
  `conda activate tesis_env` (no `base`).
- Los commits los hace el usuario salvo que pida lo contrario.
- Escribir archivos con finales LF; usar la herramienta de escritura de
  archivos y no heredocs de Git Bash para código con `\`.
- Normas de estilo Python: `docs/code-norms.md`.
- Valores de umbrales (residuo de Kabsch, error angular) fijados antes de ver
  los 850 resultados.
