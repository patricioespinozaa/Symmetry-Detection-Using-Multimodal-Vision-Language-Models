# Literatura — EXP-LIT-1/2/3 (candidatos, grilla, CoT estructurado)

**Experimentos:** A9–A12 (solo eje, `axis_v06`, 224 px, 30 objetos). Notebook:
`sandbox_literatura.ipynb`. Las tres son **simulaciones con GT** que dan techos (oráculo) y pisos
(azar). No hacen inferencias nuevas de Molmo2.

| ID | Idea | Techo ΔAUC@45 vs. baseline | Veredicto del notebook |
|---|---|---|---|
| LIT-1 | Reformular el pointing como elegir entre K=4 candidatos epipolares | +0.025 / +0.028 / −0.012 | Marginal |
| LIT-2 | Grilla 5×5 visible; anclar primero a una celda | +0.118 / +0.145 / +0.147 | Vale la pena |
| LIT-3 | CoT estructurado (categoría, orientación, polos) antes de señalar | +0.021 / +0.037 / +0.015 | Bajo; el prior modal empeora |

## Referencias citadas en el notebook

- **[repo]** *Towards Cross-View Point Correspondence in VLMs* (CVPC). arXiv:2512.04686. —
  Molmo-7B-D acierta 3.02% al predecir la coordenada continua correspondiente en otra vista,
  frente a 60–66% al elegir entre 3 candidatos. Motivación de **LIT-1**.
- **[repo]** ZeroDex. arXiv:2606.19340. — Estrategia de selección/votación multivista.
  **LIT-1**.
- **[repo]** MOKA. arXiv:2403.03174. — Grilla de coordenadas visible (notación tipo ajedrez)
  para pointing de VLM en manipulación robótica. **LIT-2**, implementado en `axis_lit2_grid`.
- **[repo]** SketchVLM. arXiv:2604.22875. — **LIT-2**.
- **[repo]** *Enhancing Spatial Reasoning in VLMs via CoT and RL*. arXiv:2507.13362. —
  Distingue el CoT libre (no ayuda) del CoT estructurado tipo scene-graph (sí ayuda). **LIT-3**,
  implementado en `axis_lit3_cot`.
- **[repo]** Implementación real de LIT-1 con inferencia:
  `Pipeline_Experiments/candidates_then_select/`, todavía sin correr
  (`docs/experimentos_pipeline_sin_malla.md` §4.3).

## Fundamentos agregados

- **[nuevo]** Yang, J. et al. (2023). *Set-of-Mark Prompting*. arXiv:2310.11441. — Antecedente
  general de marcar candidatos numerados sobre la imagen y pedir al modelo que elija (LIT-1) o
  que razone sobre regiones marcadas (LIT-2).
- **[nuevo]** Wei, J. et al. (2022). *Chain-of-Thought Prompting*. NeurIPS 2022. — Base de
  LIT-3.
- **[nuevo]** Hartley & Zisserman (2004), cap. 9 (geometría epipolar). — Generación de
  candidatos a lo largo de la línea epipolar en LIT-1.

## Advertencia metodológica

Los techos usan GT (proyección del polo real, celda real, orientación real), así que miden
**cuánto podría ganarse si el VLM acertara esa subtarea**, no cuánto gana. El paso siguiente
natural es medir la tasa real de acierto. Para LIT-2 ya se midió: hoy el modelo acierta la celda
correcta solo el 18–24% de las veces.
