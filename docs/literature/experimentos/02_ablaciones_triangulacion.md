# Literatura — Ablaciones de triangulación (EXP-A a EXP-E)

**Experimentos:** A4, A5, A7 (eje), P6 (plano, solo A/C/D). Notebooks:
`sandbox_pipeline_server_extended.ipynb` §10 (definición), `sandbox_pipeline_server_final.ipynb`
y `_updated_6_pts_v2.ipynb` §10–12 (ejecución). EXP-F se trata en
[03_consolidacion_multiplano.md](03_consolidacion_multiplano.md).

| ID | Mecanismo | Referencia principal |
|---|---|---|
| EXP-A | SVD ponderado por la separación 2D de los 2 puntos de la vista | Bartoli & Sturm 2005 |
| EXP-B | Point-then-direction: ancla y dirección por separado (solo eje) | Wu et al. 2021 |
| EXP-C | Filtro objeto + RANSAC 2D por vista | Recker et al. 2013 |
| EXP-D | Peso = longitud × penalización por cercanía al borde | AssemblyHands-X |
| EXP-E | Reponderación iterativa por residuo `|n_i·d|` (solo eje) | Hess-Flores et al.; Zhang et al. 2020 |

## Referencias citadas en los notebooks y en el repo

Fuentes: encabezados de las celdas EXP-A..E de los notebooks;
`docs/experimentos_pipeline_sin_malla.md` §4.1 y §8.3; `Pipeline_Experiments/README.md`.

- **[repo]** Bartoli, A., Sturm, P. (2005). *Structure-from-motion using lines*. CVIU, 100(3),
  416–441. DOI:10.1016/j.cviu.2005.06.001. — **EXP-A**.
- **[repo]** Wu et al. (2021). Triangulación de pose de mano, PMC7832884
  (https://www.ncbi.nlm.nih.gov/pmc/articles/PMC7832884/). — **EXP-B**. El repo aclara que es
  una instanciación propia, no un puerto de código.
- **[repo]** Recker, S., Hess-Flores, M., Joy, K. I. (2013). *Statistical Angular Error-Based
  Triangulation for Efficient and Accurate Multi-View Scene Reconstruction*. WACV 2013. —
  **EXP-C**.
- **[repo]** Zhang et al. (2020). arXiv:2008.01258. — Citado junto a EXP-C y EXP-E.
- **[repo]** AssemblyHands-X. arXiv:2509.23888. — **EXP-D**.
- **[repo]** COMPOSE. arXiv:2601.09698. — Citado junto a EXP-D.
- **[repo]** Hess-Flores, M. et al. (eScholarship). — **EXP-E**. *(El notebook no da título
  exacto; completar antes de citar.)*
- **[repo]** Iskakov, K., Burkov, E., Lempitsky, V., Malkov, Y. (2019). *Learnable
  Triangulation of Human Pose*. ICCV 2019. — Precedente de pesos de confianza por vista
  (EXP-A/D/E).
- **[repo]** Lee, S. H., Civera, J. (2019). *Closed-Form Optimal Two-View Triangulation Based
  on Angular Errors*. ICCV 2019, arXiv:1903.09115.
- **[repo]** Ghasemi, A. et al. *Improving Triangulation by Enforcing Consistency*.
  arXiv:1804.10448. — Cota de error que baja con el número de cámaras (si las vistas son
  válidas).

## Fundamentos agregados

- **[nuevo]** Fischler, M. A., Bolles, R. C. (1981). *Random Sample Consensus: A Paradigm for
  Model Fitting with Applications to Image Analysis and Automated Cartography*. Comm. ACM,
  24(6), 381–395. — RANSAC original (EXP-C). Con 2 puntos por vista RANSAC no tiene outliers
  que descartar: el repo confirmó que EXP-C puro es un no-op.
- **[nuevo]** Huber, P. J. (1964). *Robust Estimation of a Location Parameter*. Annals of
  Mathematical Statistics, 35(1), 73–101. — Base de los M-estimadores.
- **[nuevo]** Holland, P. W., Welsch, R. E. (1977). *Robust Regression Using Iteratively
  Reweighted Least-Squares*. Communications in Statistics – Theory and Methods, 6(9). — IRLS,
  el esquema que EXP-E implementa con peso `exp(−λ·residuo)`.

## Lectura de los resultados a la luz de la literatura

Ponderar (A, D, E) ayuda cuando el ruido es **heterocedástico y aleatorio** entre vistas. En el
sandbox el baseline ganó casi todas las columnas. Esto es coherente con el diagnóstico del repo
(`docs/diagnostico_conditioning_axis.md`): el error del eje parece venir de una **inconsistencia
de identidad** entre vistas, un sesgo sistemático que ningún esquema de pesos corrige. Ver
[08_diagnostico_casos_conditioning.md](08_diagnostico_casos_conditioning.md) para el sesgo de
selección en la tabla por objeto (A5).
