# Literatura — Diagnóstico de casos, conditioning y sesgo de selección

**Experimentos:** A13 (`visualizar_casos_axis.ipynb` + `Experiments/analysis/*_2d.png` /
`*_3d.png`) y A5 (tabla "mejor experimento por objeto" de
`sandbox_pipeline_server_final.ipynb` §12). Ambos dependen de la estratificación
BUENO / MEDIO / MALO de `Mapping/select_sandbox_objects.py`.

**Qué se hace:** inspección visual de los puntos 2D y de los rayos 3D para separar dos
hipótesis. (a) Ruido/condicionamiento: los rayos no convergen. (b) Inconsistencia de identidad:
los rayos convergen, pero a una estructura distinta del eje real. Complementa los diagnósticos
cuantitativos de `docs/diagnostico_conditioning_axis.md`.

## Conditioning de la triangulación

Fuente de las **[repo]**: `docs/diagnostico_conditioning_axis.md` §2.

- **[repo]** Hartley, R., Zisserman, A. (2004). *Multiple View Geometry in Computer Vision*
  (2ª ed.). — El error de triangulación escala aproximadamente como 1/sin(θ) con la separación
  entre observaciones (`pixel_sep`).
- **[repo]** Golub, G. H., Van Loan, C. F. *Matrix Computations* (4ª ed.). Johns Hopkins
  University Press. — El número de condición σ1/σ2 acota la sensibilidad del vector nulo
  (`cond_number`).
- **[repo]** Bartoli & Sturm (2005). CVIU 100(3). — Planos de interpretación y configuraciones
  degeneradas.
- **[repo]** Draper, N. R., Smith, H. *Applied Regression Analysis* (3ª ed.). Wiley. — El
  apalancamiento y la dispersión del regresor determinan la precisión de la pendiente
  (`axis_span`).
- **[repo]** York, D. (1966). *Least-squares fitting of a straight line*. Canadian Journal of
  Physics, 44(5). — Ajuste con error en ambas variables.

## Inconsistencia de identidad entre vistas

- **[repo]** Bhat & Yamasaki (2026). *Consistent Yet Wrong*. arXiv:2606.02742.
- **[repo]** Gong et al. (2025). *ZeroKey*. arXiv:2412.06292.
- **[repo]** CVPC. arXiv:2512.04686. — Correspondencia entre vistas casi al azar en Molmo-7B-D.

## Sesgo de selección: "mejor de N" por objeto (A5)

`docs/experimentos_pipeline_sin_malla.md` §5.6 lo analiza: quedarse con el mínimo de 6
estimadores ruidosos por objeto hace que alguna ablación "gane" aunque ninguna sea mejor.

- **[nuevo]** Cawley, G. C., Talbot, N. L. C. (2010). *On Over-fitting in Model Selection and
  Subsequent Selection Bias in Performance Evaluation*. JMLR, 11, 2079–2107.
- **[nuevo]** Benjamini, Y., Hochberg, Y. (1995). *Controlling the False Discovery Rate: A
  Practical and Powerful Approach to Multiple Testing*. JRSS-B, 57(1), 289–300. — Corrección
  por comparaciones múltiples cuando se comparan muchas variantes.
- **[nuevo]** Demšar, J. (2006). *Statistical Comparisons of Classifiers over Multiple Data
  Sets*. JMLR, 7, 1–30. — Tests pareados (Wilcoxon) por objeto: la forma correcta de comparar
  una ablación contra el baseline sobre los mismos objetos.

## Estratificación BUENO/MEDIO/MALO

Hay que tenerla en cuenta al leer resultados: los objetos se eligieron según el error de
`axis_v06_nomesh`/`plane_v04_1_nomesh`, de modo que las ablaciones evaluadas sobre ese mismo
prompt sufren regresión a la media (los MALO tienden a "mejorar" y los BUENO a "empeorar" por
azar).

- **[nuevo]** Barnett, A. G., van der Pols, J. C., Dobson, A. J. (2005). *Regression to the
  mean: what it is and how to deal with it*. International Journal of Epidemiology, 34(1),
  215–220.
