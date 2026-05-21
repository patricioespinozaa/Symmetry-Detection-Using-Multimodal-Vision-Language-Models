import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Datos
n_views = [1, 6, 14, 26]
ang_mean = [57.57, 48.00, 44.70, 44.30]
ang_med = [72.38, 48.27, 41.24, 37.15]

fig, ax = plt.subplots(figsize=(9, 7))

# --- DIBUJO DE LÍNEAS ---
ax.plot(n_views, ang_mean, marker='o', markersize=11, linewidth=3, label='ANG-MEAN', color='#1f77b4', zorder=4)
ax.plot(n_views, ang_med, marker='s', markersize=11, linewidth=3, label='ANG-MED', color='#ff7f0e', linestyle='--', zorder=4)

# 1 y 2. REMOVER LÍNEAS SUPERIOR Y DERECHA
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

# 3. MÁS CASILLAS PEQUEÑAS (Densidad de la cuadrícula)
ax.minorticks_on()
# Esto divide cada espacio entre números grandes en 5 sub-casillas
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

# Configuración de la rejilla (Estilo igual al original)
ax.grid(which='major', linestyle='--', linewidth=0.8, color='#d3d3d3', zorder=0)
ax.grid(which='minor', linestyle=':', linewidth=0.5, color='#e0e0e0', alpha=0.6, zorder=0)

# EJE X CON VALORES ESPECÍFICOS
ax.set_xticks(n_views)

# NÚMEROS EN NEGRILLA (Labels)
for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontweight('bold')
    label.set_fontsize(13)

# TICKS (STICKS) ENNEGRECIDOS
# Major: Gruesos y largos / Minor: Más cortos y finos
ax.tick_params(axis='both', which='major', length=10, width=2.5, direction='out', color='black')
ax.tick_params(axis='both', which='minor', length=5, width=1.2, direction='out', color='gray')

# --- DETALLES DE LEYENDA Y TÍTULO ---
ax.set_title('Planes Raw-Mesh: Angular Error vs. Views', fontsize=16, fontweight='bold', pad=20)
legend = ax.legend(loc='upper right', shadow=True, fancybox=True, borderpad=1, 
                   prop={'weight': 'bold', 'size': 12})
frame = legend.get_frame()
frame.set_edgecolor('gray')

plt.tight_layout()
plt.savefig('angular_error_vs_views.png', dpi=300)