import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

n_views = [1, 6, 14, 26]
p5 = [0.017, 0.062, 0.081, 0.107]
p10 = [0.167, 0.151, 0.203, 0.248]

fig, ax = plt.subplots(figsize=(9, 7))

# Líneas verde y morada con triángulos
ax.plot(n_views, p5, marker='^', markersize=11, linewidth=3, label='p@5°', color='#2ca02c', zorder=4)
ax.plot(n_views, p10, marker='v', markersize=11, linewidth=3, label='p@10°', color='#9467bd', linestyle='--', zorder=4)

ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['left'].set_linewidth(1.5)
ax.spines['bottom'].set_linewidth(1.5)

ax.minorticks_on()
ax.xaxis.set_minor_locator(ticker.AutoMinorLocator(5))
ax.yaxis.set_minor_locator(ticker.AutoMinorLocator(5))

ax.grid(which='major', linestyle='--', linewidth=0.8, color='#d3d3d3', zorder=0)
ax.grid(which='minor', linestyle=':', linewidth=0.5, color='#e0e0e0', alpha=0.6, zorder=0)

ax.set_xticks(n_views)

for label in (ax.get_xticklabels() + ax.get_yticklabels()):
    label.set_fontweight('bold')
    label.set_fontsize(13)

ax.tick_params(axis='both', which='major', length=10, width=2.5, direction='out', color='black')
ax.tick_params(axis='both', which='minor', length=5, width=1.2, direction='out', color='gray')

ax.set_title('Planes Raw-Mesh: Precision vs. Views', fontsize=16, fontweight='bold', pad=20)
legend = ax.legend(loc='upper left', shadow=True, fancybox=True, borderpad=1, prop={'weight': 'bold', 'size': 12})
legend.get_frame().set_edgecolor('gray')

plt.tight_layout()
plt.savefig('precision_vs_views.png', dpi=300)