"""
Figura de síntesis del análisis comparativo 2019-2024
Muestra los hallazgos clave en un solo gráfico comprehensivo
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np

# Paths
BASE_DIR = Path(r"E:\Proyectos VS CODE\Eco inference 2024")
FIGURES_DIR = BASE_DIR / "outputs" / "figures"

# Datos clave
parties = ['CA', 'PC', 'PN', 'PI']

# Defección a FA (%)
defection_2019 = [69.5, 33.8, 36.0, 94.8]
defection_2024 = [27.6, 7.2, 3.6, 2.8]
change_pp = [-41.9, -26.6, -32.4, -92.1]

# Votos totales (en miles)
votes_2019 = [268, 300, 696, 24]
votes_2024 = [59, 387, 646, 41]

# Crear figura con 3 subplots - MUCHO MÁS GRANDE
fig = plt.figure(figsize=(20, 16))
gs = fig.add_gridspec(3, 2, hspace=0.45, wspace=0.35, top=0.88, bottom=0.05)

# Colores
color_2019 = '#1976D2'
color_2024 = '#D32F2F'
color_change = '#4CAF50'

# =====================================================================
# Panel 1: Defección a FA (comparación)
# =====================================================================
ax1 = fig.add_subplot(gs[0, :])

x = np.arange(len(parties))
width = 0.35

bars1 = ax1.bar(x - width/2, defection_2019, width, label='2019', color=color_2019, alpha=0.8)
bars2 = ax1.bar(x + width/2, defection_2024, width, label='2024', color=color_2024, alpha=0.8)

# Añadir valores
for i, (v1, v2) in enumerate(zip(defection_2019, defection_2024)):
    ax1.text(i - width/2, v1 + 2, f'{v1:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax1.text(i + width/2, v2 + 2, f'{v2:.1f}%', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_ylabel('Defeccion a FA (%)', fontsize=12, fontweight='bold')
ax1.set_title('HALLAZGO CLAVE: Todos los partidos redujeron dramaticamente su defeccion',
              fontsize=14, fontweight='bold', pad=15)
ax1.set_xticks(x)
ax1.set_xticklabels(parties, fontsize=12)
ax1.legend(loc='upper right', fontsize=11)
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim([0, 105])

# Añadir anotación
ax1.text(0.02, 0.97, 'MEJORA GENERALIZADA EN COHESION',
         transform=ax1.transAxes, fontsize=11, fontweight='bold',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# =====================================================================
# Panel 2: Cambio en puntos porcentuales
# =====================================================================
ax2 = fig.add_subplot(gs[1, 0])

colors_change = [color_change if c < 0 else '#FF5722' for c in change_pp]
bars = ax2.barh(parties, change_pp, color=colors_change, alpha=0.8)

# Añadir valores
for i, (p, v) in enumerate(zip(parties, change_pp)):
    ax2.text(v - 3 if v < 0 else v + 3, i, f'{v:+.1f}pp',
             ha='right' if v < 0 else 'left', va='center',
             fontsize=10, fontweight='bold')

ax2.set_xlabel('Cambio en defeccion (puntos porcentuales)', fontsize=11, fontweight='bold')
ax2.set_title('Cambio 2019 -> 2024', fontsize=12, fontweight='bold')
ax2.axvline(0, color='black', linewidth=1, linestyle='--', alpha=0.5)
ax2.grid(axis='x', alpha=0.3)

# Añadir anotación
ax2.text(0.02, 0.97, 'CA: -41.9 pp\nMAS LEAL',
         transform=ax2.transAxes, fontsize=10, fontweight='bold',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.7))

# =====================================================================
# Panel 3: Votos totales (tamaño electoral)
# =====================================================================
ax3 = fig.add_subplot(gs[1, 1])

x = np.arange(len(parties))
width = 0.35

bars1 = ax3.bar(x - width/2, votes_2019, width, label='2019', color=color_2019, alpha=0.8)
bars2 = ax3.bar(x + width/2, votes_2024, width, label='2024', color=color_2024, alpha=0.8)

# Añadir valores
for i, (v1, v2) in enumerate(zip(votes_2019, votes_2024)):
    ax3.text(i - width/2, v1 + 20, f'{v1}k', ha='center', va='bottom', fontsize=9, fontweight='bold')
    ax3.text(i + width/2, v2 + 20, f'{v2}k', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax3.set_ylabel('Votos 1ra vuelta (miles)', fontsize=11, fontweight='bold')
ax3.set_title('Tamano electoral (primera vuelta)', fontsize=12, fontweight='bold')
ax3.set_xticks(x)
ax3.set_xticklabels(parties, fontsize=12)
ax3.legend(loc='upper right', fontsize=10)
ax3.grid(axis='y', alpha=0.3)

# Añadir anotación
ax3.text(0.02, 0.97, 'CA: -78%\nCOLAPSO',
         transform=ax3.transAxes, fontsize=10, fontweight='bold',
         verticalalignment='top', bbox=dict(boxstyle='round', facecolor='salmon', alpha=0.7))

# =====================================================================
# Panel 4: Texto de síntesis
# =====================================================================
ax4 = fig.add_subplot(gs[2, :])
ax4.axis('off')

synthesis_text = """PARADOJA ELECTORAL 2024: ¿Por que perdio la coalicion si mejoro su cohesion?

• TODOS los partidos de la coalicion redujeron drasticamente su defeccion a FA:
    CA: 69.5% → 27.6% (-41.9 pp)  |  PC: 33.8% → 7.2% (-26.6 pp)
    PN: 36.0% → 3.6% (-32.4 pp)   |  PI: 94.8% → 2.8% (-92.1 pp)

• PERO la coalicion perdio 154,251 votos en primera vuelta:
    - CA colapso: -208,722 votos (-78%)
    - PN se achico: -49,957 votos (-7%)
    - PC crecio: +87,327 votos (+29%) [no suficiente para compensar]

• CONCLUSION CLAVE:
  La cohesion en segunda vuelta NO compensa la derrota en primera vuelta.
  La coalicion debe enfocarse en CRECER su base electoral inicial, no solo en retener.

• HALLAZGO GEOGRAFICO:
  Correlacion 2019-2024 ≈ 0 (r < 0.13 para todos los partidos)
  Los patrones departamentales de 2024 NO fueron predecibles desde 2019.
  Esto sugiere RECONFIGURACION PROFUNDA del electorado uruguayo."""

ax4.text(0.5, 0.5, synthesis_text,
         transform=ax4.transAxes,
         fontsize=12,
         verticalalignment='center',
         horizontalalignment='center',
         family='monospace',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3, pad=30),
         linespacing=1.5)

# Título general
fig.suptitle('ANALISIS COMPARATIVO ELECTORAL: URUGUAY 2019 vs 2024\n' +
             'King\'s Ecological Inference - 7,200+ circuitos',
             fontsize=18, fontweight='bold')

# Guardar
output_file = FIGURES_DIR / "synthesis_2019_2024.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()

print(f"OK Generado: {output_file}")
print("\nFigura de sintesis creada exitosamente.")
