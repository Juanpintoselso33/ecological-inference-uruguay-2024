"""
Análisis comprehensivo de TODOS los partidos de la coalición.
Incluyendo CA, PC, OTROS y PN.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.size'] = 11

# Load data (with PI if available)
print("Loading data...")
try:
    regional = pd.read_csv('outputs/tables/region_with_pi.csv')
    print("  Loaded: region_with_pi.csv")
except FileNotFoundError:
    regional = pd.read_csv('outputs/tables/region_detailed_results.csv')
    print("  Loaded: region_detailed_results.csv (fallback)")

try:
    urban = pd.read_csv('outputs/tables/urban_rural_with_pi.csv')
    urban = urban[urban['stratum'] == 'urbano'].iloc[0]
    rural_data = pd.read_csv('outputs/tables/urban_rural_with_pi.csv')
    rural = rural_data[rural_data['stratum'] == 'rural'].iloc[0]
    print("  Loaded: urban_rural_with_pi.csv")
except FileNotFoundError:
    urban = pd.read_csv('outputs/tables/transition_matrix_urban.csv', index_col=0)
    rural = pd.read_csv('outputs/tables/transition_matrix_rural.csv', index_col=0)
    print("  Loaded: transition_matrix_urban/rural.csv (fallback)")

try:
    national = pd.read_csv('outputs/tables/national_transition_matrix_with_pi.csv')
    print("  Loaded: national_transition_matrix_with_pi.csv")
except FileNotFoundError:
    national = pd.read_csv('outputs/tables/national_transitions_with_ci_final.csv')
    print("  Loaded: national_transitions_with_ci_final.csv (fallback)")

print("\n" + "="*70)
print("ANÁLISIS COMPLETO DE LA COALICIÓN")
print("="*70)

# ============================================================================
# NATIONAL LEVEL - ALL PARTIES
# ============================================================================
print("\n" + "="*70)
print("NIVEL NACIONAL - DEFECCIONES A FA")
print("="*70)

national_fa = national[national['destination'] == 'FA'].copy()
national_fa = national_fa[national_fa['origin'].isin(['CA', 'PC', 'OTROS', 'PN'])]

print("\nTransferencias a Frente Amplio:")
for _, row in national_fa.iterrows():
    print(f"  {row['origin']:6s} -> FA: {row['probability']*100:5.1f}% "
          f"[{row['ci_95_lower']*100:5.1f}%, {row['ci_95_upper']*100:5.1f}%]")

# Calculate absolute votes
votes_primera = {
    'CA': 59408,
    'PC': 386703,
    'PN': 645131,
    'OTROS': 0  # Need to calculate
}

print("\n" + "="*70)
print("IMPACTO EN VOTOS ABSOLUTOS")
print("="*70)

# From national data
ca_to_fa = 30173
pc_to_fa = 43184
pn_to_fa = 49548
otros_to_fa = national_fa[national_fa['origin'] == 'OTROS']['probability'].values[0] * 212073  # Total OTROS votes

print(f"\nVotos que fueron a FA desde coalicion:")
print(f"  CA -> FA:    {ca_to_fa:>7,.0f} votos (50.8% de 59,408)")
print(f"  PC -> FA:    {pc_to_fa:>7,.0f} votos (11.2% de 386,703) - MAYOR APORTE")
print(f"  PN -> FA:    {pn_to_fa:>7,.0f} votos (7.7% de 645,131)")
print(f"  OTROS -> FA: {otros_to_fa:>7,.0f} votos (0.3% de ~212,000)")
print(f"  {'-'*50}")
print(f"  TOTAL:     {ca_to_fa + pc_to_fa + pn_to_fa:>7,.0f} votos")

# ============================================================================
# URBAN VS RURAL - ALL PARTIES
# ============================================================================
print("\n" + "="*70)
print("URBANO vs RURAL - TODOS LOS PARTIDOS")
print("="*70)

coalition_parties = ['CA', 'PC', 'PI', 'PN']

print("\nDefecciones a FA:")
print(f"{'Partido':<10} {'Urbano':>10} {'Rural':>10} {'Diferencia':>12} {'Patrón':<15}")
print("-" * 70)

for party in coalition_parties:
    urban_rate = urban.loc[party, 'FA'] * 100
    rural_rate = rural.loc[party, 'FA'] * 100
    diff = rural_rate - urban_rate

    if diff > 5:
        pattern = "Rural+ (colapso)"
    elif diff < -5:
        pattern = "Urbano+ (sorpresa)"
    else:
        pattern = "Similar"

    print(f"{party:<10} {urban_rate:>9.1f}% {rural_rate:>9.1f}% "
          f"{diff:>+10.1f} pp  {pattern:<15}")

# ============================================================================
# METROPOLITAN VS INTERIOR - ALL PARTIES
# ============================================================================
print("\n" + "="*70)
print("METROPOLITANA vs INTERIOR - TODOS LOS PARTIDOS")
print("="*70)

metro = regional[regional['region'] == 'Area_Metropolitana']
interior = regional[regional['region'] == 'Interior']

print("\nDefecciones a FA:")
print(f"{'Partido':<10} {'Metropolitana':>14} {'Interior':>10} {'Diferencia':>12} {'Patrón':<15}")
print("-" * 75)

for party in coalition_parties:
    metro_rate = metro[(metro['origin_party'] == party) &
                       (metro['destination_party'] == 'FA')]['mean'].values[0] * 100
    interior_rate = interior[(interior['origin_party'] == party) &
                            (interior['destination_party'] == 'FA')]['mean'].values[0] * 100
    diff = interior_rate - metro_rate

    if diff > 5:
        pattern = "Interior+ (crisis)"
    elif diff < -5:
        pattern = "Metro+ (sorpresa)"
    else:
        pattern = "Similar"

    print(f"{party:<10} {metro_rate:>13.1f}% {interior_rate:>9.1f}% "
          f"{diff:>+10.1f} pp  {pattern:<15}")

# ============================================================================
# KEY FINDINGS
# ============================================================================
print("\n" + "="*70)
print("HALLAZGOS CLAVE")
print("="*70)

print("\n1. PARTIDO COLORADO (PC):")
print(f"   - Nacional: 11.2% -> FA (43,184 votos - MAYOR CONTRIBUCION)")
print(f"   - Rural 2.5x urbano: 18.2% vs 7.2%")
print(f"   - Patron similar a CA: rural/interior colapsa mas")

print("\n2. CABILDO ABIERTO (CA):")
print(f"   - Nacional: 50.8% -> FA (30,173 votos)")
print(f"   - 'Rebelion rural': 60.7% rural vs 47.4% urbano")
print(f"   - Interior mas inestable: 55.5% vs 47.4% metro")

print("\n3. OTROS PARTIDOS:")
print(f"   - Nacional: 0.3% -> FA (muy leal a coalicion)")
print(f"   - PATRON OPUESTO: Metropolitana defecta MAS (21.3% vs 12.3% interior)")
print(f"   - Sugiere dinamica diferente - posiblemente Partido Independiente urbano")

print("\n4. PARTIDO NACIONAL (PN):")
print(f"   - Nacional: 7.7% -> FA (49,548 votos)")
print(f"   - Colapso interior: 8.1% vs 0.17% metro (48x diferencia)")
print(f"   - Erosion de base tradicional rural")

# ============================================================================
# CREATE COMPREHENSIVE VISUALIZATION
# ============================================================================
print("\n" + "="*70)
print("GENERANDO VISUALIZACIONES")
print("="*70)

# Figure 1: Urban vs Rural comparison - ALL PARTIES
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

parties = ['CA', 'PC', 'OTROS', 'PN']
urban_rates = [urban.loc[p, 'FA'] * 100 for p in parties]
rural_rates = [rural.loc[p, 'FA'] * 100 for p in parties]

x = np.arange(len(parties))
width = 0.35

bars1 = ax1.bar(x - width/2, urban_rates, width, label='Urbano',
                color='steelblue', alpha=0.8)
bars2 = ax1.bar(x + width/2, rural_rates, width, label='Rural',
                color='coral', alpha=0.8)

ax1.set_ylabel('Defección a FA (%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Partido de Origen', fontsize=12, fontweight='bold')
ax1.set_title('Defecciones por Urbanización', fontsize=13, fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(parties)
ax1.legend(fontsize=11)
ax1.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

# Figure 2: Metropolitan vs Interior comparison
metro_rates = []
interior_rates = []

for party in parties:
    m = metro[(metro['origin_party'] == party) &
              (metro['destination_party'] == 'FA')]['mean'].values[0] * 100
    i = interior[(interior['origin_party'] == party) &
                (interior['destination_party'] == 'FA')]['mean'].values[0] * 100
    metro_rates.append(m)
    interior_rates.append(i)

bars3 = ax2.bar(x - width/2, metro_rates, width, label='Metropolitana',
                color='mediumseagreen', alpha=0.8)
bars4 = ax2.bar(x + width/2, interior_rates, width, label='Interior',
                color='indianred', alpha=0.8)

ax2.set_ylabel('Defección a FA (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Partido de Origen', fontsize=12, fontweight='bold')
ax2.set_title('Defecciones por Región', fontsize=13, fontweight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(parties)
ax2.legend(fontsize=11)
ax2.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars3, bars4]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}%', ha='center', va='bottom', fontsize=9)

plt.tight_layout()
plt.savefig('outputs/figures/all_coalition_parties_comparison.png',
            dpi=300, bbox_inches='tight')
print("OK Saved: outputs/figures/all_coalition_parties_comparison.png")

# Figure 2: Heatmap of defection patterns
fig, ax = plt.subplots(figsize=(10, 8))

# Create matrix for heatmap
strata = ['Urbano', 'Rural', 'Metropolitana', 'Interior']
data_matrix = []

for party in parties:
    row = [
        urban.loc[party, 'FA'] * 100,
        rural.loc[party, 'FA'] * 100,
        metro[(metro['origin_party'] == party) &
              (metro['destination_party'] == 'FA')]['mean'].values[0] * 100,
        interior[(interior['origin_party'] == party) &
                (interior['destination_party'] == 'FA')]['mean'].values[0] * 100
    ]
    data_matrix.append(row)

df_heatmap = pd.DataFrame(data_matrix, index=parties, columns=strata)

sns.heatmap(df_heatmap, annot=True, fmt='.1f', cmap='RdYlBu_r',
            cbar_kws={'label': 'Defección a FA (%)'},
            linewidths=1, linecolor='white',
            vmin=0, vmax=70, ax=ax)

ax.set_title('Matriz de Defecciones: Todos los Partidos de la Coalición',
             fontsize=14, fontweight='bold', pad=20)
ax.set_xlabel('Estratificación Geográfica', fontsize=12, fontweight='bold')
ax.set_ylabel('Partido de Origen', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/figures/coalition_defection_heatmap.png',
            dpi=300, bbox_inches='tight')
print("OK Saved: outputs/figures/coalition_defection_heatmap.png")

# Figure 3: Vote impact - absolute numbers
fig, ax = plt.subplots(figsize=(10, 7))

parties_impact = ['CA', 'PC', 'PN']
votes_impact = [ca_to_fa, pc_to_fa, pn_to_fa]
colors_impact = ['#e74c3c', '#3498db', '#2ecc71']

bars = ax.barh(parties_impact, votes_impact, color=colors_impact, alpha=0.7)

ax.set_xlabel('Votos transferidos a FA', fontsize=12, fontweight='bold')
ax.set_ylabel('Partido de Origen', fontsize=12, fontweight='bold')
ax.set_title('Impacto Absoluto: Votos de Coalición que fueron a FA',
             fontsize=13, fontweight='bold')
ax.grid(axis='x', alpha=0.3)

# Add value labels
for i, (bar, votes) in enumerate(zip(bars, votes_impact)):
    width = bar.get_width()
    ax.text(width + 1000, bar.get_y() + bar.get_height()/2.,
            f'{votes:,.0f} votos', va='center', fontsize=11, fontweight='bold')

# Add percentages
rates = [50.8, 11.2, 7.7]
for i, (bar, rate) in enumerate(zip(bars, rates)):
    ax.text(bar.get_width()/2, bar.get_y() + bar.get_height()/2.,
            f'{rate}%', va='center', ha='center',
            color='white', fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig('outputs/figures/coalition_vote_impact.png',
            dpi=300, bbox_inches='tight')
print("OK Saved: outputs/figures/coalition_vote_impact.png")

print("\n" + "="*70)
print("ANÁLISIS COMPLETO FINALIZADO")
print("="*70)
print("\nArchivos generados:")
print("  • outputs/figures/all_coalition_parties_comparison.png")
print("  • outputs/figures/coalition_defection_heatmap.png")
print("  • outputs/figures/coalition_vote_impact.png")
