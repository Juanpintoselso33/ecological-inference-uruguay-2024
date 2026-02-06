"""
Análisis Comparativo Electoral 2019 vs 2024
==========================================

Compara resultados de King's Ecological Inference entre ambas elecciones
para identificar cambios en patrones de transferencia de votos.

Autor: Electoral Analysis Project
Fecha: 2026-02-05
"""

import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
BASE_DIR = Path(r"E:\Proyectos VS CODE\Eco inference 2024")
OUTPUTS_DIR = BASE_DIR / "outputs"
RESULTS_DIR = OUTPUTS_DIR / "results"
TABLES_DIR = OUTPUTS_DIR / "tables"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Asegurar directorios existen
FIGURES_DIR.mkdir(exist_ok=True, parents=True)

# Mapeo de nombres departamentales
DEPT_MAP = {
    'AR': 'Artigas', 'CA': 'Canelones', 'CL': 'Cerro Largo', 'CO': 'Colonia',
    'DU': 'Durazno', 'FD': 'Florida', 'FS': 'Flores', 'LA': 'Lavalleja',
    'MA': 'Maldonado', 'MO': 'Montevideo', 'PA': 'Paysandú', 'RN': 'Río Negro',
    'RO': 'Rocha', 'RV': 'Rivera', 'SA': 'Salto', 'SJ': 'San José',
    'SO': 'Soriano', 'TA': 'Tacuarembó', 'TT': 'Treinta y Tres'
}

# =====================================================================
# CARGA DE DATOS
# =====================================================================

print("=" * 70)
print("ANÁLISIS COMPARATIVO: ELECCIONES 2019 vs 2024")
print("=" * 70)
print()

print("Cargando datos 2019...")
dept_2019 = pd.read_csv(TABLES_DIR / "transfers_by_department_2019.csv")
with open(RESULTS_DIR / "national_transfers_2019.pkl", 'rb') as f:
    national_2019 = pickle.load(f)

print("Cargando datos 2024...")
dept_2024 = pd.read_csv(TABLES_DIR / "transfers_by_department_with_pi.csv")
# No hay nacional 2024 con PI todavía, calculamos desde departamentos ponderados
# with open(RESULTS_DIR / "national_transfers_with_pi.pkl", 'rb') as f:
#     national_2024 = pickle.load(f)

# Normalizar nombres departamentales en 2019
dept_2019['departamento_full'] = dept_2019['departamento'].map(DEPT_MAP)

# =====================================================================
# CÁLCULO NACIONAL 2024 (ponderado por votos)
# =====================================================================

print("\nCalculando promedios nacionales ponderados por votos...")

def calc_national_weighted(df, party_cols, vote_cols):
    """Calcula promedio nacional ponderado por votos"""
    results = {}
    for party, vote_col in zip(party_cols, vote_cols):
        to_fa_col = f"{party}_to_fa"
        to_pn_col = f"{party}_to_pn"

        if to_fa_col in df.columns and vote_col in df.columns:
            total_votes = df[vote_col].sum()
            weighted_to_fa = (df[to_fa_col] * df[vote_col]).sum() / total_votes
            weighted_to_pn = (df[to_pn_col] * df[vote_col]).sum() / total_votes

            results[party] = {
                'to_fa': weighted_to_fa,
                'to_pn': weighted_to_pn,
                'votes': total_votes
            }
    return results

# 2024 nacional
parties_2024 = ['ca', 'pc', 'pi', 'pn']
vote_cols_2024 = ['ca_votes', 'pc_votes', 'pi_votes', 'pn_votes']
national_2024_calc = calc_national_weighted(dept_2024, parties_2024, vote_cols_2024)

# 2019 nacional (verificar contra pickle)
parties_2019 = ['ca', 'pc', 'pn', 'pi', 'otros']
vote_cols_2019 = ['ca_votes', 'pc_votes', 'pn_votes', 'pi_votes', 'otros_votes']
national_2019_calc = calc_national_weighted(dept_2019, parties_2019, vote_cols_2019)

print("\n2019 Nacional (ponderado):")
for party, data in national_2019_calc.items():
    print(f"  {party.upper()}: {data['to_fa']*100:.1f}% -> FA, {data['to_pn']*100:.1f}% -> PN")

print("\n2024 Nacional (ponderado):")
for party, data in national_2024_calc.items():
    print(f"  {party.upper()}: {data['to_fa']*100:.1f}% -> FA, {data['to_pn']*100:.1f}% -> PN")

# =====================================================================
# TABLA 1: COMPARACIÓN NACIONAL
# =====================================================================

print("\n" + "=" * 70)
print("GENERANDO TABLA 1: Comparación Nacional")
print("=" * 70)

comparison_rows = []

for party in ['ca', 'pc', 'pn', 'pi']:
    if party in national_2019_calc and party in national_2024_calc:
        d2019 = national_2019_calc[party]
        d2024 = national_2024_calc[party]

        defection_2019 = d2019['to_fa']
        defection_2024 = d2024['to_fa']
        change_pp = (defection_2024 - defection_2019) * 100  # puntos porcentuales

        votes_defected_2019 = d2019['votes'] * defection_2019
        votes_defected_2024 = d2024['votes'] * defection_2024
        change_abs = votes_defected_2024 - votes_defected_2019

        comparison_rows.append({
            'Partido': party.upper(),
            'Defeccion_2019_%': f"{defection_2019*100:.1f}",
            'Defeccion_2024_%': f"{defection_2024*100:.1f}",
            'Cambio_pp': f"{change_pp:+.1f}",
            'Votos_defectos_2019': f"{votes_defected_2019:,.0f}",
            'Votos_defectos_2024': f"{votes_defected_2024:,.0f}",
            'Cambio_absoluto': f"{change_abs:+,.0f}",
            'Votos_totales_2019': f"{d2019['votes']:,.0f}",
            'Votos_totales_2024': f"{d2024['votes']:,.0f}"
        })

df_comparison = pd.DataFrame(comparison_rows)
output_file = TABLES_DIR / "comparison_2019_2024.csv"
df_comparison.to_csv(output_file, index=False)
print(f"\nOK Guardado: {output_file}")
print("\nComparación Nacional:")
print(df_comparison.to_string(index=False))

# =====================================================================
# TABLA 2: CAMBIOS DEPARTAMENTALES
# =====================================================================

print("\n" + "=" * 70)
print("GENERANDO TABLA 2: Cambios Departamentales")
print("=" * 70)

# Merge datasets
merged = dept_2019.merge(
    dept_2024,
    left_on='departamento_full',
    right_on='departamento',
    suffixes=('_2019', '_2024')
)

# Calcular cambios
for party in ['ca', 'pc', 'pn']:
    col_2019 = f"{party}_to_fa_2019"
    col_2024 = f"{party}_to_fa_2024"
    if col_2019 in merged.columns and col_2024 in merged.columns:
        merged[f'{party}_change'] = (merged[col_2024] - merged[col_2019]) * 100

# Top 10 cambios por partido
changes_data = []

for party in ['ca', 'pc', 'pn']:
    change_col = f'{party}_change'
    if change_col in merged.columns:
        # Mayor cambio positivo (más defección)
        top_increase = merged.nlargest(5, change_col)[['departamento_2024', change_col]]
        for _, row in top_increase.iterrows():
            changes_data.append({
                'Partido': party.upper(),
                'Tipo': 'Mayor aumento',
                'Departamento': row['departamento_2024'],
                'Cambio_pp': f"{row[change_col]:+.1f}"
            })

        # Mayor cambio negativo (menos defección)
        top_decrease = merged.nsmallest(5, change_col)[['departamento_2024', change_col]]
        for _, row in top_decrease.iterrows():
            changes_data.append({
                'Partido': party.upper(),
                'Tipo': 'Mayor reducción',
                'Departamento': row['departamento_2024'],
                'Cambio_pp': f"{row[change_col]:+.1f}"
            })

df_changes = pd.DataFrame(changes_data)
output_file = TABLES_DIR / "department_changes_2019_2024.csv"
df_changes.to_csv(output_file, index=False)
print(f"\nOK Guardado: {output_file}")

print("\nTop 5 cambios por partido:")
for party in ['CA', 'PC', 'PN']:
    print(f"\n{party}:")
    subset = df_changes[df_changes['Partido'] == party]
    print(subset.to_string(index=False))

# =====================================================================
# TABLA 3: ANÁLISIS DE CORRELACIÓN
# =====================================================================

print("\n" + "=" * 70)
print("GENERANDO TABLA 3: Correlacion Geografica")
print("=" * 70)

correlation_rows = []

for party in ['ca', 'pc', 'pn']:
    col_2019 = f"{party}_to_fa_2019"
    col_2024 = f"{party}_to_fa_2024"

    if col_2019 in merged.columns and col_2024 in merged.columns:
        # Eliminar NaNs
        valid = merged[[col_2019, col_2024]].dropna()

        if len(valid) > 2:
            # Correlacion Pearson
            r, p_value = stats.pearsonr(valid[col_2019], valid[col_2024])

            # Regresión lineal
            slope, intercept, r_value, p_val_reg, std_err = stats.linregress(
                valid[col_2019], valid[col_2024]
            )

            correlation_rows.append({
                'Partido': party.upper(),
                'Correlacion_Pearson': f"{r:.3f}",
                'P_value': f"{p_value:.4f}",
                'Pendiente_regresión': f"{slope:.3f}",
                'R_squared': f"{r_value**2:.3f}",
                'N_departamentos': len(valid)
            })

df_correlation = pd.DataFrame(correlation_rows)
output_file = TABLES_DIR / "correlation_analysis.csv"
df_correlation.to_csv(output_file, index=False)
print(f"\nOK Guardado: {output_file}")
print("\nCorrelaciones geográficas 2019-2024:")
print(df_correlation.to_string(index=False))

# =====================================================================
# FIGURA 1-3: SCATTER PLOTS DEPARTAMENTALES
# =====================================================================

print("\n" + "=" * 70)
print("GENERANDO FIGURAS: Scatter Plots Departamentales")
print("=" * 70)

for party in ['ca', 'pc', 'pn']:
    col_2019 = f"{party}_to_fa_2019"
    col_2024 = f"{party}_to_fa_2024"

    if col_2019 in merged.columns and col_2024 in merged.columns:
        fig, ax = plt.subplots(figsize=(10, 8))

        # Datos válidos
        valid = merged[[col_2019, col_2024, 'departamento_2024']].dropna()
        x = valid[col_2019] * 100
        y = valid[col_2024] * 100

        # Scatter plot
        ax.scatter(x, y, s=100, alpha=0.6, edgecolors='black')

        # Etiquetas para departamentos extremos
        for idx, row in valid.iterrows():
            dept = row['departamento_2024']
            x_val = row[col_2019] * 100
            y_val = row[col_2024] * 100

            # Etiquetar si está lejos de la diagonal
            distance = abs(y_val - x_val)
            if distance > 15:  # Más de 15pp de diferencia
                ax.annotate(
                    dept,
                    (x_val, y_val),
                    xytext=(5, 5),
                    textcoords='offset points',
                    fontsize=8,
                    alpha=0.7
                )

        # Línea de identidad (sin cambio)
        lim = [0, max(x.max(), y.max()) + 5]
        ax.plot(lim, lim, 'k--', alpha=0.3, label='Sin cambio')

        # Regresión lineal
        if len(valid) > 2:
            slope, intercept, r_value, _, _ = stats.linregress(x, y)
            x_line = np.linspace(x.min(), x.max(), 100)
            y_line = slope * x_line + intercept
            ax.plot(
                x_line, y_line,
                'r-',
                alpha=0.5,
                label=f'Regresión (R²={r_value**2:.2f})'
            )

        ax.set_xlabel('Defeccion a FA 2019 (%)', fontsize=12)
        ax.set_ylabel('Defeccion a FA 2024 (%)', fontsize=12)
        ax.set_title(
            f'Defeccion {party.upper()} -> FA: 2019 vs 2024\n' +
            f'(Cambio geografico por departamento)',
            fontsize=14,
            fontweight='bold'
        )
        ax.legend(loc='best')
        ax.grid(True, alpha=0.3)

        output_file = FIGURES_DIR / f"scatter_{party}_2019_2024.png"
        plt.tight_layout()
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"OK Generado: {output_file}")

# =====================================================================
# FIGURA 4: BARRAS COHESIÓN DE COALICIÓN
# =====================================================================

print("\n" + "=" * 70)
print("GENERANDO FIGURA: Cohesión de Coalición")
print("=" * 70)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# 2019
parties_2019_plot = ['CA', 'PC', 'PN', 'PI']
retention_2019 = []
for party in ['ca', 'pc', 'pn', 'pi']:
    if party in national_2019_calc:
        retention_2019.append(national_2019_calc[party]['to_pn'] * 100)
    else:
        retention_2019.append(0)

defection_2019 = []
for party in ['ca', 'pc', 'pn', 'pi']:
    if party in national_2019_calc:
        defection_2019.append(national_2019_calc[party]['to_fa'] * 100)
    else:
        defection_2019.append(0)

x_pos = np.arange(len(parties_2019_plot))
ax1.bar(x_pos, retention_2019, label='Retencion (->PN)', color='#2E86AB', alpha=0.8)
ax1.bar(x_pos, defection_2019, bottom=retention_2019, label='Defeccion (->FA)', color='#D32F2F', alpha=0.8)
ax1.set_xticks(x_pos)
ax1.set_xticklabels(parties_2019_plot)
ax1.set_ylabel('Porcentaje (%)', fontsize=11)
ax1.set_title('2019: Cohesion Coalicion Republicana', fontsize=12, fontweight='bold')
ax1.legend(loc='lower left')
ax1.set_ylim([0, 105])
ax1.grid(axis='y', alpha=0.3)

# Añadir valores
for i, (ret, def_) in enumerate(zip(retention_2019, defection_2019)):
    ax1.text(i, ret/2, f'{ret:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold')
    ax1.text(i, ret + def_/2, f'{def_:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold')

# 2024
parties_2024_plot = ['CA', 'PC', 'PN', 'PI']
retention_2024 = []
for party in ['ca', 'pc', 'pn', 'pi']:
    if party in national_2024_calc:
        retention_2024.append(national_2024_calc[party]['to_pn'] * 100)
    else:
        retention_2024.append(0)

defection_2024 = []
for party in ['ca', 'pc', 'pn', 'pi']:
    if party in national_2024_calc:
        defection_2024.append(national_2024_calc[party]['to_fa'] * 100)
    else:
        defection_2024.append(0)

ax2.bar(x_pos, retention_2024, label='Retencion (->PN)', color='#2E86AB', alpha=0.8)
ax2.bar(x_pos, defection_2024, bottom=retention_2024, label='Defeccion (->FA)', color='#D32F2F', alpha=0.8)
ax2.set_xticks(x_pos)
ax2.set_xticklabels(parties_2024_plot)
ax2.set_ylabel('Porcentaje (%)', fontsize=11)
ax2.set_title('2024: Cohesion Coalicion Republicana', fontsize=12, fontweight='bold')
ax2.legend(loc='lower left')
ax2.set_ylim([0, 105])
ax2.grid(axis='y', alpha=0.3)

# Añadir valores
for i, (ret, def_) in enumerate(zip(retention_2024, defection_2024)):
    ax2.text(i, ret/2, f'{ret:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold')
    ax2.text(i, ret + def_/2, f'{def_:.0f}%', ha='center', va='center', fontsize=9, fontweight='bold')

plt.tight_layout()
output_file = FIGURES_DIR / "coalition_cohesion_comparison.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"OK Generado: {output_file}")

# =====================================================================
# FIGURA 5: FOREST PLOT TEMPORAL
# =====================================================================

print("\n" + "=" * 70)
print("GENERANDO FIGURA: Forest Plot Temporal (CA)")
print("=" * 70)

# Ordenar por cambio
ca_data = merged[['departamento_2024', 'ca_to_fa_2019', 'ca_to_fa_2024',
                   'ca_to_fa_lower_2019', 'ca_to_fa_upper_2019',
                   'ca_to_fa_lower_2024', 'ca_to_fa_upper_2024']].dropna()
ca_data['change'] = ca_data['ca_to_fa_2024'] - ca_data['ca_to_fa_2019']
ca_data = ca_data.sort_values('change')

fig, ax = plt.subplots(figsize=(12, 10))

y_pos = np.arange(len(ca_data))

# 2019 (izquierda, azul)
for i, (idx, row) in enumerate(ca_data.iterrows()):
    x_lower = (row['ca_to_fa_2019'] - row['ca_to_fa_lower_2019']) * 100
    x_upper = (row['ca_to_fa_upper_2019'] - row['ca_to_fa_2019']) * 100
    ax.errorbar(
        row['ca_to_fa_2019'] * 100,
        y_pos[i] - 0.15,
        xerr=[[x_lower], [x_upper]],
        fmt='o',
        color='#1976D2',
        markersize=6,
        capsize=3,
        alpha=0.7,
        label='2019' if i == 0 else None
    )

# 2024 (derecha, rojo)
for i, (idx, row) in enumerate(ca_data.iterrows()):
    x_lower = (row['ca_to_fa_2024'] - row['ca_to_fa_lower_2024']) * 100
    x_upper = (row['ca_to_fa_upper_2024'] - row['ca_to_fa_2024']) * 100
    ax.errorbar(
        row['ca_to_fa_2024'] * 100,
        y_pos[i] + 0.15,
        xerr=[[x_lower], [x_upper]],
        fmt='s',
        color='#D32F2F',
        markersize=6,
        capsize=3,
        alpha=0.7,
        label='2024' if i == 0 else None
    )

ax.set_yticks(y_pos)
ax.set_yticklabels(ca_data['departamento_2024'])
ax.set_xlabel('Defeccion CA -> FA (%)', fontsize=12)
ax.set_title(
    'Evolucion Temporal: Defeccion Cabildo Abierto por Departamento\n' +
    '(Intervalos de credibilidad 95%)',
    fontsize=14,
    fontweight='bold'
)
ax.legend(loc='best')
ax.grid(axis='x', alpha=0.3)
ax.axvline(50, color='gray', linestyle='--', alpha=0.3, label='50%')

plt.tight_layout()
output_file = FIGURES_DIR / "temporal_evolution_forest.png"
plt.savefig(output_file, dpi=300, bbox_inches='tight')
plt.close()
print(f"OK Generado: {output_file}")

# =====================================================================
# ANÁLISIS FINAL
# =====================================================================

print("\n" + "=" * 70)
print("ANÁLISIS COMPLETADO")
print("=" * 70)

print("\nRESUMEN DE HALLAZGOS CLAVE:")
print("-" * 70)

# CA paradox
ca_2019_def = national_2019_calc['ca']['to_fa'] * 100
ca_2024_def = national_2024_calc['ca']['to_fa'] * 100
print(f"\n1. PARADOJA CA:")
print(f"   - 2019: {ca_2019_def:.1f}% defección a FA")
print(f"   - 2024: {ca_2024_def:.1f}% defección a FA")
print(f"   - Cambio: {ca_2024_def - ca_2019_def:+.1f} pp")
print(f"   -> CA fue MÁS LEAL en 2024, pero la coalición perdió")

# Otros partidos
print(f"\n2. OTROS PARTIDOS:")
for party in ['pc', 'pn']:
    if party in national_2019_calc and party in national_2024_calc:
        def_2019 = national_2019_calc[party]['to_fa'] * 100
        def_2024 = national_2024_calc[party]['to_fa'] * 100
        change = def_2024 - def_2019
        print(f"   {party.upper()}: {def_2019:.1f}% (2019) -> {def_2024:.1f}% (2024) = {change:+.1f} pp")

# Estabilidad geográfica
print(f"\n3. ESTABILIDAD GEOGRÁFICA:")
for party in ['ca', 'pc', 'pn']:
    if party.upper() in df_correlation['Partido'].values:
        row = df_correlation[df_correlation['Partido'] == party.upper()].iloc[0]
        r = float(row['Correlacion_Pearson'])
        stability = "Alta" if r > 0.7 else "Media" if r > 0.4 else "Baja"
        print(f"   {party.upper()}: r={r:.2f} ({stability} correlación 2019-2024)")

print("\n" + "=" * 70)
print("Todos los archivos generados exitosamente.")
print("=" * 70)
print("\nARCHIVOS GENERADOS:")
print(f"  Tablas: {TABLES_DIR}")
print(f"    - comparison_2019_2024.csv")
print(f"    - department_changes_2019_2024.csv")
print(f"    - correlation_analysis.csv")
print(f"\n  Figuras: {FIGURES_DIR}")
print(f"    - scatter_ca_2019_2024.png")
print(f"    - scatter_pc_2019_2024.png")
print(f"    - scatter_pn_2019_2024.png")
print(f"    - coalition_cohesion_comparison.png")
print(f"    - temporal_evolution_forest.png")
print()
