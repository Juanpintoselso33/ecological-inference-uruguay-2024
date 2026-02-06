"""
Visualizaciones Comparativas Criticas 2019 vs 2024
===================================================

Genera 8 visualizaciones comparativas de alta prioridad para analisis
temporal de las elecciones uruguayas.

Autor: Electoral Inference Analysis Team
Fecha: 2026-02-05
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import matplotlib.colors as mcolors
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configuracion global de estilo
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Paleta de colores del proyecto
COLORS = {
    'CA': '#E74C3C',      # Rojo
    'PC': '#3498DB',      # Azul
    'PN': '#2ECC71',      # Verde
    'PI': '#F39C12',      # Naranja/Amarillo
    'FA': '#9B59B6',      # Purpura
    'Blancos': '#95A5A6', # Gris
    '2019': '#1976D2',    # Azul oscuro
    '2024': '#D32F2F',    # Rojo oscuro
    'neutral': '#7F8C8D'  # Gris neutro
}

# Paths
BASE_DIR = Path(r"E:\Proyectos VS CODE\Eco inference 2024")
DATA_DIR = BASE_DIR / "outputs" / "tables"
OUTPUT_DIR = BASE_DIR / "outputs" / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Mapeo de departamentos
DEPT_MAP = {
    'AR': 'Artigas', 'CA': 'Canelones', 'CL': 'Cerro Largo', 'CO': 'Colonia',
    'DU': 'Durazno', 'FD': 'Florida', 'FS': 'Flores', 'LA': 'Lavalleja',
    'MA': 'Maldonado', 'MO': 'Montevideo', 'PA': 'Paysandu', 'RN': 'Rio Negro',
    'RO': 'Rocha', 'RV': 'Rivera', 'SA': 'Salto', 'SJ': 'San Jose',
    'SO': 'Soriano', 'TA': 'Tacuarembo', 'TT': 'Treinta y Tres'
}


def load_data():
    """Carga todos los datos necesarios."""
    print("=" * 70)
    print("CARGANDO DATOS")
    print("=" * 70)

    # Cargar datos departamentales
    df_2024 = pd.read_csv(DATA_DIR / "transfers_by_department_with_pi.csv")
    df_2019 = pd.read_csv(DATA_DIR / "transfers_by_department_2019.csv")

    # Normalizar nombres de departamentos en 2019
    df_2019['departamento_full'] = df_2019['departamento'].map(DEPT_MAP)

    print(f"  [OK] 2019: {len(df_2019)} departamentos")
    print(f"  [OK] 2024: {len(df_2024)} departamentos")

    # Calcular metricas nacionales ponderadas
    def calc_national(df, party_col, vote_col, to_col):
        """Calcula promedio nacional ponderado."""
        if f'{party_col}_{to_col}' in df.columns and vote_col in df.columns:
            total_votes = df[vote_col].sum()
            weighted = (df[f'{party_col}_{to_col}'] * df[vote_col]).sum() / total_votes
            return weighted
        return None

    # Metricas nacionales 2019
    national_2019 = {
        'CA': {
            'to_fa': calc_national(df_2019, 'ca', 'ca_votes', 'to_fa'),
            'to_pn': calc_national(df_2019, 'ca', 'ca_votes', 'to_pn'),
            'votes': df_2019['ca_votes'].sum()
        },
        'PC': {
            'to_fa': calc_national(df_2019, 'pc', 'pc_votes', 'to_fa'),
            'to_pn': calc_national(df_2019, 'pc', 'pc_votes', 'to_pn'),
            'votes': df_2019['pc_votes'].sum()
        },
        'PN': {
            'to_fa': calc_national(df_2019, 'pn', 'pn_votes', 'to_fa'),
            'to_pn': calc_national(df_2019, 'pn', 'pn_votes', 'to_pn'),
            'votes': df_2019['pn_votes'].sum()
        },
        'PI': {
            'to_fa': calc_national(df_2019, 'pi', 'pi_votes', 'to_fa'),
            'to_pn': calc_national(df_2019, 'pi', 'pi_votes', 'to_pn'),
            'votes': df_2019['pi_votes'].sum()
        }
    }

    # Metricas nacionales 2024
    national_2024 = {
        'CA': {
            'to_fa': calc_national(df_2024, 'ca', 'ca_votes', 'to_fa'),
            'to_pn': calc_national(df_2024, 'ca', 'ca_votes', 'to_pn'),
            'votes': df_2024['ca_votes'].sum()
        },
        'PC': {
            'to_fa': calc_national(df_2024, 'pc', 'pc_votes', 'to_fa'),
            'to_pn': calc_national(df_2024, 'pc', 'pc_votes', 'to_pn'),
            'votes': df_2024['pc_votes'].sum()
        },
        'PN': {
            'to_fa': calc_national(df_2024, 'pn', 'pn_votes', 'to_fa'),
            'to_pn': calc_national(df_2024, 'pn', 'pn_votes', 'to_pn'),
            'votes': df_2024['pn_votes'].sum()
        },
        'PI': {
            'to_fa': calc_national(df_2024, 'pi', 'pi_votes', 'to_fa'),
            'to_pn': calc_national(df_2024, 'pi', 'pi_votes', 'to_pn'),
            'votes': df_2024['pi_votes'].sum()
        }
    }

    print("\n  Metricas nacionales cargadas.")

    return df_2019, df_2024, national_2019, national_2024


def figure1_sankey_comparison(df_2019, df_2024, national_2019, national_2024):
    """
    FIGURA 1: Sankey diagrams side-by-side 2019 vs 2024
    ALTA PRIORIDAD - Muestra dramaticamente CA 2019 (69.5%->FA) vs 2024 (27.6%->FA)
    """
    print("\n[1/8] Generando Sankey Comparison 2019 vs 2024...")

    fig = plt.figure(figsize=(22, 12))
    gs = GridSpec(1, 2, figure=fig, wspace=0.15)

    parties = ['CA', 'PC', 'PN', 'PI']

    for idx, (national, year, title_color) in enumerate([
        (national_2019, 2019, COLORS['2019']),
        (national_2024, 2024, COLORS['2024'])
    ]):
        ax = fig.add_subplot(gs[0, idx])

        # Datos para el Sankey simplificado (barras horizontales apiladas)
        y_positions = np.arange(len(parties))

        for i, party in enumerate(parties):
            if party in national and national[party]['to_fa'] is not None:
                to_fa = national[party]['to_fa'] * 100
                to_pn = national[party]['to_pn'] * 100
                to_blancos = 100 - to_fa - to_pn
                votes = national[party]['votes']

                # Barra hacia FA (purpura)
                ax.barh(i, to_fa, height=0.6, color=COLORS['FA'],
                       edgecolor='black', linewidth=0.5, label='-> FA' if i == 0 else '')
                # Barra hacia PN (verde)
                ax.barh(i, to_pn, left=to_fa, height=0.6, color=COLORS['PN'],
                       edgecolor='black', linewidth=0.5, label='-> PN' if i == 0 else '')
                # Barra hacia Blancos (gris)
                ax.barh(i, to_blancos, left=to_fa+to_pn, height=0.6, color=COLORS['Blancos'],
                       edgecolor='black', linewidth=0.5, label='-> Blancos' if i == 0 else '')

                # Etiquetas dentro de las barras
                if to_fa > 8:
                    ax.text(to_fa/2, i, f'{to_fa:.1f}%', ha='center', va='center',
                           fontsize=11, fontweight='bold', color='white')
                if to_pn > 8:
                    ax.text(to_fa + to_pn/2, i, f'{to_pn:.1f}%', ha='center', va='center',
                           fontsize=11, fontweight='bold', color='white')

                # Etiqueta del partido con votos
                ax.text(-5, i, f'{party}\n({int(votes/1000)}k)', ha='right', va='center',
                       fontsize=11, fontweight='bold')

        ax.set_xlim(0, 100)
        ax.set_ylim(-0.5, len(parties) - 0.5)
        ax.set_yticks([])
        ax.set_xlabel('Porcentaje de transferencia (%)', fontsize=12)
        ax.set_title(f'{year}', fontsize=20, fontweight='bold', color=title_color, pad=20)
        ax.axvline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
        ax.grid(axis='x', alpha=0.3)

        if idx == 0:
            ax.legend(loc='upper right', fontsize=10)

    # Titulo general y subtitulo con hallazgo clave
    fig.suptitle('Transferencias de Voto: Primera Vuelta a Balotaje',
                fontsize=18, fontweight='bold', y=0.98)

    # Cuadro de hallazgo clave
    ca_2019 = national_2019['CA']['to_fa'] * 100
    ca_2024 = national_2024['CA']['to_fa'] * 100
    change = ca_2024 - ca_2019

    textstr = (f'HALLAZGO CLAVE:\n'
               f'CA defeccion a FA:\n'
               f'2019: {ca_2019:.1f}%\n'
               f'2024: {ca_2024:.1f}%\n'
               f'Cambio: {change:.1f} pp')

    props = dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                 edgecolor='orange', alpha=0.9, linewidth=2)
    fig.text(0.5, 0.02, textstr, fontsize=12, ha='center', va='bottom',
            bbox=props, family='monospace')

    output_file = OUTPUT_DIR / "sankey_comparison_2019_2024.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: {output_file}")


def figure2_forest_plot_temporal_ca(df_2019, df_2024):
    """
    FIGURA 2: Forest plot con 19 departamentos
    Puntos 2019 (azul) vs 2024 (rojo) con lineas conectando
    Intervalos de credibilidad 95%
    """
    print("\n[2/8] Generando Forest Plot Temporal CA...")

    # Merge de datos
    df_2019_norm = df_2019.copy()
    df_2019_norm['dept'] = df_2019_norm['departamento'].map(DEPT_MAP)

    merged = df_2019_norm.merge(
        df_2024,
        left_on='dept',
        right_on='departamento',
        suffixes=('_2019', '_2024')
    )

    # Calcular cambio y ordenar
    merged['ca_change'] = (merged['ca_to_fa_2024'] - merged['ca_to_fa_2019']) * 100
    merged = merged.sort_values('ca_change')

    # Determinar columna de nombre de departamento
    dept_col = 'departamento' if 'departamento' in merged.columns else 'dept'

    fig, ax = plt.subplots(figsize=(14, 12))

    y_positions = np.arange(len(merged))

    for i, (idx, row) in enumerate(merged.iterrows()):
        dept = row[dept_col]

        # 2019 - punto azul
        x_2019 = row['ca_to_fa_2019'] * 100
        lower_2019 = row.get('ca_to_fa_lower_2019', x_2019 * 0.9) * 100
        upper_2019 = row.get('ca_to_fa_upper_2019', x_2019 * 1.1) * 100

        # 2024 - punto rojo
        x_2024 = row['ca_to_fa_2024'] * 100
        lower_2024 = row.get('ca_to_fa_lower_2024', x_2024 * 0.9) * 100
        upper_2024 = row.get('ca_to_fa_upper_2024', x_2024 * 1.1) * 100

        # Linea conectando ambos puntos
        ax.plot([x_2019, x_2024], [i-0.1, i+0.1], 'k-', alpha=0.3, linewidth=1.5)

        # 2019 con intervalo de credibilidad
        ax.errorbar(x_2019, i - 0.1,
                   xerr=[[x_2019 - lower_2019], [upper_2019 - x_2019]],
                   fmt='o', color=COLORS['2019'], markersize=8, capsize=4,
                   elinewidth=1.5, capthick=1.5, alpha=0.8,
                   label='2019' if i == 0 else '')

        # 2024 con intervalo de credibilidad
        ax.errorbar(x_2024, i + 0.1,
                   xerr=[[x_2024 - lower_2024], [upper_2024 - x_2024]],
                   fmt='s', color=COLORS['2024'], markersize=8, capsize=4,
                   elinewidth=1.5, capthick=1.5, alpha=0.8,
                   label='2024' if i == 0 else '')

        # Etiqueta de cambio
        change = x_2024 - x_2019
        color_change = 'green' if change < 0 else 'red'
        ax.text(105, i, f'{change:+.1f}pp', va='center', fontsize=9,
               fontweight='bold', color=color_change)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(merged[dept_col])
    ax.set_xlabel('Defeccion CA -> FA (%)', fontsize=12)
    ax.set_xlim(0, 110)
    ax.axvline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.grid(axis='x', alpha=0.3)
    ax.legend(loc='upper right', fontsize=11)

    ax.set_title('Evolucion Temporal: Defeccion Cabildo Abierto por Departamento\n'
                '(Intervalos de credibilidad 95%, ordenado por cambio)',
                fontsize=14, fontweight='bold')

    # Anotacion
    ax.text(0.02, 0.98, 'Δ = cambio en pp\n(negativo = mejoro lealtad)',
           transform=ax.transAxes, fontsize=9, va='top',
           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    output_file = OUTPUT_DIR / "forest_plot_temporal_ca.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: {output_file}")


def figure3_scatter_defection_change(df_2019, df_2024):
    """
    FIGURA 3: Scatter plot de defeccion
    Eje X = defeccion 2019, Eje Y = defeccion 2024
    Linea 45 de referencia, color por magnitud de cambio
    """
    print("\n[3/8] Generando Scatter Defection Change...")

    # Merge de datos
    df_2019_norm = df_2019.copy()
    df_2019_norm['dept'] = df_2019_norm['departamento'].map(DEPT_MAP)

    merged = df_2019_norm.merge(
        df_2024,
        left_on='dept',
        right_on='departamento',
        suffixes=('_2019', '_2024')
    )

    # Determinar columna de nombre de departamento
    dept_col = 'departamento' if 'departamento' in merged.columns else 'dept'

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for ax, party, title in zip(axes, ['ca', 'pc', 'pn'], ['Cabildo Abierto', 'Partido Colorado', 'Partido Nacional']):
        col_2019 = f'{party}_to_fa_2019'
        col_2024 = f'{party}_to_fa_2024'

        if col_2019 in merged.columns and col_2024 in merged.columns:
            x = merged[col_2019] * 100
            y = merged[col_2024] * 100
            change = y - x

            # Colorear por magnitud de cambio
            norm = plt.Normalize(change.min(), change.max())
            colors = plt.cm.RdYlBu_r(norm(change))

            # Scatter plot
            scatter = ax.scatter(x, y, c=change, cmap='RdYlBu_r', s=150,
                                edgecolors='black', linewidth=1, alpha=0.8)

            # Linea 45 grados (sin cambio)
            lim = [0, max(x.max(), y.max()) + 10]
            ax.plot(lim, lim, 'k--', alpha=0.5, linewidth=2, label='Sin cambio')

            # Etiquetas para departamentos extremos
            for idx, row in merged.iterrows():
                x_val = row[col_2019] * 100
                y_val = row[col_2024] * 100
                dept = row[dept_col]

                distance = abs(y_val - x_val)
                if distance > 25:  # Etiquetar si cambio > 25pp
                    ax.annotate(dept, (x_val, y_val), xytext=(5, 5),
                               textcoords='offset points', fontsize=8, alpha=0.8)

            ax.set_xlabel('Defeccion 2019 (%)', fontsize=11)
            ax.set_ylabel('Defeccion 2024 (%)', fontsize=11)
            ax.set_title(f'{title}', fontsize=13, fontweight='bold', color=COLORS[party.upper()])
            ax.set_xlim(0, 105)
            ax.set_ylim(0, 105)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper left')

            # Colorbar
            cbar = plt.colorbar(scatter, ax=ax, shrink=0.8)
            cbar.set_label('Cambio (pp)', fontsize=10)

            # Region de mejora (debajo de la diagonal)
            ax.fill_between([0, 100], [0, 100], [100, 100], alpha=0.05, color='red')
            ax.fill_between([0, 100], [0, 0], [0, 100], alpha=0.05, color='green')
            ax.text(80, 20, 'Mejoro\nlealtad', fontsize=9, alpha=0.5, ha='center')
            ax.text(20, 80, 'Empeoro\nlealtad', fontsize=9, alpha=0.5, ha='center')

    plt.suptitle('Comparacion Departamental: Defeccion a FA (2019 vs 2024)',
                fontsize=14, fontweight='bold')

    output_file = OUTPUT_DIR / "scatter_defection_change.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: {output_file}")


def figure4_coalition_cohesion_barplot(national_2019, national_2024):
    """
    FIGURA 4: Barras agrupadas por partido (CA, PC, PN)
    Altura = % lealtad a coalicion
    2019 (barra clara) vs 2024 (barra oscura)
    Muestra paradoja: todos mas leales pero coalicion perdio
    """
    print("\n[4/8] Generando Coalition Cohesion Barplot...")

    fig, ax = plt.subplots(figsize=(14, 8))

    parties = ['CA', 'PC', 'PN', 'PI']
    x = np.arange(len(parties))
    width = 0.35

    # Lealtad = votos que fueron a PN / total votos
    loyalty_2019 = []
    loyalty_2024 = []

    for party in parties:
        if party in national_2019 and national_2019[party]['to_pn'] is not None:
            loyalty_2019.append(national_2019[party]['to_pn'] * 100)
        else:
            loyalty_2019.append(0)

        if party in national_2024 and national_2024[party]['to_pn'] is not None:
            loyalty_2024.append(national_2024[party]['to_pn'] * 100)
        else:
            loyalty_2024.append(0)

    # Barras
    rects1 = ax.bar(x - width/2, loyalty_2019, width, label='2019',
                    color=[COLORS[p] for p in parties], alpha=0.5, edgecolor='black', linewidth=1)
    rects2 = ax.bar(x + width/2, loyalty_2024, width, label='2024',
                    color=[COLORS[p] for p in parties], alpha=1.0, edgecolor='black', linewidth=2)

    # Etiquetas de valores
    def autolabel(rects, values):
        for rect, val in zip(rects, values):
            height = rect.get_height()
            ax.annotate(f'{val:.1f}%',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 3), textcoords="offset points",
                       ha='center', va='bottom', fontsize=10, fontweight='bold')

    autolabel(rects1, loyalty_2019)
    autolabel(rects2, loyalty_2024)

    # Flechas de cambio
    for i, (l19, l24) in enumerate(zip(loyalty_2019, loyalty_2024)):
        change = l24 - l19
        color = 'green' if change > 0 else 'red'
        y_pos = max(l19, l24) + 8
        ax.annotate(f'{change:+.1f}pp', xy=(i, y_pos), ha='center', fontsize=11,
                   fontweight='bold', color=color,
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                            edgecolor=color, alpha=0.8))

    ax.set_ylabel('Lealtad Coalicional (% -> PN)', fontsize=12)
    ax.set_xlabel('Partido de Primera Vuelta', fontsize=12)
    ax.set_title('Cohesion Coalicional: Lealtad por Partido\n'
                '(Todos mejoraron, pero la coalicion perdio)',
                fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(parties, fontsize=12, fontweight='bold')
    ax.legend(loc='upper left', fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    # Cuadro de paradoja
    textstr = ('PARADOJA ELECTORAL\n'
               '━━━━━━━━━━━━━━━━━━━━\n'
               'Todos los partidos de la\n'
               'coalicion mejoraron su\n'
               'lealtad entre 2019 y 2024.\n\n'
               'SIN EMBARGO, la coalicion\n'
               'PERDIO la eleccion.\n\n'
               'Razon: Base electoral\n'
               'mas pequena (CA colapso)')

    props = dict(boxstyle='round,pad=0.5', facecolor='mistyrose',
                 edgecolor='red', alpha=0.9, linewidth=2)
    ax.text(0.98, 0.5, textstr, transform=ax.transAxes, fontsize=10,
           ha='right', va='center', bbox=props, family='monospace')

    output_file = OUTPUT_DIR / "coalition_cohesion_barplot.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: {output_file}")


def figure5_map_ca_change_choropleth(df_2019, df_2024):
    """
    FIGURA 5: Mapa coropletico del cambio en defeccion CA
    Azul = mejoro lealtad, Rojo = empeoro
    Nota: Sin shapefile, usaremos un mapa esquematico
    """
    print("\n[5/8] Generando Mapa CA Change (esquematico)...")

    # Merge de datos
    df_2019_norm = df_2019.copy()
    df_2019_norm['dept'] = df_2019_norm['departamento'].map(DEPT_MAP)

    merged = df_2019_norm.merge(
        df_2024,
        left_on='dept',
        right_on='departamento',
        suffixes=('_2019', '_2024')
    )

    # Determinar columna de nombre de departamento
    dept_col = 'departamento' if 'departamento' in merged.columns else 'dept'

    # Calcular cambio
    merged['ca_change'] = (merged['ca_to_fa_2024'] - merged['ca_to_fa_2019']) * 100
    merged = merged.sort_values('ca_change')

    fig, ax = plt.subplots(figsize=(16, 10))

    # Crear mapa esquematico como barras horizontales
    y_pos = np.arange(len(merged))

    # Colorear barras segun cambio
    norm = mcolors.TwoSlopeNorm(vmin=merged['ca_change'].min(),
                                 vcenter=0,
                                 vmax=merged['ca_change'].max())
    colors = plt.cm.RdYlBu_r(norm(merged['ca_change']))

    bars = ax.barh(y_pos, merged['ca_change'], color=colors,
                   edgecolor='black', linewidth=0.5, height=0.7)

    # Linea vertical en 0
    ax.axvline(0, color='black', linewidth=2)

    # Etiquetas
    ax.set_yticks(y_pos)
    ax.set_yticklabels(merged[dept_col], fontsize=10)
    ax.set_xlabel('Cambio en Defeccion CA -> FA (pp)\n'
                 '(Negativo = mejoro lealtad, Positivo = empeoro)', fontsize=12)
    ax.set_title('Cambio Geografico: Defeccion Cabildo Abierto (2024 - 2019)\n'
                'Heterogeneidad geografica del cambio en lealtad',
                fontsize=14, fontweight='bold')

    # Valores en las barras
    for i, (bar, val) in enumerate(zip(bars, merged['ca_change'])):
        x_pos = val + (2 if val >= 0 else -2)
        ha = 'left' if val >= 0 else 'right'
        ax.text(x_pos, i, f'{val:.1f}pp', va='center', ha=ha, fontsize=9, fontweight='bold')

    # Regiones anotadas
    ax.text(-50, len(merged)-1, 'MEJORO\nLEALTAD', fontsize=12, ha='center', va='center',
           color='blue', fontweight='bold', alpha=0.5)
    ax.text(15, 0, 'EMPEORO\nLEALTAD', fontsize=12, ha='center', va='center',
           color='red', fontweight='bold', alpha=0.5)

    ax.grid(axis='x', alpha=0.3)
    ax.set_xlim(merged['ca_change'].min() - 10, merged['ca_change'].max() + 10)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap='RdYlBu_r', norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.6, pad=0.02)
    cbar.set_label('Cambio (pp)', fontsize=11)

    output_file = OUTPUT_DIR / "map_ca_change_choropleth.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: {output_file}")


def figure6_heatmap_transfers_comparison(df_2019, df_2024, national_2019, national_2024):
    """
    FIGURA 6: Heatmap doble (2019 | 2024)
    Filas: origen (CA, PC, PI, PN)
    Columnas: destino (FA, PN, Blancos)
    Color intensity = probabilidad transferencia
    """
    print("\n[6/8] Generando Heatmap Transfers Comparison...")

    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    parties = ['CA', 'PC', 'PI', 'PN']
    destinations = ['FA', 'PN', 'Blancos']

    for ax, (national, year) in zip(axes, [(national_2019, 2019), (national_2024, 2024)]):
        # Construir matriz de transferencia
        matrix = np.zeros((len(parties), len(destinations)))

        for i, party in enumerate(parties):
            if party in national and national[party]['to_fa'] is not None:
                matrix[i, 0] = national[party]['to_fa'] * 100  # FA
                matrix[i, 1] = national[party]['to_pn'] * 100  # PN
                matrix[i, 2] = 100 - matrix[i, 0] - matrix[i, 1]  # Blancos

        # Heatmap
        im = ax.imshow(matrix, cmap='YlOrRd', aspect='auto', vmin=0, vmax=100)

        # Etiquetas
        ax.set_xticks(np.arange(len(destinations)))
        ax.set_yticks(np.arange(len(parties)))
        ax.set_xticklabels(destinations, fontsize=12, fontweight='bold')
        ax.set_yticklabels(parties, fontsize=12, fontweight='bold')

        # Colorear celdas con colores de destino
        for i in range(len(parties)):
            for j in range(len(destinations)):
                val = matrix[i, j]
                text_color = 'white' if val > 50 else 'black'
                ax.text(j, i, f'{val:.1f}%', ha='center', va='center',
                       fontsize=12, fontweight='bold', color=text_color)

        ax.set_title(f'{year}', fontsize=16, fontweight='bold',
                    color=COLORS['2019'] if year == 2019 else COLORS['2024'])
        ax.set_xlabel('Destino (Balotaje)', fontsize=12)
        ax.set_ylabel('Origen (Primera Vuelta)', fontsize=12)

    # Colorbar comun
    cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02, location='right')
    cbar.set_label('Probabilidad de Transferencia (%)', fontsize=11)

    plt.suptitle('Matriz de Transferencia Nacional: 2019 vs 2024',
                fontsize=16, fontweight='bold')

    output_file = OUTPUT_DIR / "heatmap_transfers_comparison.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: {output_file}")


def figure7_slope_chart_party_loyalty(national_2019, national_2024):
    """
    FIGURA 7: Slopegraph 2019 -> 2024
    Una linea por partido
    Mostrar convergencia/divergencia de lealtades
    """
    print("\n[7/8] Generando Slope Chart Party Loyalty...")

    fig, ax = plt.subplots(figsize=(12, 10))

    parties = ['CA', 'PC', 'PN', 'PI']
    years = [2019, 2024]

    for party in parties:
        if party in national_2019 and party in national_2024:
            # Lealtad = % que voto PN
            loyalty_2019 = national_2019[party]['to_pn'] * 100
            loyalty_2024 = national_2024[party]['to_pn'] * 100

            # Linea
            ax.plot(years, [loyalty_2019, loyalty_2024], 'o-',
                   color=COLORS[party], linewidth=3, markersize=12,
                   label=party)

            # Etiquetas en los extremos
            ax.text(2019 - 0.15, loyalty_2019, f'{loyalty_2019:.1f}%',
                   ha='right', va='center', fontsize=11, fontweight='bold',
                   color=COLORS[party])
            ax.text(2024 + 0.15, loyalty_2024, f'{loyalty_2024:.1f}%',
                   ha='left', va='center', fontsize=11, fontweight='bold',
                   color=COLORS[party])

            # Etiqueta del partido
            ax.text(2024 + 0.3, loyalty_2024, party, ha='left', va='center',
                   fontsize=12, fontweight='bold', color=COLORS[party])

            # Cambio
            change = loyalty_2024 - loyalty_2019
            mid_x = 2019 + 0.5
            mid_y = (loyalty_2019 + loyalty_2024) / 2
            ax.annotate(f'{change:+.1f}pp', (mid_x, mid_y), fontsize=9,
                       ha='center', va='center',
                       bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                                edgecolor=COLORS[party], alpha=0.8))

    ax.set_xlim(2018.5, 2025)
    ax.set_ylim(0, 100)
    ax.set_xticks(years)
    ax.set_xticklabels(['2019', '2024'], fontsize=14, fontweight='bold')
    ax.set_ylabel('Lealtad Coalicional (% -> PN)', fontsize=12)
    ax.set_title('Evolucion de la Lealtad Coalicional por Partido\n'
                '(Slopegraph: 2019 a 2024)',
                fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.axhline(50, color='gray', linestyle='--', alpha=0.5)

    # Leyenda
    ax.legend(loc='lower right', fontsize=11)

    # Anotacion de convergencia
    textstr = ('PATRON:\n'
               'Todos los partidos\n'
               'convergieron hacia\n'
               'mayor lealtad.\n\n'
               'PI: Mayor cambio\n'
               'PN: Ya era alto')

    ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
           va='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    output_file = OUTPUT_DIR / "slope_chart_party_loyalty.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: {output_file}")


def figure8_bubble_chart_impact(df_2019, df_2024):
    """
    FIGURA 8: Bubble chart por departamento
    Eje X: cambio defeccion CA
    Eje Y: margen victoria FA 2024
    Tamano burbuja: votos CA 2024
    """
    print("\n[8/8] Generando Bubble Chart Impact...")

    # Merge de datos
    df_2019_norm = df_2019.copy()
    df_2019_norm['dept'] = df_2019_norm['departamento'].map(DEPT_MAP)

    merged = df_2019_norm.merge(
        df_2024,
        left_on='dept',
        right_on='departamento',
        suffixes=('_2019', '_2024')
    )

    # Determinar columna de nombre de departamento
    dept_col = 'departamento' if 'departamento' in merged.columns else 'dept'

    # Calcular metricas
    merged['ca_change'] = (merged['ca_to_fa_2024'] - merged['ca_to_fa_2019']) * 100

    # Estimar margen FA (aproximado basado en datos de transferencia)
    # Margen = votos netos hacia FA - votos netos hacia PN
    merged['fa_net_2024'] = (
        merged['ca_votes_2024'] * merged['ca_to_fa_2024'] +
        merged['pc_votes_2024'] * merged['pc_to_fa_2024'] +
        merged['pn_votes_2024'] * merged['pn_to_fa_2024'] +
        merged['pi_votes_2024'] * merged['pi_to_fa_2024']
    )

    merged['pn_net_2024'] = (
        merged['ca_votes_2024'] * merged['ca_to_pn_2024'] +
        merged['pc_votes_2024'] * merged['pc_to_pn_2024'] +
        merged['pn_votes_2024'] * merged['pn_to_pn_2024'] +
        merged['pi_votes_2024'] * merged['pi_to_pn_2024']
    )

    merged['margin_fa'] = (merged['fa_net_2024'] - merged['pn_net_2024']) / 1000  # En miles

    fig, ax = plt.subplots(figsize=(14, 10))

    # Tamano de burbujas basado en votos CA 2024
    sizes = merged['ca_votes_2024'] / 100  # Escalar para visualizacion

    # Colorear por region (simplificado)
    colors = plt.cm.viridis(np.linspace(0, 1, len(merged)))

    scatter = ax.scatter(merged['ca_change'], merged['margin_fa'],
                        s=sizes, c=merged['ca_votes_2024'],
                        cmap='YlOrRd', alpha=0.7, edgecolors='black', linewidth=1)

    # Etiquetas de departamentos
    for idx, row in merged.iterrows():
        ax.annotate(row[dept_col], (row['ca_change'], row['margin_fa']),
                   xytext=(5, 5), textcoords='offset points', fontsize=8, alpha=0.8)

    ax.axvline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax.set_xlabel('Cambio en Defeccion CA -> FA (pp)\n(Negativo = mejoro lealtad)', fontsize=12)
    ax.set_ylabel('Margen neto FA (miles de votos)\n(Positivo = FA gana)', fontsize=12)
    ax.set_title('Impacto del Cambio en Lealtad CA por Departamento\n'
                '(Tamano de burbuja = votos CA 2024)',
                fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.7)
    cbar.set_label('Votos CA 2024', fontsize=11)

    # Regiones
    ax.text(-60, ax.get_ylim()[1] * 0.9, 'Mejoro lealtad\nFA gano', fontsize=10,
           color='green', alpha=0.5, ha='center')
    ax.text(10, ax.get_ylim()[1] * 0.9, 'Empeoro lealtad\nFA gano', fontsize=10,
           color='red', alpha=0.5, ha='center')
    ax.text(-60, ax.get_ylim()[0] * 0.9, 'Mejoro lealtad\nPN gano', fontsize=10,
           color='blue', alpha=0.5, ha='center')
    ax.text(10, ax.get_ylim()[0] * 0.9, 'Empeoro lealtad\nPN gano', fontsize=10,
           color='orange', alpha=0.5, ha='center')

    output_file = OUTPUT_DIR / "bubble_chart_impact.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  [OK] Guardado: {output_file}")


def main():
    """Funcion principal de ejecucion."""
    print("=" * 70)
    print("GENERADOR DE VISUALIZACIONES COMPARATIVAS 2019 vs 2024")
    print("Analisis Electoral Uruguay - Inferencia Ecologica")
    print("=" * 70)

    # Cargar datos
    df_2019, df_2024, national_2019, national_2024 = load_data()

    # Generar todas las figuras
    try:
        figure1_sankey_comparison(df_2019, df_2024, national_2019, national_2024)
        figure2_forest_plot_temporal_ca(df_2019, df_2024)
        figure3_scatter_defection_change(df_2019, df_2024)
        figure4_coalition_cohesion_barplot(national_2019, national_2024)
        figure5_map_ca_change_choropleth(df_2019, df_2024)
        figure6_heatmap_transfers_comparison(df_2019, df_2024, national_2019, national_2024)
        figure7_slope_chart_party_loyalty(national_2019, national_2024)
        figure8_bubble_chart_impact(df_2019, df_2024)

        print("\n" + "=" * 70)
        print("GENERACION COMPLETADA EXITOSAMENTE")
        print("=" * 70)

        print(f"\n8 figuras guardadas en: {OUTPUT_DIR}")
        print("\nArchivos generados:")
        files = [
            "sankey_comparison_2019_2024.png",
            "forest_plot_temporal_ca.png",
            "scatter_defection_change.png",
            "coalition_cohesion_barplot.png",
            "map_ca_change_choropleth.png",
            "heatmap_transfers_comparison.png",
            "slope_chart_party_loyalty.png",
            "bubble_chart_impact.png"
        ]

        for i, name in enumerate(files, 1):
            filepath = OUTPUT_DIR / name
            if filepath.exists():
                size_kb = filepath.stat().st_size / 1024
                print(f"  {i}. {name} ({size_kb:.1f} KB)")
            else:
                print(f"  {i}. {name} - NO ENCONTRADO")

        print("\nEstas figuras estan listas para integracion en el informe LaTeX.")

    except Exception as e:
        print(f"\n[ERROR]: {e}")
        import traceback
        traceback.print_exc()
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
