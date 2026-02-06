#!/usr/bin/env python3
"""
Granular Department Analysis for Uruguay 2024 Elections
========================================================
Analyzes vote transfer patterns at department level to identify:
1. Pivot departments where margins were tight
2. Geographic clusters by defection patterns
3. Coalition complementarity indices
4. Temporal changes 2019-2024
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Configure paths
BASE_DIR = Path(r"E:\Proyectos VS CODE\Eco inference 2024")
DATA_DIR = BASE_DIR / "data" / "processed"
OUTPUT_TABLES = BASE_DIR / "outputs" / "tables"
OUTPUT_FIGURES = BASE_DIR / "outputs" / "figures"

# Ensure output directories exist
OUTPUT_TABLES.mkdir(parents=True, exist_ok=True)
OUTPUT_FIGURES.mkdir(parents=True, exist_ok=True)

# Department name mapping (2019 uses abbreviations)
DEPT_ABBREV_TO_NAME = {
    'AR': 'Artigas', 'CA': 'Canelones', 'CL': 'Cerro Largo',
    'CO': 'Colonia', 'DU': 'Durazno', 'FD': 'Florida',
    'FS': 'Flores', 'LA': 'Lavalleja', 'MA': 'Maldonado',
    'MO': 'Montevideo', 'PA': 'Paysandu', 'RN': 'Rio Negro',
    'RO': 'Rocha', 'RV': 'Rivera', 'SA': 'Salto',
    'SJ': 'San Jose', 'SO': 'Soriano', 'TA': 'Tacuarembo',
    'TT': 'Treinta y Tres'
}

# Reverse mapping
DEPT_NAME_TO_ABBREV = {v: k for k, v in DEPT_ABBREV_TO_NAME.items()}

# Also handle special characters
DEPT_NAME_NORMALIZE = {
    'Paysandú': 'Paysandu',
    'Río Negro': 'Rio Negro',
    'Tacuarembó': 'Tacuarembo',
    'San José': 'San Jose'
}


def load_data():
    """Load 2024 and 2019 transfer data."""
    # Load 2024 data
    df_2024 = pd.read_csv(OUTPUT_TABLES / "transfers_by_department_with_pi.csv")

    # Load 2019 data
    df_2019 = pd.read_csv(OUTPUT_TABLES / "transfers_by_department_2019.csv")

    # Load merged circuit data for actual vote totals
    df_circuits = pd.read_parquet(DATA_DIR / "circuitos_merged.parquet")

    return df_2024, df_2019, df_circuits


def normalize_department_names(df, col='departamento'):
    """Normalize department names for matching."""
    df = df.copy()
    # Apply character normalization
    for old, new in DEPT_NAME_NORMALIZE.items():
        df[col] = df[col].str.replace(old, new)
    return df


def calculate_ballotage_margins(df_circuits):
    """Calculate FA vs PN margins by department."""
    dept_totals = df_circuits.groupby('departamento').agg({
        'fa_ballotage': 'sum',
        'pn_ballotage': 'sum',
        'total_ballotage': 'sum'
    }).reset_index()

    dept_totals['fa_pct'] = dept_totals['fa_ballotage'] / dept_totals['total_ballotage'] * 100
    dept_totals['pn_pct'] = dept_totals['pn_ballotage'] / dept_totals['total_ballotage'] * 100
    dept_totals['margin_fa_pn'] = dept_totals['fa_pct'] - dept_totals['pn_pct']
    dept_totals['margin_votes'] = dept_totals['fa_ballotage'] - dept_totals['pn_ballotage']
    dept_totals['winner'] = np.where(dept_totals['margin_fa_pn'] > 0, 'FA', 'PN')

    return normalize_department_names(dept_totals)


def analyze_pivot_departments(df_2024, df_margins):
    """
    Identify pivot departments where:
    1. Margin was close (<5%)
    2. CA defection could have changed the outcome
    """
    print("\n" + "="*60)
    print("1. PIVOT DEPARTMENTS ANALYSIS")
    print("="*60)

    # Normalize names
    df_2024 = normalize_department_names(df_2024)

    # Merge transfers with margins
    df = df_2024.merge(df_margins, on='departamento', how='left')

    # Calculate CA defection impact
    df['ca_defected_to_fa'] = df['ca_votes'] * df['ca_to_fa']
    df['ca_defected_to_pn'] = df['ca_votes'] * df['ca_to_pn']
    df['ca_net_impact'] = df['ca_defected_to_fa'] - df['ca_defected_to_pn']

    # If CA had been 100% loyal to PN
    df['hypothetical_pn_gain'] = df['ca_defected_to_fa']  # Votes that would stay with coalition
    df['hypothetical_new_margin'] = df['margin_votes'] - 2 * df['hypothetical_pn_gain']
    df['would_flip'] = (df['margin_votes'] > 0) & (df['hypothetical_new_margin'] < 0)

    # Find close departments
    df['abs_margin_pct'] = df['margin_fa_pn'].abs()
    df_close = df[df['abs_margin_pct'] < 10].sort_values('abs_margin_pct')

    # Top 5 closest margins
    top5_pivot = df_close.head(10).copy()

    print("\nTop 10 Departamentos con Margen Mas Cerrado:")
    print("-" * 80)
    for _, row in top5_pivot.iterrows():
        flip_status = "HABRIA CAMBIADO" if row['would_flip'] else "Se mantiene"
        print(f"{row['departamento']:15s} | Margen: {row['margin_fa_pn']:+6.2f}% ({int(row['margin_votes']):+6d} votos) | "
              f"CA->FA: {int(row['ca_defected_to_fa']):5d} | {flip_status}")

    # Prepare output table
    pivot_output = df[['departamento', 'fa_pct', 'pn_pct', 'margin_fa_pn', 'margin_votes',
                       'ca_votes', 'ca_to_fa', 'ca_to_pn', 'ca_defected_to_fa', 'ca_defected_to_pn',
                       'ca_net_impact', 'hypothetical_new_margin', 'would_flip', 'winner']].copy()
    pivot_output = pivot_output.sort_values('margin_fa_pn', key=abs)

    # Save to CSV
    pivot_output.to_csv(OUTPUT_TABLES / "department_pivot_analysis.csv", index=False)
    print(f"\nGuardado: {OUTPUT_TABLES / 'department_pivot_analysis.csv'}")

    return pivot_output


def cluster_departments(df_2024):
    """
    Cluster departments by defection patterns using K-means (k=3).
    Features: ca_to_fa, pc_to_fa, pi_to_fa, pn_to_fa
    """
    print("\n" + "="*60)
    print("2. GEOGRAPHIC CLUSTERS BY DEFECTION PATTERN")
    print("="*60)

    df = normalize_department_names(df_2024.copy())

    # Features for clustering
    features = ['ca_to_fa', 'pc_to_fa', 'pi_to_fa', 'pn_to_fa']
    X = df[features].values

    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # K-means with k=3
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    df['cluster'] = kmeans.fit_predict(X_scaled)

    # Calculate cluster centers in original scale
    centers = scaler.inverse_transform(kmeans.cluster_centers_)

    # Determine cluster labels based on CA defection
    cluster_ca_means = df.groupby('cluster')['ca_to_fa'].mean().sort_values(ascending=False)
    cluster_labels = {
        cluster_ca_means.index[0]: 'Alta Defeccion CA',
        cluster_ca_means.index[1]: 'Defeccion Moderada',
        cluster_ca_means.index[2]: 'Baja Defeccion CA'
    }
    df['cluster_label'] = df['cluster'].map(cluster_labels)

    # Print cluster summary
    print("\nCaracteristicas de Clusters:")
    print("-" * 80)
    for cluster_id in cluster_ca_means.index:
        label = cluster_labels[cluster_id]
        cluster_data = df[df['cluster'] == cluster_id]
        print(f"\n{label} (n={len(cluster_data)}):")
        print(f"  CA->FA: {cluster_data['ca_to_fa'].mean():.1%} [{cluster_data['ca_to_fa'].min():.1%} - {cluster_data['ca_to_fa'].max():.1%}]")
        print(f"  PC->FA: {cluster_data['pc_to_fa'].mean():.1%}")
        print(f"  PI->FA: {cluster_data['pi_to_fa'].mean():.1%}")
        print(f"  PN->FA: {cluster_data['pn_to_fa'].mean():.1%}")
        print(f"  Departamentos: {', '.join(cluster_data['departamento'].tolist())}")

    # Prepare output
    cluster_output = df[['departamento', 'cluster', 'cluster_label', 'ca_to_fa', 'pc_to_fa',
                         'pi_to_fa', 'pn_to_fa', 'ca_votes', 'pc_votes', 'pi_votes', 'pn_votes']].copy()
    cluster_output = cluster_output.sort_values(['cluster', 'ca_to_fa'], ascending=[True, False])

    cluster_output.to_csv(OUTPUT_TABLES / "department_clusters.csv", index=False)
    print(f"\nGuardado: {OUTPUT_TABLES / 'department_clusters.csv'}")

    return df, cluster_labels


def analyze_coalition_complementarity(df_2024, df_margins):
    """
    Analyze coalition strength by party in each department.
    Calculate dependency indices.
    """
    print("\n" + "="*60)
    print("3. COALITION COMPLEMENTARITY ANALYSIS")
    print("="*60)

    df = normalize_department_names(df_2024.copy())
    df = df.merge(normalize_department_names(df_margins[['departamento', 'total_ballotage', 'pn_ballotage']]),
                  on='departamento', how='left')

    # Calculate total coalition votes in primera vuelta
    df['coalition_total'] = df['pn_votes'] + df['pc_votes'] + df['ca_votes'] + df['pi_votes']

    # Party share within coalition
    df['pn_coalition_share'] = df['pn_votes'] / df['coalition_total']
    df['pc_coalition_share'] = df['pc_votes'] / df['coalition_total']
    df['ca_coalition_share'] = df['ca_votes'] / df['coalition_total']
    df['pi_coalition_share'] = df['pi_votes'] / df['coalition_total']

    # Calculate "loyalty-weighted contribution" - how much each party contributes when loyal
    # Contribution = share * loyalty rate
    df['pn_loyalty_contrib'] = df['pn_coalition_share'] * df['pn_to_pn']
    df['pc_loyalty_contrib'] = df['pc_coalition_share'] * df['pc_to_pn']
    df['ca_loyalty_contrib'] = df['ca_coalition_share'] * df['ca_to_pn']
    df['pi_loyalty_contrib'] = df['pi_coalition_share'] * df['pi_to_pn']

    # Total loyalty contribution
    df['total_loyalty_contrib'] = (df['pn_loyalty_contrib'] + df['pc_loyalty_contrib'] +
                                    df['ca_loyalty_contrib'] + df['pi_loyalty_contrib'])

    # Dependency indices (how much does coalition depend on each party's loyalty)
    df['ca_dependency_idx'] = df['ca_coalition_share'] * (1 - df['ca_to_pn'])  # Higher = more loss from disloyalty
    df['pc_dependency_idx'] = df['pc_coalition_share'] * (1 - df['pc_to_pn'])
    df['pi_dependency_idx'] = df['pi_coalition_share'] * (1 - df['pi_to_pn'])

    # Identify where CA is critical vs PC
    df['ca_more_critical'] = df['ca_dependency_idx'] > df['pc_dependency_idx']

    print("\nIndice de Dependencia de CA (mayor = mas critico):")
    print("-" * 60)
    ca_critical = df.sort_values('ca_dependency_idx', ascending=False)
    for _, row in ca_critical.head(10).iterrows():
        critical_party = "CA" if row['ca_more_critical'] else "PC"
        print(f"{row['departamento']:15s} | CA dep: {row['ca_dependency_idx']:.3f} | "
              f"PC dep: {row['pc_dependency_idx']:.3f} | Mas critico: {critical_party}")

    print("\n\nDepartamentos donde PC es mas critico que CA:")
    pc_critical = df[~df['ca_more_critical']].sort_values('pc_dependency_idx', ascending=False)
    for _, row in pc_critical.iterrows():
        print(f"  {row['departamento']:15s} | PC dep: {row['pc_dependency_idx']:.3f} | CA dep: {row['ca_dependency_idx']:.3f}")

    # Prepare output
    comp_output = df[['departamento', 'coalition_total',
                      'pn_coalition_share', 'pc_coalition_share', 'ca_coalition_share', 'pi_coalition_share',
                      'pn_to_pn', 'pc_to_pn', 'ca_to_pn', 'pi_to_pn',
                      'ca_dependency_idx', 'pc_dependency_idx', 'pi_dependency_idx',
                      'ca_more_critical']].copy()
    comp_output = comp_output.sort_values('ca_dependency_idx', ascending=False)

    comp_output.to_csv(OUTPUT_TABLES / "coalition_complementarity.csv", index=False)
    print(f"\nGuardado: {OUTPUT_TABLES / 'coalition_complementarity.csv'}")

    return comp_output


def analyze_temporal_change(df_2024, df_2019):
    """
    Calculate changes in defection patterns between 2019 and 2024.
    """
    print("\n" + "="*60)
    print("4. TEMPORAL CHANGE ANALYSIS (2019 -> 2024)")
    print("="*60)

    # Normalize department names
    df_24 = normalize_department_names(df_2024.copy())
    df_19 = df_2019.copy()

    # Map 2019 abbreviations to full names
    df_19['departamento'] = df_19['departamento'].map(DEPT_ABBREV_TO_NAME)
    df_19 = normalize_department_names(df_19)

    # Merge datasets
    merged = df_24.merge(df_19, on='departamento', suffixes=('_2024', '_2019'), how='inner')

    # Calculate changes
    merged['delta_ca_to_fa'] = merged['ca_to_fa_2024'] - merged['ca_to_fa_2019']
    merged['delta_ca_to_pn'] = merged['ca_to_pn_2024'] - merged['ca_to_pn_2019']
    merged['delta_pc_to_fa'] = merged['pc_to_fa_2024'] - merged['pc_to_fa_2019']
    merged['delta_pn_to_fa'] = merged['pn_to_fa_2024'] - merged['pn_to_fa_2019']

    # Sort by CA loyalty change (positive = more loyal to PN)
    merged_sorted = merged.sort_values('delta_ca_to_fa')

    print("\nTop 5 departamentos donde CA se volvio MAS LEAL a PN (menor defeccion a FA):")
    print("-" * 80)
    for _, row in merged_sorted.head(5).iterrows():
        print(f"{row['departamento']:15s} | 2019: {row['ca_to_fa_2019']:.1%} -> 2024: {row['ca_to_fa_2024']:.1%} | "
              f"Delta: {row['delta_ca_to_fa']:+.1%}")

    print("\nTop 5 departamentos donde CA se volvio MENOS LEAL a PN (mayor defeccion a FA):")
    print("-" * 80)
    for _, row in merged_sorted.tail(5).iloc[::-1].iterrows():
        print(f"{row['departamento']:15s} | 2019: {row['ca_to_fa_2019']:.1%} -> 2024: {row['ca_to_fa_2024']:.1%} | "
              f"Delta: {row['delta_ca_to_fa']:+.1%}")

    # National average change
    avg_delta = merged['delta_ca_to_fa'].mean()
    print(f"\nCambio promedio nacional en defeccion CA->FA: {avg_delta:+.1%}")

    # Prepare output
    temporal_output = merged[['departamento',
                              'ca_to_fa_2019', 'ca_to_fa_2024', 'delta_ca_to_fa',
                              'ca_to_pn_2019', 'ca_to_pn_2024', 'delta_ca_to_pn',
                              'pc_to_fa_2019', 'pc_to_fa_2024', 'delta_pc_to_fa',
                              'pn_to_fa_2019', 'pn_to_fa_2024', 'delta_pn_to_fa']].copy()
    temporal_output = temporal_output.sort_values('delta_ca_to_fa')

    temporal_output.to_csv(OUTPUT_TABLES / "temporal_change_ranking.csv", index=False)
    print(f"\nGuardado: {OUTPUT_TABLES / 'temporal_change_ranking.csv'}")

    return temporal_output


def create_visualizations(pivot_df, cluster_df, cluster_labels, complementarity_df, temporal_df):
    """Create all visualizations."""
    print("\n" + "="*60)
    print("5. GENERATING VISUALIZATIONS")
    print("="*60)

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')

    # 1. Pivot departments barplot
    fig, ax = plt.subplots(figsize=(12, 8))
    pivot_sorted = pivot_df.sort_values('margin_fa_pn')
    colors = ['#1f77b4' if m > 0 else '#d62728' for m in pivot_sorted['margin_fa_pn']]
    bars = ax.barh(pivot_sorted['departamento'], pivot_sorted['margin_fa_pn'], color=colors)

    # Add CA defection annotations
    for i, (_, row) in enumerate(pivot_sorted.iterrows()):
        if row['would_flip']:
            ax.annotate('*PIVOTE*', xy=(row['margin_fa_pn'], i),
                       xytext=(5, 0), textcoords='offset points',
                       fontsize=8, color='red', fontweight='bold')

    ax.axvline(x=0, color='black', linewidth=0.8)
    ax.axvline(x=5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.axvline(x=-5, color='gray', linewidth=0.5, linestyle='--', alpha=0.5)
    ax.set_xlabel('Margen FA - PN (%)', fontsize=12)
    ax.set_title('Margen Electoral por Departamento (Ballotage 2024)\nAzul = Victoria FA, Rojo = Victoria PN', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "pivot_departments_barplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {OUTPUT_FIGURES / 'pivot_departments_barplot.png'}")

    # 2. Cluster scatter plot (CA vs PC defection)
    fig, ax = plt.subplots(figsize=(10, 8))

    cluster_colors = {0: '#e41a1c', 1: '#377eb8', 2: '#4daf4a'}
    for cluster_id, label in cluster_labels.items():
        cluster_data = cluster_df[cluster_df['cluster'] == cluster_id]
        ax.scatter(cluster_data['ca_to_fa'], cluster_data['pc_to_fa'],
                  label=label, s=100, alpha=0.7, c=cluster_colors.get(cluster_id, 'gray'))

        # Add department labels
        for _, row in cluster_data.iterrows():
            ax.annotate(row['departamento'][:3], (row['ca_to_fa'], row['pc_to_fa']),
                       fontsize=8, alpha=0.8)

    ax.set_xlabel('Defeccion CA -> FA', fontsize=12)
    ax.set_ylabel('Defeccion PC -> FA', fontsize=12)
    ax.set_title('Clusters de Departamentos por Patron de Defeccion', fontsize=14)
    ax.legend(loc='upper right')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 0.6)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "department_clusters_map.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {OUTPUT_FIGURES / 'department_clusters_map.png'}")

    # 3. Complementarity heatmap
    fig, ax = plt.subplots(figsize=(14, 10))

    # Prepare data for heatmap
    heatmap_data = complementarity_df.set_index('departamento')[
        ['pn_coalition_share', 'pc_coalition_share', 'ca_coalition_share', 'pi_coalition_share',
         'pn_to_pn', 'pc_to_pn', 'ca_to_pn', 'pi_to_pn']
    ].copy()

    # Rename columns for clarity
    heatmap_data.columns = ['PN Share', 'PC Share', 'CA Share', 'PI Share',
                            'PN Lealtad', 'PC Lealtad', 'CA Lealtad', 'PI Lealtad']

    # Sort by CA dependency
    heatmap_data = heatmap_data.loc[complementarity_df.sort_values('ca_dependency_idx', ascending=False)['departamento']]

    sns.heatmap(heatmap_data, annot=True, fmt='.2f', cmap='RdYlGn',
                center=0.5, ax=ax, linewidths=0.5,
                cbar_kws={'label': 'Proporcion'})
    ax.set_title('Complementariedad de Coalicion por Departamento\n(Ordenado por Dependencia de CA)', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "complementarity_heatmap.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {OUTPUT_FIGURES / 'complementarity_heatmap.png'}")

    # 4. Temporal change plot
    fig, ax = plt.subplots(figsize=(12, 8))

    temporal_sorted = temporal_df.sort_values('delta_ca_to_fa')
    colors = ['#2ca02c' if d < 0 else '#d62728' for d in temporal_sorted['delta_ca_to_fa']]

    y_pos = range(len(temporal_sorted))
    ax.barh(y_pos, temporal_sorted['delta_ca_to_fa'] * 100, color=colors, alpha=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(temporal_sorted['departamento'])
    ax.axvline(x=0, color='black', linewidth=0.8)

    # Add 2019 and 2024 values as text
    for i, (_, row) in enumerate(temporal_sorted.iterrows()):
        ax.annotate(f"{row['ca_to_fa_2019']:.0%}->{row['ca_to_fa_2024']:.0%}",
                   xy=(row['delta_ca_to_fa'] * 100, i),
                   xytext=(5 if row['delta_ca_to_fa'] >= 0 else -5, 0),
                   textcoords='offset points',
                   ha='left' if row['delta_ca_to_fa'] >= 0 else 'right',
                   fontsize=8, alpha=0.8)

    ax.set_xlabel('Cambio en Defeccion CA->FA (puntos porcentuales)', fontsize=12)
    ax.set_title('Cambio Temporal en Defeccion de Cabildo Abierto (2019 vs 2024)\nVerde = Mas leal a PN, Rojo = Menos leal', fontsize=14)
    plt.tight_layout()
    plt.savefig(OUTPUT_FIGURES / "temporal_change_barplot.png", dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Guardado: {OUTPUT_FIGURES / 'temporal_change_barplot.png'}")

    print("\nTodas las visualizaciones generadas exitosamente.")


def generate_summary_report(pivot_df, cluster_df, complementarity_df, temporal_df):
    """Generate a summary report of key findings."""
    print("\n" + "="*60)
    print("6. SUMMARY REPORT")
    print("="*60)

    print("\n--- HALLAZGOS CLAVE ---\n")

    # Pivot analysis
    flipped = pivot_df[pivot_df['would_flip']]
    print(f"1. DEPARTAMENTOS PIVOTE:")
    print(f"   - {len(flipped)} departamentos habrian cambiado de mano si CA fuera 100% leal")
    if len(flipped) > 0:
        print(f"   - Departamentos: {', '.join(flipped['departamento'].tolist())}")
        total_swing = flipped['ca_defected_to_fa'].sum() * 2
        print(f"   - Votos potencialmente en juego: {int(total_swing):,}")

    # Cluster analysis
    print(f"\n2. CLUSTERS GEOGRAFICOS:")
    for label in cluster_df['cluster_label'].unique():
        cluster_data = cluster_df[cluster_df['cluster_label'] == label]
        print(f"   - {label}: {len(cluster_data)} departamentos")
        print(f"     Promedio CA->FA: {cluster_data['ca_to_fa'].mean():.1%}")

    # Complementarity
    ca_critical_count = complementarity_df['ca_more_critical'].sum()
    print(f"\n3. CRITICIDAD DE CA:")
    print(f"   - CA es mas critico que PC en {ca_critical_count} de 19 departamentos")
    top_ca_dep = complementarity_df.nlargest(3, 'ca_dependency_idx')
    print(f"   - Top 3 dependientes de CA: {', '.join(top_ca_dep['departamento'].tolist())}")

    # Temporal
    more_loyal = (temporal_df['delta_ca_to_fa'] < 0).sum()
    less_loyal = (temporal_df['delta_ca_to_fa'] > 0).sum()
    print(f"\n4. CAMBIO TEMPORAL:")
    print(f"   - CA se volvio MAS leal en {more_loyal} departamentos")
    print(f"   - CA se volvio MENOS leal en {less_loyal} departamentos")
    avg_change = temporal_df['delta_ca_to_fa'].mean()
    print(f"   - Cambio promedio: {avg_change:+.1%} ({'mas defeccion' if avg_change > 0 else 'menos defeccion'})")


def main():
    """Main execution function."""
    print("="*60)
    print("ANALISIS GRANULAR DE DEPARTAMENTOS - URUGUAY 2024")
    print("="*60)

    # Load data
    print("\nCargando datos...")
    df_2024, df_2019, df_circuits = load_data()
    print(f"  - 2024: {len(df_2024)} departamentos")
    print(f"  - 2019: {len(df_2019)} departamentos")
    print(f"  - Circuitos: {len(df_circuits)} registros")

    # Calculate margins
    df_margins = calculate_ballotage_margins(df_circuits)

    # Run analyses
    pivot_df = analyze_pivot_departments(df_2024, df_margins)
    cluster_df, cluster_labels = cluster_departments(df_2024)
    complementarity_df = analyze_coalition_complementarity(df_2024, df_margins)
    temporal_df = analyze_temporal_change(df_2024, df_2019)

    # Create visualizations
    create_visualizations(pivot_df, cluster_df, cluster_labels, complementarity_df, temporal_df)

    # Generate summary
    generate_summary_report(pivot_df, cluster_df, complementarity_df, temporal_df)

    print("\n" + "="*60)
    print("ANALISIS COMPLETADO")
    print("="*60)
    print(f"\nArchivos generados:")
    print(f"  - {OUTPUT_TABLES / 'department_pivot_analysis.csv'}")
    print(f"  - {OUTPUT_TABLES / 'department_clusters.csv'}")
    print(f"  - {OUTPUT_TABLES / 'coalition_complementarity.csv'}")
    print(f"  - {OUTPUT_TABLES / 'temporal_change_ranking.csv'}")
    print(f"  - {OUTPUT_FIGURES / 'pivot_departments_barplot.png'}")
    print(f"  - {OUTPUT_FIGURES / 'department_clusters_map.png'}")
    print(f"  - {OUTPUT_FIGURES / 'complementarity_heatmap.png'}")
    print(f"  - {OUTPUT_FIGURES / 'temporal_change_barplot.png'}")


if __name__ == "__main__":
    main()
