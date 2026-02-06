"""
Análisis de Dinámica de Participación Electoral
Analiza cambios en participación entre primera vuelta y ballotage
y su impacto en resultados electorales
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

from src.utils.config import get_config
from src.utils.logger import get_logger
from src.visualization.styles import (
    setup_professional_style,
    apply_tableau_style,
    save_publication_figure,
    PARTY_COLORS
)

logger = get_logger(__name__)
setup_professional_style()


def load_data(config):
    """Load processed data."""
    logger.info("Cargando datos...")

    data_dirs = config.get_data_dirs()
    processed_dir = Path(data_dirs['processed'])

    # Try with covariates first
    data_path_cov = processed_dir / 'circuitos_full_covariates.parquet'
    data_path = processed_dir / 'circuitos_merged.parquet'

    if data_path_cov.exists():
        df = pd.read_parquet(data_path_cov)
        logger.info(f"Datos cargados (con covariables): {len(df)} circuitos")
    else:
        df = pd.read_parquet(data_path)
        logger.info(f"Datos cargados: {len(df)} circuitos")

    return df


def calculate_turnout_metrics(df):
    """Calculate turnout metrics."""
    logger.info("Calculando metricas de participacion...")

    df = df.copy()

    # Basic turnout change - always recalculate (don't trust pre-existing columns)
    df['participacion_change'] = df['total_ballotage'] - df['total_primera']
    df['participacion_change_pct'] = (df['participacion_change'] / df['total_primera']) * 100

    # Vote changes by party
    df['fa_change'] = df['fa_ballotage'] - df['fa_primera']
    df['pn_change'] = df['pn_ballotage'] - (df.get('pn_primera', 0))

    # Coalition change (PN + PC + CA primera → PN ballotage)
    coalition_primera = df.get('pn_primera', 0) + df.get('pc_primera', 0) + df.get('ca_primera', 0)
    df['coalition_change'] = df['pn_ballotage'] - coalition_primera

    # Party shares in primera vuelta (for classification)
    df['fa_share_primera'] = df['fa_primera'] / df['total_primera']
    df['pn_share_primera'] = df.get('pn_primera', 0) / df['total_primera']
    df['coalition_share_primera'] = coalition_primera / df['total_primera']

    # Dominant party in primera vuelta
    def get_dominant(row):
        if row['fa_share_primera'] > row['coalition_share_primera']:
            return 'FA'
        else:
            return 'Coalicion'

    df['dominant_party'] = df.apply(get_dominant, axis=1)

    # Swing (change in FA share)
    df['fa_swing'] = (df['fa_ballotage'] / df['total_ballotage']) - df['fa_share_primera']

    # Mobilization efficiency (what % of new votes went to each party)
    df['fa_mobilization_eff'] = np.where(
        df['participacion_change'] > 0,
        df['fa_change'] / df['participacion_change'],
        np.nan
    )

    df['pn_mobilization_eff'] = np.where(
        df['participacion_change'] > 0,
        df['pn_change'] / df['participacion_change'],
        np.nan
    )

    logger.info(f"Metricas calculadas para {len(df)} circuitos")

    return df


def analyze_national(df):
    """National-level turnout analysis."""
    logger.info("Analisis nacional de participacion...")

    results = {
        'total_primera': df['total_primera'].sum(),
        'total_ballotage': df['total_ballotage'].sum(),
        'participacion_change': df['participacion_change'].sum(),
        'participacion_change_pct': (df['participacion_change'].sum() / df['total_primera'].sum()) * 100,
        'fa_change': df['fa_change'].sum(),
        'pn_change': df['pn_change'].sum(),
        'coalition_change': df['coalition_change'].sum(),
        'fa_mobilization_pct': (df['fa_change'].sum() / df['participacion_change'].sum()) * 100,
        'pn_mobilization_pct': (df['pn_change'].sum() / df['participacion_change'].sum()) * 100,
        'avg_swing': df['fa_swing'].mean() * 100,
    }

    logger.info(f"  Cambio participacion: {results['participacion_change']:,} votos ({results['participacion_change_pct']:.1f}%)")
    logger.info(f"  Cambio FA: {results['fa_change']:,} votos")
    logger.info(f"  Cambio PN: {results['pn_change']:,} votos")
    logger.info(f"  Eficiencia movilizacion FA: {results['fa_mobilization_pct']:.1f}%")

    return pd.DataFrame([results])


def analyze_by_department(df):
    """Department-level turnout analysis."""
    logger.info("Analisis por departamento...")

    dept_col = 'departamento' if 'departamento' in df.columns else 'DEPARTAMENTO'
    if dept_col not in df.columns:
        logger.warning("No hay columna departamento - saltando analisis departamental")
        return None

    grouped = df.groupby(dept_col).agg({
        'total_primera': 'sum',
        'total_ballotage': 'sum',
        'participacion_change': 'sum',
        'fa_change': 'sum',
        'pn_change': 'sum',
        'coalition_change': 'sum',
        'fa_swing': 'mean'
    }).reset_index()

    grouped['participacion_change_pct'] = (grouped['participacion_change'] / grouped['total_primera']) * 100
    grouped['fa_mobilization_pct'] = (grouped['fa_change'] / grouped['participacion_change']) * 100
    grouped['pn_mobilization_pct'] = (grouped['pn_change'] / grouped['participacion_change']) * 100

    grouped = grouped.sort_values('participacion_change', ascending=False)

    logger.info(f"Analisis departamental completo: {len(grouped)} departamentos")

    return grouped


def analyze_by_dominant_party(df):
    """Analyze turnout by dominant party in primera vuelta."""
    logger.info("Analisis por partido dominante...")

    grouped = df.groupby('dominant_party').agg({
        'participacion_change': ['mean', 'sum', 'count'],
        'participacion_change_pct': 'mean',
        'fa_change': 'sum',
        'pn_change': 'sum',
        'fa_swing': 'mean',
        'fa_mobilization_eff': 'mean',
        'pn_mobilization_eff': 'mean'
    })

    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()

    logger.info("Analisis por partido dominante completo")

    return grouped


def correlation_analysis(df):
    """Correlation between turnout change and swing."""
    logger.info("Analisis de correlacion...")

    # Filter valid observations
    valid = ~df['fa_swing'].isna() & ~df['participacion_change_pct'].isna()
    df_valid = df[valid]

    corr = np.corrcoef(df_valid['participacion_change_pct'], df_valid['fa_swing'])[0, 1]

    logger.info(f"  Correlacion participacion vs swing FA: {corr:.3f}")

    return {
        'correlation_turnout_swing': corr,
        'n_observations': len(df_valid)
    }


def save_results(df_national, df_dept, df_dominant, corr_results, config):
    """Save analysis results."""
    logger.info("Guardando resultados...")

    output_dirs = config.get_output_dirs()
    tables_dir = Path(output_dirs['tables'])
    latex_dir = tables_dir / 'latex'

    tables_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    # 1. National summary
    csv_path = tables_dir / 'turnout_national_summary.csv'
    df_national.to_csv(csv_path, index=False, float_format='%.2f')
    logger.info(f"Nacional guardado: {csv_path}")

    # 2. Departmental
    if df_dept is not None:
        csv_path = tables_dir / 'turnout_by_department.csv'
        df_dept.to_csv(csv_path, index=False, float_format='%.2f')
        logger.info(f"Departamental guardado: {csv_path}")

        # LaTeX table (top 10)
        df_latex = df_dept.head(10)[['departamento', 'participacion_change', 'participacion_change_pct', 'fa_mobilization_pct']].copy()
        df_latex.columns = ['Departamento', 'Cambio (votos)', 'Cambio (%)', 'Movilizacion FA (%)']

        latex_table = df_latex.to_latex(
            index=False,
            escape=False,
            column_format='lrrr',
            caption='Top 10 departamentos por cambio en participacion',
            label='tab:turnout_by_dept',
            float_format='%.1f'
        )

        latex_path = latex_dir / 'turnout_top10_departments.tex'
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        logger.info(f"LaTeX guardado: {latex_path}")

    # 3. By dominant party
    csv_path = tables_dir / 'turnout_by_dominant_party.csv'
    df_dominant.to_csv(csv_path, index=False, float_format='%.4f')
    logger.info(f"Por partido dominante guardado: {csv_path}")

    # 4. Correlation results
    csv_path = tables_dir / 'turnout_correlation.csv'
    pd.DataFrame([corr_results]).to_csv(csv_path, index=False, float_format='%.4f')
    logger.info(f"Correlaciones guardadas: {csv_path}")


def generate_figures(df, df_dept, df_dominant, config):
    """Generate turnout analysis figures."""
    logger.info("Generando figuras...")

    output_dirs = config.get_output_dirs()
    figures_dir = Path(output_dirs['figures'])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Scatter turnout change vs FA swing
    fig, ax = plt.subplots(figsize=(10, 8))

    valid = ~df['fa_swing'].isna() & ~df['participacion_change_pct'].isna()
    x = df.loc[valid, 'participacion_change_pct'].values
    y = df.loc[valid, 'fa_swing'].values * 100

    ax.scatter(x, y, s=30, alpha=0.4, color=PARTY_COLORS['FA'], edgecolors='none')

    # Regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.7,
            label=f'Regresion: y = {z[0]:.2f}x + {z[1]:.2f}')

    apply_tableau_style(ax,
                       title='Cambio en Participacion vs Swing Electoral FA',
                       xlabel='Cambio en Participacion (%)',
                       ylabel='Swing FA (pp)')

    ax.legend(loc='best', frameon=True, fontsize=10)
    ax.axhline(0, color='#7F8C8D', linestyle='--', linewidth=1, alpha=0.5)
    ax.axvline(0, color='#7F8C8D', linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()
    save_publication_figure(fig, figures_dir / 'scatter_turnout_vs_fa_swing')
    plt.close()
    logger.info("Figura 1 guardada: scatter_turnout_vs_fa_swing")

    # Figure 2: Turnout change by dominant party
    fig, ax = plt.subplots(figsize=(10, 6))

    parties = df_dominant['dominant_party'].values
    turnout_change = df_dominant['participacion_change_pct_mean'].values

    colors = [PARTY_COLORS['FA'] if p == 'FA' else PARTY_COLORS['PN'] for p in parties]

    bars = ax.bar(parties, turnout_change, color=colors, alpha=0.8,
                  edgecolor='white', linewidth=2)

    # Value labels
    for bar, val in zip(bars, turnout_change):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.1f}%',
               ha='center', va='bottom', fontsize=12, fontweight='bold')

    apply_tableau_style(ax,
                       title='Cambio Promedio en Participacion por Partido Dominante',
                       xlabel='Partido Dominante (Primera Vuelta)',
                       ylabel='Cambio en Participacion (%)')

    plt.tight_layout()
    save_publication_figure(fig, figures_dir / 'turnout_by_dominant_party')
    plt.close()
    logger.info("Figura 2 guardada: turnout_by_dominant_party")

    # Figure 3: Departmental turnout change (map would be ideal, but bar chart for now)
    if df_dept is not None:
        fig, ax = plt.subplots(figsize=(12, 8))

        top15 = df_dept.head(15)
        depts = top15['departamento'].values
        changes = top15['participacion_change_pct'].values

        colors_dept = ['#27AE60' if c > 0 else '#E74C3C' for c in changes]

        bars = ax.barh(range(len(depts)), changes, color=colors_dept, alpha=0.8,
                      edgecolor='white', linewidth=1.5)

        ax.set_yticks(range(len(depts)))
        ax.set_yticklabels(depts, fontsize=10)
        ax.invert_yaxis()

        # Value labels
        for i, (bar, val) in enumerate(zip(bars, changes)):
            width = bar.get_width()
            ax.text(width, i, f'{val:+.1f}%',
                   ha='left' if val > 0 else 'right', va='center',
                   fontsize=9, fontweight='normal')

        ax.axvline(0, color='#2C3E50', linestyle='-', linewidth=1.5)

        apply_tableau_style(ax,
                           title='Cambio en Participacion por Departamento (Top 15)',
                           xlabel='Cambio en Participacion (%)',
                           ylabel='')

        plt.tight_layout()
        save_publication_figure(fig, figures_dir / 'turnout_change_by_department')
        plt.close()
        logger.info("Figura 3 guardada: turnout_change_by_department")


def main():
    config = get_config()

    logger.info("="*70)
    logger.info("ANALISIS DE DINAMICA DE PARTICIPACION")
    logger.info("="*70)

    # Load data
    df = load_data(config)

    # Calculate metrics
    df = calculate_turnout_metrics(df)

    # Analyses
    df_national = analyze_national(df)
    df_dept = analyze_by_department(df)
    df_dominant = analyze_by_dominant_party(df)
    corr_results = correlation_analysis(df)

    # Save results
    save_results(df_national, df_dept, df_dominant, corr_results, config)

    # Generate figures
    generate_figures(df, df_dept, df_dominant, config)

    logger.info("="*70)
    logger.info("ANALISIS DE PARTICIPACION COMPLETADO")
    logger.info("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
