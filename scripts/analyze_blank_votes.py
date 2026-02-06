"""
Análisis de Patrones de Voto en Blanco
¿El voto blanco está correlacionado con CA (protesta) o es aleatorio?
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
    data_path = processed_dir / 'circuitos_merged.parquet'

    df = pd.read_parquet(data_path)
    logger.info(f"Datos cargados: {len(df)} circuitos")

    return df


def calculate_blank_metrics(df):
    """Calculate blank vote metrics."""
    logger.info("Calculando métricas de voto en blanco...")

    df = df.copy()

    # Blank rate in ballotage
    df['blancos_ballotage_total'] = df.get('blancos_ballotage', 0) + df.get('anulados_ballotage', 0)
    df['blank_rate'] = df['blancos_ballotage_total'] / df['total_ballotage']

    # Party shares in primera vuelta
    df['ca_share'] = df['ca_primera'] / df['total_primera']
    df['pc_share'] = df.get('pc_primera', 0) / df['total_primera']
    df['pn_share'] = df.get('pn_primera', 0) / df['total_primera']
    df['fa_share'] = df['fa_primera'] / df['total_primera']
    df['pi_share'] = df.get('pi_primera', 0) / df['total_primera']

    # Coalition share
    df['coalition_share'] = df['ca_share'] + df['pc_share'] + df['pn_share']

    logger.info(f"Métricas calculadas para {len(df)} circuitos")

    return df


def analyze_national(df):
    """National-level blank vote analysis."""
    logger.info("Análisis nacional de voto en blanco...")

    results = {
        'total_blancos': df['blancos_ballotage_total'].sum(),
        'total_votes': df['total_ballotage'].sum(),
        'blank_rate_national': df['blancos_ballotage_total'].sum() / df['total_ballotage'].sum(),
        'blank_rate_mean': df['blank_rate'].mean(),
        'blank_rate_median': df['blank_rate'].median(),
        'blank_rate_std': df['blank_rate'].std()
    }

    logger.info(f"  Votos en blanco: {results['total_blancos']:,} ({results['blank_rate_national']:.2%})")
    logger.info(f"  Tasa media por circuito: {results['blank_rate_mean']:.2%} ± {results['blank_rate_std']:.2%}")

    return pd.DataFrame([results])


def analyze_by_department(df):
    """Department-level blank vote analysis."""
    logger.info("Análisis por departamento...")

    dept_col = 'departamento' if 'departamento' in df.columns else 'DEPARTAMENTO'
    if dept_col not in df.columns:
        logger.warning("No hay columna departamento - saltando análisis departamental")
        return None

    grouped = df.groupby(dept_col).agg({
        'blancos_ballotage_total': 'sum',
        'total_ballotage': 'sum',
        'blank_rate': 'mean',
        'ca_share': 'mean',
        'coalition_share': 'mean'
    }).reset_index()

    grouped['blank_rate_dept'] = grouped['blancos_ballotage_total'] / grouped['total_ballotage']
    grouped = grouped.sort_values('blank_rate_dept', ascending=False)

    logger.info(f"Análisis departamental completo: {len(grouped)} departamentos")

    return grouped


def correlation_analysis(df):
    """Correlation between blank rate and party shares."""
    logger.info("Análisis de correlación...")

    # Filter valid observations
    valid = ~df['blank_rate'].isna()
    df_valid = df[valid]

    correlations = {
        'blank_vs_ca': np.corrcoef(df_valid['blank_rate'], df_valid['ca_share'])[0, 1],
        'blank_vs_pc': np.corrcoef(df_valid['blank_rate'], df_valid['pc_share'])[0, 1],
        'blank_vs_pn': np.corrcoef(df_valid['blank_rate'], df_valid['pn_share'])[0, 1],
        'blank_vs_fa': np.corrcoef(df_valid['blank_rate'], df_valid['fa_share'])[0, 1],
        'blank_vs_pi': np.corrcoef(df_valid['blank_rate'], df_valid['pi_share'])[0, 1],
        'blank_vs_coalition': np.corrcoef(df_valid['blank_rate'], df_valid['coalition_share'])[0, 1],
        'n_observations': len(df_valid)
    }

    logger.info(f"  Correlación blank vs CA: {correlations['blank_vs_ca']:.3f}")
    logger.info(f"  Correlación blank vs PC: {correlations['blank_vs_pc']:.3f}")
    logger.info(f"  Correlación blank vs PN: {correlations['blank_vs_pn']:.3f}")
    logger.info(f"  Correlación blank vs FA: {correlations['blank_vs_fa']:.3f}")
    logger.info(f"  Correlación blank vs Coalición: {correlations['blank_vs_coalition']:.3f}")

    return correlations


def analyze_by_ca_quartile(df):
    """Blank rate by CA share quartiles."""
    logger.info("Análisis por cuartiles de CA...")

    # Create quartiles
    df['ca_quartile'] = pd.qcut(df['ca_share'], q=4, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4 (High)'])

    grouped = df.groupby('ca_quartile').agg({
        'blank_rate': ['mean', 'median', 'std', 'count'],
        'ca_share': 'mean'
    })

    grouped.columns = ['_'.join(col).strip() for col in grouped.columns.values]
    grouped = grouped.reset_index()

    logger.info("Análisis por cuartiles completo")

    return grouped


def save_results(df_national, df_dept, correlations, df_quartile, config):
    """Save blank vote analysis results."""
    logger.info("Guardando resultados...")

    output_dirs = config.get_output_dirs()
    tables_dir = Path(output_dirs['tables'])
    latex_dir = tables_dir / 'latex'

    tables_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    # 1. National summary
    csv_path = tables_dir / 'blank_votes_national.csv'
    df_national.to_csv(csv_path, index=False, float_format='%.4f')
    logger.info(f"Nacional guardado: {csv_path}")

    # 2. Departmental
    if df_dept is not None:
        csv_path = tables_dir / 'blank_votes_by_department.csv'
        df_dept.to_csv(csv_path, index=False, float_format='%.4f')
        logger.info(f"Departamental guardado: {csv_path}")

        # LaTeX table (top 10)
        df_latex = df_dept.head(10)[['departamento', 'blank_rate_dept', 'blank_rate', 'ca_share']].copy()
        df_latex.columns = ['Departamento', 'Tasa Blank (Depto)', 'Tasa Blank (Circuitos)', 'Share CA']
        df_latex['Tasa Blank (Depto)'] = df_latex['Tasa Blank (Depto)'].apply(lambda x: f"{x:.2%}")
        df_latex['Tasa Blank (Circuitos)'] = df_latex['Tasa Blank (Circuitos)'].apply(lambda x: f"{x:.2%}")
        df_latex['Share CA'] = df_latex['Share CA'].apply(lambda x: f"{x:.2%}")

        latex_table = df_latex.to_latex(
            index=False,
            escape=False,
            column_format='lccc',
            caption='Top 10 departamentos por tasa de voto en blanco',
            label='tab:blank_by_dept'
        )

        latex_path = latex_dir / 'blank_votes_top10_departments.tex'
        with open(latex_path, 'w', encoding='utf-8') as f:
            f.write(latex_table)
        logger.info(f"LaTeX guardado: {latex_path}")

    # 3. Correlations
    csv_path = tables_dir / 'blank_votes_correlation.csv'
    pd.DataFrame([correlations]).to_csv(csv_path, index=False, float_format='%.4f')
    logger.info(f"Correlaciones guardadas: {csv_path}")

    # LaTeX correlation table
    df_corr_latex = pd.DataFrame([
        {'Partido': 'CA', 'Correlación': correlations['blank_vs_ca']},
        {'Partido': 'PC', 'Correlación': correlations['blank_vs_pc']},
        {'Partido': 'PN', 'Correlación': correlations['blank_vs_pn']},
        {'Partido': 'FA', 'Correlación': correlations['blank_vs_fa']},
        {'Partido': 'PI', 'Correlación': correlations['blank_vs_pi']},
        {'Partido': 'Coalición', 'Correlación': correlations['blank_vs_coalition']}
    ])
    df_corr_latex['Correlación'] = df_corr_latex['Correlación'].apply(lambda x: f"{x:.3f}")

    latex_table = df_corr_latex.to_latex(
        index=False,
        escape=False,
        column_format='lc',
        caption='Correlación entre voto en blanco y share de partidos (primera vuelta)',
        label='tab:blank_correlation'
    )

    latex_path = latex_dir / 'blank_votes_correlation.tex'
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)
    logger.info(f"LaTeX correlaciones guardado: {latex_path}")

    # 4. By CA quartile
    csv_path = tables_dir / 'blank_votes_by_ca_quartile.csv'
    df_quartile.to_csv(csv_path, index=False, float_format='%.4f')
    logger.info(f"Por cuartil CA guardado: {csv_path}")


def generate_figures(df, df_dept, df_quartile, config):
    """Generate blank vote analysis figures."""
    logger.info("Generando figuras...")

    output_dirs = config.get_output_dirs()
    figures_dir = Path(output_dirs['figures'])
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Scatter blank rate vs CA share
    fig, ax = plt.subplots(figsize=(10, 8))

    valid = ~df['blank_rate'].isna() & ~df['ca_share'].isna()
    x = df.loc[valid, 'ca_share'].values * 100
    y = df.loc[valid, 'blank_rate'].values * 100

    ax.scatter(x, y, s=20, alpha=0.3, color=PARTY_COLORS['CA'], edgecolors='none')

    # Regression line
    z = np.polyfit(x, y, 1)
    p = np.poly1d(z)
    x_line = np.linspace(x.min(), x.max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, alpha=0.7,
            label=f'Regresión: y = {z[0]:.3f}x + {z[1]:.2f}')

    # Correlation
    corr = np.corrcoef(x, y)[0, 1]
    ax.text(0.05, 0.95, f'Correlación: {corr:.3f}',
            transform=ax.transAxes, fontsize=12,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    apply_tableau_style(ax,
                       title='Voto en Blanco vs Share CA (Primera Vuelta)',
                       xlabel='Share CA Primera Vuelta (%)',
                       ylabel='Tasa Voto en Blanco Ballotage (%)')

    ax.legend(loc='best', frameon=True, fontsize=10)

    plt.tight_layout()
    save_publication_figure(fig, figures_dir / 'scatter_blank_rate_vs_ca_share')
    plt.close()
    logger.info("Figura 1 guardada: scatter_blank_rate_vs_ca_share")

    # Figure 2: Departmental blank rate ranking
    if df_dept is not None:
        fig, ax = plt.subplots(figsize=(12, 8))

        depts = df_dept['departamento'].values
        rates = df_dept['blank_rate_dept'].values * 100

        colors_dept = ['#E67E22' if r > df_dept['blank_rate_dept'].mean() * 100 else '#3498DB' for r in rates]

        bars = ax.barh(range(len(depts)), rates, color=colors_dept, alpha=0.8,
                      edgecolor='white', linewidth=1.5)

        ax.set_yticks(range(len(depts)))
        ax.set_yticklabels(depts, fontsize=10)
        ax.invert_yaxis()

        # Value labels
        for i, (bar, val) in enumerate(zip(bars, rates)):
            width = bar.get_width()
            ax.text(width, i, f'{val:.2f}%',
                   ha='left', va='center',
                   fontsize=9, fontweight='normal')

        # National average line
        national_avg = df_dept['blank_rate_dept'].mean() * 100
        ax.axvline(national_avg, color='#E74C3C', linestyle='--', linewidth=2, alpha=0.7,
                  label=f'Promedio nacional: {national_avg:.2f}%')

        apply_tableau_style(ax,
                           title='Tasa de Voto en Blanco por Departamento (Ballotage 2024)',
                           xlabel='Tasa de Voto en Blanco (%)',
                           ylabel='')

        ax.legend(loc='best', frameon=True, fontsize=10)

        plt.tight_layout()
        save_publication_figure(fig, figures_dir / 'blank_rate_ranking')
        plt.close()
        logger.info("Figura 2 guardada: blank_rate_ranking")

    # Figure 3: Blank rate by CA quartile
    fig, ax = plt.subplots(figsize=(10, 6))

    quartiles = df_quartile['ca_quartile'].values
    blank_means = df_quartile['blank_rate_mean'].values * 100
    blank_stds = df_quartile['blank_rate_std'].values * 100

    colors_quartile = [PARTY_COLORS['CA']] * len(quartiles)

    bars = ax.bar(range(len(quartiles)), blank_means, yerr=blank_stds,
                  color=colors_quartile, alpha=0.7, capsize=5,
                  edgecolor='white', linewidth=2)

    ax.set_xticks(range(len(quartiles)))
    ax.set_xticklabels(quartiles, fontsize=10)

    # Value labels
    for bar, val in zip(bars, blank_means):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{val:.2f}%',
               ha='center', va='bottom', fontsize=11, fontweight='bold')

    apply_tableau_style(ax,
                       title='Tasa de Voto en Blanco por Cuartil de Share CA',
                       xlabel='Cuartil de Share CA (Primera Vuelta)',
                       ylabel='Tasa de Voto en Blanco (%)')

    plt.tight_layout()
    save_publication_figure(fig, figures_dir / 'blank_rate_by_ca_quartile')
    plt.close()
    logger.info("Figura 3 guardada: blank_rate_by_ca_quartile")


def main():
    config = get_config()

    logger.info("="*70)
    logger.info("ANALISIS DE PATRONES DE VOTO EN BLANCO")
    logger.info("="*70)

    # Load data
    df = load_data(config)

    # Calculate metrics
    df = calculate_blank_metrics(df)

    # Analyses
    df_national = analyze_national(df)
    df_dept = analyze_by_department(df)
    correlations = correlation_analysis(df)
    df_quartile = analyze_by_ca_quartile(df)

    # Save results
    save_results(df_national, df_dept, correlations, df_quartile, config)

    # Generate figures
    generate_figures(df, df_dept, df_quartile, config)

    logger.info("="*70)
    logger.info("ANALISIS DE VOTO EN BLANCO COMPLETADO")
    logger.info("="*70)

    return 0


if __name__ == '__main__':
    sys.exit(main())
