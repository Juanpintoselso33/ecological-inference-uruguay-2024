"""
Análisis nacional con covariables en la concentración del prior Dirichlet.

Implementa el modelo extendido de issue #13:
    log(phi_i) = alpha + beta_turnout * turnout_i + beta_logsize * log(size_i)
                       + beta_ca_share * ca_share_i + beta_urban * urban_i

Las covariables modulan la concentración del prior Dirichlet-Multinomial
por circuito, capturando heterogeneidad sistemática en la dispersión de
las transferencias de voto.

Uso:
    conda run -n ei2024 python scripts/national_analysis_with_covariates.py
"""
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.king_ei import KingEI
from src.utils.config import get_config

logging.basicConfig(level=logging.INFO,
                    format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ORIGIN_COLS = ['ca_primera', 'fa_primera', 'pc_primera', 'pn_primera',
               'pi_primera', 'otros_primera']
DEST_COLS   = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']


def build_covariates(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Construye y estandariza las covariables del issue #13.

    Covariables:
    - turnout_c   : participación ballotage (centrada)
    - logsize_c   : log(total_primera) (centrado)
    - ca_share_c  : proporción CA primera vuelta (centrada)
    - urban       : 1=urbano, 0=rural (estandarizado)
    """
    df = df.copy()

    # Participación ballotage
    df['participacion_ballotage'] = (
        df['total_ballotage'] / df['total_primera'].replace(0, np.nan)
    )

    # Proporción CA primera vuelta
    df['ca_share'] = df['ca_primera'] / df['total_primera'].replace(0, np.nan)

    # Log tamaño del circuito
    df['log_total'] = np.log(df['total_primera'].clip(lower=1))

    # Clasificación urbana (1=urbano) — tomar de columna existente si hay
    if 'clasificacion_urbana' in df.columns:
        df['urban'] = (df['clasificacion_urbana'] == 'urbano').astype(float)
    elif 'urbano_rural' in df.columns:
        df['urban'] = (df['urbano_rural'] == 'urbano').astype(float)
    else:
        logger.warning("No se encontró columna de clasificación urbana — se omite covariable urbana")
        df['urban'] = 0.0

    # Centrar/estandarizar (el modelo tiene su propio alpha_concentration,
    # pero centrar mejora la convergencia)
    cov_cols = []
    for col, name in [
        ('participacion_ballotage', 'turnout_c'),
        ('log_total',               'logsize_c'),
        ('ca_share',                'ca_share_c'),
    ]:
        mu  = df[col].mean(skipna=True)
        std = df[col].std(skipna=True).clip(lower=1e-6)
        df[name] = (df[col] - mu) / std
        cov_cols.append(name)

    # urban no se estandariza (ya es 0/1 y tiene interpretación directa)
    cov_cols.append('urban')

    logger.info("Covariables construidas: %s", cov_cols)
    logger.info("  turnout media=%.3f std=%.3f",
                df['participacion_ballotage'].mean(), df['participacion_ballotage'].std())
    logger.info("  log(size) media=%.3f std=%.3f",
                df['log_total'].mean(), df['log_total'].std())
    logger.info("  CA share media=%.3f std=%.3f",
                df['ca_share'].mean(), df['ca_share'].std())
    logger.info("  Proporción urbana=%.3f", df['urban'].mean())

    return df, cov_cols


def main():
    config = get_config()
    data_dirs = config.get_data_dirs()
    output_dirs = config.get_output_dirs()

    data_path = Path(data_dirs['processed']) / 'circuitos_merged.parquet'
    results_dir = Path(output_dirs['tables']).parent / 'results'
    tables_dir  = Path(output_dirs['tables'])
    latex_dir   = tables_dir / 'latex'

    logger.info("Cargando datos: %s", data_path)
    df = pd.read_parquet(data_path)
    logger.info("  %d circuitos cargados", len(df))

    # Construir covariables
    df, cov_cols = build_covariates(df)

    # Filtrar circuitos válidos
    min_votes = 30
    valid = (df['total_primera'] >= min_votes) & (df['total_ballotage'] >= min_votes)
    df_clean = df[valid].copy()
    logger.info("  %d circuitos válidos (mín %d votos)", len(df_clean), min_votes)

    # ── Modelo sin covariables (baseline) ────────────────────────────────────
    logger.info("="*60)
    logger.info("MODELO BASELINE (DirichletMultinomial, sin covariables)")
    logger.info("="*60)

    model_base = KingEI(
        num_samples=4000,
        num_chains=4,
        num_warmup=2000,
        target_accept=0.95,
        likelihood='dirichlet_multinomial',
        prior_alpha=1.0,
        random_seed=42,
        trace_dir='outputs/results/traces',
    )
    model_base.fit(
        data=df_clean,
        origin_cols=ORIGIN_COLS,
        destination_cols=DEST_COLS,
        total_origin='total_primera',
        total_destination='total_ballotage',
        progressbar=True,
    )

    T_base = model_base.get_transition_matrix()
    ci_base = model_base.get_credible_intervals(prob=0.95)
    diag_base = model_base.get_diagnostics()

    logger.info("Baseline CA→FA: %.1f%% [%.1f%%, %.1f%%]",
                T_base[0,0]*100, ci_base['lower'][0,0]*100, ci_base['upper'][0,0]*100)
    logger.info("Baseline R-hat max: %.4f | ESS min: %.0f",
                np.max(diag_base['rhat']), np.min(diag_base['ess']))

    # ── Modelo con covariables ────────────────────────────────────────────────
    logger.info("="*60)
    logger.info("MODELO CON COVARIABLES (turnout, log_size, ca_share, urban)")
    logger.info("="*60)

    model_cov = KingEI(
        num_samples=4000,
        num_chains=4,
        num_warmup=2000,
        target_accept=0.95,
        likelihood='dirichlet_multinomial',
        prior_alpha=1.0,
        random_seed=42,
        trace_dir='outputs/results/traces',
    )
    model_cov.fit(
        data=df_clean,
        origin_cols=ORIGIN_COLS,
        destination_cols=DEST_COLS,
        total_origin='total_primera',
        total_destination='total_ballotage',
        covariate_cols=cov_cols,
        progressbar=True,
    )

    T_cov = model_cov.get_transition_matrix()
    ci_cov = model_cov.get_credible_intervals(prob=0.95)
    diag_cov = model_cov.get_diagnostics()

    logger.info("Covariate CA→FA: %.1f%% [%.1f%%, %.1f%%]",
                T_cov[0,0]*100, ci_cov['lower'][0,0]*100, ci_cov['upper'][0,0]*100)
    logger.info("Covariate R-hat max: %.4f | ESS min: %.0f",
                np.max(diag_cov['rhat']), np.min(diag_cov['ess']))

    # ── Comparación y guardado ────────────────────────────────────────────────
    origin_names = [c.replace('_primera','').upper() for c in ORIGIN_COLS]
    dest_names   = [c.replace('_ballotage','').upper() for c in DEST_COLS]

    rows = []
    for i, orig in enumerate(origin_names):
        for j, dest in enumerate(dest_names):
            rows.append({
                'Origen': orig,
                'Destino': dest,
                'Baseline_mean':  T_base[i,j],
                'Baseline_lower': ci_base['lower'][i,j],
                'Baseline_upper': ci_base['upper'][i,j],
                'Covariate_mean':  T_cov[i,j],
                'Covariate_lower': ci_cov['lower'][i,j],
                'Covariate_upper': ci_cov['upper'][i,j],
                'CI_width_base': ci_base['upper'][i,j] - ci_base['lower'][i,j],
                'CI_width_cov':  ci_cov['upper'][i,j]  - ci_cov['lower'][i,j],
            })

    df_comp = pd.DataFrame(rows)
    df_comp['CI_width_reduction_pct'] = (
        (df_comp['CI_width_base'] - df_comp['CI_width_cov'])
        / df_comp['CI_width_base'] * 100
    )

    csv_path = tables_dir / 'covariate_comparison.csv'
    df_comp.to_csv(csv_path, index=False, float_format='%.4f')
    logger.info("Comparación guardada: %s", csv_path)

    # LaTeX table
    df_latex = df_comp[['Origen','Destino',
                         'Baseline_mean','Baseline_lower','Baseline_upper',
                         'Covariate_mean','Covariate_lower','Covariate_upper',
                         'CI_width_reduction_pct']].copy()

    def fmt_ci(row, prefix):
        m  = row[f'{prefix}_mean'] * 100
        lo = row[f'{prefix}_lower'] * 100
        hi = row[f'{prefix}_upper'] * 100
        return f"{m:.1f}\\% [{lo:.1f}, {hi:.1f}]"

    df_latex['Baseline'] = df_latex.apply(lambda r: fmt_ci(r,'Baseline'), axis=1)
    df_latex['Covariables'] = df_latex.apply(lambda r: fmt_ci(r,'Covariate'), axis=1)
    df_latex['Red. IC (\\%)'] = df_latex['CI_width_reduction_pct'].apply(lambda x: f"{x:.1f}\\%")

    latex_out = df_latex[['Origen','Destino','Baseline','Covariables','Red. IC (\\%)']].to_latex(
        index=False, escape=False,
        column_format='llccc',
        caption='Comparación modelo baseline vs.~modelo con covariables (turnout, tamaño, proporción CA, urbano/rural). Se reporta media posterior e IC 95\\%.',
        label='tab:covariate_comparison',
    )
    latex_path = latex_dir / 'covariate_comparison.tex'
    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_out)
    logger.info("LaTeX guardado: %s", latex_path)

    # Summary
    avg_reduction = df_comp['CI_width_reduction_pct'].mean()
    logger.info("="*60)
    logger.info("RESUMEN: reducción media de anchura de IC = %.1f%%", avg_reduction)
    logger.info("="*60)
    logger.info("Nota: la literatura reporta reducciones de ~44%% con covariables relevantes.")
    logger.info("Si la reducción observada es menor, indica que las covariables disponibles")
    logger.info("capturan poca varianza adicional sobre el likelihood con 7,271 observaciones.")


if __name__ == '__main__':
    main()
