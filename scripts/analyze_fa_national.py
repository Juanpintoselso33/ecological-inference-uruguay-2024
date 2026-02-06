"""
Análisis Nacional - Comportamiento Electoral del Frente Amplio 2024

Estima las tasas de transferencia de votos desde FA en primera vuelta hacia:
- FA ballotage (retención)
- PN ballotage (defección)
- Blancos/Anulados

Utiliza King's Ecological Inference con PyMC v5.
"""

import sys
from pathlib import Path
import argparse
import pickle
import warnings

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
from src.models.king_ei import KingEI
from src.utils.config import get_config
from src.utils.logger import get_logger
from src.utils.validators import validate_vote_counts

warnings.filterwarnings('ignore')
logger = get_logger(__name__)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Análisis nacional de transferencias FA 2024'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=4000,
        help='Número de muestras MCMC (default: 4000)'
    )
    parser.add_argument(
        '--chains',
        type=int,
        default=4,
        help='Número de cadenas MCMC (default: 4)'
    )
    parser.add_argument(
        '--warmup',
        type=int,
        default=2000,
        help='Número de muestras de warmup (default: 2000)'
    )
    return parser.parse_args()


def load_data(config):
    """Load processed electoral data."""
    logger.info("Cargando datos procesados...")

    data_dirs = config.get_data_dirs()
    data_path = Path(data_dirs['processed']) / 'circuitos_merged.parquet'
    if not data_path.exists():
        raise FileNotFoundError(f"Archivo no encontrado: {data_path}")

    df = pd.read_parquet(data_path)
    logger.info(f"Datos cargados: {len(df)} circuitos, {len(df.columns)} columnas")

    return df


def prepare_fa_data(df):
    """
    Prepara datos para análisis FA.

    Origin: FA primera vuelta + OTROS primera vuelta
    Destination: FA ballotage, PN ballotage, Blancos+Anulados
    """
    logger.info("Preparando datos para análisis FA...")

    # Verificar columnas requeridas
    required_cols = [
        'fa_primera', 'otros_primera', 'total_primera',
        'fa_ballotage', 'pn_ballotage', 'blancos_ballotage',
        'anulados_ballotage', 'total_ballotage'
    ]

    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Columnas faltantes: {missing}")

    # Crear columna combinada de blancos+anulados
    df['blancos_anulados_ballotage'] = (
        df['blancos_ballotage'] + df['anulados_ballotage']
    )

    # Recalcular otros_primera como complemento de FA
    # (incluye PN + PC + CA + PI + otros originales)
    df['otros_primera_recalc'] = df['total_primera'] - df['fa_primera']

    # Definir matrices origin/destination
    origin_cols = ['fa_primera', 'otros_primera_recalc']
    destination_cols = [
        'fa_ballotage',
        'pn_ballotage',
        'blancos_anulados_ballotage'
    ]

    # Validar conteos
    logger.info("Validando conteos de votos...")

    # Validar primera vuelta (debe sumar 100% por construcción)
    primera_sum = df[origin_cols].sum(axis=1)
    primera_diff = np.abs(primera_sum - df['total_primera'])
    primera_valid = primera_diff < 1  # menos de 1 voto de diferencia

    # Validar ballotage
    ballotage_sum = df[destination_cols].sum(axis=1)
    ballotage_diff = np.abs(ballotage_sum - df['total_ballotage'])
    ballotage_valid = (ballotage_diff / df['total_ballotage']) < 0.02

    # Filtrar circuitos válidos
    valid = primera_valid & ballotage_valid
    df_clean = df[valid].copy()

    logger.info(f"Circuitos válidos: {len(df_clean)} de {len(df)} ({100*len(df_clean)/len(df):.1f}%)")

    # Filtrar circuitos con participación muy baja (< 50 votos)
    min_votes = 50
    low_turnout = (df_clean['total_primera'] < min_votes) | (df_clean['total_ballotage'] < min_votes)
    df_clean = df_clean[~low_turnout]

    logger.info(f"Después de filtrar baja participación: {len(df_clean)} circuitos")

    return df_clean, origin_cols, destination_cols


def fit_king_ei(df, origin_cols, destination_cols, args):
    """
    Ajusta modelo King's EI para transferencias FA.
    """
    logger.info("Ajustando modelo King's Ecological Inference...")
    logger.info(f"Configuración MCMC: {args.samples} samples, {args.chains} chains, {args.warmup} warmup")

    # Inicializar modelo
    model = KingEI(
        num_samples=args.samples,
        num_chains=args.chains,
        num_warmup=args.warmup
    )

    # Ajustar modelo
    logger.info("Iniciando muestreo MCMC (esto tomará ~3 horas)...")

    model.fit(
        data=df,
        origin_cols=origin_cols,
        destination_cols=destination_cols,
        total_origin='total_primera',
        total_destination='total_ballotage'
    )

    logger.info("✓ Muestreo MCMC completado")

    return model


def extract_results(model, origin_cols, destination_cols):
    """
    Extrae resultados del modelo ajustado.
    """
    logger.info("Extrayendo resultados del modelo...")

    # Matriz de transición (medias posteriores)
    T_matrix = model.get_transition_matrix()

    # Intervalos de credibilidad al 95%
    ci_95 = model.get_credible_intervals(prob=0.95)

    # Diagnósticos MCMC
    diagnostics = model.get_diagnostics()

    # Crear DataFrame con resultados
    results = []

    for i, origin in enumerate(origin_cols):
        for j, dest in enumerate(destination_cols):
            # Limpiar nombres
            origin_name = origin.replace('_primera', '').upper()
            dest_name = dest.replace('_ballotage', '').replace('_anulados', '').upper()

            transition = f"{origin_name}→{dest_name}"

            results.append({
                'Transition': transition,
                'Origin': origin_name,
                'Destination': dest_name,
                'Mean': T_matrix[i, j],
                'CI_Lower': ci_95['lower'][i, j],
                'CI_Upper': ci_95['upper'][i, j],
                'CI_Width': ci_95['upper'][i, j] - ci_95['lower'][i, j],
                'Rhat': diagnostics['rhat'].flatten()[i*len(destination_cols)+j] if len(diagnostics['rhat'].shape) > 1 else diagnostics['rhat'][i*len(destination_cols)+j],
                'ESS_Bulk': diagnostics['ess'].flatten()[i*len(destination_cols)+j] if len(diagnostics['ess'].shape) > 1 else diagnostics['ess'][i*len(destination_cols)+j],
                'ESS_Tail': np.nan  # ESS_Tail not available in current diagnostics
            })

    df_results = pd.DataFrame(results)

    return df_results, diagnostics


def validate_results(df_results):
    """
    Valida resultados del modelo.
    """
    logger.info("Validando resultados del modelo...")

    # 1. Verificar convergencia MCMC
    rhat_max = df_results['Rhat'].max()
    ess_bulk_min = df_results['ESS_Bulk'].min()

    logger.info(f"  R-hat máximo: {rhat_max:.4f} (criterio: < 1.01)")
    logger.info(f"  ESS bulk mínimo: {ess_bulk_min:.0f} (criterio: > 1000)")

    if rhat_max > 1.01:
        logger.warning(f"⚠ R-hat excede 1.01 (valor: {rhat_max:.4f})")
    else:
        logger.info("  ✓ R-hat OK")

    if ess_bulk_min < 1000:
        logger.warning(f"⚠ ESS bulk < 1000 (valor: {ess_bulk_min:.0f})")
    else:
        logger.info("  ✓ ESS OK")

    # 2. Verificar que transiciones suman ~100% por origen
    fa_transitions = df_results[df_results['Origin'] == 'FA']
    fa_sum = fa_transitions['Mean'].sum()

    logger.info(f"  Suma transiciones FA: {fa_sum:.1%} (esperado: ~100%)")

    if abs(fa_sum - 1.0) > 0.01:
        logger.warning(f"⚠ Suma de transiciones FA difiere de 100%: {fa_sum:.1%}")
    else:
        logger.info("  ✓ Suma de transiciones OK")

    # 3. Verificar plausibilidad de retención FA
    fa_retention = df_results[
        (df_results['Origin'] == 'FA') &
        (df_results['Destination'] == 'FA')
    ]['Mean'].values[0]

    logger.info(f"  Retención FA: {fa_retention:.1%} (esperado: > 85%)")

    if fa_retention < 0.85:
        logger.warning(f"⚠ Retención FA inesperadamente baja: {fa_retention:.1%}")
    else:
        logger.info("  ✓ Retención FA plausible")

    # 4. Verificar que defección FA→PN es baja
    fa_defection = df_results[
        (df_results['Origin'] == 'FA') &
        (df_results['Destination'] == 'PN')
    ]['Mean'].values[0]

    logger.info(f"  Defección FA→PN: {fa_defection:.1%} (esperado: < 10%)")

    if fa_defection > 0.10:
        logger.warning(f"⚠ Defección FA→PN inesperadamente alta: {fa_defection:.1%}")
    else:
        logger.info("  ✓ Defección FA→PN plausible")

    logger.info("✓ Validación completada")


def save_results(df_results, model, config):
    """
    Guarda resultados a disco.
    """
    logger.info("Guardando resultados...")

    # Crear directorios si no existen
    output_dirs = config.get_output_dirs()
    results_dir = Path(config.project_root_path) / 'outputs' / 'results'
    tables_dir = Path(output_dirs['tables'])
    latex_dir = tables_dir / 'latex'

    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    latex_dir.mkdir(parents=True, exist_ok=True)

    # 1. Guardar trace completo (para análisis posteriores)
    trace_path = results_dir / 'fa_national_transitions_2024.pkl'
    with open(trace_path, 'wb') as f:
        pickle.dump(model.trace_, f)
    logger.info(f"  ✓ Trace guardado: {trace_path}")

    # 2. Guardar tabla CSV
    csv_path = tables_dir / 'fa_national_matrix_2024.csv'
    df_results.to_csv(csv_path, index=False, float_format='%.4f')
    logger.info(f"  ✓ Tabla CSV guardada: {csv_path}")

    # 3. Guardar tabla LaTeX
    latex_path = latex_dir / 'fa_national_matrix_2024.tex'

    # Filtrar solo transiciones FA (no OTROS)
    df_fa_only = df_results[df_results['Origin'] == 'FA'].copy()

    # Formatear para LaTeX
    df_fa_only['CI'] = df_fa_only.apply(
        lambda row: f"[{row['CI_Lower']:.1%}, {row['CI_Upper']:.1%}]",
        axis=1
    )

    df_latex = df_fa_only[['Transition', 'Mean', 'CI']].copy()
    df_latex['Mean'] = df_latex['Mean'].apply(lambda x: f"{x:.1%}")

    latex_table = df_latex.to_latex(
        index=False,
        escape=False,
        column_format='lcc',
        caption='Matriz de transferencias nacional - Frente Amplio 2024',
        label='tab:fa_national_2024'
    )

    with open(latex_path, 'w', encoding='utf-8') as f:
        f.write(latex_table)

    logger.info(f"  ✓ Tabla LaTeX guardada: {latex_path}")

    logger.info("✓ Todos los resultados guardados exitosamente")


def print_summary(df_results):
    """
    Imprime resumen de resultados en consola.
    """
    print("\n" + "="*70)
    print("RESULTADOS - TRANSFERENCIAS FRENTE AMPLIO 2024")
    print("="*70 + "\n")

    # Filtrar solo transiciones FA
    fa_trans = df_results[df_results['Origin'] == 'FA']

    for _, row in fa_trans.iterrows():
        print(f"{row['Transition']:20s} {row['Mean']:6.1%}  [{row['CI_Lower']:6.1%}, {row['CI_Upper']:6.1%}]")

    print("\n" + "="*70)
    print("DIAGNÓSTICOS MCMC")
    print("="*70 + "\n")

    print(f"R-hat máximo:    {df_results['Rhat'].max():.4f}")
    print(f"R-hat promedio:  {df_results['Rhat'].mean():.4f}")
    print(f"ESS bulk mínimo: {df_results['ESS_Bulk'].min():.0f}")
    print(f"ESS bulk medio:  {df_results['ESS_Bulk'].mean():.0f}")
    print("\n" + "="*70 + "\n")


def main():
    """Main execution function."""
    args = parse_args()
    config = get_config()

    logger.info("="*70)
    logger.info("ANÁLISIS NACIONAL - FRENTE AMPLIO 2024")
    logger.info("="*70)

    try:
        # 1. Cargar datos
        df = load_data(config)

        # 2. Preparar datos FA
        df_clean, origin_cols, destination_cols = prepare_fa_data(df)

        # 3. Ajustar modelo King's EI
        model = fit_king_ei(df_clean, origin_cols, destination_cols, args)

        # 4. Extraer resultados
        df_results, diagnostics = extract_results(model, origin_cols, destination_cols)

        # 5. Validar resultados
        validate_results(df_results)

        # 6. Guardar resultados
        save_results(df_results, model, config)

        # 7. Imprimir resumen
        print_summary(df_results)

        logger.info("✓ Análisis completado exitosamente")
        return 0

    except Exception as e:
        logger.error(f"✗ Error durante análisis: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
