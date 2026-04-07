"""Compare KingEI (DirichletMultinomial) outputs against eiCircles R package.

Usage:
    conda run -n ds python scripts/compare_eiCircles.py

This script:
1. Fits KingEI with DirichletMultinomial on the national 2024 dataset
2. Exports results to outputs/tables/comparison_eiCircles_python.csv
3. Exports R-compatible CSV for running eiCircles
4. Prints PSIS/LOO diagnostics

The eiCircles R script to run separately:
    install.packages("eiCircles")
    library(eiCircles)
    data <- read.csv("data/processed/circuitos_for_r.csv")
    # See eiCircles vignette for exact function call
    # Save output to outputs/tables/comparison_eiCircles_r.csv
"""
import sys
import logging
import pandas as pd
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.king_ei import KingEI
from src.diagnostics.loo import compute_loo, loo_summary

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ORIGIN_COLS = ['ca_primera', 'fa_primera', 'pc_primera', 'pn_primera', 'pi_primera', 'otros_primera']
DEST_COLS = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']


def main():
    data_path = Path('data/processed/circuitos_merged.parquet')
    output_dir = Path('outputs/tables')
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Loading data from %s", data_path)
    df = pd.read_parquet(data_path)
    logger.info("Loaded %d circuits", len(df))

    # Export R-compatible CSV (proportions for eiCircles)
    r_export = df[ORIGIN_COLS + DEST_COLS + ['total_primera', 'total_ballotage']].copy()
    for c in ORIGIN_COLS:
        r_export[f'{c}_prop'] = r_export[c] / r_export['total_primera'].replace(0, np.nan)
    r_csv_path = Path('data/processed/circuitos_for_r.csv')
    r_csv_path.parent.mkdir(parents=True, exist_ok=True)
    r_export.to_csv(r_csv_path, index=False)
    logger.info("Exported R-compatible CSV to %s", r_csv_path)

    # Fit DirichletMultinomial model (publication quality)
    logger.info("Fitting KingEI with DirichletMultinomial likelihood (4000 samples, 4 chains)...")
    model = KingEI(
        num_samples=4000,
        num_chains=4,
        num_warmup=2000,
        target_accept=0.9,
        random_seed=42,
        likelihood='dirichlet_multinomial',
    )
    model.fit(
        df,
        origin_cols=ORIGIN_COLS,
        destination_cols=DEST_COLS,
        total_origin='total_primera',
        total_destination='total_ballotage',
    )

    T = model.get_transition_matrix()
    ci = model.get_credible_intervals(prob=0.95)

    # Save Python results
    rows = []
    for i, orig in enumerate(ORIGIN_COLS):
        for j, dest in enumerate(DEST_COLS):
            rows.append({
                'origin': orig.replace('_primera', '').upper(),
                'destination': dest.replace('_ballotage', '').upper(),
                'mean': round(T[i, j], 4),
                'lower_95': round(ci['lower'][i, j], 4),
                'upper_95': round(ci['upper'][i, j], 4),
                'model': 'KingEI_DirichletMultinomial',
            })
    results_df = pd.DataFrame(rows)
    out_path = output_dir / 'comparison_eiCircles_python.csv'
    results_df.to_csv(out_path, index=False)
    logger.info("Saved Python results to %s", out_path)

    print("\n" + "=" * 60)
    print("NATIONAL TRANSITION MATRIX (DirichletMultinomial)")
    print("=" * 60)
    print(results_df.to_string(index=False))

    # LOO diagnostics (opcional — requiere log_likelihood en el trace)
    try:
        logger.info("Computing PSIS/LOO-CV...")
        loo_result = compute_loo(model)
        print("\n" + loo_summary(loo_result))
    except Exception as e:
        logger.warning("LOO no disponible: %s", e)

    print("\n" + "=" * 60)
    print("TO COMPARE WITH eiCircles IN R:")
    print("=" * 60)
    print(f"  Input CSV: {r_csv_path}")
    print("  Run in R:")
    print("    install.packages('eiCircles')")
    print("    library(eiCircles)")
    print(f"    data <- read.csv('{r_csv_path}')")
    print("    # See eiCircles vignette for call signature")
    print(f"    # Save output to {output_dir}/comparison_eiCircles_r.csv")
    print("=" * 60)

    logger.info("Done.")


if __name__ == '__main__':
    main()
