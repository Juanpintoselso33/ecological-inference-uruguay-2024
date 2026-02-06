"""
Run stratified analyses for 2019.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pickle
from src.models.king_ei import KingEI
from src.utils import get_logger

logger = get_logger(__name__)


def run_analysis(name, df, samples=2000, chains=3):
    """Run King's EI analysis."""

    origin_cols = ['ca_primera', 'pc_primera', 'pn_primera', 'pi_primera', 'otros_primera']
    destination_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']

    logger.info(f"\n{'='*70}")
    logger.info(f"Analyzing: {name}")
    logger.info(f"Circuits: {len(df):,}")
    logger.info(f"{'='*70}\n")

    model = KingEI(
        num_samples=samples,
        num_chains=chains,
        num_warmup=samples,
        random_seed=42
    )

    try:
        model.fit(
            data=df,
            origin_cols=origin_cols,
            destination_cols=destination_cols,
            total_origin='total_primera',
            total_destination='total_ballotage',
            progressbar=True
        )

        T = model.get_transition_matrix()
        ci = model.get_credible_intervals(0.95)
        diag = model.get_diagnostics()

        results = {
            'name': name,
            'n_circuits': int(len(df)),
            'total_primera': float(df['total_primera'].sum()),
            'total_ballotage': float(df['total_ballotage'].sum()),
            'transition_matrix': T.tolist() if hasattr(T, 'tolist') else T,
            'credible_intervals': {
                'lower': ci['lower'].tolist() if hasattr(ci['lower'], 'tolist') else ci['lower'],
                'upper': ci['upper'].tolist() if hasattr(ci['upper'], 'tolist') else ci['upper'],
            },
            'diagnostics': {
                'rhat': diag['rhat'].tolist() if hasattr(diag['rhat'], 'tolist') else diag['rhat'],
                'ess': diag['ess'].tolist() if hasattr(diag['ess'], 'tolist') else diag['ess'],
            },
            'origin_cols': origin_cols,
            'destination_cols': destination_cols,
        }

        for party in origin_cols:
            results[f'{party}_votes'] = float(df[party].sum())

        logger.info(f"\nResults for {name}:")

        # Handle rhat and ess - they can be arrays or lists
        import numpy as np
        rhat_vals = diag['rhat']
        ess_vals = diag['ess']

        if isinstance(rhat_vals, np.ndarray):
            max_rhat = np.max(rhat_vals)
        else:
            max_rhat = max(max(r) if isinstance(r, list) else r for r in rhat_vals)

        if isinstance(ess_vals, np.ndarray):
            min_ess = np.min(ess_vals)
        else:
            min_ess = min(min(e) if isinstance(e, list) else e for e in ess_vals)

        logger.info(f"  Max R-hat: {max_rhat:.4f}")
        logger.info(f"  Min ESS: {min_ess:.0f}")
        logger.info(f"\n  Transition Matrix (to FA | to PN | to blancos):")

        for i, party in enumerate(origin_cols):
            party_label = party.replace('_primera', '').upper()
            logger.info(f"    {party_label:6s}: {T[i,0]*100:5.1f}% | {T[i,1]*100:5.1f}% | {T[i,2]*100:5.1f}%")

        return results

    except Exception as e:
        logger.error(f"Failed to fit {name}: {e}")
        import traceback
        traceback.print_exc()
        return None


def create_stratified_table(strat_results):
    """Create CSV table from stratified results."""

    rows = []

    for stratum_name, result in strat_results.items():
        T = result['transition_matrix']
        ci = result['credible_intervals']

        row = {
            'stratum': stratum_name,
            'n_circuits': result['n_circuits'],

            # CA
            'ca_to_fa': T[0][0],
            'ca_to_pn': T[0][1],
            'ca_to_fa_lower': ci['lower'][0][0],
            'ca_to_fa_upper': ci['upper'][0][0],
            'ca_votes': result['ca_primera_votes'],

            # PC
            'pc_to_fa': T[1][0],
            'pc_to_pn': T[1][1],
            'pc_votes': result['pc_primera_votes'],

            # PN
            'pn_to_fa': T[2][0],
            'pn_to_pn': T[2][1],
            'pn_votes': result['pn_primera_votes'],

            # PI
            'pi_to_fa': T[3][0],
            'pi_to_pn': T[3][1],
            'pi_votes': result['pi_primera_votes'],
        }

        rows.append(row)

    return pd.DataFrame(rows)


if __name__ == '__main__':
    # Load data
    logger.info("Loading 2019 data...")
    df = pd.read_parquet('data/processed/circuitos_merged_2019.parquet')

    logger.info(f"\nDataset: {len(df):,} circuits")
    logger.info(f"Departments: {df['departamento'].unique()[:5]}...")

    # Check for Montevideo
    montevideo_variants = ['MONTEVIDEO', 'MO', 'Montevideo']
    montevideo_name = None
    for variant in montevideo_variants:
        if variant in df['departamento'].values:
            montevideo_name = variant
            break

    logger.info(f"\nMontevideo department name: {montevideo_name}")

    # ========================================================================
    # URBAN VS RURAL (Montevideo vs Rest)
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: MONTEVIDEO VS INTERIOR")
    logger.info("="*70)

    if montevideo_name:
        df_montevideo = df[df['departamento'] == montevideo_name]
        df_interior = df[df['departamento'] != montevideo_name]
    else:
        # Fallback: use 'MO' if exact match not found
        df_montevideo = df[df['departamento'] == 'MO']
        df_interior = df[df['departamento'] != 'MO']

    urban_rural_results = {}

    # Montevideo
    result_mvd = run_analysis('MONTEVIDEO', df_montevideo, samples=2000, chains=3)
    if result_mvd:
        urban_rural_results['MONTEVIDEO'] = result_mvd

    # Interior
    result_int = run_analysis('INTERIOR', df_interior, samples=2000, chains=3)
    if result_int:
        urban_rural_results['INTERIOR'] = result_int

    if urban_rural_results:
        with open('outputs/results/urban_rural_transfers_2019.pkl', 'wb') as f:
            pickle.dump(urban_rural_results, f)
        logger.info(f"\nSaved: outputs/results/urban_rural_transfers_2019.pkl")

        urban_rural_table = create_stratified_table(urban_rural_results)
        urban_rural_table.to_csv('outputs/tables/urban_rural_2019.csv', index=False)
        logger.info(f"Saved: outputs/tables/urban_rural_2019.csv")

    # ========================================================================
    # METROPOLITAN VS INTERIOR (Montevideo+Canelones vs Rest)
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: METROPOLITANA VS INTERIOR")
    logger.info("="*70)

    metropolitan_depts = ['MO', 'CA', 'MONTEVIDEO', 'CANELONES']
    df_metro = df[df['departamento'].isin(metropolitan_depts)]
    df_interior2 = df[~df['departamento'].isin(metropolitan_depts)]

    region_results = {}

    # Metropolitan
    result_metro = run_analysis('METROPOLITANA', df_metro, samples=2000, chains=3)
    if result_metro:
        region_results['METROPOLITANA'] = result_metro

    # Interior
    result_int2 = run_analysis('INTERIOR', df_interior2, samples=2000, chains=3)
    if result_int2:
        region_results['INTERIOR'] = result_int2

    if region_results:
        with open('outputs/results/region_transfers_2019.pkl', 'wb') as f:
            pickle.dump(region_results, f)
        logger.info(f"\nSaved: outputs/results/region_transfers_2019.pkl")

        region_table = create_stratified_table(region_results)
        region_table.to_csv('outputs/tables/region_2019.csv', index=False)
        logger.info(f"Saved: outputs/tables/region_2019.csv")

    logger.info("\n" + "="*70)
    logger.info("STRATIFIED ANALYSIS COMPLETE")
    logger.info("="*70)
