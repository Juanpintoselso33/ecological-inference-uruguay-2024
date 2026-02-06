"""
Analyze 2019 electoral transfers using King's Ecological Inference.

Comprehensive analysis:
- National level
- By department (19)
- Stratified: Urban vs Rural
- Stratified: Montevideo vs Interior

All parties: CA, PC, PN, PI, OTROS → FA/PN
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import pickle
from src.models.king_ei import KingEI
from src.utils import get_logger

logger = get_logger(__name__)


def run_analysis(name, df, origin_cols, destination_cols, samples=4000, chains=4):
    """
    Run King's EI analysis for a given dataset.

    Parameters
    ----------
    name : str
        Analysis name (e.g., "NACIONAL", "MONTEVIDEO")
    df : pd.DataFrame
        Circuit-level data
    origin_cols : list
        Origin party columns (primera vuelta)
    destination_cols : list
        Destination columns (ballotage)
    samples : int
        MCMC samples per chain
    chains : int
        Number of MCMC chains

    Returns
    -------
    dict
        Results with transition matrix, credible intervals, diagnostics
    """

    if len(df) < 20:
        logger.warning(f"Skipping {name}: too few circuits ({len(df)})")
        return None

    logger.info(f"\n{'='*70}")
    logger.info(f"Analyzing: {name}")
    logger.info(f"Circuits: {len(df):,}")
    logger.info(f"Total votes primera: {df['total_primera'].sum():,.0f}")
    logger.info(f"Total votes ballotage: {df['total_ballotage'].sum():,.0f}")
    logger.info(f"{'='*70}\n")

    # Create model
    model = KingEI(
        num_samples=samples,
        num_chains=chains,
        num_warmup=samples,
        random_seed=42
    )

    try:
        # Fit model
        model.fit(
            data=df,
            origin_cols=origin_cols,
            destination_cols=destination_cols,
            total_origin='total_primera',
            total_destination='total_ballotage',
            progressbar=True
        )

        # Extract results
        T = model.get_transition_matrix()
        ci = model.get_credible_intervals(0.95)
        diag = model.get_diagnostics()

        # Build results dictionary (convert numpy arrays to plain Python types for serialization)
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
            # model object is not serializable - omit it
        }

        # Add party vote totals (convert to float for serialization)
        for party in origin_cols:
            results[f'{party}_votes'] = float(df[party].sum())

        # Print summary
        logger.info(f"\nResults for {name}:")
        logger.info(f"  Max R-hat: {np.max(diag['rhat']):.4f}")
        logger.info(f"  Min ESS: {np.min(diag['ess']):.0f}")
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


def analyze_national(df, samples=4000, chains=4):
    """National level analysis."""

    origin_cols = ['ca_primera', 'pc_primera', 'pn_primera', 'pi_primera', 'otros_primera']
    destination_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']

    return run_analysis(
        name='NACIONAL',
        df=df,
        origin_cols=origin_cols,
        destination_cols=destination_cols,
        samples=samples,
        chains=chains
    )


def analyze_departments(df, samples=1000, chains=2):
    """Analyze each of the 19 departments."""

    origin_cols = ['ca_primera', 'pc_primera', 'pn_primera', 'pi_primera', 'otros_primera']
    destination_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']

    departments = sorted(df['departamento'].unique())
    results = {}

    for i, dept in enumerate(departments, 1):
        logger.info(f"\n[{i}/{len(departments)}] Processing department: {dept}")

        df_dept = df[df['departamento'] == dept]
        result = run_analysis(
            name=dept,
            df=df_dept,
            origin_cols=origin_cols,
            destination_cols=destination_cols,
            samples=samples,
            chains=chains
        )

        if result:
            results[dept] = result

    return results


def analyze_urban_rural(df, samples=2000, chains=3):
    """Stratified analysis: Urban vs Rural."""

    origin_cols = ['ca_primera', 'pc_primera', 'pn_primera', 'pi_primera', 'otros_primera']
    destination_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']

    # Define urban departments (montevideo + capitales departamentales)
    # For simplicity, classify Montevideo as urban, rest as mixed/rural
    # This is a proxy - ideally we'd have circuit-level urban/rural classification

    df_montevideo = df[df['departamento'] == 'MONTEVIDEO']
    df_interior = df[df['departamento'] != 'MONTEVIDEO']

    results = {}

    # Montevideo (predominantly urban)
    result_urban = run_analysis(
        name='MONTEVIDEO (urban proxy)',
        df=df_montevideo,
        origin_cols=origin_cols,
        destination_cols=destination_cols,
        samples=samples,
        chains=chains
    )
    if result_urban:
        results['MONTEVIDEO'] = result_urban

    # Interior (mixed urban/rural)
    result_interior = run_analysis(
        name='INTERIOR (mixed)',
        df=df_interior,
        origin_cols=origin_cols,
        destination_cols=destination_cols,
        samples=samples,
        chains=chains
    )
    if result_interior:
        results['INTERIOR'] = result_interior

    return results


def analyze_by_region(df, samples=2000, chains=3):
    """Stratified analysis: Metropolitan vs Interior."""

    origin_cols = ['ca_primera', 'pc_primera', 'pn_primera', 'pi_primera', 'otros_primera']
    destination_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']

    # Metropolitan region: Montevideo + Canelones
    metropolitan_depts = ['MONTEVIDEO', 'CANELONES']

    df_metro = df[df['departamento'].isin(metropolitan_depts)]
    df_interior = df[~df['departamento'].isin(metropolitan_depts)]

    results = {}

    # Metropolitan
    result_metro = run_analysis(
        name='METROPOLITANA',
        df=df_metro,
        origin_cols=origin_cols,
        destination_cols=destination_cols,
        samples=samples,
        chains=chains
    )
    if result_metro:
        results['METROPOLITANA'] = result_metro

    # Interior
    result_interior = run_analysis(
        name='INTERIOR',
        df=df_interior,
        origin_cols=origin_cols,
        destination_cols=destination_cols,
        samples=samples,
        chains=chains
    )
    if result_interior:
        results['INTERIOR'] = result_interior

    return results


def create_department_summary_table(dept_results):
    """Create CSV table from department results."""

    rows = []

    for dept_name, result in dept_results.items():
        T = result['transition_matrix']
        ci = result['credible_intervals']
        diag = result['diagnostics']

        # Party indices: CA=0, PC=1, PN=2, PI=3, OTROS=4
        # Destinations: FA=0, PN=1, BLANCOS=2

        row = {
            'departamento': dept_name,
            'n_circuits': result['n_circuits'],

            # CA transfers
            'ca_to_fa': T[0][0],
            'ca_to_pn': T[0][1],
            'ca_to_blancos': T[0][2],
            'ca_to_fa_lower': ci['lower'][0][0],
            'ca_to_fa_upper': ci['upper'][0][0],
            'ca_votes': result['ca_primera_votes'],

            # PC transfers
            'pc_to_fa': T[1][0],
            'pc_to_pn': T[1][1],
            'pc_to_blancos': T[1][2],
            'pc_to_fa_lower': ci['lower'][1][0],
            'pc_to_fa_upper': ci['upper'][1][0],
            'pc_votes': result['pc_primera_votes'],

            # PN transfers
            'pn_to_fa': T[2][0],
            'pn_to_pn': T[2][1],
            'pn_to_blancos': T[2][2],
            'pn_to_fa_lower': ci['lower'][2][0],
            'pn_to_fa_upper': ci['upper'][2][0],
            'pn_votes': result['pn_primera_votes'],

            # PI transfers
            'pi_to_fa': T[3][0],
            'pi_to_pn': T[3][1],
            'pi_to_blancos': T[3][2],
            'pi_to_fa_lower': ci['lower'][3][0],
            'pi_to_fa_upper': ci['upper'][3][0],
            'pi_votes': result['pi_primera_votes'],

            # OTROS transfers
            'otros_to_fa': T[4][0],
            'otros_to_pn': T[4][1],
            'otros_to_blancos': T[4][2],
            'otros_votes': result['otros_primera_votes'],

            # Diagnostics
            'max_rhat': max(max(r) if isinstance(r, list) else r for r in diag['rhat']),
            'min_ess': min(min(e) if isinstance(e, list) else e for e in diag['ess']),
        }

        rows.append(row)

    return pd.DataFrame(rows)


def create_stratified_table(strat_results, filename):
    """Create CSV table from stratified results."""

    rows = []

    for stratum_name, result in strat_results.items():
        T = result['transition_matrix']
        ci = result['credible_intervals']

        row = {
            'stratum': stratum_name,
            'n_circuits': result['n_circuits'],

            # CA
            'ca_to_fa': T[0, 0],
            'ca_to_pn': T[0, 1],
            'ca_to_fa_lower': ci['lower'][0, 0],
            'ca_to_fa_upper': ci['upper'][0, 0],
            'ca_votes': result['ca_primera_votes'],

            # PC
            'pc_to_fa': T[1, 0],
            'pc_to_pn': T[1, 1],
            'pc_votes': result['pc_primera_votes'],

            # PN
            'pn_to_fa': T[2, 0],
            'pn_to_pn': T[2, 1],
            'pn_votes': result['pn_primera_votes'],

            # PI
            'pi_to_fa': T[3, 0],
            'pi_to_pn': T[3, 1],
            'pi_votes': result['pi_primera_votes'],
        }

        rows.append(row)

    return pd.DataFrame(rows)


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description='Complete King\'s EI analysis for 2019 elections'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/circuitos_merged_2019.parquet',
        help='Path to 2019 merged data'
    )
    parser.add_argument(
        '--national-only',
        action='store_true',
        help='Run only national analysis (skip departments and stratified)'
    )
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick mode: 1000 samples, 2 chains (for testing)'
    )

    args = parser.parse_args()

    # Set sample sizes
    if args.quick:
        national_samples, national_chains = 1000, 2
        dept_samples, dept_chains = 500, 2
        strat_samples, strat_chains = 750, 2
    else:
        national_samples, national_chains = 4000, 4
        dept_samples, dept_chains = 1000, 2
        strat_samples, strat_chains = 2000, 3

    # Load data
    logger.info(f"Loading 2019 data from {args.data}...")
    df = pd.read_parquet(args.data)

    logger.info(f"\n2019 DATASET SUMMARY:")
    logger.info(f"  Total circuits: {len(df):,}")
    logger.info(f"  Departments: {df['departamento'].nunique()}")
    logger.info(f"  Total votes primera: {df['total_primera'].sum():,.0f}")
    logger.info(f"  Total votes ballotage: {df['total_ballotage'].sum():,.0f}")
    logger.info(f"\n  Party votes (primera vuelta):")
    logger.info(f"    CA: {df['ca_primera'].sum():,.0f} ({df['ca_primera'].sum()/df['total_primera'].sum()*100:.2f}%)")
    logger.info(f"    PC: {df['pc_primera'].sum():,.0f} ({df['pc_primera'].sum()/df['total_primera'].sum()*100:.2f}%)")
    logger.info(f"    PN: {df['pn_primera'].sum():,.0f} ({df['pn_primera'].sum()/df['total_primera'].sum()*100:.2f}%)")
    logger.info(f"    PI: {df['pi_primera'].sum():,.0f} ({df['pi_primera'].sum()/df['total_primera'].sum()*100:.2f}%)")
    logger.info(f"    FA: {df['fa_primera'].sum():,.0f} ({df['fa_primera'].sum()/df['total_primera'].sum()*100:.2f}%)")
    logger.info(f"    OTROS: {df['otros_primera'].sum():,.0f} ({df['otros_primera'].sum()/df['total_primera'].sum()*100:.2f}%)")

    # Output directories
    results_dir = Path('outputs/results')
    tables_dir = Path('outputs/tables')
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # ========================================================================
    # 1. NATIONAL ANALYSIS
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 1: NATIONAL ANALYSIS")
    logger.info("="*70)

    national_result = analyze_national(df, samples=national_samples, chains=national_chains)

    if national_result:
        # Save national results
        with open(results_dir / 'national_transfers_2019.pkl', 'wb') as f:
            pickle.dump(national_result, f)
        logger.info(f"\nSaved: {results_dir / 'national_transfers_2019.pkl'}")

    if args.national_only:
        logger.info("\n--national-only flag set. Skipping department and stratified analyses.")
        return

    # ========================================================================
    # 2. DEPARTMENT ANALYSIS
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 2: DEPARTMENT ANALYSIS (19 departments)")
    logger.info("="*70)

    dept_results = analyze_departments(df, samples=dept_samples, chains=dept_chains)

    if dept_results:
        # Save department results
        with open(results_dir / 'department_transfers_2019.pkl', 'wb') as f:
            pickle.dump(dept_results, f)
        logger.info(f"\nSaved: {results_dir / 'department_transfers_2019.pkl'}")

        # Create and save summary table
        dept_table = create_department_summary_table(dept_results)
        dept_table.to_csv(tables_dir / 'transfers_by_department_2019.csv', index=False)
        logger.info(f"Saved: {tables_dir / 'transfers_by_department_2019.csv'}")

        # Print summary
        logger.info("\n" + "="*70)
        logger.info("DEPARTMENT SUMMARY: CA DEFECTION TO FA (2019)")
        logger.info("="*70)
        dept_table_sorted = dept_table.sort_values('ca_to_fa', ascending=False)
        for _, row in dept_table_sorted.iterrows():
            logger.info(f"{row['departamento']:20s}: {row['ca_to_fa']*100:5.1f}% "
                       f"[{row['ca_to_fa_lower']*100:5.1f}% - {row['ca_to_fa_upper']*100:5.1f}%] "
                       f"({row['ca_votes']:,.0f} votes)")

    # ========================================================================
    # 3. URBAN VS RURAL (Montevideo vs Interior)
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 3: URBAN VS RURAL STRATIFICATION")
    logger.info("="*70)

    urban_rural_results = analyze_urban_rural(df, samples=strat_samples, chains=strat_chains)

    if urban_rural_results:
        # Save results
        with open(results_dir / 'urban_rural_transfers_2019.pkl', 'wb') as f:
            pickle.dump(urban_rural_results, f)
        logger.info(f"\nSaved: {results_dir / 'urban_rural_transfers_2019.pkl'}")

        # Create table
        urban_rural_table = create_stratified_table(urban_rural_results, 'urban_rural')
        urban_rural_table.to_csv(tables_dir / 'urban_rural_2019.csv', index=False)
        logger.info(f"Saved: {tables_dir / 'urban_rural_2019.csv'}")

    # ========================================================================
    # 4. METROPOLITAN VS INTERIOR
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("PHASE 4: METROPOLITAN VS INTERIOR STRATIFICATION")
    logger.info("="*70)

    region_results = analyze_by_region(df, samples=strat_samples, chains=strat_chains)

    if region_results:
        # Save results
        with open(results_dir / 'region_transfers_2019.pkl', 'wb') as f:
            pickle.dump(region_results, f)
        logger.info(f"\nSaved: {results_dir / 'region_transfers_2019.pkl'}")

        # Create table
        region_table = create_stratified_table(region_results, 'region')
        region_table.to_csv(tables_dir / 'region_2019.csv', index=False)
        logger.info(f"Saved: {tables_dir / 'region_2019.csv'}")

    # ========================================================================
    # FINAL SUMMARY
    # ========================================================================
    logger.info("\n" + "="*70)
    logger.info("ANALYSIS COMPLETE - 2019 ELECTIONS")
    logger.info("="*70)
    logger.info("\nFiles generated:")
    logger.info(f"  1. {results_dir / 'national_transfers_2019.pkl'}")
    logger.info(f"  2. {results_dir / 'department_transfers_2019.pkl'}")
    logger.info(f"  3. {results_dir / 'urban_rural_transfers_2019.pkl'}")
    logger.info(f"  4. {results_dir / 'region_transfers_2019.pkl'}")
    logger.info(f"  5. {tables_dir / 'transfers_by_department_2019.csv'}")
    logger.info(f"  6. {tables_dir / 'urban_rural_2019.csv'}")
    logger.info(f"  7. {tables_dir / 'region_2019.csv'}")

    logger.info("\n" + "="*70)
    logger.info("KEY FINDINGS - 2019")
    logger.info("="*70)

    if national_result:
        T = national_result['transition_matrix']
        logger.info(f"\nNATIONAL TRANSFERS (all parties to FA):")
        logger.info(f"  CA → FA: {T[0,0]*100:.1f}%")
        logger.info(f"  PC → FA: {T[1,0]*100:.1f}%")
        logger.info(f"  PN → FA: {T[2,0]*100:.1f}%")
        logger.info(f"  PI → FA: {T[3,0]*100:.1f}%")
        logger.info(f"  OTROS → FA: {T[4,0]*100:.1f}%")

        # Calculate coalition defection
        ca_votes = national_result['ca_primera_votes']
        pc_votes = national_result['pc_primera_votes']
        pn_votes = national_result['pn_primera_votes']

        ca_to_fa_votes = ca_votes * T[0,0]
        pc_to_fa_votes = pc_votes * T[1,0]
        pn_to_fa_votes = pn_votes * T[2,0]

        total_coalition_defection = ca_to_fa_votes + pc_to_fa_votes + pn_to_fa_votes

        logger.info(f"\nCOALITION DEFECTION (2019):")
        logger.info(f"  CA defectors to FA: {ca_to_fa_votes:,.0f} votes")
        logger.info(f"  PC defectors to FA: {pc_to_fa_votes:,.0f} votes")
        logger.info(f"  PN defectors to FA: {pn_to_fa_votes:,.0f} votes")
        logger.info(f"  TOTAL DEFECTION: {total_coalition_defection:,.0f} votes")


if __name__ == '__main__':
    main()
