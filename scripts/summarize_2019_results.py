"""
Comprehensive summary of 2019 analysis results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import pickle
import numpy as np


def load_results():
    """Load all available results."""

    results_dir = Path('outputs/results')
    tables_dir = Path('outputs/tables')

    results = {}

    # National
    nat_path = results_dir / 'national_transfers_2019.pkl'
    if nat_path.exists():
        with open(nat_path, 'rb') as f:
            results['national'] = pickle.load(f)

    # Departments
    dept_path = results_dir / 'department_transfers_2019.pkl'
    if dept_path.exists():
        with open(dept_path, 'rb') as f:
            results['departments'] = pickle.load(f)

    # Urban/Rural
    ur_path = results_dir / 'urban_rural_transfers_2019.pkl'
    if ur_path.exists():
        with open(ur_path, 'rb') as f:
            results['urban_rural'] = pickle.load(f)

    # Region
    reg_path = results_dir / 'region_transfers_2019.pkl'
    if reg_path.exists():
        with open(reg_path, 'rb') as f:
            results['region'] = pickle.load(f)

    # Department table
    dept_table_path = tables_dir / 'transfers_by_department_2019.csv'
    if dept_table_path.exists():
        results['dept_table'] = pd.read_csv(dept_table_path)

    return results


def print_national_summary(nat_result):
    """Print national level summary."""

    print("="*80)
    print("NATIONAL ANALYSIS - 2019 ELECTIONS")
    print("="*80)

    print(f"\nCircuits analyzed: {nat_result['n_circuits']:,}")
    print(f"Total votes primera vuelta: {nat_result['total_primera']:,.0f}")
    print(f"Total votes ballotage: {nat_result['total_ballotage']:,.0f}")

    diag = nat_result['diagnostics']
    print(f"\nMCMC Diagnostics:")
    print(f"  Max R-hat: {max(max(r) if isinstance(r, list) else r for r in diag['rhat']):.4f}")
    print(f"  Min ESS: {min(min(e) if isinstance(e, list) else e for e in diag['ess']):.0f}")

    print(f"\n{'='*80}")
    print("TRANSITION MATRIX: How first-round voters moved to ballotage")
    print("="*80)

    T = nat_result['transition_matrix']
    ci = nat_result['credible_intervals']

    print(f"\n{'Party':15s} {'To FA':>12s} {'To PN':>12s} {'To Blancos':>12s} {'95% CI (FA)':>25s}")
    print("-"*80)

    for i, party in enumerate(nat_result['origin_cols']):
        party_name = party.replace('_primera', '').upper()
        ci_str = f"[{ci['lower'][i][0]*100:5.1f}% - {ci['upper'][i][0]*100:5.1f}%]"

        print(f"{party_name:15s} {T[i][0]*100:11.1f}% {T[i][1]*100:11.1f}% {T[i][2]*100:11.1f}%  {ci_str:>25s}")

    print("\n" + "="*80)
    print("COALITION DEFECTION ANALYSIS")
    print("="*80)

    ca_votes = nat_result['ca_primera_votes']
    pc_votes = nat_result['pc_primera_votes']
    pn_votes = nat_result['pn_primera_votes']
    pi_votes = nat_result['pi_primera_votes']

    ca_to_fa = T[0][0] * ca_votes
    pc_to_fa = T[1][0] * pc_votes
    pn_to_fa = T[2][0] * pn_votes
    pi_to_fa = T[3][0] * pi_votes

    total_coalition_votes = ca_votes + pc_votes + pn_votes + pi_votes
    total_defection = ca_to_fa + pc_to_fa + pn_to_fa + pi_to_fa

    print(f"\nCoalition parties (CA + PC + PN + PI): {total_coalition_votes:,.0f} votes")
    print(f"\nDefection to FA:")
    print(f"  CA voters:     {ca_to_fa:>12,.0f} ({T[0][0]*100:5.1f}% of {ca_votes:,.0f})")
    print(f"  PC voters:     {pc_to_fa:>12,.0f} ({T[1][0]*100:5.1f}% of {pc_votes:,.0f})")
    print(f"  PN voters:     {pn_to_fa:>12,.0f} ({T[2][0]*100:5.1f}% of {pn_votes:,.0f})")
    print(f"  PI voters:     {pi_to_fa:>12,.0f} ({T[3][0]*100:5.1f}% of {pi_votes:,.0f})")
    print(f"  {'='*60}")
    print(f"  TOTAL:         {total_defection:>12,.0f} ({total_defection/total_coalition_votes*100:5.1f}% of coalition)")

    # FA gains (estimated from transition matrix)
    otros_votes = nat_result.get('otros_primera_votes', 0)

    # FA ballotage = FA primera + gains from all other parties
    fa_ballotage_est = (nat_result.get('fa_primera_votes', 0) +
                        ca_votes * T[0][0] + pc_votes * T[1][0] +
                        pn_votes * T[2][0] + pi_votes * T[3][0] +
                        otros_votes * T[4][0])

    # Coalition ballotage = Coalition primera - losses to FA - losses to blank
    coalition_ballotage_est = (total_coalition_votes -
                                (ca_votes * T[0][0] + pc_votes * T[1][0] +
                                 pn_votes * T[2][0] + pi_votes * T[3][0]) -
                                (ca_votes * T[0][2] + pc_votes * T[1][2] +
                                 pn_votes * T[2][2] + pi_votes * T[3][2]))

    print(f"\nBALLOTAGE ESTIMATES:")
    print(f"  FA estimated votes:        {fa_ballotage_est:>12,.0f}")
    print(f"  Coalition estimated votes: {coalition_ballotage_est:>12,.0f}")
    print(f"  FA margin:                 {fa_ballotage_est - coalition_ballotage_est:>12,.0f}")

    # Actual ballotage results (if known)
    print(f"\n  (Note: Coalition won 2019 with 49.98% vs FA 48.43%)")


def print_department_summary(dept_table):
    """Print department-level summary."""

    print("\n" + "="*80)
    print("DEPARTMENT ANALYSIS - CA DEFECTION TO FA")
    print("="*80)

    print(f"\n{'Department':20s} {'CA->FA Rate':>12s} {'95% CI':>25s} {'CA Votes':>12s}")
    print("-"*80)

    dept_sorted = dept_table.sort_values('ca_to_fa', ascending=False)

    for _, row in dept_sorted.iterrows():
        ci_str = f"[{row['ca_to_fa_lower']*100:5.1f}% - {row['ca_to_fa_upper']*100:5.1f}%]"
        print(f"{row['departamento']:20s} {row['ca_to_fa']*100:11.1f}% {ci_str:>25s} {row['ca_votes']:>12,.0f}")

    print("\n" + "="*80)
    print("KEY INSIGHTS")
    print("="*80)

    highest = dept_sorted.iloc[0]
    lowest = dept_sorted.iloc[-1]

    print(f"\nHighest CA defection: {highest['departamento']} ({highest['ca_to_fa']*100:.1f}%)")
    print(f"Lowest CA defection:  {lowest['departamento']} ({lowest['ca_to_fa']*100:.1f}%)")
    print(f"Range: {(highest['ca_to_fa'] - lowest['ca_to_fa'])*100:.1f} percentage points")

    # Weighted average
    weighted_avg = (dept_sorted['ca_to_fa'] * dept_sorted['ca_votes']).sum() / dept_sorted['ca_votes'].sum()
    print(f"\nWeighted national average: {weighted_avg*100:.1f}%")

    # Regional patterns
    urban_depts = ['MO', 'CA']  # Montevideo, Canelones
    interior_depts = dept_sorted[~dept_sorted['departamento'].isin(urban_depts)]

    if len(urban_depts) > 0:
        urban_rows = dept_sorted[dept_sorted['departamento'].isin(urban_depts)]
        if len(urban_rows) > 0:
            urban_avg = (urban_rows['ca_to_fa'] * urban_rows['ca_votes']).sum() / urban_rows['ca_votes'].sum()
            interior_avg = (interior_depts['ca_to_fa'] * interior_depts['ca_votes']).sum() / interior_depts['ca_votes'].sum()

            print(f"\nUrban (Montevideo + Canelones): {urban_avg*100:.1f}%")
            print(f"Interior (rest):                 {interior_avg*100:.1f}%")
            print(f"Urban-Interior gap:              {(urban_avg - interior_avg)*100:.1f} pp")


def print_stratified_summary(strat_results, name):
    """Print stratified analysis summary."""

    print("\n" + "="*80)
    print(f"{name.upper()} STRATIFICATION")
    print("="*80)

    for stratum, result in strat_results.items():
        T = result['transition_matrix']

        print(f"\n{stratum}:")
        print(f"  Circuits: {result['n_circuits']:,}")
        print(f"  CA -> FA: {T[0][0]*100:5.1f}%  (from {result['ca_primera_votes']:,.0f} CA votes)")
        print(f"  PC -> FA: {T[1][0]*100:5.1f}%  (from {result['pc_primera_votes']:,.0f} PC votes)")
        print(f"  PN -> FA: {T[2][0]*100:5.1f}%  (from {result['pn_primera_votes']:,.0f} PN votes)")


def main():
    print("\n" + "="*80)
    print("2019 ELECTORAL INFERENCE ANALYSIS - COMPREHENSIVE SUMMARY")
    print("="*80)

    # Load results
    results = load_results()

    if not results:
        print("\nNo results found. Run analysis first.")
        return

    print(f"\nAvailable results:")
    for key in results.keys():
        print(f"  - {key}")

    # National
    if 'national' in results:
        print_national_summary(results['national'])

    # Departments
    if 'dept_table' in results:
        print_department_summary(results['dept_table'])

    # Urban/Rural
    if 'urban_rural' in results:
        print_stratified_summary(results['urban_rural'], "Urban vs Rural")

    # Region
    if 'region' in results:
        print_stratified_summary(results['region'], "Metropolitan vs Interior")

    print("\n" + "="*80)
    print("END OF SUMMARY")
    print("="*80)


if __name__ == '__main__':
    main()
