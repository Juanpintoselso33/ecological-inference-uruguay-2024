"""
Generate comprehensive summary of regional 2019 vs 2024 analysis.
To be run after both analyses are complete.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pickle
import pandas as pd
import numpy as np
from datetime import datetime

def generate_summary():
    """Generate comprehensive regional comparison summary."""

    print("="*100)
    print("GENERATING COMPREHENSIVE REGIONAL ANALYSIS SUMMARY")
    print("="*100)

    # Load results
    print("\nLoading 2019 results...")
    with open('outputs/results/region_transfers_2019.pkl', 'rb') as f:
        data_2019 = pickle.load(f)
    results_2019 = data_2019['results']

    print("Loading 2024 results...")
    with open('outputs/results/region_transfers_with_pi.pkl', 'rb') as f:
        data_2024 = pickle.load(f)
    results_2024 = data_2024['results']

    print("Loading comparison CSV...")
    df_2024 = pd.read_csv('outputs/tables/region_with_pi.csv')
    df_2019 = pd.read_csv('outputs/tables/region_2019.csv')

    # Party indices
    ca_idx, fa_idx, otros_idx, pc_idx, pi_idx, pn_idx = 0, 1, 2, 3, 4, 5

    # Create detailed summary tables
    summary_rows = []

    for year, data in [(2019, data_2019), (2024, data_2024)]:
        results = data['results']

        for region in ['Metropolitana', 'Interior']:
            r = results[region]
            T = r['transition_matrix']
            ci = r['ci_95']
            diag = r['diagnostics']

            row = {
                'Year': year,
                'Region': region,
                'Circuits': r['n_circuits'],
                'CA_votes': r['votes']['ca'],
                'PC_votes': r['votes']['pc'],
                'PI_votes': r['votes']['pi'],
                'PN_votes': r['votes']['pn'],
                'Total_votes': r['votes']['total_primera'],

                # CA
                'CA_to_FA_%': T[ca_idx, 0] * 100,
                'CA_to_FA_CI_low': ci['lower'][ca_idx, 0] * 100,
                'CA_to_FA_CI_high': ci['upper'][ca_idx, 0] * 100,

                # PC
                'PC_to_FA_%': T[pc_idx, 0] * 100,
                'PC_to_FA_CI_low': ci['lower'][pc_idx, 0] * 100,
                'PC_to_FA_CI_high': ci['upper'][pc_idx, 0] * 100,

                # PI
                'PI_to_FA_%': T[pi_idx, 0] * 100,
                'PI_to_FA_CI_low': ci['lower'][pi_idx, 0] * 100,
                'PI_to_FA_CI_high': ci['upper'][pi_idx, 0] * 100,

                # PN
                'PN_to_FA_%': T[pn_idx, 0] * 100,
                'PN_to_FA_CI_low': ci['lower'][pn_idx, 0] * 100,
                'PN_to_FA_CI_high': ci['upper'][pn_idx, 0] * 100,

                # Diagnostics
                'max_rhat': np.max(diag['rhat']),
                'min_ess': np.min(diag['ess']),
            }

            summary_rows.append(row)

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv('outputs/tables/regional_summary_2019_2024.csv', index=False)
    print(f"\nSummary table saved: outputs/tables/regional_summary_2019_2024.csv")

    # Generate detailed report
    report = []
    report.append("# REGIONAL ANALYSIS SUMMARY: 2019 vs 2024\n")
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

    report.append("\n## METROPOLITANA (Montevideo + Canelones)\n")

    m19 = results_2019['Metropolitana']
    m24 = results_2024['Metropolitana']
    T19m = m19['transition_matrix']
    T24m = m24['transition_matrix']

    report.append(f"\n### Cabildo Abierto (CA) → Frente Amplio\n")
    ca_19 = T19m[ca_idx, 0] * 100
    ca_24 = T24m[ca_idx, 0] * 100
    report.append(f"- 2019: {ca_19:.1f}%\n")
    report.append(f"- 2024: {ca_24:.1f}%\n")
    report.append(f"- Change: {ca_24 - ca_19:+.1f} pp\n")

    report.append(f"\n### Partido Colorado (PC) → Frente Amplio\n")
    pc_19 = T19m[pc_idx, 0] * 100
    pc_24 = T24m[pc_idx, 0] * 100
    report.append(f"- 2019: {pc_19:.1f}%\n")
    report.append(f"- 2024: {pc_24:.1f}%\n")
    report.append(f"- Change: {pc_24 - pc_19:+.1f} pp\n")

    report.append(f"\n### Partido Nacional (PN) → Frente Amplio\n")
    pn_19 = T19m[pn_idx, 0] * 100
    pn_24 = T24m[pn_idx, 0] * 100
    report.append(f"- 2019: {pn_19:.1f}%\n")
    report.append(f"- 2024: {pn_24:.1f}%\n")
    report.append(f"- Change: {pn_24 - pn_19:+.1f} pp\n")

    report.append("\n## INTERIOR (Resto del país)\n")

    i19 = results_2019['Interior']
    i24 = results_2024['Interior']
    T19i = i19['transition_matrix']
    T24i = i24['transition_matrix']

    report.append(f"\n### Cabildo Abierto (CA) → Frente Amplio\n")
    ca_19i = T19i[ca_idx, 0] * 100
    ca_24i = T24i[ca_idx, 0] * 100
    report.append(f"- 2019: {ca_19i:.1f}%\n")
    report.append(f"- 2024: {ca_24i:.1f}%\n")
    report.append(f"- Change: {ca_24i - ca_19i:+.1f} pp\n")

    report.append(f"\n### Partido Colorado (PC) → Frente Amplio\n")
    pc_19i = T19i[pc_idx, 0] * 100
    pc_24i = T24i[pc_idx, 0] * 100
    report.append(f"- 2019: {pc_19i:.1f}%\n")
    report.append(f"- 2024: {pc_24i:.1f}%\n")
    report.append(f"- Change: {pc_24i - pc_19i:+.1f} pp\n")

    report.append(f"\n### Partido Nacional (PN) → Frente Amplio\n")
    pn_19i = T19i[pn_idx, 0] * 100
    pn_24i = T24i[pn_idx, 0] * 100
    report.append(f"- 2019: {pn_19i:.1f}%\n")
    report.append(f"- 2024: {pn_24i:.1f}%\n")
    report.append(f"- Change: {pn_24i - pn_19i:+.1f} pp\n")

    report.append("\n## KEY GAPS (Metropolitana - Interior)\n")

    report.append(f"\n### CA → FA Gap\n")
    gap_ca_19 = (T19m[ca_idx, 0] - T19i[ca_idx, 0]) * 100
    gap_ca_24 = (T24m[ca_idx, 0] - T24i[ca_idx, 0]) * 100
    report.append(f"- 2019: {gap_ca_19:+.1f} pp (Metro - Interior)\n")
    report.append(f"- 2024: {gap_ca_24:+.1f} pp (Metro - Interior)\n")
    report.append(f"- Gap evolution: {gap_ca_24 - gap_ca_19:+.1f} pp\n")

    report.append(f"\n### PN → FA Gap\n")
    gap_pn_19 = (T19m[pn_idx, 0] - T19i[pn_idx, 0]) * 100
    gap_pn_24 = (T24m[pn_idx, 0] - T24i[pn_idx, 0]) * 100
    report.append(f"- 2019: {gap_pn_19:+.1f} pp (Metro - Interior)\n")
    report.append(f"- 2024: {gap_pn_24:+.1f} pp (Metro - Interior)\n")
    report.append(f"- Gap evolution: {gap_pn_24 - gap_pn_19:+.1f} pp\n")

    report.append("\n## VOTE IMPACT (2024)\n")

    report.append(f"\n### Metropolitana\n")
    ca_impact_m = m24['votes']['ca'] * T24m[ca_idx, 0]
    pc_impact_m = m24['votes']['pc'] * T24m[pc_idx, 0]
    pn_impact_m = m24['votes']['pn'] * T24m[pn_idx, 0]
    total_m = ca_impact_m + pc_impact_m + pn_impact_m

    report.append(f"- CA → FA: {ca_impact_m:,.0f} votes\n")
    report.append(f"- PC → FA: {pc_impact_m:,.0f} votes\n")
    report.append(f"- PN → FA: {pn_impact_m:,.0f} votes\n")
    report.append(f"- **Total coalition → FA: {total_m:,.0f} votes**\n")

    report.append(f"\n### Interior\n")
    ca_impact_i = i24['votes']['ca'] * T24i[ca_idx, 0]
    pc_impact_i = i24['votes']['pc'] * T24i[pc_idx, 0]
    pn_impact_i = i24['votes']['pn'] * T24i[pn_idx, 0]
    total_i = ca_impact_i + pc_impact_i + pn_impact_i

    report.append(f"- CA → FA: {ca_impact_i:,.0f} votes\n")
    report.append(f"- PC → FA: {pc_impact_i:,.0f} votes\n")
    report.append(f"- PN → FA: {pn_impact_i:,.0f} votes\n")
    report.append(f"- **Total coalition → FA: {total_i:,.0f} votes**\n")

    report.append(f"\n### Regional Distribution\n")
    report.append(f"- Metro receives: {total_m:,.0f} votes ({total_m/(total_m+total_i)*100:.1f}%)\n")
    report.append(f"- Interior receives: {total_i:,.0f} votes ({total_i/(total_m+total_i)*100:.1f}%)\n")

    report_text = ''.join(report)
    with open('outputs/reports/regional_analysis_summary.md', 'w', encoding='utf-8') as f:
        f.write(report_text)

    print(f"Report saved: outputs/reports/regional_analysis_summary.md")

    # Print summary
    print("\n" + "="*100)
    print("SUMMARY GENERATED")
    print("="*100)
    print(report_text)


if __name__ == '__main__':
    try:
        generate_summary()
    except FileNotFoundError as e:
        print(f"ERROR: {e}")
        print("Analysis files not yet available. Ensure both analyses have completed.")
