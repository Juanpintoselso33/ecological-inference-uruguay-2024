"""
Departmental ecological inference with DirichletMultinomial likelihood.

Runs KingEI(likelihood='dirichlet_multinomial') for each of the 19
Uruguayan departments (2024 data) and saves:
  - outputs/tables/transfers_by_department_dm.csv
  - outputs/tables/latex/department_dm_results.tex

Usage (from project root, WSL2):
  /opt/miniconda3/envs/ei2024/bin/python scripts/analyze_by_department_dm.py
  /opt/miniconda3/envs/ei2024/bin/python scripts/analyze_by_department_dm.py --year 2019
  /opt/miniconda3/envs/ei2024/bin/python scripts/analyze_by_department_dm.py --quick
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import argparse
import numpy as np
import pandas as pd
from src.models.king_ei import KingEI
from src.utils import get_logger

logger = get_logger(__name__)

ORIGIN_COLS = ['ca_primera', 'fa_primera', 'otros_primera', 'pc_primera', 'pn_primera']
DEST_COLS   = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']
ORIGIN_LABELS = ['CA', 'FA', 'OTROS', 'PC', 'PN']
DEST_LABELS   = ['FA', 'PN', 'BLANCOS']

# Indices for reporting
IDX = {party: ORIGIN_LABELS.index(party) for party in ['CA', 'PC', 'PN', 'FA']}


def analyze_department(dept_name: str, df_dept: pd.DataFrame,
                       num_samples: int, num_chains: int) -> dict | None:
    if len(df_dept) < 10:
        logger.warning("Skipping %s: only %d circuits", dept_name, len(df_dept))
        return None

    model = KingEI(
        num_samples=num_samples,
        num_chains=num_chains,
        num_warmup=num_samples,
        random_seed=42,
        likelihood='dirichlet_multinomial',
        nuts_sampler='auto',
    )

    try:
        model.fit(
            data=df_dept,
            origin_cols=ORIGIN_COLS,
            destination_cols=DEST_COLS,
            total_origin='total_primera',
            total_destination='total_ballotage',
            progressbar=False,
        )
    except Exception as exc:
        logger.error("Failed to fit %s: %s", dept_name, exc)
        return None

    T  = model.get_transition_matrix()
    ci = model.get_credible_intervals(0.95)
    lo = ci['lower']
    hi = ci['upper']

    row = {
        'departamento': dept_name,
        'n_circuits': len(df_dept),
    }

    for party, pidx in IDX.items():
        for didx, dest in enumerate(DEST_LABELS):
            key = f"{party.lower()}_to_{dest.lower()}"
            row[key]            = T[pidx, didx]
            row[f"{key}_lower"] = lo[pidx, didx]
            row[f"{key}_upper"] = hi[pidx, didx]

    # Vote totals
    for col in ['ca_primera', 'pc_primera', 'pn_primera', 'fa_primera']:
        row[f"{col}_total"] = int(df_dept[col].sum())

    diag = model.get_diagnostics()
    row['max_rhat'] = float(np.max(diag['rhat']))
    row['min_ess']  = float(np.min(diag['ess']))

    logger.info(
        "%s — CA→FA: %.1f%% [%.1f%%, %.1f%%]  PC→FA: %.1f%%  PN→FA: %.1f%%  "
        "R̂≤%.3f  ESS≥%.0f",
        dept_name,
        row['ca_to_fa'] * 100, row['ca_to_fa_lower'] * 100, row['ca_to_fa_upper'] * 100,
        row['pc_to_fa'] * 100, row['pn_to_fa'] * 100,
        row['max_rhat'], row['min_ess'],
    )
    return row


def build_latex_table(df: pd.DataFrame, year: int) -> str:
    """Generate LaTeX table with CA→FA, PC→FA, PN→FA by department."""
    lines = [
        rf"% Análisis departamental DirichletMultinomial {year}",
        r"% Generado por scripts/analyze_by_department_dm.py",
        r"",
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Inferencia ecológica por departamento ({year}), "
        r"verosimilitud DirichletMultinomial. "
        r"Se reporta la media posterior con intervalo de credibilidad al 95\%.}",
        rf"\label{{tab:dept_dm_{year}}}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Departamento & CA$\to$FA & PC$\to$FA & PN$\to$FA & Circuitos \\",
        r"\midrule",
    ]

    for _, row in df.sort_values('departamento').iterrows():
        def fmt(val, lo, hi):
            return (f"{val*100:.1f} "
                    f"[{lo*100:.1f}, {hi*100:.1f}]")

        lines.append(
            rf"\textsc{{{row['departamento'].title()}}} & "
            rf"{fmt(row['ca_to_fa'], row['ca_to_fa_lower'], row['ca_to_fa_upper'])} & "
            rf"{fmt(row['pc_to_fa'], row['pc_to_fa_lower'], row['pc_to_fa_upper'])} & "
            rf"{fmt(row['pn_to_fa'], row['pn_to_fa_lower'], row['pn_to_fa_upper'])} & "
            rf"{int(row['n_circuits'])} \\"
        )

    lines += [
        r"\bottomrule",
        r"\end{tabular}",
        r"\begin{tablenotes}\small",
        r"\item Modelo: King EI, verosimilitud DirichletMultinomial, "
        r"1{,}000 muestras por cadena, 2 cadenas.",
        r"\end{tablenotes}",
        r"\end{table}",
        r"",
    ]
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Departmental DM EI analysis")
    parser.add_argument('--year', type=int, default=2024, choices=[2019, 2024])
    parser.add_argument('--quick', action='store_true',
                        help='500 samples / 2 chains (fast, less accurate)')
    parser.add_argument('--dept', type=str, default=None,
                        help='Run a single department (for testing)')
    args = parser.parse_args()

    num_samples = 500 if args.quick else 1000
    num_chains  = 2

    data_paths = {
        2024: 'data/processed/circuitos_merged.parquet',
        2019: 'data/processed/circuitos_merged_2019.parquet',
    }
    df = pd.read_parquet(data_paths[args.year])
    print(f"Loaded {len(df):,} circuits — {df['departamento'].nunique()} departments ({args.year})")

    departments = sorted(df['departamento'].unique())
    if args.dept:
        departments = [d for d in departments if d.lower() == args.dept.lower()]
        if not departments:
            print(f"Department '{args.dept}' not found. Available: {sorted(df['departamento'].unique())}")
            sys.exit(1)

    results = []
    for i, dept in enumerate(departments, 1):
        print(f"\n[{i}/{len(departments)}] {dept} ({df[df['departamento'] == dept].shape[0]} circuits)")
        row = analyze_department(dept, df[df['departamento'] == dept], num_samples, num_chains)
        if row:
            results.append(row)

    if not results:
        print("No results produced.")
        sys.exit(1)

    results_df = pd.DataFrame(results)

    # Save CSV
    out_dir = Path('outputs/tables')
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.year}" if args.year != 2024 else ""
    csv_path = out_dir / f"transfers_by_department_dm{suffix}.csv"
    results_df.to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")

    # Save LaTeX
    latex_dir = Path('outputs/tables/latex')
    latex_dir.mkdir(parents=True, exist_ok=True)
    tex_path = latex_dir / f"department_dm_results{suffix}.tex"
    tex_path.write_text(build_latex_table(results_df, args.year), encoding='utf-8')
    print(f"LaTeX saved: {tex_path}")

    # Print summary
    print("\n" + "=" * 72)
    print(f"CA → FA by department ({args.year}) — DirichletMultinomial")
    print("=" * 72)
    for _, row in results_df.sort_values('ca_to_fa', ascending=False).iterrows():
        flag = "⚠" if row['max_rhat'] > 1.05 else " "
        print(f"{flag} {row['departamento']:20s}  "
              f"CA→FA: {row['ca_to_fa']*100:5.1f}% "
              f"[{row['ca_to_fa_lower']*100:5.1f}%, {row['ca_to_fa_upper']*100:5.1f}%]  "
              f"R̂={row['max_rhat']:.3f}")

    bad = results_df[results_df['max_rhat'] > 1.05]
    if not bad.empty:
        print(f"\nConvergence warnings (R̂ > 1.05): {list(bad['departamento'])}")


if __name__ == '__main__':
    main()
