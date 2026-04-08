"""
Stratified ecological inference with DirichletMultinomial likelihood.

Strata:
  - Urban vs Rural  (total_primera > 1000 = urban, else rural)
  - Metropolitan vs Interior  (Montevideo + Canelones = metro)

Outputs:
  - outputs/tables/stratified_dm_results.csv
  - outputs/tables/latex/urban_rural_dm.tex
  - outputs/tables/latex/metro_interior_dm.tex

Usage (WSL2):
  /opt/miniconda3/envs/ei2024/bin/python scripts/analyze_stratified_dm.py
  /opt/miniconda3/envs/ei2024/bin/python scripts/analyze_stratified_dm.py --year 2019
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

ORIGIN_COLS   = ['ca_primera', 'fa_primera', 'otros_primera', 'pc_primera', 'pn_primera']
DEST_COLS     = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']
ORIGIN_LABELS = ['CA', 'FA', 'OTROS', 'PC', 'PN']
DEST_LABELS   = ['FA', 'PN', 'BLANCOS']
IDX = {p: ORIGIN_LABELS.index(p) for p in ['CA', 'FA', 'PC', 'PN']}

METRO_DEPTS = {'montevideo', 'canelones'}


def classify(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Urban/rural — use existing column if available, else heuristic
    if 'urban_rural' in df.columns:
        df['urban_rural'] = df['urban_rural'].str.capitalize()
    else:
        # Fallback: habilitados median split (~390 for 2024 data)
        threshold = df['habilitados'].median()
        df['urban_rural'] = df['habilitados'].apply(
            lambda x: 'Urbano' if x >= threshold else 'Rural'
        )
        logger.warning("No urban_rural column found; using habilitados >= %.0f as Urban", threshold)

    # Metropolitan/Interior
    df['metro_interior'] = df['departamento'].str.lower().apply(
        lambda d: 'Metropolitano' if d in METRO_DEPTS else 'Interior'
    )
    return df


def fit_stratum(label: str, df_stratum: pd.DataFrame,
                num_samples: int, num_chains: int) -> dict | None:
    n = len(df_stratum)
    logger.info("  %s — %d circuitos", label, n)
    if n < 50:
        logger.warning("  Skipping %s: too few circuits (%d)", label, n)
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
            data=df_stratum,
            origin_cols=ORIGIN_COLS,
            destination_cols=DEST_COLS,
            total_origin='total_primera',
            total_destination='total_ballotage',
            progressbar=False,
        )
    except Exception as exc:
        logger.error("  Failed %s: %s", label, exc)
        return None

    T  = model.get_transition_matrix()
    ci = model.get_credible_intervals(0.95)
    lo, hi = ci['lower'], ci['upper']
    diag = model.get_diagnostics()

    row = {'stratum': label, 'n_circuits': n}
    for party, pidx in IDX.items():
        for didx, dest in enumerate(DEST_LABELS):
            k = f"{party.lower()}_to_{dest.lower()}"
            row[k]            = T[pidx, didx]
            row[f"{k}_lower"] = lo[pidx, didx]
            row[f"{k}_upper"] = hi[pidx, didx]

    row['max_rhat'] = float(np.max(diag['rhat']))
    row['min_ess']  = float(np.min(diag['ess']))

    logger.info(
        "  %s — CA→FA: %.1f%% [%.1f%%, %.1f%%]  PC→FA: %.1f%%  PN→FA: %.1f%%  R̂≤%.3f",
        label,
        row['ca_to_fa']*100, row['ca_to_fa_lower']*100, row['ca_to_fa_upper']*100,
        row['pc_to_fa']*100, row['pn_to_fa']*100, row['max_rhat'],
    )
    return row


def fmt_ci(val, lo, hi):
    return f"{val*100:.1f} [{lo*100:.1f}, {hi*100:.1f}]"


def build_latex_table(rows: list[dict], stratification: str, year: int,
                      caption_detail: str) -> str:
    label = stratification.lower().replace('/', '_').replace(' ', '_')
    lines = [
        f"% Análisis estratificado DM — {stratification} ({year})",
        r"% Generado por scripts/analyze_stratified_dm.py",
        r"",
        r"\begin{table}[htbp]",
        r"\centering",
        rf"\caption{{Inferencia ecológica estratificada por {caption_detail} ({year}), "
        r"verosimilitud DirichletMultinomial. "
        r"Media posterior con intervalo de credibilidad al 95\%.}",
        rf"\label{{tab:stratified_dm_{label}_{year}}}",
        r"\small",
        r"\begin{tabular}{lcccc}",
        r"\toprule",
        r"Estrato & CA$\to$FA & PC$\to$FA & PN$\to$FA & Circuitos \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            rf"\textsc{{{row['stratum']}}} & "
            rf"{fmt_ci(row['ca_to_fa'], row['ca_to_fa_lower'], row['ca_to_fa_upper'])} & "
            rf"{fmt_ci(row['pc_to_fa'], row['pc_to_fa_lower'], row['pc_to_fa_upper'])} & "
            rf"{fmt_ci(row['pn_to_fa'], row['pn_to_fa_lower'], row['pn_to_fa_upper'])} & "
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
    parser = argparse.ArgumentParser()
    parser.add_argument('--year', type=int, default=2024, choices=[2019, 2024])
    parser.add_argument('--samples', type=int, default=1000)
    parser.add_argument('--chains', type=int, default=2)
    args = parser.parse_args()

    data_paths = {
        2024: ['data/processed/circuitos_full_covariates.parquet',
               'data/processed/circuitos_merged.parquet'],
        2019: ['data/processed/circuitos_merged_2019.parquet'],
    }
    df = None
    for path in data_paths[args.year]:
        p = Path(path)
        if p.exists():
            df = pd.read_parquet(p)
            print(f"Loaded from: {path}")
            break
    if df is None:
        raise FileNotFoundError(f"No data file found for {args.year}")
    df = classify(df)

    print(f"Loaded {len(df):,} circuits ({args.year})")
    print("Urban/Rural:", df['urban_rural'].value_counts().to_dict())
    print("Metro/Interior:", df['metro_interior'].value_counts().to_dict())

    all_rows = []

    # --- Urban vs Rural ---
    print("\n=== Urban vs Rural ===")
    ur_rows = []
    for stratum in ['Urbano', 'Rural']:
        row = fit_stratum(stratum, df[df['urban_rural'] == stratum],
                          args.samples, args.chains)
        if row:
            ur_rows.append(row)
            all_rows.append({**row, 'stratification': 'urban_rural'})

    # --- Metropolitan vs Interior ---
    print("\n=== Metropolitano vs Interior ===")
    mi_rows = []
    for stratum in ['Metropolitano', 'Interior']:
        row = fit_stratum(stratum, df[df['metro_interior'] == stratum],
                          args.samples, args.chains)
        if row:
            mi_rows.append(row)
            all_rows.append({**row, 'stratification': 'metro_interior'})

    # Save CSV
    out_dir = Path('outputs/tables')
    out_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.year}" if args.year != 2024 else ""
    csv_path = out_dir / f"stratified_dm_results{suffix}.csv"
    pd.DataFrame(all_rows).to_csv(csv_path, index=False)
    print(f"\nCSV saved: {csv_path}")

    # Save LaTeX
    latex_dir = Path('outputs/tables/latex')
    latex_dir.mkdir(parents=True, exist_ok=True)

    ur_tex  = latex_dir / f"urban_rural_dm{suffix}.tex"
    mi_tex  = latex_dir / f"metro_interior_dm{suffix}.tex"

    ur_tex.write_text(
        build_latex_table(ur_rows, 'Urbano/Rural', args.year, 'zona urbana/rural'),
        encoding='utf-8'
    )
    mi_tex.write_text(
        build_latex_table(mi_rows, 'Metropolitano/Interior', args.year,
                          'región metropolitana e interior'),
        encoding='utf-8'
    )
    print(f"LaTeX saved: {ur_tex}")
    print(f"LaTeX saved: {mi_tex}")

    # Summary
    print("\n" + "=" * 65)
    print(f"SUMMARY — DirichletMultinomial stratified ({args.year})")
    print("=" * 65)
    for row in all_rows:
        flag = "⚠" if row['max_rhat'] > 1.05 else " "
        print(f"{flag} [{row['stratification']}] {row['stratum']:18s}  "
              f"CA→FA: {row['ca_to_fa']*100:5.1f}% "
              f"[{row['ca_to_fa_lower']*100:5.1f}%, {row['ca_to_fa_upper']*100:5.1f}%]  "
              f"R̂={row['max_rhat']:.3f}  ESS≥{row['min_ess']:.0f}")


if __name__ == '__main__':
    main()
