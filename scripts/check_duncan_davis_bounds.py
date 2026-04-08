"""
Issue #12: Duncan-Davis / Fréchet bounds check on ecological inference estimates.

For each circuit i, the Fréchet (sharp) ecological bounds on the transfer rate
β (fraction of party-X voters going to party-Y) are:

  lower_i = max(0,  (y_i - (N_i - x_i)) / x_i)
  upper_i = min(1,  y_i / x_i)

where:
  x_i = votes for origin party in primera vuelta (e.g. CA)
  y_i = votes for destination party in ballotage (e.g. FA)
  N_i = total primera vuelta votes in circuit

Aggregating across circuits gives the Duncan-Davis bounds on the national rate:

  β_lower = Σ x_i * lower_i / Σ x_i
  β_upper = Σ x_i * upper_i / Σ x_i

Our DM estimate must fall within [β_lower, β_upper] to be logically consistent.

Outputs:
  - outputs/tables/duncan_davis_bounds.csv  (per origin-destination pair)
  - outputs/tables/latex/duncan_davis_bounds.tex
  - Flags circuits where DM point estimate is outside individual bounds

Usage:
  conda run -n ei2024 python scripts/check_duncan_davis_bounds.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd

# ── DM national estimates (from temporal_comparison_dm or national analysis) ──

# From temporal_comparison_dm.tex (2024 national DM model)
DM_ESTIMATES = {
    ("CA",    "FA"):      0.443,
    ("CA",    "PN"):      0.510,
    ("CA",    "BLANCOS"): 0.047,
    ("FA",    "FA"):      0.992,
    ("FA",    "PN"):      0.002,
    ("FA",    "BLANCOS"): 0.006,
    ("PC",    "FA"):      0.051,
    ("PC",    "PN"):      0.927,
    ("PC",    "BLANCOS"): 0.022,
    ("PN",    "FA"):      0.075,
    ("PN",    "PN"):      0.906,
    ("PN",    "BLANCOS"): 0.018,
}

ORIGIN_COLS = {
    "CA":    "ca_primera",
    "FA":    "fa_primera",
    "PC":    "pc_primera",
    "PN":    "pn_primera",
}
DEST_COLS = {
    "FA":      "fa_ballotage",
    "PN":      "pn_ballotage",
    "BLANCOS": "blancos_ballotage",
}

# ── load data ─────────────────────────────────────────────────────────────────

df = pd.read_parquet("data/processed/circuitos_merged.parquet")
N  = df["total_primera"].values
print(f"Loaded {len(df):,} circuits")

# ── compute Fréchet bounds per origin-destination pair ────────────────────────

rows = []

for (orig, dest), dm_est in DM_ESTIMATES.items():
    if orig not in ORIGIN_COLS or dest not in DEST_COLS:
        continue

    x = df[ORIGIN_COLS[orig]].values.astype(float)
    y = df[DEST_COLS[dest]].values.astype(float)

    # Skip circuits where origin party has 0 votes
    mask = x > 0
    x_m, y_m, N_m = x[mask], y[mask], N[mask]

    # Fréchet bounds per circuit
    lo_i = np.maximum(0.0, (y_m - (N_m - x_m)) / x_m)
    hi_i = np.minimum(1.0, y_m / x_m)

    # Weighted aggregation (Duncan-Davis)
    w        = x_m / x_m.sum()
    dd_lower = float(np.dot(w, lo_i))
    dd_upper = float(np.dot(w, hi_i))

    # Check if DM estimate is within bounds
    within = dd_lower <= dm_est <= dd_upper
    margin_lo = dm_est - dd_lower
    margin_hi = dd_upper - dm_est

    # Count circuits where individual bounds exclude the national estimate
    n_exclude_lo = int((lo_i > dm_est).sum())
    n_exclude_hi = int((hi_i < dm_est).sum())

    rows.append({
        "origen":      orig,
        "destino":     dest,
        "n_circuits":  int(mask.sum()),
        "dd_lower":    dd_lower,
        "dd_upper":    dd_upper,
        "dm_estimate": dm_est,
        "within_bounds": within,
        "margin_lower":  margin_lo,
        "margin_upper":  margin_hi,
        "n_circuits_exclude_lo": n_exclude_lo,
        "n_circuits_exclude_hi": n_exclude_hi,
    })

results = pd.DataFrame(rows)

# ── save CSV ──────────────────────────────────────────────────────────────────

out_dir = Path("outputs/tables")
out_dir.mkdir(parents=True, exist_ok=True)
csv_path = out_dir / "duncan_davis_bounds.csv"
results.to_csv(csv_path, index=False)
print(f"CSV saved: {csv_path}")

# ── print summary ─────────────────────────────────────────────────────────────

print("\n" + "=" * 78)
print("DUNCAN-DAVIS / FRÉCHET BOUNDS CHECK")
print("=" * 78)
print(f"{'Origen->Destino':18s}  {'DD Lower':>9}  {'DM Est.':>9}  {'DD Upper':>9}  {'Within?':>8}  {'Margin Lo':>10}  {'Margin Hi':>10}")
print("-" * 78)

for _, r in results.iterrows():
    flag = "OK" if r["within_bounds"] else "VIOLATION"
    print(
        f"{r['origen']}->{r['destino']:12s}  "
        f"{r['dd_lower']*100:8.1f}%  "
        f"{r['dm_estimate']*100:8.1f}%  "
        f"{r['dd_upper']*100:8.1f}%  "
        f"{flag:>10}  "
        f"{r['margin_lower']*100:+9.1f}pp  "
        f"{r['margin_upper']*100:+9.1f}pp"
    )

violations = results[~results["within_bounds"]]
if violations.empty:
    print("\nAll DM estimates are within Duncan-Davis bounds.")
else:
    print(f"\n{len(violations)} violations found:")
    print(violations[["origen", "destino", "dd_lower", "dm_estimate", "dd_upper"]].to_string())

# ── generate LaTeX table ──────────────────────────────────────────────────────

latex_dir = Path("outputs/tables/latex")
latex_dir.mkdir(parents=True, exist_ok=True)

lines = [
    r"% Bounds de Duncan-Davis / Fréchet para verificación de estimados DM",
    r"% Generado por scripts/check_duncan_davis_bounds.py",
    r"",
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Cotas de Duncan-Davis (Fréchet) para los estimados del modelo "
    r"DirichletMultinomial (2024). Las cotas definen el rango lógicamente posible "
    r"para cada tasa de transferencia dados los conteos electorales observados. "
    r"Todos los estimados dentro de las cotas son logicamente consistentes.}",
    r"\label{tab:duncan_davis_bounds}",
    r"\small",
    r"\begin{tabular}{llcccc}",
    r"\toprule",
    r"Origen & Destino & Cota inferior & Estimado DM & Cota superior & ¿Válido? \\",
    r"\midrule",
]

for _, r in results.sort_values(["origen", "destino"]).iterrows():
    check = r"\checkmark" if r["within_bounds"] else r"\times"
    lines.append(
        rf"\textsc{{{r['origen']}}} & {r['destino']} & "
        rf"{r['dd_lower']*100:.1f}\% & "
        rf"\textbf{{{r['dm_estimate']*100:.1f}\%}} & "
        rf"{r['dd_upper']*100:.1f}\% & "
        rf"${check}$ \\"
    )

lines += [
    r"\bottomrule",
    r"\end{tabular}",
    r"\begin{tablenotes}\small",
    r"\item Cotas calculadas usando el método de Fréchet (Duncan y Davis, 1953): "
    r"$\beta^-_i = \max(0, (y_i - N_i + x_i)/x_i)$, "
    r"$\beta^+_i = \min(1, y_i/x_i)$, agregadas ponderando por $x_i$.",
    r"\end{tablenotes}",
    r"\end{table}",
    r"",
]

tex_path = latex_dir / "duncan_davis_bounds.tex"
tex_path.write_text("\n".join(lines), encoding="utf-8")
print(f"\nLaTeX saved: {tex_path}")
