"""
Issue #8: Formal Bayesian test of geographic differences in defection rates.

Uses posterior normal approximation from DM stratified results to compute:
  - P(CA->FA rural > urban | data)
  - P(CA->FA interior > metro | data)
  - P(CA->FA interior > metro | data) for PN
  - Effect sizes (posterior mean difference + CI)

Outputs:
  - outputs/tables/geographic_difference_tests.csv
  - outputs/tables/latex/geographic_tests.tex

Usage:
  conda run -n ei2024 python scripts/test_geographic_differences.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from scipy import stats

# ── helpers ─────────────────────────────────────────────────────────────────

def ci_to_sd(lower, upper, level=0.95):
    """Approximate posterior SD from credible interval via normal approximation."""
    z = stats.norm.ppf((1 + level) / 2)   # 1.96 for 95%
    return (upper - lower) / (2 * z)


def prob_a_gt_b(mu_a, sd_a, mu_b, sd_b):
    """P(A > B) under independent normal posteriors."""
    diff_mean = mu_a - mu_b
    diff_sd   = np.sqrt(sd_a**2 + sd_b**2)
    return float(stats.norm.cdf(diff_mean / diff_sd))


def effect_ci(mu_a, sd_a, mu_b, sd_b, level=0.95):
    """Posterior mean and CI of (A - B)."""
    z      = stats.norm.ppf((1 + level) / 2)
    diff_m = mu_a - mu_b
    diff_s = np.sqrt(sd_a**2 + sd_b**2)
    return diff_m, diff_m - z * diff_s, diff_m + z * diff_s


# ── load results ─────────────────────────────────────────────────────────────

df = pd.read_csv("outputs/tables/stratified_dm_results.csv")

ur  = df[df["stratification"] == "urban_rural"].set_index("stratum")
mi  = df[df["stratification"] == "metro_interior"].set_index("stratum")

# ── run tests ────────────────────────────────────────────────────────────────

rows = []

comparisons = [
    # (label, group_df, stratum_a, stratum_b, party_key)
    ("Rural vs Urbano — CA->FA",      ur, "Rural",        "Urbano",        "ca_to_fa"),
    ("Rural vs Urbano — PC->FA",      ur, "Rural",        "Urbano",        "pc_to_fa"),
    ("Rural vs Urbano — PN->FA",      ur, "Rural",        "Urbano",        "pn_to_fa"),
    ("Interior vs Metro — CA->FA",    mi, "Interior",     "Metropolitano", "ca_to_fa"),
    ("Interior vs Metro — PC->FA",    mi, "Interior",     "Metropolitano", "pc_to_fa"),
    ("Interior vs Metro — PN->FA",    mi, "Interior",     "Metropolitano", "pn_to_fa"),
]

for label, g, stratum_a, stratum_b, key in comparisons:
    mu_a = g.loc[stratum_a, key]
    lo_a = g.loc[stratum_a, f"{key}_lower"]
    hi_a = g.loc[stratum_a, f"{key}_upper"]

    mu_b = g.loc[stratum_b, key]
    lo_b = g.loc[stratum_b, f"{key}_lower"]
    hi_b = g.loc[stratum_b, f"{key}_upper"]

    sd_a = ci_to_sd(lo_a, hi_a)
    sd_b = ci_to_sd(lo_b, hi_b)

    prob   = prob_a_gt_b(mu_a, sd_a, mu_b, sd_b)
    diff_m, diff_lo, diff_hi = effect_ci(mu_a, sd_a, mu_b, sd_b)

    rows.append({
        "comparison":  label,
        "stratum_a":   stratum_a,
        "mean_a":      mu_a,
        "stratum_b":   stratum_b,
        "mean_b":      mu_b,
        "diff_mean":   diff_m,
        "diff_lower":  diff_lo,
        "diff_upper":  diff_hi,
        "P(A>B)":      prob,
        "significant": "Sí" if (diff_lo > 0 or diff_hi < 0) else "No",
    })

results = pd.DataFrame(rows)

# ── save CSV ─────────────────────────────────────────────────────────────────

out_dir = Path("outputs/tables")
out_dir.mkdir(parents=True, exist_ok=True)
csv_path = out_dir / "geographic_difference_tests.csv"
results.to_csv(csv_path, index=False)
print(f"CSV saved: {csv_path}")

# ── print summary ─────────────────────────────────────────────────────────────

print("\n" + "=" * 75)
print("FORMAL BAYESIAN TESTS — Geographic differences (DM model)")
print("=" * 75)
for _, r in results.iterrows():
    sig = "OK" if r["significant"] == "Sí" else "--"
    print(f"{sig} {r['comparison']}")
    print(f"    {r['stratum_a']}: {r['mean_a']*100:.1f}%  vs  "
          f"{r['stratum_b']}: {r['mean_b']*100:.1f}%")
    print(f"    Diff = {r['diff_mean']*100:+.1f}pp  "
          f"[{r['diff_lower']*100:+.1f}, {r['diff_upper']*100:+.1f}]  "
          f"P(A>B) = {r['P(A>B)']:.3f}")

# ── generate LaTeX table ──────────────────────────────────────────────────────

latex_dir = Path("outputs/tables/latex")
latex_dir.mkdir(parents=True, exist_ok=True)

lines = [
    r"% Test formal de diferencias geográficas (DirichletMultinomial)",
    r"% Generado por scripts/test_geographic_differences.py",
    r"",
    r"\begin{table}[htbp]",
    r"\centering",
    r"\caption{Test bayesiano de diferencias geográficas en tasas de defección (2024). "
    r"Se reporta la diferencia posterior media con IC al 95\% y la probabilidad posterior "
    r"$P(A > B)$ bajo aproximación normal de los posteriors DirichletMultinomial.}",
    r"\label{tab:geographic_tests}",
    r"\small",
    r"\begin{tabular}{lcccc}",
    r"\toprule",
    r"Comparación & $\bar{\theta}_A$ & $\bar{\theta}_B$ & "
    r"$\Delta$ [IC 95\%] & $P(A > B)$ \\",
    r"\midrule",
]

groups = [
    ("Urbano vs Rural", [r for r in rows if "Rural vs Urbano" in r["comparison"]]),
    ("Metropolitano vs Interior", [r for r in rows if "Interior vs Metro" in r["comparison"]]),
]

for group_label, group_rows in groups:
    lines.append(rf"\multicolumn{{5}}{{l}}{{\textit{{{group_label}}}}} \\")
    for r in group_rows:
        party = r["comparison"].split("—")[1].strip()
        sig_marker = r"$^*$" if r["significant"] == "Sí" else ""
        lines.append(
            rf"\quad {party} & "
            rf"{r['mean_a']*100:.1f}\% & "
            rf"{r['mean_b']*100:.1f}\% & "
            rf"{r['diff_mean']*100:+.1f} [{r['diff_lower']*100:+.1f}, {r['diff_upper']*100:+.1f}] & "
            rf"{r['P(A>B)']:.3f}{sig_marker} \\"
        )
    lines.append(r"\midrule")

lines[-1] = r"\bottomrule"  # replace last \midrule with \bottomrule
lines += [
    r"\end{tabular}",
    r"\begin{tablenotes}\small",
    r"\item $^*$ El IC al 95\% de la diferencia no incluye cero.",
    r"\item $A$ = estrato con mayor defección esperada (Rural / Interior); "
    r"$B$ = estrato de referencia (Urbano / Metropolitano).",
    r"\end{tablenotes}",
    r"\end{table}",
    r"",
]

tex_path = latex_dir / "geographic_tests.tex"
tex_path.write_text("\n".join(lines), encoding="utf-8")
print(f"LaTeX saved: {tex_path}")
