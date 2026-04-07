"""
Comparacion temporal 2019 vs 2024 con DirichletMultinomial.

Ejecutar DESPUES de que termine compare_eiCircles.py (resultados 2024).

Fases:
1. Carga resultados 2024 DM (outputs/tables/comparison_eiCircles_python.csv)
2. Ajusta modelo DM en datos 2019 (7,213 circuitos, 4000 samples)
3. Genera tabla comparativa y figura
4. Guarda LaTeX y CSV

Uso:
    conda run -n ds python scripts/compare_temporal_dm.py
"""
import sys
import warnings
import logging
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

from src.models.king_ei import KingEI

ORIGIN      = ['ca_primera', 'fa_primera', 'pc_primera', 'pn_primera', 'pi_primera', 'otros_primera']
DEST        = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']
ORIGIN_LBLS = ['CA', 'FA', 'PC', 'PN', 'PI', 'OTROS']
DEST_LBLS   = ['FA', 'PN', 'BLANCOS']

OUT_DIR = Path('outputs/tables')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── 1. Cargar resultados 2024 DM ────────────────────────────────────────────
csv_2024 = OUT_DIR / 'comparison_eiCircles_python.csv'
if not csv_2024.exists():
    raise FileNotFoundError(
        f"No se encontró {csv_2024}.\n"
        "Ejecutar primero: conda run -n ds python scripts/compare_eiCircles.py"
    )

df_2024 = pd.read_csv(csv_2024)
logger.info("Resultados 2024 DM cargados desde %s", csv_2024)

# Reconstruir matrices T_2024
T_2024       = np.zeros((len(ORIGIN_LBLS), len(DEST_LBLS)))
T_2024_lower = np.zeros_like(T_2024)
T_2024_upper = np.zeros_like(T_2024)
for _, row in df_2024.iterrows():
    i = ORIGIN_LBLS.index(row.origin)
    j = DEST_LBLS.index(row.destination)
    T_2024[i, j]       = row['mean']
    T_2024_lower[i, j] = row['lower_95']
    T_2024_upper[i, j] = row['upper_95']

# ─── 2. Ajustar modelo DM en 2019 ────────────────────────────────────────────
df19 = pd.read_parquet('data/processed/circuitos_merged_2019.parquet')
logger.info("Datos 2019: %d circuitos", len(df19))

logger.info("Ajustando DirichletMultinomial 2019 (4000 samples, 4 chains)...")
model_2019 = KingEI(
    num_samples=4000,
    num_chains=4,
    num_warmup=2000,
    target_accept=0.9,
    random_seed=42,
    likelihood='dirichlet_multinomial',
)
model_2019.fit(
    df19,
    origin_cols=ORIGIN,
    destination_cols=DEST,
    total_origin='total_primera',
    total_destination='total_ballotage',
    progressbar=True,
)

T_2019       = model_2019.get_transition_matrix()
ci_2019      = model_2019.get_credible_intervals(prob=0.95)
T_2019_lower = ci_2019['lower']
T_2019_upper = ci_2019['upper']
diag_2019    = model_2019.get_diagnostics()
logger.info("2019 R-hat max: %.4f | ESS min: %.0f",
            np.max(diag_2019['rhat']), np.min(diag_2019['ess']))

# ─── 3. Guardar resultados 2019 ───────────────────────────────────────────────
rows_2019 = []
for i, orig in enumerate(ORIGIN_LBLS):
    for j, dest in enumerate(DEST_LBLS):
        rows_2019.append({
            'origin': orig, 'destination': dest,
            'mean': round(T_2019[i, j], 4),
            'lower_95': round(T_2019_lower[i, j], 4),
            'upper_95': round(T_2019_upper[i, j], 4),
            'model': 'KingEI_DirichletMultinomial_2019',
        })
csv_2019 = OUT_DIR / 'comparison_eiCircles_2019_python.csv'
pd.DataFrame(rows_2019).to_csv(csv_2019, index=False)
logger.info("Resultados 2019 guardados: %s", csv_2019)

# ─── 4. Tabla comparativa ─────────────────────────────────────────────────────
print("\n" + "="*72)
print("COMPARACION TEMPORAL: DirichletMultinomial  2019 vs 2024")
print("="*72)

print("\n--- Probabilidades de transicion hacia FA_ballotage ---")
print(f"\n{'Origen':<8}  {'2019':>7}  {'IC95_2019':>16}  {'2024':>7}  {'IC95_2024':>16}  {'Δ':>6}")
print("-"*70)
for i, lbl in enumerate(ORIGIN_LBLS):
    j = 0  # FA
    d19  = T_2019[i, j]
    d24  = T_2024[i, j]
    l19  = T_2019_lower[i, j]; u19 = T_2019_upper[i, j]
    l24  = T_2024_lower[i, j]; u24 = T_2024_upper[i, j]
    delta = d24 - d19
    sig = "*" if (l24 > u19 or u24 < l19) else " "
    print(f"{lbl:<8}  {d19:>6.3f}  [{l19:.3f},{u19:.3f}]  {d24:>6.3f}  [{l24:.3f},{u24:.3f}]  {delta:>+6.3f}{sig}")

print("\n* = IC95 no se solapan (cambio significativo)")

print("\n--- Probabilidades de transicion hacia PN_ballotage ---")
print(f"\n{'Origen':<8}  {'2019':>7}  {'IC95_2019':>16}  {'2024':>7}  {'IC95_2024':>16}  {'Δ':>6}")
print("-"*70)
for i, lbl in enumerate(ORIGIN_LBLS):
    j = 1  # PN
    d19  = T_2019[i, j]; d24  = T_2024[i, j]
    l19  = T_2019_lower[i, j]; u19 = T_2019_upper[i, j]
    l24  = T_2024_lower[i, j]; u24 = T_2024_upper[i, j]
    delta = d24 - d19
    sig = "*" if (l24 > u19 or u24 < l19) else " "
    print(f"{lbl:<8}  {d19:>6.3f}  [{l19:.3f},{u19:.3f}]  {d24:>6.3f}  [{l24:.3f},{u24:.3f}]  {delta:>+6.3f}{sig}")

# ─── 5. CSV comparativo completo ──────────────────────────────────────────────
rows_comp = []
for i, orig in enumerate(ORIGIN_LBLS):
    for j, dest in enumerate(DEST_LBLS):
        rows_comp.append({
            'origin': orig,
            'destination': dest,
            'T_2019': round(T_2019[i, j], 4),
            'lower_2019': round(T_2019_lower[i, j], 4),
            'upper_2019': round(T_2019_upper[i, j], 4),
            'T_2024': round(T_2024[i, j], 4),
            'lower_2024': round(T_2024_lower[i, j], 4),
            'upper_2024': round(T_2024_upper[i, j], 4),
            'delta': round(T_2024[i, j] - T_2019[i, j], 4),
        })
csv_comp = OUT_DIR / 'temporal_comparison_dm_2019_2024.csv'
pd.DataFrame(rows_comp).to_csv(csv_comp, index=False)
logger.info("Tabla comparativa guardada: %s", csv_comp)

# ─── 6. Tabla LaTeX ──────────────────────────────────────────────────────────
def fmt_ci(mean, lower, upper):
    return f"{mean:.3f} [{lower:.3f}, {upper:.3f}]"

latex_path = Path('outputs/tables/latex/temporal_comparison_dm.tex')
latex_path.parent.mkdir(parents=True, exist_ok=True)

with open(latex_path, 'w', encoding='utf-8') as f:
    f.write("% Tabla: Comparacion temporal 2019 vs 2024 (DirichletMultinomial)\n")
    f.write("% Generada por scripts/compare_temporal_dm.py\n\n")
    f.write("\\begin{table}[htbp]\n")
    f.write("\\centering\n")
    f.write("\\caption{Comparaci\\'{o}n temporal de matrices de transici\\'{o}n: "
            "2019 vs 2024 (DirichletMultinomial). "
            "Se reporta la media posterior con intervalo de credibilidad al 95\\%.}\n")
    f.write("\\label{tab:temporal_comparison_dm}\n")
    f.write("\\small\n")
    f.write("\\begin{tabular}{llcccc}\n")
    f.write("\\toprule\n")
    f.write("Origen & Destino & 2019 & 2024 & $\\Delta$ & Sign. \\\\\n")
    f.write("\\midrule\n")

    for i, orig in enumerate(ORIGIN_LBLS):
        for j, dest in enumerate(DEST_LBLS):
            l19 = T_2019_lower[i,j]; u19 = T_2019_upper[i,j]
            l24 = T_2024_lower[i,j]; u24 = T_2024_upper[i,j]
            sig = "$^*$" if (l24 > u19 or u24 < l19) else ""
            delta = T_2024[i,j] - T_2019[i,j]
            delta_str = f"{delta:+.3f}"
            f.write(f"\\textsc{{{orig}}} & {dest} & "
                    f"{fmt_ci(T_2019[i,j], l19, u19)} & "
                    f"{fmt_ci(T_2024[i,j], l24, u24)} & "
                    f"{delta_str} & {sig} \\\\\n")
        if i < len(ORIGIN_LBLS) - 1:
            f.write("\\midrule\n")

    f.write("\\bottomrule\n")
    f.write("\\end{tabular}\n")
    f.write("\\begin{tablenotes}\\small\n")
    f.write("\\item $^*$ Los intervalos de credibilidad al 95\\% no se solapan.\n")
    f.write("\\item Modelo: inferencia ecol\\'{o}gica bayesiana con "
            "verosimilitud DirichletMultinomial.\n")
    f.write("\\end{tablenotes}\n")
    f.write("\\end{table}\n")

logger.info("Tabla LaTeX guardada: %s", latex_path)

print(f"\nArchivos generados:")
print(f"  {csv_2019}")
print(f"  {csv_comp}")
print(f"  {latex_path}")
print("\nDone.")
