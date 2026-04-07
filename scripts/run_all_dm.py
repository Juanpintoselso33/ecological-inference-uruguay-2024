"""
Master script: espera resultado 2024 DM → ajusta 2019 DM → comparacion temporal.

Corre en foreground. Si compare_eiCircles.py ya termino, salta directo a 2019.
Si todavia esta corriendo, espera polling cada 30s hasta que aparezca el CSV.

Uso:
    conda run -n ds python scripts/run_all_dm.py
"""
import sys, time, warnings, logging
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
OUT_DIR     = Path('outputs/tables')
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ─── Fase 1: Esperar resultados 2024 ─────────────────────────────────────────
csv_2024 = OUT_DIR / 'comparison_eiCircles_python.csv'

if not csv_2024.exists():
    logger.info("Job 2024 todavia corriendo. Esperando (polling cada 60s)...")
    dots = 0
    while not csv_2024.exists():
        time.sleep(60)
        dots += 1
        logger.info("Esperando... %d min. (%s)", dots, time.strftime('%H:%M:%S'))
    logger.info("Resultados 2024 encontrados. Continuando.")
else:
    logger.info("Resultados 2024 ya existen. Saltando fit 2024.")

# Cargar T_2024
df_2024 = pd.read_csv(csv_2024)
T_2024       = np.zeros((len(ORIGIN_LBLS), len(DEST_LBLS)))
T_2024_lower = np.zeros_like(T_2024)
T_2024_upper = np.zeros_like(T_2024)
for _, row in df_2024.iterrows():
    i = ORIGIN_LBLS.index(row.origin)
    j = DEST_LBLS.index(row.destination)
    T_2024[i, j]       = row['mean']
    T_2024_lower[i, j] = row['lower_95']
    T_2024_upper[i, j] = row['upper_95']

logger.info("T_2024 cargado. CA->FA = %.3f [%.3f, %.3f]",
            T_2024[0,0], T_2024_lower[0,0], T_2024_upper[0,0])

# ─── Fase 2: Ajustar modelo DM 2019 ──────────────────────────────────────────
csv_2019 = OUT_DIR / 'comparison_eiCircles_2019_python.csv'

if csv_2019.exists():
    logger.info("Resultados 2019 ya existen. Cargando.")
    df_2019_res = pd.read_csv(csv_2019)
    T_2019       = np.zeros((len(ORIGIN_LBLS), len(DEST_LBLS)))
    T_2019_lower = np.zeros_like(T_2019)
    T_2019_upper = np.zeros_like(T_2019)
    for _, row in df_2019_res.iterrows():
        i = ORIGIN_LBLS.index(row.origin)
        j = DEST_LBLS.index(row.destination)
        T_2019[i, j]       = row['mean']
        T_2019_lower[i, j] = row['lower_95']
        T_2019_upper[i, j] = row['upper_95']
else:
    df19 = pd.read_parquet('data/processed/circuitos_merged_2019.parquet')
    logger.info("Datos 2019: %d circuitos", len(df19))

    logger.info("Ajustando DirichletMultinomial 2019 (4000 samples, 4 chains)...")
    model_2019 = KingEI(
        num_samples=4000, num_chains=4, num_warmup=2000,
        target_accept=0.9, random_seed=42,
        likelihood='dirichlet_multinomial',
    )
    model_2019.fit(
        df19,
        origin_cols=ORIGIN, destination_cols=DEST,
        total_origin='total_primera', total_destination='total_ballotage',
        progressbar=True,
    )
    T_2019       = model_2019.get_transition_matrix()
    ci_2019      = model_2019.get_credible_intervals(prob=0.95)
    T_2019_lower = ci_2019['lower']
    T_2019_upper = ci_2019['upper']
    diag         = model_2019.get_diagnostics()
    logger.info("2019 R-hat max=%.4f | ESS min=%.0f",
                np.max(diag['rhat']), np.min(diag['ess']))

    rows = []
    for i, orig in enumerate(ORIGIN_LBLS):
        for j, dest in enumerate(DEST_LBLS):
            rows.append({'origin': orig, 'destination': dest,
                         'mean': round(T_2019[i,j], 4),
                         'lower_95': round(T_2019_lower[i,j], 4),
                         'upper_95': round(T_2019_upper[i,j], 4),
                         'model': 'KingEI_DirichletMultinomial_2019'})
    pd.DataFrame(rows).to_csv(csv_2019, index=False)
    logger.info("Guardado: %s", csv_2019)

# ─── Fase 3: Tabla comparativa ────────────────────────────────────────────────
print("\n" + "="*72)
print("COMPARACION TEMPORAL 2019 vs 2024  —  DirichletMultinomial")
print("="*72)

for j_idx, dest_lbl in enumerate(DEST_LBLS):
    print(f"\n--- Destino: {dest_lbl} ---")
    print(f"{'Origen':<8}  {'2019 media':>10}  {'IC95 2019':>17}  "
          f"{'2024 media':>10}  {'IC95 2024':>17}  {'Δ':>7}")
    print("-"*78)
    for i, lbl in enumerate(ORIGIN_LBLS):
        d19 = T_2019[i,j_idx];  d24 = T_2024[i,j_idx]
        l19 = T_2019_lower[i,j_idx]; u19 = T_2019_upper[i,j_idx]
        l24 = T_2024_lower[i,j_idx]; u24 = T_2024_upper[i,j_idx]
        delta = d24 - d19
        sig = " *" if (l24 > u19 or u24 < l19) else "  "
        print(f"{lbl:<8}  {d19:>10.3f}  [{l19:.3f},{u19:.3f}]  "
              f"{d24:>10.3f}  [{l24:.3f},{u24:.3f}]  {delta:>+6.3f}{sig}")
print("\n * IC95 no solapan => cambio estadisticamente distinguible")

# ─── Fase 4: CSV completo ─────────────────────────────────────────────────────
rows_comp = []
for i, orig in enumerate(ORIGIN_LBLS):
    for j, dest in enumerate(DEST_LBLS):
        rows_comp.append({
            'origin': orig, 'destination': dest,
            'T_2019': round(T_2019[i,j], 4),
            'lower_2019': round(T_2019_lower[i,j], 4),
            'upper_2019': round(T_2019_upper[i,j], 4),
            'T_2024': round(T_2024[i,j], 4),
            'lower_2024': round(T_2024_lower[i,j], 4),
            'upper_2024': round(T_2024_upper[i,j], 4),
            'delta': round(T_2024[i,j] - T_2019[i,j], 4),
        })
df_comp = pd.DataFrame(rows_comp)
csv_comp = OUT_DIR / 'temporal_comparison_dm_2019_2024.csv'
df_comp.to_csv(csv_comp, index=False)
logger.info("CSV comparativo: %s", csv_comp)

# ─── Fase 5: LaTeX ────────────────────────────────────────────────────────────
def fmt_ci(mean, lower, upper):
    return f"{mean:.3f} [{lower:.3f}, {upper:.3f}]"

latex_path = Path('outputs/tables/latex/temporal_comparison_dm.tex')
latex_path.parent.mkdir(parents=True, exist_ok=True)

with open(latex_path, 'w', encoding='utf-8') as f:
    f.write("% Tabla comparacion temporal 2019 vs 2024 (DirichletMultinomial)\n")
    f.write("% Generada por scripts/run_all_dm.py\n\n")
    f.write("\\begin{table}[htbp]\n\\centering\n")
    f.write("\\caption{Comparaci\\'{o}n de matrices de transici\\'{o}n electoral: "
            "2019 vs.\\ 2024 (verosimilitud DirichletMultinomial). "
            "Se reporta la media posterior con intervalo de credibilidad al 95\\%.}\n")
    f.write("\\label{tab:temporal_comparison_dm}\n\\small\n")
    f.write("\\begin{tabular}{llcccc}\n\\toprule\n")
    f.write("Origen & Destino & 2019 & 2024 & $\\Delta$ (pp) & \\\\\n\\midrule\n")

    for i, orig in enumerate(ORIGIN_LBLS):
        for j, dest in enumerate(DEST_LBLS):
            l19 = T_2019_lower[i,j]; u19 = T_2019_upper[i,j]
            l24 = T_2024_lower[i,j]; u24 = T_2024_upper[i,j]
            sig = "$^*$" if (l24 > u19 or u24 < l19) else ""
            delta = T_2024[i,j] - T_2019[i,j]
            f.write(f"\\textsc{{{orig}}} & {dest} & "
                    f"{fmt_ci(T_2019[i,j], l19, u19)} & "
                    f"{fmt_ci(T_2024[i,j], l24, u24)} & "
                    f"{delta:+.3f} & {sig} \\\\\n")
        if i < len(ORIGIN_LBLS) - 1:
            f.write("\\midrule\n")

    f.write("\\bottomrule\n\\end{tabular}\n")
    f.write("\\begin{tablenotes}\\small\n")
    f.write("\\item $^*$ Los intervalos de credibilidad al 95\\% no se solapan.\n")
    f.write("\\item Modelo: inferencia ecol\\'{o}gica bayesiana, "
            "verosimilitud DirichletMultinomial, 4{,}000 muestras por cadena.\n")
    f.write("\\end{tablenotes}\n\\end{table}\n")

logger.info("LaTeX: %s", latex_path)

print(f"\n{'='*72}")
print("COMPLETADO. Archivos generados:")
print(f"  {csv_2019}")
print(f"  {csv_comp}")
print(f"  {latex_path}")
print(f"{'='*72}")
