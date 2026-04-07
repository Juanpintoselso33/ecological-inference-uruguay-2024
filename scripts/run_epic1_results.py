"""
EPIC 1 Results: Compare Normal vs DirichletMultinomial on Uruguay 2024 data.

Ejecutar: conda run -n ds python scripts/run_epic1_results.py

Fases:
1. Diagnosticos deterministicos (bounds, leverage) - instantaneo
2. Modelo DM en muestra de 300 circuitos  - ~5 min preview
3. Comparacion con Normal existente
"""
import sys
import pickle
import warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.diagnostics.bounds import compute_duncan_davis_bounds, bounds_to_dataframe
from src.diagnostics.leverage import compute_circuit_leverage
from src.models.king_ei import KingEI

ORIGIN = ['ca_primera', 'fa_primera', 'pc_primera', 'pn_primera', 'pi_primera', 'otros_primera']
DEST   = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']
ORIGIN_LABELS = ['CA', 'FA', 'PC', 'PN', 'PI', 'OTROS']
DEST_LABELS   = ['FA', 'PN', 'BLANCOS']

# ─── Cargar datos ────────────────────────────────────────────────────────────
print("\n" + "="*65)
print("EPIC 1: Resultados con DirichletMultinomial")
print("="*65)

df = pd.read_parquet('data/processed/circuitos_merged.parquet')
print(f"\nDatos: {len(df):,} circuitos, {df.departamento.nunique()} departamentos")

# ─── 1. Duncan-Davis Bounds (deterministico, instantaneo) ────────────────────
print("\n" + "─"*65)
print("1. COTAS DUNCAN-DAVIS (deterministicas, todos los circuitos)")
print("─"*65)

bounds = compute_duncan_davis_bounds(df, ORIGIN, DEST, 'total_primera', 'total_ballotage')
bdf = bounds_to_dataframe(bounds, ORIGIN, DEST)

# Focus: CA -> FA bounds (the key transition)
ca_bounds = bdf[bdf.origin == 'ca_primera'].copy()
ca_bounds['origin_label'] = ORIGIN_LABELS[0]
fa_bounds = bdf[(bdf.destination == 'fa_ballotage')].copy()

print("\nCotas para transicion -> FA_ballotage (media sobre circuitos):")
print(f"{'Origen':<12} {'Cota inferior':>15} {'Cota superior':>15} {'Rango':>10}")
print("-"*52)
for _, row in fa_bounds.iterrows():
    label = row.origin.replace('_primera', '').upper()
    rng = row.upper_mean - row.lower_mean
    print(f"{label:<12} {row.lower_mean:>14.3f}  {row.upper_mean:>14.3f}  {rng:>9.3f}")

# ─── 2. Leverage ─────────────────────────────────────────────────────────────
print("\n" + "─"*65)
print("2. LEVERAGE (influencia por circuito)")
print("─"*65)

lev = compute_circuit_leverage(df, ORIGIN, 'total_primera')
n_high = lev.high_leverage.sum()
print(f"\nCircuitos con leverage alto (>2*media): {n_high:,} / {len(df):,} ({n_high/len(df)*100:.1f}%)")
print(f"Leverage media: {lev.leverage.mean():.4f}  |  max: {lev.leverage.max():.4f}")

top10 = df[['departamento', 'total_primera']].copy()
top10['leverage'] = lev.leverage.values
top10 = top10.nlargest(10, 'leverage')
print("\nTop 10 circuitos por leverage:")
for _, r in top10.iterrows():
    print(f"  {r.departamento:<15}  votos_1a={r.total_primera:>5.0f}  leverage={r.leverage:.4f}")

# ─── 3. Modelo Normal existente ──────────────────────────────────────────────
print("\n" + "─"*65)
print("3. MODELO EXISTENTE (Normal likelihood, full 7271 circuitos)")
print("─"*65)

try:
    with open('outputs/results/national_transfers_with_pi.pkl', 'rb') as f:
        old_model = pickle.load(f)

    if hasattr(old_model, 'get_transition_matrix'):
        T_normal = old_model.get_transition_matrix()
        ci_normal = old_model.get_credible_intervals(prob=0.95)
        print(f"\nMatriz de transicion T (Normal) con IC 95%:")
        print(f"\n{'Origen':<8}", end="")
        for d in DEST_LABELS:
            print(f"  {d:>8}", end="")
        print()
        print("-"*38)
        for i, label in enumerate(ORIGIN_LABELS):
            print(f"{label:<8}", end="")
            for j in range(len(DEST_LABELS)):
                print(f"  {T_normal[i,j]:>7.3f}", end="")
            print()
        print()
        print(f"CA->FA: {T_normal[0,0]:.3f} [{ci_normal['lower'][0,0]:.3f}, {ci_normal['upper'][0,0]:.3f}]")
    else:
        print("(formato pkl no soportado directamente)")
        T_normal = None
except Exception as e:
    print(f"Error cargando pkl: {e}")
    T_normal = None

# ─── 4. Modelo DirichletMultinomial (muestra rapida) ─────────────────────────
print("\n" + "─"*65)
print("4. MODELO DirichletMultinomial (muestra 300 circuitos, 500 samples)")
print("   ~5-10 min  ...  iniciando")
print("─"*65)

rng = np.random.default_rng(42)
sample_idx = rng.choice(len(df), size=300, replace=False)
df_sample = df.iloc[sample_idx].reset_index(drop=True)

model_dm = KingEI(
    num_samples=500,
    num_chains=2,
    num_warmup=300,
    target_accept=0.9,
    random_seed=42,
    likelihood='dirichlet_multinomial',
)
model_dm.fit(
    df_sample,
    origin_cols=ORIGIN,
    destination_cols=DEST,
    total_origin='total_primera',
    total_destination='total_ballotage',
    progressbar=True,
)

T_dm = model_dm.get_transition_matrix()
ci_dm = model_dm.get_credible_intervals(prob=0.95)

print(f"\nMatriz de transicion T (DirichletMultinomial, n=300 circuitos):")
print(f"\n{'Origen':<8}", end="")
for d in DEST_LABELS:
    print(f"  {d:>8}", end="")
print()
print("-"*38)
for i, label in enumerate(ORIGIN_LABELS):
    print(f"{label:<8}", end="")
    for j in range(len(DEST_LABELS)):
        print(f"  {T_dm[i,j]:>7.3f}", end="")
    print()

print()
print(f"CA->FA: {T_dm[0,0]:.3f} [{ci_dm['lower'][0,0]:.3f}, {ci_dm['upper'][0,0]:.3f}]")

# ─── 5. Comparacion lado a lado ──────────────────────────────────────────────
if T_normal is not None:
    print("\n" + "─"*65)
    print("5. COMPARACION Normal vs DirichletMultinomial (CA->FA)")
    print("─"*65)
    print(f"\n{'Modelo':<35} {'CA->FA':>8} {'CA->PN':>8}")
    print("-"*55)
    print(f"{'Normal (7271 circuitos)':<35} {T_normal[0,0]:>8.3f} {T_normal[0,1]:>8.3f}")
    print(f"{'DM (300 circuitos muestra)':<35} {T_dm[0,0]:>8.3f} {T_dm[0,1]:>8.3f}")
    delta_fa = T_dm[0,0] - T_normal[0,0]
    print(f"\nDelta CA->FA (DM - Normal): {delta_fa:+.3f}")

# ─── Guardar resultados ───────────────────────────────────────────────────────
out_path = Path('outputs/tables/epic1_dm_preview_results.csv')
rows = []
for i, orig in enumerate(ORIGIN_LABELS):
    for j, dest in enumerate(DEST_LABELS):
        rows.append({
            'origin': orig, 'destination': dest,
            'T_normal': T_normal[i,j] if T_normal is not None else None,
            'T_dm_preview': T_dm[i,j],
            'dm_lower95': ci_dm['lower'][i,j],
            'dm_upper95': ci_dm['upper'][i,j],
        })
pd.DataFrame(rows).to_csv(out_path, index=False)
print(f"\nResultados guardados: {out_path}")

# Guardar bounds
bounds_path = Path('outputs/tables/duncan_davis_bounds_national.csv')
bdf.to_csv(bounds_path, index=False)
print(f"Cotas guardadas: {bounds_path}")

print("\n" + "="*65)
print("EPIC 1 completado.")
print("Para run completo de produccion (4000 samples, 7271 circuitos):")
print("  conda run -n ds python scripts/compare_eiCircles.py")
print("="*65)
