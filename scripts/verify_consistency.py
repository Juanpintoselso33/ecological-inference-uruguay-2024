"""
Comprehensive data consistency verification for electoral inference project.
Checks all numbers in LaTeX report against source data and CSV tables.
"""

import pandas as pd
import numpy as np
import re
from pathlib import Path

# Paths
BASE_DIR = Path("E:/Proyectos VS CODE/Eco inference 2024")
DATA_DIR = BASE_DIR / "data/processed"
TABLES_DIR = BASE_DIR / "outputs/tables"
LATEX_DIR = BASE_DIR / "reports/latex/sections"

# Load processed data
print("=" * 80)
print("LOADING DATA")
print("=" * 80)

df_2024 = pd.read_parquet(DATA_DIR / "circuitos_merged.parquet")
print(f"\n2024 Dataset: {len(df_2024)} circuits, {df_2024.shape[1]} columns")

# Key statistics from raw data
print("\n" + "=" * 80)
print("RAW DATA STATISTICS (2024)")
print("=" * 80)

ca_primera_total = df_2024['ca_primera'].sum()
fa_primera_total = df_2024['fa_primera'].sum()
pn_primera_total = df_2024['pn_primera'].sum()
pc_primera_total = df_2024['pc_primera'].sum()
pi_primera_total = df_2024['pi_primera'].sum()
otros_primera_total = df_2024['otros_primera'].sum()
total_primera = df_2024['total_primera'].sum()

fa_ballotage_total = df_2024['fa_ballotage'].sum()
pn_ballotage_total = df_2024['pn_ballotage'].sum()
blancos_ballotage_total = df_2024['blancos_ballotage'].sum()
total_ballotage = df_2024['total_ballotage'].sum()

print(f"\nFirst Round Votes:")
print(f"  CA: {ca_primera_total:,}")
print(f"  FA: {fa_primera_total:,}")
print(f"  PN: {pn_primera_total:,}")
print(f"  PC: {pc_primera_total:,}")
print(f"  PI: {pi_primera_total:,}")
print(f"  OTROS: {otros_primera_total:,}")
print(f"  TOTAL: {total_primera:,}")

print(f"\nBallotage Votes:")
print(f"  FA: {fa_ballotage_total:,}")
print(f"  PN: {pn_ballotage_total:,}")
print(f"  Blancos: {blancos_ballotage_total:,}")
print(f"  TOTAL: {total_ballotage:,}")

# Load national transition matrix
print("\n" + "=" * 80)
print("NATIONAL TRANSITION MATRIX (2024)")
print("=" * 80)

df_national = pd.read_csv(TABLES_DIR / "national_transition_matrix_final.csv", index_col=0)
df_national_ci = pd.read_csv(TABLES_DIR / "national_transitions_with_ci_final.csv")

print("\nTransition probabilities:")
for _, row in df_national_ci.iterrows():
    origin = row['origin']
    destination = row['destination']
    prob = row['probability']
    ci_lower = row['ci_95_lower']
    ci_upper = row['ci_95_upper']
    print(f"  {origin} -> {destination}: {prob*100:.1f}% [{ci_lower*100:.1f}%, {ci_upper*100:.1f}%]")

# Key numbers to verify in LaTeX
ca_to_fa = df_national_ci[(df_national_ci['origin'] == 'CA') & (df_national_ci['destination'] == 'FA')]['probability'].values[0]
ca_to_pn = df_national_ci[(df_national_ci['origin'] == 'CA') & (df_national_ci['destination'] == 'PN')]['probability'].values[0]
ca_to_fa_lower = df_national_ci[(df_national_ci['origin'] == 'CA') & (df_national_ci['destination'] == 'FA')]['ci_95_lower'].values[0]
ca_to_fa_upper = df_national_ci[(df_national_ci['origin'] == 'CA') & (df_national_ci['destination'] == 'FA')]['ci_95_upper'].values[0]

pc_to_fa = df_national_ci[(df_national_ci['origin'] == 'PC') & (df_national_ci['destination'] == 'FA')]['probability'].values[0]
pc_to_pn = df_national_ci[(df_national_ci['origin'] == 'PC') & (df_national_ci['destination'] == 'PN')]['probability'].values[0]

pn_to_fa = df_national_ci[(df_national_ci['origin'] == 'PN') & (df_national_ci['destination'] == 'FA')]['probability'].values[0]
pn_to_pn = df_national_ci[(df_national_ci['origin'] == 'PN') & (df_national_ci['destination'] == 'PN')]['probability'].values[0]

print("\n" + "=" * 80)
print("KEY FINDINGS - CORRECT VALUES")
print("=" * 80)

print(f"\nCA Defection:")
print(f"  Rate: {ca_to_fa*100:.1f}% to FA, {ca_to_pn*100:.1f}% to PN")
print(f"  95% CI: [{ca_to_fa_lower*100:.1f}%, {ca_to_fa_upper*100:.1f}%]")
print(f"  WARNING  LaTeX reports: 50.8% - CSV shows: {ca_to_fa*100:.1f}%")

print(f"\nPC Loyalty:")
print(f"  Rate: {pc_to_pn*100:.1f}% to PN, {pc_to_fa*100:.1f}% to FA")
print(f"  WARNING  LaTeX reports: 86.7% - CSV shows: {pc_to_pn*100:.1f}%")

print(f"\nPN Loyalty:")
print(f"  Rate: {pn_to_pn*100:.1f}% retention, {pn_to_fa*100:.1f}% to FA")
print(f"  WARNING  LaTeX reports: 90.3% - CSV shows: {pn_to_pn*100:.1f}%")

# Load departmental data
print("\n" + "=" * 80)
print("DEPARTMENTAL ANALYSIS")
print("=" * 80)

df_dept = pd.read_csv(TABLES_DIR / "transfers_by_department_with_pi.csv")
print(f"\nLoaded data for {len(df_dept)} departments")

# Check extreme cases
rio_negro = df_dept[df_dept['departamento'] == 'Río Negro']
tacuarembo = df_dept[df_dept['departamento'] == 'Tacuarembó']
montevideo = df_dept[df_dept['departamento'] == 'Montevideo']

print(f"\nRio Negro CA->FA: {rio_negro['ca_to_fa'].values[0]*100:.1f}% [{rio_negro['ca_to_fa_lower'].values[0]*100:.1f}%, {rio_negro['ca_to_fa_upper'].values[0]*100:.1f}%]")
print(f"  WARNING: LaTeX reports: 95.6% - should be {rio_negro['ca_to_fa'].values[0]*100:.1f}%")

print(f"\nTacuarembo CA->FA: {tacuarembo['ca_to_fa'].values[0]*100:.1f}% [{tacuarembo['ca_to_fa_lower'].values[0]*100:.1f}%, {tacuarembo['ca_to_fa_upper'].values[0]*100:.1f}%]")
print(f"  WARNING: LaTeX reports: 6.0% - should be {tacuarembo['ca_to_fa'].values[0]*100:.1f}%")

print(f"\nMontevideo CA->FA: {montevideo['ca_to_fa'].values[0]*100:.1f}% [{montevideo['ca_to_fa_lower'].values[0]*100:.1f}%, {montevideo['ca_to_fa_upper'].values[0]*100:.1f}%]")
print(f"  CA votes: {int(montevideo['ca_votes'].values[0]):,}")
print(f"  WARNING: LaTeX reports: 60.6% and 21,066 votes - CSV shows: {montevideo['ca_to_fa'].values[0]*100:.1f}% and {int(montevideo['ca_votes'].values[0]):,}")

# Urban-rural comparison
print("\n" + "=" * 80)
print("URBAN-RURAL COMPARISON")
print("=" * 80)

df_urban_rural = pd.read_csv(TABLES_DIR / "urban_rural_comparison.csv")
print("\nCA defection by urbanization:")
for _, row in df_urban_rural.iterrows():
    stratum = row['stratum']
    ca_to_fa_rate = row['ca_to_fa'] * 100
    ca_to_fa_lower = row['ca_to_fa_ci_lower'] * 100
    ca_to_fa_upper = row['ca_to_fa_ci_upper'] * 100
    print(f"  {stratum}: {ca_to_fa_rate:.1f}% [{ca_to_fa_lower:.1f}%, {ca_to_fa_upper:.1f}%]")

urban_ca = df_urban_rural[df_urban_rural['stratum'] == 'urban']['ca_to_fa'].values[0] * 100
rural_ca = df_urban_rural[df_urban_rural['stratum'] == 'rural']['ca_to_fa'].values[0] * 100
diff = rural_ca - urban_ca
print(f"\nDifference: {diff:.1f} pp")
print(f"  WARNING  LaTeX reports: 13.3 pp - CSV shows: {diff:.1f} pp")

# Region comparison
print("\n" + "=" * 80)
print("METROPOLITAN-INTERIOR COMPARISON")
print("=" * 80)

df_region = pd.read_csv(TABLES_DIR / "region_comparison.csv")
print("\nCA defection by region:")
for _, row in df_region.iterrows():
    region = row['region']
    ca_to_fa_rate = row['ca_to_fa_mean'] * 100
    ca_to_fa_lower = row['ca_to_fa_lower'] * 100
    ca_to_fa_upper = row['ca_to_fa_upper'] * 100
    print(f"  {region}: {ca_to_fa_rate:.1f}% [{ca_to_fa_lower:.1f}%, {ca_to_fa_upper:.1f}%]")

metro_ca = df_region[df_region['region'] == 'Area_Metropolitana']['ca_to_fa_mean'].values[0] * 100
interior_ca = df_region[df_region['region'] == 'Interior']['ca_to_fa_mean'].values[0] * 100
diff_region = interior_ca - metro_ca
print(f"\nDifference: {diff_region:.1f} pp")
print(f"  WARNING  LaTeX reports: 8.1 pp - CSV shows: {diff_region:.1f} pp")

# Temporal comparison
print("\n" + "=" * 80)
print("TEMPORAL COMPARISON (2019 vs 2024)")
print("=" * 80)

df_comparison = pd.read_csv(TABLES_DIR / "comparison_2019_2024.csv")
print("\nDefection rate changes:")
for _, row in df_comparison.iterrows():
    partido = row['Partido']
    def_2019 = float(row['Defeccion_2019_%'])
    def_2024 = float(row['Defeccion_2024_%'])
    cambio = float(row['Cambio_pp'])
    print(f"  {partido}: {def_2019:.1f}% (2019) -> {def_2024:.1f}% (2024) | Change: {cambio:.1f} pp")

# Check for inconsistencies
print("\n" + "=" * 80)
print("CRITICAL INCONSISTENCIES DETECTED")
print("=" * 80)

inconsistencies = []

# 1. National CA defection
if abs(ca_to_fa * 100 - 50.8) > 0.5:
    inconsistencies.append(f"INCONSISTENT: CA national defection: LaTeX says 50.8%, CSV shows {ca_to_fa*100:.1f}%")
else:
    print("OK: CA national defection rate: CONSISTENT")

# 2. PC loyalty
if abs(pc_to_pn * 100 - 86.7) > 0.5:
    inconsistencies.append(f"INCONSISTENT: PC loyalty: LaTeX says 86.7%, CSV shows {pc_to_pn*100:.1f}%")
else:
    print("OK: PC loyalty rate: CONSISTENT")

# 3. PN retention
if abs(pn_to_pn * 100 - 90.3) > 0.5:
    inconsistencies.append(f"INCONSISTENT: PN retention: LaTeX says 90.3%, CSV shows {pn_to_pn*100:.1f}%")
else:
    print("OK: PN retention rate: CONSISTENT")

# 4. Circuit count
if len(df_2024) != 7271:
    inconsistencies.append(f"INCONSISTENT: Circuit count: LaTeX says 7,271, data shows {len(df_2024)}")
else:
    print("OK: Circuit count: CONSISTENT")

# 5. CA first round votes
if abs(ca_primera_total - 59412) > 100:
    inconsistencies.append(f"INCONSISTENT: CA first round votes: LaTeX says 59,412, data shows {ca_primera_total:,}")
else:
    print("OK: CA first round votes: CONSISTENT")

# 6. Urban-rural gap
if abs(diff - 13.3) > 0.5:
    inconsistencies.append(f"INCONSISTENT: Urban-rural CA gap: LaTeX says 13.3 pp, CSV shows {diff:.1f} pp")
else:
    print("OK: Urban-rural gap: CONSISTENT")

# Print all inconsistencies
if inconsistencies:
    print("\n" + "=" * 80)
    print("SUMMARY OF INCONSISTENCIES")
    print("=" * 80)
    for inc in inconsistencies:
        print(inc)
else:
    print("\n" + "=" * 80)
    print("OK: ALL KEY STATISTICS ARE CONSISTENT!")
    print("=" * 80)

print("\n" + "=" * 80)
print("VERIFICATION COMPLETE")
print("=" * 80)
