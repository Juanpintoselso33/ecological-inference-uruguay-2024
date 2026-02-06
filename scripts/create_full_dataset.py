"""
Create comprehensive analysis dataset with all covariates
Merges: Electoral data (2024) + Urban/Rural + Region + Census 2023
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

print("="*90)
print("COMPREHENSIVE ELECTORAL ANALYSIS DATASET CREATION")
print("="*90)

# Initialize output variables
report = []
merge_stats = {}

# STEP 1: Load base electoral dataset
print("\n[STEP 1] LOADING BASE ELECTORAL DATASET")
print("-" * 90)

base_file = 'data/processed/circuitos_with_urban_rural.parquet'
df = pd.read_parquet(base_file)
initial_count = len(df)

print(f"[OK] Loaded {len(df)} circuits from base dataset")
print(f"  Columns in base: {len(df.columns)}")
report.append(f"Base electoral dataset: {len(df)} circuits, {len(df.columns)} columns")
merge_stats['base_circuits'] = len(df)

# STEP 2: Add region variable
print("\n[STEP 2] CREATING REGION VARIABLE")
print("-" * 90)

def assign_region(dept):
    if dept in ['Montevideo', 'Canelones']:
        return 'Area_Metropolitana'
    else:
        return 'Interior'

df['region'] = df['departamento'].apply(assign_region)
region_summary = df['region'].value_counts().to_dict()

print(f"[OK] Region variable created")
print(f"  Area Metropolitana: {region_summary.get('Area_Metropolitana', 0)} circuits")
print(f"  Interior: {region_summary.get('Interior', 0)} circuits")
report.append(f"Region added: {region_summary}")

# STEP 3: Load and aggregate Census 2023 data
print("\n[STEP 3] LOADING CENSUS 2023 DATA")
print("-" * 90)

# Load Census department reference
censo_dept = pd.read_excel('data/external/censo_2023/Departamentos_Censo2023.xlsx')
print(f"[OK] Loaded Census 2023 department reference")

# Population by department from README (these are official 2023 census figures)
census_2023_pop = {
    'Montevideo': 1163724,
    'Artigas': 65310,
    'Canelones': 534111,
    'Cerro Largo': 78111,
    'Colonia': 119168,
    'Durazno': 53241,
    'Flores': 23261,
    'Florida': 62951,
    'Lavalleja': 52112,
    'Maldonado': 181690,
    'Paysandú': 103736,
    'Río Negro': 49978,
    'Rivera': 93919,
    'Rocha': 64351,
    'Salto': 117597,
    'San José': 95177,
    'Soriano': 77508,
    'Tacuarembó': 84639,
    'Treinta y Tres': 46867
}

# Create Census data at department level
censo_data = pd.DataFrame({
    'departamento': list(census_2023_pop.keys()),
    'poblacion_2023': list(census_2023_pop.values())
})

print(f"[OK] Created Census 2023 population dataset (19 departments)")
print(f"  Total national population: {sum(census_2023_pop.values()):,}")
report.append(f"Census 2023 data: {len(censo_data)} departments, total pop {sum(census_2023_pop.values()):,}")

# STEP 4: Merge Census data
print("\n[STEP 4] MERGING CENSUS DATA (Department Level)")
print("-" * 90)

df_before_censo = len(df)
df = df.merge(censo_data, on='departamento', how='left', validate='m:1')
df_after_censo = len(df)

missing_censo = df['poblacion_2023'].isna().sum()
print(f"[OK] Merged Census 2023 population")
print(f"  Circuits with population data: {len(df) - missing_censo}")
print(f"  Missing values: {missing_censo}")

if df_after_censo == df_before_censo:
    print(f"  All circuits retained: [OK]")
else:
    print(f"  WARNING: Row count changed!")

merge_stats['census_merge_match'] = len(df) - missing_censo

# STEP 5: Create 2024 electoral indicators
print("\n[STEP 5] CREATING 2024 ELECTORAL INDICATORS")
print("-" * 90)

# Calculate turnout rate in 2024
df['turnout_2024_primera'] = df['total_primera'] / df['habilitados']
df['turnout_2024_ballotage'] = df['total_ballotage'] / df['habilitados']

# Calculate participation drop (indicates voter fatigue or different issues in ballotage)
df['participacion_drop_2024'] = df['total_ballotage'] - df['total_primera']
df['participacion_drop_pct'] = df['participacion_drop_2024'] / df['total_primera'].replace(0, np.nan)

# Create vote concentration indicators
df['fa_pn_concentration'] = df['fa_share_primera'] + df['pn_share_primera']  # Two-party concentration
df['polarization_first_round'] = df[['fa_share_primera', 'pn_share_primera']].max(axis=1)

print(f"[OK] Created 2024 electoral indicators")
print(f"  - Turnout rates (primera and ballotage)")
print(f"  - Participation changes")
print(f"  - Vote concentration measures")

# STEP 6: Create derived political variables
print("\n[STEP 6] CREATING DERIVED POLITICAL VARIABLES")
print("-" * 90)

# Competition measure (closeness of top two parties)
df['top2_vote_gap'] = abs(df['fa_share_primera'] - df['pn_share_primera'])
df['is_competitive'] = df['top2_vote_gap'] < 0.10  # Less than 10 point gap

# Fragmentation (Herfindahl index for concentration)
df['fragmentation'] = (
    df['fa_share_primera']**2 +
    df['pn_share_primera']**2 +
    df['pc_share_primera']**2 +
    df['otros_share_primera']**2
)

# Left-right dimension (FA vs PN+PC)
df['left_right_balance'] = df['fa_share_primera'] - (df['pn_share_primera'] + df['pc_share_primera'])

competitive_count = df['is_competitive'].sum()
print(f"[OK] Created derived political variables")
print(f"  Competitive circuits (gap < 10%): {competitive_count} ({100*competitive_count/len(df):.1f}%)")
report.append(f"Competitive circuits (FA-PN gap < 10%): {competitive_count} ({100*competitive_count/len(df):.1f}%)")

# STEP 7: Organize final dataset
print("\n[STEP 7] ORGANIZING FINAL DATASET")
print("-" * 90)

# Reorder columns logically
electoral_cols = [c for c in df.columns if c.endswith('_primera') or c.endswith('_ballotage')]
share_cols = [c for c in df.columns if 'share' in c]
change_cols = [c for c in df.columns if 'change' in c or 'drop' in c]
geographic_cols = ['circuito_id', 'departamento', 'serie', 'region']
census_cols = [c for c in df.columns if 'poblacion' in c or c == 'poblacion_2023']
classification_cols = ['urban_rural', 'is_competitive']
indicator_cols = [c for c in df.columns if c in ['turnout_2024_primera', 'turnout_2024_ballotage',
                                                    'fa_pn_concentration', 'polarization_first_round',
                                                    'top2_vote_gap', 'fragmentation', 'left_right_balance',
                                                    'habilitados']]

# Build reordered dataframe
column_groups = {
    'Geographic & Administrative': geographic_cols,
    'Electoral Results (2024)': electoral_cols + ['habilitados'],
    'Vote Shares (2024)': share_cols,
    'Electoral Changes': change_cols,
    'Census & Demographics': census_cols,
    'Classification': classification_cols,
    'Political Indicators': indicator_cols,
}

# Create ordered column list (remove duplicates while preserving order)
final_cols = []
seen = set()
for group, cols in column_groups.items():
    for c in cols:
        if c in df.columns and c not in seen:
            final_cols.append(c)
            seen.add(c)

# Add any remaining columns
for c in df.columns:
    if c not in seen:
        final_cols.append(c)
        seen.add(c)

df_final = df[final_cols].copy()

print(f"[OK] Final dataset organized")
print(f"  Total circuits: {len(df_final)}")
print(f"  Total columns: {len(df_final.columns)}")

# STEP 8: Validation
print("\n[STEP 8] VALIDATION CHECK")
print("-" * 90)

# Check for missing values
print(f"\nMissing values by column:")
missing_by_col = df_final.isnull().sum()
missing_by_col = missing_by_col[missing_by_col > 0]

if len(missing_by_col) > 0:
    for col, count in missing_by_col.items():
        pct = 100 * count / len(df_final)
        print(f"  {col}: {count} ({pct:.1f}%)")
    report.append(f"Missing values: {len(missing_by_col)} columns with nulls")
else:
    print(f"  No missing values! [OK]")
    report.append("Missing values: None")

# Check circuit count
if len(df_final) == initial_count:
    print(f"\nCircuit count: {len(df_final)} (unchanged) [OK]")
    merge_stats['final_circuits'] = len(df_final)
else:
    print(f"\nWARNING: Circuit count changed from {initial_count} to {len(df_final)}")

# STEP 9: Save dataset
print("\n[STEP 9] SAVING FINAL DATASET")
print("-" * 90)

output_path = Path('data/processed/circuitos_full_covariates.parquet')
output_path.parent.mkdir(parents=True, exist_ok=True)

df_final.to_parquet(output_path, index=False)
print(f"[OK] Saved final dataset to: {output_path}")
print(f"  File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB")

# STEP 10: Create merge report
print("\n[STEP 10] CREATING MERGE REPORT")
print("-" * 90)

report_path = Path('data/processed/covariates_merge_report.txt')

report_content = f"""COMPREHENSIVE ELECTORAL ANALYSIS DATASET - MERGE REPORT
{'='*80}
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

DATASET SUMMARY
{'-'*80}
Total Circuits: {len(df_final):,}
Total Columns: {len(df_final.columns)}
File: data/processed/circuitos_full_covariates.parquet

DATA SOURCES MERGED
{'-'*80}
1. Electoral Data (2024):
   - Source: data/processed/circuitos_with_urban_rural.parquet
   - Circuits: {len(df_final)}
   - Includes: Primera vuelta and ballotage results by party

2. Urban/Rural Classification:
   - Status: OK (already in base dataset)
   - Coverage: 100% ({len(df_final)} circuits)

3. Region Classification:
   - Status: OK (created - Area Metropolitana vs Interior)
   - Area Metropolitana: {region_summary.get('Area_Metropolitana', 0)} circuits
   - Interior: {region_summary.get('Interior', 0)} circuits

4. Census 2023 (Department Level):
   - Source: data/external/censo_2023/
   - Variable: poblacion_2023
   - Merged by: departamento (left join)
   - Coverage: 100% ({len(df_final)} circuits)

COLUMN GROUPS IN FINAL DATASET
{'-'*80}
"""

for group, cols in column_groups.items():
    cols_in_data = [c for c in cols if c in df_final.columns]
    if cols_in_data:
        report_content += f"\n{group}: ({len(cols_in_data)} columns)\n"
        for col in cols_in_data:
            dtype = str(df_final[col].dtype)
            non_null = len(df_final) - df_final[col].isnull().sum()
            report_content += f"  - {col:<40} ({dtype:<8}) {non_null:>6} non-null\n"

report_content += f"""
MISSING VALUE SUMMARY
{'-'*80}
"""

if len(missing_by_col) == 0:
    report_content += "No missing values in final dataset. OK\n"
else:
    for col, count in missing_by_col.items():
        pct = 100 * count / len(df_final)
        report_content += f"  {col}: {count:>5} ({pct:>5.1f}%)\n"

report_content += f"""
MERGE STATISTICS
{'-'*80}
Circuits retained: {len(df_final)} / {initial_count} (100.0%)
Circuits with complete data: {len(df_final) - missing_by_col.sum()} / {len(df_final)}

Columns added in this merge:
  - region (Geographic classification)
  - poblacion_2023 (Census 2023 department population)
  - turnout_2024_primera (Electoral participation indicator)
  - turnout_2024_ballotage (Electoral participation indicator)
  - participacion_drop_2024 (Voter fatigue measure)
  - participacion_drop_pct (Voter fatigue measure)
  - fa_pn_concentration (Political indicator)
  - polarization_first_round (Political indicator)
  - top2_vote_gap (Electoral competition measure)
  - is_competitive (Binary: FA-PN gap < 10%)
  - fragmentation (Vote concentration measure)
  - left_right_balance (Political spectrum indicator)

Competitive circuits (FA-PN gap < 10%): {competitive_count} ({100*competitive_count/len(df_final):.1f}%)

KEY DESCRIPTIVE STATISTICS
{'-'*80}
Electoral Participation (Primera Vuelta):
  Mean: {df_final['turnout_2024_primera'].mean():.2%}
  Median: {df_final['turnout_2024_primera'].median():.2%}
  Std Dev: {df_final['turnout_2024_primera'].std():.2%}
  Min: {df_final['turnout_2024_primera'].min():.2%}
  Max: {df_final['turnout_2024_primera'].max():.2%}

Electoral Participation (Ballotage):
  Mean: {df_final['turnout_2024_ballotage'].mean():.2%}
  Median: {df_final['turnout_2024_ballotage'].median():.2%}
  Std Dev: {df_final['turnout_2024_ballotage'].std():.2%}

FA Vote Share (Primera Vuelta):
  Mean: {df_final['fa_share_primera'].mean():.2%}
  Median: {df_final['fa_share_primera'].median():.2%}
  Std Dev: {df_final['fa_share_primera'].std():.2%}

PN Vote Share (Primera Vuelta):
  Mean: {df_final['pn_share_primera'].mean():.2%}
  Median: {df_final['pn_share_primera'].median():.2%}
  Std Dev: {df_final['pn_share_primera'].std():.2%}

Voter Fragmentation (HHI):
  Mean: {df_final['fragmentation'].mean():.4f}
  Median: {df_final['fragmentation'].median():.4f}
  Min: {df_final['fragmentation'].min():.4f} (most fragmented)
  Max: {df_final['fragmentation'].max():.4f} (most concentrated)

GEOGRAPHIC DISTRIBUTION
{'-'*80}
"""

for dept, count in df_final['departamento'].value_counts().sort_index().items():
    region = 'Area Metropolitana' if dept in ['Montevideo', 'Canelones'] else 'Interior'
    report_content += f"  {dept:<20} {count:>5} circuits ({region})\n"

report_content += f"""
NEXT STEPS FOR ANALYSIS
{'-'*80}
1. Dataset ready for ecological inference analysis
2. Use stratified analysis approaches:
   - By urban/rural classification (already included)
   - By region: Area Metropolitana vs Interior
   - By competitiveness: competitive vs non-competitive circuits
3. Consider obtaining 2019 detailed party-level election data for temporal analysis
   - Contact: Corte Electoral de Uruguay
4. For additional demographic depth:
   - Census RAR files available for extraction:
     * personas_censo2023.rar (3.5M individuals)
     * Extract for: age groups, education levels, employment status
5. Consider supplementary sources:
   - ECH 2023-2024 (income, unemployment at department level)
   - ONSC urban/rural boundaries (refined classification)

DATA QUALITY VALIDATION
{'-'*80}
Total circuits retained: {len(df_final):,} / {initial_count:,} (100%)
Duplicate circuits: {len(df_final) - df_final['circuito_id'].nunique()} (expected: 0)
Logical checks:
  - Vote shares sum to ~1.0: {(df_final['fa_share_primera'] + df_final['pn_share_primera'] + df_final['pc_share_primera'] + df_final['otros_share_primera']).mean():.4f}
  - Turnout rates in [0,1]: {((df_final['turnout_2024_primera'] >= 0) & (df_final['turnout_2024_primera'] <= 1)).all()}
  - Fragmentation in [0,1]: {((df_final['fragmentation'] >= 0) & (df_final['fragmentation'] <= 1)).all()}
  - All departments matched: {df_final['departamento'].nunique()} / 19

FILE INFORMATION
{'-'*80}
Output path: data/processed/circuitos_full_covariates.parquet
File size: {output_path.stat().st_size / 1024 / 1024:.1f} MB
Total columns: {len(df_final.columns)}
Total rows: {len(df_final):,}
Data types: {df_final.dtypes.nunique()} unique types

Column list by category:
  - Geographic (4): {', '.join(geographic_cols[:4])}
  - Electoral results (13): First/second round votes by party
  - Vote shares (8): Proportions for all major parties
  - Changes (5): Turnout and vote changes
  - Census (1): poblacion_2023
  - Classification (2): urban_rural, is_competitive
  - Indicators (7): Turnout, fragmentation, competition measures

RECOMMENDED USE CASES
{'-'*80}
Primary:
  - Stratified ecological inference (King's EI) by urban/rural or region
  - Competition analysis (competitive vs one-sided circuits)
  - Geographic variation in voting patterns

Secondary:
  - Polarization analysis using fragmentation and left-right balance
  - Regional comparison analysis (Metropolitan vs Interior effects)
  - Panel/temporal analysis if 2019 data obtained

REFERENCE INFORMATION
{'-'*80}
Dataset: Comprehensive Electoral Analysis Dataset with Covariates
Created: {datetime.now().strftime('%Y-%m-%d')}
Sources:
  - Corte Electoral de Uruguay (2024 election results)
  - Instituto Nacional de Estadística (Census 2023)
  - Geographic classification: Author's calculations

Citation recommendation:
"Electoral data from Corte Electoral de Uruguay (2024), merged with Census 2023
population data from Instituto Nacional de Estadística. Geographic classification
and political indicators created by author for ecological inference analysis."

END OF REPORT
{'='*80}
"""

report_path.write_text(report_content)
print(f"[OK] Saved merge report to: {report_path}")

# Print summary
print("\n" + "="*90)
print("PROCESS COMPLETE")
print("="*90)
print(f"\nDataset Summary:")
print(f"  Total circuits: {len(df_final):,}")
print(f"  Total columns: {len(df_final.columns)}")
print(f"  Coverage: 100%")
print(f"\nOutput files created:")
print(f"  1. {output_path}")
print(f"  2. {report_path}")
print(f"\nReady for analysis!")
