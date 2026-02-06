"""
Process 2019 election data to match 2024 structure.

This script processes the 2019 primera vuelta and ballotage data from the
Corte Electoral Uruguay and creates a merged dataset with the same structure
as the 2024 data for comparative analysis.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Paths
BASE_DIR = Path(__file__).parent.parent
RAW_DIR = BASE_DIR / "data" / "raw" / "2019"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

# Ensure processed directory exists
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def load_primera_vuelta_2019():
    """Load and process primera vuelta 2019 data."""
    logger.info("Loading primera vuelta 2019...")

    file_path = RAW_DIR / "primera_vuelta_2019.xlsx"
    df = pd.read_excel(file_path, sheet_name='Datos')

    logger.info(f"Loaded {len(df)} rows from primera vuelta 2019")
    logger.info(f"Unique parties: {df['Lema'].nunique()}")
    logger.info(f"Unique circuits: {df['CRV'].nunique()}")

    return df


def aggregate_primera_vuelta(df):
    """Aggregate primera vuelta by circuit and party."""
    logger.info("Aggregating primera vuelta by circuit and party...")

    # Aggregate votes by Department, CRV, and Lema
    agg = df.groupby(['Departamento', 'CRV', 'Lema'])['CantidadVotos'].sum().reset_index()

    # Pivot to wide format
    pivot = agg.pivot_table(
        index=['Departamento', 'CRV'],
        columns='Lema',
        values='CantidadVotos',
        fill_value=0
    ).reset_index()

    # Rename columns to match our convention
    column_mapping = {
        'Partido Frente Amplio': 'fa_primera',
        'Partido Nacional': 'pn_primera',
        'Partido Colorado': 'pc_primera',
        'Partido Cabildo Abierto': 'ca_primera',
        'Partido Independiente': 'pi_primera',
    }

    # Rename columns that exist
    for old_name, new_name in column_mapping.items():
        if old_name in pivot.columns:
            pivot.rename(columns={old_name: new_name}, inplace=True)

    # Aggregate all other parties as "otros"
    otros_parties = [
        'Partido Asamblea Popular',
        'Partido de los Trabajadores',
        'Partido Ecologista Radical Intransigente',
        'Partido de la Gente',
        'Partido Verde Animalista',
        'Partido Digital'
    ]

    otros_cols = [col for col in pivot.columns if col in otros_parties]
    if otros_cols:
        pivot['otros_primera'] = pivot[otros_cols].sum(axis=1)
        pivot.drop(columns=otros_cols, inplace=True)
    else:
        pivot['otros_primera'] = 0

    # Calculate total primera vuelta
    vote_cols = [col for col in pivot.columns if col.endswith('_primera')]
    pivot['total_primera'] = pivot[vote_cols].sum(axis=1)

    logger.info(f"Aggregated to {len(pivot)} circuits")
    logger.info(f"Columns: {pivot.columns.tolist()}")

    return pivot


def load_ballotage_2019():
    """Load and process ballotage 2019 data."""
    logger.info("Loading ballotage 2019...")

    file_path = RAW_DIR / "ballotage_2019.xlsx"
    df = pd.read_excel(file_path, sheet_name='Datos')

    logger.info(f"Loaded {len(df)} rows from ballotage 2019")
    logger.info(f"Unique circuits: {df['CRV'].nunique()}")

    return df


def process_ballotage(df):
    """Process ballotage data to match our structure."""
    logger.info("Processing ballotage 2019...")

    # Rename columns
    df_bal = df[['Departamento', 'CRV', 'Total_Martinez_Villar',
                 'Total_Lacalle Pou_Argimon', 'Total_EN_Blanco']].copy()

    df_bal.rename(columns={
        'Total_Martinez_Villar': 'fa_ballotage',  # FA candidate
        'Total_Lacalle Pou_Argimon': 'pn_ballotage',  # PN candidate (coalition)
        'Total_EN_Blanco': 'blancos_ballotage'
    }, inplace=True)

    # Calculate total ballotage
    df_bal['total_ballotage'] = (
        df_bal['fa_ballotage'] +
        df_bal['pn_ballotage'] +
        df_bal['blancos_ballotage']
    )

    logger.info(f"Processed {len(df_bal)} circuits")

    return df_bal


def merge_datasets(df_primera, df_ballotage):
    """Merge primera vuelta and ballotage datasets."""
    logger.info("Merging datasets...")

    # Merge on Department and CRV
    merged = pd.merge(
        df_primera,
        df_ballotage,
        on=['Departamento', 'CRV'],
        how='inner'
    )

    logger.info(f"Merged dataset has {len(merged)} circuits")

    # Create circuit_id (Department + CRV)
    merged['circuito_id'] = merged['Departamento'] + '_' + merged['CRV'].astype(str)
    merged['departamento'] = merged['Departamento']

    # Drop Departamento column (keeping departamento)
    merged.drop(columns=['Departamento'], inplace=True)

    return merged


def add_derived_columns(df):
    """Add derived columns to match 2024 dataset structure."""
    logger.info("Adding derived columns...")

    # Primera vuelta shares
    df['fa_share_primera'] = df['fa_primera'] / df['total_primera']
    df['pn_share_primera'] = df['pn_primera'] / df['total_primera']
    df['pc_share_primera'] = df['pc_primera'] / df['total_primera']
    df['ca_share_primera'] = df['ca_primera'] / df['total_primera']
    df['pi_share_primera'] = df['pi_primera'] / df['total_primera']
    df['otros_share_primera'] = df['otros_primera'] / df['total_primera']

    # Ballotage shares
    df['fa_share_ballotage'] = df['fa_ballotage'] / df['total_ballotage']
    df['pn_share_ballotage'] = df['pn_ballotage'] / df['total_ballotage']
    df['blancos_share_ballotage'] = df['blancos_ballotage'] / df['total_ballotage']

    # Vote changes
    df['fa_vote_change'] = df['fa_ballotage'] - df['fa_primera']
    df['pn_vote_change'] = df['pn_ballotage'] - df['pn_primera']
    df['fa_vote_change_pct'] = df['fa_vote_change'] / df['fa_primera']
    df['pn_vote_change_pct'] = df['pn_vote_change'] / df['pn_primera']

    # Participation change
    df['participacion_change'] = (df['total_ballotage'] - df['total_primera']) / df['total_primera']

    # Replace inf and nan with 0
    df.replace([np.inf, -np.inf], 0, inplace=True)
    df.fillna(0, inplace=True)

    return df


def validate_data(df):
    """Validate the merged dataset."""
    logger.info("Validating data...")

    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        logger.warning(f"Missing values found:\n{missing[missing > 0]}")

    # Check for negative values in vote counts
    vote_cols = ['fa_primera', 'pn_primera', 'pc_primera', 'ca_primera',
                 'pi_primera', 'otros_primera', 'fa_ballotage', 'pn_ballotage',
                 'blancos_ballotage']
    negatives = (df[vote_cols] < 0).sum()
    if negatives.any():
        logger.warning(f"Negative values found:\n{negatives[negatives > 0]}")

    # Check vote totals
    check_primera = (
        df['fa_primera'] + df['pn_primera'] + df['pc_primera'] +
        df['ca_primera'] + df['pi_primera'] + df['otros_primera']
    )
    primera_diff = (check_primera - df['total_primera']).abs()
    if primera_diff.max() > 1:
        logger.warning(f"Primera vuelta totals don't match (max diff: {primera_diff.max()})")

    check_ballotage = (
        df['fa_ballotage'] + df['pn_ballotage'] + df['blancos_ballotage']
    )
    ballotage_diff = (check_ballotage - df['total_ballotage']).abs()
    if ballotage_diff.max() > 1:
        logger.warning(f"Ballotage totals don't match (max diff: {ballotage_diff.max()})")

    # Summary statistics
    logger.info(f"\n=== SUMMARY STATISTICS ===")
    logger.info(f"Total circuits: {len(df)}")
    logger.info(f"Departments: {df['departamento'].nunique()}")
    logger.info(f"\nPrimera vuelta totals:")
    logger.info(f"  FA: {df['fa_primera'].sum():,}")
    logger.info(f"  PN: {df['pn_primera'].sum():,}")
    logger.info(f"  PC: {df['pc_primera'].sum():,}")
    logger.info(f"  CA: {df['ca_primera'].sum():,}")
    logger.info(f"  PI: {df['pi_primera'].sum():,}")
    logger.info(f"  Otros: {df['otros_primera'].sum():,}")
    logger.info(f"  Total: {df['total_primera'].sum():,}")
    logger.info(f"\nBallotage totals:")
    logger.info(f"  FA: {df['fa_ballotage'].sum():,}")
    logger.info(f"  PN: {df['pn_ballotage'].sum():,}")
    logger.info(f"  Blancos: {df['blancos_ballotage'].sum():,}")
    logger.info(f"  Total: {df['total_ballotage'].sum():,}")

    return df


def main():
    """Main processing pipeline."""
    logger.info("Starting 2019 data processing...")

    # Load primera vuelta
    df_pv_raw = load_primera_vuelta_2019()
    df_pv = aggregate_primera_vuelta(df_pv_raw)

    # Load ballotage
    df_bal_raw = load_ballotage_2019()
    df_bal = process_ballotage(df_bal_raw)

    # Merge
    df_merged = merge_datasets(df_pv, df_bal)

    # Add derived columns
    df_merged = add_derived_columns(df_merged)

    # Validate
    df_final = validate_data(df_merged)

    # Save
    output_file = PROCESSED_DIR / "circuitos_merged_2019.parquet"
    df_final.to_parquet(output_file, index=False)
    logger.info(f"Saved merged data to {output_file}")

    # Also save as CSV for easy inspection
    csv_file = PROCESSED_DIR / "circuitos_merged_2019.csv"
    df_final.to_csv(csv_file, index=False)
    logger.info(f"Saved CSV to {csv_file}")

    logger.info("2019 data processing complete!")


if __name__ == "__main__":
    main()
