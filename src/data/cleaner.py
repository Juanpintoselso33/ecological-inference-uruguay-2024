"""
Data cleaner module.
Normalizes column names, validates data, and calculates proportions.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
import re

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_config, get_logger, validate_vote_counts, log_validation_results

logger = get_logger(__name__)


class ElectoralDataCleaner:
    """Cleans and normalizes electoral data."""

    # Mapping of party lemas to standardized codes
    PARTY_MAPPING = {
        # Major parties
        'Partido Frente Amplio': 'FA',
        'Partido Nacional': 'PN',
        'Partido Colorado': 'PC',
        'Partido Cabildo Abierto': 'CA',  # Fixed: add "Partido" prefix
        # Smaller parties (will group as OTROS)
        'Partido Constitucional Ambientalista': 'PCA',
        'Partido de la Gente': 'PG',
        'Partido Ecologista Radical Intransigente': 'PERI',
        'Partido Independiente': 'PI',
        'Partido Identidad Soberana': 'PIS',
        'Partido Asamblea Popular': 'PAP',
        'Partido Por Los Cambios Necesarios (PCN)': 'PCN',
        'Partido Avanzar Republicano': 'PAR',
        # Ballotage candidates
        'Yamandú Orsi - Carolina Cosse': 'FA',
        'Álvaro Delgado - Valeria Ripoll': 'PN'
    }

    # Known special vote categories
    SPECIAL_CATEGORIES = ['EN_BLANCO', 'ANULADOS', 'BLANCO', 'ANULADO']

    def __init__(self, config_path: str = None):
        """
        Initialize data cleaner.

        Args:
            config_path: Path to config.yaml
        """
        self.config = get_config(config_path)
        self.dirs = self.config.get_data_dirs()

    def load_with_skip(self, file_path: str, header_row: int = 7) -> pd.DataFrame:
        """
        Load Excel file with header at specific row.

        Args:
            file_path: Path to Excel file
            header_row: Row number containing column headers (default 7 for Corte Electoral format)

        Returns:
            DataFrame with cleaned data
        """
        logger.info(f"Loading {file_path} (header at row {header_row})...")

        # Load with header at specified row
        df = pd.read_excel(file_path, header=header_row, engine='openpyxl')

        # The first row contains the actual column names
        # Check if first row has string values (likely column names)
        first_row = df.iloc[0]
        if all(isinstance(val, str) for val in first_row if pd.notna(val)):
            # Use first row as column names
            df.columns = first_row.tolist()
            # Drop the first row
            df = df.iloc[1:].reset_index(drop=True)
            logger.info(f"Used first row as column names: {df.columns.tolist()}")

        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        return df

    def clean_primera_vuelta(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean primera vuelta data.

        Input format (long):
        ACTO | CONVOCATORIA | DEPTO | CIRCUITO | SERIES | ESCRUTINIO | LEMA | HOJA | CNT_VOTOS

        Output format (aggregated by circuit):
        circuito_id | departamento | serie | FA_votos | PN_votos | ... | total_votos

        Args:
            df: Raw DataFrame from Excel

        Returns:
            Cleaned and aggregated DataFrame
        """
        logger.info("Cleaning primera vuelta data...")

        # Normalize column names
        df.columns = [self._normalize_column_name(col) for col in df.columns]

        logger.info(f"Columns: {df.columns.tolist()}")

        # Map lemas to party codes
        df['partido'] = df['lema'].map(self.PARTY_MAPPING)

        # Mark unknown parties as 'OTROS'
        df.loc[df['partido'].isna(), 'partido'] = 'OTROS'

        # Group smaller parties into OTROS (keep only major parties + PI separate)
        major_parties = ['FA', 'PN', 'PC', 'CA', 'PI']
        df.loc[~df['partido'].isin(major_parties), 'partido'] = 'OTROS'

        logger.info(f"Parties found: {df['partido'].unique()}")
        logger.info(f"Unique lemas: {df['lema'].nunique()}")

        # Create circuit identifier
        df['circuito_id'] = (
            df['depto'].astype(str) + '_' +
            df['circuito'].astype(str) + '_' +
            df['series'].astype(str)
        )

        # Aggregate votes by circuit and party
        logger.info("Aggregating votes by circuit and party...")
        agg_df = df.groupby(['circuito_id', 'depto', 'series', 'partido'])['cnt_votos'].sum().reset_index()

        # Pivot to wide format
        logger.info("Pivoting to wide format...")
        pivot_df = agg_df.pivot_table(
            index=['circuito_id', 'depto', 'series'],
            columns='partido',
            values='cnt_votos',
            fill_value=0
        ).reset_index()

        # Rename depto to departamento for consistency
        pivot_df.rename(columns={'depto': 'departamento', 'series': 'serie'}, inplace=True)

        # Flatten column names
        pivot_df.columns.name = None

        # Add '_primera' suffix to party columns
        party_cols = [col for col in pivot_df.columns if col not in ['circuito_id', 'departamento', 'serie']]
        rename_dict = {col: f"{col.lower()}_primera" for col in party_cols}
        pivot_df.rename(columns=rename_dict, inplace=True)

        # Calculate total votes
        vote_cols = [col for col in pivot_df.columns if col.endswith('_primera')]
        pivot_df['total_primera'] = pivot_df[vote_cols].sum(axis=1)

        logger.info(f"Cleaned primera vuelta: {len(pivot_df)} circuits")
        logger.info(f"Columns: {pivot_df.columns.tolist()}")

        return pivot_df

    def clean_ballotage(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean ballotage data.

        Input format (already aggregated by circuit):
        ACTO | CONVOCATORIA | DEPTO | CIRCUITO | SERIE | ... | Yamandú Orsi | Álvaro Delgado

        Output format:
        circuito_id | departamento | serie | FA_ballotage | PN_ballotage | blancos_ballotage | anulados_ballotage | total_ballotage | habilitados

        Args:
            df: Raw DataFrame from Excel

        Returns:
            Cleaned DataFrame
        """
        logger.info("Cleaning ballotage data...")

        # Normalize column names
        df.columns = [self._normalize_column_name(col) for col in df.columns]

        logger.info(f"Columns: {df.columns.tolist()}")

        # Create circuit identifier (same as primera vuelta)
        df['circuito_id'] = (
            df['depto'].astype(str) + '_' +
            df['circuito'].astype(str) + '_' +
            df['serie'].astype(str)
        )

        # Identify candidate columns (contains "-" or "Orsi"/"Delgado")
        candidate_cols = [col for col in df.columns if '-' in str(col) or 'orsi' in str(col).lower() or 'delgado' in str(col).lower()]

        logger.info(f"Candidate columns: {candidate_cols}")

        # Map to standardized names
        rename_dict = {
            'circuito_id': 'circuito_id',
            'depto': 'departamento',
            'serie': 'serie',
            'habilitado': 'habilitados',
            'en_blanco': 'blancos_ballotage',
            'anulados': 'anulados_ballotage',
            't_emitidos': 'total_ballotage'
        }

        # Map candidate columns to FA and PN
        for col in candidate_cols:
            col_lower = str(col).lower()
            if 'orsi' in col_lower or 'yamand' in col_lower:
                rename_dict[col] = 'fa_ballotage'
            elif 'delgado' in col_lower or 'lvaro' in col_lower or 'alvaro' in col_lower:
                rename_dict[col] = 'pn_ballotage'

        # Select and rename columns
        cols_to_keep = list(rename_dict.keys())
        clean_df = df[cols_to_keep].copy()
        clean_df.rename(columns=rename_dict, inplace=True)

        # Recalculate total from components (more reliable than t_emitidos)
        clean_df['total_ballotage'] = (
            clean_df['fa_ballotage'] +
            clean_df['pn_ballotage'] +
            clean_df['blancos_ballotage'] +
            clean_df['anulados_ballotage']
        )

        # Calculate participation rate (handle division by zero)
        if 'habilitados' in clean_df.columns:
            clean_df['participacion_ballotage'] = clean_df['total_ballotage'] / clean_df['habilitados'].replace(0, np.nan)

        logger.info(f"Cleaned ballotage: {len(clean_df)} circuits")
        logger.info(f"Columns: {clean_df.columns.tolist()}")

        return clean_df

    def calculate_proportions(self, df: pd.DataFrame, vote_cols: List[str],
                             total_col: str, suffix: str = '_share') -> pd.DataFrame:
        """
        Calculate vote share proportions.

        Args:
            df: DataFrame with vote counts
            vote_cols: Columns with vote counts
            total_col: Column with total votes
            suffix: Suffix for proportion columns

        Returns:
            DataFrame with added proportion columns
        """
        logger.info(f"Calculating proportions for {len(vote_cols)} columns...")

        for col in vote_cols:
            prop_col = col.replace('_primera', suffix).replace('_ballotage', suffix)
            # Avoid division by zero
            df[prop_col] = df[col] / df[total_col].replace(0, np.nan)

        return df

    def validate_cleaned_data(self, df: pd.DataFrame, election_type: str) -> bool:
        """
        Validate cleaned data.

        Args:
            df: Cleaned DataFrame
            election_type: 'primera_vuelta' or 'ballotage'

        Returns:
            True if all validations pass
        """
        logger.info(f"Validating {election_type} data...")

        all_valid = True

        # Get vote columns
        if election_type == 'primera_vuelta':
            vote_cols = [col for col in df.columns if col.endswith('_primera')]
            total_col = 'total_primera'
        else:
            vote_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage', 'anulados_ballotage']
            total_col = 'total_ballotage'

        # Validate vote counts
        is_valid, errors = validate_vote_counts(df, vote_cols, total_col)
        log_validation_results(f"{election_type} - vote counts", is_valid, errors)
        all_valid &= is_valid

        # Check for negative values
        for col in vote_cols + [total_col]:
            if (df[col] < 0).any():
                n_negative = (df[col] < 0).sum()
                logger.error(f"Column {col} has {n_negative} negative values")
                all_valid = False

        # Check for missing circuit IDs
        if df['circuito_id'].isna().any():
            n_missing = df['circuito_id'].isna().sum()
            logger.error(f"{n_missing} rows have missing circuito_id")
            all_valid = False

        # Check for duplicate circuits
        duplicates = df['circuito_id'].duplicated()
        if duplicates.any():
            n_dup = duplicates.sum()
            logger.error(f"{n_dup} duplicate circuit IDs found")
            all_valid = False

        return all_valid

    def _normalize_column_name(self, col: str) -> str:
        """
        Normalize column name to lowercase, remove special characters.

        Args:
            col: Original column name

        Returns:
            Normalized column name
        """
        # Convert to string
        col = str(col)

        # Replace Spanish characters
        replacements = {
            'á': 'a', 'é': 'e', 'í': 'i', 'ó': 'o', 'ú': 'u',
            'ñ': 'n', 'ü': 'u',
            'Á': 'a', 'É': 'e', 'Í': 'i', 'Ó': 'o', 'Ú': 'u',
            'Ñ': 'n', 'Ü': 'u'
        }

        for old, new in replacements.items():
            col = col.replace(old, new)

        # Remove special characters except underscore, hyphen, space
        col = re.sub(r'[^a-zA-Z0-9_\- ]', '', col)

        # Replace spaces and hyphens with underscores
        col = col.replace(' ', '_').replace('-', '_')

        # Convert to lowercase
        col = col.lower()

        # Remove consecutive underscores
        col = re.sub(r'_+', '_', col)

        # Remove leading/trailing underscores
        col = col.strip('_')

        return col

    def clean_primera_vuelta_file(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and clean primera vuelta file from disk.

        Args:
            file_path: Optional custom file path

        Returns:
            Cleaned DataFrame
        """
        if file_path is None:
            file_path = Path(self.dirs['raw']) / 'primera_vuelta_2024.xlsx'

        df = self.load_with_skip(file_path, header_row=7)
        df_clean = self.clean_primera_vuelta(df)

        # Validate
        self.validate_cleaned_data(df_clean, 'primera_vuelta')

        return df_clean

    def clean_ballotage_file(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and clean ballotage file from disk.

        Args:
            file_path: Optional custom file path

        Returns:
            Cleaned DataFrame
        """
        if file_path is None:
            file_path = Path(self.dirs['raw']) / 'ballotage_2024.xlsx'

        df = self.load_with_skip(file_path, header_row=7)
        df_clean = self.clean_ballotage(df)

        # Validate
        self.validate_cleaned_data(df_clean, 'ballotage')

        return df_clean

    def clean_all(self) -> Dict[str, pd.DataFrame]:
        """
        Clean all electoral data files.

        Returns:
            Dictionary with 'primera_vuelta' and 'ballotage' cleaned DataFrames
        """
        logger.info("Cleaning all electoral data...")

        data = {
            'primera_vuelta': self.clean_primera_vuelta_file(),
            'ballotage': self.clean_ballotage_file()
        }

        logger.info(f"✓ Cleaned primera vuelta: {data['primera_vuelta'].shape}")
        logger.info(f"✓ Cleaned ballotage: {data['ballotage'].shape}")

        return data


def main():
    """Main function for standalone script execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Clean and normalize electoral data'
    )
    parser.add_argument(
        'action',
        choices=['clean', 'validate'],
        help='Action to perform'
    )
    parser.add_argument(
        '--election',
        choices=['primera', 'ballotage', 'all'],
        default='all',
        help='Election type to clean'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        help='Output directory (default: data/processed/)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config.yaml (optional)'
    )

    args = parser.parse_args()

    # Initialize cleaner
    cleaner = ElectoralDataCleaner(config_path=args.config)

    if args.output_dir is None:
        args.output_dir = cleaner.dirs['processed']

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    if args.action == 'clean':
        if args.election == 'all':
            data = cleaner.clean_all()

            # Save cleaned data
            for election_type, df in data.items():
                output_file = Path(args.output_dir) / f'{election_type}_clean.parquet'
                df.to_parquet(output_file, index=False)
                print(f"\n✓ Saved {election_type}: {output_file}")
                print(f"  Shape: {df.shape}")
                print(f"  Columns: {df.columns.tolist()}")

        elif args.election == 'primera':
            df = cleaner.clean_primera_vuelta_file()
            output_file = Path(args.output_dir) / 'primera_vuelta_clean.parquet'
            df.to_parquet(output_file, index=False)
            print(f"\n✓ Saved primera vuelta: {output_file}")
            print(f"  Shape: {df.shape}")

        elif args.election == 'ballotage':
            df = cleaner.clean_ballotage_file()
            output_file = Path(args.output_dir) / 'ballotage_clean.parquet'
            df.to_parquet(output_file, index=False)
            print(f"\n✓ Saved ballotage: {output_file}")
            print(f"  Shape: {df.shape}")


if __name__ == '__main__':
    main()
