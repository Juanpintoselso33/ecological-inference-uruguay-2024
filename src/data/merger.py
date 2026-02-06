"""
Data merger module.
Merges primera vuelta and ballotage data by circuit.
"""

from pathlib import Path
from typing import Dict, Optional
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_config, get_logger, validate_circuit_data, log_validation_results

logger = get_logger(__name__)


class ElectoralDataMerger:
    """Merges primera vuelta and ballotage electoral data."""

    def __init__(self, config_path: str = None):
        """
        Initialize data merger.

        Args:
            config_path: Path to config.yaml
        """
        self.config = get_config(config_path)
        self.dirs = self.config.get_data_dirs()

    def merge_elections(self, df_primera: pd.DataFrame,
                       df_ballotage: pd.DataFrame,
                       how: str = 'inner') -> pd.DataFrame:
        """
        Merge primera vuelta and ballotage data by circuit_id.

        Args:
            df_primera: Primera vuelta DataFrame
            df_ballotage: Ballotage DataFrame
            how: Join type ('inner', 'outer', 'left', 'right')

        Returns:
            Merged DataFrame
        """
        logger.info(f"Merging elections (join type: {how})...")
        logger.info(f"  Primera vuelta: {len(df_primera)} circuits")
        logger.info(f"  Ballotage: {len(df_ballotage)} circuits")

        # Merge on circuito_id
        merged = pd.merge(
            df_primera,
            df_ballotage,
            on='circuito_id',
            how=how,
            suffixes=('', '_dup')  # Avoid duplicating column names
        )

        # Handle duplicate departamento and serie columns
        if 'departamento_dup' in merged.columns:
            # Use departamento from primera (should be the same)
            merged.drop(columns=['departamento_dup'], inplace=True)

        if 'serie_dup' in merged.columns:
            # Use serie from primera (should be the same)
            merged.drop(columns=['serie_dup'], inplace=True)

        logger.info(f"  Merged: {len(merged)} circuits")

        # Report unmatched circuits
        if how == 'inner':
            primera_ids = set(df_primera['circuito_id'])
            ballotage_ids = set(df_ballotage['circuito_id'])
            merged_ids = set(merged['circuito_id'])

            only_primera = primera_ids - ballotage_ids
            only_ballotage = ballotage_ids - primera_ids

            if only_primera:
                logger.warning(f"  {len(only_primera)} circuits only in primera vuelta (excluded)")
            if only_ballotage:
                logger.warning(f"  {len(only_ballotage)} circuits only in ballotage (excluded)")

        return merged

    def calculate_proportions(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate vote share proportions for all parties.

        Args:
            df: Merged DataFrame

        Returns:
            DataFrame with proportion columns added
        """
        logger.info("Calculating vote share proportions...")

        # Primera vuelta proportions
        primera_parties = ['ca', 'fa', 'otros', 'pc', 'pi', 'pn']
        for party in primera_parties:
            col = f'{party}_primera'
            if col in df.columns:
                df[f'{party}_share_primera'] = df[col] / df['total_primera'].replace(0, np.nan)

        # Ballotage proportions
        ballotage_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage', 'anulados_ballotage']
        for col in ballotage_cols:
            if col in df.columns:
                share_col = col.replace('_ballotage', '_share_ballotage')
                df[share_col] = df[col] / df['total_ballotage'].replace(0, np.nan)

        logger.info(f"  Added proportion columns")

        return df

    def add_metadata(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add metadata columns (participation rates, vote changes, etc.).

        Args:
            df: DataFrame with merged data

        Returns:
            DataFrame with metadata added
        """
        logger.info("Adding metadata columns...")

        # Participation change
        if 'total_primera' in df.columns and 'total_ballotage' in df.columns:
            df['participacion_change'] = (
                (df['total_ballotage'] - df['total_primera']) / df['total_primera'].replace(0, np.nan)
            )

        # Vote changes for major parties
        if 'fa_primera' in df.columns and 'fa_ballotage' in df.columns:
            df['fa_vote_change'] = df['fa_ballotage'] - df['fa_primera']
            df['fa_vote_change_pct'] = df['fa_vote_change'] / df['fa_primera'].replace(0, np.nan)

        if 'pn_primera' in df.columns and 'pn_ballotage' in df.columns:
            df['pn_vote_change'] = df['pn_ballotage'] - df['pn_primera']
            df['pn_vote_change_pct'] = df['pn_vote_change'] / df['pn_primera'].replace(0, np.nan)

        return df

    def filter_low_turnout(self, df: pd.DataFrame,
                          min_votes: int = None) -> pd.DataFrame:
        """
        Filter out circuits with very low turnout.

        Args:
            df: DataFrame to filter
            min_votes: Minimum votes required (from config if None)

        Returns:
            Filtered DataFrame
        """
        if min_votes is None:
            min_votes = self.config.get('validation.min_votes_per_circuit', 10)

        initial_count = len(df)

        # Filter by primera vuelta votes
        df_filtered = df[df['total_primera'] >= min_votes].copy()

        filtered_count = initial_count - len(df_filtered)
        if filtered_count > 0:
            logger.warning(f"Filtered out {filtered_count} circuits with < {min_votes} votes")

        return df_filtered

    def validate_merged_data(self, df: pd.DataFrame) -> bool:
        """
        Validate merged dataset.

        Args:
            df: Merged DataFrame

        Returns:
            True if all validations pass
        """
        logger.info("Validating merged data...")

        all_valid = True

        # Check for missing circuit IDs
        required_cols = ['circuito_id', 'departamento', 'serie']
        is_valid, errors = validate_circuit_data(df, required_cols)
        log_validation_results("Required columns", is_valid, errors)
        all_valid &= is_valid

        # Check for NaN values in vote columns
        vote_cols = [col for col in df.columns if 'primera' in col or 'ballotage' in col]
        vote_cols = [col for col in vote_cols if not col.endswith('_share_primera') and not col.endswith('_share_ballotage')]

        for col in vote_cols:
            if col in df.columns:
                n_nan = df[col].isna().sum()
                if n_nan > 0:
                    logger.warning(f"Column {col} has {n_nan} NaN values")
                    all_valid = False

        return all_valid

    def merge_from_files(self,
                        primera_path: Optional[str] = None,
                        ballotage_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load and merge data from parquet files.

        Args:
            primera_path: Path to primera vuelta parquet (default: data/processed/)
            ballotage_path: Path to ballotage parquet (default: data/processed/)

        Returns:
            Merged DataFrame
        """
        # Default paths
        if primera_path is None:
            primera_path = Path(self.dirs['processed']) / 'primera_vuelta_clean.parquet'
        if ballotage_path is None:
            ballotage_path = Path(self.dirs['processed']) / 'ballotage_clean.parquet'

        # Load data
        logger.info(f"Loading primera vuelta from {primera_path}...")
        df_primera = pd.read_parquet(primera_path)

        logger.info(f"Loading ballotage from {ballotage_path}...")
        df_ballotage = pd.read_parquet(ballotage_path)

        # Merge
        df_merged = self.merge_elections(df_primera, df_ballotage, how='inner')

        # Calculate proportions
        df_merged = self.calculate_proportions(df_merged)

        # Add metadata
        df_merged = self.add_metadata(df_merged)

        # Filter low turnout
        df_merged = self.filter_low_turnout(df_merged)

        # Validate
        self.validate_merged_data(df_merged)

        return df_merged


def main():
    """Main function for standalone script execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Merge primera vuelta and ballotage electoral data'
    )
    parser.add_argument(
        '--primera',
        type=str,
        help='Path to primera vuelta parquet (default: data/processed/primera_vuelta_clean.parquet)'
    )
    parser.add_argument(
        '--ballotage',
        type=str,
        help='Path to ballotage parquet (default: data/processed/ballotage_clean.parquet)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for merged data (default: data/processed/circuitos_merged.parquet)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config.yaml (optional)'
    )

    args = parser.parse_args()

    # Initialize merger
    merger = ElectoralDataMerger(config_path=args.config)

    # Merge data
    df_merged = merger.merge_from_files(
        primera_path=args.primera,
        ballotage_path=args.ballotage
    )

    # Save merged data
    if args.output is None:
        args.output = str(Path(merger.dirs['processed']) / 'circuitos_merged.parquet')

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df_merged.to_parquet(args.output, index=False)

    print(f"\nMerged data saved to: {args.output}")
    print(f"  Shape: {df_merged.shape}")
    print(f"  Circuits: {len(df_merged)}")
    print(f"  Columns: {len(df_merged.columns)}")
    print(f"\nColumn groups:")
    print(f"  Primera vuelta votes: {len([c for c in df_merged.columns if c.endswith('_primera') and 'share' not in c])}")
    print(f"  Ballotage votes: {len([c for c in df_merged.columns if c.endswith('_ballotage') and 'share' not in c])}")
    print(f"  Proportions: {len([c for c in df_merged.columns if 'share' in c])}")
    print(f"  Metadata: {len([c for c in df_merged.columns if 'change' in c or c in ['habilitados', 'participacion_ballotage']])}")


if __name__ == '__main__':
    main()
