"""
Data loader module.
Parses Excel files from Corte Electoral and extracts vote data by circuit.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_config, get_logger

logger = get_logger(__name__)


class ElectoralDataLoader:
    """Loads and parses electoral data from Excel files."""

    def __init__(self, config_path: str = None):
        """
        Initialize data loader.

        Args:
            config_path: Path to config.yaml
        """
        self.config = get_config(config_path)
        self.dirs = self.config.get_data_dirs()

    def load_excel_file(self, file_path: str,
                       sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Load Excel file into DataFrame.

        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name to load (if None, loads first sheet)

        Returns:
            DataFrame with loaded data
        """
        logger.info(f"Loading Excel file: {file_path}")

        try:
            if sheet_name:
                df = pd.read_excel(file_path, sheet_name=sheet_name, engine='openpyxl')
                logger.info(f"Loaded sheet '{sheet_name}': {len(df)} rows")
            else:
                df = pd.read_excel(file_path, engine='openpyxl')
                logger.info(f"Loaded first sheet: {len(df)} rows")

            return df

        except Exception as e:
            logger.error(f"Failed to load Excel file: {e}")
            raise

    def list_sheets(self, file_path: str) -> List[str]:
        """
        List all sheet names in an Excel file.

        Args:
            file_path: Path to Excel file

        Returns:
            List of sheet names
        """
        try:
            xl_file = pd.ExcelFile(file_path, engine='openpyxl')
            sheets = xl_file.sheet_names
            logger.info(f"Found {len(sheets)} sheets: {sheets}")
            return sheets

        except Exception as e:
            logger.error(f"Failed to read Excel file: {e}")
            raise

    def parse_primera_vuelta(self, file_path: str,
                            sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Parse primera vuelta (first round) data.

        Expected columns (approximate):
        - Departamento
        - Circuito or Serie
        - Frente Amplio (FA)
        - Partido Nacional (PN)
        - Partido Colorado (PC)
        - Cabildo Abierto (CA)
        - Otros partidos
        - Blancos
        - Anulados
        - Total

        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name (if None, auto-detect)

        Returns:
            DataFrame with parsed data
        """
        logger.info("Parsing primera vuelta data...")

        # Load data
        df = self.load_excel_file(file_path, sheet_name)

        logger.info(f"Original columns: {df.columns.tolist()}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"First few rows:\n{df.head()}")

        # Return raw data for now - cleaning will be done in cleaner.py
        return df

    def parse_ballotage(self, file_path: str,
                       sheet_name: Optional[str] = None) -> pd.DataFrame:
        """
        Parse ballotage (second round) data.

        Expected columns (approximate):
        - Departamento
        - Circuito or Serie
        - Frente Amplio (FA) / Lema 609
        - Partido Nacional (PN) / Lema 404
        - Blancos
        - Anulados
        - Total

        Args:
            file_path: Path to Excel file
            sheet_name: Sheet name (if None, auto-detect)

        Returns:
            DataFrame with parsed data
        """
        logger.info("Parsing ballotage data...")

        # Load data
        df = self.load_excel_file(file_path, sheet_name)

        logger.info(f"Original columns: {df.columns.tolist()}")
        logger.info(f"Shape: {df.shape}")
        logger.info(f"First few rows:\n{df.head()}")

        # Return raw data for now - cleaning will be done in cleaner.py
        return df

    def explore_file_structure(self, file_path: str) -> Dict:
        """
        Explore structure of Excel file to understand format.

        Args:
            file_path: Path to Excel file

        Returns:
            Dictionary with file structure information
        """
        logger.info(f"Exploring file structure: {file_path}")

        info = {
            'file_path': file_path,
            'sheets': [],
            'file_size': Path(file_path).stat().st_size
        }

        # List all sheets
        sheet_names = self.list_sheets(file_path)
        info['sheet_count'] = len(sheet_names)

        # Analyze each sheet
        for sheet_name in sheet_names:
            try:
                df = self.load_excel_file(file_path, sheet_name)

                sheet_info = {
                    'name': sheet_name,
                    'shape': df.shape,
                    'columns': df.columns.tolist(),
                    'dtypes': df.dtypes.to_dict(),
                    'missing_values': df.isna().sum().to_dict(),
                    'sample_rows': df.head(3).to_dict('records')
                }

                info['sheets'].append(sheet_info)

            except Exception as e:
                logger.warning(f"Could not analyze sheet '{sheet_name}': {e}")

        return info

    def load_primera_vuelta(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load primera vuelta data from default location.

        Args:
            file_path: Optional custom file path

        Returns:
            DataFrame with primera vuelta data
        """
        if file_path is None:
            file_path = Path(self.dirs['raw']) / 'primera_vuelta_2024.xlsx'

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Primera vuelta file not found: {file_path}")

        return self.parse_primera_vuelta(str(file_path))

    def load_ballotage(self, file_path: Optional[str] = None) -> pd.DataFrame:
        """
        Load ballotage data from default location.

        Args:
            file_path: Optional custom file path

        Returns:
            DataFrame with ballotage data
        """
        if file_path is None:
            file_path = Path(self.dirs['raw']) / 'ballotage_2024.xlsx'

        if not Path(file_path).exists():
            raise FileNotFoundError(f"Ballotage file not found: {file_path}")

        return self.parse_ballotage(str(file_path))

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """
        Load all electoral data.

        Returns:
            Dictionary with 'primera_vuelta' and 'ballotage' DataFrames
        """
        logger.info("Loading all electoral data...")

        data = {
            'primera_vuelta': self.load_primera_vuelta(),
            'ballotage': self.load_ballotage()
        }

        logger.info(f"✓ Loaded primera vuelta: {data['primera_vuelta'].shape}")
        logger.info(f"✓ Loaded ballotage: {data['ballotage'].shape}")

        return data


def main():
    """Main function for standalone script execution."""
    import argparse
    import json

    parser = argparse.ArgumentParser(
        description='Load and explore electoral data from Excel files'
    )
    parser.add_argument(
        'action',
        choices=['load', 'explore', 'sheets'],
        help='Action to perform'
    )
    parser.add_argument(
        '--file',
        type=str,
        help='Path to Excel file (optional)'
    )
    parser.add_argument(
        '--election',
        choices=['primera', 'ballotage'],
        help='Election type (for load action)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config.yaml (optional)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for exploration results'
    )

    args = parser.parse_args()

    # Initialize loader
    loader = ElectoralDataLoader(config_path=args.config)

    if args.action == 'sheets':
        # List sheets in file
        if not args.file:
            print("Error: --file required for 'sheets' action")
            return

        sheets = loader.list_sheets(args.file)
        print(f"\nSheets in {args.file}:")
        for i, sheet in enumerate(sheets, 1):
            print(f"  {i}. {sheet}")

    elif args.action == 'explore':
        # Explore file structure
        if not args.file:
            print("Error: --file required for 'explore' action")
            return

        info = loader.explore_file_structure(args.file)

        # Print summary
        print("\n" + "="*60)
        print("FILE STRUCTURE")
        print("="*60)
        print(f"\nFile: {info['file_path']}")
        print(f"Size: {info['file_size']:,} bytes")
        print(f"Sheets: {info['sheet_count']}")

        for sheet in info['sheets']:
            print(f"\n--- Sheet: {sheet['name']} ---")
            print(f"Shape: {sheet['shape']}")
            print(f"Columns ({len(sheet['columns'])}):")
            for col in sheet['columns']:
                missing = sheet['missing_values'].get(col, 0)
                dtype = sheet['dtypes'].get(col, 'unknown')
                print(f"  - {col} ({dtype}) - {missing} missing")

        # Save to JSON if output specified
        if args.output:
            with open(args.output, 'w', encoding='utf-8') as f:
                json.dump(info, f, indent=2, default=str)
            print(f"\nExploration results saved to: {args.output}")

    elif args.action == 'load':
        # Load data
        if args.election == 'primera':
            df = loader.load_primera_vuelta(args.file)
            print(f"\n✓ Loaded primera vuelta data: {df.shape}")
        elif args.election == 'ballotage':
            df = loader.load_ballotage(args.file)
            print(f"\n✓ Loaded ballotage data: {df.shape}")
        else:
            data = loader.load_all()
            print(f"\n✓ Loaded all electoral data:")
            print(f"  - Primera vuelta: {data['primera_vuelta'].shape}")
            print(f"  - Ballotage: {data['ballotage'].shape}")

        print("\nPreview:")
        print(df.head() if args.election else data['primera_vuelta'].head())


if __name__ == '__main__':
    main()
