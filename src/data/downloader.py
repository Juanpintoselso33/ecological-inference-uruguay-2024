"""
Data downloader module.
Downloads electoral data from Corte Electoral Uruguay and searches for shapefiles.
"""

import os
import hashlib
from pathlib import Path
from typing import Dict, Optional
import requests
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_config, get_logger

logger = get_logger(__name__)


class DataDownloader:
    """Downloads electoral data and shapefiles."""

    def __init__(self, config_path: str = None):
        """
        Initialize downloader.

        Args:
            config_path: Path to config.yaml
        """
        self.config = get_config(config_path)
        self.urls = self.config.get_data_urls()
        self.dirs = self.config.get_data_dirs()

        # Ensure directories exist
        for dir_path in self.dirs.values():
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    def download_file(self, url: str, output_path: str,
                     chunk_size: int = 8192, force: bool = False) -> bool:
        """
        Download a file from URL to output path with progress bar.

        Args:
            url: URL to download from
            output_path: Local path to save file
            chunk_size: Download chunk size in bytes
            force: If True, re-download even if file exists

        Returns:
            True if download successful, False otherwise
        """
        output_file = Path(output_path)

        # Check if file already exists
        if output_file.exists() and not force:
            logger.info(f"File already exists: {output_path}")
            return True

        logger.info(f"Downloading from {url}")

        try:
            # Send GET request
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()

            # Get total file size
            total_size = int(response.headers.get('content-length', 0))

            # Download with progress bar
            with open(output_path, 'wb') as f, tqdm(
                desc=output_file.name,
                total=total_size,
                unit='B',
                unit_scale=True,
                unit_divisor=1024,
            ) as pbar:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        pbar.update(len(chunk))

            logger.info(f"Downloaded successfully: {output_path}")
            return True

        except requests.exceptions.RequestException as e:
            logger.error(f"Failed to download {url}: {e}")
            if output_file.exists():
                output_file.unlink()  # Remove partial download
            return False

    def compute_checksum(self, file_path: str, algorithm: str = 'md5') -> str:
        """
        Compute checksum of a file.

        Args:
            file_path: Path to file
            algorithm: Hash algorithm ('md5', 'sha1', 'sha256')

        Returns:
            Hexadecimal checksum string
        """
        hash_func = hashlib.new(algorithm)

        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b''):
                hash_func.update(chunk)

        return hash_func.hexdigest()

    def verify_file(self, file_path: str, min_size: int = 1000,
                   expected_checksum: str = None,
                   checksum_algorithm: str = 'md5') -> bool:
        """
        Verify integrity of downloaded file.

        Args:
            file_path: Path to file
            min_size: Minimum expected file size in bytes
            expected_checksum: Expected checksum (if known)
            checksum_algorithm: Hash algorithm for checksum

        Returns:
            True if file is valid
        """
        file_path = Path(file_path)

        if not file_path.exists():
            logger.error(f"File does not exist: {file_path}")
            return False

        # Check file size
        file_size = file_path.stat().st_size
        if file_size < min_size:
            logger.error(
                f"File too small: {file_size} bytes (expected >= {min_size})"
            )
            return False

        # Check checksum if provided
        if expected_checksum:
            actual_checksum = self.compute_checksum(file_path, checksum_algorithm)
            if actual_checksum != expected_checksum:
                logger.error(
                    f"Checksum mismatch: {actual_checksum} != {expected_checksum}"
                )
                return False

        logger.info(f"File verified: {file_path} ({file_size:,} bytes)")
        return True

    def download_electoral_data(self, force: bool = False) -> Dict[str, str]:
        """
        Download electoral data from Corte Electoral.

        Args:
            force: If True, re-download even if files exist

        Returns:
            Dictionary with paths to downloaded files
        """
        downloaded_files = {}

        # Download primera vuelta
        primera_url = self.urls['primera_vuelta']
        primera_path = Path(self.dirs['raw']) / 'primera_vuelta_2024.xlsx'
        success = self.download_file(primera_url, str(primera_path), force=force)

        if success and self.verify_file(primera_path, min_size=10000):
            downloaded_files['primera_vuelta'] = str(primera_path)
            logger.info(f"✓ Primera vuelta data: {primera_path}")
        else:
            logger.error("✗ Failed to download/verify primera vuelta data")

        # Download ballotage
        ballotage_url = self.urls['ballotage']
        ballotage_path = Path(self.dirs['raw']) / 'ballotage_2024.xlsx'
        success = self.download_file(ballotage_url, str(ballotage_path), force=force)

        if success and self.verify_file(ballotage_path, min_size=10000):
            downloaded_files['ballotage'] = str(ballotage_path)
            logger.info(f"✓ Ballotage data: {ballotage_path}")
        else:
            logger.error("✗ Failed to download/verify ballotage data")

        return downloaded_files

    def search_shapefiles(self, use_tavily: bool = True) -> Optional[str]:
        """
        Search for Uruguay electoral circuit shapefiles.

        Args:
            use_tavily: If True, use Tavily search (requires MCP server)

        Returns:
            URL to shapefile or None if not found
        """
        logger.info("Searching for electoral circuit shapefiles...")

        if use_tavily:
            try:
                # This would use the Tavily MCP tool if available
                # For now, we'll provide known URLs as fallback
                logger.warning(
                    "Tavily search not implemented yet. "
                    "Using fallback shapefile sources."
                )
            except Exception as e:
                logger.warning(f"Tavily search failed: {e}")

        # Known fallback URLs for Uruguay shapefiles
        fallback_sources = [
            {
                'name': 'IDE Uruguay - Circuitos Electorales',
                'url': 'https://visualizador.ide.uy/',
                'type': 'WFS service (requires API access)'
            },
            {
                'name': 'Corte Electoral - Geodatos',
                'url': 'https://www.gub.uy/corte-electoral/',
                'type': 'Check official website for geodata section'
            },
            {
                'name': 'OpenData Uruguay - Catálogo',
                'url': 'https://catalogodatos.gub.uy/',
                'type': 'Search for "circuitos electorales"'
            }
        ]

        logger.info("Potential shapefile sources:")
        for source in fallback_sources:
            logger.info(f"  - {source['name']}: {source['url']}")
            logger.info(f"    Type: {source['type']}")

        logger.warning(
            "Shapefile download requires manual intervention. "
            "Please download shapefiles to: " + self.dirs['shapefiles']
        )

        return None

    def download_shapefile(self, url: str, force: bool = False) -> Optional[str]:
        """
        Download shapefile from URL.

        Args:
            url: URL to shapefile (usually .zip)
            force: If True, re-download even if file exists

        Returns:
            Path to downloaded file or None if failed
        """
        output_path = Path(self.dirs['shapefiles']) / 'circuitos_uruguay.zip'

        success = self.download_file(url, str(output_path), force=force)

        if success and self.verify_file(output_path, min_size=1000):
            logger.info(f"✓ Shapefile downloaded: {output_path}")

            # Try to extract if it's a zip file
            if output_path.suffix == '.zip':
                try:
                    import zipfile
                    with zipfile.ZipFile(output_path, 'r') as zip_ref:
                        zip_ref.extractall(self.dirs['shapefiles'])
                    logger.info(f"✓ Shapefile extracted to: {self.dirs['shapefiles']}")
                except Exception as e:
                    logger.error(f"Failed to extract shapefile: {e}")

            return str(output_path)
        else:
            logger.error("✗ Failed to download/verify shapefile")
            return None

    def download_all(self, force: bool = False,
                    shapefile_url: Optional[str] = None) -> Dict[str, any]:
        """
        Download all required data.

        Args:
            force: If True, re-download existing files
            shapefile_url: Optional direct URL to shapefile

        Returns:
            Dictionary with download status
        """
        logger.info("Starting data download...")

        results = {
            'electoral_data': self.download_electoral_data(force=force),
            'shapefile': None
        }

        # Try to download shapefile if URL provided
        if shapefile_url:
            results['shapefile'] = self.download_shapefile(shapefile_url, force=force)
        else:
            results['shapefile'] = self.search_shapefiles()

        logger.info("Download complete!")
        logger.info(f"Electoral data files: {len(results['electoral_data'])}")
        logger.info(f"Shapefile: {'downloaded' if results['shapefile'] else 'not available'}")

        return results


def main():
    """Main function for standalone script execution."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Download electoral data from Corte Electoral Uruguay'
    )
    parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Force re-download of existing files'
    )
    parser.add_argument(
        '--shapefile-url',
        type=str,
        help='Direct URL to shapefile (optional)'
    )
    parser.add_argument(
        '--config',
        type=str,
        help='Path to config.yaml (optional)'
    )

    args = parser.parse_args()

    # Initialize downloader
    downloader = DataDownloader(config_path=args.config)

    # Download all data
    results = downloader.download_all(
        force=args.force,
        shapefile_url=args.shapefile_url
    )

    # Print summary
    print("\n" + "="*60)
    print("DOWNLOAD SUMMARY")
    print("="*60)

    if results['electoral_data']:
        print("\n✓ Electoral Data Downloaded:")
        for key, path in results['electoral_data'].items():
            print(f"  - {key}: {path}")
    else:
        print("\n✗ No electoral data downloaded")

    if results['shapefile']:
        print(f"\n✓ Shapefile: {results['shapefile']}")
    else:
        print("\n⚠ Shapefile not downloaded (requires manual intervention)")

    print("\n" + "="*60)


if __name__ == '__main__':
    main()
