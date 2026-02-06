"""
Script to download electoral data from Corte Electoral Uruguay.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.downloader import DataDownloader, main as downloader_main

if __name__ == '__main__':
    downloader_main()
