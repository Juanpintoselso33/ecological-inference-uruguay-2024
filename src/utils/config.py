"""
Configuration loader utility.
Loads and validates configuration from config.yaml.
"""

import os
from pathlib import Path
from typing import Any, Dict
import yaml


class Config:
    """Configuration manager for the electoral inference project."""

    def __init__(self, config_path: str = None):
        """
        Initialize configuration loader.

        Args:
            config_path: Path to config.yaml. If None, searches in project root.
        """
        if config_path is None:
            # Find project root (directory containing config.yaml)
            current = Path(__file__).resolve()
            while current.parent != current:
                config_file = current / "config.yaml"
                if config_file.exists():
                    config_path = str(config_file)
                    break
                current = current.parent
            else:
                raise FileNotFoundError("config.yaml not found in project tree")

        self.config_path = Path(config_path)
        self.project_root = self.config_path.parent
        self._config = self._load_config()
        self._resolve_paths()

    def _load_config(self) -> Dict[str, Any]:
        """Load YAML configuration file."""
        with open(self.config_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        return config

    def _resolve_paths(self):
        """Convert relative paths to absolute paths based on project root."""
        # Data directories
        for key in ['raw_dir', 'processed_dir', 'shapefiles_dir']:
            if key in self._config['data']:
                rel_path = self._config['data'][key]
                self._config['data'][key] = str(self.project_root / rel_path)

        # Output directories
        for key in ['figures_dir', 'tables_dir', 'reports_dir']:
            if key in self._config['outputs']:
                rel_path = self._config['outputs'][key]
                self._config['outputs'][key] = str(self.project_root / rel_path)

        # Log file
        if 'file' in self._config.get('logging', {}):
            rel_path = self._config['logging']['file']
            self._config['logging']['file'] = str(self.project_root / rel_path)

    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.

        Args:
            key_path: Dot-separated path (e.g., 'data.raw_dir')
            default: Default value if key not found

        Returns:
            Configuration value

        Example:
            >>> config = Config()
            >>> config.get('models.king_ei.num_samples')
            10000
        """
        keys = key_path.split('.')
        value = self._config

        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default

        return value

    def get_data_urls(self) -> Dict[str, str]:
        """Get data download URLs."""
        return self._config['data']['urls']

    def get_data_dirs(self) -> Dict[str, str]:
        """Get data directory paths."""
        return {
            'raw': self._config['data']['raw_dir'],
            'processed': self._config['data']['processed_dir'],
            'shapefiles': self._config['data']['shapefiles_dir']
        }

    def get_output_dirs(self) -> Dict[str, str]:
        """Get output directory paths."""
        tables_dir = Path(self._config['outputs']['tables_dir'])
        return {
            'figures': self._config['outputs']['figures_dir'],
            'tables': self._config['outputs']['tables_dir'],
            'tables_latex': str(tables_dir / 'latex'),
            'reports': self._config['outputs']['reports_dir']
        }

    def get_model_config(self, model_name: str) -> Dict[str, Any]:
        """Get configuration for a specific model."""
        return self._config['models'].get(model_name, {})

    def get_parties(self, election_type: str) -> list:
        """
        Get list of parties for a given election type.

        Args:
            election_type: 'primera_vuelta' or 'ballotage'

        Returns:
            List of party codes
        """
        return self._config['parties'].get(election_type, [])

    def ensure_directories(self):
        """Create all necessary directories if they don't exist."""
        dirs_to_create = [
            self._config['data']['raw_dir'],
            self._config['data']['processed_dir'],
            self._config['data']['shapefiles_dir'],
            self._config['outputs']['figures_dir'],
            self._config['outputs']['tables_dir'],
            self._config['outputs']['reports_dir'],
        ]

        for dir_path in dirs_to_create:
            Path(dir_path).mkdir(parents=True, exist_ok=True)

    @property
    def project_root_path(self) -> Path:
        """Get project root directory as Path object."""
        return self.project_root

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-style access."""
        return self._config[key]

    def __repr__(self) -> str:
        return f"Config(config_path='{self.config_path}')"


# Global config instance
_global_config = None


def get_config(config_path: str = None) -> Config:
    """
    Get global configuration instance.

    Args:
        config_path: Path to config.yaml (only used on first call)

    Returns:
        Config instance
    """
    global _global_config
    if _global_config is None:
        _global_config = Config(config_path)
    return _global_config


def reload_config(config_path: str = None) -> Config:
    """
    Force reload of configuration.

    Args:
        config_path: Path to config.yaml

    Returns:
        New Config instance
    """
    global _global_config
    _global_config = Config(config_path)
    return _global_config
