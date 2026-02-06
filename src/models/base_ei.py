"""
Base class for ecological inference models.
Defines the interface that all EI models must implement.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
import pandas as pd
import numpy as np
from pathlib import Path

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.utils import get_logger

logger = get_logger(__name__)


class BaseEIModel(ABC):
    """
    Abstract base class for ecological inference models.

    All EI models must implement this interface to ensure consistency
    and interoperability.
    """

    def __init__(self, name: str = "BaseEI"):
        """
        Initialize base EI model.

        Args:
            name: Name of the model
        """
        self.name = name
        self.is_fitted = False
        self.results_ = None

    @abstractmethod
    def fit(self, data: pd.DataFrame,
            origin_cols: List[str],
            destination_cols: List[str],
            total_origin: str,
            total_destination: str) -> 'BaseEIModel':
        """
        Fit the ecological inference model.

        Args:
            data: DataFrame with circuit-level vote data
            origin_cols: Column names for origin election parties
            destination_cols: Column names for destination election parties
            total_origin: Column name for total votes in origin election
            total_destination: Column name for total votes in destination election

        Returns:
            Self (fitted model)

        Example:
            >>> model.fit(
            ...     data=df,
            ...     origin_cols=['fa_primera', 'pn_primera', 'pc_primera', 'ca_primera', 'otros_primera'],
            ...     destination_cols=['fa_ballotage', 'pn_ballotage', 'blancos_ballotage'],
            ...     total_origin='total_primera',
            ...     total_destination='total_ballotage'
            ... )
        """
        pass

    @abstractmethod
    def get_transition_matrix(self) -> np.ndarray:
        """
        Get the estimated transition matrix.

        Returns:
            Transition matrix as numpy array.
            Shape: (n_origin_parties, n_destination_parties)
            Each row sums to 1.0
            matrix[i, j] = proportion of party i voters who voted for party j

        Raises:
            ValueError: If model not fitted yet
        """
        pass

    @abstractmethod
    def get_uncertainty(self) -> Dict[str, np.ndarray]:
        """
        Get uncertainty estimates for the transition matrix.

        Returns:
            Dictionary with uncertainty information:
            - 'lower_bound': Lower confidence/credible interval (2D array)
            - 'upper_bound': Upper confidence/credible interval (2D array)
            - 'std_error': Standard error for each cell (2D array) [optional]
            - 'credible_interval': Credible interval width (scalar) [for Bayesian models]

        Raises:
            ValueError: If model not fitted yet
        """
        pass

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Predict destination votes from origin votes using the transition matrix.

        Args:
            data: DataFrame with origin vote data

        Returns:
            Predicted destination votes as numpy array

        Raises:
            ValueError: If model not fitted yet
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} model not fitted yet. Call fit() first.")

        # Get transition matrix
        T = self.get_transition_matrix()

        # Extract origin votes as matrix
        origin_votes = data[self.origin_cols_].values

        # Predict: destination = origin @ transition_matrix
        predicted = origin_votes @ T

        return predicted

    def validate_inputs(self, data: pd.DataFrame,
                       origin_cols: List[str],
                       destination_cols: List[str],
                       total_origin: str,
                       total_destination: str) -> None:
        """
        Validate input data for fitting.

        Args:
            data: DataFrame with vote data
            origin_cols: Origin election columns
            destination_cols: Destination election columns
            total_origin: Total origin votes column
            total_destination: Total destination votes column

        Raises:
            ValueError: If inputs are invalid
        """
        # Check required columns exist
        required_cols = origin_cols + destination_cols + [total_origin, total_destination]
        missing_cols = set(required_cols) - set(data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Check for negative values
        for col in origin_cols + destination_cols:
            if (data[col] < 0).any():
                raise ValueError(f"Column {col} contains negative values")

        # Check for NaN values
        for col in required_cols:
            if data[col].isna().any():
                n_nan = data[col].isna().sum()
                logger.warning(f"Column {col} has {n_nan} NaN values (will be dropped)")

        # Check dimensions match
        if len(data) == 0:
            raise ValueError("Input data is empty")

        logger.info(f"Input validation passed: {len(data)} circuits, "
                   f"{len(origin_cols)} origin parties, "
                   f"{len(destination_cols)} destination parties")

    def get_party_names(self) -> Tuple[List[str], List[str]]:
        """
        Get origin and destination party names.

        Returns:
            Tuple of (origin_party_names, destination_party_names)

        Raises:
            ValueError: If model not fitted yet
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} model not fitted yet. Call fit() first.")

        return self.origin_party_names_, self.destination_party_names_

    def get_results_dataframe(self) -> pd.DataFrame:
        """
        Get transition matrix as a formatted DataFrame.

        Returns:
            DataFrame with transition matrix
            Rows: origin parties
            Columns: destination parties

        Raises:
            ValueError: If model not fitted yet
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} model not fitted yet. Call fit() first.")

        T = self.get_transition_matrix()

        df = pd.DataFrame(
            T,
            index=self.origin_party_names_,
            columns=self.destination_party_names_
        )

        return df

    def get_results_summary(self) -> str:
        """
        Get a text summary of the results.

        Returns:
            Formatted string with results summary
        """
        if not self.is_fitted:
            return f"{self.name} model not fitted yet."

        summary = []
        summary.append(f"\n{'='*60}")
        summary.append(f"{self.name} - Transition Matrix")
        summary.append(f"{'='*60}\n")

        df = self.get_results_dataframe()
        summary.append(df.to_string())

        summary.append(f"\n{'='*60}")
        summary.append("Row sums (should be ~1.0):")
        summary.append(f"{'='*60}")
        row_sums = self.get_transition_matrix().sum(axis=1)
        for i, (party, row_sum) in enumerate(zip(self.origin_party_names_, row_sums)):
            summary.append(f"  {party}: {row_sum:.6f}")

        return "\n".join(summary)

    def save_results(self, output_path: str) -> None:
        """
        Save model results to file.

        Args:
            output_path: Path to save results (supports .csv, .parquet, .json)
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} model not fitted yet. Call fit() first.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = self.get_results_dataframe()

        if output_path.suffix == '.csv':
            df.to_csv(output_path)
        elif output_path.suffix == '.parquet':
            df.to_parquet(output_path)
        elif output_path.suffix == '.json':
            df.to_json(output_path, indent=2)
        else:
            raise ValueError(f"Unsupported output format: {output_path.suffix}")

        logger.info(f"Results saved to: {output_path}")

    def __repr__(self) -> str:
        """String representation of the model."""
        status = "fitted" if self.is_fitted else "not fitted"
        return f"{self.name}({status})"

    def __str__(self) -> str:
        """String representation for printing."""
        return self.get_results_summary() if self.is_fitted else repr(self)
