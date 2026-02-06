"""
Goodman Ecological Regression implementation.
Simple OLS regression-based method for ecological inference (baseline).
"""

from typing import Dict, List
import pandas as pd
import numpy as np
from pathlib import Path

# Try to import statsmodels, fall back to sklearn if not available
try:
    import statsmodels.api as sm
    HAS_STATSMODELS = True
except ImportError:
    from sklearn.linear_model import LinearRegression
    HAS_STATSMODELS = False
    import warnings
    warnings.warn("statsmodels not available, using sklearn LinearRegression instead")

import sys
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.models.base_ei import BaseEIModel
from src.utils import get_logger

logger = get_logger(__name__)


class GoodmanRegression(BaseEIModel):
    """
    Goodman Ecological Regression model.

    Uses OLS regression to estimate transition probabilities:
        votes_destination_j = β0 + Σ_i β_i * votes_origin_i

    Where β_i represents the proportion of origin party i voters who voted for destination party j.

    Limitations:
    - May produce estimates outside [0, 1] bounds
    - Assumes linear relationship
    - No uncertainty quantification beyond standard errors
    - Aggregation bias

    Use as baseline for comparison with King's EI.
    """

    def __init__(self, add_constant: bool = True):
        """
        Initialize Goodman Regression model.

        Args:
            add_constant: Whether to add intercept term (default True)
        """
        super().__init__(name="GoodmanRegression")
        self.add_constant = add_constant
        self.models_ = {}  # Store one model per destination party

    def fit(self, data: pd.DataFrame,
            origin_cols: List[str],
            destination_cols: List[str],
            total_origin: str,
            total_destination: str) -> 'GoodmanRegression':
        """
        Fit Goodman regression model.

        Fits separate regressions for each destination party:
            destination_j ~ origin_1 + origin_2 + ... + origin_k

        Args:
            data: DataFrame with circuit-level vote data
            origin_cols: Column names for origin election parties
            destination_cols: Column names for destination election parties
            total_origin: Column name for total votes in origin election (not used)
            total_destination: Column name for total votes in destination election (not used)

        Returns:
            Self (fitted model)
        """
        # Validate inputs
        self.validate_inputs(data, origin_cols, destination_cols, total_origin, total_destination)

        # Store column names
        self.origin_cols_ = origin_cols
        self.destination_cols_ = destination_cols
        self.origin_party_names_ = [col.replace('_primera', '').upper() for col in origin_cols]
        self.destination_party_names_ = [col.replace('_ballotage', '').upper() for col in destination_cols]

        # Drop rows with NaN values
        data_clean = data[origin_cols + destination_cols].dropna()
        logger.info(f"Fitting on {len(data_clean)} circuits (dropped {len(data) - len(data_clean)} with NaN)")

        # Extract data
        X = data_clean[origin_cols].values
        n_samples, n_origin = X.shape
        n_dest = len(destination_cols)

        # Add constant if requested
        if self.add_constant:
            if HAS_STATSMODELS:
                X_reg = sm.add_constant(X)
            else:
                X_reg = np.hstack([np.ones((n_samples, 1)), X])
        else:
            X_reg = X

        # Fit separate regression for each destination party
        logger.info(f"Fitting {n_dest} regressions (one per destination party)...")

        self.coefficients_ = np.zeros((n_origin, n_dest))
        self.intercepts_ = np.zeros(n_dest) if self.add_constant else None
        self.r_squared_ = np.zeros(n_dest)
        self.std_errors_ = np.zeros((n_origin, n_dest))

        for j, dest_col in enumerate(destination_cols):
            y = data_clean[dest_col].values

            if HAS_STATSMODELS:
                # Fit with statsmodels (provides more diagnostics)
                model = sm.OLS(y, X_reg)
                results = model.fit()

                if self.add_constant:
                    self.intercepts_[j] = results.params[0]
                    self.coefficients_[:, j] = results.params[1:]
                    self.std_errors_[:, j] = results.bse[1:]
                else:
                    self.coefficients_[:, j] = results.params
                    self.std_errors_[:, j] = results.bse

                self.r_squared_[j] = results.rsquared
                self.models_[dest_col] = results

                logger.info(f"  {dest_col}: R² = {results.rsquared:.4f}")

            else:
                # Fit with sklearn
                model = LinearRegression(fit_intercept=self.add_constant)
                model.fit(X, y)

                self.coefficients_[:, j] = model.coef_
                if self.add_constant:
                    self.intercepts_[j] = model.intercept_

                # Calculate R²
                y_pred = model.predict(X)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                self.r_squared_[j] = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

                self.models_[dest_col] = model

                logger.info(f"  {dest_col}: R² = {self.r_squared_[j]:.4f}")

                # Std errors not available with sklearn
                self.std_errors_[:, j] = np.nan

        self.is_fitted = True

        # Check for bound violations
        self._check_bounds()

        return self

    def get_transition_matrix(self) -> np.ndarray:
        """
        Get the estimated transition matrix.

        Returns:
            Transition matrix as numpy array.
            Shape: (n_origin_parties, n_destination_parties)

        Note:
            Goodman regression may produce values outside [0, 1].
            Values are NOT normalized to sum to 1.0 per row.
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} model not fitted yet. Call fit() first.")

        return self.coefficients_

    def get_transition_matrix_normalized(self) -> np.ndarray:
        """
        Get normalized transition matrix (rows sum to 1).

        Returns:
            Normalized transition matrix

        Note:
            Normalization may not be appropriate if coefficients are negative.
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} model not fitted yet. Call fit() first.")

        T = self.coefficients_.copy()

        # Clip negative values to 0 before normalizing
        T_clipped = np.maximum(T, 0)

        # Normalize rows to sum to 1
        row_sums = T_clipped.sum(axis=1, keepdims=True)
        T_normalized = T_clipped / np.where(row_sums > 0, row_sums, 1)

        return T_normalized

    def get_uncertainty(self) -> Dict[str, np.ndarray]:
        """
        Get uncertainty estimates (standard errors from OLS).

        Returns:
            Dictionary with:
            - 'std_error': Standard error for each coefficient
            - 'lower_bound': Lower 95% CI (coef - 1.96*SE)
            - 'upper_bound': Upper 95% CI (coef + 1.96*SE)
        """
        if not self.is_fitted:
            raise ValueError(f"{self.name} model not fitted yet. Call fit() first.")

        # 95% confidence intervals
        z_score = 1.96
        lower = self.coefficients_ - z_score * self.std_errors_
        upper = self.coefficients_ + z_score * self.std_errors_

        return {
            'std_error': self.std_errors_,
            'lower_bound': lower,
            'upper_bound': upper
        }

    def _check_bounds(self) -> None:
        """Check for bound violations and issue warnings."""
        # Check for negative coefficients
        n_negative = (self.coefficients_ < 0).sum()
        if n_negative > 0:
            logger.warning(
                f"{n_negative}/{self.coefficients_.size} coefficients are negative "
                f"(min: {self.coefficients_.min():.4f})"
            )

        # Check for coefficients > 1
        n_above_one = (self.coefficients_ > 1).sum()
        if n_above_one > 0:
            logger.warning(
                f"{n_above_one}/{self.coefficients_.size} coefficients > 1.0 "
                f"(max: {self.coefficients_.max():.4f})"
            )

        # Check row sums
        row_sums = self.coefficients_.sum(axis=1)
        row_sums_off = np.abs(row_sums - 1.0) > 0.1
        if row_sums_off.any():
            logger.warning(
                f"{row_sums_off.sum()}/{len(row_sums)} rows have sum far from 1.0 "
                f"(range: [{row_sums.min():.4f}, {row_sums.max():.4f}])"
            )

    def get_results_summary(self) -> str:
        """
        Get a text summary of the results with diagnostics.

        Returns:
            Formatted string with results summary
        """
        if not self.is_fitted:
            return f"{self.name} model not fitted yet."

        summary = []
        summary.append(f"\n{'='*60}")
        summary.append(f"{self.name} - Transition Matrix (Coefficients)")
        summary.append(f"{'='*60}\n")

        df = self.get_results_dataframe()
        summary.append(df.to_string())

        summary.append(f"\n{'='*60}")
        summary.append("Model Diagnostics:")
        summary.append(f"{'='*60}")

        # R-squared for each destination
        for j, (dest_name, r2) in enumerate(zip(self.destination_party_names_, self.r_squared_)):
            summary.append(f"  {dest_name}: R² = {r2:.4f}")

        # Bound violations
        summary.append(f"\n{'='*60}")
        summary.append("Bound Violations:")
        summary.append(f"{'='*60}")
        n_negative = (self.coefficients_ < 0).sum()
        n_above_one = (self.coefficients_ > 1).sum()
        summary.append(f"  Negative coefficients: {n_negative}/{self.coefficients_.size}")
        summary.append(f"  Coefficients > 1.0: {n_above_one}/{self.coefficients_.size}")

        # Row sums
        summary.append(f"\n{'='*60}")
        summary.append("Row sums:")
        summary.append(f"{'='*60}")
        row_sums = self.coefficients_.sum(axis=1)
        for i, (party, row_sum) in enumerate(zip(self.origin_party_names_, row_sums)):
            status = "OK" if abs(row_sum - 1.0) < 0.1 else "WARNING"
            summary.append(f"  {party}: {row_sum:.6f} [{status}]")

        return "\n".join(summary)


def main():
    """Test Goodman Regression on sample data."""
    import argparse

    parser = argparse.ArgumentParser(
        description='Test Goodman Regression on electoral data'
    )
    parser.add_argument(
        '--data',
        type=str,
        default='data/processed/circuitos_merged.parquet',
        help='Path to merged data (default: data/processed/circuitos_merged.parquet)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Output path for results (optional)'
    )

    args = parser.parse_args()

    # Load data
    logger.info(f"Loading data from {args.data}...")
    df = pd.read_parquet(args.data)

    # Define columns
    origin_cols = ['ca_primera', 'fa_primera', 'otros_primera', 'pc_primera', 'pn_primera']
    destination_cols = ['fa_ballotage', 'pn_ballotage', 'blancos_ballotage']

    # Fit model
    logger.info("Fitting Goodman Regression...")
    model = GoodmanRegression(add_constant=True)
    model.fit(
        data=df,
        origin_cols=origin_cols,
        destination_cols=destination_cols,
        total_origin='total_primera',
        total_destination='total_ballotage'
    )

    # Print results
    print(model.get_results_summary())

    # Save if output path provided
    if args.output:
        model.save_results(args.output)
        print(f"\nResults saved to: {args.output}")


if __name__ == '__main__':
    main()
