"""
Data validation utilities for electoral inference analysis.
Validates data integrity, bounds, and consistency checks.
"""

import numpy as np
import pandas as pd
from typing import List, Tuple, Optional, Dict
from .logger import get_logger
from .config import get_config

logger = get_logger(__name__)


class ValidationError(Exception):
    """Custom exception for validation errors."""
    pass


def validate_vote_counts(df: pd.DataFrame, vote_columns: List[str],
                        total_column: str = 'total') -> Tuple[bool, List[str]]:
    """
    Validate that vote counts sum to total and are non-negative.

    Args:
        df: DataFrame with vote data
        vote_columns: List of column names containing vote counts
        total_column: Name of total votes column

    Returns:
        Tuple of (is_valid, list_of_errors)

    Example:
        >>> is_valid, errors = validate_vote_counts(df, ['fa', 'pn', 'pc'], 'total')
    """
    errors = []
    config = get_config()
    tolerance = config.get('validation.tolerance', 0.001)

    # Check for negative values
    for col in vote_columns + [total_column]:
        if col not in df.columns:
            errors.append(f"Column '{col}' not found in DataFrame")
            continue

        negative_mask = df[col] < 0
        if negative_mask.any():
            n_negative = negative_mask.sum()
            errors.append(
                f"Column '{col}' has {n_negative} negative values"
            )

    # Check if votes sum to total
    if total_column in df.columns and all(col in df.columns for col in vote_columns):
        vote_sum = df[vote_columns].sum(axis=1)
        total = df[total_column]

        diff = np.abs(vote_sum - total)
        mismatch_mask = diff > tolerance

        if mismatch_mask.any():
            n_mismatch = mismatch_mask.sum()
            max_diff = diff[mismatch_mask].max()
            errors.append(
                f"{n_mismatch} rows have vote sum != total "
                f"(max difference: {max_diff:.2f})"
            )

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_proportions(df: pd.DataFrame, proportion_columns: List[str],
                        allow_nan: bool = False) -> Tuple[bool, List[str]]:
    """
    Validate that proportions are in [0, 1] range.

    Args:
        df: DataFrame with proportion data
        proportion_columns: List of column names containing proportions
        allow_nan: Whether to allow NaN values

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []
    config = get_config()
    min_val = config.get('validation.min_proportion', 0.0)
    max_val = config.get('validation.max_proportion', 1.0)

    for col in proportion_columns:
        if col not in df.columns:
            errors.append(f"Column '{col}' not found in DataFrame")
            continue

        values = df[col]

        # Check for NaN
        if values.isna().any() and not allow_nan:
            n_nan = values.isna().sum()
            errors.append(f"Column '{col}' has {n_nan} NaN values")

        # Check bounds (excluding NaN)
        valid_values = values.dropna()

        below_min = valid_values < min_val
        if below_min.any():
            n_below = below_min.sum()
            min_found = valid_values[below_min].min()
            errors.append(
                f"Column '{col}' has {n_below} values < {min_val} "
                f"(min: {min_found:.6f})"
            )

        above_max = valid_values > max_val
        if above_max.any():
            n_above = above_max.sum()
            max_found = valid_values[above_max].max()
            errors.append(
                f"Column '{col}' has {n_above} values > {max_val} "
                f"(max: {max_found:.6f})"
            )

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_row_sums(matrix: np.ndarray, expected_sum: float = 1.0,
                     tolerance: float = None) -> Tuple[bool, np.ndarray]:
    """
    Validate that each row of a matrix sums to expected value.

    Args:
        matrix: 2D numpy array
        expected_sum: Expected sum for each row (default 1.0 for probabilities)
        tolerance: Tolerance for comparison (if None, reads from config)

    Returns:
        Tuple of (is_valid, array_of_differences)
    """
    if tolerance is None:
        config = get_config()
        tolerance = config.get('validation.tolerance', 0.001)

    row_sums = matrix.sum(axis=1)
    differences = np.abs(row_sums - expected_sum)
    is_valid = np.all(differences <= tolerance)

    return is_valid, differences


def validate_transition_matrix(matrix: np.ndarray,
                               row_labels: List[str] = None,
                               col_labels: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate a transition matrix for ecological inference.

    Args:
        matrix: Transition matrix (rows=origin, cols=destination)
        row_labels: Labels for rows (for error messages)
        col_labels: Labels for columns (for error messages)

    Returns:
        Tuple of (is_valid, list_of_errors)

    Checks:
        1. All values in [0, 1]
        2. Each row sums to 1.0
        3. No NaN or inf values
    """
    errors = []
    config = get_config()
    tolerance = config.get('validation.tolerance', 0.001)

    # Check for NaN or inf
    if np.isnan(matrix).any():
        n_nan = np.isnan(matrix).sum()
        errors.append(f"Matrix contains {n_nan} NaN values")

    if np.isinf(matrix).any():
        n_inf = np.isinf(matrix).sum()
        errors.append(f"Matrix contains {n_inf} infinite values")

    # Check bounds [0, 1]
    if (matrix < 0).any():
        n_negative = (matrix < 0).sum()
        errors.append(f"Matrix contains {n_negative} negative values")

    if (matrix > 1).any():
        n_above = (matrix > 1).sum()
        errors.append(f"Matrix contains {n_above} values > 1")

    # Check row sums
    is_valid_sums, differences = validate_row_sums(matrix, 1.0, tolerance)
    if not is_valid_sums:
        bad_rows = np.where(differences > tolerance)[0]
        for i in bad_rows:
            row_label = row_labels[i] if row_labels else f"Row {i}"
            row_sum = matrix[i, :].sum()
            errors.append(
                f"{row_label}: sum = {row_sum:.6f} (diff = {differences[i]:.6f})"
            )

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_circuit_data(df: pd.DataFrame, required_columns: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate circuit-level electoral data.

    Args:
        df: DataFrame with circuit data
        required_columns: List of required column names

    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    errors = []

    # Check required columns
    missing_cols = set(required_columns) - set(df.columns)
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
        return False, errors

    # Check for duplicate circuits
    if 'circuito_id' in df.columns:
        duplicates = df['circuito_id'].duplicated()
        if duplicates.any():
            n_dup = duplicates.sum()
            dup_ids = df.loc[duplicates, 'circuito_id'].tolist()
            errors.append(
                f"Found {n_dup} duplicate circuit IDs: {dup_ids[:10]}"
            )

    # Check minimum votes per circuit
    if 'total_primera' in df.columns:
        config = get_config()
        min_votes = config.get('validation.min_votes_per_circuit', 10)

        low_turnout = df['total_primera'] < min_votes
        if low_turnout.any():
            n_low = low_turnout.sum()
            logger.warning(
                f"{n_low} circuits have fewer than {min_votes} votes "
                f"(may want to exclude from analysis)"
            )

    is_valid = len(errors) == 0
    return is_valid, errors


def validate_aggregation_consistency(df: pd.DataFrame,
                                     official_totals: Dict[str, int]) -> Tuple[bool, List[str]]:
    """
    Validate that aggregated totals match official national results.

    Args:
        df: DataFrame with circuit-level data
        official_totals: Dictionary of official vote totals by party

    Returns:
        Tuple of (is_valid, list_of_warnings)
    """
    warnings = []
    config = get_config()
    tolerance = config.get('validation.tolerance', 0.001)

    for party, official_total in official_totals.items():
        col_name = f"{party.lower()}_primera"
        if col_name in df.columns:
            calculated_total = df[col_name].sum()
            diff = abs(calculated_total - official_total)
            rel_diff = diff / official_total if official_total > 0 else 0

            if rel_diff > tolerance:
                warnings.append(
                    f"{party}: calculated={calculated_total:,}, "
                    f"official={official_total:,}, "
                    f"diff={diff:,} ({rel_diff:.2%})"
                )

    is_valid = len(warnings) == 0
    return is_valid, warnings


def check_data_quality(df: pd.DataFrame) -> Dict[str, any]:
    """
    Comprehensive data quality check.

    Args:
        df: DataFrame to check

    Returns:
        Dictionary with quality metrics
    """
    quality_report = {
        'n_rows': len(df),
        'n_columns': len(df.columns),
        'missing_values': {},
        'duplicates': 0,
        'numeric_columns': [],
        'categorical_columns': []
    }

    # Missing values
    for col in df.columns:
        n_missing = df[col].isna().sum()
        if n_missing > 0:
            quality_report['missing_values'][col] = {
                'count': n_missing,
                'percentage': n_missing / len(df) * 100
            }

    # Duplicates
    if 'circuito_id' in df.columns:
        quality_report['duplicates'] = df['circuito_id'].duplicated().sum()

    # Column types
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            quality_report['numeric_columns'].append(col)
        elif pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_categorical_dtype(df[col]):
            quality_report['categorical_columns'].append(col)

    return quality_report


def log_validation_results(validation_name: str, is_valid: bool,
                          errors_or_warnings: List[str]):
    """
    Log validation results in a standardized format.

    Args:
        validation_name: Name of the validation check
        is_valid: Whether validation passed
        errors_or_warnings: List of error/warning messages
    """
    if is_valid:
        logger.info(f"✓ {validation_name}: PASSED")
    else:
        logger.error(f"✗ {validation_name}: FAILED")
        for msg in errors_or_warnings:
            logger.error(f"  - {msg}")
