"""
Data Processing Module

This module contains utility functions for data extraction, validation, 
and statistical processing used across multiple fitting methods.

Functions:
    - scan_fidelity_grid: Evaluate F_max over a grid of fidelities
    - compute_Fmax_grid: Compute F_max grid for multiple k values
    - extract_valid_log_data: Extract and validate logarithmic data
    - perform_weighted_linear_regression: Weighted regression with k² weighting
"""

import numpy as np
from scipy.stats import linregress
from loguru import logger
from quantum_core import generate_constraint_matrix_vectorized, solve_linear_program


def scan_fidelity_grid(k: int, fidelity_values: np.ndarray) -> tuple:
    """
    Evaluate the maximum fidelity (F_max) over a grid of initial fidelities.

    Args:
        k (int): Number of tensor copies of the state.
        fidelity_values (np.ndarray): Array of fidelity values to evaluate.

    Returns:
        tuple: Tuple (F_values, Fmax_values) containing:
            - F_values: np.ndarray of fidelity values
            - Fmax_values: np.ndarray of corresponding maximum fidelities
    """
    Fmax_values = []

    A = generate_constraint_matrix_vectorized(k)
    logger.info(f"Constraint matrix generated for k = {k}")

    for F in fidelity_values:
        F_max = solve_linear_program(F, k, A)
        Fmax_values.append(F_max)
        logger.debug(f"F = {F:.3f} -> F_max = {F_max:.6f}")

    return fidelity_values, np.array(Fmax_values)


def compute_Fmax_grid(fidelity_values: np.ndarray, k_list: list[int]) -> dict:
    """
    Compute the F_max values for each fidelity and for each k in the list.

    Args:
        fidelity_values (np.ndarray): Array of fidelity values f ∈ [0.5, 1].
        k_list (list[int]): List of tensor copy counts (k values).

    Returns:
        dict: Dictionary mapping each k to a list of corresponding F_max values.
    """
    Fmax_dict = {}
    for k in k_list:
        A = generate_constraint_matrix_vectorized(k)  # Precompute constraint matrix
        logger.info(f"Constraint matrix for k={k} computed.")
        Fmax_k = []
        for f in fidelity_values:
            Fmax = solve_linear_program(f, k, A)  # Solve LP for each f
            Fmax_k.append(Fmax)
        Fmax_dict[k] = Fmax_k
    return Fmax_dict


def extract_valid_log_data(fidelity_values: np.ndarray, Fmax_grid: dict, 
                          k_values: np.ndarray) -> list:
    """
    Extract valid logarithmic data points for phi estimation.
    
    This helper function standardizes the data extraction and validation process
    used across multiple fitting methods, ensuring consistent handling of edge cases.
    
    Args:
        fidelity_values (np.ndarray): Array of fidelity values.
        Fmax_grid (dict): Dictionary mapping k to arrays of F_max values.
        k_values (np.ndarray): Array of k values to process.
        
    Returns:
        list: List of tuples (i, y_vals, ks) where:
            - i: fidelity index
            - y_vals: valid log2(1 - F_max) values
            - ks: corresponding k values
    """
    processed_data = []
    
    for i in range(len(fidelity_values)):
        y_vals, ks = [], []
        
        for k in k_values:
            Fmax = Fmax_grid[k][i]
            
            # Skip invalid or near-perfect fidelities
            if np.isnan(Fmax) or Fmax >= 0.999:
                continue
            
            # Safe logarithm calculation with epsilon protection
            epsilon = 1e-4
            log_val = np.log2(max(1 - Fmax, epsilon))
            
            # Filter reasonable log values (avoid extreme outliers)
            if log_val > -50 and log_val < 0:
                y_vals.append(log_val)
                ks.append(k)
        
        processed_data.append((i, y_vals, ks))
    
    return processed_data


def perform_weighted_linear_regression(ks: list, y_vals: list) -> float:
    """
    Perform weighted linear regression with k² weighting.
    
    Higher k values receive more weight in the regression to emphasize
    asymptotic behavior, which is more relevant for the theoretical limit.
    
    Args:
        ks (list): K values (independent variable).
        y_vals (list): Log values (dependent variable).
        
    Returns:
        float: Estimated slope (phi value) or NaN if regression fails.
    """
    try:
        # k² weighting favors high k values for asymptotic behavior
        weights = np.array(ks) ** 2
        
        # Weighted least squares: (X^T W X)^{-1} X^T W y
        W = np.diag(weights)
        X = np.column_stack([ks, np.ones(len(ks))])  # Design matrix [k, 1]
        y = np.array(y_vals)
        
        # Solve weighted normal equation
        XTW = X.T @ W
        beta = np.linalg.solve(XTW @ X, XTW @ y)
        
        return -beta[0]  # Return -slope as phi estimate
        
    except:
        # Fallback to standard linear regression
        try:
            slope, _, _, _, _ = linregress(ks, y_vals)
            return -slope
        except:
            return np.nan
