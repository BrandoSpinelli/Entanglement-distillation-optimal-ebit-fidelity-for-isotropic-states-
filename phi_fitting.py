"""
Phi Fitting Methods Module

This module contains all the different methods for estimating the error exponent φ(f)
from quantum fidelity optimization data.

Methods:
    - fit_phi_nonlinear: Nonlinear curve fitting using F_max model
    - phi_fit_linear: Weighted linear regression approach
    - estimate_phi_numerical: Direct numerical estimation
    - phi_fit_asymptotic: Asymptotic analysis using high k values
    - phi_fit_extrapolation: Extrapolation to infinite k limit
"""

import numpy as np
from scipy.optimize import curve_fit
from scipy.stats import linregress
from loguru import logger
import matplotlib.pyplot as plt
from quantum_core import Fmax_model
from data_processing import extract_valid_log_data, perform_weighted_linear_regression


def fit_phi_nonlinear(fidelity_values: np.ndarray, Fmax_grid: dict) -> tuple:
    """
    Fit the function F_max(k, f) with the model 1 - C(f) * 2^(-k * phi(f))
    to estimate C(f) and phi(f) for each fidelity value.

    Args:
        fidelity_values (np.ndarray): Array of fidelity values.
        Fmax_grid (dict): Dictionary mapping k to F_max arrays.

    Returns:
        tuple: Two arrays containing the fitted values of C(f) and φ(f).
    """
    k_values = np.array(sorted(Fmax_grid.keys()))
    phi_list, C_list = [], []

    for i, f in enumerate(fidelity_values):
        # Extract F_max values at fixed f across all ks
        y = np.array([Fmax_grid[k][i] for k in k_values])

        # Remove invalid points (NaN or converging to 1)
        mask = (~np.isnan(y)) & (y < 1.1)
        if mask.sum() < 3:
            # Not enough points to perform a meaningful fit
            phi_list.append(np.nan)
            C_list.append(np.nan)
            continue

        ks = k_values[mask]
        y = y[mask]

        try:
            # Fit the model using non-linear least squares
            popt, _ = curve_fit(Fmax_model, ks, y, p0=(0.5, 0.1), bounds=([0, 0], [1, 10]))
            C, phi = popt
            C_list.append(C)
            phi_list.append(phi)
        except Exception as e:
            logger.warning(f"Fit failed at f={f:.3f}: {e}")
            phi_list.append(np.nan)
            C_list.append(np.nan)
        
    # Optional: Plot C(f) for validation
    plt.figure(figsize=(6, 4))
    plt.plot(fidelity_values, C_list, label='Estimated C(f)')
    plt.axhline(1.0, color='r', linestyle='--', label='C = 1')
    plt.xlabel('Fidelity f')
    plt.ylabel('C parameter')
    plt.title('Normalization Parameter C(f)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()

    return np.array(C_list), np.array(phi_list)


def phi_fit_linear(fidelity_values: np.ndarray, Fmax_grid: dict) -> np.ndarray:
    """
    Estimate φ(f) using weighted linear regression on log-transformed data.
    
    This method fits the linear model: log₂(1 - F_max(k,f)) ≈ -k * φ(f)
    using k² weighting to emphasize asymptotic behavior (higher k values).
    
    The weighting scheme favors data points at large k, which are more relevant
    for the theoretical asymptotic limit k → ∞.
    
    Args:
        fidelity_values (np.ndarray): Array of fidelity values f ∈ [0.5, 1].
        Fmax_grid (dict): Dictionary mapping k → array of F_max values.
        
    Returns:
        np.ndarray: Estimated φ(f) values using weighted linear regression.
    """
    k_values = np.array(sorted(Fmax_grid.keys()))
    phi_estimates = []
    
    # Process data for all fidelity values
    processed_data = extract_valid_log_data(fidelity_values, Fmax_grid, k_values)
    
    for i, y_vals, ks in processed_data:
        if len(ks) < 3:  # Need minimum points for reliable regression
            phi_estimates.append(np.nan)
            continue
        
        # Perform weighted linear regression
        phi_est = perform_weighted_linear_regression(ks, y_vals)
        phi_estimates.append(phi_est)

    return np.array(phi_estimates)


def estimate_phi_numerical(fidelity_values: np.ndarray, Fmax_grid: dict) -> np.ndarray:
    """
    Estimate φ(f) using direct numerical calculation from individual k values.
    
    This method computes φ(f) ≈ -(1/k) * log₂(1 - F_max(k,f)) for each k,
    then averages the estimates across all valid k values.
    
    Unlike fitting methods, this approach provides a more direct (though potentially
    noisier) estimate by directly applying the theoretical relationship.
    
    Args:
        fidelity_values (np.ndarray): Array of fidelity values f ∈ [0.5, 1].
        Fmax_grid (dict): Dictionary mapping k → arrays of F_max values.
        
    Returns:
        np.ndarray: Numerically estimated φ(f) values.
    """
    k_values = np.array(sorted(Fmax_grid.keys()))
    phi_estimates = []

    for i in range(len(fidelity_values)):
        phi_k = []
        
        for k in k_values:
            Fmax = Fmax_grid[k][i]
            
            # Skip invalid or near-perfect fidelities
            if np.isnan(Fmax) or Fmax >= 0.999:
                continue
            
            # Safe logarithm calculation with epsilon protection
            epsilon = 1e-15
            stable_val = max(1 - Fmax, epsilon)
            phi_val = -np.log2(stable_val) / k
            
            # Keep reasonable phi values (positive and bounded)
            if phi_val > 0 and phi_val < 10:
                phi_k.append(phi_val)
        
        # Average valid estimates or return NaN if no valid data
        if len(phi_k) == 0:
            phi_estimates.append(np.nan)
        else:
            phi_estimates.append(np.mean(phi_k))

    return np.array(phi_estimates)


def phi_fit_asymptotic(fidelity_values: np.ndarray, Fmax_grid: dict, k_min_threshold: int = 30) -> np.ndarray:
    """
    Estimate φ(f) focusing on asymptotic behavior using only high k values.
    
    This method restricts the analysis to k ≥ k_min_threshold to focus on the
    asymptotic regime where the theoretical limit k → ∞ is most relevant.
    
    By filtering out low k values, this approach reduces finite-size effects
    and provides estimates closer to the true asymptotic behavior.
    
    Args:
        fidelity_values (np.ndarray): Array of fidelity values f ∈ [0.5, 1].
        Fmax_grid (dict): Dictionary mapping k → array of F_max values.
        k_min_threshold (int): Minimum k value to include (default: 30).
        
    Returns:
        np.ndarray: Asymptotic φ(f) estimates using only high k values.
    """
    k_values = np.array(sorted(Fmax_grid.keys()))
    
    # Filter to high k values only for asymptotic behavior
    high_k_mask = k_values >= k_min_threshold
    k_values_high = k_values[high_k_mask]
    
    if len(k_values_high) < 3:
        logger.warning(f"Insufficient high k values (>= {k_min_threshold}) for asymptotic fitting")
        return np.full(len(fidelity_values), np.nan)
    
    # Process only high k data
    processed_data = extract_valid_log_data(fidelity_values, Fmax_grid, k_values_high)
    phi_estimates = []

    for i, y_vals, ks in processed_data:
        if len(ks) < 2:  # Need minimum 2 points for asymptotic estimate
            phi_estimates.append(np.nan)
            continue
        
        try:
            if len(ks) >= 3:
                # Linear regression for asymptotic slope
                slope, _, _, _, _ = linregress(ks, y_vals)
                phi_estimates.append(-slope)
            else:
                # Direct asymptotic estimate using highest available k
                k_max = max(ks)
                idx_max = ks.index(k_max)
                phi_direct = -y_vals[idx_max] / k_max
                phi_estimates.append(phi_direct)
                
        except:
            phi_estimates.append(np.nan)

    return np.array(phi_estimates)


def phi_fit_extrapolation(fidelity_values: np.ndarray, Fmax_grid: dict) -> np.ndarray:
    """
    Estimate φ(f) by extrapolating φ(k) → φ(∞) using asymptotic modeling.
    
    This sophisticated method first computes φ(k) for each k value, then fits
    the asymptotic model φ(k) = φ_∞ + A/k to extrapolate the infinite-k limit.
    
    The extrapolation approach accounts for finite-size corrections and provides
    the most theoretically sound estimate of the asymptotic error exponent.
    
    Args:
        fidelity_values (np.ndarray): Array of fidelity values f ∈ [0.5, 1].
        Fmax_grid (dict): Dictionary mapping k → array of F_max values.
        
    Returns:
        np.ndarray: Extrapolated φ(∞) estimates.
    """
    k_values = np.array(sorted(Fmax_grid.keys()))
    phi_estimates = []

    for i in range(len(fidelity_values)):
        phi_k_values = []
        valid_ks = []
        
        # Calculate φ(k) for each k value
        for k in k_values:
            Fmax = Fmax_grid[k][i]
            
            # Skip invalid or near-perfect fidelities
            if np.isnan(Fmax) or Fmax >= 0.999:
                continue
            
            # Compute φ(k) estimate for this specific k
            epsilon = 1e-4
            log_val = np.log2(max(1 - Fmax, epsilon))
            
            if log_val > -50 and log_val < 0:
                phi_k = -log_val / k  # φ(k) estimate
                if phi_k > 0 and phi_k < 10:
                    phi_k_values.append(phi_k)
                    valid_ks.append(k)
        
        if len(valid_ks) < 3:
            phi_estimates.append(np.nan)
            continue
        
        try:
            # Fit asymptotic model: φ(k) = φ_∞ + A/k
            def phi_model(k, phi_inf, A):
                """Asymptotic model with 1/k correction term."""
                return phi_inf + A / k
            
            # Perform curve fitting with reasonable bounds
            popt, _ = curve_fit(phi_model, valid_ks, phi_k_values, 
                              p0=(0.5, 1.0), bounds=([0, -10], [10, 10]))
            phi_infinity = popt[0]  # Extract φ_∞
            phi_estimates.append(phi_infinity)
            
        except:
            # Fallback: use highest k value as best approximation
            max_k_idx = np.argmax(valid_ks)
            phi_estimates.append(phi_k_values[max_k_idx])

    return np.array(phi_estimates)
