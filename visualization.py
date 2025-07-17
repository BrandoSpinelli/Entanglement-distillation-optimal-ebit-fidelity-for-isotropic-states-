"""
Visualization Module

This module contains all plotting functions with consistent IEEE-style formatting
for visualizing quantum fidelity optimization results and phi estimation comparisons.

Functions:
    - IEEE style helper functions for consistent plotting
    - Individual plotting functions for each fitting method
    - Comprehensive comparison plotting
    - Fidelity vs F_max visualization
"""

import numpy as np
import matplotlib.pyplot as plt
from quantum_core import theoretical_phi


def _setup_ieee_plot_style():
    """
    Configure matplotlib with IEEE-style plotting parameters.
    
    Sets consistent styling across all plots including fonts, sizes,
    grid appearance, and line widths suitable for academic publications.
    """
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif", 
        "font.size": 11,
        "axes.labelsize": 11,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.5,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })


def _setup_ieee_axes(ax, xlabel: str, ylabel: str, title: str):
    """
    Apply consistent IEEE-style formatting to plot axes.
    
    Args:
        ax: Matplotlib axes object to format.
        xlabel (str): X-axis label text.
        ylabel (str): Y-axis label text. 
        title (str): Plot title text.
    """
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=4)
    ax.set_xlim([0.5, 1.0])
    ax.legend(loc="best", frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


def _calculate_plot_ylimits(phi_estimated: np.ndarray, phi_theory: np.ndarray) -> tuple:
    """
    Calculate appropriate y-axis limits for phi plots.
    
    Considers only valid (non-NaN) values and adds 5% padding around the data range.
    
    Args:
        phi_estimated (np.ndarray): Estimated phi values (may contain NaN).
        phi_theory (np.ndarray): Theoretical phi values.
        
    Returns:
        tuple: (ymin, ymax) for plot limits.
    """
    valid_vals = ~np.isnan(phi_estimated)
    if np.sum(valid_vals) > 0:
        ymin = np.nanmin([np.min(phi_estimated[valid_vals]), np.min(phi_theory)]) * 0.95
        ymax = np.nanmax([np.max(phi_estimated[valid_vals]), np.max(phi_theory)]) * 1.05
    else:
        ymin, ymax = np.min(phi_theory) * 0.95, np.max(phi_theory) * 1.05
    return ymin, ymax


def _create_phi_plot_template(fidelity_values: np.ndarray, phi_estimated: np.ndarray, 
                             phi_theory: np.ndarray, method_label: str, method_color: str,
                             title: str) -> None:
    """
    Create a standardized phi estimation plot with consistent formatting.
    
    This template function reduces code duplication across all phi plotting functions
    by providing a common structure and styling.
    
    Args:
        fidelity_values (np.ndarray): Input fidelity values.
        phi_estimated (np.ndarray): Estimated phi values from specific method.
        phi_theory (np.ndarray): Theoretical phi values for comparison.
        method_label (str): Label for the estimation method (e.g., "Linear Fit").
        method_color (str): Color for the estimation method plot line.
        title (str): Plot title.
    """
    _setup_ieee_plot_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # IEEE single-column size
    
    # Plot estimated values
    ax.plot(fidelity_values, phi_estimated, 'o-', color=method_color, label=method_label)
    
    # Plot theoretical bound
    ax.plot(fidelity_values, phi_theory, '--', color='black',
            label=r'Theory $\phi_{\mathrm{th}}(f) = \log \left(\frac{1}{2\sqrt{f(1-f)}}\right)$')
    
    # Setup axes and limits
    _setup_ieee_axes(ax, r"Input fidelity $f$", r"Error rate decay $\phi(f)$", title)
    ymin, ymax = _calculate_plot_ylimits(phi_estimated, phi_theory)
    ax.set_ylim([ymin, ymax])
    
    plt.tight_layout()
    plt.show()


def plot_fidelity_vs_fmax(F_values: np.ndarray, Fmax_values: np.ndarray, k: int):
    """
    Plot F_max(f) for a given k value using IEEE-style formatting.

    Args:
        F_values (np.ndarray): Array of input fidelity values.
        Fmax_values (np.ndarray): Array of maximum fidelity values.
        k (int): Number of tensor copies.
    """
    _setup_ieee_plot_style()
    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # IEEE column size

    ax.plot(
        F_values,
        Fmax_values,
        label=fr"$\mathcal{{F}}_\max(\rho_F^{{\otimes {k}}})$",
        color='black'
    )

    ax.set_xlabel(r"Initial Fidelity $F$")
    ax.set_ylabel(r"Optimized Fidelity $\mathcal{F}_{\max}$")
    ax.set_title(rf"$k={k}$", pad=4)

    ax.set_xlim([0.5, 1.0])
    ax.set_ylim([np.nanmin(Fmax_values) * 0.98, np.nanmax(Fmax_values) * 1.02])

    ax.legend(loc="best", frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


def plot_phi_fit_vs_theory(fidelity_values: np.ndarray, phi_fit: np.ndarray):
    """
    Plot nonlinear fit φ(f) estimates against theoretical bounds.
    
    Visualizes the results from nonlinear curve fitting using the model
    F_max(k,f) = 1 - C(f) * 2^(-k*φ(f)).
    
    Args:
        fidelity_values (np.ndarray): Array of input fidelity values f.
        phi_fit (np.ndarray): Estimated φ(f) from nonlinear fitting.
    """
    phi_theory = theoretical_phi(fidelity_values)
    _create_phi_plot_template(
        fidelity_values, phi_fit, phi_theory,
        method_label=r'Fitted $\phi_{\mathrm{fit}}(f)$',
        method_color='black',
        title=r"Comparison: Fit vs Theory"
    )


def plot_phi_fit_linear(fidelity_values: np.ndarray, phi_linear_fit: np.ndarray):
    """
    Plot linear fit φ(f) estimates against theoretical bounds.
    
    Visualizes results from weighted linear regression on log₂(1-F_max) vs k,
    using k² weighting to emphasize asymptotic behavior.
    
    Args:
        fidelity_values (np.ndarray): Array of input fidelity values.
        phi_linear_fit (np.ndarray): Estimated φ(f) from linear fitting.
    """
    phi_theory = theoretical_phi(fidelity_values)
    _create_phi_plot_template(
        fidelity_values, phi_linear_fit, phi_theory,
        method_label=r'Linear Fit $\phi_{\mathrm{fit}}(f)$',
        method_color='black',
        title="Linear Fit vs Theory"
    )


def plot_phi_numerical(fidelity_values: np.ndarray, phi_numerical: np.ndarray):
    """
    Plot numerical φ(f) estimates against theoretical bounds.
    
    Visualizes results from direct numerical calculation:
    φ(f) ≈ -(1/k) * log₂(1 - F_max(k,f)), averaged over k values.
    
    Args:
        fidelity_values (np.ndarray): Input fidelity values.
        phi_numerical (np.ndarray): Numerically estimated φ(f).
    """
    phi_theory = theoretical_phi(fidelity_values)
    _create_phi_plot_template(
        fidelity_values, phi_numerical, phi_theory,
        method_label=r'Numerical $\phi_{\mathrm{num}}(f)$',
        method_color='darkorange',
        title="Numerical Estimate vs Theory"
    )


def plot_phi_asymptotic(fidelity_values: np.ndarray, phi_asymptotic: np.ndarray):
    """
    Plot asymptotic φ(f) estimates against theoretical bounds.
    
    Visualizes results from fitting using only high k values (k ≥ threshold)
    to focus on the asymptotic regime where k → ∞.
    
    Args:
        fidelity_values (np.ndarray): Input fidelity values.
        phi_asymptotic (np.ndarray): Asymptotically estimated φ(f).
    """
    phi_theory = theoretical_phi(fidelity_values)
    _create_phi_plot_template(
        fidelity_values, phi_asymptotic, phi_theory,
        method_label=r'Asymptotic $\phi_{\mathrm{asym}}(f)$',
        method_color='blue',
        title="Asymptotic Estimate vs Theory"
    )


def plot_phi_extrapolation(fidelity_values: np.ndarray, phi_extrapolation: np.ndarray):
    """
    Plot extrapolated φ(f) estimates against theoretical bounds.
    
    Visualizes results from extrapolating φ(k) → φ(∞) using the model
    φ(k) = φ_∞ + A/k and extracting the asymptotic limit φ_∞.
    
    Args:
        fidelity_values (np.ndarray): Input fidelity values.
        phi_extrapolation (np.ndarray): Extrapolated φ(f).
    """
    phi_theory = theoretical_phi(fidelity_values)
    _create_phi_plot_template(
        fidelity_values, phi_extrapolation, phi_theory,
        method_label=r'Extrapolated $\phi_{\infty}(f)$',
        method_color='green',
        title="Extrapolation Estimate vs Theory"
    )


def plot_phi_comparison_all(fidelity_values: np.ndarray, 
                           phi_linear: np.ndarray,
                           phi_numerical: np.ndarray, 
                           phi_nonlinear: np.ndarray,
                           phi_asymptotic: np.ndarray,
                           phi_extrapolation: np.ndarray):
    """
    Create a comprehensive comparison plot of all φ(f) estimation methods.
    
    This unified visualization allows direct comparison of all implemented 
    estimation approaches against the theoretical upper bound, facilitating
    method evaluation and selection.
    
    Args:
        fidelity_values (np.ndarray): Input fidelity values f ∈ [0.5, 1].
        phi_linear (np.ndarray): Standard weighted linear fit estimates.
        phi_numerical (np.ndarray): Direct numerical estimates.
        phi_nonlinear (np.ndarray): Nonlinear curve fit estimates.
        phi_asymptotic (np.ndarray): High-k asymptotic estimates.
        phi_extrapolation (np.ndarray): Extrapolated infinite-k estimates.
    """
    phi_theory = theoretical_phi(fidelity_values)

    # Enhanced IEEE-style settings for comprehensive comparison
    plt.rcParams.update({
        "text.usetex": False,
        "font.family": "serif",
        "font.size": 12,
        "axes.labelsize": 12,
        "axes.titlesize": 14,
        "legend.fontsize": 10,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "lines.linewidth": 2,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--"
    })

    fig, ax = plt.subplots(figsize=(10, 7))  # Larger figure for comprehensive view

    # Plot all estimation methods with distinct styling
    method_configs = [
        (phi_linear, 'o-', 'Standard Linear Fit', 'red'),
        (phi_numerical, 's-', 'Numerical Estimate', 'orange'), 
        (phi_nonlinear, 'd-', 'Nonlinear Fit', 'purple'),
        (phi_asymptotic, '^-', 'Asymptotic (high k)', 'blue'),
        (phi_extrapolation, 'v-', 'Extrapolated φ(∞)', 'green')
    ]
    
    for phi_data, marker_style, label, color in method_configs:
        ax.plot(fidelity_values, phi_data, marker_style, label=label, 
                color=color, alpha=0.8, markersize=4)
    
    # Theoretical bound with prominent styling
    ax.plot(fidelity_values, phi_theory, '--', label='Theoretical bound', 
            color='black', linewidth=3, alpha=0.9)

    # Setup axes and limits
    ax.set_xlabel(r"Input fidelity $f$")
    ax.set_ylabel(r"Error rate decay $\phi(f)$")
    ax.set_title(r"Comprehensive Comparison: $\phi(f)$ Estimation Methods", pad=15)

    # Calculate appropriate y-limits considering all valid data
    all_phi_arrays = [phi_linear, phi_numerical, phi_nonlinear, phi_asymptotic, phi_extrapolation]
    valid_values = []
    
    for phi_array in all_phi_arrays:
        valid_mask = ~np.isnan(phi_array)
        if np.sum(valid_mask) > 0:
            valid_values.extend(phi_array[valid_mask])
    
    if valid_values:
        ymin = min(np.min(valid_values), np.min(phi_theory)) * 0.95
        ymax = max(np.max(valid_values), np.max(phi_theory)) * 1.05
    else:
        ymin, ymax = np.min(phi_theory) * 0.95, np.max(phi_theory) * 1.05
    
    ax.set_xlim([0.5, 1.0])
    ax.set_ylim([ymin, ymax])

    # Enhanced legend and styling
    ax.legend(loc="best", frameon=True, fancybox=True, shadow=True, 
             framealpha=0.9, edgecolor='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#fafafa')  # Subtle background

    plt.tight_layout()
    plt.show()
