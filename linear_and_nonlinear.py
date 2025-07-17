import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import cvxpy
from scipy.optimize import curve_fit
from scipy.special import comb, gammaln
from scipy.stats import linregress
from loguru import logger
import warnings


def Fmax_model(k, C, phi):
    """
    Theoretical model for maximum achievable fidelity as a function of k.
    
    This model describes the asymptotic behavior of fidelity optimization protocols:
    F_max(k) = 1 - C * 2^(-phi * k)
    
    where:
    - k is the number of tensor copies
    - C is a normalization constant (ideally close to 1)
    - phi is the error exponent determining the exponential decay rate
    
    Args:
        k (int or np.ndarray): Number of tensor copies.
        C (float): Amplitude parameter for the exponential decay.
        phi (float): Error exponent (decay rate parameter).
        
    Returns:
        float or np.ndarray: Model prediction for F_max(k).
    """
    return 1 - C * 2 ** (-phi * k)


def theoretical_phi(f: float) -> float:
    """
    Compute the theoretical upper bound for the error exponent φ(f).
    
    This bound is derived from quantum information theory and represents the
    asymptotic scaling of fidelity optimization protocols.
    
    Args:
        f (float): Input fidelity value in range [0, 1].
        
    Returns:
        float: Theoretical upper bound φ_th(f) = log₂(1 / (2√(f(1-f)))).
    """
    return np.log2(1 / (2*np.sqrt(f * (1 - f))))

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


def generate_p_bar(F: float, k: int) -> np.ndarray:
    """
    Generate the binomial probability distribution for quantum state fidelity.
    
    This function computes the probability vector p̄ for k independent Bernoulli
    trials with success probability F, representing the statistical model
    underlying quantum fidelity optimization.
    
    The resulting distribution models the outcomes when measuring k tensor
    copies of a quantum state with fidelity F.
    
    Args:
        F (float): Input fidelity parameter in range [0, 1].
        k (int): Number of tensor copies (independent trials).
        
    Returns:
        np.ndarray: Probability vector of length (k + 1) where element j
                   represents P(j successes out of k trials).
    """
    return np.array([(1 - F) ** j * F ** (k - j) * comb(k, j) for j in range(k + 1)])


def generate_constraint_matrix_vectorized(k: int) -> np.ndarray:
    """
    Efficient and numerically stable generation of the constraint matrix A
    used in the linear program for fidelity optimization, for a given number
    of copies `k`.

    This implementation avoids nested loops by using a partially vectorized
    approach: it retains a single loop over the summation index `s`, while all
    operations for indices j and t are vectorized using NumPy broadcasting.

    The matrix element A[j, t] is defined as a sum over `s`:
    
        A[j, t] = sum_{s=0}^{min(j,t)} 2^(j-k) * (-2)^(s-j) * 6^(-s)
                  * binom(k - j, t - s) * binom(j, s)

    To maintain numerical stability (avoid overflow/underflow) for large values of `k`, this code works in
    the **log-domain** for the binomial coefficients and powers. Specifically:

    - `binom(n, r)` is replaced with `exp(gammaln(n+1) - gammaln(r+1) - gammaln(n-r+1))`
      using the identity `log(n!) = gammaln(n + 1)`
    - Powers like `2^(j-k)` are handled as `exp((j-k)*log(2))`, which is numerically stable
    - Similarly, `(-2)^(s-j)` is split into a sign term and a log-magnitude component

    Args:
        k (int): The number of copies in the quantum setting (defines matrix size).

    Returns:
        np.ndarray: A (k + 1) x (k + 1) constraint matrix A.
    """
    log_2 = np.log(2)
    log_6 = np.log(6)

    # Indices for j and t
    j, t = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing='ij')

    A = np.zeros((k + 1, k + 1), dtype=np.float64)

    # Loop over s but keep everything else vectorized
    for s in range(k + 1):
        valid_mask = (s <= j) & (s <= t)

        log_comb1 = (
            gammaln(k - j + 1) -
            gammaln(t - s + 1) -
            gammaln(k - j - (t - s) + 1)
        )
        log_comb2 = (
            gammaln(j + 1) -
            gammaln(s + 1) -
            gammaln(j - s + 1)
        )

        # log of the coefficient (excluding the alternating sign)
        log_coeff = (
            (j - k) * log_2 +
            (s - j) * log_2 -
            s * log_6 +
            log_comb1 +
            log_comb2
        )

        sign = (-1.0) ** (j - s)
        term = np.where(valid_mask, sign * np.exp(log_coeff), 0.0)

        A += term  # accumulate over s

    return A


def solve_linear_program(F: float, k: int, A: np.ndarray) -> float:
    """
    Solve the linear program to compute the maximum achievable fidelity F_max 
    from an initial fidelity F and a given constraint matrix A.

    Args:
        F (float): Initial fidelity.
        k (int): Number of tensor copies.
        A (np.ndarray): Constraint matrix of shape (k+1, k+1).

    Returns:
        float: Optimal value of the fidelity maximization problem. Returns NaN if LP fails.
    """
    p_bar = generate_p_bar(F, k)               # Binomial probability distribution
    b = cvxpy.Variable(k + 1)                  # Optimization variable (vector of probabilities)

    # Define linear constraints
    constraints = [
        b >= 0,                # Non-negativity
        b <= 1,                # Upper bound
        A @ b <= 0.5,          # Fidelity constraint (upper)
        A @ b >= -0.5          # Fidelity constraint (lower)
    ]

    # Define the objective: maximize expected fidelity
    objective = cvxpy.Maximize(p_bar @ b)

    # Solve the optimization problem
    problem = cvxpy.Problem(objective, constraints)
    problem.solve(solver=cvxpy.MOSEK, verbose=False)

    # Log warning if solution is not optimal
    if problem.status not in ["optimal", "optimal_inaccurate"]:
        logger.warning(f"F = {F:.3f}, LP not solved optimally (status: {problem.status})")
        return np.nan

    return problem.value

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

def plot_fidelity_vs_fmax(F_values: np.ndarray, Fmax_values: np.ndarray, k: int):
    """
    Plot F_max(f) for a given k value using IEEE-style formatting.

    Args:
        F_values (np.ndarray): Array of input fidelity values.
        Fmax_values (np.ndarray): Array of maximum fidelity values.
        k (int): Number of tensor copies.
    """
    # Configure global style for IEEE-like aesthetics
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
        
    plt.plot(fidelity_values, C_list, label='Estimated C(f)')
    plt.axhline(1.0, color='r', linestyle='--', label='C = 1')
    plt.legend()
    plt.show()

    return np.array(C_list), np.array(phi_list)


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
    processed_data = _extract_valid_log_data(fidelity_values, Fmax_grid, k_values)
    
    for i, y_vals, ks in processed_data:
        if len(ks) < 3:  # Need minimum points for reliable regression
            phi_estimates.append(np.nan)
            continue
        
        # Perform weighted linear regression
        phi_est = _perform_weighted_linear_regression(ks, y_vals)
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


def plot_phi_numerical(fidelity_values: np.ndarray, phi_numerical: np.ndarray):
    """
    Plot φ(f) estimated numerically vs. theoretical φ(f) with IEEE-style formatting.

    Args:
        fidelity_values (np.ndarray): Input fidelity values.
        phi_numerical (np.ndarray): Numerically estimated φ(f).
    """
    phi_theory = theoretical_phi(fidelity_values)

    # IEEE-style plot settings
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

    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # IEEE single-column size

    ax.plot(
        fidelity_values, phi_numerical, 'o-', color='darkorange',
        label=r'Numerical $\phi_{\mathrm{num}}(f)$'
    )
    ax.plot(
        fidelity_values, phi_theory, '--', color='black',
        label=r'Theory $\phi_{\mathrm{th}}(f) = \log \left(\frac{1}{2\sqrt{f(1-f)}}\right)$'
    )

    ax.set_xlabel(r"Input fidelity $f$")
    ax.set_ylabel(r"Error rate decay $\phi(f)$")
    ax.set_title("Numerical Estimate vs Theory", pad=4)

    # Define plot limits based on valid values only
    valid_vals = ~np.isnan(phi_numerical)
    ymin = np.nanmin([np.min(phi_numerical[valid_vals]), np.min(phi_theory)]) * 0.95
    ymax = np.nanmax([np.max(phi_numerical[valid_vals]), np.max(phi_theory)]) * 1.05
    ax.set_xlim([0.5, 1.0])
    ax.set_ylim([ymin, ymax])

    ax.legend(loc="best", frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()


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
    processed_data = _extract_valid_log_data(fidelity_values, Fmax_grid, k_values_high)
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

def _extract_valid_log_data(fidelity_values: np.ndarray, Fmax_grid: dict, 
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

def _perform_weighted_linear_regression(ks: list, y_vals: list) -> float:
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

if __name__ == "__main__":
    """
    Main execution pipeline for quantum fidelity optimization and error exponent estimation.
    
    This script implements a comprehensive analysis of quantum error exponent φ(f) using
    multiple estimation approaches. The pipeline includes:
    
    1. Linear program-based fidelity optimization
    2. Multiple φ(f) estimation methods (linear, numerical, nonlinear, asymptotic, extrapolation)
    3. IEEE-style visualization and comparison of results
    
    The analysis provides insights into the asymptotic behavior of quantum fidelity
    optimization protocols and validates theoretical bounds through numerical methods.
    """
    logger.info("=== Quantum Fidelity Optimization & Error Exponent Analysis ===")

    # Configuration parameters
    k_value = 30  # Number of tensor copies for initial fidelity scan
    fidelity_values = np.linspace(0.5, 0.95, 100)  # Range of input fidelities
    k_list = list(range(10, 101, 10))  # Extended k range for asymptotic analysis

    logger.info(f"Configuration: k_scan={k_value}, fidelity_range=[0.5, 0.95], k_grid={k_list}")

    # === STEP 1: Single-k fidelity optimization scan ===
    logger.info(f"Step 1: Performing fidelity scan for k = {k_value}")
    F_values, Fmax_values = scan_fidelity_grid(k_value, fidelity_values)
    logger.info("Fidelity scan completed, generating visualization")
    plot_fidelity_vs_fmax(F_values, Fmax_values, k=k_value)

    # === STEP 2: Multi-k grid computation for φ(f) estimation ===
    logger.info("Step 2: Computing F_max grid across multiple k values...")
    Fmax_grid = compute_Fmax_grid(fidelity_values, k_list)
    logger.info(f"Grid computation completed: {len(k_list)} k values × {len(fidelity_values)} fidelities")

    # === STEP 3: φ(f) estimation using multiple methods ===
    
    # Method 1: Nonlinear curve fitting
    logger.info("Step 3a: Nonlinear fitting (F_max = 1 - C*2^(-k*φ))...")
    C_fit, phi_fit = fit_phi_nonlinear(fidelity_values, Fmax_grid)
    plot_phi_fit_vs_theory(fidelity_values, phi_fit)

    # Method 2: Weighted linear regression
    logger.info("Step 3b: Weighted linear regression (k² weighting)...")
    phi_linear_fit = phi_fit_linear(fidelity_values, Fmax_grid)
    plot_phi_fit_linear(fidelity_values, phi_linear_fit)

    # Method 3: Direct numerical estimation
    logger.info("Step 3c: Direct numerical estimation...")
    phi_numerical = estimate_phi_numerical(fidelity_values, Fmax_grid)
    plot_phi_numerical(fidelity_values, phi_numerical)
    
    # Method 4: Asymptotic analysis (high k only)
    logger.info("Step 3d: Asymptotic estimation (k ≥ 30)...")
    phi_asymptotic = phi_fit_asymptotic(fidelity_values, Fmax_grid, k_min_threshold=30)
    plot_phi_asymptotic(fidelity_values, phi_asymptotic)
    
    # Method 5: Extrapolation to infinite k
    logger.info("Step 3e: Extrapolation analysis (φ(k) → φ(∞))...")
    phi_extrapolated = phi_fit_extrapolation(fidelity_values, Fmax_grid)
    plot_phi_extrapolation(fidelity_values, phi_extrapolated)

    # === STEP 4: Comprehensive comparison ===
    logger.info("Step 4: Generating comprehensive method comparison...")
    plot_phi_comparison_all(fidelity_values, phi_linear_fit, phi_numerical, 
                           phi_fit, phi_asymptotic, phi_extrapolated)

    logger.info("=== Analysis Pipeline Completed Successfully ===")
    logger.info("All φ(f) estimation methods executed and visualized.")




