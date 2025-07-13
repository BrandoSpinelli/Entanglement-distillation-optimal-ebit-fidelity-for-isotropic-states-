import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm 
import cvxpy
from scipy.optimize import curve_fit
from scipy.special import comb, gammaln
from loguru import logger


def Fmax_model(k, C, phi):
    """Nonlinear model: 1 - C * 2^{-phi * k}"""
    return 1 - C * 2 ** (-phi * k)


def theoretical_phi(f: float) -> float:
    """Theoretical lower bound for φ(f)."""
    return np.log2(1 / (2 * np.sqrt(f * (1 - f))))


def generate_p_bar(F: float, k: int) -> np.ndarray:
    """
    Generate the probability distribution p̄ for a given fidelity F and number of copies k.
    This vector represents the binomial distribution for k Bernoulli trials with success probability F.

    Args:
        F (float): Fidelity in the range [0, 1].
        k (int): Number of tensor copies of the quantum state.

    Returns:
        np.ndarray: Probability vector of length (k + 1).
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

    To maintain numerical stability for large values of `k`, this code works in
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
        num_points (int): Number of fidelity values to evaluate in the range [0.5, 1.0].

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
    Plot the initial fidelity F versus the optimized fidelity F_max, styled for academic (IEEE) publication.

    Args:
        F_values (np.ndarray): Initial fidelity values.
        Fmax_values (np.ndarray): Corresponding maximum fidelities after optimization.
        k (int): Number of tensor copies of the state.
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
        mask = (~np.isnan(y)) & (y < 1.0)
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

    return np.array(C_list), np.array(phi_list)


def plot_phi_fit_vs_theory(fidelity_values: np.ndarray, phi_fit: np.ndarray):
    """
    Plot the estimated decay parameter φ_fit(f) against the theoretical φ_th(f),
    using IEEE-style aesthetics for academic publication.

    Args:
        fidelity_values (np.ndarray): Array of initial fidelity values f.
        phi_fit (np.ndarray): Estimated values φ_fit(f) from nonlinear fitting.
    """
    # Compute theoretical values
    phi_theory = theoretical_phi(fidelity_values)

    # IEEE-like plot settings
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

    fig, ax = plt.subplots(figsize=(3.5, 2.5))  # Standard IEEE column width

    # Plot numerical fit and theoretical curve
    ax.plot(
        fidelity_values, phi_fit, 'o-',
        label=r'Fitted $\phi_{\mathrm{fit}}(f)$', color='black'
    )
    ax.plot(
        fidelity_values, phi_theory, '--',
        label=r'Theoretical $\phi_{\mathrm{th}}(f)$', color='red'
    )

    ax.set_xlabel(r"Fidelity $F$")
    ax.set_ylabel(r"Decay Parameter $\phi(f)$")
    ax.set_title(r"Comparison: Fit vs Theory", pad=4)

    ax.set_xlim([0.5, 1.0])
    ax.set_ylim([
        np.nanmin(np.concatenate([phi_fit, phi_theory])) * 0.95,
        np.nanmax(np.concatenate([phi_fit, phi_theory])) * 1.05
    ])

    ax.legend(loc="best", frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.tight_layout()
    plt.show()



if __name__ == "__main__":
    logger.info("Starting fidelity scan and error exponent estimation")

    # Number of tensor copies for fidelity scan
    k_value = 100
    logger.info(f"Starting fidelity scan for k = {k_value}")

    # Range of fidelities and k values to build 2D Fmax grid
    fidelity_values = np.linspace(0.5, 0.95, 100)
    k_list = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]

    # Compute and visualize F vs Fmax
    F_values, Fmax_values = scan_fidelity_grid(k_value, fidelity_values)
    logger.info("Fidelity scan completed, plotting results")
    plot_fidelity_vs_fmax(F_values, Fmax_values, k=k_value)



    logger.info("Computing Fmax grid...")
    Fmax_grid = compute_Fmax_grid(fidelity_values, k_list)

    logger.info("Performing nonlinear fitting...")
    C_fit, phi_fit = fit_phi_nonlinear(fidelity_values, Fmax_grid)

    logger.info("Plotting φ(f) vs theoretical bound...")
    plot_phi_fit_vs_theory(fidelity_values, phi_fit)