import cvxpy
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import gammaln
from scipy.special import comb
from loguru import logger

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


def scan_fidelity_grid(k: int, num_points: int) -> tuple:
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
    F_values = np.linspace(0.5, 1.0, num_points)
    Fmax_values = []

    A = generate_constraint_matrix_vectorized(k)
    logger.info(f"Constraint matrix generated for k = {k}")

    for F in F_values:
        F_max = solve_linear_program(F, k, A)
        Fmax_values.append(F_max)
        logger.debug(f"F = {F:.3f} -> F_max = {F_max:.6f}")

    return F_values, np.array(Fmax_values)


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

def compute_error_exponent_grid(f_values: np.ndarray, k_list: list[int]) -> dict:
    """
    Estimate the error exponent function φ(f) ≈ -1/k * log2(1 - Fmax)
    for various fidelity values and increasing k.

    Args:
        f_values (np.ndarray): Array of input fidelities f ∈ [0.5, 1].
        k_list (list[int]): List of k values for which to estimate Fmax and φ(f).

    Returns:
        dict: Mapping of each k to a tuple (f_values, phi_values).
    """
    exponent_results = {}

    for k in k_list:
        logger.info(f"Computing φ(f) for k = {k}")
        A = generate_constraint_matrix_vectorized(k)
        phi_values = []

        for f in f_values:
            Fmax = solve_linear_program(f, k, A)
            if np.isnan(Fmax) or Fmax >= 1.0:
                phi_values.append(np.nan)
            else:
                phi = -1.0 / k * np.log2(1 - Fmax)
                phi_values.append(phi)

        exponent_results[k] = (f_values, np.array(phi_values))

    return exponent_results


def plot_error_exponent_curves(exponent_results: dict):
    """
    Plot the estimated error exponent φ(f) vs input fidelity f,
    for different values of k.

    Args:
        exponent_results (dict): Dictionary mapping k to (f_values, phi_values).
    """
    plt.figure(figsize=(4, 2.8))
    for k, (f_values, phi_values) in exponent_results.items():
        plt.plot(f_values, phi_values, label=fr"$k={k}$")

    plt.xlabel(r"Input Fidelity $f$")
    plt.ylabel(r"Error Exponent $\phi(f)$")
    plt.title("Estimated Error Exponent Curves")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    logger.info("Starting fidelity scan and error exponent estimation")

    # You can change the value of k to control the number of tensor copies.
    k_value = 100
    logger.info(f"Starting fidelity scan for k = {k_value}")

    # Perform fidelity scan and plot results
    F_values, Fmax_values = scan_fidelity_grid(k=k_value, num_points=100)
    logger.info("Fidelity scan completed, plotting results")
    plot_fidelity_vs_fmax(F_values, Fmax_values, k=k_value)

    # Initial fidelity grid
    f_values = np.linspace(0.5, 0.99, 100)

    # Choose values of k for approximation
    k_values = np.linspace(10, 300, 10, dtype=int)

    # Compute error exponent φ(f) for each k
    logger.info("Computing error exponent φ(f) for various k values")
    results = compute_error_exponent_grid(f_values, k_values)

    # Plot the results
    logger.info("Plotting error exponent curves")
    plot_error_exponent_curves(results)

