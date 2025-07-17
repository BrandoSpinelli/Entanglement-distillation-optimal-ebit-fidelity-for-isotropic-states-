"""
Quantum Core Functions Module

This module contains the fundamental quantum mechanics and optimization functions
for fidelity optimization and constraint matrix generation.

Functions:
    - Fmax_model: Theoretical model for maximum achievable fidelity
    - theoretical_phi: Theoretical upper bound for error exponent
    - generate_p_bar: Binomial probability distribution for quantum states
    - generate_constraint_matrix_vectorized: Efficient constraint matrix generation
    - solve_linear_program: Linear program solver for fidelity optimization
"""

import numpy as np
import cvxpy
from scipy.special import comb, gammaln
from loguru import logger


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
