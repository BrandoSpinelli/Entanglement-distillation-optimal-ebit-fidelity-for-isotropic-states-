"""
Module for computing the feasibility of vector b.
Contains functions to calculate S_j(b) for any value of k.
"""

import numpy as np
from scipy.special import comb
from fractions import Fraction
import math

def compute_vector_b(k):
    """
    Computes the optimal vector b for a given k.
    
    Args:
        k (int): Value of k
    
    Returns:
        list: Vector b
    """
    b = []
    
    if k % 2 == 0:  # k even
        for t in range(k + 1):
            if t < k // 2:
                b.append(1)
            elif t == k // 2:
                b.append(0.5)
            else:
                b.append(0)
    else:  # k odd
        for t in range(k + 1):
            if t < (k + 1) // 2:
                b.append(1)
            else:
                b.append(0)
    
    return b

def compute_coefficient(j, t, k):
    """
    Computes the coefficient for S_j(b).
    
    Args:
        j (int): Index j
        t (int): Index t
        k (int): Value of k
    
    Returns:
        float: Computed coefficient
    """
    coefficient = 0
    
    for s in range(min(j, t) + 1):
        term = (2**(j - k) * 
                (-2)**(s - j) * 
                (6**(-s)) * 
                comb(k - j, t - s, exact=True) * 
                comb(j, s, exact=True))
        coefficient += term
    
    return coefficient

def compute_S_j(j, k):
    """
    Computes S_j(b) for given j and k.
    
    Args:
        j (int): Index j
        k (int): Value of k
    
    Returns:
        float: Value of S_j(b)
    """
    b = compute_vector_b(k)
    S_j = 0
    
    for t in range(k + 1):
        coefficient = compute_coefficient(j, t, k)
        S_j += b[t] * coefficient
    
    return S_j

def compute_all_S_values(k):
    """
    Computes all S_j(b) values for j = 0, 1, ..., k.
    
    Args:
        k (int): Value of k
    
    Returns:
        list: List of S_j(b) values
    """
    S_values = []
    
    for j in range(k + 1):
        S_j = compute_S_j(j, k)
        S_values.append(S_j)
    
    return S_values

def check_feasibility(k):
    """
    Checks if vector b satisfies the feasibility condition for a given k.
    
    Args:
        k (int): Value of k
    
    Returns:
        tuple: (is_feasible, S_values, violations)
    """
    S_values = compute_all_S_values(k)
    violations = []
    is_feasible = True
    
    for j, S_j in enumerate(S_values):
        if S_j < -0.5 or S_j > 0.5:
            is_feasible = False
            violation = max(S_j - 0.5, -0.5 - S_j)
            violations.append((j, S_j, violation))
        else:
            violations.append((j, S_j, 0))
    
    return is_feasible, S_values, violations

def compute_feasibility_batch(k_range):
    """
    Computes feasibility for a range of k values.
    
    Args:
        k_range (list): List of k values to test
    
    Returns:
        dict: Dictionary with results for each k
    """
    results = {}
    
    for k in k_range:
        print(f"Computing feasibility for k = {k}...")
        is_feasible, S_values, violations = check_feasibility(k)
        
        results[k] = {
            'is_feasible': is_feasible,
            'S_values': S_values,
            'violations': violations,
            'vector_b': compute_vector_b(k)
        }
        
        if is_feasible:
            print(f"  ✓ k = {k}: FEASIBLE")
        else:
            max_violation = max(viol[2] for viol in violations)
            print(f"  ✗ k = {k}: NOT FEASIBLE (max violation: {max_violation:.6f})")
    
    return results

def convert_to_fractions(S_values, max_denominator=10000):
    """
    Converts S_j values to fractions for cleaner visualization.
    
    Args:
        S_values (list): List of S_j values
        max_denominator (int): Maximum denominator for fractions
    
    Returns:
        list: List of Fraction objects
    """
    fractions = []
    
    for val in S_values:
        if abs(val) < 1e-15:  # Consider zero
            fractions.append(Fraction(0))
        else:
            frac = Fraction(val).limit_denominator(max_denominator)
            fractions.append(frac)
    
    return fractions

def print_results_table(results):
    """
    Prints a formatted table with the results.
    
    Args:
        results (dict): Feasibility computation results
    """
    print("\n" + "="*80)
    print("FEASIBILITY RESULTS SUMMARY TABLE")
    print("="*80)
    
    # Header
    k_values = sorted(results.keys())
    max_j = max(len(results[k]['S_values']) for k in k_values)
    
    print(f"{'j':<3}", end="")
    for k in k_values:
        print(f"{'k=' + str(k):<15}", end="")
    print()
    print("-" * (3 + 15 * len(k_values)))
    
    # Righe della tabella
    for j in range(max_j):
        print(f"{j:<3}", end="")
        for k in k_values:
            if j < len(results[k]['S_values']):
                val = results[k]['S_values'][j]
                frac = Fraction(val).limit_denominator(10000)
                
                if frac.denominator == 1:
                    text = str(frac.numerator)
                elif abs(val) < 1e-10:
                    text = "0"
                else:
                    text = f"{frac.numerator}/{frac.denominator}"
                
                print(f"{text:<15}", end="")
            else:
                print(f"{'-':<15}", end="")
        print()
    
    print("-" * (3 + 15 * len(k_values)))
    
    # Feasibility summary
    print("\nFEASIBILITY SUMMARY:")
    for k in k_values:
        status = "✓ FEASIBLE" if results[k]['is_feasible'] else "✗ NOT FEASIBLE"
        print(f"k = {k:2d}: {status}")

if __name__ == "__main__":
    # Module test
    print("Testing feasibility calculation module...")
    
    # Test for small values
    test_k_values = [2, 3, 4, 5, 6]
    results = compute_feasibility_batch(test_k_values)
    print_results_table(results)
