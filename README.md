# Quantum Fidelity Optimization Analysis

A comprehensive modular Python toolkit for analyzing quantum error exponents φ(f) through multiple estimation methods.

## 🏗️ **Modular Architecture**

The project is organized into specialized modules for maintainability and reusability:

```
📦 Quantum Analysis Toolkit
├── 🧮 quantum_core.py          # Core quantum mechanics & optimization
├── 📊 data_processing.py       # Data extraction & statistical processing  
├── 🔬 phi_fitting.py           # Multiple φ(f) estimation methods
├── 📈 visualization.py         # IEEE-style plotting & comparisons
├── 🚀 main_quantum_analysis.py # Main execution pipeline
└── 📋 requirements.txt         # Dependencies
```

## 🧮 **quantum_core.py**
- `Fmax_model()`: Theoretical F_max(k) = 1 - C*2^(-phi*k)
- `theoretical_phi()`: Theoretical upper bound φ_th(f)
- `generate_p_bar()`: Binomial probability distributions
- `generate_constraint_matrix_vectorized()`: Efficient constraint matrix generation
- `solve_linear_program()`: CVXPY-based fidelity optimization

## 📊 **data_processing.py** 
- `scan_fidelity_grid()`: F_max evaluation over fidelity grids
- `compute_Fmax_grid()`: Multi-k F_max grid computation
- `extract_valid_log_data()`: Standardized data extraction & validation
- `perform_weighted_linear_regression()`: k²-weighted regression

## 🔬 **phi_fitting.py**
- `fit_phi_nonlinear()`: Nonlinear curve fitting F_max = 1 - C*2^(-k*φ)
- `phi_fit_linear()`: Weighted linear regression log₂(1-F_max) vs k
- `estimate_phi_numerical()`: Direct numerical estimation φ = -(1/k)*log₂(1-F_max)
- `phi_fit_asymptotic()`: High-k asymptotic analysis (k ≥ threshold)
- `phi_fit_extrapolation()`: Extrapolation φ(k) → φ(∞)

## 📈 **visualization.py**
- IEEE-style plotting templates with consistent formatting
- Individual plotting functions for each estimation method
- `plot_phi_comparison_all()`: Comprehensive method comparison
- `plot_fidelity_vs_fmax()`: F_max visualization

## 🚀 **main_quantum_analysis.py**
- Complete analysis pipeline orchestration
- Environment validation
- Structured logging and progress tracking
- Method comparison and summary

## ⚙️ **Installation & Usage**

```bash
# Install dependencies
pip install -r requirements.txt

# Run complete analysis
python main_quantum_analysis.py
```

## 📊 **Analysis Methods**

1. **Nonlinear Fitting**: F_max(k,f) = 1 - C(f)*2^(-k*φ(f))
2. **Weighted Linear Regression**: log₂(1-F_max) vs k with k² weighting  
3. **Direct Numerical**: φ(f) ≈ -(1/k)*log₂(1-F_max), averaged over k
4. **Asymptotic Analysis**: Using only high k values (k ≥ 30)
5. **Extrapolation**: φ(k) = φ_∞ + A/k → extract φ_∞

## 🎯 **Key Features**

- **Modular Design**: Clean separation of concerns
- **IEEE-Style Plots**: Publication-ready visualizations
- **Multiple Methods**: Comprehensive φ(f) estimation approaches
- **Asymptotic Focus**: Special emphasis on k → ∞ behavior
- **Error Handling**: Robust numerical stability
- **Documentation**: Extensive docstrings and comments

## 🔬 **Scientific Background**

The analysis focuses on quantum error exponents φ(f) that characterize the asymptotic scaling of fidelity optimization protocols:

**F_max(k,f) ≈ 1 - C(f) * 2^(-k*φ(f))**

Where φ(f) represents the rate at which optimization errors decay exponentially with the number of tensor copies k.

## 📈 **Expected Results**

- Comparison of estimation methods against theoretical bounds
- Validation of asymptotic behavior k → ∞  
- Method performance evaluation across different fidelity regimes
- Publication-ready IEEE-style visualizations
