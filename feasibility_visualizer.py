"""
Module for visualizing feasibility results.
Contains functions to create different types of IEEE-style plots.
"""

import numpy as np
import matplotlib.pyplot as plt
from fractions import Fraction
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap

# IEEE style configuration
try:
    # Prova prima con seaborn se disponibile
    plt.style.use('seaborn-v0_8-whitegrid')
except OSError:
    try:
        # Fallback a seaborn classico
        plt.style.use('seaborn-whitegrid')
    except OSError:
        # Usa stile matplotlib standard con griglia
        plt.style.use('default')

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif', 'serif'],
    'axes.titlesize': 14,
    'axes.labelsize': 12,
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'figure.titlesize': 16,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'axes.axisbelow': True,
    'axes.edgecolor': 'black',
    'axes.linewidth': 0.8,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white'
})

def plot_feasibility_bars(results, save_path=None):
    """
    Creates a bar chart of S_j(b) values for all k.
    
    Args:
        results (dict): Feasibility computation results
        save_path (str): Path to save the plot (optional)
    """
    k_values = sorted(results.keys())
    
    # Data preparation
    max_j = max(len(results[k]['S_values']) for k in k_values)
    
    # IEEE style colors
    colors = plt.cm.Set1(np.linspace(0, 1, len(k_values)))
    
    # Adaptive figure configuration - single large plot
    fig_height = max(10, len(k_values) * 0.8 + 6)  # Adaptive height
    fig, ax = plt.subplots(1, 1, figsize=(18, fig_height))
    
    # ========== MAIN PLOT: S_j(b) Values ==========
    
    x_positions = np.arange(max_j)
    width = 0.8 / len(k_values)
    
    for i, k in enumerate(k_values):
        S_values = results[k]['S_values']
        
        # Extend with NaN to standardize length
        extended_values = S_values + [np.nan] * (max_j - len(S_values))
        
        # Bar positions
        positions = x_positions + (i - len(k_values)/2 + 0.5) * width
        
        # Create bars
        bars = ax.bar(positions, extended_values, width,
                      label=f'k={k}', color=colors[i], alpha=0.8,
                      edgecolor='black', linewidth=0.5)
        
        # Add labels with fractions
        for j, (pos, val) in enumerate(zip(positions, extended_values)):
            if not np.isnan(val) and abs(val) > 1e-10:
                frac = Fraction(val).limit_denominator(10000)
                
                if frac.denominator == 1:
                    text = str(frac.numerator)
                else:
                    text = f"{frac.numerator}/{frac.denominator}"
                
                # Position text
                y_offset = 0.02 if val >= 0 else -0.04
                ax.text(pos, val + y_offset, text, ha='center', 
                        va='bottom' if val >= 0 else 'top',
                        fontsize=10, fontweight='bold', rotation=45)
    
    # Reference lines
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.7, 
                linewidth=2, label='Upper limit')
    ax.axhline(y=-0.5, color='red', linestyle='--', alpha=0.7, 
                linewidth=2, label='Lower limit')
    ax.axhline(y=0, color='gray', linestyle='-', alpha=0.5, linewidth=1)
    
    ax.set_xlabel('Index j', fontsize=14, fontweight='bold')
    ax.set_ylabel('$S_j(b)$', fontsize=14, fontweight='bold')
    ax.set_title('Feasibility Values $S_j(b)$ for the Optimal Vector', 
                  fontsize=16, fontweight='bold')
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f'j={i}' for i in range(max_j)])
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=12)
    ax.set_ylim(-0.6, 0.6)
    
    # Improve spacing and layout
    plt.tight_layout(pad=2.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to: {save_path}")
    
    plt.show()

def plot_feasibility_heatmap(results, save_path=None):
    """
    Creates a compact heatmap of S_j(b) values.
    
    Args:
        results (dict): Feasibility computation results
        save_path (str): Path to save the plot (optional)
    """
    k_values = sorted(results.keys())
    max_j = max(len(results[k]['S_values']) for k in k_values)
    
    # Prepare data matrix
    data_matrix = []
    for k in k_values:
        S_values = results[k]['S_values']
        padded = S_values + [np.nan] * (max_j - len(S_values))
        data_matrix.append(padded)
    
    data_matrix = np.array(data_matrix)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(max_j * 1.2 + 2, len(k_values) * 0.8 + 2))
    
    # Heatmap with divergent colormap
    im = ax.imshow(data_matrix, cmap='RdBu_r', aspect='auto',
                   vmin=-0.5, vmax=0.5, interpolation='nearest')
    
    # Add values as text
    for i in range(len(k_values)):
        for j in range(max_j):
            if not np.isnan(data_matrix[i, j]):
                val = data_matrix[i, j]
                
                # Convert to fraction
                if abs(val) < 1e-15:
                    text = "0"
                else:
                    frac = Fraction(val).limit_denominator(10000)
                    if frac.denominator == 1:
                        text = str(frac.numerator)
                    else:
                        text = f"{frac.numerator}/{frac.denominator}"
                
                # Text color
                color = 'white' if abs(val) > 0.25 else 'black'
                ax.text(j, i, text, ha='center', va='center',
                       color=color, fontsize=10, fontweight='bold')
    
    # Axis configuration
    ax.set_xticks(range(max_j))
    ax.set_xticklabels([f'j={i}' for i in range(max_j)])
    ax.set_yticks(range(len(k_values)))
    ax.set_yticklabels([f'k={k}' for k in k_values])
    
    ax.set_xlabel('Index j', fontsize=12, fontweight='bold')
    ax.set_ylabel('Value of k', fontsize=12, fontweight='bold')
    ax.set_title('Heatmap of Feasibility Values $S_j(b)$',
                 fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('$S_j(b)$', fontsize=12, fontweight='bold')
    
    # Grid
    ax.set_xticks(np.arange(-0.5, max_j, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(k_values), 1), minor=True)
    ax.grid(which='minor', color='gray', linestyle='-', linewidth=1, alpha=0.7)
    
    plt.tight_layout(pad=2.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Heatmap saved to: {save_path}")
    
    plt.show()

def plot_feasibility_validation(results, save_path=None):
    """
    Creates a feasibility validation plot.
    
    Args:
        results (dict): Feasibility computation results
        save_path (str): Path to save the plot (optional)
    """
    k_values = sorted(results.keys())
    
    # Calculate violations
    max_violations = []
    min_violations = []
    feasible_count = 0
    
    for k in k_values:
        S_values = results[k]['S_values']
        max_val = max(S_values)
        min_val = min(S_values)
        
        max_viol = max(0, max_val - 0.5)
        min_viol = max(0, -0.5 - min_val)
        
        max_violations.append(max_viol)
        min_violations.append(min_viol)
        
        if results[k]['is_feasible']:
            feasible_count += 1
    
    # Create plot
    fig, ax = plt.subplots(figsize=(12, 8))
    
    width = 0.35
    x_pos = np.arange(len(k_values))
    
    bars1 = ax.bar(x_pos - width/2, max_violations, width,
                   label='Upper limit violation (>0.5)', 
                   color='red', alpha=0.7)
    bars2 = ax.bar(x_pos + width/2, min_violations, width,
                   label='Lower limit violation (<-0.5)',
                   color='blue', alpha=0.7)
    
    # Perfect feasibility line
    ax.axhline(y=0, color='green', linestyle='-', linewidth=3,
               label='Perfect feasibility (no violations)')
    
    # Configuration
    ax.set_xlabel('Value of k', fontsize=12, fontweight='bold')
    ax.set_ylabel('Violation magnitude', fontsize=12, fontweight='bold')
    ax.set_title('Feasibility Condition Validation',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.legend()
    
    # Summary box
    feasibility_rate = (feasible_count / len(k_values)) * 100
    summary_text = (f'FEASIBILITY RESULTS\\n'
                   f'Values tested: {len(k_values)}\\n'
                   f'Feasible: {feasible_count}\\n'
                   f'Success rate: {feasibility_rate:.1f}%')
    
    box_color = 'lightgreen' if feasible_count == len(k_values) else 'lightyellow'
    ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
            fontsize=11, fontweight='bold',
            bbox=dict(boxstyle="round,pad=0.5", facecolor=box_color, alpha=0.8),
            verticalalignment='top')
    
    plt.tight_layout(pad=2.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Validation plot saved to: {save_path}")
    
    plt.show()

def plot_convergence_analysis(results, save_path=None):
    """
    Analyzes the convergence of S_j(b) values as k increases.
    
    Args:
        results (dict): Feasibility computation results
        save_path (str): Path to save the plot (optional)
    """
    k_values = sorted(results.keys())
    
    if len(k_values) < 3:
        print("At least 3 k values are needed for convergence analysis")
        return
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(18, 14))
    
    # ========== S_0 convergence ==========
    S_0_values = [results[k]['S_values'][0] for k in k_values]
    ax1.plot(k_values, S_0_values, 'o-', linewidth=2, markersize=8, color='blue')
    ax1.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
    ax1.axhline(y=-0.5, color='red', linestyle='--', alpha=0.7)
    ax1.set_xlabel('k')
    ax1.set_ylabel('$S_0(b)$')
    ax1.set_title('Convergence of $S_0(b)$')
    ax1.grid(True, alpha=0.3)
    
    # ========== S_1 convergence ==========
    S_1_values = [results[k]['S_values'][1] if len(results[k]['S_values']) > 1 
                  else np.nan for k in k_values]
    valid_k = [k for k, val in zip(k_values, S_1_values) if not np.isnan(val)]
    valid_S1 = [val for val in S_1_values if not np.isnan(val)]
    
    if valid_S1:
        ax2.plot(valid_k, valid_S1, 'o-', linewidth=2, markersize=8, color='orange')
        ax2.axhline(y=0.5, color='red', linestyle='--', alpha=0.7)
        ax2.axhline(y=-0.5, color='red', linestyle='--', alpha=0.7)
    ax2.set_xlabel('k')
    ax2.set_ylabel('$S_1(b)$')
    ax2.set_title('Convergence of $S_1(b)$')
    ax2.grid(True, alpha=0.3)
    
    # ========== Range of S_j values ==========
    ranges = []
    for k in k_values:
        S_vals = results[k]['S_values']
        range_val = max(S_vals) - min(S_vals)
        ranges.append(range_val)
    
    ax3.plot(k_values, ranges, 'o-', linewidth=2, markersize=8, color='green')
    ax3.set_xlabel('k')
    ax3.set_ylabel('Range($S_j(b)$)')
    ax3.set_title('Range of $S_j(b)$ Values')
    ax3.grid(True, alpha=0.3)
    
    # ========== Number of non-zero terms ==========
    non_zero_counts = []
    for k in k_values:
        S_vals = results[k]['S_values']
        non_zero = sum(1 for val in S_vals if abs(val) > 1e-10)
        non_zero_counts.append(non_zero)
    
    ax4.plot(k_values, non_zero_counts, 'o-', linewidth=2, markersize=8, color='purple')
    ax4.set_xlabel('k')
    ax4.set_ylabel('Number of non-zero terms')
    ax4.set_title('Sparsity of Vector $S_j(b)$')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout(pad=3.0, h_pad=3.0, w_pad=2.0)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Convergence analysis saved to: {save_path}")
    
    plt.show()

def create_complete_report(results, save_dir=None):
    """
    Creates a complete report with all plots.
    
    Args:
        results (dict): Feasibility computation results
        save_dir (str): Directory to save plots (optional)
    """
    print("Creating complete feasibility results report...")
    
    base_path = save_dir + "/" if save_dir else ""
    
    print("1. Creating bar chart...")
    plot_feasibility_bars(results, f"{base_path}feasibility_bars.png" if save_dir else None)
    
    print("2. Creating heatmap...")
    plot_feasibility_heatmap(results, f"{base_path}feasibility_heatmap.png" if save_dir else None)
    
    print("3. Creating validation plot...")
    plot_feasibility_validation(results, f"{base_path}feasibility_validation.png" if save_dir else None)
    
    if len(results) >= 3:
        print("4. Creating convergence analysis...")
        plot_convergence_analysis(results, f"{base_path}convergence_analysis.png" if save_dir else None)
    
    print("Complete report created!")

if __name__ == "__main__":
    # Test module with example data
    print("Testing visualization module...")
    
    # Example data
    example_results = {
        2: {'S_values': [0.5, -1/3, 5/24], 'is_feasible': True},
        3: {'S_values': [0.5, -1/3, 1/6, 0], 'is_feasible': True},
        4: {'S_values': [0.5, -17/48, 7/36, -13/192, 0], 'is_feasible': True}
    }
    
    plot_feasibility_bars(example_results)
