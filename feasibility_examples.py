"""
Example script to demonstrate the use of the feasibility analysis system.
Shows different use cases and configurations.
"""

from feasibility_calculator import compute_feasibility_batch, print_results_table
from feasibility_visualizer import create_complete_report, plot_feasibility_bars, plot_feasibility_heatmap
import os

def example_basic_analysis():
    """
    Basic example: analysis for k from 2 to 6
    """
    print("="*60)
    print("EXAMPLE 1: BASIC ANALYSIS (k = 2 to 6)")
    print("="*60)
    
    k_values = [2, 3, 4, 5, 6]
    results = compute_feasibility_batch(k_values)
    print_results_table(results)
    
    # Create only the bar chart
    plot_feasibility_bars(results)

def example_extended_analysis():
    """
    Extended example: analysis for larger k
    """
    print("\\n" + "="*60)
    print("EXAMPLE 2: EXTENDED ANALYSIS (k = 2 to 12)")
    print("="*60)
    
    k_values = list(range(2, 13))
    results = compute_feasibility_batch(k_values)
    
    # Show only a summary
    feasible_k = [k for k, r in results.items() if r['is_feasible']]
    non_feasible_k = [k for k, r in results.items() if not r['is_feasible']]
    
    print(f"\\nSUMMARY RESULTS:")
    print(f"Feasible k values: {feasible_k}")
    print(f"NON-feasible k values: {non_feasible_k}")
    
    # Create complete report
    create_complete_report(results)

def example_custom_analysis():
    """
    Custom example: analysis for specific k values
    """
    print("\\n" + "="*60)
    print("EXAMPLE 3: CUSTOM ANALYSIS")
    print("="*60)
    
    # Test some specific interesting values
    k_values = [2, 5, 10, 15, 20]
    results = compute_feasibility_batch(k_values)
    
    print("\\nResults table:")
    print_results_table(results)
    
    # Create heatmap
    plot_feasibility_heatmap(results)

def example_convergence_study():
    """
    Convergence study: many k values to see behavior
    """
    print("\\n" + "="*60)
    print("EXAMPLE 4: CONVERGENCE STUDY")
    print("="*60)
    
    # K values with variable step to study convergence
    k_values = [2, 3, 4, 5, 6, 8, 10, 12, 15, 20, 25, 30]
    
    print(f"Testing convergence for k = {k_values}")
    
    results = compute_feasibility_batch(k_values)
    
    # Convergence analysis
    print("\\nCONVERGENCE ANALYSIS:")
    print("-" * 40)
    
    # Check if S_0 converges to 0.5
    S_0_values = [results[k]['S_values'][0] for k in k_values]
    print("S_0(b) values:")
    for k, s0 in zip(k_values, S_0_values):
        print(f"  k={k:2d}: S_0 = {s0:8.6f}")
    
    # Check feasibility for large k
    large_k_feasible = [k for k in k_values if k >= 10 and results[k]['is_feasible']]
    print(f"\\nK values ≥ 10 that are feasible: {large_k_feasible}")
    
    # Create complete report with convergence analysis
    create_complete_report(results)

def example_save_results():
    """
    Example of saving results
    """
    print("\\n" + "="*60)
    print("EXAMPLE 5: SAVING RESULTS")
    print("="*60)
    
    # Create directory for results if it doesn't exist
    save_dir = "feasibility_results"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        print(f"Created directory: {save_dir}")
    
    k_values = [2, 3, 4, 5, 6, 7, 8]
    results = compute_feasibility_batch(k_values)
    
    # Save all plots
    create_complete_report(results, save_dir)
    
    print(f"\\nAll plots have been saved to: {save_dir}")

def run_all_examples():
    """
    Runs all examples in sequence
    """
    print("RUNNING ALL EXAMPLES")
    print("="*60)
    
    try:
        example_basic_analysis()
        
        input("\\nPress ENTER to continue with example 2...")
        example_extended_analysis()
        
        input("\\nPress ENTER to continue with example 3...")
        example_custom_analysis()
        
        input("\\nPress ENTER to continue with example 4...")
        example_convergence_study()
        
        input("\\nPress ENTER to continue with example 5...")
        example_save_results()
        
        print("\\n" + "="*60)
        print("ALL EXAMPLES COMPLETED!")
        print("="*60)
        
    except KeyboardInterrupt:
        print("\\nExecution interrupted by user")

if __name__ == "__main__":
    print("FEASIBILITY ANALYSIS EXAMPLE SCRIPT")
    print("="*60)
    print("Available examples:")
    print("1. Basic analysis (k = 2 to 6)")
    print("2. Extended analysis (k = 2 to 12)")
    print("3. Custom analysis")
    print("4. Convergence study")
    print("5. Save results")
    print("6. Run all examples")
    
    choice = input("\\nSelect an example (1-6): ").strip()
    
    if choice == "1":
        example_basic_analysis()
    elif choice == "2":
        example_extended_analysis()
    elif choice == "3":
        example_custom_analysis()
    elif choice == "4":
        example_convergence_study()
    elif choice == "5":
        example_save_results()
    elif choice == "6":
        run_all_examples()
    else:
        print("Invalid option. Running basic example...")
        example_basic_analysis()
