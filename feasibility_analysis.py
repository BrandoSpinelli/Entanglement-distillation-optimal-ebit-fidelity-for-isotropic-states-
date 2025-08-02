"""

Main script to run complete feasibility analysis.
Computes S_j(b) for different k values and creates complete visualizations.
"""

import sys
import os
from feasibility_calculator import compute_feasibility_batch, print_results_table
from feasibility_visualizer import create_complete_report
import argparse

def main():
    """
    Main function to run complete analysis.
    """
    parser = argparse.ArgumentParser(description='Vector b Feasibility Analysis')
    parser.add_argument('--k_min', type=int, default=2, help='Minimum k value (default: 2)')
    parser.add_argument('--k_max', type=int, default=10, help='Maximum k value (default: 10)')
    parser.add_argument('--k_list', nargs='+', type=int, help='Specific list of k values to test')
    parser.add_argument('--save_dir', type=str, help='Directory to save plots')
    parser.add_argument('--no_plots', action='store_true', help='Do not show plots, only calculations')
    parser.add_argument('--table_only', action='store_true', help='Show only results table')
    
    args = parser.parse_args()
    
    # Determine k values to test
    if args.k_list:
        k_values = sorted(args.k_list)
        print(f"Testing specified k values: {k_values}")
    else:
        k_values = list(range(args.k_min, args.k_max + 1))
        print(f"Testing k values from {args.k_min} to {args.k_max}")
    
    # Verify that save directory exists
    if args.save_dir and not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        print(f"Created directory: {args.save_dir}")
    
    print("\\nStarting feasibility value computation...")
    print("="*60)
    
    # Compute results
    results = compute_feasibility_batch(k_values)
    
    print("\\nComputation completed!")
    
    # Print summary table
    print_results_table(results)
    
    # Summary analysis
    print("\\nSUMMARY ANALYSIS:")
    print("="*60)
    
    feasible_count = sum(1 for r in results.values() if r['is_feasible'])
    total_count = len(results)
    
    print(f"K values tested: {total_count}")
    print(f"Feasible values: {feasible_count}")
    print(f"Success rate: {(feasible_count/total_count)*100:.1f}%")
    
    if feasible_count < total_count:
        print("\\nNON-feasible values:")
        for k, result in results.items():
            if not result['is_feasible']:
                max_violation = max(abs(max(result['S_values']) - 0.5),
                                  abs(min(result['S_values']) + 0.5))
                print(f"  k = {k}: maximum violation = {max_violation:.6f}")
    
    # Statistics on S_j values
    print("\\nS_j VALUES STATISTICS:")
    print("-" * 30)
    
    all_S_values = []
    for result in results.values():
        all_S_values.extend(result['S_values'])
    
    if all_S_values:
        import numpy as np
        print(f"Total number of S_j values: {len(all_S_values)}")
        print(f"Minimum value: {min(all_S_values):.6f}")
        print(f"Maximum value: {max(all_S_values):.6f}")
        print(f"Mean: {np.mean(all_S_values):.6f}")
        print(f"Standard deviation: {np.std(all_S_values):.6f}")
    
    # Create visualizations if requested
    if not args.no_plots and not args.table_only:
        print("\\nCreating visualizations...")
        create_complete_report(results, args.save_dir)
    elif args.table_only:
        print("\\nVisualizations skipped (table-only mode)")
    else:
        print("\\nVisualizations skipped (--no_plots flag)")
    
    print("\\nAnalysis completed!")

def interactive_mode():
    """
    Interactive mode for feasibility analysis.
    """
    print("="*60)
    print("FEASIBILITY ANALYSIS - INTERACTIVE MODE")
    print("="*60)
    
    while True:
        print("\\nAvailable options:")
        print("1. Test a single k value")
        print("2. Test a range of k values")
        print("3. Test a custom list of k values")
        print("4. Load results from previous file")
        print("5. Exit")
        
        choice = input("\\nSelect an option (1-5): ").strip()
        
        if choice == "1":
            try:
                k = int(input("Enter k value: "))
                results = compute_feasibility_batch([k])
                print_results_table(results)
                
                create_plots = input("Create plots? (y/n): ").strip().lower()
                if create_plots in ['y', 'yes']:
                    create_complete_report(results)
                    
            except ValueError:
                print("Error: enter a valid integer")
        
        elif choice == "2":
            try:
                k_min = int(input("Minimum k value: "))
                k_max = int(input("Maximum k value: "))
                
                if k_min > k_max:
                    print("Error: k_min must be <= k_max")
                    continue
                
                k_values = list(range(k_min, k_max + 1))
                results = compute_feasibility_batch(k_values)
                print_results_table(results)
                
                create_plots = input("Create plots? (y/n): ").strip().lower()
                if create_plots in ['y', 'yes']:
                    create_complete_report(results)
                    
            except ValueError:
                print("Error: enter valid integers")
        
        elif choice == "3":
            try:
                k_input = input("Enter k values separated by spaces: ")
                k_values = [int(x) for x in k_input.split()]
                
                if not k_values:
                    print("Error: no values entered")
                    continue
                
                results = compute_feasibility_batch(sorted(k_values))
                print_results_table(results)
                
                create_plots = input("Create plots? (y/n): ").strip().lower()
                if create_plots in ['y', 'yes']:
                    create_complete_report(results)
                    
            except ValueError:
                print("Error: enter valid integers separated by spaces")
        
        elif choice == "4":
            print("Feature not yet implemented")
        
        elif choice == "5":
            print("Exiting program...")
            break
        
        else:
            print("Invalid option, try again")

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No command line arguments, start interactive mode
        interactive_mode()
    else:
        # Use command line arguments
        main()
