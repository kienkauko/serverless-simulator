import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def power_law_with_offset(x, A, B, x0, C):
    """
    Power Law with offset function: y = A * (x - x0)^B + C
    
    Args:
        x: Input values
        A: Amplitude
        B: Power (exponent)
        x0: x-offset
        C: y-offset
    """
    # Handle potential domain errors when x < x0
    with np.errstate(all='ignore'):
        result = np.zeros_like(x, dtype=float)
        valid = x > x0
        # Apply power law only to valid values
        result[valid] = A * np.power(x[valid] - x0, B) + C
        # Set the rest to the offset
        result[~valid] = C
        return result

def perform_power_law_fitting(csv_file, initial_params=None):
    """
    Perform Power Law with offset fitting on theta vs ram_usage_per_request from a CSV file.
    
    Args:
        csv_file: Path to the CSV file
        initial_params: Initial parameter guesses [A, B, x0, C]
        
    Returns:
        DataFrame with original data and the model
    """
    print(f"Reading data from: {csv_file}")
    
    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Extract the features (theta) and target (ram_usage_per_request)
    X = df['theta'].values
    y = df['ram_usage_per_request'].values
    
    # Sort data for better visualization
    sorted_indices = np.argsort(X)
    X_sorted = X[sorted_indices]
    y_sorted = y[sorted_indices]
    
    # Set initial parameter guesses if not provided
    if initial_params is None:
        # Reasonable default guesses:
        # A: scale factor (try the max y value)
        # B: exponent (try -0.5, common for power laws)
        # x0: x offset (try 0)
        # C: y offset (try min y value)
        initial_params = [np.max(y), -0.5, 0, np.min(y)]
    
    try:
        # Perform curve fitting
        params, covariance = curve_fit(
            power_law_with_offset, X, y, 
            p0=initial_params,
            bounds=([0, -10, -1, 0], [np.inf, 10, 1, np.inf]),  # Set reasonable bounds
            maxfev=10000  # Increase max function evaluations
        )
        
        A, B, x0, C = params
        
        # Generate predictions for the original data points
        y_pred = power_law_with_offset(X, A, B, x0, C)
        
        # Create a smooth curve for plotting
        X_smooth = np.linspace(X.min(), X.max(), 200)
        y_smooth = power_law_with_offset(X_smooth, A, B, x0, C)
        
        # Add predictions to dataframe
        df['ram_usage_per_request_poly'] = y_pred
        
        # Calculate R-squared (coefficient of determination)
        ss_total = np.sum((y - np.mean(y)) ** 2)
        ss_residual = np.sum((y - y_pred) ** 2)
        r_squared = 1 - (ss_residual / ss_total)
        
        # Plot the data points and the fitted curve
        plt.figure(figsize=(10, 6))
        plt.scatter(X, y, color='blue', label='Data points')
        plt.plot(X_smooth, y_smooth, color='red', 
                 label=f'Power Law fit: y = {A:.4f}(x - {x0:.4f})^{B:.4f} + {C:.4f}\nR² = {r_squared:.4f}')
        
        plt.title('Theta vs RAM Usage Per Request (Power Law with Offset)')
        plt.xlabel('Theta')
        plt.ylabel('RAM Usage Per Request')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()
        
        return df, params, r_squared
        
    except Exception as e:
        print(f"Fitting error: {e}")
        print("Try adjusting the initial parameters or bounds.")
        return df, None, None

def display_power_law_formula(params, r_squared):
    """Display the power law formula"""
    if params is None:
        print("\nCould not fit the power law model to the data.")
        return
    
    A, B, x0, C = params
    
    print("\nPower Law with Offset Formula:")
    print(f"ram_usage_per_request = {A:.6f} × (theta - {x0:.6f})^{B:.6f} + {C:.6f}")
    print(f"R² (coefficient of determination): {r_squared:.6f}")
    
    print("\nTo calculate approximation for a specific theta value:")
    print(f"ram_usage_per_request = {A:.6f} × (theta - {x0:.6f})^{B:.6f} + {C:.6f}")
    print("\nExample calculation for theta = 0.1:")
    if 0.1 > x0:
        result = A * (0.1 - x0)**B + C
        print(f"ram_usage_per_request = {A:.6f} × (0.1 - {x0:.6f})^{B:.6f} + {C:.6f} = {result:.6f}")
    else:
        print(f"Cannot calculate: theta value must be greater than x0 ({x0:.6f})")
    print("")

def main():
    # Get all CSV files in the current directory
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Change working directory to script directory
    os.chdir(script_dir)
    # Find all CSV files in this directory
    csv_files = glob.glob("*.csv")
    
    if not csv_files:
        print("No CSV files found in the current directory.")
        return
    
    # Let the user select a file
    print("Available CSV files:")
    for i, file in enumerate(csv_files):
        print(f"{i+1}. {file}")
    
    try:
        file_idx = int(input("\nSelect a file number: ")) - 1
        if file_idx < 0 or file_idx >= len(csv_files):
            print("Invalid selection. Exiting.")
            return
    except ValueError:
        print("Invalid input. Exiting.")
        return
    
    # Ask if user wants to provide custom initial parameters
    custom_params = input("\nDo you want to provide custom initial parameters? (y/n): ").lower().strip()
    initial_params = None
    if custom_params == 'y' or custom_params == 'yes':
        try:
            print("\nPlease enter initial guesses for the Power Law with Offset parameters:")
            print("Formula: y = A * (x - x0)^B + C")
            A = float(input("A (amplitude): "))
            B = float(input("B (exponent): "))
            x0 = float(input("x0 (x offset): "))
            C = float(input("C (y offset): "))
            initial_params = [A, B, x0, C]
            print(f"Using parameters: A={A}, B={B}, x0={x0}, C={C}")
        except ValueError:
            print("Invalid input. Using default parameter guesses.")
            initial_params = None
    
    # Perform the power law fitting
    df, params, r_squared = perform_power_law_fitting(csv_files[file_idx], initial_params)
    
    # Ask if user wants to see the formula
    if params is not None:
        show_formula = input("\nDo you want to see the formula details? (y/n): ").lower().strip()
        if show_formula == 'y' or show_formula == 'yes':
            display_power_law_formula(params, r_squared)
    
    # Ask if user wants to save the updated CSV
    save_csv = input("Do you want to save the CSV with the power law approximation? (y/n): ").lower().strip()
    if save_csv == 'y' or save_csv == 'yes':
        # Save to the original file (overwrite)
        df.to_csv(csv_files[file_idx], index=False)
        print(f"Updated original file: {csv_files[file_idx]}")

if __name__ == "__main__":
    main()