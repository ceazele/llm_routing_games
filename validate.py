import os
import pandas as pd
import glob

# Specify the input folder
input_folder = 'summarized runs'  # Replace with your input folder path

# Define valid routes for both cases
valid_routes_no_bridge = ['O-L-D', 'O-R-D']
valid_routes_with_bridge = ['O-L-D', 'O-R-D', 'O-L-R-D']

# Initialize a list to collect error reports
error_reports = []

# Find all run folders in the input directory
run_folders = sorted(glob.glob(os.path.join(input_folder, 'run *')))

for run_folder in run_folders:
    # Paths to game_1A.csv and game_1B.csv
    game_1A_path = os.path.join(run_folder, 'game_1A', 'game_1A.csv')
    game_1B_path = os.path.join(run_folder, 'game_1B', 'game_1B.csv')
    
    # Validate No Bridge data
    if os.path.exists(game_1A_path):
        df_no_bridge = pd.read_csv(game_1A_path)
        
        # Check if 'Route' column exists
        if 'Route' in df_no_bridge.columns:
            # Identify invalid routes
            invalid_routes = set(df_no_bridge['Route'].unique()) - set(valid_routes_no_bridge)
            if invalid_routes:
                error_reports.append({
                    'file': game_1A_path,
                    'invalid_routes': invalid_routes
                })
        else:
            error_reports.append({
                'file': game_1A_path,
                'error': "'Route' column is missing."
            })
    else:
        error_reports.append({
            'file': game_1A_path,
            'error': "File does not exist."
        })
    
    # Validate With Bridge data
    if os.path.exists(game_1B_path):
        df_with_bridge = pd.read_csv(game_1B_path)
        
        # Check if 'Route' column exists
        if 'Route' in df_with_bridge.columns:
            # Identify invalid routes
            invalid_routes = set(df_with_bridge['Route'].unique()) - set(valid_routes_with_bridge)
            if invalid_routes:
                error_reports.append({
                    'file': game_1B_path,
                    'invalid_routes': invalid_routes
                })
        else:
            error_reports.append({
                'file': game_1B_path,
                'error': "'Route' column is missing."
            })
    else:
        error_reports.append({
            'file': game_1B_path,
            'error': "File does not exist."
        })

# Print out the error reports
if error_reports:
    print("Data validation completed. Issues found in the following files:")
    for report in error_reports:
        print(f"\nFile: {report['file']}")
        if 'invalid_routes' in report:
            print(f"Invalid routes found: {report['invalid_routes']}")
        else:
            print(f"Error: {report['error']}")
else:
    print("Data validation completed. No issues found.")
