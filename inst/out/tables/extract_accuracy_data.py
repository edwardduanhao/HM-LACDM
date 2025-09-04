import pandas as pd
import glob
import os
import re
from pathlib import Path

# Find all summary_table.csv files
csv_files = glob.glob("/Users/eduardo/Library/CloudStorage/OneDrive-UCLAITServices/UCLA/Jeon GSR/Code/Good-to-use/HMLCDM/inst/out/tables/multiplerun/*/summary_table.csv")

# Initialize data storage
results = {'strong': [], 'moderate': [], 'weak': []}

for csv_file in csv_files:
    # Extract parameters from folder name
    folder_name = Path(csv_file).parent.name
    
    # Parse folder name: i_200_k_3_t_2_j_21_signal_strong
    match = re.match(r'i_(\d+)_k_(\d+)_t_(\d+)_j_\d+_signal_(strong|moderate|weak)', folder_name)
    
    if match:
        i_val, k_val, t_val, signal = match.groups()
        i_val, k_val, t_val = int(i_val), int(k_val), int(t_val)
        
        # Read CSV file
        df = pd.read_csv(csv_file)
        
        # Extract metrics for both 0.05 and 0.01 thresholds
        fpr_05 = df[df['Metric'] == 'Q-matrix false positive rate (0.05)']['Mean'].iloc[0]
        fnr_05 = df[df['Metric'] == 'Q-matrix false negative rate (0.05)']['Mean'].iloc[0]
        acc_05 = df[df['Metric'] == 'Q-matrix accuracy (0.05)']['Mean'].iloc[0]
        
        fpr_01 = df[df['Metric'] == 'Q-matrix false positive rate (0.01)']['Mean'].iloc[0]
        fnr_01 = df[df['Metric'] == 'Q-matrix false negative rate (0.01)']['Mean'].iloc[0]
        acc_01 = df[df['Metric'] == 'Q-matrix accuracy (0.01)']['Mean'].iloc[0]
        
        # Store results
        results[signal].append({
            'I': i_val,
            'K': k_val,
            'T': t_val,
            'FPR_05': fpr_05,
            'FNR_05': fnr_05,
            'ACC_05': acc_05,
            'FPR_01': fpr_01,
            'FNR_01': fnr_01,
            'ACC_01': acc_01
        })

# Sort results by I, K, T
for signal in results:
    results[signal].sort(key=lambda x: (x['I'], x['K'], x['T']))

# Function to format percentages
def format_percent_pair(val_05, val_01):
    """Format two values as percentage pair: val_05% / val_01%"""
    if val_05 == 0 and val_01 == 0:
        return "0\\% / 0\\%"
    else:
        return f"{val_05*100:.2f}\\% / {val_01*100:.2f}\\%"

# Generate LaTeX tables with ALL configurations like the RMSE script
for signal in ['strong', 'moderate', 'weak']:
    print(f"\n% {signal.upper()} SIGNAL TABLE")
    print("\\begin{tabular}{cccccc}")
    print("\\hline")
    print("$I$ & $K$ & $T$ & FPR & FNR & Accuracy \\\\")
    print("\\hline")
    
    for row in results[signal]:
        i_str = f"{row['I']:,}" if row['I'] >= 1000 else str(row['I'])
        fpr_str = format_percent_pair(row['FPR_05'], row['FPR_01'])
        fnr_str = format_percent_pair(row['FNR_05'], row['FNR_01'])
        acc_str = format_percent_pair(row['ACC_05'], row['ACC_01'])
        
        print(f"{i_str} & {row['K']} & {row['T']} & {fpr_str} & {fnr_str} & {acc_str} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print()