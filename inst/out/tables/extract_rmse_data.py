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
        
        # Extract RMSE values
        beta_rmse = df[df['Metric'] == 'Beta RMSE']['Mean'].iloc[0]
        beta_rmse_sd = df[df['Metric'] == 'Beta RMSE']['SD'].iloc[0]
        
        pii_rmse = df[df['Metric'] == 'Pii RMSE']['Mean'].iloc[0]
        pii_rmse_sd = df[df['Metric'] == 'Pii RMSE']['SD'].iloc[0]
        
        tau_rmse = df[df['Metric'] == 'Tau RMSE']['Mean'].iloc[0]
        tau_rmse_sd = df[df['Metric'] == 'Tau RMSE']['SD'].iloc[0]
        
        # Store results
        results[signal].append({
            'I': i_val,
            'K': k_val,
            'T': t_val,
            'beta_rmse': beta_rmse,
            'beta_rmse_sd': beta_rmse_sd,
            'pii_rmse': pii_rmse,
            'pii_rmse_sd': pii_rmse_sd,
            'tau_rmse': tau_rmse,
            'tau_rmse_sd': tau_rmse_sd
        })

# Sort results by I, K, T
for signal in results:
    results[signal].sort(key=lambda x: (x['I'], x['K'], x['T']))

# Function to format numbers for LaTeX
def format_rmse(mean, sd):
    if mean >= 1:
        mean_str = f"{mean:.2f}"
    elif mean >= 0.1:
        mean_str = f"{mean:.2f}"
    elif mean >= 0.01:
        mean_str = f"{mean:.3f}"
    else:
        mean_str = f"{mean:.4f}"
    
    # Format SD in scientific notation if very small
    if sd < 0.01:
        sd_str = f"{sd:.1e}"
    else:
        sd_str = f"{sd:.3f}"
    
    return f"{mean_str} (\\num{{{sd_str}}})"

# Generate LaTeX tables
for signal in ['strong', 'moderate', 'weak']:
    print(f"\n% {signal.upper()} SIGNAL TABLE")
    print("\\begin{tabular}{ccccccc}")
    print("\\hline")
    print("$I$ & $K$ & $T$ & $\\beta$ RMSE & $\\tau$ RMSE & $\\pi$ RMSE \\\\")
    print("\\hline")
    
    for row in results[signal]:
        i_str = f"{row['I']:,}" if row['I'] >= 1000 else str(row['I'])
        beta_str = format_rmse(row['beta_rmse'], row['beta_rmse_sd'])
        tau_str = format_rmse(row['tau_rmse'], row['tau_rmse_sd'])
        pii_str = format_rmse(row['pii_rmse'], row['pii_rmse_sd'])
        
        print(f"{i_str} & {row['K']} & {row['T']} & {beta_str} & {tau_str} & {pii_str} \\\\")
    
    print("\\hline")
    print("\\end{tabular}")
    print()