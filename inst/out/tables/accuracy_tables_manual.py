#!/usr/bin/env python3
import csv
import os

# Helper function to read CSV and extract metrics
def extract_metrics(file_path):
    metrics = {}
    try:
        with open(file_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                metrics[row['Metric']] = {
                    'mean': float(row['Mean']),
                    'sd': float(row['SD'])
                }
    except:
        return None
    return metrics

# Helper function to format percentage pairs
def format_percent_pair(val_05, val_01):
    if val_05 == 0 and val_01 == 0:
        return "0\\% / 0\\%"
    else:
        return f"{val_05*100:.2f}\\% / {val_01*100:.2f}\\%"

# Base path
base_path = "/Users/eduardo/Library/CloudStorage/OneDrive-UCLAITServices/UCLA/Jeon GSR/Code/Good-to-use/HMLCDM/inst/out/tables/multiplerun"

# All configurations in the data
configs = []
for i in [200, 1000]:
    for k in [3, 4]:
        for t in [2, 3, 5]:
            j = 21 if k == 3 else 30
            configs.append((i, k, t, j))

# Extract data for each signal type
results = {'strong': [], 'moderate': [], 'weak': []}

for signal in ['strong', 'moderate', 'weak']:
    for i, k, t, j in configs:
        folder_name = f"i_{i}_k_{k}_t_{t}_j_{j}_signal_{signal}"
        file_path = os.path.join(base_path, folder_name, "summary_table.csv")
        
        if os.path.exists(file_path):
            metrics = extract_metrics(file_path)
            if metrics:
                fpr_05 = metrics.get('Q-matrix false positive rate (0.05)', {}).get('mean', 0)
                fnr_05 = metrics.get('Q-matrix false negative rate (0.05)', {}).get('mean', 0)  
                acc_05 = metrics.get('Q-matrix accuracy (0.05)', {}).get('mean', 0)
                
                fpr_01 = metrics.get('Q-matrix false positive rate (0.01)', {}).get('mean', 0)
                fnr_01 = metrics.get('Q-matrix false negative rate (0.01)', {}).get('mean', 0)
                acc_01 = metrics.get('Q-matrix accuracy (0.01)', {}).get('mean', 0)
                
                results[signal].append({
                    'I': i, 'K': k, 'T': t,
                    'FPR': format_percent_pair(fpr_05, fpr_01),
                    'FNR': format_percent_pair(fnr_05, fnr_01),
                    'ACC': format_percent_pair(acc_05, acc_01)
                })

# Generate LaTeX tables
output_lines = []

for signal in ['strong', 'moderate', 'weak']:
    output_lines.append(f"% {signal.upper()} SIGNAL TABLE")
    output_lines.append("\\begin{tabular}{cccccc}")
    output_lines.append("\\hline")
    output_lines.append("$I$ & $K$ & $T$ & FPR & FNR & Accuracy \\\\")
    output_lines.append("\\hline")
    
    # Sort by I, K, T
    sorted_results = sorted(results[signal], key=lambda x: (x['I'], x['K'], x['T']))
    
    for row in sorted_results:
        i_str = f"{row['I']:,}" if row['I'] >= 1000 else str(row['I'])
        output_lines.append(f"{i_str} & {row['K']} & {row['T']} & {row['FPR']} & {row['FNR']} & {row['ACC']} \\\\")
    
    output_lines.append("\\hline")
    output_lines.append("\\end{tabular}")
    output_lines.append("")

# Write to file
with open('accuracy_tables_complete.tex', 'w') as f:
    f.write('\n'.join(output_lines))

print("Tables generated and saved to accuracy_tables_complete.tex")
print("\nPreview:")
for line in output_lines[:20]:  # Show first 20 lines
    print(line)