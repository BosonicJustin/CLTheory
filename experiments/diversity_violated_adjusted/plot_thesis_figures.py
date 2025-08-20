import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import re
from glob import glob

def parse_mean_std(value_str):
    """Parse 'mean±std' format into separate mean and std values."""
    if '±' in str(value_str):
        parts = str(value_str).split('±')
        return float(parts[0]), float(parts[1])
    else:
        return float(value_str), 0.0

def extract_experiment_info(filename):
    """Extract generative process and encoder type from filename."""
    name = os.path.basename(filename).replace('.csv', '')
    match = re.match(r'g_(.+)_f_(.+)', name)
    if match:
        process = match.group(1).title()
        encoder = match.group(2).replace('_', ' ').title()
        
        # Standardize encoder types for comparison
        if 'inverse' in encoder.lower():
            encoder_type = 'Inverse'
        elif 'mlp' in encoder.lower():
            encoder_type = 'MLP'
        elif 'linear' in encoder.lower():
            encoder_type = 'Inverse'  # Linear is inverse for Identity and Linear processes
        else:
            encoder_type = encoder
            
        return process, encoder_type
    return None, None

def load_and_standardize_csv(filepath):
    """Load CSV and standardize both formats."""
    df = pd.read_csv(filepath)
    process, encoder_type = extract_experiment_info(filepath)
    
    if process is None:
        return None
        
    standardized_data = []
    
    # Detect format: check if we have separate std columns
    if 'Linear_Std' in df.columns:
        # Format 1: Separate columns (Ratio,Linear,Linear_Std,Perm,Perm_Std,...)
        for _, row in df.iterrows():
            standardized_data.append({
                'constraint_ratio': float(row['Ratio']),
                'linear_mean': float(row['Linear']),
                'linear_std': float(row['Linear_Std']),
                'perm_mean': float(row['Perm']),
                'perm_std': float(row['Perm_Std']),
                'angle_mean': float(row['Angle']),
                'angle_std': float(row['Angle_Std']),
                'loss_mean': float(row['Loss']),
                'loss_std': float(row['Loss_Std']),
                'generative_process': process,
                'encoder_type': encoder_type,
                'experiment_file': os.path.basename(filepath)
            })
    else:
        # Format 2: Combined mean±std format (Ratio,Linear,Perm,Angle,Loss)
        for _, row in df.iterrows():
            linear_mean, linear_std = parse_mean_std(row['Linear'])
            perm_mean, perm_std = parse_mean_std(row['Perm'])
            angle_mean, angle_std = parse_mean_std(row['Angle'])
            loss_mean, loss_std = parse_mean_std(row['Loss'])
            
            standardized_data.append({
                'constraint_ratio': float(row['Ratio']),
                'linear_mean': linear_mean,
                'linear_std': linear_std,
                'perm_mean': perm_mean,
                'perm_std': perm_std,
                'angle_mean': angle_mean,
                'angle_std': angle_std,
                'loss_mean': loss_mean,
                'loss_std': loss_std,
                'generative_process': process,
                'encoder_type': encoder_type,
                'experiment_file': os.path.basename(filepath)
            })
    
    return pd.DataFrame(standardized_data)

def create_process_comparison_figures(csv_directory):
    """Create 5 figures (one per generative process) with 2x2 metric grids."""
    
    # Load all CSV files
    csv_files = glob(os.path.join(csv_directory, "*.csv"))
    all_data = []
    
    print(f"🔍 Loading {len(csv_files)} CSV files...")
    for filepath in csv_files:
        df = load_and_standardize_csv(filepath)
        if df is not None:
            all_data.append(df)
            process, encoder = extract_experiment_info(filepath)
            print(f"  ✅ {process} Process → {encoder} Encoder")
    
    if not all_data:
        print("❌ No valid CSV files found!")
        return
    
    # Combine all experiments
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Define metrics for the 2x2 grid
    metrics = [
        ('linear_mean', 'linear_std', 'Linear Performance'),
        ('perm_mean', 'perm_std', 'Permutation Performance'),
        ('angle_mean', 'angle_std', 'Angle Error'),
        ('loss_mean', 'loss_std', 'Loss')
    ]
    
    # Get all unique generative processes
    processes = sorted(combined_df['generative_process'].unique())
    print(f"\n🎯 Creating figures for processes: {processes}")
    
    # Create one figure per generative process
    for process_idx, target_process in enumerate(processes):
        print(f"\n📊 Creating Figure {process_idx + 1}: {target_process} Process")
        
        # Get all data for this process
        process_data = combined_df[combined_df['generative_process'] == target_process]
        available_encoders = sorted(process_data['encoder_type'].unique())
        print(f"   Available encoders: {available_encoders}")
        
        # Create 2x2 subplot figure
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Handle special case for MLP process (only has MLP encoder)
        if target_process == 'Mlp' and len(available_encoders) == 1:
            special_case = True
        else:
            special_case = False
        
        # Plot each metric in the 2x2 grid
        for metric_idx, (mean_col, std_col, metric_name) in enumerate(metrics):
            row = metric_idx // 2
            col = metric_idx % 2
            ax = axes[row, col]
            
            # Colors and markers for different encoders
            colors = {'MLP': 'blue', 'Inverse': 'red'}
            markers = {'MLP': 'o', 'Inverse': 's'}
            
            lines_plotted = 0
            
            # Plot each encoder type
            for encoder in available_encoders:
                encoder_data = process_data[process_data['encoder_type'] == encoder]
                encoder_data = encoder_data.sort_values('constraint_ratio')
                
                if not encoder_data.empty:
                    x = encoder_data['constraint_ratio']
                    y = encoder_data[mean_col]
                    yerr = encoder_data[std_col]
                    
                    # Determine label - clean naming without "inverse" 
                    if encoder == 'MLP':
                        label = 'MLP Encoder'
                    elif encoder == 'Inverse':
                        if target_process in ['Identity', 'Linear']:
                            label = 'Linear Encoder'
                        else:
                            label = f'{target_process} Encoder'
                    else:
                        label = f'{encoder} Encoder'
                    
                    ax.errorbar(x, y, yerr=yerr,
                               label=label,
                               color=colors.get(encoder, 'gray'),
                               marker=markers.get(encoder, '^'),
                               markersize=6,
                               linewidth=2.5,
                               capsize=4,
                               capthick=1.5,
                               alpha=0.8)
                    lines_plotted += 1
            
            # Customize subplot
            ax.set_xlabel('Constraint Ratio', fontsize=12)
            ax.set_ylabel(metric_name, fontsize=12)
            ax.set_title(metric_name, fontsize=13, pad=10)
            ax.grid(True, alpha=0.3)
            ax.set_xlim(-0.05, 1.05)
            ax.set_xticks([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
            ax.tick_params(axis='x', rotation=45)
            
            # Add legend to each subplot
            if lines_plotted > 0:
                ax.legend(loc='best', frameon=True, fancybox=True, shadow=True, fontsize=10)
            
            # Add special annotation for missing inverse case
            if lines_plotted == 0:
                ax.text(0.5, 0.5, 'No Data Available', 
                       ha='center', va='center', fontsize=12, 
                       transform=ax.transAxes, alpha=0.5)
        
        plt.tight_layout()
        
        # Save figure with descriptive name
        process_name_clean = target_process.lower().replace(' ', '_')
        filename = f'figure_{process_idx + 1}_{process_name_clean}_process_comparison.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   💾 Saved: {filename}")
        
        plt.show()
    
    # Print summary
    print(f"\n🎉 Created {len(processes)} figures successfully!")
    print("Each figure shows constraint ratio effects on all 4 metrics for one generative process.")
    print("Ready for thesis inclusion! 📚")

def print_dataset_summary(csv_directory):
    """Print a summary of the available dataset."""
    csv_files = glob(os.path.join(csv_directory, "*.csv"))
    all_data = []
    
    for filepath in csv_files:
        df = load_and_standardize_csv(filepath)
        if df is not None:
            all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        
        print("📋 DATASET SUMMARY")
        print("=" * 50)
        print(f"Total experiments: {len(csv_files)}")
        print(f"Generative processes: {sorted(combined_df['generative_process'].unique())}")
        print(f"Encoder types: {sorted(combined_df['encoder_type'].unique())}")
        print(f"Constraint ratios: {sorted(combined_df['constraint_ratio'].unique())}")
        print("\nProcess-Encoder combinations:")
        for process in sorted(combined_df['generative_process'].unique()):
            encoders = sorted(combined_df[combined_df['generative_process'] == process]['encoder_type'].unique())
            print(f"  {process}: {encoders}")

if __name__ == "__main__":
    csv_directory = "raw_results"
    
    print("🎨 Creating Thesis Figures for Constraint Ratio Experiments")
    print("=" * 70)
    
    # Print dataset summary first
    print_dataset_summary(csv_directory)
    
    print("\n" + "=" * 70)
    print("🚀 Generating Figures...")
    
    # Create the process comparison figures
    create_process_comparison_figures(csv_directory)