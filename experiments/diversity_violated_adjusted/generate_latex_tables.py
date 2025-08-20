import pandas as pd
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
        
        # Clean up names for LaTeX
        if 'inverse' in encoder.lower():
            if 'patches' in encoder.lower():
                encoder_clean = 'Patches'
            elif 'spiral' in encoder.lower():
                encoder_clean = 'Spiral'
            else:
                encoder_clean = encoder.replace('Inverse ', '')
        elif 'mlp' in encoder.lower():
            encoder_clean = 'MLP'
        elif 'linear' in encoder.lower():
            encoder_clean = 'Linear'
        else:
            encoder_clean = encoder
            
        return process, encoder_clean
    return None, None

def load_and_standardize_csv(filepath):
    """Load CSV and convert both formats to standardized format."""
    df = pd.read_csv(filepath)
    process, encoder = extract_experiment_info(filepath)
    
    if process is None:
        return None, None, None
        
    standardized_data = []
    
    # Detect format and parse accordingly
    if 'Linear_Std' in df.columns:
        # Format 1: Separate columns
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
                'loss_std': float(row['Loss_Std'])
            })
    else:
        # Format 2: Combined mean±std format
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
                'loss_std': loss_std
            })
    
    return pd.DataFrame(standardized_data), process, encoder

def generate_latex_table(df, process, encoder, table_counter):
    """Generate a LaTeX table for one experiment."""
    
    # Find best values for highlighting
    max_linear_idx = df['linear_mean'].idxmax()
    max_perm_idx = df['perm_mean'].idxmax()
    min_angle_idx = df['angle_mean'].idxmin()
    min_loss_idx = df['loss_mean'].idxmin()
    
    # Create table header
    latex = f"\\begin{{table}}[h]\n"
    latex += f"  \\centering\n"
    latex += f"  \\small\n"
    latex += f"  \\renewcommand{{\\arraystretch}}{{1.5}}\n"
    latex += f"  \\begin{{tabular}}{{|c|c|c|c|c|}}\n"
    latex += f"  \\hline\n"
    latex += f"  Constraint Ratio & Linear & Perm & Angle & Loss \\\\\n"
    latex += f"  \\hline\n"
    
    # Add data rows
    for idx, row in df.iterrows():
        ratio = f"{row['constraint_ratio']:.1f}"
        
        # Format with highlighting for best values
        if idx == max_linear_idx:
            linear = f"$\\textbf{{{row['linear_mean']:.3f}}} \\pm {row['linear_std']:.3f}$"
        else:
            linear = f"${row['linear_mean']:.3f} \\pm {row['linear_std']:.3f}$"
            
        if idx == max_perm_idx:
            perm = f"$\\textbf{{{row['perm_mean']:.3f}}} \\pm {row['perm_std']:.3f}$"
        else:
            perm = f"${row['perm_mean']:.3f} \\pm {row['perm_std']:.3f}$"
            
        if idx == min_angle_idx:
            angle = f"$\\textbf{{{row['angle_mean']:.3f}}} \\pm {row['angle_std']:.3f}$"
        else:
            angle = f"${row['angle_mean']:.3f} \\pm {row['angle_std']:.3f}$"
            
        if idx == min_loss_idx:
            loss = f"$\\textbf{{{row['loss_mean']:.3f}}} \\pm {row['loss_std']:.3f}$"
        else:
            loss = f"${row['loss_mean']:.3f} \\pm {row['loss_std']:.3f}$"
        
        latex += f"  {ratio} & {linear} & {perm} & {angle} & {loss} \\\\\n"
        latex += f"  \\hline\n"
    
    # Close table with caption and label
    latex += f"  \\end{{tabular}}\n"
    
    # Create descriptive caption
    if encoder == process:  # Special case for MLP process
        caption = f"Constraint ratio experiment results for {process} generative process with {encoder} encoder learning approximate inverse"
    else:
        caption = f"Constraint ratio experiment results for {process} generative process with {encoder} encoder"
    
    latex += f"  \\caption{{{caption}}}\n"
    
    # Create meaningful label
    process_label = process.lower()
    encoder_label = encoder.lower().replace(' ', '_')
    latex += f"  \\label{{tab:constraint_{process_label}_{encoder_label}}}\n"
    latex += f"\\end{{table}}\n\n"
    
    return latex

def generate_all_latex_tables(csv_directory, output_file):
    """Generate LaTeX tables for all CSV files and save to one output file."""
    
    csv_files = glob(os.path.join(csv_directory, "*.csv"))
    all_latex = ""
    table_counter = 1
    
    print(f"🔍 Processing {len(csv_files)} CSV files...")
    
    # Sort files for consistent ordering
    csv_files.sort()
    
    for filepath in csv_files:
        print(f"  Processing: {os.path.basename(filepath)}")
        
        df, process, encoder = load_and_standardize_csv(filepath)
        
        if df is not None:
            latex_table = generate_latex_table(df, process, encoder, table_counter)
            all_latex += latex_table
            table_counter += 1
            print(f"    ✅ Generated table for {process} → {encoder}")
        else:
            print(f"    ❌ Could not process {os.path.basename(filepath)}")
    
    # Add document structure comments for easy copying
    header = "% LaTeX tables for constraint ratio experiments\n"
    header += "% Generated automatically from CSV files\n"
    header += "% Copy and paste the tables below into your appendix\n\n"
    
    footer = "% End of generated tables\n"
    footer += f"% Total tables generated: {table_counter - 1}\n"
    
    final_latex = header + all_latex + footer
    
    # Save to output file
    with open(output_file, 'w') as f:
        f.write(final_latex)
    
    print(f"\n✅ Generated {table_counter - 1} LaTeX tables")
    print(f"📁 Saved to: {output_file}")
    print(f"📋 Ready to copy and paste into your thesis appendix!")
    
    return final_latex

def print_experiment_summary(csv_directory):
    """Print a summary of available experiments."""
    csv_files = glob(os.path.join(csv_directory, "*.csv"))
    
    print("📊 EXPERIMENT SUMMARY")
    print("=" * 50)
    
    experiments = []
    for filepath in csv_files:
        process, encoder = extract_experiment_info(filepath)
        if process and encoder:
            experiments.append((process, encoder, os.path.basename(filepath)))
    
    experiments.sort()
    
    for i, (process, encoder, filename) in enumerate(experiments, 1):
        print(f"Table {i:2}: {process:<8} Process → {encoder:<8} Encoder ({filename})")
    
    print(f"\nTotal experiments: {len(experiments)}")
    print("All tables will be generated in appendix-ready format.\n")

if __name__ == "__main__":
    csv_directory = "raw_results"
    output_file = "appendix_tables.tex"
    
    print("📝 Generating LaTeX Tables for Thesis Appendix")
    print("=" * 60)
    
    # Print summary first
    print_experiment_summary(csv_directory)
    
    print("🚀 Generating LaTeX code...")
    
    # Generate all tables
    latex_content = generate_all_latex_tables(csv_directory, output_file)
    
    print(f"\n🎉 SUCCESS!")
    print(f"Open {output_file} and copy all content into your thesis appendix.")
    print("Each table has proper captions and labels for referencing.")