import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def load_and_process_data(csv_path):
    """Load and process the progressive diversity violation data."""
    df = pd.read_csv(csv_path)
    
    # Clean up encoder names for better plotting
    df['encoder_clean'] = df['process_name'].str.replace('_10D_70D', '').str.replace('InjectiveLinearDecoder', 'Linear Process ($g$)').str.replace('MonomialEmbedding', 'Monomial Process ($g$)')
    
    return df

def create_diversity_violation_plots(csv_path, save_path=None):
    """Create 4 plots showing the effect of diversity violation on different metrics."""
    
    # Load data
    df = load_and_process_data(csv_path)
    
    # Set up the figure with 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Define metrics and their properties
    metrics = [
        ('avg_final_linear', 'std_final_linear', 'Final Linear Metric', 'Linear Performance'),
        ('avg_final_perm', 'std_final_perm', 'Final Perm Metric', 'Permutation Performance'),  
        ('avg_final_angle', 'std_final_angle', 'Final Angle Metric', 'Angle Performance'),
        ('avg_final_loss', 'std_final_loss', 'Final Loss', 'Loss')
    ]
    
    # Colors and markers for different encoders
    colors = ['#1f77b4', '#ff7f0e']  # Blue and orange
    markers = ['o', 's']  # Circle and square
    
    # Create each subplot
    for idx, (mean_col, std_col, title, ylabel) in enumerate(metrics):
        row = idx // 2
        col = idx % 2
        ax = axes[row, col]
        
        # Plot for each encoder type
        for i, encoder in enumerate(df['encoder_clean'].unique()):
            encoder_data = df[df['encoder_clean'] == encoder].sort_values('d_fixed')
            
            x = encoder_data['d_fixed']
            y = encoder_data[mean_col]
            yerr = encoder_data[std_col]
            
            # Plot line with error bars
            ax.errorbar(x, y, yerr=yerr, 
                       label=encoder, 
                       color=colors[i], 
                       marker=markers[i], 
                       markersize=6,
                       linewidth=2, 
                       capsize=3, 
                       capthick=1)
        
        # Customize subplot
        ax.set_xlabel('$d_{\\mathrm{fixed}}$', fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fancybox=True, shadow=True)
        
        # Set x-axis to show all integer values from 0 to 10
        ax.set_xticks(range(0, 11))
        
        # Add secondary x-axis showing violation ratio
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(range(0, 11))
        ax2.set_xticklabels([f'{i/10:.1f}' for i in range(0, 11)], fontsize=9)
        ax2.set_xlabel('Diversity Violation Ratio', fontsize=9, color='gray')
        
    # Adjust layout to prevent overlap
    plt.tight_layout()
    
    # Save or show
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()
    
    return fig

def create_individual_plots(csv_path, save_dir=None):
    """Create individual plots for each metric (for more detailed analysis)."""
    
    df = load_and_process_data(csv_path)
    
    # Define metrics and their properties
    metrics = [
        ('avg_final_linear', 'std_final_linear', 'Final Linear Metric', 'Linear Performance'),
        ('avg_final_perm', 'std_final_perm', 'Final Perm Metric', 'Permutation Performance'),  
        ('avg_final_angle', 'std_final_angle', 'Final Angle Metric', 'Angle Performance'),
        ('avg_final_loss', 'std_final_loss', 'Final Loss', 'Loss')
    ]
    
    colors = ['#1f77b4', '#ff7f0e']
    markers = ['o', 's']
    
    figures = []
    
    for mean_col, std_col, title, ylabel in metrics:
        fig, ax = plt.subplots(1, 1, figsize=(8, 6))
        
        # Plot for each encoder type
        for i, encoder in enumerate(df['encoder_clean'].unique()):
            encoder_data = df[df['encoder_clean'] == encoder].sort_values('d_fixed')
            
            x = encoder_data['d_fixed']
            y = encoder_data[mean_col]
            yerr = encoder_data[std_col]
            
            ax.errorbar(x, y, yerr=yerr, 
                       label=encoder, 
                       color=colors[i], 
                       marker=markers[i], 
                       markersize=8,
                       linewidth=2.5, 
                       capsize=4, 
                       capthick=1.5)
        
        # Customize plot
        ax.set_xlabel('$d_{\\mathrm{fixed}}$', fontsize=14)
        ax.set_ylabel(ylabel, fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=12)
        
        # Set x-axis
        ax.set_xticks(range(0, 11))
        
        # Add secondary x-axis showing violation ratio
        ax2 = ax.twiny()
        ax2.set_xlim(ax.get_xlim())
        ax2.set_xticks(range(0, 11))
        ax2.set_xticklabels([f'{i/10:.1f}' for i in range(0, 11)])
        ax2.set_xlabel('Diversity Violation Ratio', fontsize=12, color='gray')
        
        plt.tight_layout()
        
        if save_dir:
            filename = f"{save_dir}/diversity_violation_{mean_col}.png"
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            print(f"Individual plot saved to {filename}")
            
        figures.append(fig)
    
    return figures

# Main execution
if __name__ == "__main__":
    csv_path = "experiment_results_10d/progressive_d_fixed_10d_70d_averages_20250814_061808.csv"
    
    print("Creating combined 2x2 plot...")
    fig_combined = create_diversity_violation_plots(csv_path, "diversity_violation_combined.png")
    
    print("\nCreating individual plots...")
    figs_individual = create_individual_plots(csv_path, ".")
    
    print("\nVisualization complete!")
    print("This shows how increasing the number of fixed dimensions (diversity violation)")
    print("affects the performance of different encoder types across multiple metrics.")