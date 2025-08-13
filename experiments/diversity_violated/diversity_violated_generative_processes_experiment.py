import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), '..'))

import torch
from torch import nn
import numpy as np
import pandas as pd
from datetime import datetime
import time
import gc

# Imports from existing codebase
from encoders import SphericalEncoder
from data.generation import InjectiveLinearDecoder, Patches, SpiralRotation
from invertible_network_utils import construct_invertible_mlp
from spaces import NSphereSpace
from simclr.simclr import SimCLR
from evals.disentanglement import linear_disentanglement, permutation_disentanglement
from evals.distance_preservation import calculate_angle_preservation_error

# Experiment Configuration - diversity_violated with fixed dimensions
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🔧 Using device: {device}')

# Settings from diversity_violated experiments
latent_dim = 3
tau = 0.3
kappa = 1 / tau
batch_size = 2000
iterations = 100
n_runs = 5
fixed_dims_on_sample = 1  # KEY: Fix first dimension (d_fixed = 1)

# Initialize sphere spaces - SAME AS DIVERSITY_VIOLATED
full_sphere = NSphereSpace(3)
sub_sphere = NSphereSpace(2)

# CONDITIONAL SAMPLING WITH FIXED DIMENSIONS - FROM DIVERSITY_VIOLATED
def sample_conditional_with_dims_fixed(z, batch, u_dim):
    """Sample conditionally with first u_dim dimensions fixed"""
    u = z[:, :u_dim]  # Fix first u_dim dimensions
    v = z[:, u_dim:]  # Remaining dimensions
    
    v_norm = torch.nn.functional.normalize(v, dim=-1, p=2)
    aug_samples_v = sub_sphere.von_mises_fisher(v_norm, kappa, batch) * torch.norm(v, p=2, dim=-1, keepdim=True)
    
    return torch.cat((u, aug_samples_v), dim=-1)

def sample_pair_with_fixed_dimension(batch, u_dim):
    """Sample pairs with fixed dimensions - SAME AS DIVERSITY_VIOLATED"""
    z = full_sphere.uniform(batch)  # (N, 3)
    return z, sample_conditional_with_dims_fixed(z, batch, u_dim)

def sample_uniform_sphere(batch):
    """Sample uniform from sphere"""
    return full_sphere.uniform(batch)

# Create sample functions - WITH FIXED DIMENSIONS
sample_pair_fixed = lambda batch: sample_pair_with_fixed_dimension(batch, fixed_dims_on_sample)
sample_uniform_fixed = lambda batch: sample_uniform_sphere(batch)

def initialize_generative_processes():
    """Initialize all 5 generative processes on CPU"""
    processes = {}
    
    # 1. InjectiveLinearDecoder: 3D -> 7D
    processes['linear'] = {
        'name': 'InjectiveLinearDecoder',
        'model': InjectiveLinearDecoder(latent_dim=3, output_dim=7),
        'input_dim': 7,  # What the encoder will receive
        'description': '3D sphere → 7D linear transformation'
    }
    
    # 2. Patches: 3D -> 3D
    processes['patches'] = {
        'name': 'Patches', 
        'model': Patches(slice_number=4, device='cpu'),  # Initialize on CPU first
        'input_dim': 3,
        'description': '3D sphere → 3D patches transformation'
    }
    
    # 3. SpiralRotation: 3D -> 3D
    processes['spiral'] = {
        'name': 'SpiralRotation',
        'model': SpiralRotation(2),
        'input_dim': 3,
        'description': '3D sphere → 3D spiral rotation'
    }

    # 4. Invertible MLP: 3D -> 3D
    processes['invertible_mlp'] = {
        'name': 'InvertibleMLP',
        'model': construct_invertible_mlp(n=3, n_layers=3, act_fct="leaky_relu"),
        'input_dim': 3,
        'description': '3D sphere → 3D invertible MLP'
    }
    
    # 5. Identity: 3D -> 3D
    processes['identity'] = {
        'name': 'Identity',
        'model': torch.nn.Identity(),
        'input_dim': 3,
        'description': '3D sphere → 3D identity (no transformation)'
    }
    
    print("🏗️  Initialized 5 generative processes with conditional sampling (d_fixed=1):")
    for key, proc in processes.items():
        print(f"  • {proc['name']}: {proc['description']}")
    
    return processes

def train_simclr_for_process(process_info, run_id, verbose=False):
    """Train SimCLR for a single generative process"""
    
    process_name = process_info['name']
    g_model = process_info['model']
    encoder_input_dim = process_info['input_dim']
    
    print(f"\n🔄 [{process_name}] Run {run_id+1}/{n_runs}")
    print(f"   Architecture: 3D sphere → {process_name}({encoder_input_dim}D) → SphericalEncoder → 3D")
    print(f"   Sampling: Conditional VMF with d_fixed={fixed_dims_on_sample} (DIVERSITY VIOLATED)")
    
    # Move generative process to device
    g_model = g_model.to(device)
    
    # For Patches, also update the internal device parameter
    if hasattr(g_model, 'device'):
        g_model.device = device
    
    # Initialize fresh encoder for this run - same architecture
    f_model = SphericalEncoder(
        input_dim=encoder_input_dim, 
        hidden_dims=[128, 256, 256, 256, 128], 
        output_dim=3  # Always output to 3D sphere
    ).to(device)
    
    # Initialize SimCLR - with conditional sampling
    simclr = SimCLR(
        encoder=f_model,
        decoder=g_model, 
        sample_pair=sample_pair_fixed,      # Conditional sampling with fixed dims
        sample_uniform=sample_uniform_fixed, # Uniform sphere sampling
        temperature=tau,
        device=device
    )
    
    # Train
    start_time = time.time()
    trained_encoder, scores = simclr.train(batch_size, iterations)
    training_time = time.time() - start_time
    
    # Final evaluation
    trained_encoder.eval()
    with torch.no_grad():
        # Generate test data using same sampling as training
        z_test, z_aug_test = sample_pair_fixed(batch_size)
        z_test = z_test.to(device)
        z_aug_test = z_aug_test.to(device)
        
        # Forward pass through full model: h = f(g(z))
        h = lambda latent: trained_encoder(g_model(latent))
        z_enc_test = h(z_test)
        
        # Compute final metrics
        lin_dis, _ = linear_disentanglement(z_test.cpu(), z_enc_test.cpu())
        linear_score, _ = lin_dis
        
        perm_dis, _ = permutation_disentanglement(z_test.cpu(), z_enc_test.cpu(), mode="pearson", solver="munkres")
        perm_score, _ = perm_dis
        
        angle_error = calculate_angle_preservation_error(z_test.cpu(), z_enc_test.cpu())
        
        # Get final loss from scores
        final_loss = scores['eval_losses'][-1] if scores['eval_losses'] else 0.0
    
    # Cleanup GPU memory
    del trained_encoder, simclr, f_model, g_model
    torch.cuda.empty_cache()
    gc.collect()
    
    result = {
        'process_name': process_name,
        'run_id': run_id,
        'linear_score': linear_score,
        'perm_score': perm_score, 
        'angle_error': angle_error,
        'final_loss': final_loss,
        'training_time_seconds': training_time,
        'encoder_input_dim': encoder_input_dim,
        'fixed_dims': fixed_dims_on_sample,
        # Historical metrics from training
        'historical_linear_scores': scores['linear_scores'],
        'historical_perm_scores': scores['perm_scores'], 
        'historical_angle_errors': scores['angle_preservation_errors'],
        'historical_losses': scores['eval_losses'],
        'historical_pos_losses': scores['eval_pos_losses'],
        'historical_neg_losses': scores['eval_neg_losses']
    }
    
    print(f"   ✅ Results: Linear={linear_score:.3f}, Perm={perm_score:.3f}, Angle={angle_error:.3f}, Loss={final_loss:.3f} ({training_time:.1f}s)")
    
    return result

def main():
    """Main experiment function"""
    print("🚀 Starting Diversity Violated Generative Processes Experiment")
    print(f"Device: {device}, Batch size: {batch_size}, Iterations: {iterations}")
    print(f"Temperature: {tau}, Kappa: {kappa}")
    print(f"🔴 DIVERSITY VIOLATED: Conditional sampling with d_fixed={fixed_dims_on_sample}")
    print(f"Total experiments: 5 processes × {n_runs} runs = {5 * n_runs}")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "experiment_results"
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f"diversity_violated_generative_processes_{timestamp}.csv")
    
    print(f"📁 Results will be saved to: {results_file}")
    print("=" * 80)
    
    # Initialize all processes
    processes = initialize_generative_processes()
    
    # Results storage
    all_results = []
    total_start_time = time.time()
    
    # Run experiments
    for process_key, process_info in processes.items():
        process_start_time = time.time()
        print(f"\n[{process_info['name']}] Starting {n_runs} runs...")
        
        process_results = []
        for run in range(n_runs):
            result = train_simclr_for_process(process_info, run, verbose=True)
            process_results.append(result)
            all_results.append(result)
            
            # Save individual result immediately
            df = pd.DataFrame(all_results)
            df.to_csv(results_file, index=False)
            print(f"   💾 Saved run {run+1} to {results_file}")
        
        # Process summary
        linear_scores = [r['linear_score'] for r in process_results]
        perm_scores = [r['perm_score'] for r in process_results]
        angle_errors = [r['angle_error'] for r in process_results]
        
        process_time = time.time() - process_start_time
        print(f"\n📊 [{process_info['name']}] Summary ({process_time:.1f}s):")
        print(f"   Linear: {np.mean(linear_scores):.3f} ± {np.std(linear_scores):.3f}")
        print(f"   Perm: {np.mean(perm_scores):.3f} ± {np.std(perm_scores):.3f}")
        print(f"   Angle: {np.mean(angle_errors):.3f} ± {np.std(angle_errors):.3f}")
    
    total_time = time.time() - total_start_time
    print(f"\n⏱️  Total experiment time: {total_time/60:.1f} minutes")
    
    # Final summary
    print("\n" + "=" * 80)
    print("📊 FINAL RESULTS SUMMARY (Diversity Violated)")
    print("=" * 80)
    
    df_final = pd.DataFrame(all_results)
    summary = df_final.groupby('process_name').agg({
        'linear_score': ['mean', 'std'],
        'perm_score': ['mean', 'std'],
        'angle_error': ['mean', 'std'],
        'final_loss': ['mean', 'std'],
        'training_time_seconds': ['mean']
    }).round(4)
    
    print(summary)
    
    # Save final summary
    summary_file = os.path.join(results_dir, f"diversity_violated_generative_processes_summary_{timestamp}.csv")
    summary.to_csv(summary_file)
    
    # Compute and save averages across all runs
    print("\n📊 Computing averages across all runs...")
    averages_data = []
    for process_key, process_info in processes.items():
        process_results = [r for r in all_results if r['process_name'] == process_info['name']]
        
        # Collect all historical metrics across runs
        all_historical_linear = []
        all_historical_perm = []
        all_historical_angle = []
        all_historical_losses = []
        
        for result in process_results:
            all_historical_linear.extend(result['historical_linear_scores'])
            all_historical_perm.extend(result['historical_perm_scores'])
            all_historical_angle.extend(result['historical_angle_errors'])
            all_historical_losses.extend(result['historical_losses'])
        
        averages_data.append({
            'process_name': process_info['name'],
            'fixed_dims': fixed_dims_on_sample,
            'avg_final_linear': np.mean([r['linear_score'] for r in process_results]),
            'std_final_linear': np.std([r['linear_score'] for r in process_results]),
            'avg_final_perm': np.mean([r['perm_score'] for r in process_results]),
            'std_final_perm': np.std([r['perm_score'] for r in process_results]),
            'avg_final_angle': np.mean([r['angle_error'] for r in process_results]),
            'std_final_angle': np.std([r['angle_error'] for r in process_results]),
            'avg_final_loss': np.mean([r['final_loss'] for r in process_results]),
            'std_final_loss': np.std([r['final_loss'] for r in process_results]),
            'avg_historical_linear': np.mean(all_historical_linear) if all_historical_linear else 0,
            'avg_historical_perm': np.mean(all_historical_perm) if all_historical_perm else 0,
            'avg_historical_angle': np.mean(all_historical_angle) if all_historical_angle else 0,
            'avg_historical_loss': np.mean(all_historical_losses) if all_historical_losses else 0,
            'num_runs': len(process_results),
            'total_training_time': sum([r['training_time_seconds'] for r in process_results])
        })
    
    # Save averages
    averages_df = pd.DataFrame(averages_data)
    averages_file = os.path.join(results_dir, f"diversity_violated_generative_processes_averages_{timestamp}.csv")
    averages_df.to_csv(averages_file, index=False)
    
    print("\n📊 PROCESS AVERAGES (Diversity Violated):")
    print("-" * 80)
    for avg_data in averages_data:
        print(f"{avg_data['process_name']} (d_fixed={avg_data['fixed_dims']}):")
        print(f"  Final: Linear={avg_data['avg_final_linear']:.3f}±{avg_data['std_final_linear']:.3f}, "
              f"Perm={avg_data['avg_final_perm']:.3f}±{avg_data['std_final_perm']:.3f}")
        print(f"  Historical Avg: Linear={avg_data['avg_historical_linear']:.3f}, "
              f"Perm={avg_data['avg_historical_perm']:.3f}")
    
    print(f"\n📁 Final files saved:")
    print(f"  • Raw results: {results_file}")
    print(f"  • Summary: {summary_file}")
    print(f"  • Averages: {averages_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()