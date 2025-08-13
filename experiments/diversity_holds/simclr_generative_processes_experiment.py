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

# Experiment Configuration - matching diversity_holds experiments
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🔧 Using device: {device}')

# Settings from diversity_holds experiments
latent_dim = 3
tau = 0.3
kappa = 1 / tau
batch_size = 2000
iterations = 100
n_runs = 5

# Initialize sphere space - same as diversity_holds
sphere = NSphereSpace(latent_dim)

# Sampling functions - SAME AS DIVERSITY_HOLDS experiments
sample_pair_fixed = lambda batch: sphere.sample_pair_vmf(batch, kappa)
sample_uniform_fixed = lambda batch: sphere.uniform(batch)

def initialize_generative_processes():
    """Initialize all 4 generative processes on CPU"""
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
        'model': Patches(slice_number=4, device='cpu'),
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
    
    print("🏗️  Initialized 4 generative processes:")
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
    
    # Move generative process to device
    g_model = g_model.to(device)
    
    # Initialize fresh encoder for this run - same architecture as diversity_holds
    f_model = SphericalEncoder(
        input_dim=encoder_input_dim, 
        hidden_dims=[128, 256, 256, 256, 128], 
        output_dim=3  # Always output to 3D sphere
    ).to(device)

    # For Patches, also update the internal device parameter
    if hasattr(g_model, 'device'):
        g_model.device = device
    
    # Initialize SimCLR - same as diversity_holds experiments
    simclr = SimCLR(
        encoder=f_model,
        decoder=g_model, 
        sample_pair=sample_pair_fixed,      # VMF sampling
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
        'encoder_input_dim': encoder_input_dim
    }
    
    print(f"   ✅ Results: Linear={linear_score:.3f}, Perm={perm_score:.3f}, Angle={angle_error:.3f}, Loss={final_loss:.3f} ({training_time:.1f}s)")
    
    return result

def main():
    """Main experiment function"""
    print("🚀 Starting SimCLR Generative Processes Experiment")
    print(f"Device: {device}, Batch size: {batch_size}, Iterations: {iterations}")
    print(f"Temperature: {tau}, Kappa: {kappa}")
    print(f"Sampling: VMF pairs + Uniform negatives (same as diversity_holds)")
    print(f"Total experiments: 4 processes × {n_runs} runs = {4 * n_runs}")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "experiment_results"
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f"simclr_generative_processes_{timestamp}.csv")
    
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
    print("📊 FINAL RESULTS SUMMARY")
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
    summary_file = os.path.join(results_dir, f"simclr_generative_processes_summary_{timestamp}.csv")
    summary.to_csv(summary_file)
    
    print(f"\n📁 Final files saved:")
    print(f"  • Raw results: {results_file}")
    print(f"  • Summary: {summary_file}")
    print("=" * 80)

if __name__ == "__main__":
    main()