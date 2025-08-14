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
from data.generation import InjectiveLinearDecoder, MonomialEmbedding
from spaces import NSphereSpace
from simclr.simclr import SimCLR
from evals.disentanglement import linear_disentanglement, permutation_disentanglement
from evals.distance_preservation import calculate_angle_preservation_error

# Experiment Configuration - Progressive d_fixed from 0 to 10 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'🔧 Using device: {device}')

# Settings for 10D experiments
latent_dim = 10
output_dim = 70
tau = 0.3
kappa = 1 / tau
batch_size = 2000
iterations = 100
n_runs = 5

# Test d_fixed values from 0 (no diversity violation) to 10 (all dimensions fixed)
d_fixed_values = list(range(0, 11))  # [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Pre-initialize all sphere spaces once (efficient!)
full_sphere_10d = NSphereSpace(10)
sub_spheres = {}  # Cache sub-spheres by dimension (only for ≥2D, handle 1D separately)
for remaining_dims in range(2, latent_dim + 1):
    sub_spheres[remaining_dims] = NSphereSpace(remaining_dims)

def sample_conditional_10d_fixed(z, batch, u_dim):
    """Sample conditionally with first u_dim dimensions fixed (10D version)"""
    if u_dim == 0:
        # No dimensions fixed - return standard VMF sample
        return full_sphere_10d.von_mises_fisher(z, kappa, batch)
    elif u_dim >= latent_dim:
        # All dimensions fixed - return original z
        return z.expand(batch, -1)
    else:
        u = z[:, :u_dim]  # Fix first u_dim dimensions
        v = z[:, u_dim:]  # Remaining dimensions
        
        remaining_dims = latent_dim - u_dim
        
        if remaining_dims == 1:
            # Special case: 1D "sphere" is just [-1, 1]
            # For 1D, VMF reduces to sampling ±1 with probabilities based on kappa
            v_norm = torch.nn.functional.normalize(v, dim=-1, p=2)
            # Simple approach: sample uniformly on 1D sphere (just ±1)
            aug_samples_v = torch.sign(torch.randn(batch, 1, device=z.device)) * torch.norm(v, p=2, dim=-1, keepdim=True)
        else:
            # Standard case: use sub-sphere VMF sampling (works fine for 2D, 3D, ...)
            sub_sphere = sub_spheres[remaining_dims]
            v_norm = torch.nn.functional.normalize(v, dim=-1, p=2)
            aug_samples_v = sub_sphere.von_mises_fisher(v_norm, kappa, batch) * torch.norm(v, p=2, dim=-1, keepdim=True)
        
        return torch.cat((u.expand(batch, u_dim), aug_samples_v), dim=-1)

def sample_pair_10d_fixed(batch, u_dim):
    """Sample pairs with fixed dimensions (10D version)"""
    z = full_sphere_10d.uniform(batch)  # (N, 10)
    z_aug = sample_conditional_10d_fixed(z, batch, u_dim)
    return z, z_aug

def sample_uniform_10d(batch):
    """Sample uniform from 10D sphere"""
    return full_sphere_10d.uniform(batch)

def initialize_10d_70d_processes():
    """Initialize the 2 generative processes ONCE: InjectiveLinearDecoder and MonomialEmbedding"""
    processes = {}
    
    # 1. InjectiveLinearDecoder: 10D -> 70D (FIXED WEIGHTS - initialized once!)
    linear_decoder = InjectiveLinearDecoder(latent_dim=10, output_dim=70)
    processes['injective_linear'] = {
        'name': 'InjectiveLinearDecoder_10D_70D',
        'model': linear_decoder,
        'input_dim': 70,
        'description': '10D sphere → 70D injective linear transformation (FIXED WEIGHTS)'
    }
    
    # 2. MonomialEmbedding: 10D -> 70D (DETERMINISTIC - no random weights)
    monomial_embedding = MonomialEmbedding(latent_dim=10, max_degree=7)
    processes['monomial'] = {
        'name': 'MonomialEmbedding_10D_70D',
        'model': monomial_embedding, 
        'input_dim': 70,
        'description': '10D sphere → 70D monomial embedding (DETERMINISTIC)'
    }
    
    print("🏗️  Initialized 2 generative processes for 10D → 70D (CONTROLLED EXPERIMENT):")
    for key, proc in processes.items():
        print(f"  • {proc['name']}: {proc['description']}")
    
    return processes

def train_simclr_for_d_fixed(process_info, d_fixed, run_id, verbose=False):
    """Train SimCLR for a single generative process with specific d_fixed"""
    
    process_name = process_info['name']
    g_model = process_info['model'].to(device)  # Move to device
    encoder_input_dim = process_info['input_dim']
    
    print(f"\n🔄 [{process_name}] d_fixed={d_fixed}, Run {run_id+1}/{n_runs}")
    print(f"   Architecture: 10D sphere → {process_name}(70D) → DeepSphericalEncoder → 10D")
    diversity_status = "NO VIOLATION" if d_fixed == 0 else f"DIVERSITY VIOLATED (d_fixed={d_fixed})"
    print(f"   Sampling: {diversity_status}")
    
    # Create sampling functions for this specific d_fixed
    sample_pair_current = lambda batch: sample_pair_10d_fixed(batch, d_fixed)
    sample_uniform_current = lambda batch: sample_uniform_10d(batch)
    
    # Initialize much deeper and more complex encoder for 70D input
    f_model = SphericalEncoder(
        input_dim=encoder_input_dim,  # 70D input
        hidden_dims=[512, 1024, 1024, 512, 256, 128, 64, 32],  # Much deeper: 8 layers
        output_dim=10  # Output to 10D sphere
    ).to(device)
    
    hidden_dims_str = "[512,1024,1024,512,256,128,64,32]"
    print(f"   Encoder: {encoder_input_dim}D → {hidden_dims_str} → 10D ({sum(p.numel() for p in f_model.parameters())} params)")
    
    # Initialize SimCLR
    simclr = SimCLR(
        encoder=f_model,
        decoder=g_model,
        sample_pair=sample_pair_current,
        sample_uniform=sample_uniform_current,
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
        z_test, z_aug_test = sample_pair_current(batch_size)
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
    del trained_encoder, simclr, f_model
    torch.cuda.empty_cache()
    gc.collect()
    
    result = {
        'process_name': process_name,
        'd_fixed': d_fixed,
        'run_id': run_id,
        'linear_score': linear_score,
        'perm_score': perm_score,
        'angle_error': angle_error,
        'final_loss': final_loss,
        'training_time_seconds': training_time,
        'encoder_input_dim': encoder_input_dim,
        'diversity_violation_ratio': d_fixed / latent_dim,  # 0.0 to 1.0
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
    print("🚀 Starting Progressive d_fixed Experiment (10D → 70D)")
    print(f"Device: {device}, Batch size: {batch_size}, Iterations: {iterations}")
    print(f"Temperature: {tau}, Kappa: {kappa}")
    print(f"Pre-initialized sub-spheres: {list(sub_spheres.keys())}")
    print(f"Testing d_fixed values: {d_fixed_values}")
    print(f"Encoder architecture: 70D → [512,1024,1024,512,256,128,64,32] → 10D (Deep & Complex)")
    print(f"Total experiments: 2 processes × {len(d_fixed_values)} d_fixed × {n_runs} runs = {2 * len(d_fixed_values) * n_runs}")
    
    # Create results directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_dir = "experiment_results_10d"
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f"progressive_d_fixed_10d_70d_{timestamp}.csv")
    
    print(f"📁 Results will be saved to: {results_file}")
    print("=" * 100)
    
    # Initialize processes (do this once and reuse)
    processes = initialize_10d_70d_processes()
    
    # Results storage
    all_results = []
    total_start_time = time.time()
    
    # Run experiments: for each process, for each d_fixed, for each run
    for process_key, process_info in processes.items():
        print(f"\n{'='*50} [{process_info['name']}] {'='*50}")
        
        for d_fixed in d_fixed_values:
            d_fixed_start_time = time.time()
            print(f"\n🔍 Testing d_fixed={d_fixed} (fixing {d_fixed}/{latent_dim} dimensions)")
            
            d_fixed_results = []
            for run in range(n_runs):
                # Use the SAME decoder for all runs/d_fixed values (controlled experiment!)
                result = train_simclr_for_d_fixed(process_info, d_fixed, run, verbose=True)
                d_fixed_results.append(result)
                all_results.append(result)
                
                # Save individual result immediately
                df = pd.DataFrame(all_results)
                df.to_csv(results_file, index=False)
                print(f"   💾 Saved run {run+1} to {results_file}")
            
            # d_fixed summary
            linear_scores = [r['linear_score'] for r in d_fixed_results]
            perm_scores = [r['perm_score'] for r in d_fixed_results]
            angle_errors = [r['angle_error'] for r in d_fixed_results]
            
            d_fixed_time = time.time() - d_fixed_start_time
            print(f"\n📊 [{process_info['name']}] d_fixed={d_fixed} Summary ({d_fixed_time:.1f}s):")
            print(f"   Linear: {np.mean(linear_scores):.3f} ± {np.std(linear_scores):.3f}")
            print(f"   Perm: {np.mean(perm_scores):.3f} ± {np.std(perm_scores):.3f}")
            print(f"   Angle: {np.mean(angle_errors):.3f} ± {np.std(angle_errors):.3f}")
    
    total_time = time.time() - total_start_time
    print(f"\n⏱️  Total experiment time: {total_time/60:.1f} minutes")
    
    # Final summary analysis
    print("\n" + "=" * 100)
    print("📊 FINAL RESULTS SUMMARY (Progressive d_fixed 10D → 70D)")
    print("=" * 100)
    
    df_final = pd.DataFrame(all_results)
    
    # Group by process and d_fixed for detailed summary
    summary = df_final.groupby(['process_name', 'd_fixed']).agg({
        'linear_score': ['mean', 'std'],
        'perm_score': ['mean', 'std'],
        'angle_error': ['mean', 'std'],
        'final_loss': ['mean', 'std'],
        'training_time_seconds': ['mean'],
        'diversity_violation_ratio': ['first']  # This is constant per group
    }).round(4)
    
    print(summary)
    
    # Save detailed summary
    summary_file = os.path.join(results_dir, f"progressive_d_fixed_10d_70d_summary_{timestamp}.csv")
    summary.to_csv(summary_file)
    
    # Compute averages for plotting/analysis
    print("\n📊 Computing averages across all runs...")
    averages_data = []
    
    for process_key, process_info in processes.items():
        for d_fixed in d_fixed_values:
            process_d_fixed_results = [
                r for r in all_results 
                if r['process_name'] == process_info['name'] and r['d_fixed'] == d_fixed
            ]
            
            if not process_d_fixed_results:
                continue
                
            # Collect historical metrics
            all_historical_linear = []
            all_historical_perm = []
            all_historical_angle = []
            all_historical_losses = []
            
            for result in process_d_fixed_results:
                all_historical_linear.extend(result['historical_linear_scores'])
                all_historical_perm.extend(result['historical_perm_scores'])
                all_historical_angle.extend(result['historical_angle_errors'])
                all_historical_losses.extend(result['historical_losses'])
            
            averages_data.append({
                'process_name': process_info['name'],
                'd_fixed': d_fixed,
                'diversity_violation_ratio': d_fixed / latent_dim,
                'avg_final_linear': np.mean([r['linear_score'] for r in process_d_fixed_results]),
                'std_final_linear': np.std([r['linear_score'] for r in process_d_fixed_results]),
                'avg_final_perm': np.mean([r['perm_score'] for r in process_d_fixed_results]),
                'std_final_perm': np.std([r['perm_score'] for r in process_d_fixed_results]),
                'avg_final_angle': np.mean([r['angle_error'] for r in process_d_fixed_results]),
                'std_final_angle': np.std([r['angle_error'] for r in process_d_fixed_results]),
                'avg_final_loss': np.mean([r['final_loss'] for r in process_d_fixed_results]),
                'std_final_loss': np.std([r['final_loss'] for r in process_d_fixed_results]),
                'avg_historical_linear': np.mean(all_historical_linear) if all_historical_linear else 0,
                'avg_historical_perm': np.mean(all_historical_perm) if all_historical_perm else 0,
                'avg_historical_angle': np.mean(all_historical_angle) if all_historical_angle else 0,
                'avg_historical_loss': np.mean(all_historical_losses) if all_historical_losses else 0,
                'num_runs': len(process_d_fixed_results),
                'total_training_time': sum([r['training_time_seconds'] for r in process_d_fixed_results])
            })
    
    # Save averages
    averages_df = pd.DataFrame(averages_data)
    averages_file = os.path.join(results_dir, f"progressive_d_fixed_10d_70d_averages_{timestamp}.csv")
    averages_df.to_csv(averages_file, index=False)
    
    print("\n📊 PROGRESSIVE d_fixed RESULTS:")
    print("-" * 100)
    for process_key, process_info in processes.items():
        print(f"\n🔹 {process_info['name']}:")
        process_avgs = [d for d in averages_data if d['process_name'] == process_info['name']]
        for avg_data in process_avgs:
            violation_pct = avg_data['diversity_violation_ratio'] * 100
            print(f"  d_fixed={avg_data['d_fixed']:2d} ({violation_pct:4.1f}%): "
                  f"Linear={avg_data['avg_final_linear']:.3f}±{avg_data['std_final_linear']:.3f}, "
                  f"Perm={avg_data['avg_final_perm']:.3f}±{avg_data['std_final_perm']:.3f}, "
                  f"Angle={avg_data['avg_final_angle']:.3f}±{avg_data['std_final_angle']:.3f}")
    
    print(f"\n📁 Final files saved:")
    print(f"  • Raw results: {results_file}")
    print(f"  • Summary: {summary_file}")
    print(f"  • Averages: {averages_file}")
    print("=" * 100)
    print(f"🎉 Experiment completed! Total time: {total_time/60:.1f} minutes")

if __name__ == "__main__":
    main()