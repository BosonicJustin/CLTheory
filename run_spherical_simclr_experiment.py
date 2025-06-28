#!/usr/bin/env python3
"""
Spherical SimCLR Experiment Runner

This script runs the spherical toy experiment with SimCLR multiple times,
saving the results and model weights for analysis.
"""

import torch
import os
import json
import pickle
import argparse
from pathlib import Path

from data.generation import SpiralRotation, Patches
from encoders import SphericalEncoder
from simclr.simclr import SimCLR
from spaces import NSphereSpace


def run_spherical_experiment(batch_size=6144, iterations=100000, run_name=None, number_of_runs=1, 
                            diversity_condition="holds"):
    """
    Run the spherical SimCLR experiment multiple times.
    
    Args:
        batch_size: Batch size for training (default: 6144)
        iterations: Number of training iterations (default: 100k)
        run_name: Name for the experiment run (mandatory)
        number_of_runs: Number of times to repeat the experiment (mandatory)
        diversity_condition: Whether diversity condition holds or is violated (default: "holds")
    """
    
    if run_name is None:
        raise ValueError("run_name is mandatory")
    if number_of_runs is None:
        raise ValueError("number_of_runs is mandatory")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Print diversity condition status
    print(f"Diversity condition: {diversity_condition.upper()}")
    
    # Create run directory
    run_dir = Path(f"runs/{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup sphere space and transformations
    sphere = NSphereSpace(3)
    g_spiral = SpiralRotation(2)
    
    # Setup sampling functions based on diversity condition
    tau = 0.3
    kappa = 1 / tau
    
    if diversity_condition == "holds":
        # Normal vMF sampling
        sample_pair_fixed = lambda batch: sphere.sample_pair_vmf(batch, kappa)
    else:  # diversity_condition == "violated"
        # Fixed dimensions sampling - violates diversity condition
        fixed_dims_on_sample = 1
        sub_sphere = NSphereSpace(2)  # For the remaining dimensions
        
        def sample_conditional_with_dims_fixed(z, batch, u_dim):
            u = z[:, :u_dim]
            v = z[:, u_dim:]
            
            v_norm = torch.nn.functional.normalize(v, dim=-1, p=2)
            aug_samples_v = sub_sphere.von_mises_fisher(v_norm, kappa, batch, device=device) * torch.norm(v, p=2, dim=-1, keepdim=True)
            
            return torch.cat((u, aug_samples_v), dim=-1)
        
        def sample_pair_with_fixed_dimension(batch, u_dim):
            z = sphere.uniform(batch, device=device)
            return z, sample_conditional_with_dims_fixed(z, batch, u_dim)
        
        sample_pair_fixed = lambda batch: sample_pair_with_fixed_dimension(batch, fixed_dims_on_sample)
    
    sample_uniform_fixed = lambda batch: sphere.uniform(batch, device=device)
    
    # Store all results
    all_results = {
        'experiment_config': {
            'batch_size': batch_size,
            'iterations': iterations,
            'run_name': run_name,
            'number_of_runs': number_of_runs,
            'diversity_condition': diversity_condition,
            'tau': tau,
            'kappa': kappa,
            'fixed_dims_on_sample': 1 if diversity_condition == "violated" else None,
            'device': str(device)
        },
        'runs': []
    }
    
    print(f"Starting {number_of_runs} runs of spherical SimCLR experiment")
    print(f"Config: batch_size={batch_size}, iterations={iterations}")
    if diversity_condition == "violated":
        print(f"Fixed dimensions for sampling: 1")
    print(f"Results will be saved to: {run_dir}")
    
    for run_idx in range(number_of_runs):
        print(f"\n--- Run {run_idx + 1}/{number_of_runs} ---")
        
        # Initialize fresh encoder for each run
        f = SphericalEncoder()
        
        # Setup SimCLR
        simclr_vmf = SimCLR(
            f, g_spiral, sample_pair_fixed, sample_uniform_fixed, tau, device
        )
        
        # Train the model
        print(f"Training SimCLR for {iterations} iterations...")
        trained_encoder, scores = simclr_vmf.train(batch_size, iterations)
        
        # Save model weights
        model_path = run_dir / f"model_run_{run_idx + 1}.pth"
        torch.save({
            'model_state_dict': trained_encoder.state_dict(),
            'run_idx': run_idx,
            'final_scores': scores,
            'diversity_condition': diversity_condition
        }, model_path)
        
        # Store run results
        run_result = {
            'run_idx': run_idx + 1,
            'model_path': str(model_path),
            'scores': scores
        }
        all_results['runs'].append(run_result)
        
        print(f"Run {run_idx + 1} completed!")
        print(f"Final linear score: {scores['linear_scores'][-1]:.4f}")
        print(f"Final permutation score: {scores['perm_scores'][-1]:.4f}")
        print(f"Final angle preservation error: {scores['angle_preservation_errors'][-1]:.4f}")
        print(f"Model saved to: {model_path}")
    
    # Save aggregated results
    results_path = run_dir / "experiment_results.json"
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Also save as pickle for easier loading in Python
    pickle_path = run_dir / "experiment_results.pkl"
    with open(pickle_path, 'wb') as f:
        pickle.dump(all_results, f)
    
    print(f"\n=== Experiment Complete ===")
    print(f"Completed {number_of_runs} runs")
    print(f"Results saved to: {results_path}")
    print(f"Results also saved as pickle: {pickle_path}")
    
    # Print summary statistics
    final_linear_scores = [run['scores']['linear_scores'][-1] for run in all_results['runs']]
    final_perm_scores = [run['scores']['perm_scores'][-1] for run in all_results['runs']]
    final_angle_errors = [run['scores']['angle_preservation_errors'][-1] for run in all_results['runs']]
    
    print(f"\n=== Summary Statistics ===")
    print(f"Linear Scores - Mean: {torch.tensor(final_linear_scores).mean():.4f}, Std: {torch.tensor(final_linear_scores).std():.4f}")
    print(f"Permutation Scores - Mean: {torch.tensor(final_perm_scores).mean():.4f}, Std: {torch.tensor(final_perm_scores).std():.4f}")
    print(f"Angle Preservation Errors - Mean: {torch.tensor(final_angle_errors).mean():.4f}, Std: {torch.tensor(final_angle_errors).std():.4f}")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description='Run spherical SimCLR experiment')
    parser.add_argument('--batch_size', type=int, default=6144, 
                        help='Batch size for training (default: 6144)')
    parser.add_argument('--iterations', type=int, default=100000,
                        help='Number of training iterations (default: 100000)')
    parser.add_argument('--run_name', type=str, required=True,
                        help='Name for the experiment run (required)')
    parser.add_argument('--number_of_runs', type=int, required=True,
                        help='Number of times to repeat the experiment (required)')
    parser.add_argument('--diversity_condition', type=str, default='holds',
                        choices=['holds', 'violated'],
                        help='Whether diversity condition holds or is violated (default: holds)')
    
    args = parser.parse_args()
    
    run_spherical_experiment(
        batch_size=args.batch_size,
        iterations=args.iterations,
        run_name=args.run_name,
        number_of_runs=args.number_of_runs,
        diversity_condition=args.diversity_condition
    )


if __name__ == "__main__":
    main() 