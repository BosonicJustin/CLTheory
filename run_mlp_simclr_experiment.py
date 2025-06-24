#!/usr/bin/env python3
"""
MLP SimCLR Experiment Runner

This script runs the spherical toy experiment with SimCLR using an invertible MLP 
transformation multiple times, saving the results and model weights for analysis.
"""

import torch
import os
import json
import pickle
import argparse
from pathlib import Path

from typing import Literal
from invertible_network_utils import construct_invertible_mlp
from encoders import SphericalEncoder
from simclr.simclr import SimCLR
from spaces import NSphereSpace


def run_mlp_experiment(batch_size=6144, iterations=100000, run_name=None, number_of_runs=1, 
                       n_layers=3, act_fct: Literal['relu', 'leaky_relu', 'elu', 'smooth_leaky_relu', 'softplus'] = "leaky_relu", 
                       cond_thresh_ratio=0.25):
    """
    Run the spherical SimCLR experiment with invertible MLP transformation multiple times.
    
    Args:
        batch_size: Batch size for training (default: 6144)
        iterations: Number of training iterations (default: 100k)
        run_name: Name for the experiment run (mandatory)
        number_of_runs: Number of times to repeat the experiment (mandatory)
        n_layers: Number of layers in the invertible MLP (default: 3)
        act_fct: Activation function for the MLP (default: "leaky_relu")
        cond_thresh_ratio: Condition threshold ratio for MLP construction (default: 0.25)
    """
    
    if run_name is None:
        raise ValueError("run_name is mandatory")
    if number_of_runs is None:
        raise ValueError("number_of_runs is mandatory")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    # Create run directory
    run_dir = Path(f"runs/{run_name}")
    run_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup sphere space and transformations
    sphere = NSphereSpace(3)
    
    # Setup sampling functions
    tau = 0.3
    kappa = 1 / tau
    sample_pair_fixed = lambda batch: sphere.sample_pair_vmf(batch, kappa)
    sample_uniform_fixed = lambda batch: sphere.uniform(batch, device=device)
    
    # Store all results
    all_results = {
        'experiment_config': {
            'batch_size': batch_size,
            'iterations': iterations,
            'run_name': run_name,
            'number_of_runs': number_of_runs,
            'n_layers': n_layers,
            'act_fct': act_fct,
            'cond_thresh_ratio': cond_thresh_ratio,
            'tau': tau,
            'kappa': kappa,
            'device': str(device)
        },
        'runs': []
    }
    
    print(f"Starting {number_of_runs} runs of MLP SimCLR experiment")
    print(f"Config: batch_size={batch_size}, iterations={iterations}")
    print(f"MLP Config: n_layers={n_layers}, act_fct={act_fct}, cond_thresh_ratio={cond_thresh_ratio}")
    print(f"Results will be saved to: {run_dir}")
    
    for run_idx in range(number_of_runs):
        print(f"\n--- Run {run_idx + 1}/{number_of_runs} ---")
        
        # Initialize fresh encoder and MLP for each run
        f = SphericalEncoder()
        g_mlp = construct_invertible_mlp(
            n=3, 
            n_layers=n_layers, 
            act_fct=act_fct,
            cond_thresh_ratio=cond_thresh_ratio
        ).to(device)
        
        # Setup SimCLR
        simclr_mlp = SimCLR(
            f, g_mlp, sample_pair_fixed, sample_uniform_fixed, tau, device
        )
        
        # Train the model
        print(f"Training SimCLR with invertible MLP for {iterations} iterations...")
        trained_encoder, scores = simclr_mlp.train(batch_size, iterations)
        
        # Save model weights (both encoder and MLP)
        model_path = run_dir / f"model_run_{run_idx + 1}.pth"
        torch.save({
            'encoder_state_dict': trained_encoder.state_dict(),
            'mlp_state_dict': g_mlp.state_dict(),
            'run_idx': run_idx,
            'final_scores': scores,
            'mlp_config': {
                'n_layers': n_layers,
                'act_fct': act_fct,
                'cond_thresh_ratio': cond_thresh_ratio
            }
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
    parser = argparse.ArgumentParser(description='Run MLP SimCLR experiment')
    parser.add_argument('--batch_size', type=int, default=6144, 
                        help='Batch size for training (default: 6144)')
    parser.add_argument('--iterations', type=int, default=100000,
                        help='Number of training iterations (default: 100000)')
    parser.add_argument('--run_name', type=str, required=True,
                        help='Name for the experiment run (required)')
    parser.add_argument('--number_of_runs', type=int, required=True,
                        help='Number of times to repeat the experiment (required)')
    parser.add_argument('--n_layers', type=int, default=3,
                        help='Number of layers in the invertible MLP (default: 3)')
    parser.add_argument('--act_fct', type=str, default='leaky_relu',
                        choices=['relu', 'leaky_relu', 'elu', 'smooth_leaky_relu', 'softplus'],
                        help='Activation function for the MLP (default: leaky_relu)')
    parser.add_argument('--cond_thresh_ratio', type=float, default=0.25,
                        help='Condition threshold ratio for MLP construction (default: 0.25)')
    
    args = parser.parse_args()
    
    run_mlp_experiment(
        batch_size=args.batch_size,
        iterations=args.iterations,
        run_name=args.run_name,
        number_of_runs=args.number_of_runs,
        n_layers=args.n_layers,
        act_fct=args.act_fct,
        cond_thresh_ratio=args.cond_thresh_ratio
    )


if __name__ == "__main__":
    main() 