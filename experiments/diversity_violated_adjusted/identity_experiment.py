import sys
import os
sys.path.append(os.path.join(os.path.dirname(os.getcwd()), '..'))

import torch
from torch import nn
from torch import functional as F
from encoders import SphericalEncoder, LinearEncoder, InverseSpiralEncoder, InversePatchesEncoder
from data.generation import InjectiveLinearDecoder, SpiralRotation, Patches
from spaces import NSphereSpace
from visualization_utils.spheres import visualize_spheres_side_by_side, scatter3d_sphere
import matplotlib.pyplot as plt
from evals.disentanglement import linear_disentanglement, permutation_disentanglement
from evals.distance_preservation import calculate_angle_preservation_error
import numpy as np

# Setup
full_sphere = NSphereSpace(3)
sub_sphere = NSphereSpace(2)

d_fix = 1
d_input = 3  # Input dimension (from sphere)
d_intermediate = 3  # Intermediate dimension (after g, before f) - Patches outputs 3D
d_output = 3  # Final output dimension (after f)
tau = 0.3
kappa = 1 / tau
batch_size = 2000
neg_samples = 2000
iterations = 5000  # Reduced for faster experimentation
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Enable TensorFloat32 for faster matrix operations on modern GPUs
if device == 'cuda':
    torch.set_float32_matmul_precision('high')

# Initialize SpiralRotation ONCE - reused for all experiments (3D → 3D)
g = SpiralRotation(device=device)
g = torch.compile(g)  # Compile for 20-40% speedup since g is never updated
# Warm up compilation
_ = g(torch.randn(100, 3, device=device))
print(f"🔧 Initialized and compiled shared SpiralRotation: {d_input}D → {d_intermediate}D")

def sample_negative_samples(Z, M, constraint_ratio):
    """
    Hybrid sampling: mix constrained and unconstrained negatives
    """
    N = Z.shape[0]
    
    # Split negatives based on constraint ratio
    M_constrained = int(constraint_ratio * M)
    M_unconstrained = M - M_constrained
    
    negatives_list = []
    
    # CONSTRAINED NEGATIVES
    if M_constrained > 0:
        z_fixed = Z[:, :d_fix].to(device)
        radii = torch.sqrt(1 - (z_fixed ** 2).sum(dim=1)).to(device)
        neg_constrained = torch.randn(N, M_constrained, d_input - d_fix, device=device)
        neg_constrained = neg_constrained / neg_constrained.norm(dim=2, keepdim=True)
        neg_constrained = neg_constrained * radii.view(-1, 1, 1)
        neg_constrained = torch.cat((z_fixed.unsqueeze(1).expand(-1, M_constrained, -1), neg_constrained), dim=2)
        negatives_list.append(neg_constrained)
    
    # UNCONSTRAINED NEGATIVES (full sphere)
    if M_unconstrained > 0:
        neg_unconstrained = full_sphere.uniform(N * M_unconstrained).to(device).reshape(N, M_unconstrained, d_input)
        negatives_list.append(neg_unconstrained)
    
    # COMBINE
    if len(negatives_list) > 1:
        return torch.cat(negatives_list, dim=1)
    elif len(negatives_list) == 1:
        return negatives_list[0]
    else:
        # Fallback - shouldn't happen
        return torch.randn(N, M, d_input, device=device)

def sample_conditional_with_dims_fixed(z, batch, u_dim):
    u = z[:, :u_dim].to(device)
    v = z[:, u_dim:].to(device)
    v_norm = torch.nn.functional.normalize(v, dim=-1, p=2).to(device)
    aug_samples_v = sub_sphere.von_mises_fisher(v_norm, kappa, batch).to(device) * torch.norm(v, p=2, dim=-1, keepdim=True).to(device)
    return torch.cat((u, aug_samples_v), dim=-1)

def sample_pair_with_fixed_dimension(batch, u_dim):
    z = full_sphere.uniform(batch).to(device)
    return z, sample_conditional_with_dims_fixed(z, batch, u_dim)

class InfoNceLossAdjusted(nn.Module):
    def __init__(self, temperature=0.1):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, anchors, positives, negatives):
        N, d = anchors.shape
        N_pos, d_pos = positives.shape
        N_neg, _, d_neg = negatives.shape
        
        assert N == N_pos == N_neg, f"Batch sizes must match: {N}, {N_pos}, {N_neg}"
        assert d == d_pos == d_neg, f"Embedding dimensions must match: {d}, {d_pos}, {d_neg}"
        
        pos_similarities = torch.sum(anchors * positives, dim=1) / self.temperature
        neg_cosines = (anchors.unsqueeze(1) * negatives).sum(dim=-1) / self.temperature
        neg_similarities = torch.log(torch.exp(neg_cosines).sum(dim=-1) + torch.exp(pos_similarities))

        return neg_similarities.mean() - pos_similarities.mean()

def profile_bottlenecks():
    """Quick profiling to identify bottlenecks - run once before main experiment"""
    print("🔍 Profiling bottlenecks...")
    
    # Use same setup as training
    f_test = SphericalEncoder(input_dim=d_intermediate, hidden_dims=[128, 256, 256, 256, 128], output_dim=d_output).to(device)
    f_test = torch.compile(f_test)
    h_test = lambda latent: f_test(g(latent))
    
    # Warm up compilation
    _ = h_test(torch.randn(100, 3, device=device))
    
    # Profile 10 iterations
    times = {'neg_sampling': [], 'forward': [], 'backward': []}
    
    for i in range(10):
        # Negative sampling
        start = time.time()
        z = full_sphere.uniform(batch_size).to(device)
        z_aug = sample_conditional_with_dims_fixed(z, batch_size, d_fix)
        negs = sample_negative_samples(z, neg_samples, 0.5)  # Use middle constraint ratio
        times['neg_sampling'].append(time.time() - start)
        
        # Forward pass
        start = time.time()
        z_enc = h_test(z)
        z_enc_sim = h_test(z_aug)
        negs_flat = negs.contiguous().view(-1, d_input).to(device)
        z_enc_neg = h_test(negs_flat).view(batch_size, neg_samples, d_output)
        times['forward'].append(time.time() - start)
        
        # Backward pass
        start = time.time()
        objective = InfoNceLossAdjusted(tau)
        loss = objective(z_enc, z_enc_sim, z_enc_neg)
        loss.backward()
        times['backward'].append(time.time() - start)
    
    # Report averages
    for phase, timings in times.items():
        avg_time = sum(timings) / len(timings)
        print(f"  {phase}: {avg_time:.3f}s avg ({avg_time/sum(sum(times.values(), []))*100:.1f}%)")
    
    total_time = sum(sum(times.values(), []))
    print(f"  Total per iteration: {total_time/10:.3f}s")
    print(f"  Estimated time for 5000 iterations: {total_time/10 * 5000/60:.1f} minutes")
    print()

def train_model(constraint_ratio, run_id=0, verbose=False):
    """Train model with given constraint ratio and return final metrics"""
    
    # Initialize fresh InversePatchesEncoder for each experiment (f is reinitialized)
    # f = InversePatchesEncoder(input_dim=d_intermediate, latent_dim=d_output, slice_number=4).to(device)
    f = SphericalEncoder(input_dim=d_intermediate, hidden_dims=[128, 256, 256, 256, 128], output_dim=d_output).to(device)
    f = torch.compile(f)  # Compile f for 20-40% speedup during training
    objective = InfoNceLossAdjusted(tau)
    # Composition: sphere data -> g (Patches 3D→3D) -> f (InversePatchesEncoder 3D→3D) 
    h = lambda latent: f(g(latent))
    optimizer = torch.optim.Adam(f.parameters(), lr=0.001)  # Only optimize f, g is fixed
    
    sample_pair_fixed = lambda batch: sample_pair_with_fixed_dimension(batch, d_fix)
    
    # Training loop with timing
    segment_times = []  # Track time for each 500-iteration segment
    segment_start = time.time()
    
    for i in range(iterations):
        optimizer.zero_grad()
        z, z_aug = sample_pair_fixed(batch_size)
        negs = sample_negative_samples(z, neg_samples, constraint_ratio)

        z_enc = h(z.to(device))
        z_enc_sim = h(z_aug.to(device))
        # More efficient negative processing - use view instead of reshape
        negs_flat = negs.contiguous().view(-1, d_input).to(device)
        z_enc_neg = h(negs_flat).view(batch_size, neg_samples, d_output)

        loss = objective(z_enc, z_enc_sim, z_enc_neg)
        loss.backward()
        optimizer.step()

        if i % 500 == 0:  # Reduced print frequency for speed
            # Calculate segment timing
            segment_time = time.time() - segment_start
            segment_times.append(segment_time)
            
            # Calculate moving average and estimate remaining time
            if len(segment_times) >= 3:
                avg_segment_time = sum(segment_times[-3:]) / 3  # Last 3 segments
            else:
                avg_segment_time = sum(segment_times) / len(segment_times)  # All available segments
            
            remaining_segments = (iterations - i) // 500
            estimated_remaining = avg_segment_time * remaining_segments / 60  # minutes
            
            print(f"    Run {run_id+1}, Iteration {i}: Loss: {loss.item():.4f} | Segment: {segment_time:.1f}s | Avg: {avg_segment_time:.1f}s | Est. remaining: {estimated_remaining:.1f}min")
            
            # Reset for next segment
            if i < iterations - 500:  # Don't reset on last segment
                segment_start = time.time()
    
    # Final evaluation
    f.eval()
    test_batch_size = 6144
    with torch.no_grad():
        z_test, z_aug_test = sample_pair_fixed(test_batch_size)
        z_enc_test = h(z_test.to(device))
        
        # Compute final metrics
        lin_dis, _ = linear_disentanglement(z_test.cpu(), z_enc_test.cpu())
        lin_score, _ = lin_dis
        
        perm_dis, _ = permutation_disentanglement(z_test.cpu(), z_enc_test.cpu(), mode="pearson", solver="munkres")
        perm_score, _ = perm_dis
        
        angle_error = calculate_angle_preservation_error(z_test.cpu(), z_enc_test.cpu())
        
        # Final loss
        negs_test = sample_negative_samples(z_test, neg_samples, constraint_ratio)
        z_enc_sim_test = h(z_aug_test.to(device))
        z_enc_neg_test = h(negs_test.contiguous().view(-1, d_input).to(device)).view(test_batch_size, neg_samples, d_output)
        final_loss = objective(z_enc_test, z_enc_sim_test, z_enc_neg_test)
    
    return {
        'constraint_ratio': constraint_ratio,
        'linear_score': lin_score,
        'perm_score': perm_score,
        'angle_error': angle_error,
        'final_loss': final_loss.item()
    }

# 🧪 EXPERIMENT: Run training for different constraint ratios
import time
import json
from datetime import datetime

# Test different constraint ratios
constraint_ratios = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
n_runs = 5

# Create results directory and filename with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_dir = "experiment_results"
os.makedirs(results_dir, exist_ok=True)
results_file = os.path.join(results_dir, f"constraint_ratio_experiment_{timestamp}.json")
summary_file = os.path.join(results_dir, f"constraint_ratio_summary_{timestamp}.txt")

# Run quick profiling first
profile_bottlenecks()

print("🚀 Starting Constraint Ratio Experiment (5 runs per ratio)")
print(f"Device: {device}, Batch size: {batch_size}, Iterations: {iterations}")
print(f"Temperature: {tau}, Fixed dimensions: {d_fix}, Negative samples: {neg_samples}")
print(f"Architecture: 3D sphere → Patches({d_input}→{d_intermediate}) → InversePatchesEncoder({d_intermediate}→{d_output})")
print(f"Training setup: f (InversePatchesEncoder) reinitialized each run, g (Patches) shared across all {len(constraint_ratios) * n_runs} experiments")
print(f"Estimated time: ~{len(constraint_ratios) * n_runs * 2:.0f} minutes")
print(f"📁 Results will be saved to: {results_file}")

# Check for existing incomplete experiments (optional resume functionality)
existing_files = [f for f in os.listdir(results_dir) if f.startswith("constraint_ratio_experiment_") and f.endswith(".json")]
if existing_files:
    print(f"📋 Found {len(existing_files)} previous experiment file(s) in {results_dir}")
    print("   (Starting fresh experiment - previous results preserved)")

print("=" * 80)

all_results = []

# Save experiment metadata
experiment_metadata = {
    "timestamp": timestamp,
    "device": str(device),
    "batch_size": batch_size,
    "iterations": iterations,
    "temperature": tau,
    "d_fix": d_fix,
    "d_input": d_input,
    "d_intermediate": d_intermediate,
    "d_output": d_output,
    "neg_samples": neg_samples,
    "constraint_ratios": constraint_ratios,
    "n_runs": n_runs,
    "architecture": f"3D sphere → Patches({d_input}→{d_intermediate}) → InversePatchesEncoder({d_intermediate}→{d_output})",
    "training_setup": "f (InversePatchesEncoder) reinitialized each run, g (Patches) shared across all experiments",
    "results": []
}

total_start_time = time.time()

for i, ratio in enumerate(constraint_ratios):
    ratio_start_time = time.time()
    print(f"\n[{i+1}/{len(constraint_ratios)}] Training with constraint ratio: {ratio:.1f}")
    print(f"  ({int(ratio*100)}% constrained, {int((1-ratio)*100)}% unconstrained negatives)")
    
    # Run multiple times for this ratio
    ratio_results = []
    for run in range(n_runs):
        run_start_time = time.time()
        print(f"  🔄 Run {run+1}/{n_runs}:")
        
        result = train_model(ratio, run_id=run, verbose=True)
        ratio_results.append(result)
        
        run_time = time.time() - run_start_time
        print(f"      ✅ Final: Linear {result['linear_score']:.3f}, Perm {result['perm_score']:.3f}, Angle {result['angle_error']:.3f}, Loss {result['final_loss']:.3f} ({run_time:.1f}s)")
        
        # 💾 SAVE INDIVIDUAL RUN RESULT IMMEDIATELY
        experiment_metadata["results"].append({
            "constraint_ratio": ratio,
            "run_id": run,
            "result": result,
            "run_time_seconds": run_time,
            "timestamp": datetime.now().isoformat()
        })
        
        # Save to file after each run
        with open(results_file, 'w') as f:
            json.dump(experiment_metadata, f, indent=2)
        print(f"      💾 Saved run {run+1} to {results_file}")
    
    # Calculate statistics for this ratio
    linear_scores = [r['linear_score'] for r in ratio_results]
    perm_scores = [r['perm_score'] for r in ratio_results]
    angle_errors = [r['angle_error'] for r in ratio_results]
    final_losses = [r['final_loss'] for r in ratio_results]
    
    stats = {
        'constraint_ratio': ratio,
        'linear_mean': np.mean(linear_scores),
        'linear_std': np.std(linear_scores),
        'perm_mean': np.mean(perm_scores),
        'perm_std': np.std(perm_scores),
        'angle_mean': np.mean(angle_errors),
        'angle_std': np.std(angle_errors),
        'loss_mean': np.mean(final_losses),
        'loss_std': np.std(final_losses),
        'raw_results': ratio_results
    }
    all_results.append(stats)
    
    ratio_time = time.time() - ratio_start_time
    remaining_ratios = len(constraint_ratios) - (i + 1)
    est_remaining = remaining_ratios * ratio_time / 60
    
    print(f"  📊 Summary for ratio {ratio:.1f} ({ratio_time:.1f}s):")
    print(f"     Linear: {stats['linear_mean']:.3f} ± {stats['linear_std']:.3f}")
    print(f"     Perm: {stats['perm_mean']:.3f} ± {stats['perm_std']:.3f}")
    print(f"     Angle: {stats['angle_mean']:.3f} ± {stats['angle_std']:.3f}")
    print(f"     Loss: {stats['loss_mean']:.3f} ± {stats['loss_std']:.3f}")
    if remaining_ratios > 0:
        print(f"     ⏱️  Est. remaining: {est_remaining:.1f} minutes")
    
    # 💾 SAVE SUMMARY AFTER EACH CONSTRAINT RATIO
    with open(summary_file, 'w') as f:
        f.write(f"Constraint Ratio Experiment Results\n")
        f.write(f"Generated: {datetime.now().isoformat()}\n")
        f.write(f"Experiment timestamp: {timestamp}\n\n")
        
        f.write("COMPLETED RATIOS:\n")
        f.write("-" * 50 + "\n")
        for completed_stats in all_results:
            r = completed_stats['constraint_ratio']
            f.write(f"Ratio {r:.1f}: ")
            f.write(f"Linear {completed_stats['linear_mean']:.3f}±{completed_stats['linear_std']:.3f}, ")
            f.write(f"Perm {completed_stats['perm_mean']:.3f}±{completed_stats['perm_std']:.3f}\n")
        
        if remaining_ratios > 0:
            f.write(f"\nREMAINING: {remaining_ratios} ratios (~{est_remaining:.1f} minutes)\n")
        else:
            f.write(f"\n✅ EXPERIMENT COMPLETE!\n")

total_time = time.time() - total_start_time
print(f"\n⏱️  Total experiment time: {total_time/60:.1f} minutes")

# 📊 RESULTS SUMMARY
print("\n" + "=" * 90)
print("📊 EXPERIMENT RESULTS SUMMARY (Mean ± Std over 5 runs)")
print("=" * 90)
print(f"{'Ratio':<8} {'Linear':<15} {'Perm':<15} {'Angle':<15} {'Loss':<15}")
print("-" * 90)

# Find best performing ratios
best_linear = max(all_results, key=lambda x: x['linear_mean'])
best_perm = max(all_results, key=lambda x: x['perm_mean'])
best_angle = min(all_results, key=lambda x: x['angle_mean'])
best_loss = min(all_results, key=lambda x: x['loss_mean'])

for result in all_results:
    ratio = result['constraint_ratio']
    
    # Format: mean ± std
    linear_str = f"{result['linear_mean']:.3f}±{result['linear_std']:.3f}"
    perm_str = f"{result['perm_mean']:.3f}±{result['perm_std']:.3f}"
    angle_str = f"{result['angle_mean']:.3f}±{result['angle_std']:.3f}"
    loss_str = f"{result['loss_mean']:.3f}±{result['loss_std']:.3f}"
    
    # Mark best results
    markers = ""
    if result == best_linear: markers += "⭐"
    if result == best_perm: markers += "🏆"
    if result == best_angle: markers += "🎯"
    if result == best_loss: markers += "🔥"
    
    print(f"{ratio:<8.1f} {linear_str:<15} {perm_str:<15} {angle_str:<15} {loss_str:<15} {markers}")

print("\n🏆 BEST PERFORMERS (by mean):")
print(f"Best Linear Score: {best_linear['linear_mean']:.4f} ± {best_linear['linear_std']:.4f} at ratio {best_linear['constraint_ratio']:.1f}")
print(f"Best Perm Score: {best_perm['perm_mean']:.4f} ± {best_perm['perm_std']:.4f} at ratio {best_perm['constraint_ratio']:.1f}")
print(f"Best Angle Error: {best_angle['angle_mean']:.4f} ± {best_angle['angle_std']:.4f} at ratio {best_angle['constraint_ratio']:.1f}")
print(f"Best Loss: {best_loss['loss_mean']:.4f} ± {best_loss['loss_std']:.4f} at ratio {best_loss['constraint_ratio']:.1f}")

# 📈 PLOT RESULTS WITH ERROR BARS
fig, axes = plt.subplots(2, 2, figsize=(15, 12))
ratios = [r['constraint_ratio'] for r in all_results]

# Extract means and stds for plotting
linear_means = [r['linear_mean'] for r in all_results]
linear_stds = [r['linear_std'] for r in all_results]
perm_means = [r['perm_mean'] for r in all_results]
perm_stds = [r['perm_std'] for r in all_results]
angle_means = [r['angle_mean'] for r in all_results]
angle_stds = [r['angle_std'] for r in all_results]
loss_means = [r['loss_mean'] for r in all_results]
loss_stds = [r['loss_std'] for r in all_results]

# Linear scores
axes[0, 0].errorbar(ratios, linear_means, yerr=linear_stds, fmt='bo-', linewidth=2, markersize=8, capsize=5)
axes[0, 0].set_xlabel('Constraint Ratio')
axes[0, 0].set_ylabel('Linear Disentanglement Score')
axes[0, 0].set_title('Linear Disentanglement vs Constraint Ratio')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].axvline(x=best_linear['constraint_ratio'], color='red', linestyle='--', alpha=0.7, 
                   label=f'Best: {best_linear["constraint_ratio"]:.1f}')
axes[0, 0].legend()

# Permutation scores
axes[0, 1].errorbar(ratios, perm_means, yerr=perm_stds, fmt='go-', linewidth=2, markersize=8, capsize=5)
axes[0, 1].set_xlabel('Constraint Ratio')
axes[0, 1].set_ylabel('Permutation Disentanglement Score')
axes[0, 1].set_title('Permutation Disentanglement vs Constraint Ratio')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].axvline(x=best_perm['constraint_ratio'], color='red', linestyle='--', alpha=0.7, 
                   label=f'Best: {best_perm["constraint_ratio"]:.1f}')
axes[0, 1].legend()

# Angle errors
axes[1, 0].errorbar(ratios, angle_means, yerr=angle_stds, fmt='ro-', linewidth=2, markersize=8, capsize=5)
axes[1, 0].set_xlabel('Constraint Ratio')
axes[1, 0].set_ylabel('Angle Preservation Error')
axes[1, 0].set_title('Angle Error vs Constraint Ratio')
axes[1, 0].grid(True, alpha=0.3)
axes[1, 0].axvline(x=best_angle['constraint_ratio'], color='blue', linestyle='--', alpha=0.7, 
                   label=f'Best: {best_angle["constraint_ratio"]:.1f}')
axes[1, 0].legend()

# Final losses
axes[1, 1].errorbar(ratios, loss_means, yerr=loss_stds, fmt='mo-', linewidth=2, markersize=8, capsize=5)
axes[1, 1].set_xlabel('Constraint Ratio')
axes[1, 1].set_ylabel('Final Loss')
axes[1, 1].set_title('Final Loss vs Constraint Ratio')
axes[1, 1].grid(True, alpha=0.3)
axes[1, 1].axvline(x=best_loss['constraint_ratio'], color='orange', linestyle='--', alpha=0.7, 
                   label=f'Best: {best_loss["constraint_ratio"]:.1f}')
axes[1, 1].legend()

plt.tight_layout()
plot_filename = os.path.join(results_dir, f'constraint_ratio_experiment_plots_{timestamp}.png')
plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
plt.show()

# 💾 SAVE FINAL COMPLETE RESULTS
final_results = {
    "experiment_metadata": experiment_metadata,
    "summary_statistics": all_results,
    "best_performers": {
        "best_linear": {"ratio": best_linear['constraint_ratio'], "score": best_linear['linear_mean'], "std": best_linear['linear_std']},
        "best_perm": {"ratio": best_perm['constraint_ratio'], "score": best_perm['perm_mean'], "std": best_perm['perm_std']},
        "best_angle": {"ratio": best_angle['constraint_ratio'], "score": best_angle['angle_mean'], "std": best_angle['angle_std']},
        "best_loss": {"ratio": best_loss['constraint_ratio'], "score": best_loss['loss_mean'], "std": best_loss['loss_std']}
    },
    "total_experiment_time_minutes": total_time/60
}

final_results_file = os.path.join(results_dir, f"constraint_ratio_final_results_{timestamp}.json")
with open(final_results_file, 'w') as f:
    json.dump(final_results, f, indent=2)

print(f"\n📁 Final results saved to:")
print(f"  - Raw data: {results_file}")
print(f"  - Summary: {final_results_file}")
print(f"  - Plots: {plot_filename}")
print(f"  - Live summary: {summary_file}")
print("=" * 90)