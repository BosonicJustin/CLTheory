# Spherical SimCLR Experiments

This directory contains scripts for running spherical toy experiments with SimCLR using different transformations.

## Available Scripts

### 1. Spiral Rotation Experiment (`run_spherical_simclr_experiment.py`)
Runs SimCLR with the SpiralRotation transformation.

### 2. Patches Experiment (`run_patches_simclr_experiment.py`)
Runs SimCLR with the Patches transformation.

### 3. MLP Experiment (`run_mlp_simclr_experiment.py`)
Runs SimCLR with an invertible MLP transformation.

## Usage

### Basic Usage

For **Spiral Rotation** experiment:
```bash
python run_spherical_simclr_experiment.py --run_name "spiral_test" --number_of_runs 3
```

For **Patches** experiment:
```bash
python run_patches_simclr_experiment.py --run_name "patches_test" --number_of_runs 3
```

For **MLP** experiment:
```bash
python run_mlp_simclr_experiment.py --run_name "mlp_test" --number_of_runs 3
```

### Advanced Usage

#### Spiral Rotation with custom parameters:
```bash
python run_spherical_simclr_experiment.py \
    --run_name "spiral_detailed" \
    --number_of_runs 5 \
    --batch_size 4096 \
    --iterations 50000
```

#### Patches with custom parameters:
```bash
python run_patches_simclr_experiment.py \
    --run_name "patches_detailed" \
    --number_of_runs 5 \
    --batch_size 4096 \
    --iterations 50000 \
    --slice_number 8
```

#### MLP with custom parameters:
```bash
python run_mlp_simclr_experiment.py \
    --run_name "mlp_detailed" \
    --number_of_runs 5 \
    --batch_size 4096 \
    --iterations 50000 \
    --n_layers 5 \
    --act_fct elu
```

## Parameters

### Common Parameters
- `--run_name`: **REQUIRED** - Name for the experiment run
- `--number_of_runs`: **REQUIRED** - Number of times to repeat the experiment
- `--batch_size`: Batch size for training (default: 6144)
- `--iterations`: Number of training iterations (default: 100000)

### Patches-specific Parameters
- `--slice_number`: Number of slices for Patches transformation (default: 4)

### MLP-specific Parameters
- `--n_layers`: Number of layers in the invertible MLP (default: 3)
- `--act_fct`: Activation function (default: leaky_relu, choices: relu, leaky_relu, elu, smooth_leaky_relu, softplus)
- `--cond_thresh_ratio`: Condition threshold ratio for MLP construction (default: 0.25)

## Output Structure

Each experiment creates a directory structure like:
```
runs/
└── {run_name}/
    ├── experiment_results.json      # Human-readable results
    ├── experiment_results.pkl       # Python-loadable results
    ├── model_run_1.pth              # Model weights for run 1
    ├── model_run_2.pth              # Model weights for run 2
    └── ...
```

## Results Format

The `experiment_results.json` contains:
- **experiment_config**: All parameters used in the experiment
- **runs**: List of results for each run, including:
  - Final scores (linear, permutation, angle preservation)
  - Path to saved model weights
  - Complete training curves

## Loading Results

To load and analyze results in Python:

```python
import pickle
import torch

# Load results
with open('runs/my_experiment/experiment_results.pkl', 'rb') as f:
    results = pickle.load(f)

# Print summary
config = results['experiment_config']
runs = results['runs']

print(f"Experiment: {config['run_name']}")
print(f"Number of runs: {len(runs)}")

# Load a specific model
model_path = runs[0]['model_path']
checkpoint = torch.load(model_path)
model_state = checkpoint['model_state_dict']

# Analyze scores across runs
final_linear_scores = [run['scores']['linear_scores'][-1] for run in runs]
print(f"Final linear scores: {final_linear_scores}")
```

## Example Experiments

### Quick Test (3 runs, 10k iterations)
```bash
python run_spherical_simclr_experiment.py --run_name "quick_test" --number_of_runs 3 --iterations 10000
```

### Full Experiment (10 runs, 100k iterations)
```bash
python run_spherical_simclr_experiment.py --run_name "full_spiral" --number_of_runs 10 --iterations 100000
python run_patches_simclr_experiment.py --run_name "full_patches" --number_of_runs 10 --iterations 100000
python run_mlp_simclr_experiment.py --run_name "full_mlp" --number_of_runs 10 --iterations 100000
```

### Comparing Different Slice Numbers for Patches
```bash
python run_patches_simclr_experiment.py --run_name "patches_slice_2" --number_of_runs 5 --slice_number 2
python run_patches_simclr_experiment.py --run_name "patches_slice_4" --number_of_runs 5 --slice_number 4
python run_patches_simclr_experiment.py --run_name "patches_slice_8" --number_of_runs 5 --slice_number 8
```

### Comparing Different MLP Architectures
```bash
python run_mlp_simclr_experiment.py --run_name "mlp_layers_2" --number_of_runs 5 --n_layers 2
python run_mlp_simclr_experiment.py --run_name "mlp_layers_3" --number_of_runs 5 --n_layers 3
python run_mlp_simclr_experiment.py --run_name "mlp_layers_5" --number_of_runs 5 --n_layers 5
```

### Comparing Different Activation Functions
```bash
python run_mlp_simclr_experiment.py --run_name "mlp_relu" --number_of_runs 5 --act_fct relu
python run_mlp_simclr_experiment.py --run_name "mlp_leaky_relu" --number_of_runs 5 --act_fct leaky_relu
python run_mlp_simclr_experiment.py --run_name "mlp_elu" --number_of_runs 5 --act_fct elu
``` 