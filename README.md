# Proximal Policy Optimization Algorithm

This project reproduces the Proximal Policy Optimization (PPO) algorithm using PyTorch, featuring two specialized implementations:

- **`ppo_descrete.py`**: Optimized for **classic control tasks** (CartPole, LunarLander) with MLP networks
- **`ppo_atari.py`**: Optimized for **Atari games** (Breakout, Pong) with CNN networks

The code supports logging to TensorBoard and Weights & Biases (wandb) for experiment tracking and visualization.

## Demo Videos

### Classic Control - LunarLander-v2
![LunarLander Training](assets/lunar.mp4)

### Atari Games - Breakout
![Breakout Training](assets/breakout.mp4)

## Table of Contents
- [Demo Videos](#demo-videos)
- [Requirements](#requirements)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [How to Run](#how-to-run)
- [Script Comparison](#script-comparison)
- [Arguments](#arguments)

## Requirements

- Python 3.8 or higher
- Conda (Miniconda or Anaconda)
- Dependencies are listed in `requirements.txt`.

## Installation

### Option 1: Using Conda (Recommended)

1. **Create a conda environment with Python 3.8:**
   ```bash
   conda create -n ppo-algorithm python=3.8
   conda activate ppo-algorithm
   ```

2. **Install box2d-py from conda-forge (avoids compilation issues):**
   ```bash
   conda install -c conda-forge box2d-py
   ```

3. **Install remaining dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

### Option 2: Using pip only

If you prefer to use pip only, ensure you have the necessary build tools installed:

**For Ubuntu/Debian:**
```bash
sudo apt update
sudo apt install -y build-essential python3-dev swig
pip install -r requirements.txt
```

**For macOS:**
```bash
brew install swig
pip install -r requirements.txt
```

**Note:** The conda approach is recommended as it avoids compilation issues with `box2d-py` and provides pre-compiled binaries.

## Quick Start

1. **Activate the conda environment:**
   ```bash
   conda activate ppo-algorithm
   ```

2. **Choose your implementation:**

   **For Classic Control Tasks (CartPole, LunarLander):**
   ```bash
   python ppo_descrete.py --gym-id CartPole-v1 --track --wandb-project-name ppo-reproduction
   ```

   **For Atari Games (Breakout, Pong):**
   ```bash
   python ppo_atari.py --gym-id BreakoutNoFrameskip-v4 --track --wandb-project-name ppo-reproduction
   ```

## How to Run

1. **Activate the conda environment:**
   ```bash
   conda activate ppo-algorithm
   ```

2. **Choose the appropriate script and environment:**

   ### Classic Control Tasks (ppo_descrete.py)
   
   **CartPole-v1:**
   ```bash
   python ppo_descrete.py --gym-id CartPole-v1 --total-timesteps 25000 --track --wandb-project-name ppo-reproduction
   ```

   **LunarLander-v2:**
   ```bash
   python ppo_descrete.py --gym-id LunarLander-v2 --total-timesteps 100000 --track --wandb-project-name ppo-reproduction
   ```

   ### Atari Games (ppo_atari.py)
   
   **Breakout:**
   ```bash
   python ppo_atari.py --gym-id BreakoutNoFrameskip-v4 --total-timesteps 10000000 --track --wandb-project-name ppo-reproduction
   ```

   **Pong:**
   ```bash
   python ppo_atari.py --gym-id PongNoFrameskip-v4 --total-timesteps 10000000 --track --wandb-project-name ppo-reproduction
   ```

**Note**: Logging to Weights & Biases requires you to set `--track`, `--wandb-project-name`, and optionally `--wandb-entity` for organizational logging.

## Script Comparison

| Feature | `ppo_descrete.py` | `ppo_atari.py` |
|---------|-------------------|----------------|
| **Target Environments** | Classic Control (CartPole, LunarLander) | Atari Games (Breakout, Pong) |
| **Neural Network** | MLP (Multi-Layer Perceptron) | CNN (Convolutional Neural Network) |
| **Input Processing** | Raw observations | Preprocessed frames (84x84, grayscale, stacked) |
| **Default Timesteps** | 25,000 | 10,000,000 |
| **Default Environments** | 4 | 8 |
| **Default Clip Coef** | 0.2 | 0.1 |
| **GPU Utilization** | Low (3-10%) | High (50-90%) |
| **Training Time** | Fast (minutes) | Slow (hours) |

### When to Use Which Script

**Use `ppo_descrete.py` when:**
- Working with classic control tasks
- Need fast experimentation
- Limited computational resources
- Learning PPO fundamentals

**Use `ppo_atari.py` when:**
- Working with visual environments
- Need high GPU utilization
- Researching computer vision + RL
- Have sufficient computational resources

### Customizing Hyperparameters

**For Classic Control (ppo_descrete.py):**
```bash
python ppo_descrete.py --learning-rate 3e-4 --gamma 0.98 --clip-coef 0.2 --gae-lambda 0.95 --num-envs 8
```

**For Atari Games (ppo_atari.py):**
```bash
python ppo_atari.py --learning-rate 2.5e-4 --gamma 0.99 --clip-coef 0.1 --gae-lambda 0.95 --num-envs 16
```

### Using Weights & Biases
To log runs in Weights & Biases, provide:
- `--track`: Enable wandb logging.
- `--wandb-project-name`: Name of your wandb project.
- `--wandb-entity`: (Optional) Wandb team or user entity.

**Examples:**
```bash
# For classic control tasks
python ppo_descrete.py --track --wandb-project-name ppo-experiments --wandb-entity your_team

# For Atari games
python ppo_atari.py --track --wandb-project-name ppo-experiments --wandb-entity your_team
```

## Arguments

Below is a list of commonly used arguments for both scripts:

| Argument              | Description                                               | ppo_descrete.py | ppo_atari.py    |
|-----------------------|-----------------------------------------------------------|-----------------|-----------------|
| `--gym-id`            | ID of the Gym environment                                 | `CartPole-v1`   | `BreakoutNoFrameskip-v4` |
| `--total-timesteps`   | Total timesteps to run the experiment                     | `25000`         | `10000000`      |
| `--learning-rate`     | Optimizer learning rate                                   | `2.5e-4`        | `2.5e-4`        |
| `--seed`              | Random seed for reproducibility                           | `1`             | `1`             |
| `--track`             | Enable wandb logging                                      | `False`         | `False`         |
| `--wandb-project-name`| Wandb project name                                        | `ppo-test-new`  | `ppo-test-new`  |
| `--num-envs`          | Number of parallel environments                           | `4`             | `8`             |
| `--num-steps`         | Steps per environment rollout                             | `128`           | `128`           |
| `--clip-coef`         | Clipping coefficient for PPO                              | `0.2`           | `0.1`           |
| `--gae`               | Enable Generalized Advantage Estimation (GAE)             | `True`          | `True`          |
| `--gae-lambda`        | GAE lambda parameter                                      | `0.95`          | `0.95`          |

## Logging and Visualization

- **TensorBoard**: Logs are stored in the `runs/` directory. Use `tensorboard --logdir=runs` to visualize training metrics.
- **Weights & Biases**: Track your experiment online with wandb by setting `--track` and specifying the project name and entity.

### Example Logs in Weights & Biases
Running with `--track` enables detailed experiment tracking in wandb, including episode rewards, loss, and training progress visualizations.

## Experiment Logging

To visualize experiment logs, the project is configured to use [Weights & Biases (wandb)](https://wandb.ai/). You can access the project’s experiment dashboard [here on wandb](https://wandb.ai/adhiisetiawan/ppo-algorithm?nw=nwuseradhiisetiawan) to monitor the training progress, compare results, and analyze performance across runs.


