# Proximal Policy Optimization Algorithm

This project reproduces the Proximal Policy Optimization (PPO) algorithm using PyTorch, focusing on environments with discrete and continues action spaces, specifically `CartPole-v1` and `LunarLander-v2` for descrete and using MuJoCo environments for continues action space. The code supports logging to TensorBoard and Weights & Biases (wandb) for experiment tracking and visualization.

## Table of Contents
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [How to Run](#how-to-run)
- [Arguments](#arguments)
- [To Do](#to-do)

## Requirements

- Python 3.8 or higher
- Dependencies are listed in `requirements.txt`.

To install the required packages:
```bash
pip install -r requirements.txt
```

## Quick Start

Run the PPO algorithm on the CartPole environment with default settings:
```bash
python ppo.py --gym-id CartPole-v1 --track --wandb-project-name ppo-reproduction
```

## How to Run

To run PPO on either `CartPole-v1` or `LunarLander-v2`:
```bash
python ppo.py --gym-id [ENVIRONMENT_NAME] --total-timesteps [TIMESTEPS]
```

Example for LunarLander:
```bash
python ppo.py --gym-id LunarLander-v2 --total-timesteps 100000 --track --wandb-project-name ppo-reproduction
```

**Note**: Logging to Weights & Biases requires you to set `--track`, `--wandb-project-name`, and optionally `--wandb-entity` for organizational logging.

### Customizing Hyperparameters

To customize hyperparameters, pass the arguments as shown below:
```bash
python ppo.py --learning-rate 3e-4 --gamma 0.98 --clip-coef 0.2 --gae-lambda 0.95
```

### Using Weights & Biases
To log runs in Weights & Biases, provide:
- `--track`: Enable wandb logging.
- `--wandb-project-name`: Name of your wandb project.
- `--wandb-entity`: (Optional) Wandb team or user entity.

Example:
```bash
python ppo.py --track --wandb-project-name ppo-experiments --wandb-entity your_team
```

## Arguments

Below is a list of commonly used arguments:

| Argument              | Description                                               | Default         |
|-----------------------|-----------------------------------------------------------|-----------------|
| `--gym-id`            | ID of the Gym environment                                 | `CartPole-v1`   |
| `--total-timesteps`   | Total timesteps to run the experiment                     | `25000`         |
| `--learning-rate`     | Optimizer learning rate                                   | `2.5e-4`        |
| `--seed`              | Random seed for reproducibility                           | `1`             |
| `--track`             | Enable wandb logging                                      | `False`         |
| `--wandb-project-name`| Wandb project name                                        | `ppo-test`      |
| `--num-envs`          | Number of parallel environments                           | `4`             |
| `--num-steps`         | Steps per environment rollout                             | `128`           |
| `--clip-coef`         | Clipping coefficient for PPO                              | `0.2`           |
| `--gae`               | Enable Generalized Advantage Estimation (GAE)             | `True`          |
| `--gae-lambda`        | GAE lambda parameter                                      | `0.95`          |

## Logging and Visualization

- **TensorBoard**: Logs are stored in the `runs/` directory. Use `tensorboard --logdir=runs` to visualize training metrics.
- **Weights & Biases**: Track your experiment online with wandb by setting `--track` and specifying the project name and entity.

### Example Logs in Weights & Biases
Running with `--track` enables detailed experiment tracking in wandb, including episode rewards, loss, and training progress visualizations.

## Experiment Logging

To visualize experiment logs, the project is configured to use [Weights & Biases (wandb)](https://wandb.ai/). You can access the project’s experiment dashboard [here on wandb](https://wandb.ai/adhiisetiawan/ppo-algorithm?nw=nwuseradhiisetiawan) to monitor the training progress, compare results, and analyze performance across runs.

## To Do
Here's an updated README with a **To-Do** section, including checkboxes with elaborations:

---

# PPO Algorithm Reproduction

This project reproduces the Proximal Policy Optimization (PPO) algorithm using PyTorch, focusing on environments with discrete action spaces, specifically `CartPole-v1` and `LunarLander-v2`. The code supports logging to both TensorBoard and Weights & Biases (wandb) for experiment tracking and visualization.

## Experiment Logging

To visualize experiment logs, the project is configured to use [Weights & Biases (wandb)](https://wandb.ai/). You can access the project’s experiment dashboard [here on wandb](https://wandb.ai/adhiisetiawan/ppo-algorithm?nw=nwuseradhiisetiawan) to monitor the training progress, compare results, and analyze performance across runs.

## To-Do

- [ ] **Explore MuJoCo Environment**  
- [ ] **Implement MuJoCo in PPO** 
