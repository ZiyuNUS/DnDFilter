# DnDFilter
This repository is the official implementation of the paper 
["DnD Filter: Differentiable State Estimation for Dynamic Systems using Diffusion 
Models"](https://arxiv.org/abs/2503.01274), which has been submitted 
to 2025 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS 2025)

DnD Filter is a differentiable filter that utilizes diffusion models for state 
estimation of dynamic systems. Unlike conventional differentiable filters, which 
often impose restrictive assumptions on process noise (e.g., Gaussianity), DnD Filter 
enables a nonlinear state update without such constraints by conditioning a diffusion 
model on both the predicted state and observational data, capitalizing on its ability to
approximate complex distributions. To the best of our knowledge, DnD Filter represents
the first successful attempt to leverage diffusion models for state estimation, offering
a flexible and powerful framework for nonlinear estimation under noisy measurements.

# Overview
This repository provides training and validating code for DnD Filter and trained model 
checkpoints.

- `./train/train.py`: training script to train DnD Filters and baselines.
- `./train/test_*.py`: validating script for DnD Filters and baselines.
- `./train/config/`: training configurations for DnD Filter and baselines.
- `./train/dataset/`: dataset used for training and validating.
- `./train/logs/`: the pretrained model checkpoints for DnD Filter and baselines.
- `./train/DND_train/`: contains model files for DND Filter and baselines.

# Getting Started
Run the commands below inside the topmost directory:
1. Set up the conda environment:
    ```bash
    conda env create -f train/train_environment.yml
    ```
2. Source the conda environment:
    ```
    conda activate DnD_Filter
    ```
3. Install the `diffusion_policy` package from this [repo](https://github.com/real-stanford/diffusion_policy):
    ```bash
    git clone git@github.com:real-stanford/diffusion_policy.git
    pip install -e diffusion_policy/
    ```
# More details are coming soon.
