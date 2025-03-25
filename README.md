# DnDFilter
This repository is the official implementation of the paper 
["DnD Filter: Differentiable State Estimation for Dynamic Systems using Diffusion 
Models"](https://arxiv.org/abs/2503.01274).
![](./image/pipeline.png)

DnD Filter is a differentiable filter that utilizes diffusion models for state 
estimation of dynamic systems. Unlike conventional differentiable filters, which 
often impose restrictive assumptions on process noise (e.g., Gaussianity), DnD Filter 
enables a nonlinear state update without such constraints by conditioning a diffusion 
model on both the predicted state and observational data, capitalizing on its ability to
approximate complex distributions. To the best of our knowledge, DnD Filter represents
the first successful attempt to leverage diffusion models for state estimation, offering
a flexible and powerful framework for nonlinear estimation under noisy measurements.

# Overview
This repository contains the training and validation code for DnD Filter, 
as well as pre-trained model checkpoints. 
Below is an outline of the key scripts and directories:

- `./train/train.py`: Training script for DnD Filter and baseline methods.
- `./train/test_*.py`: Validation scripts for DnD Filter and baseline methods.
- `./train/config/`: Configuration files for training DnD Filter and baseline methods.
- `./train/dataset/`: Datasets used for both training and validation.
- `./train/logs/`: Directory containing trained model checkpoints for DnD Filter and baseline methods.
- `./train/DND_train/`: Model implementation files for DnD Filter and its baselines.

# Getting Started
Run the commands below inside the topmost directory:
1. Set up the conda environment:
    ```bash
    conda env create -f train_environment.yml
    ```
2. Source the conda environment:
    ```
    conda activate DnD_Filter
    ```
3. Install the `diffusion_policy` package from this [repo](https://github.com/real-stanford/diffusion_policy) into the `state_estimation_*` folders:
    ```bash
    git clone git@github.com:real-stanford/diffusion_policy.git
    pip install -e diffusion_policy/
    ```
# Training and Validating
For training, modify the configuration file in `./train/train.py` to match the 
desired training objective or model, then simply run `train.py` to start the training process.

For validating, run the test scripts found in the `./train/test_*` or `./train/test` files.

To train from a existing checkpoints, Add 
```bash
    load_run: <project_name>/<log_run_name>
```
to `.yaml` config file in `./train/config/`. The `*.pth` of the file you are 
loading to be saved in this file structure and renamed to “latest”: 
```bash
   state_estimation_*/train/logs/<project_name>/<log_run_name>/latest.pth. 
```

# Dataset
1. The simulated disk tracking dataset can be generated using codes in 
[repo](https://github.com/tiboat/BackpropKF_Reproduction). 
2. KITTI Visual Odometry Dataset 
(https://www.cvlibs.net/datasets/kitti/eval_odometry.php)

The dataset should be processed into following structure:
```
├── <dataset_name>
│   ├── <name_of_traj1>
│   │   ├── 0
│   │   ├── 1
│   │   ├── ...
│   │   ├── T-1
│   │   ├── traj_data.pkl
│   │   └── traj_data.txt
│   ├── <name_of_traj2>
│   │   ├── 0
│   │   ├── 1
│   │   ├── ...
│   │   ├── T-1
│   │   ├── traj_data.pkl
│   │   └── traj_data.txt
│   ...
└── └── <name_of_trajN>
    	├── 0
    	├── 1
    	├── ...
        ├── T-1
        ├── traj_data.pkl
        └── traj_data.txt
```  
Files `0` to `T-1` contain the high-dimensional observations (e.g.,images), while `traj_data.pkl`
and `traj_data.txt` store metadata and additional information related to the sequence.

The processed dataset used in our experiments is available at
[DataLink](https://huggingface.co/datasets/ZIYUNUS/DnD_Filter). After downloading, place the dataset into the
`./train/dataset` directory.

## Citation

```
@article{Wan2025DnD,
  title={DnD Filter: Differentiable State Estimation for Dynamic Systems using Diffusion Models},
  author={Ziyu Wan, Lin Zhao},
  journal={arXiv preprint arXiv:2503.01274},
  year={2025}
}
```
