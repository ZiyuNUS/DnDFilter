import wandb
import os
import numpy as np
import yaml
import tqdm
import itertools
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)

ACTION_STATS = {}
for key in data_config['state_stats']:
    ACTION_STATS[key] = np.array(data_config['state_stats'][key])

def train_dnd(
        model: nn.Module,
        ema_model: EMAModel,
        optimizer: Adam,
        dataloader: DataLoader,
        transform: transforms,
        device: torch.device,
        noise_scheduler: DDPMScheduler,
        epoch: int,
        print_log_freq: int = 100,
        wandb_log_freq: int = 10,
        use_wandb: bool = True,
):
    model.train()
    total_train_loss = 0

    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image,
                ground_truth,
            ) = data

            obs_images = torch.split(obs_image, 3, dim=1)
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)

            ground_truth = ground_truth[:,:,:4]
            ground_truth = normalize_data(ground_truth, ACTION_STATS).to(device)[:, 0, :2]

            def action_reduce(unreduced_loss: torch.Tensor):
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                return unreduced_loss.mean()

            obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images)
            groundtruth = torch.empty_like(obsgoal_cond)
            groundtruth.copy_(ground_truth)

            loss = torch.sqrt(action_reduce(F.mse_loss(obsgoal_cond, groundtruth, reduction="none")))
            total_train_loss = total_train_loss + loss.item() * 128 * 128

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema_model.step(model)

            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
        if use_wandb and wandb_log_freq != 0:
            log_data = {'pos_loss (train)': total_train_loss / (i + 1)}
            wandb.log(log_data, commit=True)

def evaluate_dnd(
        eval_type: str,
        ema_model: EMAModel,
        dataloader: DataLoader,
        transform: transforms,
        device: torch.device,
        noise_scheduler: DDPMScheduler,
        epoch: int,
        print_log_freq: int = 100,
        wandb_log_freq: int = 10,
        eval_fraction: float = 0.25,
        use_wandb: bool = True,
):
    ema_model = ema_model.averaged_model
    ema_model.eval()
    num_batches = len(dataloader)
    total_test_loss = 0

    with tqdm.tqdm(
            itertools.islice(dataloader, num_batches),
            total=num_batches,
            dynamic_ncols=True,
            desc=f"Evaluating {eval_type} for epoch {epoch}",
            leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image,
                ground_truth,
            ) = data

            obs_images = torch.split(obs_image, 3, dim=1)
            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)

            ground_truth = ground_truth[:, :, :4]
            ground_truth = normalize_data(ground_truth, ACTION_STATS).to(device)[:, 0, :2]

            obsgoal_cond = ema_model("vision_encoder", obs_img=batch_obs_images)

            obsgoal_cond = obsgoal_cond * 128
            ground_truth = ground_truth * 128
            goal_mask_loss = nn.functional.mse_loss(obsgoal_cond, ground_truth, reduction="none")
            def action_reduce(unreduced_loss: torch.Tensor):
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                return unreduced_loss.mean()
            goal_mask_loss = torch.sqrt(action_reduce(goal_mask_loss))

            loss_cpu = goal_mask_loss.item()
            tepoch.set_postfix(loss=loss_cpu)

            total_test_loss = total_test_loss + goal_mask_loss
        if use_wandb and wandb_log_freq != 0:
            log_data = {'pos_loss (test)': total_test_loss / (i + 1)}
            wandb.log(log_data, commit=True)

def normalize_data(data, stats):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
    if ndata.is_cuda:
        ndata = ndata.cpu()
    data = (ndata + 1) / 2
    data = data * (128 + 128) - 128
    return data



