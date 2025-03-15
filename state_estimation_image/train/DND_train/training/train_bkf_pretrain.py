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
from DND_train.training import Kalman

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

            ground_truth = ground_truth[:,:,:4]/128

            def action_reduce(unreduced_loss: torch.Tensor):
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                return unreduced_loss.mean()

            obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images)

            z_list = obsgoal_cond[:, :2]

            position_prediction = z_list[1:]

            groundtruth = torch.empty_like(position_prediction)  #
            groundtruth.copy_(ground_truth[1:, 0, :2])

            loss = action_reduce(F.mse_loss(position_prediction, groundtruth, reduction="none"))
            total_train_loss = total_train_loss + math.sqrt(loss.item() * 128 * 128)

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
        wandb_log_freq: int = 10,
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

            ground_truth = ground_truth[:, :, :4] / 128

            obsgoal_cond = ema_model("vision_encoder", obs_img=batch_obs_images)

            z_list = obsgoal_cond[:, :2]

            position_prediction = z_list[1:]

            groundtruth = torch.empty_like(position_prediction)
            groundtruth.copy_(ground_truth[1:, 0, :2])

            def action_reduce(unreduced_loss: torch.Tensor):
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                return unreduced_loss.mean()

            loss = action_reduce(F.mse_loss(position_prediction, groundtruth, reduction="none"))
            loss = loss * 128 * 128
            total_test_loss = total_test_loss + math.sqrt(loss.item())
        if use_wandb and wandb_log_freq != 0:
            log_data = {'pos_loss (test)': total_test_loss / (i + 1)}
            wandb.log(log_data, commit=True)




