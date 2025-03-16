import wandb
import os
import numpy as np
import yaml
import tqdm
import itertools
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel
import torch
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

def _compute_losses_dnd(
        gc_actions,
        ground_truth,
):

    def action_reduce(unreduced_loss: torch.Tensor):
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        return unreduced_loss.mean()

    gc_actions = gc_actions.cpu() * 128
    ground_truth = ground_truth.cpu() * 128

    gc_action_loss = action_reduce(F.mse_loss(gc_actions, ground_truth, reduction="none"))

    return gc_action_loss

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
            groundtruth = torch.empty_like(obsgoal_cond)  # 创建一个与 obsgoal_cond 具有相同属性的新张量
            groundtruth.copy_(ground_truth)

            loss = action_reduce(F.mse_loss(obsgoal_cond, groundtruth, reduction="none"))
            total_train_loss += loss.item()

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
    total_loss = 0
    num_batches = len(dataloader)

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

            losses = _compute_losses_dnd(obsgoal_cond, ground_truth)
            if use_wandb:
                total_loss += torch.sqrt(losses).item()
                if i == (num_batches - 1):
                    wandb.log({"test_loss": total_loss / num_batches})



def normalize_data(data, stats):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    ndata = ndata * 2 - 1
    return ndata




