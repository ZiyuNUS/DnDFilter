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

def _compute_losses_nomad(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        ground_truth,
        device: torch.device,
):
    pred_horizon = ground_truth.shape[1]
    action_dim = ground_truth.shape[2]

    model_output_dict = model_output(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        pred_horizon,
        action_dim,
        device=device,
    )
    gc_actions = model_output_dict['gc_actions']

    def action_reduce(unreduced_loss: torch.Tensor):
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        return unreduced_loss.mean()

    gc_actions = unnormalize_data(gc_actions.cpu(), ACTION_STATS)
    ground_truth = unnormalize_data(ground_truth.cpu(), ACTION_STATS)
    gc_action_loss = action_reduce(F.mse_loss(gc_actions, ground_truth, reduction="none"))
    results = {"gc_action_loss": gc_action_loss}
    return results

def _compute_losses_nomad_evaluate(
        predict_positions,
        ground_truth,
):
    def action_reduce(unreduced_loss: torch.Tensor):
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        return unreduced_loss.mean()

    squared_errors = F.mse_loss(predict_positions[:, 0, :].cpu(), ground_truth[:, 0, :].cpu(), reduction="none")
    gc_action_loss = torch.sqrt(action_reduce(squared_errors))
    results = {"gc_action_loss": gc_action_loss,}
    return results

def train_nomad(
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

            block = ground_truth[:, 0, 4]
            ground_truth = ground_truth[:, :, :4]
            obs_images = torch.split(obs_image, 3, dim=1)

            pos = ground_truth[:, 0, :2].to(device)
            predict_pos = torch.zeros(ground_truth.shape[0], 2).to(device)
            predict_pos[0] = pos[0]

            for count_block in range(1, ground_truth.shape[0]):
                if block[count_block - 1] < 10:
                    predict_pos[count_block] = pos[count_block]
                else:
                    predict_pos[count_block] = 0

            gt_pos = predict_pos.long()

            img_batch_tensor = torch.zeros(gt_pos.shape[0], 3, 128, 128)
            for count in range(gt_pos.shape[0]):
                if block[count] < 10:
                    x_center, y_center = gt_pos[count, 0].item(), gt_pos[count, 1].item()
                    for x in range(max(0, x_center - 3), min(128, x_center + 3 + 1)):
                        for y in range(max(0, y_center - 3), min(128, y_center + 3 + 1)):
                            if (x - x_center) ** 2 + (y - y_center) ** 2 <= 3 ** 2:
                                img_batch_tensor[count, 0, y, x] = 1.0
                                img_batch_tensor[count, 1, y, x] = 0.0
                                img_batch_tensor[count, 2, y, x] = 0.0
                else:
                    img_batch_tensor[count] = obs_images[3][count]

            obs_images += (img_batch_tensor,)

            batch_obs_images = [transform(obs) for obs in obs_images]
            batch_obs_images = torch.cat(batch_obs_images, dim=1).to(device)

            B = obs_image.shape[0]

            obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images)
            ground_truth = normalize_data(ground_truth, ACTION_STATS).to(device)[:, :, :2]

            noise = torch.randn(ground_truth.shape, device=device)

            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()

            noisy_action = noise_scheduler.add_noise(ground_truth, noise, timesteps).to(torch.float32)

            noise_pred = model("noise_pred_net", sample=noisy_action, timestep=timesteps, global_cond=obsgoal_cond)

            def action_reduce(unreduced_loss: torch.Tensor):
                while unreduced_loss.dim() > 1:
                    unreduced_loss = unreduced_loss.mean(dim=-1)
                return unreduced_loss.mean()

            diffusion_loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
            loss = diffusion_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema_model.step(model)

            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            if use_wandb:
                wandb.log({"diffusion_loss": diffusion_loss.item()})

            if i % print_log_freq == 0:
                losses = _compute_losses_nomad(
                    ema_model.averaged_model,
                    noise_scheduler,
                    batch_obs_images,
                    ground_truth,
                    device,
                )
                total_train_loss = total_train_loss + losses['gc_action_loss'].item()
        if use_wandb and wandb_log_freq != 0:
            log_data = {'pos_loss (train)': total_train_loss / (i + 1)}
            wandb.log(log_data, commit=True)


def evaluate_nomad(
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
            B = obs_image.shape[0]
            conditions = []
            positions = []
            ground_truth = ground_truth[:, :, :4]
            vel = ground_truth[0, 1, 2:4].to(device).unsqueeze(0)
            pos = ground_truth[0, 1, :2].to(device).unsqueeze(0)

            for ii in range(B):
                indicator_image = torch.zeros(3, 128, 128)
                x_center, y_center = round(pos[0, 0].item()), round(pos[0, 1].item())

                y_grid, x_grid = torch.meshgrid(torch.arange(128), torch.arange(128), indexing='ij')
                distance_squared = (x_grid - x_center) ** 2 + (y_grid - y_center) ** 2
                mask = distance_squared <= 9
                indicator_image[0][mask] = 1.0
                indicator_image[1][mask] = 0.0
                indicator_image[2][mask] = 0.0
                obs_images = torch.split(obs_image[ii], 3, dim=0)
                obs_images += (indicator_image,)
                batch_obs_images = [transform(obs) for obs in obs_images]
                batch_obs_images = torch.cat(batch_obs_images, dim=0).to(device)
                model_output_dict, condition = model_output_evaluate(
                    ema_model,
                    noise_scheduler,
                    batch_obs_images.unsqueeze(0),
                    pred_horizon=4,
                    action_dim=2,
                    device=device)
                conditions.append(condition)
                positions.append(model_output_dict['gc_actions'] * 128)
                pos = model_output_dict['gc_actions'][:, 0, :] * 128 + vel
            predict_positions = torch.cat(positions, dim=0)
            ground_truth = ground_truth[:, :, :2]
            losses = _compute_losses_nomad_evaluate(predict_positions, ground_truth)
            total_test_loss = total_test_loss + losses['gc_action_loss'].item()
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


def model_output(
        model: nn.Module,
        noise_scheduler: DDPMScheduler,
        batch_obs_images: torch.Tensor,
        pred_horizon: int,
        action_dim: int,
        device: torch.device,
):
    obs_cond = model("vision_encoder", obs_img=batch_obs_images)
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps[:]:
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obs_cond
        )
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample
    gc_actions = diffusion_output
    return {'gc_actions': gc_actions}

def model_output_evaluate(
        model: nn.Module,
        noise_scheduler: DDPMScheduler,
        batch_obs_images: torch.Tensor,
        pred_horizon: int,
        action_dim: int,
        device: torch.device,
):
    obs_cond = model("vision_encoder", obs_img=batch_obs_images)
    noisy_diffusion_output = torch.randn(
        (len(obs_cond), pred_horizon, action_dim), device=device)
    diffusion_output = noisy_diffusion_output

    for k in noise_scheduler.timesteps[:]:
        noise_pred = model(
            "noise_pred_net",
            sample=diffusion_output,
            timestep=k.unsqueeze(-1).repeat(diffusion_output.shape[0]).to(device),
            global_cond=obs_cond
        )
        diffusion_output = noise_scheduler.step(
            model_output=noise_pred,
            timestep=k,
            sample=diffusion_output
        ).prev_sample
    gc_actions = diffusion_output
    return {'gc_actions': gc_actions,}, obs_cond

