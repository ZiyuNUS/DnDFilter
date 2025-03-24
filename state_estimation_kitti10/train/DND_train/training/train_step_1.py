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

with open(os.path.join(os.path.dirname(__file__), "../data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)
ACTION_STATS = {}
for key in data_config['state_stats']:
    ACTION_STATS[key] = np.array(data_config['state_stats'][key])

def _compute_losses_dnd(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        model_info,
        ground_truth,
        device: torch.device,
):
    pred_horizon = 4
    action_dim = 2

    model_output_dict = model_output(
        ema_model,
        noise_scheduler,
        batch_obs_images,
        model_info,
        pred_horizon,
        action_dim,
        device=device,
    )
    gc_actions = model_output_dict['gc_actions']
    gc_actions = unnormalize_data(gc_actions.cpu(), ACTION_STATS).to(device)
    gc_actions = gc_actions[:, 1, :].cpu().detach().numpy()

    theta0 = ground_truth[0, 0, 2].cpu().detach().numpy()
    p0 = ground_truth[0, 0, 3:5].cpu().detach().numpy()
    pT = ground_truth[0, 0, 5:7].cpu().detach().numpy()

    ground_truth = ground_truth[:, 1, :2].cpu().detach().numpy()

    predict_dpos = gc_actions[:, 0]
    predict_dang = gc_actions[:, 1]
    gt_dpos = ground_truth[:, 0]
    gt_dangle = ground_truth[:, 1]

    predict_ang = np.cumsum(predict_dang, axis=0) + theta0
    gt_angle = np.cumsum(gt_dangle, axis=0) + theta0
    predict_ang = np.insert(predict_ang, 0, theta0)
    gt_angle = np.insert(gt_angle, 0, theta0)

    delta_x = predict_dpos * np.sin(np.deg2rad(predict_ang[1:]))
    delta_y = predict_dpos * np.cos(np.deg2rad(predict_ang[1:]))
    predict_x = np.cumsum(delta_x, axis=0) + p0[0]
    predict_y = np.cumsum(delta_y, axis=0) + p0[1]
    errors_pos = np.sqrt((pT[0] - predict_x[-1]) ** 2 + (pT[1] - predict_y[-1]) ** 2)

    errors_angle = np.abs(predict_ang[-1] - gt_angle[-1])
    total_distance = np.sum(gt_dpos)
    mean_error_pos = errors_pos / total_distance
    mean_error_angle = errors_angle / total_distance

    results = {"gc_action_loss_pos": mean_error_pos,
               "gc_action_loss_ang": mean_error_angle, }
    return results


def train_dnd(
        model: nn.Module,
        ema_model: EMAModel,
        optimizer: Adam,
        dataloader: DataLoader,
        device: torch.device,
        noise_scheduler: DDPMScheduler,
        print_log_freq: int = 100,
        wandb_log_freq: int = 10,
        use_wandb: bool = True,
):
    model.train()
    total_train_loss_pos = 0
    total_train_loss_ang = 0

    with tqdm.tqdm(dataloader, desc="Train Batch", leave=False) as tepoch:
        for i, data in enumerate(tepoch):
            (
                obs_image,
                ground_truth,
            ) = data

            model_info = normalize_data(ground_truth[:, 0, :2], ACTION_STATS).to(device).float()
            mask = (torch.rand(99) < 1).to(device)
            model_info[mask] = 0

            all_ground_truth = ground_truth
            ground_truth = ground_truth[:, :, :2]
            B = obs_image.shape[0]

            obs_images = torch.split(obs_image, 3, dim=1)
            batch_obs_images = torch.cat(obs_images, dim=1).to(device)
            obsgoal_cond = model("vision_encoder", obs_img=batch_obs_images)
            obsgoal_cond = model("state_extractor", state_vector = model_info, obs_img = obsgoal_cond)

            ground_truth = normalize_data(ground_truth, ACTION_STATS).to(device)
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

            loss = action_reduce(F.mse_loss(noise_pred, noise, reduction="none"))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            ema_model.step(model)

            loss_cpu = loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            if use_wandb:
                wandb.log({"diffusion_loss": loss.item()})
            if i % print_log_freq == 0:
                losses = _compute_losses_dnd(
                    ema_model.averaged_model,
                    noise_scheduler,
                    batch_obs_images,
                    model_info,
                    all_ground_truth,
                    device,
                )
                total_train_loss_pos = total_train_loss_pos + losses['gc_action_loss_pos'].item()
                total_train_loss_ang = total_train_loss_ang + losses['gc_action_loss_ang'].item()
        if use_wandb and wandb_log_freq != 0:
            log_data = {
                'pos_loss (train)': total_train_loss_pos / (i + 1),
                'ang_loss (train)': total_train_loss_ang / (i + 1)
            }
            wandb.log(log_data, commit=True)


def evaluate_dnd(
        eval_type: str,
        ema_model: EMAModel,
        dataloader: DataLoader,
        device: torch.device,
        noise_scheduler: DDPMScheduler,
        epoch: int,
        print_log_freq: int = 100,
        wandb_log_freq: int = 10,
        use_wandb: bool = True,
):
    ema_model = ema_model.averaged_model
    ema_model.eval()

    num_batches = len(dataloader)
    total_test_loss_pos = 0
    total_test_loss_ang = 0

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

            model_info = normalize_data(ground_truth[:, 0, :2], ACTION_STATS).to(device).float()
            mask = (torch.rand(99) < 1).to(device)
            model_info[mask] = 0

            all_ground_truth = ground_truth
            ground_truth = ground_truth[:, :, :2]
            B = obs_image.shape[0]

            obs_images = torch.split(obs_image, 3, dim=1)
            batch_obs_images = torch.cat(obs_images, dim=1).to(device)
            obsgoal_cond = ema_model("vision_encoder", obs_img=batch_obs_images)
            obsgoal_cond = ema_model("state_extractor", state_vector = model_info, obs_img = obsgoal_cond)

            ground_truth = normalize_data(ground_truth[:, :, :2], ACTION_STATS).to(device)
            noise = torch.randn(ground_truth.shape, device=device)
            timesteps = torch.randint(
                0, noise_scheduler.config.num_train_timesteps,
                (B,), device=device
            ).long()
            noisy_actions = noise_scheduler.add_noise(ground_truth, noise, timesteps).to(torch.float32)

            goal_mask_noise_pred = ema_model("noise_pred_net", sample=noisy_actions, timestep=timesteps,
                                             global_cond=obsgoal_cond)
            goal_mask_loss = nn.functional.mse_loss(goal_mask_noise_pred, noise)

            loss_cpu = goal_mask_loss.item()
            tepoch.set_postfix(loss=loss_cpu)
            if use_wandb:
                wandb.log({"diffusion_eval_loss (test)": goal_mask_loss})
            if i % print_log_freq == 0 and print_log_freq != 0:
                losses = _compute_losses_dnd(
                    ema_model,
                    noise_scheduler,
                    batch_obs_images,
                    model_info,
                    all_ground_truth,
                    device,
                )
                total_test_loss_pos = total_test_loss_pos + losses['gc_action_loss_pos'].item()
                total_test_loss_ang = total_test_loss_ang + losses['gc_action_loss_ang'].item()
        if use_wandb and wandb_log_freq != 0:
            log_data = {
                'pos_loss (test)': total_test_loss_pos / (i + 1),
                'ang_loss (test)': total_test_loss_ang / (i + 1)
            }
            wandb.log(log_data, commit=True)


def normalize_data(data, stats):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    ndata = ndata * 2 - 1
    return ndata


def unnormalize_data(ndata, stats):
    # 反归一化过程
    data = (ndata + 1) / 2  # 将数据从 [-1, 1] 转回到 [0, 1]
    data = data * (stats['max'] - stats['min']) + stats['min']  # 按原始数据范围缩放
    return data


def model_output(
        model: nn.Module,
        noise_scheduler: DDPMScheduler,
        batch_obs_images: torch.Tensor,
        model_info: torch.Tensor,
        pred_horizon: int,
        action_dim: int,
        device: torch.device,
):
    obs_cond = model("vision_encoder", obs_img=batch_obs_images)
    obs_cond = model("state_extractor", state_vector = model_info, obs_img = obs_cond)
    noisy_diffusion_output = torch.randn((len(obs_cond), pred_horizon, action_dim), device=device)
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

    return {
        'gc_actions': gc_actions,
    }






