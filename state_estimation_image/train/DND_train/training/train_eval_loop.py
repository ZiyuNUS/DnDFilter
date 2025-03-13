import wandb
import os

from typing import Dict

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam
from torchvision import transforms

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from diffusers.training_utils import EMAModel

import importlib

def dynamic_import(train_module_path):
    module = importlib.import_module(train_module_path)
    return module.train_nomad, module.evaluate_nomad

def train_eval_loop(
        train_model: bool,
        model: nn.Module,
        optimizer: Adam,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        noise_scheduler: DDPMScheduler,
        train_loader: DataLoader,
        test_dataloaders: Dict[str, DataLoader],
        transform: transforms,
        epochs: int,
        device: torch.device,
        project_folder: str,
        training_setting: str,
        print_log_freq: int = 100,
        wandb_log_freq: int = 10,
        current_epoch: int = 0,
        use_wandb: bool = True,
        eval_freq: int = 1,
):
    latest_path = os.path.join(project_folder, f"latest.pth")
    ema_model = EMAModel(model=model, power=0.75)

    train_module_path = f"DND_train.training.train_{training_setting}"  # 直接用字符串
    train_nomad, evaluate_nomad = dynamic_import(train_module_path)

    for epoch in range(current_epoch, current_epoch + epochs):
        if train_model:
            print(
                f"Start ViNT DP Training Epoch {epoch}/{current_epoch + epochs - 1}"
            )
            torch.cuda.empty_cache()
            train_nomad(
                model=model,
                ema_model=ema_model,
                optimizer=optimizer,
                dataloader=train_loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                epoch=epoch,
                print_log_freq=print_log_freq,
                wandb_log_freq=wandb_log_freq,
                use_wandb=use_wandb,
            )
            lr_scheduler.step()

        numbered_path = os.path.join(project_folder, f"ema_{epoch}.pth")
        torch.save(ema_model.averaged_model.state_dict(), numbered_path)
        numbered_path = os.path.join(project_folder, f"ema_latest.pth")
        print(f"Saved EMA model to {numbered_path}")

        numbered_path = os.path.join(project_folder, f"{epoch}.pth")
        torch.save(model.state_dict(), numbered_path)
        torch.save(model.state_dict(), latest_path)
        print(f"Saved model to {numbered_path}")

        # save optimizer
        latest_optimizer_path = os.path.join(project_folder, f"optimizer_latest.pth")
        torch.save(optimizer.state_dict(), latest_optimizer_path)

        # save scheduler
        latest_scheduler_path = os.path.join(project_folder, f"scheduler_latest.pth")
        torch.save(lr_scheduler.state_dict(), latest_scheduler_path)

        if (epoch + 1) % eval_freq == 0:
            for dataset_type in test_dataloaders:
                print(
                    f"Start {dataset_type} ViNT DP Testing Epoch {epoch}/{current_epoch + epochs - 1}"
                )
                loader = test_dataloaders[dataset_type]
                evaluate_nomad(
                    eval_type=dataset_type,
                    ema_model=ema_model,
                    dataloader=loader,
                    transform=transform,
                    device=device,
                    noise_scheduler=noise_scheduler,
                    epoch=epoch,
                    wandb_log_freq=wandb_log_freq,
                    use_wandb=use_wandb,
                )
        if use_wandb:
            wandb.log({
                "lr": optimizer.param_groups[0]["lr"],
            }, commit=False)

        if lr_scheduler is not None:
            lr_scheduler.step()
        if use_wandb:
            wandb.log({}, commit=False)
            wandb.log({
                "lr": optimizer.param_groups[0]["lr"],
            }, commit=False)
    if use_wandb:  # Flush the last set of eval logs
        wandb.log({})
    print()


def load_model(model, model_type, checkpoint: dict) -> None:
    """Load model from checkpoint."""
    if model_type == "nomad":
        state_dict = checkpoint
        model.load_state_dict(state_dict, strict=False)
