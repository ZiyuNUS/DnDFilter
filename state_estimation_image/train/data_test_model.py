import tqdm
import itertools
from diffusers.training_utils import EMAModel
import torch.nn.functional as F
import os
import time
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
from DND_train.models.dnd.dnd import DnD
from DND_train.models.dnd.vint import ViNT, replace_bn_with_gn
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from DND_train.data.dnd_dataset import DnD_Dataset
from typing import Dict
import matplotlib.pyplot as plt

with open(os.path.join(os.path.dirname(__file__), "DND_train/data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)
ACTION_STATS = {}
for key in data_config['state_stats']:
    ACTION_STATS[key] = np.array(data_config['state_stats'][key])
i = 0
total_loss = 0
total_var = 0
def _compute_losses_nomad(
        predict_positions,
        ground_truth,
):
    def action_reduce(unreduced_loss: torch.Tensor):
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        return unreduced_loss.mean()

    squared_errors = F.mse_loss(predict_positions[:, 0, :].cpu(), ground_truth[:, 0, :].cpu(), reduction="none")
    gc_action_loss = torch.sqrt(action_reduce(squared_errors))

    sample_rmse = torch.sqrt(squared_errors.mean(dim=1))
    gc_action_loss_variance = sample_rmse.std()

    results = {
        "gc_action_loss": gc_action_loss,
        "gc_action_loss_variance": gc_action_loss_variance,
    }
    return results

def evaluate_nomad(
        eval_type: str,
        ema_model: EMAModel,
        dataloader: DataLoader,
        transform: transforms,
        device: torch.device,
        noise_scheduler: DDPMScheduler,
        epoch: int,
        use_gt: int,
):
    ema_model = ema_model.averaged_model
    ema_model.eval()
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
            B = obs_image.shape[0]
            conditions = []
            positions = []
            ground_truth = ground_truth[:, :, :4]
            vel = ground_truth[0, 1, 2:4].to(device).unsqueeze(0)
            pos = ground_truth[0, 1, :2].to(device).unsqueeze(0)
            a = time.time()
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
                model_output_dict, condition = model_output(
                    ema_model,
                    noise_scheduler,
                    batch_obs_images.unsqueeze(0).to(device),
                    pred_horizon=4,
                    action_dim=2,
                    device=device,
                )
                conditions.append(condition)
                positions.append(model_output_dict['gc_actions'] * 128)
                pos = model_output_dict['gc_actions'][:, 0, :] * 128 + vel
                if (ii + 1 < 93) & use_gt:
                    pos = ground_truth[ii+1, 1, :2].to(device).unsqueeze(0)
            b = time.time()
            print(b-a)
            predict_positions = torch.cat(positions, dim=0)
            ground_truth = ground_truth[:, :, :2]

            losses = _compute_losses_nomad(predict_positions,ground_truth)
            global total_loss, total_var
            total_loss += losses['gc_action_loss'].item()
            total_var += losses['gc_action_loss_variance'].item()
            print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {losses}")
            if i == (num_batches - 1):
                print(total_loss / num_batches)
                print(total_var / num_batches)

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
    model = model.to(device)
    obs_cond = model("vision_encoder", obs_img=batch_obs_images)
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

    return {'gc_actions': gc_actions,}, obs_cond

def eval_loop_nomad(
        model: nn.Module,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        noise_scheduler: DDPMScheduler,
        test_dataloaders: Dict[str, DataLoader],
        transform: transforms,
        epochs: int,
        use_gt: int,
        device: torch.device,
        current_epoch: int = 0,
        eval_freq: int = 1,
):
    ema_model = EMAModel(model=model, power=0.75)
    for epoch in range(1):
        for dataset_type in test_dataloaders:
            print(f"Start {dataset_type} ViNT DP Testing Epoch {epoch}/{current_epoch + epochs - 1}")
            loader = test_dataloaders[dataset_type]
            evaluate_nomad(
                eval_type=dataset_type,
                ema_model=ema_model,
                dataloader=loader,
                transform=transform,
                device=device,
                noise_scheduler=noise_scheduler,
                epoch=epoch,
                use_gt=use_gt,
            )
        if lr_scheduler is not None:
            lr_scheduler.step()

def load_model(model, checkpoint: dict) -> None:
    state_dict = checkpoint
    model.load_state_dict(state_dict, strict=False)

def main(config):
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in config["gpu_ids"]])
    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu")

    cudnn.benchmark = True
    transform = ([transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),])
    transform = transforms.Compose(transform)

    test_dataloaders = {}
    for dataset_name in config["datasets"]:
        data_config = config["datasets"][dataset_name]
        if "end_slack" not in data_config:
            data_config["end_slack"] = 0
        if "record_spacing" not in data_config:
            data_config["record_spacing"] = 1
        for data_split_type in ["test"]:
            dataset = DnD_Dataset(
                data_folder=data_config[f"{data_split_type}_data_folder"],
                data_split_folder=data_config[data_split_type],
                dataset_name=dataset_name,
                image_size=config["image_size"],
                record_spacing=data_config["record_spacing"],
                len_traj_pred=config["len_traj_pred"],
                context_size=config["context_size"],
                context_type=config["context_type"],
                end_slack=data_config["end_slack"],
                normalize=config["normalize"],
            )
            dataset_type = f"{dataset_name}_{data_split_type}"
            test_dataloaders[dataset_type] = dataset
    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )
    vision_encoder = ViNT(
        obs_encoding_size=config["encoding_size"],
        context_size=config["context_size"] + 1,
        mha_num_attention_heads=config["mha_num_attention_heads"],
        mha_num_attention_layers=config["mha_num_attention_layers"],
        mha_ff_dim_factor=config["mha_ff_dim_factor"],
    )
    vision_encoder = replace_bn_with_gn(vision_encoder)
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config["encoding_size"],
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
    )
    model = DnD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
    )
    total_params = sum(p.numel() for p in vision_encoder.parameters() if p.requires_grad)
    print(total_params)
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )
    if config["clipping"]:
        for p in model.parameters():
            if not p.requires_grad:
                continue
            p.register_hook(
                lambda grad: torch.clamp(
                    grad, -1 * config["max_norm"], config["max_norm"]
                )
            )
    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    optimizer = AdamW(model.parameters(), lr=lr)
    scheduler = None
    if config["scheduler"] is not None:
        config["scheduler"] = config["scheduler"].lower()
        if config["scheduler"] == "cosine":
            print("Using cosine annealing with T_max", config["epochs"])
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=config["epochs"]
            )
        if config["warmup"]:
            print("Using warmup scheduler")
            scheduler = GradualWarmupScheduler(
                optimizer,
                multiplier=1,
                total_epoch=config["warmup_epochs"],
                after_scheduler=scheduler,
            )
    current_epoch = 0
    if "load_run" in config:
        load_project_folder = os.path.join("logs", config["load_run"])
        print("Loading model from ", load_project_folder)
        latest_path = os.path.join(load_project_folder, "final.pth")
        latest_checkpoint = torch.load(latest_path)  # f"cuda:{}" if torch.cuda.is_available() else "cpu")
        load_model(model, latest_checkpoint)
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())
    eval_loop_nomad(
        model=model,
        lr_scheduler=scheduler,
        noise_scheduler=noise_scheduler,
        test_dataloaders=test_dataloaders,
        transform=transform,
        epochs=config["epochs"],
        device=device,
        current_epoch=current_epoch,
        eval_freq=config["eval_freq"],
        use_gt=config["use_gt"],
    )
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    config_route = "config/DnD(10s with gt).yaml"
    with open(config_route, "r") as f:
        user_config = yaml.safe_load(f)
    config = user_config
    config['load_run'] = 'state_image/DnD(10s)'
    main(config)
