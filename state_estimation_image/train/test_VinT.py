import tqdm
import itertools
from diffusers.training_utils import EMAModel
import torch.nn.functional as F
import os
import numpy as np
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler
from DND_train.models.dnd.dnd import DnD
from DND_train.models.dnd.vint import replace_bn_with_gn

from DND_train.models.dnd.vint_VinT_only import VinT_only
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from DND_train.data.dnd_dataset import DnD_Dataset
from typing import Dict

with open(os.path.join(os.path.dirname(__file__), "DND_train/data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)
ACTION_STATS = {}
for key in data_config['state_stats']:
    ACTION_STATS[key] = np.array(data_config['state_stats'][key])
i = 0
total_loss = 0
total_var = 0
def _compute_losses_dnd(
        predict_positions,
        ground_truth,
):
    def action_reduce(unreduced_loss: torch.Tensor):
        while unreduced_loss.dim() > 1:
            unreduced_loss = unreduced_loss.mean(dim=-1)
        return unreduced_loss.mean()

    squared_errors = F.mse_loss(predict_positions.cpu(), ground_truth.cpu(), reduction="none")
    gc_action_loss = torch.sqrt(action_reduce(squared_errors))

    sample_rmse = torch.sqrt(squared_errors.mean(dim=1))
    gc_action_loss_variance = sample_rmse.std()

    results = {
        "gc_action_loss": gc_action_loss,
        "gc_action_loss_variance": gc_action_loss_variance,
    }
    return results

def evaluate_dnd(
        eval_type: str,
        ema_model: EMAModel,
        dataloader: DataLoader,
        transform: transforms,
        device: torch.device,
        epoch: int,
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
            positions = []
            ground_truth = ground_truth[:, :, :4]
            vel = ground_truth[0, 1, 2:4].to(device).unsqueeze(0)
            pos = ground_truth[0, 1, :2].to(device).unsqueeze(0)
            for ii in range(B):
                indicator_image = torch.zeros(3, 128, 128)
                x_center, y_center = round(pos[0, 0].item()), round(pos[0, 1].item())

                y_grid, x_grid = torch.meshgrid(torch.arange(128), torch.arange(128), indexing='ij')
                distance_squared = (x_grid - x_center) ** 2 + (y_grid - y_center) ** 2
                mask = distance_squared <= 10
                indicator_image[0][mask] = 1.0
                indicator_image[1][mask] = 0.0
                indicator_image[2][mask] = 0.0
                obs_images = torch.split(obs_image[ii], 3, dim=0)
                obs_images += (indicator_image,)

                batch_obs_images = [transform(obs) for obs in obs_images]
                batch_obs_images = torch.cat(batch_obs_images, dim=0).to(device)
                model_output_dict = model_output(
                    ema_model,
                    batch_obs_images.unsqueeze(0).to(device),
                    device=device,
                )
                positions.append(model_output_dict * 128)
                pos = model_output_dict * 128 + vel

            predict_positions = torch.cat(positions, dim=0)
            ground_truth = ground_truth[:, 0, :2]

            losses = _compute_losses_dnd(predict_positions,ground_truth)
            global total_loss, total_var
            total_loss += losses['gc_action_loss'].item()
            total_var += losses['gc_action_loss_variance'].item()
            print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {losses}")
            if i == (num_batches - 1):
                print(total_loss / num_batches)
                print(total_var / num_batches)

def model_output(
    model: nn.Module,
    batch_obs_images: torch.Tensor,
    device: torch.device,
):
    model = model.to(device)
    obs_cond = model("vision_encoder", obs_img=batch_obs_images)
    return obs_cond

def eval_loop_dnd(
        model: nn.Module,
        test_dataloaders: Dict[str, DataLoader],
        transform: transforms,
        epochs: int,
        device: torch.device,
        current_epoch: int = 0,
):
    ema_model = EMAModel(model=model, power=0.75)
    for epoch in range(1):
        for dataset_type in test_dataloaders:
            print(f"Start {dataset_type} ViNT DP Testing Epoch {epoch}/{current_epoch + epochs - 1}")
            loader = test_dataloaders[dataset_type]
            evaluate_dnd(
                eval_type=dataset_type,
                ema_model=ema_model,
                dataloader=loader,
                transform=transform,
                device=device,
                epoch=epoch,
            )

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
    vision_encoder = VinT_only(
        obs_encoding_size=config["encoding_size"],
        context_size=config["visual_size"],
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
    eval_loop_dnd(
        model=model,
        test_dataloaders=test_dataloaders,
        transform=transform,
        epochs=config["epochs"],
        device=device,
        current_epoch=current_epoch,
    )
if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    config_route = "config/VinT.yaml"
    with open(config_route, "r") as f:
        user_config = yaml.safe_load(f)
    config = user_config
    config['load_run'] = 'state_image/VinT'
    main(config)
