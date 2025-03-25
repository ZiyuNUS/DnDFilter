import tqdm
import itertools
from diffusers.training_utils import EMAModel
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
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler
import time
from DND_train.models.dnd.dnd import DnD
from DND_train.models.dnd.CNN import SensorModel
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from DND_train.data.dnd_dataset import DnD_Dataset
from typing import Dict

with open(os.path.join(os.path.dirname(__file__), "/home/yuyu/diffusion_model/state_estimation_vio/train/vint_train/data/data_config.yaml"), "r") as f:
    data_config = yaml.safe_load(f)
ACTION_STATS = {}
for key in data_config['state_stats']:
    ACTION_STATS[key] = np.array(data_config['state_stats'][key])
i = 0
total_loss = []
total_loss_angle = []

def _compute_losses_dnd(
    predict_dpositions,
    ground_truth,theta0,p0,pT
):
    gc_actions = unnormalize_data(predict_dpositions.cpu(), ACTION_STATS)
    gc_actions = gc_actions[:, 1, :].cpu().detach().numpy()

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
    errors_pos = np.sqrt((pT[0] - predict_x[-1])**2 + (pT[1] - predict_y[-1])**2)

    errors_angle = np.abs(predict_ang[-1] - gt_angle[-1])
    total_distance = np.sum(gt_dpos)
    mean_error_pos = errors_pos / total_distance
    mean_error_angle = errors_angle / total_distance

    results = {"gc_action_loss_pos": mean_error_pos,
               "gc_action_loss_ang": mean_error_angle,}
    return results

def evaluate_dnd(
        eval_type: str,
        ema_model: EMAModel,
        dataloader: DataLoader,
        transform: transforms,
        device: torch.device,
        noise_scheduler: DDPMScheduler,
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
            conditions = []
            dpositions = []
            pre_pt = []
            theta0 = ground_truth[0, 0, 2].cpu().detach().numpy()
            p0 = ground_truth[0, 0, 3:5].cpu().detach().numpy()
            pT = ground_truth[0, 0, 5:7].cpu().detach().numpy()
            dpos = ground_truth[0, 0, 0]
            dangle = ground_truth[0, 0, 1]

            a = time.time()
            for ii in range(B):
                [dpos, dangle] = normalize_data(torch.tensor([dpos, dangle]), ACTION_STATS)
                pre_pt.append(torch.stack([dpos, dangle]))

                obs_images = torch.split(obs_image[ii], 3, dim=0)
                batch_obs_images = torch.cat(obs_images, dim=0).to(device)

                model_output_dict, condition = model_output(
                    ema_model.to(device),
                    noise_scheduler,
                    batch_obs_images.unsqueeze(0).to(device),
                    torch.stack([dpos, dangle]),
                    pred_horizon=4,
                    action_dim=2,
                    device=device,
                )
                conditions.append(condition)
                dpositions.append(model_output_dict['gc_actions'])
                dpos_predict = model_output_dict['gc_actions'][0, 0, 0]
                dangle_predict = model_output_dict['gc_actions'][0, 0, 1]
                [dpos, dangle] = unnormalize_data(torch.tensor([dpos_predict, dangle_predict]), ACTION_STATS)
            b = time.time()
            print(b-a)
            predict_dpositions = torch.cat(dpositions, dim=0)

            losses = _compute_losses_dnd(predict_dpositions, ground_truth, theta0, p0, pT)

            global total_loss, total_loss_angle
            total_loss.append(losses['gc_action_loss_pos'].item())
            total_loss_angle.append(losses['gc_action_loss_ang'].item())
            print(f"(epoch {epoch}) (batch {i}/{num_batches - 1}) {losses}")
            if i == (num_batches - 1):
                print(np.mean(total_loss))
                print(np.std(total_loss) / np.sqrt(len(total_loss)))
                print(np.mean(total_loss_angle))
                print(np.std(total_loss_angle) / np.sqrt(len(total_loss_angle)))

def normalize_data(data, stats):
    ndata = (data - stats['min']) / (stats['max'] - stats['min'])
    ndata = ndata * 2 - 1
    return ndata

def unnormalize_data(ndata, stats):
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
    obs_cond = model("state_extractor", state_vector = model_info.to(device).unsqueeze(0).float(), obs_img = obs_cond)
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

def eval_loop_dnd(
        model: nn.Module,
        lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
        noise_scheduler: DDPMScheduler,
        test_dataloaders: Dict[str, DataLoader],
        transform: transforms,
        epochs: int,
        device: torch.device,
        current_epoch: int = 0,
        eval_freq: int = 1,
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
                noise_scheduler=noise_scheduler,
                epoch=epoch,
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
    vision_encoder = SensorModel()
    noise_pred_net = ConditionalUnet1D(
        input_dim=2,
        global_cond_dim=config["encoding_size"] + 2,
        down_dims=config["down_dims"],
        cond_predict_scale=config["cond_predict_scale"],
    )
    class state_extractor(nn.Module):
        def __init__(self, input_dim_high=128, input_dim_low=2, fused_dim=256):
            super(state_extractor, self).__init__()
            self.low_dim_to_high = nn.Linear(input_dim_low, input_dim_high)
            self.fusion_mlp = nn.Sequential(
                nn.Linear(input_dim_high * 2, fused_dim),
                nn.ReLU(),
                nn.Linear(fused_dim, fused_dim),
            )

        def forward(self, vector_high, vector_low):
            return torch.cat((vector_high, vector_low), dim=1)

    state_extractor = state_extractor()
    model = DnD(
        vision_encoder=vision_encoder,
        noise_pred_net=noise_pred_net,
        state_extractor=state_extractor
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
    eval_loop_dnd(
        model=model,
        lr_scheduler=scheduler,
        noise_scheduler=noise_scheduler,
        test_dataloaders=test_dataloaders,
        transform=transform,
        epochs=config["epochs"],
        device=device,
        current_epoch=current_epoch,
        eval_freq=config["eval_freq"],
    )

if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    config_route = "config/kitti10_fold_11_sim.yaml"
    with open(config_route, "r") as f:
        user_config = yaml.safe_load(f)
    config = user_config
    config['load_run'] = 'kitti10_fold11/s2simple'
    main(config)