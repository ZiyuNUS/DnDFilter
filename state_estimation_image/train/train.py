import os
import wandb
import numpy as np
import yaml
import time

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim import AdamW
from torchvision import transforms
import torch.backends.cudnn as cudnn
from warmup_scheduler import GradualWarmupScheduler

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from DND_train.models.dnd.dnd import DnD
from DND_train.models.dnd.vint import ViNT, replace_bn_with_gn
from DND_train.models.dnd.vint_bkf import ViNT_bkf
from DND_train.models.dnd.vint_VinT_wo_pred import VinT_wo_pred
from diffusion_policy.model.diffusion.conditional_unet1d import ConditionalUnet1D
from DND_train.data.dnd_dataset import DnD_Dataset
from DND_train.training.train_eval_loop import train_eval_loop, load_model

def main(config):

    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = ",".join([str(x) for x in config["gpu_ids"]])
        print("Using cuda devices:", os.environ["CUDA_VISIBLE_DEVICES"])
    else:
        print("Using cpu")
    first_gpu_id = config["gpu_ids"][0]
    device = torch.device(f"cuda:{first_gpu_id}" if torch.cuda.is_available() else "cpu")

    if "seed" in config:
        np.random.seed(config["seed"])
        torch.manual_seed(config["seed"])
        cudnn.deterministic = True

    cudnn.benchmark = True

    transform = ([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    transform = transforms.Compose(transform)

    train_dataset = []
    test_dataloaders = {}

    data_config = config["datasets"]["RED"]

    for data_split_type in ["train", "test"]:
        dataset = DnD_Dataset(
            data_folder=data_config[f"{data_split_type}_data_folder"],
            data_split_folder=data_config[data_split_type],
            dataset_name="RED",
            image_size=config["image_size"],
            record_spacing=data_config["record_spacing"],
            len_traj_pred=config["len_traj_pred"],
            context_size=config["context_size"],
            context_type=config["context_type"],
            end_slack=data_config["end_slack"],
            normalize=config["normalize"],
        )
        if data_split_type == "train":
            train_dataset.append(dataset)
        else:
            dataset_type = f"RED_{data_split_type}"
            if dataset_type not in test_dataloaders:
                test_dataloaders[dataset_type] = {}
            test_dataloaders[dataset_type] = dataset

    train_dataset = ConcatDataset(train_dataset)
    # set up the training data
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        drop_last=False,
        persistent_workers=True,
    )

    for dataset_type, dataset in test_dataloaders.items():
        test_dataloaders[dataset_type] = DataLoader(
            dataset,
            batch_size=config["eval_batch_size"],
            shuffle=False,
            num_workers=0,
            drop_last=False,
        )

    if config["vision_encoder"] == "vint":
        vision_encoder = ViNT(
            obs_encoding_size=config["encoding_size"],
            context_size=config["visual_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
        vision_encoder = replace_bn_with_gn(vision_encoder)
    elif config["vision_encoder"] == "bkf":
        vision_encoder = ViNT_bkf(
            obs_encoding_size=config["encoding_size"],
            context_size=config["context_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    elif config["vision_encoder"] == "VinT(w.o. pred)":
        vision_encoder = VinT_wo_pred(
            obs_encoding_size=config["encoding_size"],
            context_size=config["context_size"],
            mha_num_attention_heads=config["mha_num_attention_heads"],
            mha_num_attention_layers=config["mha_num_attention_layers"],
            mha_ff_dim_factor=config["mha_ff_dim_factor"],
        )
    else:
        raise ValueError(f"Vision encoder {config['vision_encoder']} not supported")

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

    noise_scheduler = DDPMScheduler(
        num_train_timesteps=config["num_diffusion_iters"],
        beta_schedule='squaredcos_cap_v2',
        clip_sample=True,
        prediction_type='epsilon'
    )

    lr = float(config["lr"])
    config["optimizer"] = config["optimizer"].lower()
    if config["optimizer"] == "adamw":
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
        latest_path = os.path.join(load_project_folder, "latest.pth")
        latest_checkpoint = torch.load(latest_path)  # f"cuda:{}" if torch.cuda.is_available() else "cpu")
        load_model(model, latest_checkpoint)
        if "epoch" in latest_checkpoint:
            current_epoch = latest_checkpoint["epoch"] + 1

    model = model.to(device)

    if "load_run" in config:  # load optimizer and scheduler after data parallel
        if "optimizer" in latest_checkpoint:
            optimizer.load_state_dict(latest_checkpoint["optimizer"].state_dict())
        if scheduler is not None and "scheduler" in latest_checkpoint:
            scheduler.load_state_dict(latest_checkpoint["scheduler"].state_dict())

    train_eval_loop(
        train_model=config["train"],
        model=model,
        optimizer=optimizer,
        lr_scheduler=scheduler,
        noise_scheduler=noise_scheduler,
        train_loader=train_loader,
        test_dataloaders=test_dataloaders,
        transform=transform,
        epochs=config["epochs"],
        device=device,
        project_folder=config["project_folder"],
        print_log_freq=config["print_log_freq"],
        wandb_log_freq=config["wandb_log_freq"],
        current_epoch=current_epoch,
        use_wandb=config["use_wandb"],
        eval_freq=config["eval_freq"],
        training_setting=config["training_setting"]
    )

    print("FINISHED TRAINING")


if __name__ == "__main__":
    torch.multiprocessing.set_start_method("spawn")
    config_route = "config/VinT(w.o. pred).yaml"

    with open(config_route, "r") as f:
        user_config = yaml.safe_load(f)
    config = user_config
    config["run_name"] += "_" + time.strftime("%Y_%m_%d_%H_%M_%S")
    config["project_folder"] = os.path.join(
        "logs", config["project_name"], config["run_name"]
    )
    os.makedirs(config["project_folder"])

    if config["use_wandb"]:
        wandb.login()
        wandb.init(
            project=config["project_name"],
            settings=wandb.Settings(start_method="fork"),
        )
        wandb.save(config_route, policy="now")
        wandb.run.name = config["run_name"]
        if wandb.run:
            wandb.config.update(config)

    main(config)
