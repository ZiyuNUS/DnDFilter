import os
import pickle
import yaml
from typing import Tuple
import tqdm
import io
import lmdb
import numpy as np
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset

from vint_train.data.data_utils import (
    img_path_to_data,
    get_data_path,
)

class ViNT_Dataset(Dataset):
    def __init__(
        self,
        data_folder: str,
        data_split_folder: str,
        dataset_name: str,
        image_size: Tuple[int, int],
        record_spacing: int,
        len_traj_pred: int,
        context_size: int,
        context_type: str = "temporal",
        end_slack: int = 0,
        normalize: bool = True,
        obs_type: str = "image",
        goal_type: str = "image",
    ):
        self.data_folder = data_folder
        self.data_split_folder = data_split_folder
        self.dataset_name = dataset_name
        
        traj_names_file = os.path.join(data_split_folder, "traj_names.txt")
        with open(traj_names_file, "r") as f:
            file_lines = f.read()
            self.traj_names = file_lines.split("\n")
        if "" in self.traj_names:
            self.traj_names.remove("")

        self.image_size = image_size
        self.record_spacing = record_spacing
        self.len_traj_pred = len_traj_pred
        self.context_size = context_size
        assert context_type in {
            "temporal",
            "randomized",
            "randomized_temporal",
        }, "context_type must be one of temporal, randomized, randomized_temporal"
        self.context_type = context_type
        self.end_slack = end_slack
        self.normalize = normalize
        self.obs_type = obs_type
        self.goal_type = goal_type

        with open(
            os.path.join(os.path.dirname(__file__), "data_config.yaml"), "r"
        ) as f:
            all_data_config = yaml.safe_load(f)
        assert (
            self.dataset_name in all_data_config
        ), f"Dataset {self.dataset_name} not found in data_config.yaml"
        dataset_names = list(all_data_config.keys())
        dataset_names.sort()
        # use this index to retrieve the dataset name from the data_config.yaml
        self.dataset_index = dataset_names.index(self.dataset_name)
        self.data_config = all_data_config[self.dataset_name]
        self.trajectory_cache = {}
        self._load_index()
        print(len(self.index_to_data))
        self._build_caches()


    def __getstate__(self):
        state = self.__dict__.copy()
        state["_image_cache"] = None
        return state
    
    def __setstate__(self, state):
        self.__dict__ = state
        self._build_caches()

    def _build_caches(self, use_tqdm: bool = True):
        cache_filename = os.path.join(
            self.data_split_folder,
            f"dataset_{self.dataset_name}.lmdb",
        )
        for traj_name in self.traj_names:
            self._get_trajectory(traj_name)

        if not os.path.exists(cache_filename):
            tqdm_iterator = tqdm.tqdm(
                self.goals_index,
                disable=not use_tqdm,
                dynamic_ncols=True,
                desc=f"Building LMDB cache for {self.dataset_name}"
            )
            with lmdb.open(cache_filename, map_size=2**40) as image_cache:
                with image_cache.begin(write=True) as txn:
                    for traj_name, time in tqdm_iterator:
                        image_path = get_data_path(self.data_folder, traj_name, time)
                        with open(image_path, "rb") as f:
                            txn.put(image_path.encode(), f.read())
        # Reopen the cache file in read-only mode
        self._image_cache: lmdb.Environment = lmdb.open(cache_filename, readonly=True)

    def _build_index(self, use_tqdm: bool = False):
        samples_index = []
        goals_index = []

        for traj_name in tqdm.tqdm(self.traj_names, disable=not use_tqdm, dynamic_ncols=True):
            traj_data = self._get_trajectory(traj_name)
            traj_len = len(traj_data["position"])

            for goal_time in range(0, traj_len):
                goals_index.append((traj_name, goal_time))

            begin_time = self.context_size * self.record_spacing
            end_time = traj_len
            for curr_time in range(begin_time, end_time):
                samples_index.append((traj_name, curr_time))
        return samples_index, goals_index

    def _load_index(self) -> None:
        index_to_data_path = os.path.join(
            self.data_split_folder,
            f"dataset_context_{self.context_type}_n{self.context_size}_slack_{self.end_slack}.pkl",
        )
        try:
            with open(index_to_data_path, "rb") as f:
                self.index_to_data, self.goals_index = pickle.load(f)
        except:
            self.index_to_data, self.goals_index = self._build_index()
            with open(index_to_data_path, "wb") as f:
                pickle.dump((self.index_to_data, self.goals_index), f)

    def _load_image(self, trajectory_name, time):
        image_path = get_data_path(self.data_folder, trajectory_name, time)
        try:
            with self._image_cache.begin() as txn:
                image_buffer = txn.get(image_path.encode())
                image_bytes = torch.load(io.BytesIO(image_buffer))
            return image_bytes
        except TypeError:
            print(f"Failed to load image {image_path}")

    def _get_trajectory(self, trajectory_name):
        if trajectory_name in self.trajectory_cache:
            return self.trajectory_cache[trajectory_name]
        else:
            with open(os.path.join(self.data_folder, trajectory_name, "traj_data.pkl"), "rb") as f:
                traj_data = pickle.load(f)
            self.trajectory_cache[trajectory_name] = traj_data
            return traj_data

    def _load_gt_vel_pos(self, trajectory, current_time):
        trajectory_data = self._get_trajectory(trajectory)
        pos_x = trajectory_data['position'][current_time, 0]
        pos_y = trajectory_data['position'][current_time, 1]
        vel_x = trajectory_data['velocity'][current_time, 0]
        vel_y = trajectory_data['velocity'][current_time, 1]
        block = trajectory_data['num'][current_time]
        ground_truth = np.array([pos_x, pos_y, vel_x, vel_y, block])
        return ground_truth

    def __len__(self) -> int:
        return len(self.index_to_data)

    def __getitem__(self, i: int) -> Tuple[torch.Tensor]:
        f_curr, curr_time = self.index_to_data[i]

        if self.context_type == "temporal":
            context_times = list(
                range(
                    curr_time + -self.context_size * self.record_spacing,
                    curr_time + 1,
                    self.record_spacing,
                )
            )
            context = [(f_curr, t) for t in context_times]
        else:
            raise ValueError(f"Invalid context type {self.context_type}")

        ground_truth_times = [curr_time for _ in range(4)]
        ground_truth_group = [(f_curr, t) for t in ground_truth_times]

        obs_image = torch.cat([
            self._load_image(f, t) for f, t in context
        ])

        ground_truth = torch.stack(
            [torch.tensor(self._load_gt_vel_pos(f, t)) for f, t in ground_truth_group])

        return (
            torch.as_tensor(obs_image, dtype=torch.float32),
            ground_truth,
        )
