import pandas as pd
import numpy as np
import pickle
import os
import glob

# 定义要遍历的根目录路径
root_dir = '/home/yuyu/diffusion_model/BackpropKF_Reproduction/dataset/linear_moving_circles_ff/circles=50_frames=100_noise=True/test=100/'

# 递归遍历根目录及所有子目录，查找 CSV 文件
for csv_file in glob.glob(os.path.join(root_dir, '**/*/*.csv'), recursive=True):
    if csv_file == '/home/yuyu/diffusion_model/BackpropKF_Reproduction/dataset/linear_moving_circles_ff/circles=50_frames=100_noise=True/test=100/4/4.csv':
        df = pd.read_csv(csv_file)

        # 提取 x, y, vx, vy 列
        x = df['x'].values
        y = df['y'].values
        vx = df['vx'].values
        vy = df['vy'].values
        # theta = df['theta'].values
        # omega = df['omega'].values
        # cx = df['center_x'].values
        # cy = df['center_y'].values
        # R = df['R'].values

        positions = np.column_stack((x, y))
        velocity = np.column_stack((vx, vy))
        # the_ome = np.column_stack((theta, omega))
        # traj_cir = np.column_stack((cx, cy, R))

        result = {
            'position': positions,
            'velocity': velocity
            # 'the_ome': the_ome,
            # 'traj_cir': traj_cir
        }
        directory_path = os.path.dirname(csv_file)
        # 构建 pkl 文件的输出路径
        pkl_file = os.path.join(directory_path, 'images', 'traj_data.pkl')


        # 确保目标文件夹存在
        os.makedirs(os.path.dirname(pkl_file), exist_ok=True)

        # 保存为 pkl 文件
        with open(pkl_file, 'wb') as file:
            pickle.dump(result, file)

        print(f"已处理并保存: {pkl_file}")

# import os
# import shutil
#
# base_dir = '/home/yuyu/diffusion_model/BackpropKF_Reproduction/dataset/linear_moving_circles_ff/circles=50_frames=100_noise=True/train=500/'
#
# for subdir in os.listdir(base_dir):
#     subdir_path = os.path.join(base_dir, subdir)
#     if os.path.isdir(subdir_path):
#         images_dir = os.path.join(subdir_path, 'images')
#         if os.path.isdir(images_dir):
#             traj_file = os.path.join(images_dir, 'traj_data.pkl')
#             if os.path.isfile(traj_file):
#                 shutil.move(traj_file, subdir_path)
#                 print(f"Moved {traj_file} to {subdir_path}")
#             shutil.rmtree(images_dir)
#             print(f"Deleted {images_dir}")
