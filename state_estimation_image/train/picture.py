# import numpy as np
# import matplotlib.pyplot as plt
#
# # ======= 读取数据 =======
# gc_x = np.load('gc_actions_x8.npy')
# gc_y = np.load('gc_actions_y8.npy')
# gt_x = np.load('ground_truth_x8.npy')
# gt_y = np.load('ground_truth_y8.npy')
# gc_x_m = np.load('gc_actions_x_m8.npy')
# gc_y_m = np.load('gc_actions_y_m8.npy')
# gt_x_m = np.load('ground_truth_x_m8.npy')
# gt_y_m = np.load('ground_truth_y_m8.npy')
# gc_x_g = np.load('gc_actions_x_g8.npy')
# gc_y_g = np.load('gc_actions_y_g8.npy')
# gt_x_g = np.load('ground_truth_x_g8.npy')
# gt_y_g = np.load('ground_truth_y_g8.npy')
# time_steps = np.arange(len(gc_x))  # 时间步
#
# # ======= 同时绘制两张图 =======
# plt.figure(figsize=(9, 4))
#
# # 绘制 X 轴数据
# plt.subplot(1, 2, 1)  # 第1个子图
# plt.plot(time_steps, gc_x, label='DiffDF (no pred)')
# plt.plot(time_steps, gc_x_m, label='DiffDF (10s)')
# plt.plot(time_steps, gc_x_g, label='DiffDF (10s with gt)')
# plt.plot(time_steps, gt_x, label='Ground Truth')
# plt.xlabel('Time Steps',fontsize=13)
# plt.ylabel('trajectory on X axis(pixels)',fontsize=13)
# plt.xlim(30, 94)
#
# # 绘制 Y 轴数据
# plt.subplot(1, 2, 2)  # 第2个子图
# plt.plot(time_steps, gc_y, label='DiffDF (no pred)')
# plt.plot(time_steps, gc_y_m, label='DiffDF (10s)')
# plt.plot(time_steps, gc_y_g, label='DiffDF (10s with gt)')
# plt.plot(time_steps, gt_y, label='Ground Truth')
# plt.xlabel('Time Steps',fontsize=13)
# plt.ylabel('trajectory on Y axis(pixels)',fontsize=13)
# plt.legend(fontsize=13)
# plt.yticks([])
# plt.xlim(30, 90)
#
# plt.tight_layout()  # 自动调整子图间距
# plt.show()
#

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

def load_and_diff(image_path1, image_path2):
    img1 = Image.open(image_path1)
    img2 = Image.open(image_path2)
    img1_np = np.array(img1)
    img2_np = np.array(img2)
    diff_image = np.abs(img2_np.astype(np.int16) - img1_np.astype(np.int16)).astype(np.uint8)
    return img1_np, img2_np, diff_image

# 加载13号文件夹的图像
img1_13, img2_13, diff_13 = load_and_diff(
    '/home/yuyu/diffusion_model/KITTI/data_odometry_color/13/0.png',
    '/home/yuyu/diffusion_model/KITTI/data_odometry_color/13/1.png'
)

# 加载14号文件夹的图像
img1_14, img2_14, diff_14 = load_and_diff(
    '/home/yuyu/diffusion_model/KITTI/data_odometry_color/755/0.png',
    '/home/yuyu/diffusion_model/KITTI/data_odometry_color/755/1.png'
)

# 可视化
plt.figure(figsize=(15, 10))

# 第1排：13号文件夹
plt.subplot(2, 3, 1)
plt.imshow(img1_13)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.imshow(img2_13)
plt.axis('off')

plt.subplot(2, 3, 3)
plt.imshow(diff_13)
plt.axis('off')

# 第2排：14号文件夹
plt.subplot(2, 3, 4)
plt.imshow(img1_14)
plt.axis('off')

plt.subplot(2, 3, 5)
plt.imshow(img2_14)
plt.axis('off')

plt.subplot(2, 3, 6)
plt.imshow(diff_14)
plt.axis('off')

plt.tight_layout()
plt.show()


