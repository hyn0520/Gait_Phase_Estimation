import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import torch
import gpytorch
from gpytorch.mlls import VariationalELBO
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.models import ApproximateGP
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from scipy.signal import butter, lfilter
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import os
import random
import math
from torch.utils.data import DataLoader, TensorDataset, random_split


# Allow duplicates of OpenMP libraries without error
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Restrict PyTorch to a single CPU thread
torch.set_num_threads(1)

# Set a fixed seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Select the device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

#################### 1. 数据读取与预处理 ####################
def lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist 频率
    normal_cutoff = cutoff / nyquist  # 归一化截止频率
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # 设计滤波器
    return lfilter(b, a, data)  # 单向滤波

# 仅示例取部分行，防止数据过大
data = pd.read_csv("/home/yinan/Documents/sconepy_environment_ps_2425/output/motion_data_with_ground_truth_-0.2_15.csv")
data = data.sort_values('time')

# 简单的裁剪
data['imu_acc_x'] = data['imu_acc_x'].clip(-10, 5)
data['imu_acc_y'] = data['imu_acc_y'].clip(-10, 5)
data['imu_ang_acc_z'] = data['imu_ang_acc_z'].clip(-40, 20)

cutoff_freq = 5.0  # 截止频率
fs = 100           # 采样率
filtered_columns = ['imu_acc_x', 'imu_acc_y', 'imu_ang_vel_z']
for col in filtered_columns:
    if col in data.columns:
        data[f'{col}_filtered'] = lowpass_filter(data[col], cutoff_freq, fs)

train_ratio = 0.8
split_index = int(len(data) * train_ratio)


X = data[['imu_acc_x_filtered', 'imu_acc_y_filtered', 'imu_ang_vel_z_filtered']].values
y = data['gait_phase'].values
times = data['time'].values

X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.float32)

# Normalize features to [0,1]
X_min = X_tensor.min(dim=0, keepdim=True).values
X_max = X_tensor.max(dim=0, keepdim=True).values
X_tensor = (X_tensor - X_min) / (X_max - X_min)


dataset = TensorDataset(X_tensor, y_tensor)
num_sequences = X_tensor.shape[0]
train_size = int(0.8 * num_sequences)
X_train = X_tensor[:train_size]
y_train = y_tensor[:train_size]
X_val   = X_tensor[train_size:]
y_val   = y_tensor[train_size:]

# 构造 TensorDataset
train_dataset = TensorDataset(X_train, y_train)
val_dataset   = TensorDataset(X_val, y_val)
times_test = times[train_size:]
# 创建 DataLoader
batch_size = 500
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
# val_size = num_sequences - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# # Create DataLoaders
# batch_size = 100
# train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
# val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)


M = 500
Z_init = X_tensor[:M]

class SparseGPModel(ApproximateGP):
    def __init__(self, inducing_points):
        # Variational Distribution & Variational Strategy
        variational_distribution = CholeskyVariationalDistribution(num_inducing_points=inducing_points.size(0))
        variational_strategy = VariationalStrategy(
            self,
            inducing_points=inducing_points,
            variational_distribution=variational_distribution,
            learn_inducing_locations=True
        )
        super().__init__(variational_strategy)

        # 平均函数、核函数
        self.mean_module = gpytorch.means.ConstantMean()
        # 可以叠加多个核, 比如 RBF + 近似Periodic, 这里仅用 RBF 做示例
        self.periodic = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.PeriodicKernel(
                 period_length=100.0,  # 可选设置初值
            )
        )
        self.rbf = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(
                # lengthscale=10.0,
            )
        )
        # 最终叠加
        self.covar_module = self.periodic + self.rbf


    def forward(self, x):
        # x shape: [batch_size, 3]
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

#################### 3. 定义Likelihood & 模型实例 ####################
likelihood = GaussianLikelihood().cuda()
model = SparseGPModel(inducing_points=Z_init).cuda()
mll = VariationalELBO(likelihood, model, num_data=len(train_dataset))
# optimizer = torch.optim.Adam([
#     {'params': model.parameters()},
#     {'params': likelihood.parameters()},
# ], lr=0.001)

# scheduler = StepLR(optimizer, step_size=10, gamma=0.5)
# scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
# num_epochs = 100
# start_time = time.time()
# for epoch in range(num_epochs):
#     #########################
#     #      Training phase   #
#     #########################
#     model.train()
#     likelihood.train()
#     train_loss = 0.0
    
#     for batch_X, batch_y in train_loader:
#         batch_X = batch_X.to(device)
#         batch_y = batch_y.to(device)
        
#         optimizer.zero_grad()
#         # 前向传播: 得到分布
#         output_dist = model(batch_X)
#         # 损失函数: -mll
#         loss = -mll(output_dist, batch_y)
#         loss.backward()
#         optimizer.step()
#         train_loss += loss.item()
#     train_loss /= len(train_loader)

#     model.eval()
#     likelihood.eval()
#     val_loss = 0.0
#     with torch.no_grad():
#         for batch_X, batch_y in val_loader:
#             batch_X = batch_X.to(device)
#             batch_y = batch_y.to(device)

#             output_dist = model(batch_X)
#             loss = -mll(output_dist, batch_y)
#             val_loss += loss.item()
#     val_loss /= len(val_loader)
#     scheduler.step(val_loss)
    
#     # 打印信息
#     current_lr = optimizer.param_groups[0]['lr']
#     print(f"Epoch [{epoch+1}/{num_epochs}], "
#           f"Train Loss: {train_loss:.6f}, "
#           f"Val Loss: {val_loss:.6f}, "
#           f"LR: {current_lr:.6f}")
# end_time = time.time()
# print(f"Training finished. Time used: {end_time - start_time:.2f}s")
# model_save_path = "sparse_gp_model.pth"
# likelihood_save_path = "likelihood.pth"
# torch.save(model.state_dict(), model_save_path)
# torch.save(likelihood.state_dict(), likelihood_save_path)
# print(f"Model saved to {model_save_path}")

# val_indices = val_dataset.indices  # Subset里的原索引
# 对应时间
val_times = times_test     # times是原始numpy数组
y_pred_list = []
y_true_list = []
# # 重新构造相同的模型结构
# model = SparseGPModel(inducing_points=Z_init).to(device)
# likelihood = GaussianLikelihood().to(device)

# # 加载权重
model.load_state_dict(torch.load("sparse_gp_model.pth"))
likelihood.load_state_dict(torch.load("likelihood.pth"))

# # 切换到 eval 模式
# model_load.eval()
# likelihood_load.eval()


# 推理
model.eval()
likelihood.eval()
with torch.no_grad():
    for batch_X, batch_y in val_loader:
        batch_X = batch_X.to(device)
        out_dist = model(batch_X)
        # 对于回归，只取均值即可
        y_pred = out_dist.mean  # shape: (batch_size,) if 1D output
        y_pred = y_pred.cpu().numpy()
        y_pred = np.clip(y_pred, 0.0, 1.0)
        y_pred_list.append(y_pred)
        y_true_list.append(batch_y.cpu().numpy())

y_pred_all = np.concatenate(y_pred_list, axis=0)  # (N_val,)
y_true_all = np.concatenate(y_true_list, axis=0)  # (N_val,)


plt.figure(figsize=(16, 8), dpi=150)
plt.plot(val_times, y_true_all, label='True Phase', color='red', linewidth=0.8)
plt.plot(val_times, y_pred_all, label='Predicted Phase', color='blue', linestyle='--', linewidth=0.8)
plt.xlabel('Time')
plt.ylabel('Gait Phase')
plt.title('Gaussian Process')
plt.legend()
plt.grid(True)
plt.show()

