import pandas as pd
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ExpSineSquared, RBF, WhiteKernel
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import joblib
from scipy.signal import butter, lfilter
import time
from scipy.ndimage import gaussian_filter1d
from scipy.signal import savgol_filter





def lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs  # Nyquist 频率
    normal_cutoff = cutoff / nyquist  # 归一化截止频率
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  # 设计滤波器
    return lfilter(b, a, data)  # 单向滤波


def remove_outliers(data, lower_bound, upper_bound, threshold, sigma):
    data = np.clip(data, lower_bound, upper_bound)
    smoothed_data = data.copy()
    for i in range(len(data)):
        if data[i] < threshold:  # 仅对小于阈值的值进行平滑
            # 构造高斯权重，仅考虑过去的点
            weights = np.exp(-0.5 * ((np.arange(i + 1) - i) / sigma) ** 2)
            weights /= weights.sum()  # 归一化
            # 平滑当前点，仅使用过去的点
            smoothed_data[i] = np.dot(weights, data[:i + 1])
    return smoothed_data

data = pd.read_csv('/home/yinan/Documents/sconepy_environment_ps_2425/output/motion_data_with_ground_truth_-0.2_15.csv')

data = data.sort_values('time')

data['imu_acc_x'] = data['imu_acc_x'].clip(-10, 5)
data['imu_acc_y'] = data['imu_acc_y'].clip(-10, 5)
data['imu_ang_acc_z'] = data['imu_ang_acc_z'].clip(-40, 20)

cutoff_freq = 5.0  # Cutoff frequency (Hz)
fs = 100  # Sampling frequency (Hz)
filtered_columns = ['imu_acc_x', 'imu_acc_y', 'imu_ang_vel_z' ]

for col in filtered_columns:
    if col in data.columns:
        data[f'{col}_filtered'] = lowpass_filter(data[col], cutoff_freq, fs)

train_ratio = 0.8  # 80%的数据用于训练，20%用于测试
split_index = int(len(data) * train_ratio)

X = data[['imu_acc_x_filtered', 'imu_acc_y_filtered', 'imu_ang_vel_z_filtered']].values
y = data['gait_phase'].values
times = data['time'].values

X_train = X[:split_index]
y_train = y[:split_index]
times_train = times[:split_index]

X_test = X[split_index:]
y_test = y[split_index:]
times_test = times[split_index:]

kernel = (1.0 * ExpSineSquared(length_scale=1.0, periodicity=1.0)
          + 0.5 * RBF(length_scale=1.0)
          + WhiteKernel(noise_level=1e-3))

gp = joblib.load('/home/yinan/Documents/sconepy_environment_ps_2425/trained_gpr_model_10000.pkl')
# gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=10, normalize_y=True)
# print("training-----------------------------------------")
# start_time = time.time()
# gp.fit(X_train, y_train)
y_pred, y_std = gp.predict(X_test, return_std=True)
# end_time = time.time()
# fit_time = end_time - start_time
# print(f"GP fitting took {fit_time:.3f} seconds")

# 保存训练好的模型
# model_filename = '/home/yinan/Documents/sconepy_environment_ps_2425/trained_gpr_model_10000.pkl'
# joblib.dump(gp, model_filename)
# print(f"Model saved to {model_filename}")

y_pred= remove_outliers(y_pred, lower_bound=0.0, upper_bound=1.0, threshold=0.9, sigma=2)
#y_pred= gaussian_filter1d(y_pred, sigma=1)  # 设置标准差

plt.figure(figsize=(16, 8), dpi=150)  # 增大图像尺寸和分辨率，增加dpi可以提高图像清晰度

# 绘制真实值为细红线，不使用标记点
plt.plot(times_test, y_test, label='True Phase', color='red', linestyle='-',
         linewidth=0.8, alpha=0.9)

# # 绘制预测值为细蓝线，不使用标记点
plt.plot(times_test, y_pred, label='Predicted Phase', color='blue', linestyle='--',
         linewidth=0.8, alpha=0.9)
# 绘制预测值为离散点，使用蓝色圆点
# plt.plot(times_test, y_pred, label='Predicted Phase', color='blue', marker='x',
#          markersize=3, linestyle='', alpha=0.9)

# 绘制预测区间带（标准差）
plt.fill_between(times_test, y_pred - y_std, y_pred + y_std,
                 color='blue', alpha=0.1, label='Prediction Std', linewidth=0)



plt.xlabel('Time')
plt.ylabel('Gait Phase')
plt.title('Gaussian Process')
plt.legend()
plt.grid(True)
plt.show()




