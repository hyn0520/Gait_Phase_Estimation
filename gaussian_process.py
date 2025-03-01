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


seed = 42
np.random.seed(seed)


def lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs  
    normal_cutoff = cutoff / nyquist  
    b, a = butter(order, normal_cutoff, btype='low', analog=False)  
    return lfilter(b, a, data)  


def remove_outliers(data, lower_bound, upper_bound, threshold, sigma):
    data = np.clip(data, lower_bound, upper_bound)
    smoothed_data = data.copy()
    for i in range(len(data)):
        if data[i] < threshold:  
            weights = np.exp(-0.5 * ((np.arange(i + 1) - i) / sigma) ** 2)
            weights /= weights.sum()  
            smoothed_data[i] = np.dot(weights, data[:i + 1])
    return smoothed_data

data = pd.read_csv('motion_data_with_ground_truth_-0.2_15.csv')

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

train_ratio = 0.8  
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

gp = joblib.load('gp_model_10000.pkl')
# gp = GaussianProcessRegressor(kernel=kernel, alpha=0.0, n_restarts_optimizer=10, normalize_y=True)
# print("training-----------------------------------------")
# start_time = time.time()
# gp.fit(X_train, y_train)
y_pred, y_std = gp.predict(X_test, return_std=True)
# end_time = time.time()
# fit_time = end_time - start_time
# print(f"GP fitting took {fit_time:.3f} seconds")


# model_filename = 'gp_model.pkl'
# joblib.dump(gp, model_filename)
# print(f"Model saved to {model_filename}")

y_pred= remove_outliers(y_pred, lower_bound=0.0, upper_bound=1.0, threshold=0.9, sigma=2)
#y_pred= gaussian_filter1d(y_pred, sigma=1)  # 设置标准差

plt.figure(figsize=(16, 8), dpi=150)  


plt.plot(times_test, y_test, label='True Phase', color='red', linestyle='-',
         linewidth=0.8, alpha=0.9)
plt.plot(times_test, y_pred, label='Predicted Phase', color='blue', linestyle='--',
         linewidth=0.8, alpha=0.9)
plt.fill_between(times_test, y_pred - y_std, y_pred + y_std,
                 color='blue', alpha=0.1, label='Prediction Std', linewidth=0)

plt.xlabel('Time')
plt.ylabel('Gait Phase')
plt.title('Gaussian Process')
plt.legend()
plt.grid(True)
plt.show()




