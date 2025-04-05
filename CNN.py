import os
import math
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from matplotlib.widgets import Slider
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch.optim.lr_scheduler import ReduceLROnPlateau
from scipy.signal import butter, lfilter

seed = 42
window_size = 30
batch_size = 256
num_epochs = 100
learning_rate = 1e-3

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
torch.set_num_threads(1)

random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

class CNNModel(nn.Module):
    def __init__(self, in_channels=6, window_size=30, out_dim=2):
        super(CNNModel, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(16)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()

        self.fc = nn.Linear(32 * window_size, out_dim)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)

        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
            return
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


file_path = './motion_data_with_ground_truth_-0.2_45.csv'
data = pd.read_csv(file_path)

cutoff_freq = 5.0
fs = 100
filtered_columns = [
    'imu_acc_x', 'imu_acc_y', 'imu_acc_z',
    'imu_ang_vel_x', 'imu_ang_vel_y', 'imu_ang_vel_z'
]
for col in filtered_columns:
    if col in data.columns:
        data[f'{col}_filtered'] = lowpass_filter(data[col], cutoff_freq, fs)

features = [f'{col}_filtered' for col in filtered_columns if f'{col}_filtered' in data.columns]
label = 'gait_phase'

X = data[features].values
y = data[label].values

X_tensor = torch.tensor(X, dtype=torch.float32)
X_min = X_tensor.min(dim=0, keepdim=True).values
X_max = X_tensor.max(dim=0, keepdim=True).values
X_tensor = (X_tensor - X_min) / (X_max - X_min)
X = X_tensor.numpy()

y_phase_original = torch.tensor(y, dtype=torch.float32)
angle = 2 * math.pi * y_phase_original
y_cos = torch.cos(angle)
y_sin = torch.sin(angle)
y_encoded = torch.stack([y_cos, y_sin], dim=-1).numpy()

X_windows = []
y_windows = []
for i in range(len(X) - window_size + 1):
    window_data = X[i:i + window_size].T
    label_data = y_encoded[i + window_size - 1]
    X_windows.append(window_data)
    y_windows.append(label_data)

X_windows = np.array(X_windows, dtype=np.float32)
y_windows = np.array(y_windows, dtype=np.float32)

X_tensor = torch.from_numpy(X_windows)
y_tensor = torch.from_numpy(y_windows)

dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

model = CNNModel(in_channels=len(features), window_size=window_size, out_dim=2).to(device)
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5, threshold=1e-3)

for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X, batch_y = batch_X.to(device), batch_y.to(device)
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    val_loss /= len(val_loader)
    scheduler.step(val_loss)
    print(
        f"Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss / len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

    if scheduler.optimizer.param_groups[0]['lr'] < 1e-6:
        print("Early stopping triggered!")
        break

model_output_dir = "./CNN/model"
os.makedirs(model_output_dir, exist_ok=True)
model_save_path = os.path.join(model_output_dir, "cnn_model_0.2_45.pth")
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

model_load_path = model_save_path
loaded_model = CNNModel(in_channels=len(features), window_size=window_size, out_dim=2).to(device)
loaded_model.load_state_dict(torch.load(model_load_path, map_location=device))
loaded_model.eval()
print(f"Model loaded from {model_load_path}")

X_all = X_tensor.to(device)  # [num_windows, 6, window_size]
y_all = y_tensor.to(device)  # [num_windows, 2]
with torch.no_grad():
    y_pred = loaded_model(X_all).cpu().numpy()  # [num_windows, 2]
    y_actual = y_all.cpu().numpy()  # [num_windows, 2]


pred_angle = np.arctan2(y_pred[:, 1], y_pred[:, 0])
pred_phase = (pred_angle / (2 * math.pi)) % 1.0


y_phase = y_phase_original[window_size - 1:].numpy()

differences = np.abs(y_phase - pred_phase)
def calculate_gait_phase_error_sample(ground_truth, prediction):
    if np.abs(ground_truth - prediction) < 0.5:
        return abs(ground_truth - prediction)
    elif ground_truth > prediction:
        return abs(ground_truth - (1 + prediction))
    else:
        return abs((1 + ground_truth) - prediction)


calculate_gait_phase_error = np.vectorize(calculate_gait_phase_error_sample)
corrected_error = calculate_gait_phase_error(y_phase, pred_phase)

max_difference_circular = np.max(corrected_error)
min_difference_circular = np.min(corrected_error)
mean_difference_circular = np.mean(corrected_error)
print("------- Circular Distance Results -------")
print(f"Maximum Difference (corrected_error): {max_difference_circular}")
print(f"Minimum Difference (corrected_error): {min_difference_circular}")
print(f"Mean Difference (corrected_error): {mean_difference_circular}")

best_offset = None
best_mean_diff = float('inf')
for offset in np.linspace(0, 1, 101):
    y_pred_shifted = (pred_phase + offset) % 1.0
    linear_diffs = np.abs(y_phase - y_pred_shifted)
    mean_diff = np.mean(linear_diffs)
    if mean_diff < best_mean_diff:
        best_mean_diff = mean_diff
        best_offset = offset

y_pred_aligned = (pred_phase + best_offset) % 1.0
final_diffs = np.abs(y_phase - y_pred_aligned)
max_difference_linear = np.max(final_diffs)
min_difference_linear = np.min(final_diffs)
mean_difference_linear = np.mean(final_diffs)

print("------- Linear Distance with Offset Results -------")
print(f"Best offset found: {best_offset}")
print(f"Maximum Difference (original differences): {max_difference_linear}")
print(f"Minimum Difference (original differences): {min_difference_linear}")
print(f"Mean Difference (original differences): {mean_difference_linear}")

visible_time_range = 1000
fig_width_cm = visible_time_range / 10
fig_height = 8

fig_width_inches = fig_width_cm / 2.54
fig_height_inches = fig_height / 2.54

fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))
current_start_time = 0


def update_plot(start_idx):
    ax.clear()
    end_idx = start_idx + visible_time_range
    actual_data = y_phase[int(start_idx):int(end_idx)]
    predicted_data = y_pred_aligned[int(start_idx):int(end_idx)]

    ax.plot(actual_data, label='Actual', color='blue', linestyle='-', linewidth=1.5)
    ax.plot(predicted_data, label='Predicted', color='red', linestyle='-', linewidth=1.5)

    ax.set_xlabel('Window Index', fontsize=12)
    ax.set_ylabel('Gait Cycle Value', fontsize=12)
    ax.set_title('Actual vs Predicted Gait Cycle (Aligned)', fontsize=16)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.draw()


update_plot(current_start_time)

slider_ax_time = plt.axes([0.1, 0.02, 0.8, 0.03], facecolor='lightgrey')
slider_time = Slider(
    slider_ax_time,
    'Window Scroll',
    0,
    max(0, len(y_phase) - visible_time_range),
    valinit=0,
    valstep=1
)


def on_time_slider_update(val):
    global current_start_time
    current_start_time = slider_time.val
    update_plot(current_start_time)


slider_time.on_changed(on_time_slider_update)
plt.tight_layout(rect=[0.05, 0.1, 0.95, 0.95])
plt.show()
