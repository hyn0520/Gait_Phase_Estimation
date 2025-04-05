import pandas as pd
import torch
import os
from scipy.signal import butter, lfilter
import numpy as np
import random
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from matplotlib.widgets import Slider
from scipy.interpolate import NearestNDInterpolator

# Set environment variable to handle potential library conflicts
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Set single thread for PyTorch operations
torch.set_num_threads(1)

# Set random seeds for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Set device for computation (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

def lowpass_filter(data, cutoff, fs, order=4):
    """
    Apply a low-pass Butterworth filter to the data
    Args:
        data: Input signal to be filtered
        cutoff: Cutoff frequency
        fs: Sampling frequency
        order: Filter order
    Returns:
        Filtered signal
    """
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

def resample_to_uniform_length(X, y, y_phase, num_points=100):
    """
    Resample each cycle of X and y to the same length using interpolation
    Args:
        X: Input features
        y: Target values
        y_phase: Phase information for cycle identification
        num_points: Number of points to resample to
    Returns:
        Resampled X and y arrays
    """
    start_indices = np.where(y_phase == 0)[0]  # Find start indices of each cycle
    X_resampled, y_resampled = [], []

    for i in range(len(start_indices) - 1):
        start, end = start_indices[i], start_indices[i + 1]
        X_segment = X[start:end]
        y_segment = y[start:end]

        # Create uniform time points for interpolation
        t_original = np.linspace(0, 1, len(y_segment))
        t_uniform = np.linspace(0, 1, num_points)

        X_interp = [interp1d(t_original, X_segment[:, j], kind='linear')(t_uniform) for j in range(X.shape[1])]
        y_interp = interp1d(t_original, y_segment, kind='linear')(t_uniform)

        X_resampled.append(np.vstack(X_interp).T)
        y_resampled.append(y_interp)

    return np.array(X_resampled), np.array(y_resampled)

# Load motion data from CSV file
file_path = "./motion_data_with_ground_truth_-0.2_15.csv"
data = pd.read_csv(file_path)

# Set filter parameters and apply low-pass filter to IMU data
cutoff_freq = 5.0
fs = 100
columns_to_plot = []
filtered_columns = ['imu_acc_x', 'imu_acc_y', 'imu_acc_z','imu_ang_vel_x', 'imu_ang_vel_y', 'imu_ang_vel_z']

# Apply filtering and prepare columns for plotting
for col in filtered_columns:
    if col in data.columns:
        data[f'{col}_filtered'] = lowpass_filter(data[col], cutoff_freq, fs)
        columns_to_plot.append((f'{col}_filtered', "filtered"))
    if 'gait_phase' in data.columns:
        columns_to_plot.append(('gait_phase', "gait_phase"))

# Default to plotting all columns if none specified
if not columns_to_plot:
    columns_to_plot = [(col, "original") for col in data.columns[1:]]

# Set up visualization parameters
visible_rows = 4  # Number of rows displayed at once
visible_time_range = 10  # Initial horizontal display time range (in seconds)
fig_width = 5  # Figure width
fig_height = visible_rows * 3  # Each subplot is 3 units high

# Create figure and subplots
fig, ax = plt.subplots(visible_rows, 1, figsize=(fig_width, fig_height), sharex=True)
current_start_col = 0  # Current starting column index
current_start_time = 0  # Current starting time

def update_plot(start_col, start_time):
    """
    Update the plot with new column and time range
    Args:
        start_col: Starting column index
        start_time: Starting time for x-axis
    """
    start_col = int(start_col)
    for i in range(visible_rows):
        ax[i].clear()
        if start_col + i < len(columns_to_plot):
            column, col_type = columns_to_plot[start_col + i]
            # Filter data within the horizontal time range
            mask = (data['time'] >= start_time) & (data['time'] <= start_time + visible_time_range)
            if col_type == "gait_phase":
                ax[i].plot(data['time'][mask], data[column][mask], label=column, color='red')
            elif col_type == "filtered":
                ax[i].plot(data['time'][mask], data[column][mask], label=column, color='green')
            else:
                ax[i].plot(data['time'][mask], data[column][mask], label=column)
            ax[i].set_ylabel(column, fontsize=10)
            ax[i].legend(loc='upper right', fontsize=8)
            ax[i].grid(True)
        else:
            ax[i].axis('off')

    ax[-1].set_xlabel('Time (s)', fontsize=12)
    plt.draw()

update_plot(current_start_col, current_start_time)

# Add sliders for navigation
slider_ax_col = plt.axes([0.01, 0.2, 0.02, 0.6], facecolor='lightgrey')
slider_col = Slider(slider_ax_col, 'Column Scroll', 0, max(0, len(columns_to_plot) - visible_rows), valinit=0, valstep=1, orientation='vertical')

slider_ax_time = plt.axes([0.1, 0.02, 0.8, 0.03], facecolor='lightgrey')
slider_time = Slider(slider_ax_time, 'Time Scroll', 0, max(0, data['time'].max() - visible_time_range), valinit=0, valstep=0.1)

# Slider event handlers
def on_col_slider_update(val):
    global current_start_col
    current_start_col = int(slider_col.val)
    update_plot(current_start_col, current_start_time)

def on_time_slider_update(val):
    global current_start_time
    current_start_time = slider_time.val
    update_plot(current_start_col, current_start_time)

slider_col.on_changed(on_col_slider_update)
slider_time.on_changed(on_time_slider_update)

plt.tight_layout(rect=[0.05, 0.1, 0.95, 0.95])
plt.show()

# Prepare features and labels for analysis
features = [f'{col}_filtered' for col in filtered_columns if f'{col}_filtered' in data.columns]
label = 'gait_phase'
X = data[features].values
y = data[label].values

# Resample data to uniform length
X_resampled, y_resampled = resample_to_uniform_length(X, y, y)

# Calculate means for feature visualization
X_means = np.mean(X_resampled, axis=0)
y_mean = np.mean(y_resampled, axis=0)

time_steps = np.linspace(0, 1, X_resampled.shape[1])

# Plot individual features across cycles
for i, col in enumerate(features):
    plt.figure(figsize=(12, 6))
    
    for cycle_idx, X_cycle in enumerate(X_resampled):
        plt.plot(time_steps, X_cycle[:, i], color='gray', alpha=0.3)
    print(X_means[:, i])

    plt.plot(time_steps, X_means[:, i], color='red', linewidth=2, label='Mean')
    plt.xlabel('Normalized Time Step')
    plt.ylabel(f'{col}')
    plt.title(f'Comparison of {col} Across Cycles with Mean')
    plt.legend()
    plt.grid()
    plt.show()

# Plot output signal across cycles
plt.figure(figsize=(12, 6))
for cycle_idx, y_cycle in enumerate(y_resampled):
    plt.plot(time_steps, y_cycle, color='gray', alpha=0.3)
    print(y_cycle)

plt.plot(time_steps, y_mean, color='red', linewidth=2, label='Mean')
plt.xlabel('Normalized Time Step')
plt.ylabel('Output Signal (y)')
plt.title('Comparison of Output Signal Across Cycles with Mean')
plt.legend()
plt.grid()
plt.show()

print(X_means.shape, y_mean.shape)
X_mean_flat = X_means
y_mean_flat = y_mean

# Save lookup table data
np.savez("./Lookup_table/lookup_table.npz", X_mean_flat=X_mean_flat, y_mean_flat=y_mean_flat)

# Load lookup table data
data = np.load("./Lookup_table/lookup_table.npz")
X_mean_flat_loaded = data['X_mean_flat']
y_mean_flat_loaded = data['y_mean_flat']

# Create interpolator for prediction
interpolator = NearestNDInterpolator(X_mean_flat_loaded, y_mean_flat_loaded)

# Load new data for prediction
file_path = './motion_data_with_ground_truth_-0.2_15.csv'
data = pd.read_csv(file_path)

# Apply filtering to new data
cutoff_freq = 5.0
fs = 100
filtered_columns = ['imu_acc_x', 'imu_acc_y', 'imu_acc_z','imu_ang_vel_x', 'imu_ang_vel_y', 'imu_ang_vel_z']
for col in filtered_columns:
    if col in data.columns:
        data[f'{col}_filtered'] = lowpass_filter(data[col], cutoff_freq, fs)

# Prepare features and labels for prediction
features = [f'{col}_filtered' for col in filtered_columns if f'{col}_filtered' in data.columns]
label = 'gait_phase'
X_new = data[features].values
y_new = data[label].values

# Make predictions using interpolator
y_pred = []
for x in X_new:
    y_pred.append(interpolator(x))
y_pred = np.array(y_pred)
y_true = y_new

# Set up visualization parameters for prediction results
visible_time_range = 500
fig_width_cm = visible_time_range / 50
fig_height = 8

fig_width_inches = fig_width_cm / 2.54
fig_height_inches = fig_height / 2.54

num_points = len(y)
y_true = y
pred_start = 0
y_pred_aligned = y_pred

# Create figure for prediction visualization
fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))
current_start_time = 0

def update_plot(start_time):
    """
    Update the prediction visualization plot
    Args:
        start_time: Starting time index for visualization
    """
    ax.clear()
    end_time = start_time + visible_time_range

    if end_time > num_points:
        end_time = num_points

    # Plot actual data
    actual_time_index = np.arange(start_time, end_time)
    actual_data = y_true[start_time:end_time]
    ax.plot(actual_time_index, actual_data, label='Actual', color='blue', linestyle='-', linewidth=1.5)

    # Plot predicted data
    pred_indices = np.arange(pred_start, pred_start + len(y_pred_aligned))
    pred_mask = (pred_indices >= start_time) & (pred_indices < end_time)
    if np.any(pred_mask):
        visible_pred_indices = pred_indices[pred_mask]
        visible_pred_data = y_pred_aligned[pred_mask]
        ax.plot(visible_pred_indices, visible_pred_data, label='Predicted', color='red', linestyle='-', linewidth=1.5)

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Gait Cycle Value', fontsize=12)
    ax.set_title('Gait Phase Prediction vs Actual', fontsize=16)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)

    plt.draw()

update_plot(current_start_time)

# Add time slider for navigation
slider_ax_time = plt.axes([0.1, 0.02, 0.8, 0.03], facecolor='lightgrey')
slider_time = Slider(
    slider_ax_time,
    'Time Scroll',
    0,
    max(0, num_points - visible_time_range),
    valinit=0,
    valstep=1
)

def on_time_slider_update(val):
    global current_start_time
    current_start_time = int(slider_time.val)
    update_plot(current_start_time)

slider_time.on_changed(on_time_slider_update)

plt.tight_layout(rect=[0.05, 0.1, 0.95, 0.95])
plt.show()




