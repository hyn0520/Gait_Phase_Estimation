import pandas as pd
import torch
import os
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset, random_split
from scipy.signal import butter, lfilter
from torch.optim.lr_scheduler import ReduceLROnPlateau
import numpy as np
import math
import random
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider

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

class MLPModel(nn.Module):
    """
    Multi-Layer Perceptron model with batch normalization and dropout
    """
    def __init__(self, input_size, hidden_size, output_size, dropout_rate=0.0):
        """
        Initialize the MLP model
        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            output_size: Number of output features
            dropout_rate: Dropout probability
        """
        super(MLPModel, self).__init__()
        # First layer with batch normalization and dropout
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout_rate)
        
        # Second layer
        self.fc2 = nn.Linear(hidden_size, hidden_size * 2)
        self.bn2 = nn.BatchNorm1d(hidden_size * 2)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout_rate)
        
        # Third layer
        self.fc3 = nn.Linear(hidden_size * 2, hidden_size * 2)
        self.bn3 = nn.BatchNorm1d(hidden_size * 2)
        self.relu3 = nn.ReLU()
        self.dropout3 = nn.Dropout(dropout_rate)
        
        # Fourth layer
        self.fc4 = nn.Linear(hidden_size * 2, hidden_size * 4)
        self.bn4 = nn.BatchNorm1d(hidden_size * 4)
        self.relu4 = nn.ReLU()
        self.dropout4 = nn.Dropout(dropout_rate)
        
        # Fifth layer
        self.fc5 = nn.Linear(hidden_size * 4, hidden_size * 2)
        self.bn5 = nn.BatchNorm1d(hidden_size * 2)
        self.relu5 = nn.ReLU()
        self.dropout5 = nn.Dropout(dropout_rate)
        
        # Sixth layer
        self.fc6 = nn.Linear(hidden_size * 2, hidden_size)
        self.bn6 = nn.BatchNorm1d(hidden_size)
        self.relu6 = nn.ReLU()
        self.dropout6 = nn.Dropout(dropout_rate)
        
        # Output layer
        self.fc7 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        """
        Forward pass through the network
        Args:
            x: Input tensor
        Returns:
            Network output
        """
        # Sequential processing through all layers
        out = self.fc1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        out = self.fc2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dropout2(out)
        
        out = self.fc3(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.dropout3(out)
        
        out = self.fc4(out)
        out = self.bn4(out)
        out = self.relu4(out)
        out = self.dropout4(out)
        
        out = self.fc5(out)
        out = self.bn5(out)
        out = self.relu5(out)
        out = self.dropout5(out)
        
        out = self.fc6(out)
        out = self.bn6(out)
        out = self.relu6(out)
        out = self.dropout6(out)
        
        out = self.fc7(out)
        return out

class EarlyStopping:
    """
    Early stopping implementation to prevent overfitting
    """
    def __init__(self, patience=10, min_delta=1e-4):
        """
        Initialize early stopping
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change in loss to be considered as improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Check if training should be stopped
        Args:
            val_loss: Current validation loss
        """
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

# Set file paths for data and model
file_path = './motion_data_with_ground_truth_-0.2_45.csv'
model_name = "./MLP/model/mlp_model_0.2_45.pth"

# Load and preprocess data
data = pd.read_csv(file_path)

# Apply low-pass filter to IMU data
cutoff_freq = 5.0
fs = 100
filtered_columns = ['imu_acc_x', 'imu_acc_y', 'imu_acc_z', 'imu_ang_vel_x', 'imu_ang_vel_y', 'imu_ang_vel_z']
for col in filtered_columns:
    if col in data.columns:
        data[f'{col}_filtered'] = lowpass_filter(data[col], cutoff_freq, fs)

# Prepare features and labels
features = [f'{col}_filtered' for col in filtered_columns if f'{col}_filtered' in data.columns]
label = 'gait_phase'

X = data[features].values
y = data[label].values

# Convert to PyTorch tensors
X_tensor = torch.tensor(X, dtype=torch.float32)
y_phase_original = torch.tensor(y, dtype=torch.float32)

# Convert phase to sine and cosine components
angle = 2 * math.pi * y_phase_original
y_cos = torch.cos(angle).unsqueeze(-1)  # [N,1]
y_sin = torch.sin(angle).unsqueeze(-1)  # [N,1]
y_tensor = torch.cat([y_cos, y_sin], dim=-1)  # [N,2]

# Global normalization of input features
X_min = X_tensor.min(dim=0, keepdim=True).values
X_max = X_tensor.max(dim=0, keepdim=True).values
X_tensor = (X_tensor - X_min) / (X_max - X_min)

# Create windowed data for temporal context
window_size = 10
X_windowed = []
y_windowed = []
num_samples = X_tensor.shape[0]

for i in range(num_samples - window_size + 1):
    # Stack window_size consecutive samples
    X_windowed.append(X_tensor[i:i+window_size].reshape(-1))
    # Use the last timestamp's label
    y_windowed.append(y_tensor[i+window_size-1])

X_tensor = torch.stack(X_windowed)
y_tensor = torch.stack(y_windowed)
print(f"Final X_tensor shape: {X_tensor.shape}")
print(f"Final y_tensor shape: {y_tensor.shape}")

# Create dataset and split into train/validation sets
dataset = TensorDataset(X_tensor, y_tensor)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Set model parameters
input_size = len(features) * window_size
hidden_size = 128 
output_size = 2  

# Create data loaders
batch_size = 1000
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
learning_rate = 0.001

# Initialize model, loss function, optimizer and scheduler
model = MLPModel(input_size, hidden_size, output_size).to(device)
criterion = nn.SmoothL1Loss() 
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = ReduceLROnPlateau(
    optimizer,
    mode='min',
    factor=0.1,
    patience=5,
    threshold=1e-3,
    threshold_mode='rel',
    cooldown=0,
    min_lr=1e-12
)

early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

# Training loop
num_epochs = 300
for epoch in range(num_epochs):
    # Training phase
    model.train()
    train_loss = 0
    for batch_X, batch_y in train_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        
        optimizer.zero_grad()
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    # Validation phase
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch_X, batch_y in val_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            outputs = model(batch_X)
            loss = criterion(outputs, batch_y)
            val_loss += loss.item()
    val_loss /= len(val_loader)

    # Update learning rate based on validation loss
    scheduler.step(val_loss)

    # Print training progress
    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.8f}, "
          f"Val Loss: {val_loss:.8f}, LR: {current_lr:.12f}")

    # Check for early stopping
    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

# Save trained model
model_save_path = model_name
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Load model for evaluation
model_save_path = model_name
model_load_path = model_save_path
loaded_model = MLPModel(input_size, hidden_size, output_size).to(device)
loaded_model.load_state_dict(torch.load(model_load_path, map_location=device))
loaded_model.eval()
print(f"Model loaded from {model_load_path}")

# Generate predictions
with torch.no_grad():
    X_tensor = X_tensor.to(device)
    y_pred = loaded_model(X_tensor).cpu().numpy()  # [M,2]
    y_actual = y_tensor.cpu().numpy()              # [M,2]

# Convert predictions back to phase angles
pred_angle = np.arctan2(y_pred[:,1], y_pred[:,0])
pred_phase = (pred_angle / (2*math.pi)) % 1.0

# Get actual phase values
y_phase = y_phase_original[window_size-1:].numpy()

# Calculate circular distance metrics
differences = np.abs(y_phase - pred_phase)
differences = np.minimum(differences, 1 - differences)
max_difference_circular = np.max(differences)
min_difference_circular = np.min(differences)
mean_difference_circular = np.mean(differences)

print("------- Circular Distance Results -------")
print(f"Maximum Difference (circular): {max_difference_circular}")
print(f"Minimum Difference (circular): {min_difference_circular}")
print(f"Mean Difference (circular): {mean_difference_circular}")

# Find best phase offset
best_offset = None
best_mean_diff = float('inf')
for offset in np.linspace(0, 1, 101):
    y_pred_shifted = (pred_phase + offset) % 1.0
    linear_diffs = np.abs(y_phase - y_pred_shifted)
    mean_diff = np.mean(linear_diffs)
    if mean_diff < best_mean_diff:
        best_mean_diff = mean_diff
        best_offset = offset

# Apply best offset and calculate final metrics
y_pred_aligned = (pred_phase + best_offset) % 1.0
final_diffs = np.abs(y_phase - y_pred_aligned)
max_difference_linear = np.max(final_diffs)
min_difference_linear = np.min(final_diffs)
mean_difference_linear = np.mean(final_diffs)

print("------- Linear Distance with Offset Results -------")
print(f"Best offset found: {best_offset}")
print(f"Maximum Difference (linear after offset): {max_difference_linear}")
print(f"Minimum Difference (linear after offset): {min_difference_linear}")
print(f"Mean Difference (linear after offset): {mean_difference_linear}")

# Set up visualization parameters
visible_time_range = 500
fig_width_cm = visible_time_range / 50
fig_height = 8

fig_width_inches = fig_width_cm / 2.54
fig_height_inches = fig_height / 2.54

# Create figure for visualization
fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))

# Set up indices for plotting
num_points = len(y_phase_original)
pred_start = window_size - 1  
pred_indices = np.arange(pred_start, pred_start + len(y_pred_aligned))

current_start_time = 0

def update_plot(start_time):
    """
    Update the visualization plot
    Args:
        start_time: Starting time index for visualization window
    """
    ax.clear()
    end_time = start_time + visible_time_range
    
    if end_time > num_points:
        end_time = num_points
    
    # Plot actual gait phase data
    actual_time_index = np.arange(start_time, end_time)
    actual_data = y_phase_original[start_time:end_time].numpy()
    ax.plot(actual_time_index, actual_data, label='Actual', color='blue', linestyle='-', linewidth=1.5)

    # Plot predicted gait phase data
    pred_mask = (pred_indices >= start_time) & (pred_indices < end_time)
    if np.any(pred_mask):
        visible_pred_indices = pred_indices[pred_mask]
        visible_pred_data = y_pred_aligned[pred_mask]
        ax.plot(visible_pred_indices, visible_pred_data, label='Predicted', color='red', linestyle='-', linewidth=1.5)
    
    # Set plot labels and style
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Gait Cycle Value', fontsize=12)
    ax.set_title('MLP', fontsize=16)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.draw()

# Initialize plot with starting time
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
    """
    Callback function for time slider updates
    Args:
        val: New slider value
    """
    global current_start_time
    current_start_time = slider_time.val
    update_plot(current_start_time)

# Connect slider to update function
slider_time.on_changed(on_time_slider_update)

# Adjust layout and display plot
plt.tight_layout(rect=[0.05, 0.1, 0.95, 0.95])
plt.show()





