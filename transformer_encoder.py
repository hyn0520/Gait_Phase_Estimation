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


# Allow duplicate OpenMP libraries without errors
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

# Limit PyTorch to a single CPU thread
torch.set_num_threads(1)

# Set a fixed seed for reproducibility
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

# Select device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Define a low-pass filter function with a Butterworth filter
def lowpass_filter(data, cutoff, fs, order=4):
    nyquist = 0.5 * fs               # Nyquist frequency
    normal_cutoff = cutoff / nyquist # Normalized cutoff frequency
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return lfilter(b, a, data)

# Positional encoding for Transformer-based architectures
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        
        pe = torch.zeros(max_len, d_model)
        # Even indices: sin; Odd indices: cos
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)
        
        pe = pe.unsqueeze(0)  # Shape: [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: [batch_size, seq_length, d_model]
        seq_length = x.size(1)
        # Add the positional encoding to the input
        x = x + self.pe[:, :seq_length, :]
        return self.dropout(x)

# Transformer encoder model class
class TransformerEncoderModel(nn.Module):
    def __init__(self, input_dim, d_model=128, nhead=8, num_layers=6, dim_feedforward=512, dropout=0.0, output_size=2):
        super(TransformerEncoderModel, self).__init__()
        # Project input to d_model dimension
        self.input_projection = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, dropout)

        # Build Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Output fully-connected layer
        self.fc_out = nn.Linear(d_model, output_size)

    def forward(self, x):
        # x shape: [batch_size, seq_length, input_dim]
        x = self.input_projection(x)
        x = self.pos_encoder(x)
        encoded = self.transformer_encoder(x)
        out = self.fc_out(encoded)
        return out


# Early stopping mechanism
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        # If no previous best_loss set or if current val_loss is significantly better
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

early_stopping = EarlyStopping(patience=10, min_delta=1e-4)

# Models and data can be replaced as needed
file_path = './motion_data_with_ground_truth_-0.35_45.csv'
model_name = "./Transformer_encoder/model/transformer_encoder_model_0.35_45.pth"

# Read CSV data
data = pd.read_csv(file_path)

# Apply a low-pass filter to relevant columns
cutoff_freq = 5.0
fs = 100
filtered_columns = ['imu_acc_x', 'imu_acc_y', 'imu_acc_z', 'imu_ang_vel_x', 'imu_ang_vel_y', 'imu_ang_vel_z']
for col in filtered_columns:
    if col in data.columns:
        data[f'{col}_filtered'] = lowpass_filter(data[col], cutoff_freq, fs)

# Select features and label
features = [f'{col}_filtered' for col in filtered_columns if f'{col}_filtered' in data.columns]
label = 'gait_phase'

# Convert to PyTorch tensors
X = data[features].values
y = data[label].values
X_tensor = torch.tensor(X, dtype=torch.float32)
y_phase_original = torch.tensor(y, dtype=torch.float32)

# Convert gait phase to cos/sin representation
angle = 2 * math.pi * y_phase_original
y_cos = torch.cos(angle).unsqueeze(-1)  # shape: [N,1]
y_sin = torch.sin(angle).unsqueeze(-1)  # shape: [N,1]
y_tensor = torch.cat([y_cos, y_sin], dim=-1)  # shape: [N,2]

# Normalize features to [0,1]
X_min = X_tensor.min(dim=0, keepdim=True).values
X_max = X_tensor.max(dim=0, keepdim=True).values
X_tensor = (X_tensor - X_min) / (X_max - X_min)

# Define sequence length for Transformer
seq_length = 10
num_samples = X_tensor.shape[0]
input_dim = X_tensor.shape[1]

X_seq_list = []
y_seq_list = []

# Build sequences for Transformer input
for i in range(num_samples - seq_length + 1):
    # Slices of length seq_length
    X_seq = X_tensor[i:i+seq_length]         # shape: [seq_length, input_dim]
    y_seq = y_tensor[i:i+seq_length]         # shape: [seq_length, 2]
    
    # Add batch dimension
    X_seq_list.append(X_seq.unsqueeze(0))
    y_seq_list.append(y_seq.unsqueeze(0))

# Concatenate all sequences into final tensors
X_tensor_seq = torch.cat(X_seq_list, dim=0)  # [num_sequences, seq_length, input_dim]
y_tensor_seq = torch.cat(y_seq_list, dim=0)  # [num_sequences, seq_length, 2]

# Prepare dataset and split into training/validation
dataset = TensorDataset(X_tensor_seq, y_tensor_seq)
num_sequences = X_tensor_seq.shape[0]
train_size = int(0.8 * num_sequences)
val_size = num_sequences - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# Create DataLoaders
batch_size = 100
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

# Define Transformer encoder model
model = TransformerEncoderModel(
    input_dim=input_dim,
    d_model=128,
    nhead=8,
    num_layers=6,
    dim_feedforward=512,
    dropout=0.0,
    output_size=2
).to(device)

# Define loss function and optimizer
criterion = nn.SmoothL1Loss()
learning_rate = 0.001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Learning rate scheduler to reduce LR when validation loss doesn't improve
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

# Number of epochs
num_epochs = 100

# Training loop
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    
    # Training phase
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

    # Step the scheduler based on validation loss
    scheduler.step(val_loss)
    # early_stopping(val_loss)

    current_lr = optimizer.param_groups[0]['lr']
    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {train_loss/len(train_loader):.12f}, "
          f"Val Loss: {val_loss:.12f}, LR: {current_lr:.12f}")

    if early_stopping.early_stop:
        print("Early stopping triggered!")
        break

# Save the trained model
model_save_path = model_name
torch.save(model.state_dict(), model_save_path)
print(f"Model saved to {model_save_path}")

# Load the model back
model_save_path = model_name
model_load_path = model_save_path
loaded_model = TransformerEncoderModel(
    input_dim=input_dim,
    d_model=128,
    nhead=8,
    num_layers=6,
    dim_feedforward=512,
    dropout=0.0,
    output_size=2
).to(device)

loaded_model.load_state_dict(torch.load(model_load_path, map_location=device))
loaded_model.eval()
print(f"Model loaded from {model_load_path}")

# Prepare data for inference
X_tensor_seq = X_tensor_seq  # [num_sequences, seq_length, input_dim]
# Only the last element in the sequence is used as label
y_tensor_seq = y_tensor_seq[:, -1, :]  # shape: [num_sequences, 2]

dataset = TensorDataset(X_tensor_seq, y_tensor_seq)
val_loader_seq = DataLoader(dataset, batch_size=1, shuffle=False)

y_pred_list = []
y_actual_list = []

# Run inference on each sequence
with torch.no_grad():
    for batch_X, batch_y in val_loader_seq:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)
        pred = loaded_model(batch_X)
        # Collect the prediction of the last element
        y_pred_list.append(pred[:, -1, :].cpu().numpy())
        y_actual_list.append(batch_y.cpu().numpy())

# Concatenate predictions and ground truths
y_pred_seq = np.concatenate(y_pred_list, axis=0)   # shape: [num_val_seq, 2]
y_actual_seq = np.concatenate(y_actual_list, axis=0)  # shape: [num_val_seq, 2]

# Flatten the original gait phase for plotting
y_phase_val = y_phase_original.reshape(-1).numpy()

# Convert (cos, sin) to angles in [0,1]
pred_angle = np.arctan2(y_pred_seq[:, 1], y_pred_seq[:, 0])
pred_phase = (pred_angle / (2 * math.pi)) % 1.0

actual_angle = np.arctan2(y_actual_seq[:, 1], y_actual_seq[:, 0])
actual_phase = (actual_angle / (2 * math.pi)) % 1.0

y_pred = pred_phase
y_actual = actual_phase

# Calculate circular distance
differences = np.abs(y_actual - y_pred)
max_difference_circular = np.max(differences)
min_difference_circular = np.min(differences)
mean_difference_circular = np.mean(differences)

print("------- Linear Distance with Offset Results -------")
print(f"Maximum Difference (circular): {max_difference_circular}")
print(f"Minimum Difference (circular): {min_difference_circular}")
print(f"Mean Difference (circular): {mean_difference_circular}")

# Visualization using shifted predictions
visible_time_range = 500
fig_width_cm = visible_time_range / 50
fig_height = 8
fig_width_inches = fig_width_cm / 2.54
fig_height_inches = fig_height / 2.54

fig, ax = plt.subplots(figsize=(fig_width_inches, fig_height_inches))
current_start_time = 0

def update_plot(start_time, seq_length):
    ax.clear()
    end_time = start_time + visible_time_range

    # Actual data in the specified range
    actual_data = y_phase_val[int(start_time):int(end_time)]
    
    # Construct a shifted prediction array
    predicted_data_shifted = [None] * (seq_length - 1) + list(y_pred)
    predicted_data = predicted_data_shifted[int(start_time):int(end_time)]

    # Plot the actual data (blue line)
    ax.plot(actual_data, label='Actual', color='blue', linestyle='-', linewidth=1.5)
    
    # Plot the shifted predictions (red line)
    ax.plot(predicted_data, label='Predicted', color='red', linestyle='-', linewidth=1.5)

    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Gait Cycle Value', fontsize=12)
    ax.set_title('Transformer encoder', fontsize=16)
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.5)
    
    plt.draw()

# Initial plot with seq_length
update_plot(current_start_time, seq_length)

# Create a slider for time scrolling
slider_ax_time = plt.axes([0.1, 0.02, 0.8, 0.03], facecolor='lightgrey')
slider_time = Slider(
    slider_ax_time,
    'Time Scroll',
    0,
    max(0, len(y_actual) - visible_time_range),
    valinit=0,
    valstep=1
)

def on_time_slider_update(val):
    global current_start_time
    current_start_time = slider_time.val
    update_plot(current_start_time, seq_length)

slider_time.on_changed(on_time_slider_update)
plt.tight_layout(rect=[0.05, 0.1, 0.95, 0.95]) 
plt.show()
