import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import argparse

from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataclasses import dataclass
from typing import List, Optional

from model.lstm import LSTMModel
from utils.augmentation import IMUAugmentor
from utils.causal_filter import butterworth_filter
from utils.phase_conversion import phase_to_sincos, sincos_to_phase

@dataclass
class TrainingConfig:
    # training data path
    data_paths: List[str] = None
    
    result_dir: str = './results/default'
    
    features_path: Optional[str] = None
    labels_path: Optional[str] = None
    model_save_path: Optional[str] = None
    
    sequence_length: int = 10
    
    filtered_columns = [
        'imu_acc_x', 
        'imu_acc_y',
        'imu_acc_z',
        'imu_ang_vel_x',
        'imu_ang_vel_y',
        'imu_ang_vel_z'
    ]
    
    # signal processing parameters
    sampling_rate: int = 100
    cutoff_freq: int = 5
    
    # clip limits for IMU data
    imu_acc_x_clip: tuple = (-25, 40)
    imu_acc_y_clip: tuple = (-20, 40)
    imu_ang_vel_z_clip: tuple = (-3, 4)
    
    # training parameters
    batch_size: int = 512
    num_epochs: int = 100
    learning_rate: float = 0.01
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    
    # RMSprop specific parameters
    alpha: float = 0.99    # smoothing constant
    eps: float = 1e-8      # term added to denominator for numerical stability
    momentum: float = 0    # momentum factor
    
    # Learning rate scheduler parameters
    scheduler_mode: str = 'min'
    scheduler_factor: float = 0.2
    scheduler_patience: int = 6
    scheduler_threshold: float = 1e-3
    scheduler_threshold_mode: str = 'rel'
    scheduler_cooldown: int = 0
    scheduler_min_lr: float = 1e-8
    
    def __post_init__(self):
        if self.data_paths is None:
            self.data_paths = [
                './motion_data_with_ground_truth_-0.2_15.csv',
                './motion_data_with_ground_truth_-0.2_30.csv',
                './motion_data_with_ground_truth_-0.2_45.csv',
                './motion_data_with_ground_truth_-0.25_15.csv',
                './motion_data_with_ground_truth_-0.25_30.csv',
                './motion_data_with_ground_truth_-0.25_45.csv',
                './motion_data_with_ground_truth_-0.3_15.csv',
                './motion_data_with_ground_truth_-0.3_30.csv',
                './motion_data_with_ground_truth_-0.3_45.csv',
                './motion_data_with_ground_truth_-0.35_15.csv',
                './motion_data_with_ground_truth_-0.35_30.csv',
                './motion_data_with_ground_truth_-0.35_45.csv',
            ]
            
        os.makedirs(self.result_dir, exist_ok=True)
        
        # self.features_path = os.path.join(self.result_dir, 'saved_features.pt')
        # self.labels_path = os.path.join(self.result_dir, 'saved_labels.pt')
        self.model_save_path = os.path.join(self.result_dir, 'lstm_model.pth')
        
class GaitDataset(Dataset):
    def __init__(self, features, labels, sequence_length=100):
        """
        Args:
            features: (n_samples, n_features)
            labels: (n_samples,)
            sequence_length
        """
        self.sequence_length = sequence_length
        
        # create sliding windows
        self.sequences = []
        self.sequence_labels = []
        
        sincos_labels = phase_to_sincos(labels)
        
        for i in range(len(features) - sequence_length + 1):
            self.sequences.append(features[i:i + sequence_length])
            self.sequence_labels.append(sincos_labels[i + sequence_length - 1])
            
        self.sequences = np.array(self.sequences)
        self.sequence_labels = np.array(self.sequence_labels)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.sequences[idx]), torch.FloatTensor(self.sequence_labels[idx])

def preprocess_data(data_paths: List[str], filtered_columns: List[str], config: TrainingConfig):
    """
    Preprocess multiple CSV files while maintaining data separation
        
    Returns:
        Tuple[List[np.ndarray], List[np.ndarray]]: Lists of features and labels for each file
    """
    all_features = []
    all_labels = []
    
    for data_path in data_paths:
        # Read and process each file separately
        print(f"Processing {data_path}")
        data = pd.read_csv(data_path)
        
        # # Clip acceleration data using config parameters
        # data['imu_acc_x'] = data['imu_acc_x'].clip(*config.imu_acc_x_clip)
        # data['imu_acc_y'] = data['imu_acc_y'].clip(*config.imu_acc_y_clip)
        # data['imu_ang_vel_z'] = data['imu_ang_vel_z'].clip(*config.imu_ang_vel_z_clip)
        
        # Apply low-pass filter to specified columns
        for col in filtered_columns:
            data[col] = butterworth_filter(data[col].values, config.cutoff_freq, config.sampling_rate)

        # Prepare features and labels for this file
        features = data[filtered_columns].values
        labels = data['gait_phase'].values
        
        # Store processed data for this file
        all_features.append(features)
        all_labels.append(labels)
    
    return all_features, all_labels

def prepare_data(features_list: List[np.ndarray], labels_list: List[np.ndarray], config: TrainingConfig):
    """
    Prepare data loaders while maintaining separation between different CSV files
    
    Args:
        features_list: List of feature arrays, one per file
        labels_list: List of label arrays, one per file
        config: Training configuration
        
    Returns:
        Tuple[DataLoader, DataLoader]: Train and validation data loaders
    """
    all_train_datasets = []
    all_val_datasets = []
    
    # Process each file's data separately
    for features, labels in zip(features_list, labels_list):
        # # Optional data augmentation
        # augmentor = IMUAugmentor(seed=42)
        # features = augmentor.augment_data(
        #     features,
        #     num_augmentations=2,
        #     methods=['gaussian_noise', 'random_bias', 'sensor_drift', 'magnitude_scaling', 'spike_noise'],
        #     probabilities=[0.2, 0.2, 0.2, 0.2, 0.2]
        # )
        
        # Split into train/val for this file
        total_samples = len(features)
        train_size = int(0.8 * total_samples)
        
        train_features = features[:train_size]
        train_labels = labels[:train_size]
        val_features = features[train_size:]
        val_labels = labels[train_size:]
        
        # Create datasets with sliding windows for this file
        train_dataset = GaitDataset(train_features, train_labels, config.sequence_length)
        val_dataset = GaitDataset(val_features, val_labels, config.sequence_length)
        
        all_train_datasets.append(train_dataset)
        all_val_datasets.append(val_dataset)
    
    # Concatenate all datasets
    train_dataset = torch.utils.data.ConcatDataset(all_train_datasets)
    val_dataset = torch.utils.data.ConcatDataset(all_val_datasets)
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=False)
    
    # Save combined features and labels for later use in predictions
    all_features = np.concatenate(features_list)
    all_labels = np.concatenate(labels_list)
    
    torch.save(torch.FloatTensor(all_features), config.features_path)
    torch.save(torch.FloatTensor(all_labels), config.labels_path)
    
    print(f"Total number of sequences in training set: {len(train_dataset):,}")
    print(f"Total number of sequences in validation set: {len(val_dataset):,}")
    
    return train_loader, val_loader

def train_model(model, train_loader, val_loader, config: TrainingConfig):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)
    criterion = nn.MSELoss()
    # criterion = nn.SmoothL1Loss()
    # criterion = nn.HuberLoss()
    # optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    optimizer = torch.optim.RMSprop(
        model.parameters(),
        lr=config.learning_rate,
        alpha=config.alpha,
        eps=config.eps,
        momentum=config.momentum
    )
    
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode=config.scheduler_mode,
        factor=config.scheduler_factor,
        patience=config.scheduler_patience,
        threshold=config.scheduler_threshold,
        threshold_mode=config.scheduler_threshold_mode,
        cooldown=config.scheduler_cooldown,
        min_lr=config.scheduler_min_lr
    )

    best_val_loss = float('inf')

    # Print dataset sizes
    print("\nDataset Information:")
    print("-" * 30)
    print(f"Training set size: {len(train_loader.dataset):,} sequences")
    print(f"Validation set size: {len(val_loader.dataset):,} sequences")
    print(f"Batch size: {config.batch_size}")
    print(f"Steps per epoch: {len(train_loader):,}")
    print("-" * 30 + "\n")

    for epoch in range(config.num_epochs):
        model.train()
        train_loss = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        print(f'Epoch {epoch + 1}/{config.num_epochs}:')
        print(f'Training Loss: {train_loss:.6f}')
        print(f'Validation Loss: {val_loss:.6f}')
        print(f'Learning Rate: {current_lr:.8f}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'loss': best_val_loss,
            }, config.model_save_path)
            print(f'Saved model checkpoint (best validation loss: {best_val_loss:.6f})')

def main():
    parser = argparse.ArgumentParser(description='LSTM training')
    parser.add_argument('--result_dir', type=str, required=True,
                       help='result save directory')
    parser.add_argument('--data_paths', type=str, nargs='+',
                       help='training data path, one or multiple csv files')
    args = parser.parse_args()
    
    # Create configuration
    config = TrainingConfig()
    config.result_dir = args.result_dir
    if args.data_paths:
        config.data_paths = args.data_paths
    config.__post_init__()
    
    print("Starting training process...")
    print(f"Using data from {len(config.data_paths)} files")
    
    # Preprocess data
    features_list, labels_list = preprocess_data(config.data_paths, config.filtered_columns, config)
    
    # Prepare data loaders
    train_loader, val_loader = prepare_data(features_list, labels_list, config)

    # Initialize and train model
    model = LSTMModel(input_size=len(config.filtered_columns))
    train_model(model, train_loader, val_loader, config)

    print("\nTraining completed!")
    print(f"Model saved to: {config.model_save_path}")
    
if __name__ == "__main__":
    main()