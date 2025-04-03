#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import cv2
import numpy as np
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional, Any
import random
from videomae_model import FencingTemporalModel
from torchvision.transforms import Compose, Resize, ToTensor, Normalize, CenterCrop
import glob
import re

class FencingVideoDataset(Dataset):
    """
    Dataset for loading fencing video clips, assuming labels are in filenames.
    Expects video files directly inside a 'videos' subdirectory.
    Automatically splits into train/val sets.
    """
    def __init__(self, data_path, mode='train', split_ratio=0.8, temporal_window=16, img_size=224):
        """
        Args:
            data_path: Path to the root dataset directory (containing 'videos').
            mode: 'train' or 'val' to select the dataset split.
            split_ratio: Ratio of data to use for training.
            temporal_window: Number of frames per clip.
            img_size: Size to resize frames to.
        """
        self.data_path = data_path
        self.mode = mode
        self.temporal_window = temporal_window
        self.img_size = img_size
        self.video_dir = os.path.join(data_path, 'videos')
        
        self.samples = []
        self.class_to_idx = {}
        self.idx_to_class = {}
        self._load_data(split_ratio)
        
        # Define transformations
        self.transform = Compose([
            ToTensor(), # Converts images to [C, H, W] and scales to [0, 1]
            Resize((self.img_size, self.img_size), antialias=True),
            Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # Imagenet stats
        ])
        
    def _load_data(self, split_ratio):
        """Load video paths and extract labels from filenames, then split."""
        if not os.path.isdir(self.video_dir):
            print(f"Error: Video directory not found: {self.video_dir}")
            return
            
        video_files = glob.glob(os.path.join(self.video_dir, '*.avi')) + \
                      glob.glob(os.path.join(self.video_dir, '*.mp4'))
        
        if not video_files:
            print(f"Warning: No video files found in {self.video_dir}")
            return

        # Extract labels from filenames and build class mapping
        all_samples = []
        current_idx = 0
        for video_path in video_files:
            filename = os.path.basename(video_path)
            label_match = re.match(r"^([a-zA-Z0-9_\-]+)\.", filename)
            if label_match:
                label = label_match.group(1).lower() # Extract label (part before extension) and lowercase
                if label not in self.class_to_idx:
                    self.class_to_idx[label] = current_idx
                    self.idx_to_class[current_idx] = label
                    current_idx += 1
                all_samples.append((video_path, self.class_to_idx[label]))
            else:
                print(f"Warning: Could not parse label from filename: {filename}")

        # Shuffle and split
        random.shuffle(all_samples)
        split_idx = int(len(all_samples) * split_ratio)
        
        if self.mode == 'train':
            self.samples = all_samples[:split_idx]
            print(f"Loaded {len(self.samples)} samples for train from {len(all_samples)} total videos.")
        elif self.mode == 'val':
            self.samples = all_samples[split_idx:]
            print(f"Loaded {len(self.samples)} samples for val from {len(all_samples)} total videos.")
        else:
            raise ValueError(f"Invalid mode: {self.mode}. Choose 'train' or 'val'.")
            
        if not self.samples:
             print(f"Warning: No samples loaded for mode '{self.mode}'. Check data path and filenames.")
             
        print(f"Found {len(self.class_to_idx)} classes: {list(self.class_to_idx.keys())}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        video_path, label_idx = self.samples[idx]
        
        try:
            # Load video frames
            frames = self._load_video_frames(video_path)
            
            # Select temporal window
            clip = self._select_temporal_clip(frames)
            
            # Apply transformations
            transformed_clip = torch.stack([self.transform(frame) for frame in clip])
            
            # Permute to [C, T, H, W] for 3D ConvNet
            transformed_clip = transformed_clip.permute(1, 0, 2, 3)
            
            return transformed_clip, label_idx
        except Exception as e:
            print(f"Error loading sample {idx} ({video_path}): {e}")
            # Return a dummy sample or raise error, depending on desired robustness
            # For simplicity, return None and handle in DataLoader collation
            return None, None 

    def _load_video_frames(self, video_path):
        """Load all frames from a video file."""
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        cap.release()
        if not frames:
             raise IOError(f"Could not load any frames from {video_path}")
        return frames

    def _select_temporal_clip(self, frames):
        """Select a clip of temporal_window frames from the video."""
        num_frames = len(frames)
        
        if num_frames < self.temporal_window:
            # Pad by repeating the last frame
            padded_frames = frames + [frames[-1]] * (self.temporal_window - num_frames)
            return padded_frames
        elif num_frames == self.temporal_window:
            return frames
        else:
            # Randomly select a starting point
            start_idx = random.randint(0, num_frames - self.temporal_window)
            return frames[start_idx : start_idx + self.temporal_window]


def train_model(
    model: FencingTemporalModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    args: argparse.Namespace
) -> FencingTemporalModel:
    """
    Train the temporal model
    
    Args:
        model: The model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        args: Training arguments
        
    Returns:
        Trained model
    """
    device = args.device
    model = model.to(device)
    
    # Loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=args.lr_min)
    
    # Training loop
    best_val_acc = 0.0
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    
    for epoch in range(args.epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch_idx, (inputs, targets) in enumerate(pbar):
            inputs, targets = inputs.to(device), targets.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            # Update metrics
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            # Update progress bar
            avg_loss = train_loss / (batch_idx + 1)
            acc = 100.0 * train_correct / train_total
            pbar.set_postfix({"Loss": f"{avg_loss:.4f}", "Acc": f"{acc:.2f}%"})
        
        # Calculate epoch metrics
        train_loss = train_loss / len(train_loader)
        train_acc = 100.0 * train_correct / train_total
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for inputs, targets in tqdm(val_loader, desc="Validation"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                # Forward pass
                outputs = model(inputs)
                loss = criterion(outputs, targets)
                
                # Update metrics
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        # Calculate epoch metrics
        val_loss = val_loss / len(val_loader)
        val_acc = 100.0 * val_correct / val_total
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        # Update learning rate
        scheduler.step()
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{args.epochs} - "
              f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, "
              f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.output_dir, f"best_temporal_model_epoch{epoch+1}.pth")
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model to {save_path} with val_acc: {val_acc:.2f}%")
        
        # Save latest model
        save_path = os.path.join(args.output_dir, "latest_temporal_model.pth")
        torch.save(model.state_dict(), save_path)
    
    # Plot training curves
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss Curves')
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Accuracy Curves')
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'))
    
    # Save training history
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_acc': train_accs,
        'val_acc': val_accs,
    }
    
    with open(os.path.join(args.output_dir, 'training_history.json'), 'w') as f:
        json.dump(history, f)
    
    return model


def main():
    """Main function to run the training script"""
    parser = argparse.ArgumentParser(description='Train the temporal model for fencing analysis')
    parser.add_argument('--data_path', required=True, help='Path to the dataset')
    parser.add_argument('--output_dir', default='models', help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=25, help='Number of epochs')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--lr_min', type=float, default=1e-5, help='Minimum learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
    parser.add_argument('--temporal_window', type=int, default=16, help='Temporal window size')
    parser.add_argument('--img_size', type=int, default=224, help='Image size')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {args.device}")
    
    # Create datasets
    train_dataset = FencingVideoDataset(
        data_path=args.data_path,
        mode='train',
        temporal_window=args.temporal_window,
        img_size=args.img_size
    )
    
    val_dataset = FencingVideoDataset(
        data_path=args.data_path,
        mode='val',
        temporal_window=args.temporal_window,
        img_size=args.img_size
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model
    num_classes = len(train_dataset.class_to_idx)
    model = FencingTemporalModel(
        model_path=None,  # Start from scratch
        num_classes=num_classes,
        temporal_size=args.temporal_window,
        img_size=args.img_size,
        device=args.device
    )
    
    # Save class mapping
    class_mapping = {
        'idx_to_class': train_dataset.idx_to_class,
        'class_to_idx': train_dataset.class_to_idx
    }
    
    with open(os.path.join(args.output_dir, 'class_mapping.json'), 'w') as f:
        json.dump(class_mapping, f)
    
    # Train model
    print(f"Starting training for {args.epochs} epochs")
    trained_model = train_model(model, train_loader, val_loader, args)
    
    # Save final model
    final_path = os.path.join(args.output_dir, 'final_temporal_model.pth')
    torch.save(trained_model.state_dict(), final_path)
    print(f"Training complete. Final model saved to {final_path}")


if __name__ == "__main__":
    main() 