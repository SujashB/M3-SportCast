import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import yaml
from tqdm import tqdm

def create_dataset_structure():
    """Create the dataset directory structure"""
    dirs = [
        'dataset/images/train',
        'dataset/images/val',
        'dataset/labels/train',
        'dataset/labels/val'
    ]
    for dir_path in dirs:
        Path(dir_path).mkdir(parents=True, exist_ok=True)

def extract_frames(video_path, output_dir, max_frames=None):
    """Extract frames from video"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    frame_count = 0
    while cap.isOpened() and (max_frames is None or frame_count < max_frames):
        ret, frame = cap.read()
        if not ret:
            break
        
        # Save frame
        frame_path = os.path.join(output_dir, f"frame_{frame_count:06d}.jpg")
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    
    cap.release()
    return frame_count

def create_yaml():
    """Create YAML configuration file for YOLOv8 training"""
    yaml_content = {
        'path': 'dataset',  # dataset root dir
        'train': 'images/train',  # train images (relative to 'path')
        'val': 'images/val',  # val images (relative to 'path')
        'names': {0: 'fencer'},  # class names
        'nc': 1  # number of classes
    }
    
    with open('dataset/data.yaml', 'w') as f:
        yaml.dump(yaml_content, f)

def main():
    # Create dataset structure
    create_dataset_structure()
    
    # Extract frames from the video
    video_path = "evenevenmorecropped (1).mp4"
    train_dir = "dataset/images/train"
    val_dir = "dataset/images/val"
    
    print("Extracting frames from video...")
    total_frames = extract_frames(video_path, train_dir)
    
    # Move some frames to validation set (10%)
    val_frames = total_frames // 10
    for i in range(val_frames):
        src = os.path.join(train_dir, f"frame_{i:06d}.jpg")
        dst = os.path.join(val_dir, f"frame_{i:06d}.jpg")
        shutil.move(src, dst)
    
    # Create YAML configuration
    create_yaml()
    
    print("\nDataset preparation complete!")
    print(f"Total frames: {total_frames}")
    print(f"Training frames: {total_frames - val_frames}")
    print(f"Validation frames: {val_frames}")
    print("\nNext steps:")
    print("1. Label the frames using a tool like CVAT or LabelImg")
    print("2. Place the label files in dataset/labels/train and dataset/labels/val")
    print("3. Train the model using: python train_fencer_detector.py")

if __name__ == "__main__":
    main() 