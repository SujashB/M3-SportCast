import os
import cv2
import numpy as np
from pathlib import Path
import shutil
import json
from tqdm import tqdm

def create_cvat_structure():
    """Create directory structure for CVAT preparation"""
    dirs = ['cvat_preparation']
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

def create_cvat_task_json():
    """Create CVAT task configuration JSON"""
    task_config = {
        "name": "Fencer Detection Dataset",
        "labels": [
            {
                "name": "fencer",
                "color": "#FF0000",
                "attributes": []
            }
        ],
        "segments": [
            {
                "start": 0,
                "stop": 150,
                "subset": "train"
            }
        ],
        "image_quality": 100,
        "overlap": 0,
        "segment_size": 150,
        "z_order": False,
        "frame_filter": "",
        "frame_step": 1,
        "task_size": 150,
        "start_frame": 0,
        "stop_frame": 150,
        "frame_count": 150,
        "image_quality": 100,
        "use_zip_chunks": True,
        "use_cache": True,
        "source_storage": {
            "location": "local",
            "path": "cvat_preparation"
        },
        "target_storage": {
            "location": "local",
            "path": "cvat_preparation"
        }
    }
    
    with open('cvat_preparation/task.json', 'w') as f:
        json.dump(task_config, f, indent=2)

def main():
    # Create directory structure
    create_cvat_structure()
    
    # Extract frames from video
    video_path = "evenevenmorecropped (1).mp4"
    output_dir = "cvat_preparation"
    
    print("Extracting frames from video...")
    total_frames = extract_frames(video_path, output_dir)
    
    # Create CVAT task configuration
    create_cvat_task_json()
    
    print("\nCVAT preparation complete!")
    print(f"Total frames: {total_frames}")
    print("\nNext steps:")
    print("1. Go to https://app.cvat.ai/")
    print("2. Create a new task")
    print("3. Upload the frames from cvat_preparation directory")
    print("4. Label the fencers in each frame")
    print("5. Export the annotations in YOLO format")
    print("6. Place the exported labels in dataset/labels/train and dataset/labels/val")
    print("7. Train the model using: python train_fencer_detector.py")

if __name__ == "__main__":
    main() 