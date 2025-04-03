import os
import argparse
import torch
import cv2
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path

# Import our CNN model and dataset utilities
from fencing_cnn import (
    FencingPoseClassifier,
    create_dataset_from_yolo_dataset,
    create_dataloader_from_directory
)

# Import the enhanced detector
from enhanced_fencer_detector import EnhancedFencerDetector, train_sword_detector

def train_model(dataset_path=None, epochs=10, batch_size=16, lr=0.001):
    """
    Train the fencing pose classifier
    
    Args:
        dataset_path: Path to dataset directory. If None, will create one from YOLOv8 dataset.
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Learning rate
        
    Returns:
        model_path: Path to trained model
    """
    print("Training Fencing CNN model...")
    
    # If no dataset path provided, create from YOLOv8 dataset
    if dataset_path is None:
        print("Preparing dataset from YOLOv8 annotations...")
        dataset_path = 'fencing_pose_dataset'
        
        # Check if dataset directory exists
        if not os.path.exists(dataset_path):
            print(f"Creating dataset at {dataset_path}...")
            
            # Default dataset path
            yolo_dataset_path = 'fencingdataset/Fencer Detection.v7i.yolov8-obb'
            
            # Create dataset
            for split in ['train', 'valid']:
                create_dataset_from_yolo_dataset(
                    dataset_path=yolo_dataset_path,
                    output_path=dataset_path,
                    split=split
                )
        else:
            print(f"Using existing dataset at {dataset_path}")
    
    # Create model directory
    os.makedirs('models', exist_ok=True)
    
    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader = create_dataloader_from_directory(
        data_dir=dataset_path,
        batch_size=batch_size,
        train_ratio=0.8
    )
    
    # Initialize classifier
    print("Initializing model...")
    classifier = FencingPoseClassifier()
    
    # Train model
    print(f"Training model for {epochs} epochs...")
    model_path = 'models/pose_classifier.pth'
    classifier.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        lr=lr,
        save_path=model_path
    )
    
    print(f"Model trained and saved to {model_path}")
    return model_path


def train_sword_model(epochs=50, batch_size=16):
    """
    Train the sword detector model using YOLOv8
    
    Args:
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        model_path: Path to trained model
    """
    print("Training Sword Detection Model...")
    
    # Default dataset path
    dataset_path = 'fencingdataset/Fencing Blade.v8i.yolov8-obb'
    
    # Verify dataset exists
    data_yaml = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(data_yaml):
        print(f"Error: Dataset YAML not found at {data_yaml}")
        return None
    
    # Train sword detector
    model_path = train_sword_detector(
        dataset_path=dataset_path,
        epochs=epochs,
        batch_size=batch_size
    )
    
    return model_path


def test_model_on_image(image_path, model_path=None, sword_model_path=None, output_path=None):
    """
    Test the trained model on a single image
    
    Args:
        image_path: Path to input image
        model_path: Path to trained pose classifier model
        sword_model_path: Path to trained sword detector model
        output_path: Path to save output image
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Could not read image {image_path}")
        return
    
    # Initialize detector
    detector = EnhancedFencerDetector(
        pose_model_path=model_path,
        sword_model_path=sword_model_path
    )
    
    # Detect and classify
    tracked_items, sword_detections = detector.track_and_classify(image)
    
    # Draw results
    annotated_image = detector.draw_enhanced_detections(
        image.copy(), 
        tracked_items, 
        sword_detections
    )
    
    # Save or display result
    if output_path:
        cv2.imwrite(output_path, annotated_image)
        print(f"Saved result to {output_path}")
    else:
        # Display with matplotlib
        plt.figure(figsize=(12, 8))
        plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
        plt.axis('off')
        plt.title("Enhanced Fencer Detection with Pose and Sword Classification")
        plt.show()


def test_model_on_video(video_path, model_path=None, sword_model_path=None, output_path=None, max_frames=None):
    """
    Test the trained model on a video
    
    Args:
        video_path: Path to input video
        model_path: Path to trained pose classifier model
        sword_model_path: Path to trained sword detector model
        output_path: Path to save output video
        max_frames: Maximum number of frames to process
    """
    # Set default output path if not specified
    if not output_path:
        output_path = f"enhanced_{os.path.basename(video_path)}"
    
    # Initialize detector
    detector = EnhancedFencerDetector(
        pose_model_path=model_path,
        sword_model_path=sword_model_path
    )
    
    # Process video
    print(f"Processing video {video_path}...")
    results = detector.process_video(
        video_path=video_path,
        output_path=output_path,
        max_frames=max_frames
    )
    
    print(f"Video processed and saved to {output_path}")
    
    # Plot pose statistics
    plot_stats(results, output_path.replace('.mp4', '_stats.png'))


def plot_stats(results, output_path=None):
    """
    Plot fencer pose and sword statistics
    
    Args:
        results: Dictionary containing 'pose_stats' and 'sword_stats'
        output_path: Path to save output image
    """
    pose_stats = results.get('pose_stats', {})
    sword_stats = results.get('sword_stats', {})
    
    # Create a figure with subplots
    num_fencers = len(pose_stats)
    fig = plt.figure(figsize=(15, 10))
    
    # Define grid layout
    if num_fencers > 0:
        # Create grid: one row for fencers, one row for swords
        grid = plt.GridSpec(2, max(num_fencers, 1), height_ratios=[2, 1])
        
        # Plot pose stats for each fencer
        for i, (fencer_id, stats) in enumerate(pose_stats.items()):
            # Extract pose counts (excluding 'total_frames')
            poses = []
            counts = []
            
            for pose, count in sorted([(k, v) for k, v in stats.items() if k != 'total_frames'], 
                                    key=lambda x: x[1], reverse=True):
                if count > 0:
                    poses.append(pose)
                    counts.append(count)
            
            # Define colors for poses
            colors = {
                'neutral': 'gray',
                'attack': 'red',
                'defense': 'green',
                'lunge': 'blue'
            }
            
            # Get colors for poses
            pose_colors = [colors.get(pose, 'gray') for pose in poses]
            
            # Create fencer pie chart
            ax = fig.add_subplot(grid[0, i])
            if poses:
                ax.pie(
                    counts,
                    labels=poses,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=pose_colors
                )
                ax.set_title(f"Fencer {fencer_id} Pose Distribution")
            else:
                ax.text(0.5, 0.5, 'No pose data', ha='center', va='center')
                ax.axis('off')
    
    # Plot sword statistics
    ax_sword = fig.add_subplot(grid[1, :])
    
    # Extract sword counts
    sword_parts = []
    sword_counts = []
    
    for part, count in sorted([(k, v) for k, v in sword_stats.items() if k != 'total_frames'], 
                            key=lambda x: x[1], reverse=True):
        if count > 0:
            sword_parts.append(part)
            sword_counts.append(count)
    
    # Define colors for sword parts
    sword_colors = {
        'blade-guard': 'orange',
        'blade-tip': 'red',
        'fencing-blade': 'blue'
    }
    
    if sword_parts:
        ax_sword.bar(
            sword_parts,
            sword_counts,
            color=[sword_colors.get(part, 'gray') for part in sword_parts]
        )
        ax_sword.set_xlabel('Sword Parts')
        ax_sword.set_ylabel('Detections')
        ax_sword.set_title('Sword Detection Statistics')
        ax_sword.tick_params(axis='x', rotation=45)
    else:
        ax_sword.text(0.5, 0.5, 'No sword detections', ha='center', va='center')
        ax_sword.axis('off')
    
    plt.suptitle("Fencing Analysis", fontsize=16)
    plt.tight_layout()
    
    # Save or display plot
    if output_path:
        plt.savefig(output_path)
        print(f"Saved statistics plot to {output_path}")
    else:
        plt.show()


def find_sample_images():
    """
    Find sample images to test the model
    
    Returns:
        List of sample image paths
    """
    # Look for sample images in dataset
    dataset_path = 'fencingdataset/Fencer Detection.v7i.yolov8-obb/train/images'
    
    if os.path.exists(dataset_path):
        # Get a few sample images
        image_paths = [os.path.join(dataset_path, f) for f in os.listdir(dataset_path) 
                      if f.endswith(('.jpg', '.jpeg', '.png'))]
        
        # Return up to 5 sample images
        return image_paths[:5]
    
    return []


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train and test Fencing CNN model")
    parser.add_argument("--mode", choices=['train_pose', 'train_sword', 'train_both', 'test_image', 'test_video'], 
                      default='train_both', help="Mode: train_pose, train_sword, train_both, test_image, test_video")
    parser.add_argument("--dataset", default=None,
                      help="Path to dataset directory for pose classifier")
    parser.add_argument("--pose_model", default=None,
                      help="Path to trained pose classifier model")
    parser.add_argument("--sword_model", default=None,
                      help="Path to trained sword detector model")
    parser.add_argument("--input", default=None,
                      help="Path to input image or video")
    parser.add_argument("--output", default=None,
                      help="Path to save output image or video")
    parser.add_argument("--pose_epochs", type=int, default=10,
                      help="Number of training epochs for pose classifier")
    parser.add_argument("--sword_epochs", type=int, default=50,
                      help="Number of training epochs for sword detector")
    parser.add_argument("--batch_size", type=int, default=16,
                      help="Batch size for training")
    parser.add_argument("--lr", type=float, default=0.001,
                      help="Learning rate for pose classifier")
    parser.add_argument("--max_frames", type=int, default=None,
                      help="Maximum number of frames to process in video")
    
    args = parser.parse_args()
    
    if args.mode == 'train_pose':
        # Train pose classifier
        pose_model_path = train_model(
            dataset_path=args.dataset,
            epochs=args.pose_epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        
        print(f"Pose classifier trained and saved to {pose_model_path}")
    
    elif args.mode == 'train_sword':
        # Train sword detector
        sword_model_path = train_sword_model(
            epochs=args.sword_epochs,
            batch_size=args.batch_size
        )
        
        print(f"Sword detector trained and saved to {sword_model_path}")
    
    elif args.mode == 'train_both':
        # Train both models
        print("=== Training Pose Classifier ===")
        pose_model_path = train_model(
            dataset_path=args.dataset,
            epochs=args.pose_epochs,
            batch_size=args.batch_size,
            lr=args.lr
        )
        
        print("\n=== Training Sword Detector ===")
        sword_model_path = train_sword_model(
            epochs=args.sword_epochs,
            batch_size=args.batch_size
        )
        
        print("\n=== Training Complete ===")
        print(f"Pose classifier trained and saved to {pose_model_path}")
        print(f"Sword detector trained and saved to {sword_model_path}")
        
        # Test models on sample images if available
        sample_images = find_sample_images()
        if sample_images and os.path.exists(pose_model_path) and os.path.exists(sword_model_path):
            print("\nTesting models on sample images...")
            for img_path in sample_images:
                print(f"Testing on {img_path}...")
                test_model_on_image(
                    image_path=img_path,
                    model_path=pose_model_path,
                    sword_model_path=sword_model_path,
                    output_path=f"test_{os.path.basename(img_path)}"
                )
    
    elif args.mode == 'test_image':
        if not args.input:
            print("Error: Please provide input image path with --input")
            exit(1)
        
        # Test model on image
        test_model_on_image(
            image_path=args.input,
            model_path=args.pose_model,
            sword_model_path=args.sword_model,
            output_path=args.output
        )
    
    elif args.mode == 'test_video':
        if not args.input:
            print("Error: Please provide input video path with --input")
            exit(1)
        
        # Test model on video
        test_model_on_video(
            video_path=args.input,
            model_path=args.pose_model,
            sword_model_path=args.sword_model,
            output_path=args.output,
            max_frames=args.max_frames
        )
    
    else:
        print(f"Error: Unknown mode {args.mode}")
        exit(1) 