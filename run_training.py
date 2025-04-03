#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import torch
from train_temporal_model import main as train_temporal_model_main
from enhanced_fencer_detector import SwordDetector

def main():
    """Main function to run the training script with improved parameters"""
    parser = argparse.ArgumentParser(description='Run improved training for fencing analysis models')
    parser.add_argument('--data_path', default='fencingdataset/organized_videos', 
                        help='Path to the organized dataset')
    parser.add_argument('--output_dir', default='models/retrained', 
                        help='Directory to save models')
    parser.add_argument('--epochs', type=int, default=20, 
                        help='Number of epochs for training')
    parser.add_argument('--data_augmentation', action='store_true', 
                        help='Use data augmentation')
    parser.add_argument('--sword_conf_threshold', type=float, default=0.3,
                        help='Confidence threshold for sword detection')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Print training configuration
    print("=" * 50)
    print("Running improved training with the following configuration:")
    print(f"  Dataset path: {args.data_path}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Epochs: {args.epochs}")
    print(f"  Data augmentation: {args.data_augmentation}")
    print(f"  Sword confidence threshold: {args.sword_conf_threshold}")
    print("=" * 50)
    
    # Train temporal model
    print("\nTraining temporal model...")
    train_args = [
        "--data_path", args.data_path,
        "--output_dir", args.output_dir,
        "--epochs", str(args.epochs),
        "--learning_rate", "0.0001",
        "--weight_decay", "0.001",
        "--temporal_window", "16"
    ]
    
    if args.data_augmentation:
        train_args.append("--data_augmentation")
    
    # Convert args list to argv format expected by train_temporal_model_main
    import sys
    original_argv = sys.argv
    sys.argv = [original_argv[0]] + train_args
    
    try:
        train_temporal_model_main()
        print("\nTemporal model training completed.")
    except Exception as e:
        print(f"\nError during temporal model training: {e}")
    finally:
        sys.argv = original_argv
    
    print("\nTraining completed!")
    print(f"\nFinal models saved to {args.output_dir}")
    print("\nTo use the retrained model, run the analyzer with:")
    print(f"python integrated_fencing_analyzer.py \"evenevenmorecropped (1).mp4\" --output_dir test_retrained --bout_mode --temporal_model {args.output_dir}/final_temporal_model.pth")
    
if __name__ == "__main__":
    main() 