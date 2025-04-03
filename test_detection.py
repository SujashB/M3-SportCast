#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import argparse
import cv2
import numpy as np
import torch
from videomae_model import FencingTemporalModel
from integrated_fencing_analyzer import IntegratedFencingAnalyzer
import matplotlib.pyplot as plt
from tqdm import tqdm

def extract_frames(video_path, num_frames=None):
    """Extract frames from a video file"""
    cap = cv2.VideoCapture(video_path)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        if num_frames and len(frames) >= num_frames:
            break
    
    cap.release()
    return frames

def test_temporal_model(model_path, video_path, output_dir='test_results'):
    """Test the temporal model on a video file"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = FencingTemporalModel(
        model_path=model_path,
        temporal_size=16,
        img_size=224,
        device=device
    )
    
    # Extract frames from video
    frames = extract_frames(video_path)
    if not frames:
        print(f"Error: No frames extracted from {video_path}")
        return
    
    print(f"Extracted {len(frames)} frames from {video_path}")
    
    # Classify the sequence
    class_id, class_name, confidence = model.classify_sequence(frames)
    
    # Print results
    print("\nTemporal Model Results:")
    print(f"  Class: {class_name}")
    print(f"  Confidence: {confidence:.4f}")
    
    # Save a visualization
    height, width = frames[0].shape[:2]
    result_img = np.zeros((height + 100, width, 3), dtype=np.uint8)
    
    # Copy the middle frame
    middle_idx = len(frames) // 2
    result_img[:height, :width] = frames[middle_idx]
    
    # Add text with results
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(result_img, f"Class: {class_name}", (10, height + 40), 
                font, 1, (0, 255, 0), 2)
    cv2.putText(result_img, f"Confidence: {confidence:.4f}", (10, height + 80), 
                font, 1, (0, 255, 0), 2)
    
    # Save the result
    output_path = os.path.join(output_dir, "temporal_result.jpg")
    cv2.imwrite(output_path, result_img)
    
    print(f"Saved visualization to {output_path}")
    
    return class_name, confidence

def test_integrated_analyzer(model_path, video_path, output_dir='test_results'):
    """Test the integrated analyzer on a video file"""
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize analyzer
    analyzer = IntegratedFencingAnalyzer(
        temporal_model_path=model_path,
        bout_mode=True
    )
    
    # Run analysis
    results = analyzer.analyze_video(
        video_path=video_path,
        output_path=os.path.join(output_dir, "integrated_analysis.mp4"),
        save_visualization=True
    )
    
    # Print technique summary
    print("\nIntegrated Analysis Results:")
    if 'techniques' in results and results['techniques']:
        for i, technique in enumerate(results['techniques']):
            print(f"  Technique {i+1}: {technique['technique']} (Confidence: {technique['confidence']:.4f})")
    else:
        print("  No techniques detected")
    
    # Save a summary of the analysis
    summary_path = os.path.join(output_dir, "analysis_summary.txt")
    with open(summary_path, 'w') as f:
        f.write(f"Video: {video_path}\n")
        f.write(f"Model: {model_path}\n\n")
        
        f.write("Detected Techniques:\n")
        if 'techniques' in results and results['techniques']:
            for i, technique in enumerate(results['techniques']):
                f.write(f"  Technique {i+1}: {technique['technique']} (Confidence: {technique['confidence']:.4f})\n")
        else:
            f.write("  No techniques detected\n")
    
    print(f"Saved analysis summary to {summary_path}")
    
    return results

def main():
    """Main function to run testing"""
    parser = argparse.ArgumentParser(description='Test improved fencing analysis models')
    parser.add_argument('--model_path', required=True, 
                        help='Path to the trained temporal model')
    parser.add_argument('--video_path', required=True, 
                        help='Path to the test video')
    parser.add_argument('--output_dir', default='test_results', 
                        help='Directory to save results')
    parser.add_argument('--test_mode', choices=['temporal', 'integrated', 'both'], 
                        default='both', help='Which test to run')
    
    args = parser.parse_args()
    
    # Print test configuration
    print("=" * 50)
    print("Running model testing with the following configuration:")
    print(f"  Model path: {args.model_path}")
    print(f"  Test video: {args.video_path}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Test mode: {args.test_mode}")
    print("=" * 50)
    
    # Run tests
    if args.test_mode in ['temporal', 'both']:
        print("\nTesting temporal model...")
        class_name, confidence = test_temporal_model(
            model_path=args.model_path,
            video_path=args.video_path,
            output_dir=args.output_dir
        )
    
    if args.test_mode in ['integrated', 'both']:
        print("\nTesting integrated analyzer...")
        results = test_integrated_analyzer(
            model_path=args.model_path,
            video_path=args.video_path,
            output_dir=args.output_dir
        )
    
    print("\nTesting completed!")
    
if __name__ == "__main__":
    main() 