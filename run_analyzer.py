#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
import argparse
import json
from typing import List, Dict, Tuple, Optional, Any
import time
from tqdm import tqdm
import networkx as nx

# Import our custom components
from videomae_model import FencingTemporalModel
from temporal_segmentation import TemporalSegmentation
from fencing_knowledge_graph import FencingKnowledgeGraph, FencingTacticalAnalyzer, generate_coaching_tips

# Import original components - bypass the problematic advanced_fencing_analyzer.py
from enhanced_fencer_detector import EnhancedFencerDetector
from pose_estimation_helper import PoseEstimator

class SimpleRunner:
    """
    A simple runner for the fencing analysis system that bypasses advanced_fencing_analyzer.py
    """
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
    def run_analysis(self, video_path, output_dir="results", bout_mode=True, max_frames=None):
        """Run analysis on a video file"""
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Initialize components
        try:
            sword_model_path = "models/yolov8n_blade.pt"
            pose_model_path = "models/pose_classifier.pth"
            
            fencer_detector = EnhancedFencerDetector(
                pose_model_path=pose_model_path, 
                sword_model_path=sword_model_path,
                bout_mode=bout_mode
            )
            
            temporal_seg = TemporalSegmentation(
                temporal_window=16,
                motion_threshold=1.5,
                min_segment_length=8,
                max_segment_length=64
            )
            
            knowledge_graph = FencingKnowledgeGraph()
            tactical_analyzer = FencingTacticalAnalyzer(knowledge_graph)
            
            print("All components initialized successfully!")
        except Exception as e:
            print(f"Error initializing components: {e}")
            return
        
        # Process video
        try:
            # Create output file paths
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            output_video_path = os.path.join(output_dir, f"{video_name}_analyzed.mp4")
            output_json_path = os.path.join(output_dir, f"{video_name}_results.json")
            
            # Open video
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open video {video_path}")
                return
            
            # Get video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            
            # Create output video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
            
            # Analysis results
            results = {
                'video_path': video_path,
                'frame_analyses': [],
                'segments': [],
                'techniques': {},
                'fencer_feedback': {}
            }
            
            # Process frames
            frame_idx = 0
            technique_history = []
            
            # Progress bar
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if max_frames:
                total_frames = min(total_frames, max_frames)
            
            pbar = tqdm(total=total_frames, desc="Processing video")
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Check if we've hit max frames
                if max_frames and frame_idx >= max_frames:
                    break
                
                # 1. Detect fencers and swords
                tracked_items, sword_detections = fencer_detector.track_and_classify(frame)
                
                # 2. Perform temporal segmentation
                seg_results = temporal_seg.process_frame(frame)
                
                # 3. Create frame analysis
                frame_analysis = {
                    'frame_idx': frame_idx,
                    'fencer_detections': tracked_items,
                    'sword_detections': sword_detections
                }
                
                # Check if we have a new segment
                if 'segment_frames' in seg_results:
                    segment = {
                        'start_idx': seg_results.get('new_segment', (0, 0))[0],
                        'end_idx': seg_results.get('new_segment', (0, 0))[1],
                    }
                    results['segments'].append(segment)
                    
                    # Add random technique for demonstration 
                    # (would normally use the temporal model)
                    techniques = ['lunge', 'parry', 'riposte', 'advance']
                    technique = techniques[frame_idx % len(techniques)]
                    technique_history.append(technique)
                    
                    # Get coaching feedback
                    if tactical_analyzer:
                        feedback = tactical_analyzer.generate_coaching_feedback(
                            technique_history[-5:] if len(technique_history) > 5 else technique_history
                        )
                        frame_analysis['feedback'] = feedback
                
                # Save frame analysis
                results['frame_analyses'].append(frame_analysis)
                
                # Create visualization
                vis_frame = self.create_visualization(frame, frame_analysis, fencer_detector)
                writer.write(vis_frame)
                
                frame_idx += 1
                pbar.update(1)
            
            # Close resources
            cap.release()
            writer.release()
            pbar.close()
            
            # Save results to JSON
            with open(output_json_path, 'w') as f:
                json.dump(results, f, indent=2)
            
            print(f"\nProcessed {frame_idx} frames total.")
            print(f"Output video saved to: {output_video_path}")
            print(f"Results saved to: {output_json_path}")
        
        except Exception as e:
            print(f"Error processing video: {e}")
    
    def create_visualization(self, frame, frame_analysis, fencer_detector):
        """Create visualization with detections"""
        vis_frame = frame.copy()
        
        # Draw fencer detections
        if 'fencer_detections' in frame_analysis and fencer_detector:
            vis_frame = fencer_detector.draw_enhanced_detections(
                vis_frame, 
                frame_analysis['fencer_detections'], 
                frame_analysis.get('sword_detections', [])
            )
        
        # Draw feedback if available
        if 'feedback' in frame_analysis:
            feedback = frame_analysis['feedback']
            
            # Draw at the top of the frame
            cv2.rectangle(vis_frame, (0, 0), (frame.shape[1], 40), (0, 0, 0), -1)
            cv2.putText(vis_frame, feedback[:50], (10, 30), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Run Fencing Analysis")
    parser.add_argument("video_path", help="Path to the fencing video")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--bout_mode", action="store_true", help="Optimize for two fencers (bouts)")
    
    args = parser.parse_args()
    
    # Run analysis
    runner = SimpleRunner()
    runner.run_analysis(
        video_path=args.video_path,
        output_dir=args.output_dir,
        bout_mode=args.bout_mode,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    main() 