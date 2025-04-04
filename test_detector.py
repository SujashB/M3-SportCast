import cv2
import torch
import os
import argparse
import numpy as np
from enhanced_fencer_detector import EnhancedFencerDetector

def main():
    parser = argparse.ArgumentParser(description='Test fencer detector with specific fencer tracking')
    parser.add_argument('--video', type=str, default='fencing_demo_video.mp4', help='Path to input video')
    parser.add_argument('--output', type=str, default='output_detection.mp4', help='Path to output video')
    parser.add_argument('--fencer_ids', type=str, default=None, help='Comma-separated IDs of fencers to track (e.g., "0,1")')
    parser.add_argument('--box', type=str, default=None, help='Comma-separated box coordinates to focus on (e.g., "x1,y1,x2,y2")')
    parser.add_argument('--max_frames', type=int, default=None, help='Maximum number of frames to process')
    
    args = parser.parse_args()
    
    # Set output directory
    output_dir = os.path.dirname(args.output) if os.path.dirname(args.output) else '.'
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize enhanced fencer detector
    detector = EnhancedFencerDetector()
    
    # Set specific fencer IDs to track if provided
    if args.fencer_ids:
        fencer_ids = [int(id.strip()) for id in args.fencer_ids.split(',')]
        detector.set_target_fencers(fencer_ids)
        print(f"Focusing on fencers with IDs: {fencer_ids}")
    
    # Parse initial bounding box if provided
    init_fencer_boxes = None
    if args.box:
        try:
            coords = [int(c.strip()) for c in args.box.split(',')]
            if len(coords) == 4:
                init_fencer_boxes = [coords]
                print(f"Using initial bounding box: {coords}")
            else:
                print("Warning: Box format should be x1,y1,x2,y2")
        except ValueError:
            print("Warning: Invalid box format. Expected comma-separated integers.")
    
    # Process video
    print(f"Processing video: {args.video}")
    detector.process_video(args.video, args.output, args.max_frames, init_fencer_boxes)
    print(f"Output saved to: {args.output}")

if __name__ == "__main__":
    main() 