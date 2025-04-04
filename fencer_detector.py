import cv2
import numpy as np
from ultralytics import YOLO
import torch
from pathlib import Path
import time
from tqdm import tqdm

class FencerDetector:
    def __init__(self, model_path=None):
        """
        Initialize fencer detector using YOLOv8
        
        Args:
            model_path: Path to custom trained YOLOv8 model. If None, uses pretrained model.
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Initialize YOLOv8 model
        if model_path and Path(model_path).exists():
            print(f"Loading custom model from {model_path}")
            self.model = YOLO(model_path)
        else:
            print("Loading pretrained YOLOv8 model")
            self.model = YOLO('yolov8n.pt')
        
        # Initialize tracking state
        self.prev_detections = []
        self.track_history = {}
        self.next_id = 0
    
    def calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (area1 + area2 - intersection + 1e-6)
    
    def detect_fencers(self, frame, conf_threshold=0.5):
        """
        Detect fencers in a frame
        
        Args:
            frame: Input video frame (BGR)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            detections: List of detections (boxes, confidences, class_ids)
        """
        # Run YOLOv8 inference
        results = self.model(frame, conf=conf_threshold, verbose=False)[0]
        
        # Extract detections
        detections = []
        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, conf, cls = r
            if conf >= conf_threshold and cls == 0:  # Only keep person class
                detections.append({
                    'box': [int(x1), int(y1), int(x2), int(y2)],
                    'confidence': float(conf),
                    'class_id': int(cls)
                })
        
        return detections
    
    def track_fencers(self, frame, detections):
        """
        Track fencers across frames using IoU matching
        
        Args:
            frame: Current video frame
            detections: List of detections from detect_fencers()
            
        Returns:
            tracked_boxes: List of tracked bounding boxes with IDs
        """
        tracked_boxes = []
        
        # If this is the first frame
        if not self.prev_detections:
            for det in detections:
                det['track_id'] = self.next_id
                self.track_history[self.next_id] = {'box': det['box'], 'confidence': det['confidence']}
                tracked_boxes.append((self.next_id, det['box']))
                self.next_id += 1
        else:
            # Calculate IoU between current detections and previous detections
            matched_prev_ids = set()
            matched_curr_indices = set()
            
            for i, det in enumerate(detections):
                best_iou = 0.3  # IoU threshold
                best_id = None
                
                for prev_det in self.prev_detections:
                    if prev_det['track_id'] in matched_prev_ids:
                        continue
                        
                    iou = self.calculate_iou(det['box'], prev_det['box'])
                    if iou > best_iou:
                        best_iou = iou
                        best_id = prev_det['track_id']
                
                if best_id is not None:
                    det['track_id'] = best_id
                    matched_prev_ids.add(best_id)
                    matched_curr_indices.add(i)
                    self.track_history[best_id] = {'box': det['box'], 'confidence': det['confidence']}
                    tracked_boxes.append((best_id, det['box']))
            
            # Handle unmatched detections
            for i, det in enumerate(detections):
                if i not in matched_curr_indices:
                    det['track_id'] = self.next_id
                    self.track_history[self.next_id] = {'box': det['box'], 'confidence': det['confidence']}
                    tracked_boxes.append((self.next_id, det['box']))
                    self.next_id += 1
        
        # Update previous detections
        self.prev_detections = detections
        
        return tracked_boxes
    
    def draw_detections(self, frame, tracked_boxes):
        """
        Draw bounding boxes and tracking information on frame
        
        Args:
            frame: Input video frame
            tracked_boxes: List of tracked bounding boxes with IDs
            
        Returns:
            frame: Frame with visualizations
        """
        for fencer_id, box in tracked_boxes:
            x1, y1, x2, y2 = map(int, box)
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw fencer ID
            cv2.putText(frame, f"Fencer {fencer_id}", (x1, y1-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
            # Draw confidence if available
            if fencer_id in self.track_history:
                conf = self.track_history[fencer_id]['confidence']
                cv2.putText(frame, f"Conf: {conf:.2f}", (x1, y2+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame
    
    def process_video(self, video_path, output_path=None, max_frames=None):
        """
        Process a video file for fencer detection and tracking
        
        Args:
            video_path: Path to input video
            output_path: Optional path to save output video
            max_frames: Optional maximum number of frames to process
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error opening video file: {video_path}")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames:
            total_frames = min(total_frames, max_frames)
        
        # Initialize video writer if output path is provided
        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        # Process frames
        frame_count = 0
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while cap.isOpened() and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Detect fencers
                detections = self.detect_fencers(frame)
                
                # Track fencers
                tracked_boxes = self.track_fencers(frame, detections)
                
                # Draw results
                frame = self.draw_detections(frame, tracked_boxes)
                
                # Write frame if output path is provided
                if writer:
                    writer.write(frame)
                
                frame_count += 1
                pbar.update(1)
        
        # Clean up
        cap.release()
        if writer:
            writer.release()
        
        print(f"Processed {frame_count} frames")

    def detect_and_classify(self, frame):
        """Detect and classify fencers in frame
        
        Args:
            frame: Input frame
            
        Returns:
            detections: List of fencer detections with poses
            sword_detections: List of sword detections (empty for base class)
        """
        # Run inference
        results = self.model(frame)[0]
        
        # Process detections
        detections = []
        for i, r in enumerate(results.boxes.data.tolist()):
            x1, y1, x2, y2, conf, class_id = r
            
            detection = {
                'box': [x1, y1, x2, y2],
                'confidence': conf,
                'class_id': int(class_id),
                'pose_class': 'neutral',  # Default pose class
                'pose_confidence': 0.0,
                'fencer_id': i + 1  # Add fencer ID starting from 1
            }
            detections.append(detection)
            
        return detections, []  # Empty list for sword detections 