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

# Import original components
from enhanced_fencer_detector import EnhancedFencerDetector, SwordDetector
from pose_estimation_helper import PoseEstimator
from advanced_fencing_analyzer import SimplifiedFencingAnalyzer

# Helper function to convert numpy arrays in results for JSON serialization
def convert_numpy_to_list(obj):
    if isinstance(obj, dict):
        return {k: convert_numpy_to_list(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_to_list(elem) for elem in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist() 
    elif isinstance(obj, (np.int_, np.intc, np.intp, np.int8, np.int16, np.int32, np.int64, np.uint8,
                          np.uint16, np.uint32, np.uint64)):
        return int(obj)
    elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
        return {'real': obj.real, 'imag': obj.imag}
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif isinstance(obj, (np.void)):
        return None # Or other representation as needed
    return obj

class IntegratedFencingAnalyzer:
    """
    Advanced fencing analysis system that integrates 3D ConvNet, temporal segmentation,
    and tactical knowledge graph components
    """
    def __init__(
        self,
        temporal_model_path: Optional[str] = None,
        pose_model_path: Optional[str] = None,
        sword_model_path: Optional[str] = None,
        knowledge_graph_path: Optional[str] = None,
        device: Optional[torch.device] = None,
        temporal_window: int = 16,
        img_size: int = 224,
        bout_mode: bool = True
    ):
        """
        Initialize the integrated analyzer
        
        Args:
            temporal_model_path: Path to pretrained 3D temporal model weights
            pose_model_path: Path to pose classifier model
            sword_model_path: Path to sword detector model
            knowledge_graph_path: Path to fencing knowledge graph
            device: Torch device (cpu or cuda)
            temporal_window: Size of temporal window for analysis
            img_size: Image size for model input
            bout_mode: Whether to optimize for two-fencer bouts
        """
        print("Initializing Integrated Fencing Analyzer...")
        
        # Set device
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        # Initialize components
        self._init_temporal_model(temporal_model_path, temporal_window, img_size)
        self._init_segmentation(temporal_window)
        self._init_knowledge_graph(knowledge_graph_path)
        self._init_detectors(pose_model_path, sword_model_path, bout_mode)
        
        # Initialize buffer for temporal processing
        self.frame_buffer = []
        self.technique_history = []
        self.detected_segments = []
        
        # Parameters
        self.temporal_window = temporal_window
        self.img_size = img_size
        self.bout_mode = bout_mode
        
        print("Integrated Fencing Analyzer initialization complete")
    
    def _init_temporal_model(self, model_path, temporal_window, img_size):
        """Initialize the 3D temporal model"""
        print("Loading 3D Temporal ConvNet + Transformer model...")
        try:
            self.temporal_model = FencingTemporalModel(
                model_path=model_path,
                temporal_size=temporal_window,
                img_size=img_size,
                device=self.device
            )
            print(f"Temporal model loaded from: {model_path if model_path else 'random initialization'}")
        except Exception as e:
            print(f"Error initializing temporal model: {e}")
            self.temporal_model = None
    
    def _init_segmentation(self, temporal_window):
        """Initialize the temporal segmentation module"""
        print("Initializing temporal segmentation...")
        try:
            self.segmentation = TemporalSegmentation(
                temporal_window=temporal_window,
                motion_threshold=1.5,
                min_segment_length=temporal_window // 2,
                max_segment_length=temporal_window * 4
            )
            print("Temporal segmentation initialized")
        except Exception as e:
            print(f"Error initializing temporal segmentation: {e}")
            self.segmentation = None
    
    def _init_knowledge_graph(self, graph_path):
        """Initialize the fencing knowledge graph"""
        print("Loading fencing knowledge graph...")
        try:
            if graph_path and os.path.exists(graph_path):
                self.knowledge_graph = FencingKnowledgeGraph.load(graph_path)
                print(f"Knowledge graph loaded from: {graph_path}")
            else:
                self.knowledge_graph = FencingKnowledgeGraph()
                print("Default knowledge graph initialized")
            
            # Initialize tactical analyzer
            self.tactical_analyzer = FencingTacticalAnalyzer(self.knowledge_graph)
            print("Tactical analyzer initialized")
        except Exception as e:
            print(f"Error initializing knowledge graph: {e}")
            self.knowledge_graph = None
            self.tactical_analyzer = None
    
    def _init_detectors(self, pose_model_path, sword_model_path, bout_mode):
        """Initialize fencer detector and pose estimator"""
        print("Loading fencer detector and pose estimator...")
        try:
            # Initialize fencer detector
            self.fencer_detector = EnhancedFencerDetector(
                pose_model_path=pose_model_path,
                sword_model_path=sword_model_path,
                bout_mode=bout_mode
            )
            print("Enhanced fencer detector initialized")
            
            # Initialize pose estimator
            try:
                from pose_estimation_helper import MMPOSE_AVAILABLE
                self.pose_estimator = PoseEstimator(use_mmpose=MMPOSE_AVAILABLE)
                print(f"Pose estimator initialized (using MMPose: {MMPOSE_AVAILABLE})")
            except ImportError as e:
                print(f"Error initializing pose estimator: {e}")
                self.pose_estimator = None
        except Exception as e:
            print(f"Error initializing detectors: {e}")
            self.fencer_detector = None
            self.pose_estimator = None
    
    def detect_fencing_strip(self, frame):
        """
        Detect the fencing strip in the frame using edge detection
        
        Args:
            frame: Input video frame
            
        Returns:
            strip_mask: Binary mask of the detected strip
            strip_bbox: Bounding box of the strip [x1, y1, x2, y2]
            is_valid: Whether a valid strip was detected
        """
        # Convert to grayscale for edge detection
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, 50, 150)
        
        # Dilate edges to connect nearby lines
        kernel = np.ones((3, 3), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=2)
        
        # Find contours in the edge map
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Initialize strip parameters
        strip_mask = np.zeros_like(gray)
        strip_bbox = [0, 0, 0, 0]
        is_valid = False
        
        if contours:
            # Sort contours by area (descending)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            
            # Look for rectangular contours that could be the strip
            for contour in contours[:10]:  # Check the 10 largest contours
                # Approximate the contour
                epsilon = 0.02 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                
                # Check if it's roughly rectangular (4-8 sides)
                if 4 <= len(approx) <= 8:
                    # Check aspect ratio to find elongated rectangle
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = float(w) / h if h > 0 else 0
                    
                    # Fencing strips are typically elongated rectangles
                    if (aspect_ratio > 3 or aspect_ratio < 0.33) and \
                       w * h > 0.15 * frame.shape[0] * frame.shape[1]:  # Must be at least 15% of frame
                        # Draw the contour on the mask
                        cv2.drawContours(strip_mask, [contour], 0, 255, -1)
                        strip_bbox = [x, y, x + w, y + h]
                        is_valid = True
                        break
        
        # If no valid strip found, use a horizontal band in the middle as fallback
        if not is_valid:
            h, w = frame.shape[:2]
            strip_y1 = int(h * 0.4)  # Top at 40% of height
            strip_y2 = int(h * 0.9)  # Bottom at 90% of height
            strip_mask[strip_y1:strip_y2, :] = 255
            strip_bbox = [0, strip_y1, w, strip_y2]
            is_valid = True
        
        return strip_mask, strip_bbox, is_valid
    
    def process_frame(
        self, 
        frame: np.ndarray,
        fencer_ids: Optional[List[int]] = None
    ) -> Dict[str, Any]:
        """
        Process a single frame with all analysis components
        
        Args:
            frame: Video frame to process
            fencer_ids: Optional list of specific fencer IDs to track
            
        Returns:
            results: Dictionary with analysis results
        """
        # Initialize results
        results = {
            'fencer_detections': [],
            'sword_detections': [],
            'temporal_segment': None,
            'techniques': {},
            'coaching_feedback': {},
            'strip_detected': False,
            'strip_bbox': None
        }
        
        # Add frame to buffer
        self.frame_buffer.append(frame.copy())
        if len(self.frame_buffer) > self.temporal_window * 4:
            self.frame_buffer.pop(0)
        
        # 1. Detect fencing strip
        strip_mask, strip_bbox, strip_detected = self.detect_fencing_strip(frame)
        results['strip_detected'] = strip_detected
        results['strip_bbox'] = strip_bbox
        
        # 2. Detect fencers and swords
        if self.fencer_detector:
            tracked_items, sword_detections = self.fencer_detector.track_and_classify(frame)
            
            # Filter by fencer IDs if specified
            if fencer_ids:
                tracked_items = [item for item in tracked_items if item['fencer_id'] in fencer_ids]
            elif self.bout_mode:
                # If in bout mode, filter fencers based on strip position
                if strip_detected:
                    # Score each fencer based on overlap with the strip
                    for item in tracked_items:
                        box = item['box']
                        # Calculate overlap between fencer box and strip
                        overlap = self._calculate_box_overlap(box, strip_bbox)
                        item['strip_overlap'] = overlap
                    
                    # Sort by strip overlap (descending)
                    tracked_items.sort(key=lambda x: x.get('strip_overlap', 0.0), reverse=True)
                    
                    # Keep only the 2 most overlapping fencers for bout mode
                    tracked_items = tracked_items[:2] if len(tracked_items) > 2 else tracked_items
                else:
                    # If strip not detected, fallback to confidence-based selection
                    tracked_items.sort(key=lambda x: x.get('confidence', 0.0), reverse=True)
                    tracked_items = tracked_items[:2] if len(tracked_items) > 2 else tracked_items
            
            results['fencer_detections'] = tracked_items
            results['sword_detections'] = sword_detections
            
            # Add pose estimation for each fencer
            if self.pose_estimator:
                for item in tracked_items:
                    # Estimate pose for this fencer
                    fencer_box = item['box']
                    keypoints = self.pose_estimator.estimate_pose(frame, fencer_box)
                    # Add keypoints to the fencer detection
                    if keypoints is not None:
                        item['keypoints'] = keypoints
            
            # Create fencer masks for segmentation
            fencer_masks = []
            for item in tracked_items:
                x1, y1, x2, y2 = item['box']
                mask = np.zeros(frame.shape[:2], dtype=np.uint8)
                mask[int(y1):int(y2), int(x1):int(x2)] = 255
                fencer_masks.append(mask)
        else:
            fencer_masks = []
        
        # 3. Perform temporal segmentation
        if self.segmentation:
            seg_results = self.segmentation.process_frame(frame, fencer_masks=fencer_masks)
            
            # Check if we have a new segment
            if 'segment_frames' in seg_results:
                segment_frames = seg_results['segment_frames']
                
                # Skip if too few frames
                if len(segment_frames) >= self.temporal_window // 2:
                    new_segment = {
                        'start_idx': seg_results.get('new_segment', (0, 0))[0],
                        'end_idx': seg_results.get('new_segment', (0, 0))[1],
                        'frames': segment_frames
                    }
                    self.detected_segments.append(new_segment)
                    results['temporal_segment'] = new_segment
                    
                    # Process segment with 3D temporal model
                    if self.temporal_model:
                        # Process entire segment
                        class_id, class_name, confidence = self.temporal_model.classify_sequence(segment_frames)
                        
                        # Save technique
                        new_technique = {
                            'technique': class_name,
                            'confidence': confidence,
                            'segment_idx': len(self.detected_segments) - 1
                        }
                        
                        # Add to technique history
                        self.technique_history.append(class_name)
                        
                        # Get temporal attention for visualization
                        if len(segment_frames) >= self.temporal_window:
                            attention = self.temporal_model.get_temporal_attention(segment_frames)
                            new_technique['attention'] = attention
                        
                        # Get coaching feedback
                        if self.tactical_analyzer:
                            # Get feedback for this technique in context
                            feedback = self.tactical_analyzer.generate_coaching_feedback(
                                self.technique_history[-5:] if len(self.technique_history) > 5 else self.technique_history
                            )
                            new_technique['coaching_feedback'] = feedback
                        
                        # Save to results
                        results['techniques'][class_name] = new_technique
        
        # 4. Generate per-fencer feedback
        if results['fencer_detections'] and self.technique_history:
            for fencer in results['fencer_detections']:
                fencer_id = fencer['fencer_id']
                
                # Get current pose class from CNN
                pose_class = fencer.get('pose_class', 'neutral')
                
                # Create coaching feedback
                if self.tactical_analyzer:
                    # Create context with recent techniques and current pose
                    technique_context = self.technique_history[-4:] if len(self.technique_history) >= 4 else self.technique_history.copy()
                    if pose_class not in technique_context:
                        technique_context.append(pose_class)
                    
                    # Generate feedback
                    feedback = self.tactical_analyzer.generate_coaching_feedback(technique_context)
                    results['coaching_feedback'][fencer_id] = feedback
        
        return results
    
    def analyze_video(
        self,
        video_path: str,
        output_path: Optional[str] = None,
        fencer_ids: Optional[List[int]] = None,
        max_frames: Optional[int] = None,
        save_visualization: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze a fencing video with the integrated pipeline
        
        Args:
            video_path: Path to video file
            output_path: Path to save output video
            fencer_ids: Optional list of specific fencer IDs to track
            max_frames: Maximum number of frames to process
            save_visualization: Whether to create visualization
            
        Returns:
            results: Dictionary with analysis results
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return {'error': f"Could not open video {video_path}"}
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create output video writer if needed
        writer = None
        if output_path:
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Reset state
        self.frame_buffer = []
        self.technique_history = []
        self.detected_segments = []
        
        # Analysis results
        analysis_results = {
            'video_path': video_path,
            'frame_analyses': [],
            'segments': [],
            'techniques': [],
            'fencer_feedback': {}
        }
        
        # Process frames
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if we've hit max frames
            if max_frames and frame_idx >= max_frames:
                break
            
            # Process frame
            results = self.process_frame(frame, fencer_ids=fencer_ids)
            
            # Save frame analysis
            frame_analysis = {
                'frame_idx': frame_idx,
                'fencer_detections': results['fencer_detections'],
                'sword_detections': results['sword_detections'],
                'techniques': results['techniques'],
                'coaching_feedback': results['coaching_feedback']
            }
            analysis_results['frame_analyses'].append(frame_analysis)
            
            # Save segment if we have one
            if results['temporal_segment']:
                segment = results['temporal_segment'].copy()
                segment['frame_idx'] = frame_idx
                analysis_results['segments'].append(segment)
            
            # Save techniques
            for technique_name, technique_data in results['techniques'].items():
                technique_entry = technique_data.copy()
                technique_entry['frame_idx'] = frame_idx
                analysis_results['techniques'].append(technique_entry)
            
            # Update fencer feedback
            for fencer_id, feedback in results['coaching_feedback'].items():
                if fencer_id not in analysis_results['fencer_feedback']:
                    analysis_results['fencer_feedback'][fencer_id] = []
                
                analysis_results['fencer_feedback'][fencer_id].append({
                    'frame_idx': frame_idx,
                    'feedback': feedback
                })
            
            # Create visualization if needed
            if save_visualization and writer:
                vis_frame = self.create_visualization(frame, results)
                writer.write(vis_frame)
            
            frame_idx += 1
            
            # Print progress
            if frame_idx % 10 == 0:
                print(f"Processed {frame_idx} frames", end='\r')
        
        # Close resources
        cap.release()
        if writer:
            writer.release()
        
        # Compute summary statistics
        analysis_results['summary'] = self.generate_summary(analysis_results)
        
        print(f"\nProcessed {frame_idx} frames total.")
        print(f"Detected {len(analysis_results['segments'])} temporal segments.")
        # Recalculate technique counts from the final list
        technique_counts = {}
        for tech in analysis_results.get('techniques', []):
            name = tech.get('technique', 'unknown')
            technique_counts[name] = technique_counts.get(name, 0) + 1
        print(f"Identified techniques: {technique_counts}")
        
        # Remove large attention arrays before serialization
        if 'techniques' in analysis_results:
            for technique in analysis_results['techniques']:
                if 'attention' in technique:
                    del technique['attention']
        
        # Remove raw segment frames before serialization
        if 'segments' in analysis_results:
            for segment in analysis_results['segments']:
                if 'frames' in segment:
                    del segment['frames']
        
        # Convert numpy arrays before saving
        analysis_results_serializable = convert_numpy_to_list(analysis_results)
        
        return analysis_results_serializable
    
    def create_visualization(
        self, 
        frame: np.ndarray, 
        results: Dict[str, Any]
    ) -> np.ndarray:
        """
        Create visualization with all analysis components
        
        Args:
            frame: Current video frame
            results: Analysis results for this frame
            
        Returns:
            vis_frame: Visualization frame
        """
        # Create a copy of the frame
        vis_frame = frame.copy()
        
        # Draw fencing strip if detected
        if results.get('strip_detected', False) and results.get('strip_bbox'):
            strip_bbox = results['strip_bbox']
            x1, y1, x2, y2 = map(int, strip_bbox)
            
            # Draw blue outline around the strip
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)  # Blue color
            
            # Add transparent overlay to highlight the strip area
            overlay = vis_frame.copy()
            alpha = 0.2  # Transparency factor
            cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), -1)  # Filled rectangle
            vis_frame = cv2.addWeighted(overlay, alpha, vis_frame, 1 - alpha, 0)
            
            # Add strip label
            cv2.putText(vis_frame, "FENCING STRIP", (x1 + 10, y1 + 20), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        # 1. Draw fencer detections with bounding boxes and pose classes
        if results['fencer_detections'] and self.fencer_detector:
            vis_frame = self.fencer_detector.draw_enhanced_detections(
                vis_frame, 
                results['fencer_detections'], 
                results['sword_detections']
            )
            
            # 2. Draw pose keypoints using pose_estimator
            if self.pose_estimator:
                for fencer in results['fencer_detections']:
                    if 'keypoints' in fencer:
                        vis_frame = self.pose_estimator.draw_pose(vis_frame, fencer['keypoints'])
        
        # Overlay Temporal Technique Name if available
        if results['techniques']:
            # Get the first technique name for this frame
            technique_name = list(results['techniques'].keys())[0].upper()
            
            for fencer in results.get('fencer_detections', []):
                x1, y1, _, _ = map(int, fencer['box'])
                # Position the text slightly above the bounding box
                text_pos = (x1, y1 - 10) 
                
                # Add background rectangle for better visibility
                (w, h), _ = cv2.getTextSize(technique_name, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(vis_frame, (text_pos[0], text_pos[1] - h - 2), (text_pos[0] + w, text_pos[1] + 2), (0, 0, 0), -1)
                
                cv2.putText(vis_frame, technique_name, text_pos, 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (50, 255, 50), 2) # Bright green text
        
        # 3. Draw segment and technique information if available
        if 'temporal_segment' in results and results['temporal_segment']:
            # Draw a banner at the top of the frame
            banner_height = 40
            overlay = vis_frame.copy()
            cv2.rectangle(overlay, (0, 0), (frame.shape[1], banner_height), (0, 0, 0), -1)
            vis_frame = cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0)
            
            # Show technique information
            technique_text = "Detected: "
            for technique_name, technique_data in results['techniques'].items():
                confidence = technique_data.get('confidence', 0.0)
                technique_text += f"{technique_name.upper()} ({confidence:.2f}) "
            
            cv2.putText(vis_frame, technique_text, (10, banner_height - 10), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        return vis_frame
    
    def generate_summary(self, analysis_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Generate summary of the analysis results
        
        Args:
            analysis_results: Full analysis results
            
        Returns:
            summary: Summary statistics and findings
        """
        # Count techniques
        technique_counts = {}
        for technique in analysis_results['techniques']:
            technique_name = technique['technique']
            technique_counts[technique_name] = technique_counts.get(technique_name, 0) + 1
        
        # Count per-fencer feedback
        fencer_feedback_counts = {}
        for fencer_id, feedbacks in analysis_results['fencer_feedback'].items():
            fencer_feedback_counts[fencer_id] = len(feedbacks)
        
        # Calculate segment statistics
        segment_durations = []
        for segment in analysis_results['segments']:
            if 'start_idx' in segment and 'end_idx' in segment:
                duration = segment['end_idx'] - segment['start_idx']
                segment_durations.append(duration)
        
        avg_segment_duration = np.mean(segment_durations) if segment_durations else 0
        
        # Assemble summary
        summary = {
            'total_frames': len(analysis_results['frame_analyses']),
            'total_segments': len(analysis_results['segments']),
            'total_techniques': len(analysis_results['techniques']),
            'technique_counts': technique_counts,
            'fencer_feedback_counts': fencer_feedback_counts,
            'avg_segment_duration': avg_segment_duration
        }
        
        return summary
    
    def visualize_knowledge_graph(self, output_path: Optional[str] = None) -> None:
        """
        Visualize the fencing knowledge graph
        
        Args:
            output_path: Path to save the visualization
        """
        if self.knowledge_graph:
            self.knowledge_graph.visualize(output_path)
        else:
            print("Error: Knowledge graph not initialized")

    def _calculate_box_overlap(self, box1, box2):
        """
        Calculate the overlap ratio between two bounding boxes
        
        Args:
            box1: First bounding box [x1, y1, x2, y2]
            box2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            overlap_ratio: Ratio of intersection area to box1 area
        """
        # Calculate intersection
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Check if there is any intersection
        if x2 < x1 or y2 < y1:
            return 0.0
        
        intersection = (x2 - x1) * (y2 - y1)
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        
        # Avoid division by zero
        if box1_area == 0:
            return 0.0
        
        # Return the ratio of intersection to box1 area
        return intersection / box1_area


def main():
    """Main function to run the integrated analyzer"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Integrated Fencing Analyzer")
    parser.add_argument("video_path", help="Path to the fencing video")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--fencer_ids", help="Comma-separated list of fencer IDs to track")
    parser.add_argument("--temporal_model", default=None, help="Path to trained temporal model")
    parser.add_argument("--pose_model", default="models/pose_classifier.pth", help="Path to trained pose classifier model")
    parser.add_argument("--sword_model", default="models/yolov8n_blade.pt", help="Path to trained sword detector model")
    parser.add_argument("--knowledge_graph", default=None, help="Path to fencing knowledge graph")
    parser.add_argument("--no_vis", action="store_true", help="Disable visualization output")
    parser.add_argument("--bout_mode", action="store_true", help="Optimize for two fencers (bouts)")
    parser.add_argument("--enable_sword_detection", action="store_true", help="Enable sword detection and tracking")
    
    args = parser.parse_args()
    
    # Parse fencer IDs
    fencer_ids = None
    if args.fencer_ids:
        try:
            fencer_ids = [int(id.strip()) for id in args.fencer_ids.split(',')]
        except ValueError:
            print("Error: Invalid fencer IDs format. Expected comma-separated integers.")
            return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set sword model path to None if sword detection is disabled
    sword_model_path = args.sword_model if args.enable_sword_detection else None
    if args.enable_sword_detection:
        print(f"Sword detection enabled, using model: {sword_model_path}")
    else:
        print("Sword detection disabled")
    
    # Initialize analyzer
    analyzer = IntegratedFencingAnalyzer(
        temporal_model_path=args.temporal_model,
        pose_model_path=args.pose_model,
        sword_model_path=sword_model_path,
        knowledge_graph_path=args.knowledge_graph,
        bout_mode=args.bout_mode
    )
    
    # Create output paths
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    output_video_path = os.path.join(args.output_dir, f"{video_name}_integrated_analysis.mp4")
    output_json_path = os.path.join(args.output_dir, f"{video_name}_integrated_analysis.json")
    knowledge_graph_viz_path = os.path.join(args.output_dir, f"{video_name}_knowledge_graph.png")
    
    # Run analysis
    results = analyzer.analyze_video(
        video_path=args.video_path,
        output_path=output_video_path if not args.no_vis else None,
        fencer_ids=fencer_ids,
        max_frames=args.max_frames,
        save_visualization=not args.no_vis
    )
    
    # Save results to JSON
    try:
        with open(output_json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to {output_json_path}")
    except TypeError as e:
        print(f"Error saving results to JSON: {e}")
        # Fallback: Try saving without potentially problematic fields
        try:
            fallback_results = results.copy()
            if 'segments' in fallback_results:
                for segment in fallback_results['segments']:
                    if 'frames' in segment: # Remove raw frames
                        del segment['frames']
                    if 'attention' in segment: # Remove attention arrays
                        del segment['attention']
            if 'techniques' in fallback_results:
                 for technique in fallback_results['techniques']:
                    if 'attention' in technique: # Remove attention arrays
                        del technique['attention']
                        
            fallback_results = convert_numpy_to_list(fallback_results) # Ensure conversion
            
            fallback_path = os.path.join(args.output_dir, f"{video_name}_integrated_analysis_fallback.json")
            with open(fallback_path, 'w') as f:
                json.dump(fallback_results, f, indent=2)
            print(f"Saved fallback results (excluding problematic data) to {fallback_path}")
        except Exception as fallback_e:
            print(f"Could not save fallback JSON either: {fallback_e}")
    
    # Visualize knowledge graph if available
    if analyzer.knowledge_graph:
        analyzer.visualize_knowledge_graph(output_path=knowledge_graph_viz_path)
    
    print("\nAnalysis complete!")
    print(f"Processed {results['summary']['total_frames']} frames")
    print(f"Detected {results['summary']['total_segments']} temporal segments")
    print(f"Identified techniques: {results['summary']['technique_counts']}")
    

if __name__ == "__main__":
    main() 