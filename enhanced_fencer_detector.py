import cv2
import numpy as np
import torch
from pathlib import Path
import time
from tqdm import tqdm
import os
import yaml
import shutil

# Import base FencerDetector
from fencer_detector import FencerDetector
# Import our CNN pose classifier
from fencing_cnn import FencingPoseClassifier, extract_fencer_crops

class SwordDetector:
    """
    Detector for fencing blades using YOLOv8
    """
    def __init__(self, model_path=None):
        """
        Initialize sword detector
        
        Args:
            model_path: Path to custom trained YOLOv8 model. If None, uses default model.
        """
        # Load YOLOv8 model
        from ultralytics import YOLO
        
        # Set default path if not provided
        if model_path is None:
            model_path = 'models/yolov8n_blade.pt'
            
            # Check if model exists
            if not os.path.exists(model_path):
                print(f"Sword detection model not found at {model_path}. Using default YOLO model.")
                model_path = 'yolov8n.pt'
        
        self.model = YOLO(model_path)
        print(f"Loaded sword detector model from {model_path}")
        
        # Class names for the blade model
        self.class_names = {
            0: 'blade-guard',
            1: 'blade-tip',
            2: 'fencing-blade'
        }
        
        # Colors for visualization
        self.colors = {
            'blade-guard': (0, 165, 255),   # Orange
            'blade-tip': (0, 0, 255),       # Red
            'fencing-blade': (255, 0, 0)    # Blue
        }
        
        # Line thickness for sword visualization
        self.line_thickness = 2
    
    def detect_swords(self, frame, conf_threshold=0.15):
        """
        Detect swords and sword parts in an image
        
        Args:
            frame: Input image (BGR)
            conf_threshold: Confidence threshold
            
        Returns:
            detections: List of sword detections
        """
        # Create a copy of the frame with slight enhancement to help detection
        enhanced_frame = frame.copy()
        
        # Simple image enhancement (may help with blade detection)
        # Adjust brightness and contrast
        alpha = 1.3  # Contrast control (1.0 means no change)
        beta = 10    # Brightness control (0 means no change)
        enhanced_frame = cv2.convertScaleAbs(enhanced_frame, alpha=alpha, beta=beta)
        
        # Create a grayscale version and apply edge detection
        gray = cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 75, 150)
        
        # Convert edges back to BGR for YOLO
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Run inference on original, enhanced, and edge-detected frames
        results_original = self.model(frame, conf=conf_threshold, verbose=False)[0]
        results_enhanced = self.model(enhanced_frame, conf=conf_threshold, verbose=False)[0]
        results_edges = self.model(edges_bgr, conf=conf_threshold*0.8, verbose=False)[0]  # Lower threshold for edges
        
        # Combine detections from all sources
        all_detections = []
        
        # Process all results and combine
        for results, source_name in zip(
            [results_original, results_enhanced, results_edges],
            ['original', 'enhanced', 'edges']
        ):
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = r
                if conf >= conf_threshold:
                    class_id = int(cls)
                    class_name = self.class_names.get(class_id, f"class_{class_id}")
                    
                    # Convert coordinates to integers
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    # Check if this is a duplicate detection
                    new_detection = {
                        'box': [x1, y1, x2, y2],
                        'confidence': float(conf),
                        'class_id': class_id,
                        'class_name': class_name,
                        'source': source_name
                    }
                    
                    # Check for overlap with existing detections
                    is_duplicate = False
                    for i, det in enumerate(all_detections):
                        if det['class_name'] == class_name:
                            iou = self.calculate_iou(new_detection['box'], det['box'])
                            if iou > 0.3:  # Lower threshold for considering duplicates
                                is_duplicate = True
                                # Keep the one with higher confidence
                                if conf > det['confidence']:
                                    all_detections[i] = new_detection
                                break
                    
                    if not is_duplicate:
                        all_detections.append(new_detection)
        
        return all_detections
    
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
    
    def draw_detections(self, frame, detections):
        """
        Draw sword detections on frame
        
        Args:
            frame: Input frame
            detections: List of detections from detect_swords()
            
        Returns:
            frame: Annotated frame
        """
        for det in detections:
            box = det['box']
            class_name = det['class_name']
            conf = det['confidence']
            
            x1, y1, x2, y2 = box
            
            # Get color for this class
            color = self.colors.get(class_name, (255, 255, 255))
            
            # Draw different visualizations based on class
            if class_name == 'fencing-blade':
                # Draw a more prominent box for the blade
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)
                
                # Draw the center line of the blade (approximation)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Calculate the length and angle (rough approximation)
                width = x2 - x1
                height = y2 - y1
                
                # Draw the center line if the blade is long enough
                if max(width, height) > 50:
                    if height > width:  # Vertical blade
                        cv2.line(frame, (center_x, y1), (center_x, y2), color, self.line_thickness)
                    else:  # Horizontal blade
                        cv2.line(frame, (x1, center_y), (x2, center_y), color, self.line_thickness)
            
            elif class_name == 'blade-tip':
                # Draw a circle for the tip
                radius = min(10, (x2-x1)//2, (y2-y1)//2)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                cv2.circle(frame, (center_x, center_y), radius, color, self.line_thickness)
            
            elif class_name == 'blade-guard':
                # Draw a filled box for the guard
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, self.line_thickness)
                
                # Add cross lines
                cv2.line(frame, (x1, y1), (x2, y2), color, 1)
                cv2.line(frame, (x1, y2), (x2, y1), color, 1)
            
            # Draw label with higher contrast
            label = f"{class_name}: {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Draw label background with full opacity
            cv2.rectangle(frame, (x1, y1-20), (x1+text_size[0]+10, y1), color, -1)
            
            # Draw label text with black color
            cv2.putText(frame, label, (x1+5, y1-5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        return frame


class EnhancedFencerDetector(FencerDetector):
    """
    Enhanced fencer detector that combines YOLOv8 with a CNN pose classifier and sword detection
    """
    def __init__(self, model_path=None, pose_model_path=None, sword_model_path=None):
        """
        Initialize enhanced fencer detector
        
        Args:
            model_path: Path to custom trained YOLOv8 model. If None, uses pretrained model.
            pose_model_path: Path to trained pose classifier model
            sword_model_path: Path to trained sword detector model
        """
        # Initialize base FencerDetector
        super().__init__(model_path)
        
        # Initialize pose classifier
        self.pose_classifier = FencingPoseClassifier(
            model_path=pose_model_path,
            device=self.device
        )
        
        # Initialize sword detector
        self.sword_detector = SwordDetector(model_path=sword_model_path)
        
        # Track pose history for each fencer
        self.pose_history = {}
        
        # Track box history for each fencer (to reduce flickering)
        self.box_history = {}
        
        # Track sword detection history
        self.sword_history = []
        self.max_sword_history = 10
        
        # Define pose colors for visualization
        self.pose_colors = {
            'neutral': (255, 255, 255),  # White
            'attack': (0, 0, 255),       # Red (BGR)
            'defense': (0, 255, 0),      # Green
            'lunge': (255, 0, 0)         # Blue
        }
    
    def detect_and_classify(self, frame, conf_threshold=0.3):
        """
        Detect fencers and classify their poses, also detect swords
        
        Args:
            frame: Input video frame (BGR)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            detections: List of detections with pose classifications
            sword_detections: List of sword detections
        """
        # Detect fencers using YOLOv8
        detections = self.detect_fencers(frame, conf_threshold)
        
        # Detect swords with lower threshold for better recall
        sword_detections = self.sword_detector.detect_swords(frame, conf_threshold=0.15)
        
        # Extract bounding boxes
        boxes = [det['box'] for det in detections]
        
        # Extract crops of detected fencers
        if boxes:
            crops = extract_fencer_crops(frame, boxes)
            
            # Classify pose for each crop
            for i, crop in enumerate(crops):
                try:
                    # Classify pose
                    class_id, class_name, confidence = self.pose_classifier.classify_pose(crop)
                    
                    # Add pose classification to detection
                    detections[i]['pose_class_id'] = class_id
                    detections[i]['pose_class'] = class_name
                    detections[i]['pose_confidence'] = confidence
                except Exception as e:
                    # Handle errors gracefully
                    print(f"Error classifying pose: {e}")
                    detections[i]['pose_class_id'] = 0
                    detections[i]['pose_class'] = 'neutral'
                    detections[i]['pose_confidence'] = 0.0
        
        return detections, sword_detections
    
    def smooth_bounding_box(self, current_box, history, alpha=0.7):
        """
        Apply temporal smoothing to bounding box coordinates to reduce flickering
        
        Args:
            current_box: Current box coordinates [x1, y1, x2, y2]
            history: List of previous box coordinates
            alpha: Smoothing factor (0-1, higher value gives more weight to current)
            
        Returns:
            smoothed_box: Smoothed box coordinates
        """
        if not history:
            return current_box
        
        # Calculate average of previous boxes
        avg_box = [0, 0, 0, 0]
        for prev_box in history[-5:]:  # Use last 5 frames
            for i in range(4):
                avg_box[i] += prev_box[i]
        
        num_boxes = min(len(history), 5)
        avg_box = [coord / num_boxes for coord in avg_box]
        
        # Apply smoothing using exponential moving average
        smoothed_box = [
            int(alpha * current_box[i] + (1 - alpha) * avg_box[i])
            for i in range(4)
        ]
        
        return smoothed_box
    
    def track_and_classify(self, frame, detections=None, sword_detections=None):
        """
        Track fencers across frames and classify their poses
        
        Args:
            frame: Current video frame
            detections: Optional list of detections from detect_and_classify()
            sword_detections: Optional list of sword detections
            
        Returns:
            tracked_items: List of tracked fencers with pose classifications
            smoothed_sword_detections: List of temporally smoothed sword detections
        """
        # If detections not provided, run detection and classification
        if detections is None:
            detections, sword_detections = self.detect_and_classify(frame)
        
        # Track fencers using base class method
        tracked_boxes = self.track_fencers(frame, detections)
        
        # Apply temporal smoothing to sword detections
        if sword_detections:
            # Store current detections in history
            self.sword_history.append(sword_detections)
            if len(self.sword_history) > self.max_sword_history:
                self.sword_history.pop(0)
            
            # Only keep detections that appear consistently
            if len(self.sword_history) >= 3:  # Require at least 3 frames
                # Count how many times each class appears
                class_counts = {}
                for frame_dets in self.sword_history[-3:]:
                    for det in frame_dets:
                        class_name = det['class_name']
                        if class_name not in class_counts:
                            class_counts[class_name] = 0
                        class_counts[class_name] += 1
                
                # Filter out detections that don't appear consistently
                smoothed_sword_detections = []
                for det in sword_detections:
                    class_name = det['class_name']
                    if class_counts.get(class_name, 0) >= 2:  # At least 2 occurrences
                        smoothed_sword_detections.append(det)
                
                sword_detections = smoothed_sword_detections
        
        # Add pose classifications to tracked fencers
        tracked_items = []
        for fencer_id, box in tracked_boxes:
            # Find corresponding detection
            pose_class = 'neutral'
            pose_confidence = 0.0
            
            for det in detections:
                if 'track_id' in det and det['track_id'] == fencer_id:
                    pose_class = det.get('pose_class', 'neutral')
                    pose_confidence = det.get('pose_confidence', 0.0)
                    break
            
            # Update pose history
            if fencer_id not in self.pose_history:
                self.pose_history[fencer_id] = []
            
            # Add pose to history (limit to last 10 poses)
            self.pose_history[fencer_id].append(pose_class)
            if len(self.pose_history[fencer_id]) > 10:
                self.pose_history[fencer_id].pop(0)
            
            # Track most common pose in history for temporal smoothing
            if self.pose_history[fencer_id]:
                from collections import Counter
                pose_counts = Counter(self.pose_history[fencer_id])
                pose_class = pose_counts.most_common(1)[0][0]
            
            # Apply temporal smoothing to bounding box
            if fencer_id not in self.box_history:
                self.box_history[fencer_id] = []
            
            # Add current box to history
            self.box_history[fencer_id].append(box)
            if len(self.box_history[fencer_id]) > 10:
                self.box_history[fencer_id].pop(0)
            
            # Apply smoothing
            smoothed_box = self.smooth_bounding_box(box, self.box_history[fencer_id])
            
            tracked_items.append({
                'fencer_id': fencer_id,
                'box': smoothed_box,
                'pose_class': pose_class,
                'pose_confidence': pose_confidence
            })
        
        return tracked_items, sword_detections
    
    def draw_enhanced_detections(self, frame, tracked_items, sword_detections=None):
        """
        Draw bounding boxes, tracking information, and pose classifications on frame
        
        Args:
            frame: Input video frame
            tracked_items: List of tracked items from track_and_classify()
            sword_detections: Optional list of sword detections
            
        Returns:
            frame: Frame with visualizations
        """
        # Make a copy of the frame to avoid modifying the original
        annotated_frame = frame.copy()
        
        # Draw sword detections first (so they appear behind fencers)
        if sword_detections:
            annotated_frame = self.sword_detector.draw_detections(annotated_frame, sword_detections)
        
        # Draw fencer detections
        for item in tracked_items:
            fencer_id = item['fencer_id']
            box = item['box']
            pose_class = item.get('pose_class', 'neutral')
            pose_confidence = item.get('pose_confidence', 0.0)
            
            # Ensure box coordinates are integers
            x1, y1, x2, y2 = map(int, box)
            
            # Get color for pose
            color = self.pose_colors.get(pose_class, (255, 255, 255))
            
            # Draw bounding box with pose-specific color
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Determine text background rectangle dimensions
            text_thickness = 1
            text_size_id = cv2.getTextSize(f"Fencer {fencer_id}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_thickness)[0]
            text_size_pose = cv2.getTextSize(f"Pose: {pose_class}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_thickness)[0]
            text_size_conf = cv2.getTextSize(f"Conf: {pose_confidence:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_thickness)[0]
            
            # Get the max width needed for the text background
            max_width = max(text_size_id[0], text_size_pose[0], text_size_conf[0]) + 10
            
            # Draw background rectangle for text with full opacity
            cv2.rectangle(annotated_frame, (x1, y1-60), (x1+max_width, y1), color, -1)
            
            # Draw text with better contrast
            text_color = (0, 0, 0)  # Black text for better visibility
            cv2.putText(annotated_frame, f"Fencer {fencer_id}", (x1+5, y1-40),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, text_thickness)
            cv2.putText(annotated_frame, f"Pose: {pose_class}", (x1+5, y1-20),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, text_thickness)
            cv2.putText(annotated_frame, f"Conf: {pose_confidence:.2f}", (x1+5, y1-5),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, text_thickness)
        
        return annotated_frame
    
    def process_video(self, video_path, output_path=None, max_frames=None):
        """
        Process a video file for enhanced fencer detection and pose classification
        
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
        pose_stats = {}  # To track pose statistics per fencer
        sword_stats = {'blade-guard': 0, 'blade-tip': 0, 'fencing-blade': 0, 'total_frames': 0}  # To track sword statistics
        
        with tqdm(total=total_frames, desc="Processing video") as pbar:
            while cap.isOpened() and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process every 3rd frame to improve speed
                if frame_count % 3 == 0:
                    # Detect, classify, and track fencers and swords
                    tracked_items, sword_detections = self.track_and_classify(frame)
                    
                    # Draw enhanced visualizations
                    frame = self.draw_enhanced_detections(frame, tracked_items, sword_detections)
                    
                    # Update pose statistics
                    for item in tracked_items:
                        fencer_id = item['fencer_id']
                        pose_class = item['pose_class']
                        
                        # Initialize fencer stats if needed
                        if fencer_id not in pose_stats:
                            pose_stats[fencer_id] = {
                                'neutral': 0,
                                'attack': 0,
                                'defense': 0,
                                'lunge': 0,
                                'total_frames': 0
                            }
                        
                        # Update stats
                        if pose_class in pose_stats[fencer_id]:
                            pose_stats[fencer_id][pose_class] += 1
                        pose_stats[fencer_id]['total_frames'] += 1
                    
                    # Update sword statistics
                    if sword_detections:
                        sword_stats['total_frames'] += 1
                        for det in sword_detections:
                            class_name = det['class_name']
                            if class_name in sword_stats:
                                sword_stats[class_name] += 1
                
                # Write frame if output path is provided
                if writer:
                    writer.write(frame)
                
                frame_count += 1
                pbar.update(1)
        
        # Clean up
        cap.release()
        if writer:
            writer.release()
        
        # Print pose statistics
        print(f"\nProcessed {frame_count} frames")
        print("\nPose Statistics per Fencer:")
        for fencer_id, stats in pose_stats.items():
            print(f"\nFencer {fencer_id}:")
            total = stats['total_frames']
            for pose, count in sorted([(k, v) for k, v in stats.items() if k != 'total_frames'], 
                                     key=lambda x: x[1], reverse=True):
                if count > 0:
                    print(f"  {pose}: {count} frames ({count/total*100:.1f}%)")
        
        # Print sword statistics
        print("\nSword Detection Statistics:")
        total_frames_with_swords = sword_stats['total_frames']
        if total_frames_with_swords > 0:
            for part, count in sorted([(k, v) for k, v in sword_stats.items() if k != 'total_frames'], 
                                    key=lambda x: x[1], reverse=True):
                print(f"  {part}: {count} detections ({count/total_frames_with_swords*100:.1f}% of frames)")
        else:
            print("  No sword detections found")
        
        results = {
            'pose_stats': pose_stats,
            'sword_stats': sword_stats
        }
        
        return results


def train_sword_detector(dataset_path=None, epochs=50, batch_size=16):
    """
    Train YOLOv8 model for sword detection
    
    Args:
        dataset_path: Path to dataset. If None, uses fencingdataset/Fencing Blade.v8i.yolov8-obb
        epochs: Number of training epochs
        batch_size: Batch size for training
        
    Returns:
        Path to trained model
    """
    from ultralytics import YOLO
    import yaml
    import shutil
    
    # Default dataset path
    if dataset_path is None:
        dataset_path = 'fencingdataset/Fencing Blade.v8i.yolov8-obb'
    
    # Create absolute paths to dataset
    dataset_path = os.path.abspath(dataset_path)
    
    # Create a custom data.yaml with absolute paths
    original_data_yaml = os.path.join(dataset_path, 'data.yaml')
    if not os.path.exists(original_data_yaml):
        print(f"Error: Dataset YAML not found at {original_data_yaml}")
        return None
    
    # Read the original data.yaml and modify paths to be absolute
    with open(original_data_yaml, 'r') as f:
        data_config = yaml.safe_load(f)
    
    # Update paths to be absolute
    for key in ['train', 'val', 'test']:
        if key in data_config and data_config[key]:
            # Check if it's a relative path
            if not os.path.isabs(data_config[key]):
                data_config[key] = os.path.join(dataset_path, data_config[key])
    
    # Create models directory
    os.makedirs('models', exist_ok=True)
    
    # Create a custom data.yaml in the models directory
    custom_data_yaml = os.path.join('models', 'sword_data.yaml')
    with open(custom_data_yaml, 'w') as f:
        yaml.dump(data_config, f)
    
    print(f"Created custom data config at {custom_data_yaml}")
    
    # Initialize YOLOv8 model
    model = YOLO('yolov8n.pt')
    
    # Train model
    print(f"Training sword detection model for {epochs} epochs...")
    model.train(
        data=custom_data_yaml,
        epochs=epochs,
        batch=batch_size,
        imgsz=640,
        project='runs/sword_detection',
        name='train',
        exist_ok=True
    )
    
    # Export trained model
    model_path = 'models/yolov8n_blade.pt'
    
    # Get the best model path - in YOLOv8 the best weights are saved to a specific path
    best_model_path = os.path.join('runs/sword_detection/train/weights/best.pt')
    
    # Copy the best model to our models directory
    if os.path.exists(best_model_path):
        shutil.copy(best_model_path, model_path)
        print(f"Copied best model from {best_model_path} to {model_path}")
    else:
        # Fallback to using the last weights
        last_model_path = os.path.join('runs/sword_detection/train/weights/last.pt')
        if os.path.exists(last_model_path):
            shutil.copy(last_model_path, model_path)
            print(f"Copied last model from {last_model_path} to {model_path}")
        else:
            # Final fallback: export the current model
            model.export(format='pt')
            exported_path = model.export().path
            shutil.copy(exported_path, model_path)
            print(f"Exported current model to {model_path}")
    
    print(f"Sword detection model trained and saved to {model_path}")
    return model_path


def create_empty_model():
    """
    Create an empty pose classifier model for training
    
    Returns:
        Path to saved empty model
    """
    from fencing_cnn import FencingPoseClassifier
    import torch
    
    # Create model directory
    os.makedirs('models', exist_ok=True)
    
    # Initialize classifier
    classifier = FencingPoseClassifier()
    
    # Save empty model
    model_path = 'models/empty_pose_classifier.pth'
    torch.save(classifier.model.state_dict(), model_path)
    
    print(f"Created empty model at {model_path}")
    return model_path


def prepare_training_data():
    """
    Prepare training data from YOLOv8 dataset for pose classifier
    
    Returns:
        Path to prepared dataset
    """
    from fencing_cnn import create_dataset_from_yolo_dataset
    
    dataset_path = 'fencingdataset/Fencer Detection.v7i.yolov8-obb'
    output_path = 'fencing_pose_dataset'
    
    # Create dataset
    for split in ['train', 'valid']:
        create_dataset_from_yolo_dataset(
            dataset_path=dataset_path,
            output_path=output_path,
            split=split
        )
    
    return output_path


def train_pose_classifier(dataset_path):
    """
    Train pose classifier on prepared dataset
    
    Args:
        dataset_path: Path to prepared dataset
        
    Returns:
        Path to trained model
    """
    from fencing_cnn import FencingPoseClassifier, create_dataloader_from_directory
    
    # Create model directory
    os.makedirs('models', exist_ok=True)
    
    # Create dataloaders
    train_loader, val_loader = create_dataloader_from_directory(
        data_dir=dataset_path,
        batch_size=16,
        train_ratio=0.8
    )
    
    # Initialize classifier
    classifier = FencingPoseClassifier()
    
    # Train model
    model_path = 'models/pose_classifier.pth'
    classifier.train(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=10,
        lr=0.001,
        save_path=model_path
    )
    
    return model_path


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Fencer Detection with Pose Classification")
    parser.add_argument("--mode", choices=['train_pose', 'train_sword', 'test', 'process'], default='process',
                      help="Mode: train_pose, train_sword, test (on image), process (video)")
    parser.add_argument("--input", default=None, 
                      help="Input file: video file for 'process' mode, image file for 'test' mode")
    parser.add_argument("--output", default=None,
                      help="Output file path")
    parser.add_argument("--yolo_model", default=None,
                      help="Path to custom YOLOv8 model")
    parser.add_argument("--pose_model", default=None,
                      help="Path to trained pose classifier model")
    parser.add_argument("--sword_model", default=None,
                      help="Path to trained sword detector model")
    parser.add_argument("--epochs", type=int, default=50,
                      help="Number of training epochs")
    
    args = parser.parse_args()
    
    if args.mode == 'train_pose':
        print("Preparing training data...")
        dataset_path = prepare_training_data()
        
        print("Training pose classifier...")
        model_path = train_pose_classifier(dataset_path)
        
        print(f"Pose classifier trained and saved to {model_path}")
    
    elif args.mode == 'train_sword':
        print("Training sword detector...")
        model_path = train_sword_detector(epochs=args.epochs)
        
        print(f"Sword detector trained and saved to {model_path}")
    
    elif args.mode == 'test':
        if not args.input:
            print("Error: Please provide an input image file with --input")
            exit(1)
        
        # Load the image
        image = cv2.imread(args.input)
        if image is None:
            print(f"Error: Could not read image {args.input}")
            exit(1)
        
        # Initialize detector
        detector = EnhancedFencerDetector(
            model_path=args.yolo_model,
            pose_model_path=args.pose_model,
            sword_model_path=args.sword_model
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
        if args.output:
            cv2.imwrite(args.output, annotated_image)
            print(f"Saved result to {args.output}")
        else:
            cv2.imshow("Enhanced Fencer Detection", annotated_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    elif args.mode == 'process':
        if not args.input:
            print("Error: Please provide an input video file with --input")
            exit(1)
        
        # Set default output path if not specified
        output_path = args.output
        if not output_path:
            output_path = f"enhanced_{os.path.basename(args.input)}"
        
        # Initialize detector
        detector = EnhancedFencerDetector(
            model_path=args.yolo_model,
            pose_model_path=args.pose_model,
            sword_model_path=args.sword_model
        )
        
        # Process video
        results = detector.process_video(
            video_path=args.input,
            output_path=output_path
        )
        
        print(f"Video processed and saved to {output_path}")
    
    else:
        print(f"Error: Unknown mode {args.mode}")
        exit(1) 