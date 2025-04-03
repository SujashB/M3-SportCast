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
    
    def detect_swords(self, frame, conf_threshold=0.10):
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
        alpha = 1.5  # Contrast control (1.0 means no change)
        beta = 15    # Brightness control (0 means no change)
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
        # Create a copy for drawing
        annotated_frame = frame.copy()
        
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
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, self.line_thickness)
                
                # Draw the center line of the blade (approximation)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Calculate the length and angle (rough approximation)
                width = x2 - x1
                height = y2 - y1
                
                # Draw the center line if the blade is long enough
                if max(width, height) > 50:
                    if height > width:  # Vertical blade
                        cv2.line(annotated_frame, (center_x, y1), (center_x, y2), color, self.line_thickness + 1)
                    else:  # Horizontal blade
                        cv2.line(annotated_frame, (x1, center_y), (x2, center_y), color, self.line_thickness + 1)
                
                # Add a glowing effect for enhanced visibility
                blur_amount = 5
                glow_size = 3
                
                # Create a mask for the blade
                mask = np.zeros_like(annotated_frame)
                cv2.rectangle(mask, (x1-glow_size, y1-glow_size), (x2+glow_size, y2+glow_size), color, -1)
                
                # Blur the mask
                mask = cv2.GaussianBlur(mask, (blur_amount*2+1, blur_amount*2+1), 0)
                
                # Blend the mask with the frame to create a glow effect
                alpha = 0.3  # Adjust for glow intensity
                annotated_frame = cv2.addWeighted(annotated_frame, 1.0, mask, alpha, 0)
                
                # Redraw the original box to make it stand out from the glow
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, self.line_thickness)
            
            elif class_name == 'blade-tip':
                # Draw a circle for the tip with glow effect
                radius = min(10, (x2-x1)//2, (y2-y1)//2)
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Create glow effect
                mask = np.zeros_like(annotated_frame)
                cv2.circle(mask, (center_x, center_y), radius+5, color, -1)
                mask = cv2.GaussianBlur(mask, (15, 15), 0)
                
                # Blend the mask with the frame to create a glow effect
                alpha = 0.4  # Adjust for glow intensity
                annotated_frame = cv2.addWeighted(annotated_frame, 1.0, mask, alpha, 0)
                
                # Draw the actual circle
                cv2.circle(annotated_frame, (center_x, center_y), radius, color, self.line_thickness)
                
                # Add a small dot in the center for emphasis
                cv2.circle(annotated_frame, (center_x, center_y), 2, (255, 255, 255), -1)
            
            elif class_name == 'blade-guard':
                # Draw a filled box for the guard
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, self.line_thickness)
                
                # Add glow effect
                mask = np.zeros_like(annotated_frame)
                cv2.rectangle(mask, (x1-3, y1-3), (x2+3, y2+3), color, -1)
                mask = cv2.GaussianBlur(mask, (15, 15), 0)
                
                # Blend the mask with the frame
                alpha = 0.3  # Adjust for glow intensity
                annotated_frame = cv2.addWeighted(annotated_frame, 1.0, mask, alpha, 0)
                
                # Add cross lines
                cv2.line(annotated_frame, (x1, y1), (x2, y2), color, 1)
                cv2.line(annotated_frame, (x1, y2), (x2, y1), color, 1)
            
            # Draw label with higher contrast
            label = f"{class_name}: {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
            
            # Draw label background with full opacity
            cv2.rectangle(annotated_frame, (x1, y1-20), (x1+text_size[0]+10, y1), color, -1)
            
            # Draw label text with black color
            cv2.putText(annotated_frame, label, (x1+5, y1-5), 
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
            
        return annotated_frame


class EnhancedFencerDetector(FencerDetector):
    """
    Enhanced fencer detector that combines YOLOv8 with a CNN pose classifier and sword detection
    """
    def __init__(self, model_path=None, pose_model_path=None, sword_model_path=None, bout_mode=True):
        """
        Initialize enhanced fencer detector
        
        Args:
            model_path: Path to custom trained YOLOv8 model. If None, uses pretrained model.
            pose_model_path: Path to trained pose classifier model
            sword_model_path: Path to trained sword detector model. If None, sword detection is disabled.
            bout_mode: Whether to optimize for bout detection (2 fencers)
        """
        # Initialize base FencerDetector
        super().__init__(model_path)
        
        # Initialize pose classifier
        self.pose_classifier = FencingPoseClassifier(
            model_path=pose_model_path,
            device=self.device
        )
        
        # Initialize sword detector if model path is provided
        if sword_model_path:
            self.sword_detector = SwordDetector(model_path=sword_model_path)
            print(f"Sword detector initialized with model: {sword_model_path}")
        else:
            self.sword_detector = None
            print("Sword detection disabled")
        
        # Track pose history for each fencer
        self.pose_history = {}
        
        # Track box history for each fencer (to reduce flickering)
        self.box_history = {}
        
        # Track sword detection history
        self.sword_history = []
        self.max_sword_history = 10
        
        # Set bout mode (optimizes for 2 fencers)
        self.bout_mode = bout_mode
        
        # Define pose colors for visualization
        self.pose_colors = {
            'neutral': (255, 255, 255),  # White
            'attack': (0, 0, 255),       # Red (BGR)
            'defense': (0, 255, 0),      # Green
            'lunge': (255, 0, 0)         # Blue
        }
    
    def detect_fencers(self, frame, conf_threshold=0.3):
        """
        Detect fencers in a frame using YOLOv8 with bout-optimized processing
        
        Args:
            frame: Input video frame (BGR)
            conf_threshold: Confidence threshold for detections
            
        Returns:
            detections: List of fencer detections
        """
        # First use the base class method to get initial detections
        detections = super().detect_fencers(frame, conf_threshold)
        
        # Filter detections based on fencing-specific criteria
        filtered_detections = []
        
        for detection in detections:
            box = detection['box']
            x1, y1, x2, y2 = box
            
            # Calculate aspect ratio of the bounding box
            width = x2 - x1
            height = y2 - y1
            aspect_ratio = width / height if height > 0 else 0
            
            # Fencers typically have a certain aspect ratio (height > width)
            # and occupy a significant portion of the frame height
            is_likely_fencer = (
                aspect_ratio < 0.8 and  # Taller than wide
                height > frame.shape[0] * 0.15 and  # Take up reasonable height
                height < frame.shape[0] * 0.9  # Not too tall (full frame)
            )
            
            if is_likely_fencer:
                # Extract fencer crop for CNN classification
                fencer_crop = frame[int(y1):int(y2), int(x1):int(x2)]
                
                # Ensure crop is valid
                if fencer_crop.size == 0:
                    continue
                
                # Use pose classifier to check if this looks like a fencer stance
                class_id, pose_class, pose_confidence = self.pose_classifier.classify_pose(fencer_crop)
                
                # Boost confidence for detections with recognized fencing poses
                if pose_class in ['attack', 'defense', 'lunge']:
                    detection['confidence'] *= 1.5  # Boost confidence
                
                # Check for distinctive fencing equipment
                # This could be improved with a dedicated equipment detector
                has_equipment = False
                
                # For now, we'll use a simple heuristic
                # Add detection with additional fencer-specific data
                detection['pose_class'] = pose_class
                detection['pose_confidence'] = pose_confidence
                detection['has_equipment'] = has_equipment
                
                filtered_detections.append(detection)
        
        # If in bout mode and we have exactly 2 detections, we're done
        if self.bout_mode and len(filtered_detections) == 2:
            return filtered_detections
        
        # If in bout mode and we don't have 2 detections, try to recover
        if self.bout_mode:
            # If we have more than 2 detections, keep the 2 most confident ones
            if len(filtered_detections) > 2:
                # Sort detections by confidence
                filtered_detections.sort(key=lambda x: x['confidence'], reverse=True)
                # Keep only the 2 most confident detections
                filtered_detections = filtered_detections[:2]
            
            # If we have fewer than 2 detections, try lowering the threshold
            if len(filtered_detections) < 2:
                # Try with a lower threshold to see if we can find a second fencer
                lower_threshold = conf_threshold * 0.7  # Use 70% of the original threshold
                low_conf_detections = super().detect_fencers(frame, lower_threshold)
                
                # Apply the same filtering criteria
                low_conf_filtered = []
                for detection in low_conf_detections:
                    box = detection['box']
                    x1, y1, x2, y2 = box
                    
                    width = x2 - x1
                    height = y2 - y1
                    aspect_ratio = width / height if height > 0 else 0
                    
                    is_likely_fencer = (
                        aspect_ratio < 0.8 and 
                        height > frame.shape[0] * 0.15 and
                        height < frame.shape[0] * 0.9
                    )
                    
                    if is_likely_fencer:
                        low_conf_filtered.append(detection)
                
                # Sort by confidence
                low_conf_filtered.sort(key=lambda x: x['confidence'], reverse=True)
                
                # If we have exactly 0 detections, take the top 2 from low confidence
                if len(filtered_detections) == 0 and len(low_conf_filtered) >= 2:
                    filtered_detections = low_conf_filtered[:2]
                # If we have 1 detection, see if we can add a second one that doesn't overlap
                elif len(filtered_detections) == 1 and len(low_conf_filtered) > 0:
                    # Get the current detection's box
                    box1 = filtered_detections[0]['box']
                    
                    # Find a second detection that doesn't overlap much with the first
                    for det in low_conf_filtered:
                        box2 = det['box']
                        # Skip if it's the same as our high-confidence detection
                        if self._boxes_are_same(box1, box2):
                            continue
                        
                        # Check if the IoU is low (boxes don't overlap much)
                        iou = self._calculate_iou(box1, box2)
                        if iou < 0.3:  # Low overlap
                            filtered_detections.append(det)
                            break
        
        return filtered_detections
    
    def _boxes_are_same(self, box1, box2, tolerance=10):
        """Check if two boxes are approximately the same"""
        for i in range(4):
            if abs(box1[i] - box2[i]) > tolerance:
                return False
        return True
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union between two boxes"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        return intersection / (area1 + area2 - intersection + 1e-6)
    
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
        sword_detections = self.sword_detector.detect_swords(frame, conf_threshold=0.10)
        
        # Extract bounding boxes
        boxes = [det['box'] for det in detections]
        
        # Extract crops of detected fencers
        if boxes:
            crops = extract_fencer_crops(frame, boxes)
            
            # Classify pose for each crop
            for i, crop in enumerate(crops):
                try:
                    # Classify pose
                    class_id, pose_class, pose_confidence = self.pose_classifier.classify_pose(crop)
                    
                    # Add pose classification to detection
                    detections[i]['pose_class_id'] = class_id
                    detections[i]['pose_class'] = pose_class
                    detections[i]['pose_confidence'] = pose_confidence
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
    
    def track_and_classify(self, frame, prev_boxes=None):
        """
        Track fencers in a frame and classify their poses
        
        Args:
            frame: Input video frame
            prev_boxes: Optional list of previous frame's bounding boxes
            
        Returns:
            tracked_items: List of tracked items with pose classifications
            sword_detections: List of sword detections
        """
        # 1. Detect fencers with YOLOv8
        detections = self.detect_fencers(frame)
        
        # 2. Track fencers across frames
        tracked_boxes = self.track_fencers(frame, detections)
        
        # 3. Detect swords if sword detector is available
        sword_detections = []
        if self.sword_detector:
            sword_detections = self.sword_detector.detect_swords(frame)
            
            # Store sword detection history
            self.sword_history.append(sword_detections)
            if len(self.sword_history) > self.max_sword_history:
                self.sword_history.pop(0)
        
        # 4. Classify poses for each tracked fencer
        tracked_items = []
        
        for fencer_id, box in tracked_boxes:
            # Extract fencer crop from frame
            x1, y1, x2, y2 = map(int, box)
            # Ensure coordinates are within frame bounds
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                continue  # Skip invalid boxes
                
            fencer_crop = frame[y1:y2, x1:x2]
            
            # Classify pose using CNN
            class_id, pose_class, pose_confidence = self.pose_classifier.classify_pose(fencer_crop)
            
            # For better stability, maintain a history of pose classifications for each fencer
            if fencer_id not in self.pose_history:
                self.pose_history[fencer_id] = []
            
            # Add current prediction to history
            self.pose_history[fencer_id].append((pose_class, pose_confidence))
            
            # Keep history at a reasonable size (last 5 frames)
            if len(self.pose_history[fencer_id]) > 5:
                self.pose_history[fencer_id].pop(0)
            
            # Weight recent predictions more heavily
            if len(self.pose_history[fencer_id]) >= 3:
                # Count occurrences with more weight to recent frames
                pose_counts = {}
                total_weight = 0
                
                for i, (cls, conf) in enumerate(self.pose_history[fencer_id]):
                    # Exponential weighting - more recent frames have higher weight
                    weight = conf * (1.5 ** i)
                    total_weight += weight
                    
                    if cls not in pose_counts:
                        pose_counts[cls] = 0
                    pose_counts[cls] += weight
                
                # Normalize weights and find most likely pose
                for cls in pose_counts:
                    pose_counts[cls] /= total_weight
                
                # Get the pose with highest weighted probability
                final_pose_class = max(pose_counts, key=pose_counts.get)
                final_pose_confidence = pose_counts[final_pose_class]
            else:
                # Not enough history, use current prediction
                final_pose_class = pose_class
                final_pose_confidence = pose_confidence
            
            # Create tracked item with all information
            item = {
                'fencer_id': fencer_id,
                'box': box,
                'pose_class': final_pose_class,
                'pose_confidence': final_pose_confidence,
                'confidence': detections[0]['confidence'] if detections else 0.5  # Default if no detections
            }
            
            tracked_items.append(item)
        
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
        if sword_detections and self.sword_detector:
            annotated_frame = self.sword_detector.draw_detections(annotated_frame, sword_detections)
            # Draw a small indicator to show sword detection is active
            cv2.putText(annotated_frame, "Sword Detection: ACTIVE", 
                       (10, frame.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 
                       0.5, (0, 255, 255), 1)
        
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
            
            # Add special visual effects for lunge detection
            if pose_class == 'lunge':
                # Create a pulsing glow effect for lunges
                # Create a mask for the glow
                mask = np.zeros_like(annotated_frame)
                # Create an expanded box for the glow
                glow_size = 10
                cv2.rectangle(mask, (x1-glow_size, y1-glow_size), (x2+glow_size, y2+glow_size), color, -1)
                # Blur the mask
                mask = cv2.GaussianBlur(mask, (21, 21), 0)
                # Add the glow to the frame
                alpha = 0.4  # Adjust for glow intensity
                annotated_frame = cv2.addWeighted(annotated_frame, 1.0, mask, alpha, 0)
                
                # Add arrows indicating movement direction for lunges
                arrow_length = 30
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
                
                # Draw arrows in multiple directions
                cv2.arrowedLine(annotated_frame, (center_x, center_y), (center_x + arrow_length, center_y), color, 2, tipLength=0.3)
                
                # Make the box thicker for lunges
                box_thickness = 4
            else:
                box_thickness = 2
            
            # Draw bounding box with pose-specific color
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, box_thickness)
            
            # Determine text background rectangle dimensions
            text_thickness = 1
            text_size_id = cv2.getTextSize(f"Fencer {fencer_id}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_thickness)[0]
            
            # Make pose label more prominent with larger font and bold text
            pose_text = f"POSE: {pose_class.upper()}"
            pose_text_size = cv2.getTextSize(pose_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            text_size_conf = cv2.getTextSize(f"Conf: {pose_confidence:.2f}", cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_thickness)[0]
            
            # Get the max width needed for the text background
            max_width = max(text_size_id[0], pose_text_size[0], text_size_conf[0]) + 10
            
            # Draw background rectangle for text with full opacity
            cv2.rectangle(annotated_frame, (x1, y1-75), (x1+max_width, y1), color, -1)
            
            # Draw text with better contrast
            text_color = (0, 0, 0)  # Black text for better visibility
            cv2.putText(annotated_frame, f"Fencer {fencer_id}", (x1+5, y1-50),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, text_thickness)
            
            # Draw pose label with emphasis
            cv2.putText(annotated_frame, pose_text, (x1+5, y1-25),
                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)
            
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