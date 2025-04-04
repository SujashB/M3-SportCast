import cv2
import numpy as np
import torch
from pathlib import Path
import time
from tqdm import tqdm
import os
import yaml
import shutil
from torchvision.models import mobilenet_v2
from ultralytics import YOLO
from collections import defaultdict

# Import base FencerDetector
from fencer_detector import FencerDetector
# Import our CNN pose classifier
from fencing_cnn import FencingPoseClassifier, extract_fencer_crops

class EnhancedFencerDetector(FencerDetector):
    """
    Enhanced fencer detector that combines YOLOv8 with a CNN pose classifier
    """
    def __init__(self, model_path=None, pose_model_path=None):
        """Initialize the detector
        
        Args:
            model_path: Path to trained YOLOv8 model
            pose_model_path: Path to trained CNN pose classifier model
        """
        super().__init__(model_path)
        
        # Initialize pose classifier
        self.pose_classifier = None
        if pose_model_path and os.path.exists(pose_model_path):
            print("Loading pose classifier from", pose_model_path)
            try:
                # Instantiate the model first
                # The FencingPoseClassifier class wraps the FencingPoseCNN
                self.pose_classifier = FencingPoseClassifier() # Instantiate the wrapper
                # Load the state dictionary into the underlying CNN model
                state_dict = torch.load(pose_model_path, map_location=torch.device('cpu'))
                # Load into the actual nn.Module instance within the wrapper
                self.pose_classifier.model.load_state_dict(state_dict)
                # Set the underlying model to evaluation mode
                self.pose_classifier.model.eval()
                print("Pose classifier model loaded successfully.")
            except Exception as e:
                print(f"Error loading pose classifier model: {e}")
                print("Initializing pose classifier with random weights instead.")
                self.pose_classifier = FencingPoseClassifier() # Fallback
        else:
            # This block is now only for when no pose_model_path is given
            print("Initializing pose classifier with random weights (no model path provided)")
            self.pose_classifier = FencingPoseClassifier()
        
        print("Enhanced fencer detector initialized with CNN pose classifier")
        
        # Initialize pose history for temporal smoothing
        self.pose_history = {}
        
        # Track box history for each fencer (to reduce flickering)
        self.box_history = {}
        
        # Define pose colors for visualization
        self.pose_colors = {
            'neutral': (255, 255, 255),  # White
            'attack': (0, 0, 255),       # Red (BGR)
            'defense': (0, 255, 0),      # Green
            'lunge': (255, 0, 0)         # Blue
        }
        
        # Add specific fencer IDs to track (defaults to None, meaning track all)
        self.target_fencer_ids = None
    
    def set_target_fencers(self, fencer_ids):
        """
        Set specific fencer IDs to track
        
        Args:
            fencer_ids: List of fencer IDs to track, or None to track all
        """
        if fencer_ids:
            print(f"Setting target fencers to IDs: {fencer_ids}")
            self.target_fencer_ids = fencer_ids
        else:
            print("Tracking all detected fencers")
            self.target_fencer_ids = None
    
    def detect_and_classify(self, frame, conf_threshold=0.3):
        """
        Detect fencers and classify their poses with the CNN model
        
        Args:
            frame: Input video frame
            conf_threshold: Confidence threshold for detection
            
        Returns:
            detections: List of detected fencers with pose classifications
        """
        # Detect fencers using the base detector
        base_detections = self.detect_fencers(frame, conf_threshold)
        
        # Process each detection with the pose classifier
        detections = []
        for det in base_detections:
            # Extract person crop
            box = det['box']
            x1, y1, x2, y2 = map(int, box)
            
            # Ensure box is within frame
            height, width = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            # Skip if invalid box
            if x1 >= x2 or y1 >= y2 or x2 <= 0 or y2 <= 0:
                continue
            
            person_crop = frame[y1:y2, x1:x2]
            
            # Skip if empty crop
            if person_crop.size == 0:
                continue
            
            # Classify pose if classifier is available
            pose_class = 'neutral'
            pose_confidence = 0.0
            
            if self.pose_classifier is not None:
                try:
                    # Assuming pose_classifier has a classify_pose method
                    pose_class, pose_confidence = self.pose_classifier.classify_pose(person_crop)
                except Exception as e:
                    print(f"Error classifying pose: {e}")
            
            # Add detection with pose classification
            det['pose_class'] = pose_class
            det['pose_confidence'] = float(pose_confidence)
            detections.append(det)
        
        return detections # Return only fencer detections with poses
    
    def box_inside(self, box1, box2):
        """
        Check if box1 is fully inside box2
        
        Args:
            box1: First box coordinates [x1, y1, x2, y2]
            box2: Second box coordinates [x1, y1, x2, y2]
            
        Returns:
            True if box1 is inside box2, False otherwise
        """
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        return (x1_1 >= x1_2 and y1_1 >= y1_2 and x2_1 <= x2_2 and y2_1 <= y2_2)
    
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
    
    def track_and_classify(self, frame):
        """Track fencers and classify their poses
        
        Args:
            frame: Input frame
            
        Returns:
            tracked_items: List of tracked fencer detections with poses
        """
        # Get fencer detections and track using base class method
        # This returns items with 'box', 'conf', 'class_id', and importantly 'track_id' (renamed to 'fencer_id')
        tracked_items, _ = super().detect_and_classify(frame) # Call base method
        
        # Now, iterate through tracked items and add CNN pose classification
        final_tracked_items = []
        for item in tracked_items:
            box = item.get('box')
            if box is None:
                continue
            
            x1, y1, x2, y2 = map(int, box)
            height, width = frame.shape[:2]
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(width, x2)
            y2 = min(height, y2)
            
            if x1 >= x2 or y1 >= y2 or x2 <= 0 or y2 <= 0:
                continue
            
            person_crop = frame[y1:y2, x1:x2]
            if person_crop.size == 0:
                continue
            
            # Classify pose using the CNN model
            pose_class = 'neutral'
            pose_confidence = 0.0
            if self.pose_classifier is not None:
                try:
                    pose_class, pose_confidence = self.pose_classifier.classify_pose(person_crop)
                except Exception as e:
                    print(f"Error classifying pose for fencer {item.get('fencer_id', 'N/A')}: {e}")
            
            # Add pose info to the item
            item['pose_class'] = pose_class
            item['pose_confidence'] = float(pose_confidence)
            final_tracked_items.append(item)
        
        return final_tracked_items # Return only tracked fencers with poses
    
    def draw_enhanced_detections(self, frame, tracked_items):
        """
        Draw enhanced detection visualizations including pose classifications
        
        Args:
            frame: Input video frame
            tracked_items: List of tracked fencers with pose classifications
            
        Returns:
            annotated_frame: Frame with visualizations
        """
        annotated_frame = frame.copy()
        height, width = annotated_frame.shape[:2]
        
        # Draw grid lines (optional, keep for now)
        grid_color = (70, 70, 200)
        grid_alpha = 0.5
        grid_size = 7
        grid_frame = annotated_frame.copy()
        for i in range(1, grid_size):
            x = int(width * i / grid_size)
            cv2.line(grid_frame, (x, 0), (x, height), grid_color, 1)
        for i in range(1, grid_size):
            y = int(height * i / grid_size)
            cv2.line(grid_frame, (0, y), (width, y), grid_color, 1)
        cv2.addWeighted(grid_frame, grid_alpha, annotated_frame, 1 - grid_alpha, 0, annotated_frame)
        
        # Draw fencer detections
        for item in tracked_items:
            # Extract information
            fencer_id = item.get('fencer_id', item.get('id', -1))
            box = item.get('box', item.get('bbox', None))
            pose_class = item.get('pose_class', 'neutral')
            pose_confidence = item.get('pose_confidence', 0.0)
            
            if box is not None:
                x1, y1, x2, y2 = map(int, box)
                
                # Different color based on fencer ID for consistent visualization
                colors = [
                    (0, 255, 0),    # Green for fencer 0
                    (255, 0, 0),    # Blue for fencer 1
                    (0, 0, 255),    # Red for fencer 2
                    (255, 255, 0),  # Cyan for fencer 3
                    (255, 0, 255),  # Magenta for fencer 4
                    (0, 255, 255),  # Yellow for fencer 5
                ]
                
                color_idx = fencer_id % len(colors)
                color = colors[color_idx]
                
                # Make box thicker for better visibility
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 3)
                
                # Draw fencer ID and pose information
                fencer_label = f"Fencer {fencer_id}"
                pose_label = f"Pose: {pose_class}"
                conf_label = f"Conf: {pose_confidence:.2f}"
                
                # Position labels inside the box for better visibility
                cv2.putText(annotated_frame, fencer_label, (x1+5, y1+20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                cv2.putText(annotated_frame, pose_label, (x1+5, y1+45),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                # Optionally display pose confidence
                # cv2.putText(annotated_frame, conf_label, (x1+5, y1+70),
                #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Draw keypoints if available (assuming PoseEstimator adds them elsewhere)
                if 'keypoints' in item:
                    # Need to ensure draw_keypoints is available or implement it here/import it
                    # from pose_estimation_helper import draw_keypoints # Example import
                    # draw_keypoints(annotated_frame, item['keypoints'])
                    pass # Placeholder - drawing happens in process_video
        
        return annotated_frame
    
    def process_video(self, video_path, output_path=None, max_frames=None, init_fencer_boxes=None):
        # This method seems complex and might need adjustments based on changes
        # For now, assume the caller handles the changes in return values from track_and_classify
        # Let's review this method in advanced_fencing_analyzer.py context later
        pass # Placeholder - actual processing logic is in advanced_fencing_analyzer.py

# Need to ensure FencingPoseClassifier is defined or imported correctly
# class FencingPoseClassifier: # Placeholder if not imported
#     def __init__(self, num_classes=4): self.num_classes = num_classes
#     def classify_pose(self, crop): return 'neutral', 0.0


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
            pose_model_path=args.pose_model
        )
        
        # Detect and classify
        tracked_items = detector.track_and_classify(image)
        
        # Draw results
        annotated_image = detector.draw_enhanced_detections(
            image.copy(), 
            tracked_items
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
            pose_model_path=args.pose_model
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