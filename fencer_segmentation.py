import os
import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import time
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore")

# Check if torch with CUDA is available
CUDA_AVAILABLE = torch.cuda.is_available()
if CUDA_AVAILABLE:
    print(f"CUDA available: {CUDA_AVAILABLE}, using GPU")
else:
    print(f"CUDA not available, using CPU (segmentation will be slower)")

# Try to import SAMURAI model (Segment Anything Model with User Refinement for Action Investigation)
# If not available, we'll use standard OpenCV segmentation methods
try:
    from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
    SAM_AVAILABLE = True
    print("SAM model available for advanced segmentation")
except ImportError:
    SAM_AVAILABLE = False
    print("SAM model not available, using basic OpenCV segmentation")
    print("To install SAM: pip install segment-anything")

# Try to import pycocotools for mask operations
try:
    from pycocotools import mask as mask_utils
    PYCOCOTOOLS_AVAILABLE = True
except ImportError:
    PYCOCOTOOLS_AVAILABLE = False
    print("pycocotools not available, using basic mask operations")
    print("To install: pip install pycocotools")

class FencerSegmentation:
    """
    Class for segmenting fencers in videos using SAM model or fall back to basic techniques
    """
    def __init__(self, sam_checkpoint=None, model_type="vit_h"):
        """
        Initialize segmentation model
        
        Args:
            sam_checkpoint: Path to SAM model checkpoint
            model_type: SAM model type ('vit_h', 'vit_l', 'vit_b')
        """
        self.sam_available = SAM_AVAILABLE
        self.device = torch.device("cuda" if CUDA_AVAILABLE else "cpu")
        self.sam_predictor = None
        self.mask_generator = None
        
        # Try to initialize SAM model if available
        if SAM_AVAILABLE and sam_checkpoint and os.path.exists(sam_checkpoint):
            try:
                print(f"Loading SAM model from {sam_checkpoint}...")
                sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
                sam.to(self.device)
                
                # Initialize predictor and mask generator
                self.sam_predictor = SamPredictor(sam)
                self.mask_generator = SamAutomaticMaskGenerator(
                    model=sam,
                    points_per_side=32,
                    pred_iou_thresh=0.9,
                    stability_score_thresh=0.8,
                    crop_n_layers=1,
                    crop_n_points_downscale_factor=2,
                    min_mask_region_area=100  # Smaller value for smaller objects
                )
                print("SAM model loaded successfully")
            except Exception as e:
                print(f"Error loading SAM model: {e}")
                self.sam_available = False
        elif SAM_AVAILABLE:
            print("No SAM checkpoint provided. Using model without fine-tuning.")
            try:
                sam = sam_model_registry[model_type]()
                sam.to(self.device)
                self.sam_predictor = SamPredictor(sam)
                print("Base SAM model loaded")
            except Exception as e:
                print(f"Error loading base SAM model: {e}")
                self.sam_available = False
        
        # Initialize OpenCV background subtractor as fallback
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=50, detectShadows=False
        )
        
        # Tracking attributes
        self.fencer_history = []  # Track bounding boxes over time
        self.fencer_masks = []    # Track masks over time
        self.fencer_ids = []      # Track fencer IDs
        self.next_id = 1          # ID counter for new fencers
    
    def segment_fencers_sam(self, frame):
        """
        Segment fencers in a frame using SAM model
        
        Args:
            frame: Input video frame (BGR)
            
        Returns:
            masks: List of segmentation masks
            boxes: List of bounding boxes
        """
        if not self.sam_available or self.mask_generator is None:
            return None, None
        
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Generate masks
        masks = self.mask_generator.generate(rgb_frame)
        
        # Filter masks to keep only likely fencer masks
        filtered_masks = []
        boxes = []
        
        for mask in masks:
            # Extract mask information
            segmentation = mask["segmentation"]
            area = mask["area"]
            bbox = mask["bbox"]  # x, y, w, h
            
            # Filter criteria (adjust based on your needs)
            # Keep masks that are reasonably sized (not too small, not too large)
            if area > 1000 and area < 0.7 * frame.shape[0] * frame.shape[1]:
                # Convert bbox from xywh to xyxy
                x, y, w, h = bbox
                box = [x, y, x + w, y + h]
                
                filtered_masks.append(segmentation)
                boxes.append(box)
        
        return filtered_masks, boxes
    
    def segment_fencers_opencv(self, frame, history_frames=None):
        """
        Segment fencers in a frame using OpenCV
        
        Args:
            frame: Input video frame (BGR)
            history_frames: Optional list of previous frames for better segmentation
            
        Returns:
            masks: List of segmentation masks
            boxes: List of bounding boxes
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours
        filtered_contours = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 500:  # Minimum area threshold
                filtered_contours.append(contour)
        
        # Create masks and bounding boxes
        masks = []
        boxes = []
        
        for contour in filtered_contours:
            # Create mask for this contour
            mask = np.zeros_like(fg_mask)
            cv2.drawContours(mask, [contour], 0, 255, -1)
            
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)
            
            # Only keep reasonably sized detections
            if w > 30 and h > 60:  # Minimum width and height
                masks.append(mask > 0)
                boxes.append([x, y, x + w, y + h])
        
        return masks, boxes
    
    def segment_frame(self, frame):
        """
        Segment fencers in a frame using the best available method
        
        Args:
            frame: Input video frame (BGR)
            
        Returns:
            masks: List of segmentation masks
            boxes: List of bounding boxes
        """
        # Try SAM first if available
        if self.sam_available and self.mask_generator is not None:
            try:
                masks, boxes = self.segment_fencers_sam(frame)
                if masks and len(masks) > 0:
                    return masks, boxes
            except Exception as e:
                print(f"SAM segmentation failed: {e}")
        
        # Fall back to OpenCV method
        return self.segment_fencers_opencv(frame)
    
    def track_fencers(self, frame, masks, boxes):
        """
        Track fencers across frames
        
        Args:
            frame: Current video frame
            masks: List of segmentation masks
            boxes: List of bounding boxes
            
        Returns:
            tracked_boxes: List of tracked bounding boxes with IDs
            tracked_masks: List of tracked masks with IDs
        """
        tracked_boxes = []
        tracked_masks = []
        
        # If this is the first frame, initialize trackers
        if not self.fencer_history:
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                fencer_id = self.next_id
                self.next_id += 1
                
                self.fencer_history.append({
                    'id': fencer_id,
                    'box': box,
                    'missing_frames': 0
                })
                
                tracked_boxes.append((fencer_id, box))
                tracked_masks.append((fencer_id, mask))
        else:
            # For each detected object, find the closest match in history
            matched_indices = set()
            
            for i, (mask, box) in enumerate(zip(masks, boxes)):
                x1, y1, x2, y2 = box
                box_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                
                best_match = None
                min_distance = float('inf')
                
                for j, tracker in enumerate(self.fencer_history):
                    if j in matched_indices:
                        continue
                    
                    hist_box = tracker['box']
                    hist_x1, hist_y1, hist_x2, hist_y2 = hist_box
                    hist_center = ((hist_x1 + hist_x2) / 2, (hist_y1 + hist_y2) / 2)
                    
                    # Calculate distance between centers
                    distance = np.sqrt((box_center[0] - hist_center[0])**2 + 
                                     (box_center[1] - hist_center[1])**2)
                    
                    # Calculate IoU overlap
                    iou = self._calculate_iou(box, hist_box)
                    
                    # Use a combination of distance and IoU for matching
                    match_score = distance * (1 - iou)
                    
                    if match_score < min_distance and match_score < 200:  # Threshold
                        min_distance = match_score
                        best_match = j
                
                if best_match is not None:
                    # Update tracker with new position
                    self.fencer_history[best_match]['box'] = box
                    self.fencer_history[best_match]['missing_frames'] = 0
                    matched_indices.add(best_match)
                    
                    # Add to tracked results
                    fencer_id = self.fencer_history[best_match]['id']
                    tracked_boxes.append((fencer_id, box))
                    tracked_masks.append((fencer_id, mask))
                else:
                    # New object detected
                    fencer_id = self.next_id
                    self.next_id += 1
                    
                    self.fencer_history.append({
                        'id': fencer_id,
                        'box': box,
                        'missing_frames': 0
                    })
                    
                    tracked_boxes.append((fencer_id, box))
                    tracked_masks.append((fencer_id, mask))
            
            # Update missing frames count for unmatched trackers
            for j, tracker in enumerate(self.fencer_history):
                if j not in matched_indices:
                    tracker['missing_frames'] += 1
            
            # Remove trackers that have been missing for too long
            self.fencer_history = [
                tracker for tracker in self.fencer_history 
                if tracker['missing_frames'] < 10  # Keep if missing for less than 10 frames
            ]
        
        return tracked_boxes, tracked_masks
    
    def _calculate_iou(self, box1, box2):
        """Calculate intersection over union between two boxes"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection coordinates
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Check if boxes overlap
        if x2_i < x1_i or y2_i < y1_i:
            return 0.0
        
        # Calculate areas
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0.0
        return iou
    
    def draw_segmentation(self, frame, tracked_boxes, tracked_masks=None, draw_ids=True):
        """
        Draw segmentation results on frame
        
        Args:
            frame: Input video frame
            tracked_boxes: List of (fencer_id, box) tuples
            tracked_masks: Optional list of (fencer_id, mask) tuples
            draw_ids: Whether to draw fencer IDs
            
        Returns:
            output_frame: Frame with visualization
        """
        output_frame = frame.copy()
        
        # Draw masks if available
        if tracked_masks:
            # Create a color overlay
            overlay = output_frame.copy()
            
            for fencer_id, mask in tracked_masks:
                # Generate a unique color for this fencer
                color = self._id_to_color(fencer_id)
                
                # Apply mask
                if isinstance(mask, np.ndarray) and mask.shape[:2] == frame.shape[:2]:
                    overlay[mask > 0] = color
            
            # Blend the overlay with original image
            cv2.addWeighted(overlay, 0.4, output_frame, 0.6, 0, output_frame)
        
        # Draw bounding boxes and IDs
        for fencer_id, box in tracked_boxes:
            x1, y1, x2, y2 = [int(coord) for coord in box]
            
            # Generate a unique color for this fencer
            color = self._id_to_color(fencer_id)
            
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw ID
            if draw_ids:
                cv2.putText(output_frame, f"Fencer {fencer_id}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
        
        return output_frame
    
    def _id_to_color(self, fencer_id):
        """Convert fencer ID to a unique color"""
        # Use a different hue for each fencer
        hue = (fencer_id * 137) % 180  # Multiply by prime for better distribution
        
        # Use HSV to get a color with full saturation and value
        hsv_color = np.array([[[hue, 255, 255]]], dtype=np.uint8)
        
        # Convert to BGR
        bgr_color = cv2.cvtColor(hsv_color, cv2.COLOR_HSV2BGR)[0, 0]
        
        # Convert to tuple
        return tuple(map(int, bgr_color))
    
    def get_fencer_features(self, frame, tracked_boxes, tracked_masks):
        """
        Extract features for each tracked fencer
        
        Args:
            frame: Current video frame
            tracked_boxes: List of (fencer_id, box) tuples
            tracked_masks: List of (fencer_id, mask) tuples
            
        Returns:
            features: Dict of fencer features by ID
        """
        features = {}
        
        for (fencer_id, box), (_, mask) in zip(tracked_boxes, tracked_masks):
            x1, y1, x2, y2 = box
            
            # Apply mask to get only the fencer pixels
            masked_frame = frame.copy()
            
            # Ensure mask is boolean and same shape as frame
            if isinstance(mask, np.ndarray) and mask.shape[:2] == frame.shape[:2]:
                # Create a 3-channel mask
                mask_3d = np.stack([mask] * 3, axis=2)
                # Zero out non-fencer pixels
                masked_frame = np.where(mask_3d, masked_frame, 0)
            
            # Crop to bounding box
            cropped = masked_frame[int(y1):int(y2), int(x1):int(x2)]
            
            # Skip invalid crops
            if cropped.size == 0:
                continue
                
            # Extract basic features
            fencer_features = {
                'position': ((x1 + x2) / 2, (y1 + y2) / 2),  # Center position
                'size': (x2 - x1, y2 - y1),                  # Width and height
                'box': box,
                'area': np.sum(mask) if isinstance(mask, np.ndarray) else (x2 - x1) * (y2 - y1),
                'aspect_ratio': (x2 - x1) / (y2 - y1) if (y2 - y1) > 0 else 0
            }
            
            # Add to features dict
            features[fencer_id] = fencer_features
        
        return features
    
    def process_video(self, video_path, output_path=None, max_frames=None, progress_bar=True):
        """
        Process a video to segment and track fencers
        
        Args:
            video_path: Path to input video
            output_path: Path to output video (if None, no video is saved)
            max_frames: Maximum number of frames to process (None for all)
            progress_bar: Whether to show progress bar
            
        Returns:
            fencer_data: Dict of fencer data including trajectories
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if max_frames is None:
            max_frames = total_frames
        
        # Initialize output video writer if needed
        out = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Reset tracking state
        self.fencer_history = []
        self.next_id = 1
        
        # Data for each fencer
        fencer_data = {
            'trajectories': {},  # Fencer ID -> list of positions
            'frames': {},        # Fencer ID -> list of frame indices
            'boxes': {},         # Fencer ID -> list of boxes
            'metadata': {
                'video_path': video_path,
                'fps': fps,
                'width': width,
                'height': height,
                'processed_frames': 0
            }
        }
        
        # Process frames
        frame_idx = 0
        
        # Create progress bar if requested
        pbar = tqdm(total=min(max_frames, total_frames), 
                   desc="Processing video") if progress_bar else None
        
        try:
            while cap.isOpened() and frame_idx < max_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process every 2nd frame for speed
                if frame_idx % 2 == 0:
                    # Segment fencers
                    masks, boxes = self.segment_frame(frame)
                    
                    # Track fencers
                    if masks and boxes:
                        tracked_boxes, tracked_masks = self.track_fencers(frame, masks, boxes)
                        
                        # Extract features
                        fencer_features = self.get_fencer_features(frame, tracked_boxes, tracked_masks)
                        
                        # Save trajectories
                        for fencer_id, features in fencer_features.items():
                            if fencer_id not in fencer_data['trajectories']:
                                fencer_data['trajectories'][fencer_id] = []
                                fencer_data['frames'][fencer_id] = []
                                fencer_data['boxes'][fencer_id] = []
                            
                            fencer_data['trajectories'][fencer_id].append(features['position'])
                            fencer_data['frames'][fencer_id].append(frame_idx)
                            fencer_data['boxes'][fencer_id].append(features['box'])
                        
                        # Draw visualization and save to output video
                        if out is not None:
                            output_frame = self.draw_segmentation(frame, tracked_boxes, tracked_masks)
                            out.write(output_frame)
                
                frame_idx += 1
                if pbar:
                    pbar.update(1)
        
        finally:
            # Clean up
            if pbar:
                pbar.close()
            cap.release()
            if out:
                out.release()
        
        # Update metadata
        fencer_data['metadata']['processed_frames'] = frame_idx
        
        return fencer_data
    
    def generate_heatmap(self, fencer_data, fencer_id=None, output_path=None, 
                       frame_width=None, frame_height=None):
        """
        Generate a heatmap of fencer positions
        
        Args:
            fencer_data: Fencer data from process_video
            fencer_id: Optional fencer ID to filter (None for all fencers)
            output_path: Path to save the heatmap image
            frame_width: Video frame width (if not in metadata)
            frame_height: Video frame height (if not in metadata)
            
        Returns:
            heatmap: Numpy array of the heatmap
        """
        # Get frame dimensions
        if frame_width is None:
            frame_width = fencer_data['metadata'].get('width', 640)
        if frame_height is None:
            frame_height = fencer_data['metadata'].get('height', 480)
        
        # Create empty heatmap
        heatmap = np.zeros((frame_height, frame_width), dtype=np.float32)
        
        # Add all trajectories to heatmap
        for id, trajectory in fencer_data['trajectories'].items():
            if fencer_id is not None and id != fencer_id:
                continue
            
            # Add points to heatmap
            for x, y in trajectory:
                if 0 <= x < frame_width and 0 <= y < frame_height:
                    # Use a Gaussian kernel to make a smoother heatmap
                    y_idx, x_idx = int(y), int(x)
                    kernel_size = 15
                    sigma = 5
                    
                    for i in range(max(0, y_idx - kernel_size), min(frame_height, y_idx + kernel_size + 1)):
                        for j in range(max(0, x_idx - kernel_size), min(frame_width, x_idx + kernel_size + 1)):
                            dist_sq = (i - y_idx) ** 2 + (j - x_idx) ** 2
                            weight = np.exp(-dist_sq / (2 * sigma ** 2))
                            heatmap[i, j] += weight
        
        # Normalize heatmap
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap)
        
        # Create visualization
        plt.figure(figsize=(10, 8))
        plt.imshow(heatmap, cmap='hot')
        plt.colorbar(label='Density')
        
        if fencer_id is not None:
            plt.title(f"Movement Heatmap for Fencer {fencer_id}")
        else:
            plt.title("Movement Heatmap for All Fencers")
        
        # Save to file if requested
        if output_path:
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Heatmap saved to {output_path}")
        else:
            plt.show()
        
        return heatmap
    
    def detect_hits(self, fencer_data, min_distance=50, output_path=None):
        """
        Detect potential hits based on fencer proximity
        
        Args:
            fencer_data: Fencer data from process_video
            min_distance: Minimum distance between fencers to register as a hit
            output_path: Path to save visualization
            
        Returns:
            hits: List of potential hits with frame indices and positions
        """
        hits = []
        
        # Get all fencer IDs
        fencer_ids = list(fencer_data['trajectories'].keys())
        
        # No hits possible with less than 2 fencers
        if len(fencer_ids) < 2:
            return hits
        
        # For each pair of fencers
        for i in range(len(fencer_ids) - 1):
            for j in range(i + 1, len(fencer_ids)):
                id1, id2 = fencer_ids[i], fencer_ids[j]
                
                # Get trajectories
                traj1 = fencer_data['trajectories'][id1]
                traj2 = fencer_data['trajectories'][id2]
                frames1 = fencer_data['frames'][id1]
                frames2 = fencer_data['frames'][id2]
                
                # Check if trajectories overlap in time
                common_frames = set(frames1).intersection(set(frames2))
                
                for frame in common_frames:
                    # Get positions at this frame
                    idx1 = frames1.index(frame)
                    idx2 = frames2.index(frame)
                    
                    pos1 = traj1[idx1]
                    pos2 = traj2[idx2]
                    
                    # Calculate distance
                    distance = np.sqrt((pos1[0] - pos2[0])**2 + (pos1[1] - pos2[1])**2)
                    
                    # Check if close enough to be a hit
                    if distance < min_distance:
                        # Estimate hit position (midpoint)
                        hit_pos = ((pos1[0] + pos2[0]) / 2, (pos1[1] + pos2[1]) / 2)
                        
                        hits.append({
                            'frame': frame,
                            'position': hit_pos,
                            'fencers': (id1, id2),
                            'distance': distance
                        })
        
        # Create visualization
        if output_path and hits:
            # Create a timeline of hits
            plt.figure(figsize=(12, 6))
            
            # Plot frame numbers
            frames = [hit['frame'] for hit in hits]
            fencer_pairs = [f"{hit['fencers'][0]} vs {hit['fencers'][1]}" for hit in hits]
            distances = [hit['distance'] for hit in hits]
            
            plt.scatter(frames, [1] * len(frames), c=distances, cmap='cool', s=100)
            
            for i, (frame, pair) in enumerate(zip(frames, fencer_pairs)):
                plt.annotate(pair, (frame, 1.05), ha='center', fontsize=8, rotation=45)
            
            plt.title("Potential Hits Timeline")
            plt.xlabel("Frame")
            plt.yticks([])
            plt.colorbar(label="Distance (pixels)")
            
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"Hit timeline saved to {output_path}")
        
        return hits

# Example usage
if __name__ == "__main__":
    # Initialize segmentation module
    segmenter = FencerSegmentation()
    
    # Process a video
    video_path = "evenevenmorecropped (1).mp4"
    output_path = "fencer_segmentation.mp4"
    
    if os.path.exists(video_path):
        print(f"Processing video: {video_path}")
        fencer_data = segmenter.process_video(
            video_path=video_path,
            output_path=output_path
        )
        
        # Generate heatmap
        segmenter.generate_heatmap(
            fencer_data=fencer_data,
            output_path="fencer_heatmap.png"
        )
        
        # Detect potential hits
        hits = segmenter.detect_hits(
            fencer_data=fencer_data,
            output_path="fencer_hits.png"
        )
        
        print(f"Detected {len(hits)} potential hits")
    else:
        print(f"Video file not found: {video_path}")
        print("Please specify a valid video file path") 