import cv2
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional, Any
from scipy.signal import find_peaks, savgol_filter
import matplotlib.pyplot as plt

class MotionAnalyzer:
    """
    Analyzes motion in video frames to detect strokes and segment the video
    """
    def __init__(self, window_size: int = 15):
        """
        Initialize motion analyzer
        
        Args:
            window_size: Size of the window for motion analysis
        """
        self.window_size = window_size
        self.motion_history = []
        
    def compute_frame_difference(
        self, 
        prev_frame: np.ndarray, 
        curr_frame: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute difference between two frames
        
        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            mask: Optional mask to focus on specific regions
            
        Returns:
            diff_score: Difference score between frames
        """
        # Convert to grayscale if needed
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
            
        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame
        
        # Apply mask if provided
        if mask is not None:
            prev_gray = cv2.bitwise_and(prev_gray, prev_gray, mask=mask)
            curr_gray = cv2.bitwise_and(curr_gray, curr_gray, mask=mask)
        
        # Compute absolute difference
        frame_diff = cv2.absdiff(prev_gray, curr_gray)
        
        # Calculate mean difference
        diff_score = np.mean(frame_diff)
        
        return diff_score
    
    def compute_dense_optical_flow(
        self, 
        prev_frame: np.ndarray, 
        curr_frame: np.ndarray,
        mask: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float]:
        """
        Compute dense optical flow between two frames
        
        Args:
            prev_frame: Previous frame
            curr_frame: Current frame
            mask: Optional mask to focus on specific regions
            
        Returns:
            flow_vis: Visualization of optical flow
            flow_magnitude: Magnitude of optical flow
        """
        # Convert to grayscale if needed
        if len(prev_frame.shape) == 3:
            prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_frame
            
        if len(curr_frame.shape) == 3:
            curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        else:
            curr_gray = curr_frame
        
        # Apply mask if provided
        if mask is not None:
            prev_gray = cv2.bitwise_and(prev_gray, prev_gray, mask=mask)
            curr_gray = cv2.bitwise_and(curr_gray, curr_gray, mask=mask)
        
        # Calculate optical flow using Farneback method
        flow = cv2.calcOpticalFlowFarneback(
            prev_gray, curr_gray, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # Convert to polar coordinates (magnitude, angle)
        magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        
        # Calculate mean magnitude of flow
        flow_magnitude = np.mean(magnitude)
        
        # Visualize optical flow
        flow_vis = np.zeros_like(prev_frame)
        if len(flow_vis.shape) == 3:
            # Map angles to hue and magnitude to value
            hsv = np.zeros((prev_gray.shape[0], prev_gray.shape[1], 3), dtype=np.uint8)
            hsv[..., 0] = angle * 180 / np.pi / 2  # Hue
            hsv[..., 1] = 255  # Saturation
            hsv[..., 2] = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)  # Value
            flow_vis = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        return flow_vis, flow_magnitude

    def update_motion_history(self, motion_score: float) -> None:
        """
        Update motion history with new score
        
        Args:
            motion_score: New motion score to add
        """
        self.motion_history.append(motion_score)
        # Keep only recent history
        if len(self.motion_history) > self.window_size * 10:  # Limit to 10x window size
            self.motion_history = self.motion_history[-self.window_size * 10:]
    
    def smooth_signal(self, signal: List[float], window_length: int = 11) -> np.ndarray:
        """
        Smooth signal using Savitzky-Golay filter
        
        Args:
            signal: Signal to smooth
            window_length: Length of the filter window
            
        Returns:
            smoothed_signal: Smoothed signal
        """
        if len(signal) < window_length:
            return np.array(signal)
        
        # Ensure window length is odd
        if window_length % 2 == 0:
            window_length += 1
        
        # Apply Savitzky-Golay filter
        try:
            smoothed_signal = savgol_filter(signal, window_length, 3)
        except ValueError:
            # If window is too large, try smaller window
            if window_length > 3:
                smoothed_signal = savgol_filter(signal, 3, 1)
            else:
                smoothed_signal = np.array(signal)
        
        return smoothed_signal

    def detect_motion_peaks(
        self, 
        threshold_factor: float = 1.5, 
        min_distance: int = 10
    ) -> List[int]:
        """
        Detect peaks in motion history that represent potential strokes
        
        Args:
            threshold_factor: Factor to multiply mean motion for threshold
            min_distance: Minimum distance between peaks
            
        Returns:
            peak_indices: Indices of detected peaks
        """
        if len(self.motion_history) < self.window_size:
            return []
        
        # Smooth the motion signal
        smoothed_signal = self.smooth_signal(self.motion_history, window_length=min(self.window_size, 11))
        
        # Calculate threshold as a factor of the mean
        baseline = np.median(smoothed_signal)
        std_dev = np.std(smoothed_signal)
        threshold = baseline + threshold_factor * std_dev
        
        # Find peaks in the smoothed signal
        peak_indices, _ = find_peaks(
            smoothed_signal, 
            height=threshold, 
            distance=min_distance
        )
        
        # Convert indices from smoothed signal back to original motion history
        if len(peak_indices) > 0:
            peak_indices = peak_indices.tolist()
        
        return peak_indices

    def visualize_motion_signal(self, peak_indices: List[int] = None) -> np.ndarray:
        """
        Create visualization of motion signal with detected peaks
        
        Args:
            peak_indices: Indices of detected peaks
            
        Returns:
            visualization: Image of the motion signal plot
        """
        plt.figure(figsize=(12, 4))
        
        # Plot motion history
        plt.plot(self.motion_history, 'b-', label='Motion Signal')
        
        # Plot smoothed signal
        if len(self.motion_history) >= self.window_size:
            smoothed_signal = self.smooth_signal(self.motion_history, window_length=min(self.window_size, 11))
            plt.plot(smoothed_signal, 'g-', label='Smoothed Signal')
        
        # Plot peaks if provided
        if peak_indices:
            peak_values = [self.motion_history[i] for i in peak_indices]
            plt.plot(peak_indices, peak_values, 'ro', label='Detected Strokes')
        
        # Add labels and legend
        plt.xlabel('Frame')
        plt.ylabel('Motion Score')
        plt.title('Motion Analysis for Stroke Detection')
        plt.legend()
        plt.grid(True)
        
        # Convert plot to image
        plt.tight_layout()
        fig = plt.gcf()
        plt.close()
        
        # Convert to image array
        fig.canvas.draw()
        img = np.array(fig.canvas.renderer.buffer_rgba())
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
        
        return img


class StrokeDetector:
    """
    Detects fencing strokes and segments video into technique-specific clips
    """
    def __init__(
        self, 
        temporal_window: int = 16,
        motion_threshold: float = 1.5,
        min_segment_length: int = 8
    ):
        """
        Initialize stroke detector
        
        Args:
            temporal_window: Size of the window for motion analysis
            motion_threshold: Threshold factor for peak detection
            min_segment_length: Minimum length of a valid segment
        """
        self.motion_analyzer = MotionAnalyzer(window_size=temporal_window)
        self.motion_threshold = motion_threshold
        self.min_segment_length = min_segment_length
        self.temporal_window = temporal_window
        self.prev_frame = None
        self.frame_idx = 0
        self.stroke_indices = []
        self.segments = []
    
    def process_frame(self, frame: np.ndarray, masks: List[np.ndarray] = None) -> Dict[str, Any]:
        """
        Process a single frame to analyze motion and detect strokes
        
        Args:
            frame: Current frame
            masks: Optional masks for different regions of interest (e.g., for each fencer)
            
        Returns:
            results: Dictionary with motion scores and any detected strokes
        """
        results = {'frame_idx': self.frame_idx, 'motion_score': 0.0, 'stroke_detected': False}
        
        # Skip if first frame
        if self.prev_frame is None:
            self.prev_frame = frame
            self.frame_idx += 1
            return results
        
        # Calculate motion score based on optical flow
        if masks:
            # Process each mask separately
            total_score = 0.0
            flow_visuals = []
            
            for mask in masks:
                _, flow_magnitude = self.motion_analyzer.compute_dense_optical_flow(
                    self.prev_frame, frame, mask=mask
                )
                total_score += flow_magnitude
                
            # Average across all masks
            motion_score = total_score / len(masks)
            
        else:
            # Process entire frame
            _, motion_score = self.motion_analyzer.compute_dense_optical_flow(
                self.prev_frame, frame
            )
        
        # Update motion history
        self.motion_analyzer.update_motion_history(motion_score)
        
        # Detect peaks only periodically to save computation
        if self.frame_idx % 5 == 0 and len(self.motion_analyzer.motion_history) >= self.temporal_window:
            peak_indices = self.motion_analyzer.detect_motion_peaks(
                threshold_factor=self.motion_threshold,
                min_distance=self.temporal_window // 2
            )
            
            # Check if we have a new peak
            current_history_len = len(self.motion_analyzer.motion_history)
            for peak_idx in peak_indices:
                # Convert local peak index to global frame index
                global_peak_idx = self.frame_idx - (current_history_len - peak_idx)
                
                # Check if this is a new peak
                if (global_peak_idx > 0 and 
                    global_peak_idx not in self.stroke_indices and
                    current_history_len - peak_idx < 10):  # Only recent peaks
                    
                    self.stroke_indices.append(global_peak_idx)
                    results['stroke_detected'] = True
                    
                    # Segment from previous stroke to current one
                    if len(self.stroke_indices) > 1:
                        start_idx = self.stroke_indices[-2]
                        end_idx = global_peak_idx
                        
                        # Make sure it's long enough
                        if end_idx - start_idx >= self.min_segment_length:
                            self.segments.append((start_idx, end_idx))
                            results['new_segment'] = (start_idx, end_idx)
        
        # Save results
        results['motion_score'] = motion_score
        
        # Update state
        self.prev_frame = frame
        self.frame_idx += 1
        
        return results
    
    def get_current_segments(self) -> List[Tuple[int, int]]:
        """
        Get list of detected segments
        
        Returns:
            segments: List of (start_idx, end_idx) tuples
        """
        return self.segments
    
    def visualize_motion_analysis(self) -> np.ndarray:
        """
        Create visualization of motion analysis
        
        Returns:
            visualization: Image of the motion analysis
        """
        return self.motion_analyzer.visualize_motion_signal(
            peak_indices=[idx - (self.frame_idx - len(self.motion_analyzer.motion_history)) 
                         for idx in self.stroke_indices 
                         if idx >= self.frame_idx - len(self.motion_analyzer.motion_history)]
        )


class TemporalSegmentation:
    """
    Main temporal segmentation module for fencing videos
    """
    def __init__(
        self,
        temporal_window: int = 16,
        motion_threshold: float = 1.5,
        min_segment_length: int = 8,
        max_segment_length: int = 64
    ):
        """
        Initialize temporal segmentation
        
        Args:
            temporal_window: Size of the window for motion analysis
            motion_threshold: Threshold factor for peak detection
            min_segment_length: Minimum length of a valid segment (in frames)
            max_segment_length: Maximum length of a segment (in frames)
        """
        self.stroke_detector = StrokeDetector(
            temporal_window=temporal_window,
            motion_threshold=motion_threshold,
            min_segment_length=min_segment_length
        )
        self.max_segment_length = max_segment_length
        self.frame_buffer = []
        self.segments = []
        self.current_start_idx = 0
    
    def process_frame(self, frame: np.ndarray, fencer_masks: List[np.ndarray] = None) -> Dict[str, Any]:
        """
        Process a single frame and detect strokes/segments
        
        Args:
            frame: Current video frame
            fencer_masks: Optional masks for each fencer
            
        Returns:
            results: Dictionary with motion scores and detected segments
        """
        # Add frame to buffer
        self.frame_buffer.append(frame)
        
        # Keep buffer size manageable
        if len(self.frame_buffer) > self.max_segment_length * 2:
            self.frame_buffer = self.frame_buffer[-self.max_segment_length * 2:]
        
        # Process frame for stroke detection
        results = self.stroke_detector.process_frame(frame, masks=fencer_masks)
        
        # Handle segmentation
        if 'new_segment' in results:
            start_idx, end_idx = results['new_segment']
            
            # Adjust indices to buffer indices
            buffer_start = max(0, len(self.frame_buffer) - (self.stroke_detector.frame_idx - start_idx))
            buffer_end = min(len(self.frame_buffer) - 1, 
                            len(self.frame_buffer) - (self.stroke_detector.frame_idx - end_idx))
            
            # Ensure segment isn't too long
            if buffer_end - buffer_start > self.max_segment_length:
                buffer_start = buffer_end - self.max_segment_length
            
            # Extract frames for the segment
            if buffer_end > buffer_start:
                segment_frames = self.frame_buffer[buffer_start:buffer_end]
                self.segments.append({
                    'start_idx': start_idx,
                    'end_idx': end_idx,
                    'frames': segment_frames
                })
                results['segment_frames'] = segment_frames
        
        # Check if we should create a new segment based on maximum length
        if (self.stroke_detector.frame_idx - self.current_start_idx) >= self.max_segment_length:
            # Create a new segment
            buffer_start = max(0, len(self.frame_buffer) - self.max_segment_length)
            segment_frames = self.frame_buffer[buffer_start:]
            
            if len(segment_frames) >= self.stroke_detector.min_segment_length:
                segment = {
                    'start_idx': self.stroke_detector.frame_idx - len(segment_frames),
                    'end_idx': self.stroke_detector.frame_idx - 1,
                    'frames': segment_frames
                }
                self.segments.append(segment)
                results['segment_frames'] = segment_frames
                results['max_length_segment'] = True
            
            self.current_start_idx = self.stroke_detector.frame_idx
        
        return results
    
    def get_segments(self) -> List[Dict[str, Any]]:
        """
        Get list of all detected segments
        
        Returns:
            segments: List of detected segments
        """
        return self.segments
    
    def get_visualization(self) -> np.ndarray:
        """
        Get visualization of the motion analysis
        
        Returns:
            visualization: Image of the motion analysis
        """
        return self.stroke_detector.visualize_motion_analysis()
    
    @staticmethod
    def extract_segments_from_video(
        video_path: str,
        output_base_path: Optional[str] = None,
        visualize: bool = False,
        max_frames: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Process a video file and extract segments
        
        Args:
            video_path: Path to video file
            output_base_path: Base path for saving extracted segments
            visualize: Whether to create and save visualizations
            max_frames: Maximum number of frames to process
            
        Returns:
            segments: List of detected segments
        """
        # Create temporal segmentation
        segmentation = TemporalSegmentation()
        
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return []
        
        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Create visualization video if requested
        vis_writer = None
        if visualize and output_base_path:
            vis_path = f"{output_base_path}_segmentation_vis.mp4"
            vis_writer = cv2.VideoWriter(
                vis_path, 
                cv2.VideoWriter_fourcc(*'mp4v'), 
                fps, 
                (width, height + 300)  # Extra space for motion plot
            )
        
        # Process frames
        frame_idx = 0
        segments = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Check if we've reached max frames
            if max_frames and frame_idx >= max_frames:
                break
            
            # Process frame
            results = segmentation.process_frame(frame)
            
            # Check for new segments
            if 'segment_frames' in results:
                segment_idx = len(segments)
                segment_frames = results['segment_frames']
                segment_info = {
                    'segment_idx': segment_idx,
                    'start_idx': results.get('new_segment', (frame_idx - len(segment_frames), frame_idx))[0],
                    'end_idx': results.get('new_segment', (frame_idx - len(segment_frames), frame_idx))[1],
                    'num_frames': len(segment_frames)
                }
                segments.append(segment_info)
                
                # Save segment video if output path is provided
                if output_base_path:
                    segment_path = f"{output_base_path}_segment_{segment_idx:03d}.mp4"
                    segment_writer = cv2.VideoWriter(
                        segment_path, 
                        cv2.VideoWriter_fourcc(*'mp4v'), 
                        fps, 
                        (width, height)
                    )
                    
                    for seg_frame in segment_frames:
                        segment_writer.write(seg_frame)
                    
                    segment_writer.release()
                    segment_info['segment_path'] = segment_path
            
            # Create visualization
            if visualize and vis_writer:
                # Get motion visualization
                motion_vis = segmentation.get_visualization()
                
                # Resize to match frame width
                motion_vis = cv2.resize(motion_vis, (width, 300))
                
                # Add text overlay
                frame_with_text = frame.copy()
                cv2.putText(
                    frame_with_text, 
                    f"Frame: {frame_idx} | Motion: {results['motion_score']:.2f}", 
                    (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    1, 
                    (0, 255, 255), 
                    2
                )
                
                if results.get('stroke_detected', False):
                    cv2.putText(
                        frame_with_text, 
                        "STROKE DETECTED", 
                        (width // 2 - 100, height // 2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1, 
                        (0, 0, 255), 
                        2
                    )
                
                # Combine frame and motion visualization
                combined_vis = np.vstack([frame_with_text, motion_vis])
                vis_writer.write(combined_vis)
            
            frame_idx += 1
        
        # Release resources
        cap.release()
        if vis_writer:
            vis_writer.release()
        
        return segments 