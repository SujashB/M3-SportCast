import cv2
import numpy as np
import os
import torch
import matplotlib.pyplot as plt
from PIL import Image
import mediapipe as mp
import math
from typing import List, Dict, Tuple, Optional

# Optional mmpose import - we'll use if available, otherwise fall back to mediapipe
try:
    from mmpose.apis import inference_topdown, init_model
    from mmdet.apis import inference_detector, init_detector
    MMPOSE_AVAILABLE = True
except ImportError:
    print("Warning: mmpose not available. Using mediapipe as fallback.")
    MMPOSE_AVAILABLE = False

class PoseEstimator:
    def __init__(self, use_mmpose=True):
        """Initialize pose estimator with either mmpose or mediapipe"""
        self.use_mmpose = use_mmpose and MMPOSE_AVAILABLE
        
        if self.use_mmpose:
            try:
                # Load mmpose models
                det_config = 'mmdetection_configs/faster_rcnn_r50_fpn_coco.py'
                det_checkpoint = 'checkpoints/faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth'
                pose_config = 'mmpose_configs/hrnet_w48_coco_256x192.py'
                pose_checkpoint = 'checkpoints/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth'
                
                # Check if model files exist
                if not (os.path.exists(det_checkpoint) and os.path.exists(pose_checkpoint)):
                    print("MMPose checkpoint files not found. Using mediapipe as fallback.")
                    self.use_mmpose = False
                else:
                    # Initialize models
                    self.det_model = init_detector(det_config, det_checkpoint, device='cuda:0')
                    self.pose_model = init_model(pose_config, pose_checkpoint, device='cuda:0')
            except Exception as e:
                print(f"Error initializing mmpose: {e}")
                print("Using mediapipe as fallback.")
                self.use_mmpose = False
        
        if not self.use_mmpose:
            # Initialize mediapipe pose detector
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=2,
                enable_segmentation=False,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
    
    def estimate_pose(self, frame):
        """Estimate pose in a single frame"""
        if self.use_mmpose:
            return self._estimate_pose_mmpose(frame)
        else:
            return self._estimate_pose_mediapipe(frame)
    
    def _estimate_pose_mmpose(self, frame):
        """Estimate pose using mmpose"""
        # Person detection
        det_results = inference_detector(self.det_model, frame)
        person_results = det_results[0]  # Class 0 in COCO is person
        
        # Keep only high-confidence detections
        person_results = [pr for pr in person_results if pr[4] > 0.5]
        
        if not person_results:
            return None
        
        # Keypoint detection
        pose_results = inference_topdown(self.pose_model, frame, person_results)
        
        if not pose_results:
            return None
        
        # Extract keypoints from the first person
        keypoints = pose_results[0]['keypoints']
        return keypoints
    
    def _estimate_pose_mediapipe(self, frame):
        """Estimate pose using mediapipe"""
        # Convert BGR to RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(rgb_frame)
        
        if not results.pose_landmarks:
            return None
        
        # Convert to numpy array
        landmarks = results.pose_landmarks.landmark
        keypoints = []
        height, width, _ = frame.shape
        
        # MediaPipe has different keypoint order than COCO
        # Map MediaPipe landmarks to common format
        for idx, landmark in enumerate(landmarks):
            x, y = int(landmark.x * width), int(landmark.y * height)
            # Add visibility score
            keypoints.append([x, y, landmark.visibility])
        
        return np.array(keypoints)
    
    def draw_pose(self, frame, keypoints):
        """Draw pose keypoints on frame"""
        if keypoints is None:
            return frame
        
        output_frame = frame.copy()
        
        if self.use_mmpose:
            # Draw keypoints for mmpose
            for kp in keypoints:
                x, y, conf = kp
                if conf > 0.5:
                    cv2.circle(output_frame, (int(x), int(y)), 5, (0, 255, 0), -1)
            
            # Draw connections for important limbs
            limbs = [
                # Torso
                (5, 6), (5, 11), (6, 12), (11, 12),
                # Arms
                (5, 7), (7, 9), (6, 8), (8, 10),
                # Legs
                (11, 13), (13, 15), (12, 14), (14, 16)
            ]
            
            for limb in limbs:
                pt1, pt2 = limbs
                if keypoints[pt1, 2] > 0.5 and keypoints[pt2, 2] > 0.5:
                    cv2.line(output_frame, 
                             (int(keypoints[pt1, 0]), int(keypoints[pt1, 1])),
                             (int(keypoints[pt2, 0]), int(keypoints[pt2, 1])),
                             (0, 255, 255), 2)
        else:
            # Draw pose using mediapipe's built-in function
            mp_drawing = mp.solutions.drawing_utils
            mp_pose = mp.solutions.pose
            
            # Simplified drawing approach
            height, width, _ = frame.shape
            
            # Draw keypoints directly
            for idx, (x, y, conf) in enumerate(keypoints):
                if conf > 0.5:
                    # Draw circle for each keypoint
                    cv2.circle(
                        output_frame, 
                        (int(x), int(y)), 
                        5, (0, 255, 0), -1
                    )
            
            # Draw skeleton with manually defined connections
            connections = [
                # Face connections
                (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
                # Upper body connections
                (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
                # Torso
                (11, 23), (12, 24), (23, 24),
                # Lower body
                (23, 25), (25, 27), (27, 29), (29, 31), 
                (24, 26), (26, 28), (28, 30), (30, 32)
            ]
            
            for connection in connections:
                idx1, idx2 = connection
                if (idx1 < len(keypoints) and idx2 < len(keypoints) and
                    keypoints[idx1, 2] > 0.5 and keypoints[idx2, 2] > 0.5):
                    cv2.line(
                        output_frame,
                        (int(keypoints[idx1, 0]), int(keypoints[idx1, 1])),
                        (int(keypoints[idx2, 0]), int(keypoints[idx2, 1])),
                        (0, 255, 255), 2
                    )
        
        return output_frame
    
    def calculate_angles(self, keypoints, prev_keypoints=None):
        """Calculate relevant angles for fencing analysis"""
        if keypoints is None:
            return {}
        
        angles = {}
        
        # Define keypoint indices based on estimator
        if self.use_mmpose:
            # COCO keypoint indices
            nose = 0
            left_shoulder, right_shoulder = 5, 6
            left_elbow, right_elbow = 7, 8
            left_wrist, right_wrist = 9, 10
            left_hip, right_hip = 11, 12
            left_knee, right_knee = 13, 14
            left_ankle, right_ankle = 15, 16
        else:
            # MediaPipe keypoint indices
            nose = 0
            left_shoulder, right_shoulder = 11, 12
            left_elbow, right_elbow = 13, 14
            left_wrist, right_wrist = 15, 16
            left_hip, right_hip = 23, 24
            left_knee, right_knee = 25, 26
            left_ankle, right_ankle = 27, 28
        
        # Calculate right arm extension (angle between shoulder, elbow, and wrist)
        angles['right_arm'] = self._calculate_angle(
            keypoints[right_shoulder][:2], 
            keypoints[right_elbow][:2], 
            keypoints[right_wrist][:2]
        )
        
        # Calculate left arm extension
        angles['left_arm'] = self._calculate_angle(
            keypoints[left_shoulder][:2], 
            keypoints[left_elbow][:2], 
            keypoints[left_wrist][:2]
        )
        
        # Calculate right leg bend (angle between hip, knee, and ankle)
        angles['right_leg'] = self._calculate_angle(
            keypoints[right_hip][:2], 
            keypoints[right_knee][:2], 
            keypoints[right_ankle][:2]
        )
        
        # Calculate left leg bend
        angles['left_leg'] = self._calculate_angle(
            keypoints[left_hip][:2], 
            keypoints[left_knee][:2], 
            keypoints[left_ankle][:2]
        )
        
        # Calculate torso angle (vertical angle between shoulders and hips)
        shoulder_center = (
            (keypoints[left_shoulder][0] + keypoints[right_shoulder][0]) / 2,
            (keypoints[left_shoulder][1] + keypoints[right_shoulder][1]) / 2
        )
        hip_center = (
            (keypoints[left_hip][0] + keypoints[right_hip][0]) / 2,
            (keypoints[left_hip][1] + keypoints[right_hip][1]) / 2
        )
        vertical_point = (shoulder_center[0], shoulder_center[1] - 100)  # Point directly above shoulder
        angles['torso_vertical'] = self._calculate_angle(
            vertical_point, 
            shoulder_center, 
            hip_center
        )
        
        # Calculate movement direction if previous keypoints are available
        if prev_keypoints is not None:
            # Calculate hip center movement
            prev_hip_center = (
                (prev_keypoints[left_hip][0] + prev_keypoints[right_hip][0]) / 2,
                (prev_keypoints[left_hip][1] + prev_keypoints[right_hip][1]) / 2
            )
            
            # X movement (positive = right, negative = left)
            x_movement = hip_center[0] - prev_hip_center[0]
            angles['movement_x'] = x_movement
            
            # Calculate overall movement angle
            movement_angle = np.arctan2(
                hip_center[1] - prev_hip_center[1],
                hip_center[0] - prev_hip_center[0]
            ) * 180 / np.pi
            angles['movement_angle'] = movement_angle
        
        return angles
    
    def _calculate_angle(self, a, b, c):
        """Calculate angle between three points (in degrees)"""
        a = np.array(a)
        b = np.array(b)
        c = np.array(c)
        
        # Calculate vectors
        ba = a - b
        bc = c - b
        
        # Calculate angle
        cosine_angle = np.dot(ba, bc) / (np.linalg.norm(ba) * np.linalg.norm(bc))
        cosine_angle = np.clip(cosine_angle, -1.0, 1.0)  # Ensure value is in domain of arccos
        angle = np.arccos(cosine_angle) * 180 / np.pi
        
        return angle
    
    def analyze_fencing_movement(self, angles, movement_history=None):
        """Analyze fencing movements based on angles"""
        # Default movement history if not provided
        if movement_history is None:
            movement_history = []
        
        observations = []
        
        # Analyze arm extension (extended arm indicates attack or lunge)
        if angles.get('right_arm', 180) > 160:
            observations.append("arm fully extended forward")
        elif angles.get('right_arm', 180) > 130:
            observations.append("arm partially extended")
        else:
            observations.append("arm bent")
        
        # Analyze leg bend (bent front leg indicates lunge)
        if angles.get('right_leg', 180) < 120:
            observations.append("right leg deeply bent")
        elif angles.get('right_leg', 180) < 150:
            observations.append("right leg slightly bent")
            
        if angles.get('left_leg', 180) < 120:
            observations.append("left leg deeply bent")
        elif angles.get('left_leg', 180) < 150:
            observations.append("left leg slightly bent")
        
        # Analyze torso angle (forward lean indicates aggressive stance or lunge)
        if angles.get('torso_vertical', 0) > 30:
            observations.append("torso leaning forward significantly")
        elif angles.get('torso_vertical', 0) > 15:
            observations.append("torso leaning forward slightly")
        elif angles.get('torso_vertical', 0) < -15:
            observations.append("torso leaning backward")
        
        # Analyze movement direction if available
        if 'movement_x' in angles:
            if angles['movement_x'] > 10:
                observations.append("advancing forward quickly")
            elif angles['movement_x'] > 3:
                observations.append("advancing forward")
            elif angles['movement_x'] < -10:
                observations.append("retreating quickly")
            elif angles['movement_x'] < -3:
                observations.append("retreating")
        
        # Identify specific fencing movements
        fencing_movement = self._identify_fencing_technique(angles, observations)
        if fencing_movement:
            observations.append(f"performing {fencing_movement}")
        
        return observations
    
    def _identify_fencing_technique(self, angles, observations):
        """Identify specific fencing techniques based on angles and observations"""
        # Lunge detection - extended arm, deeply bent front leg, forward-leaning torso
        if (angles.get('right_arm', 0) > 160 and 
                angles.get('right_leg', 180) < 120 and 
                angles.get('torso_vertical', 0) > 15):
            return "lunge"
        
        # Advance detection - forward movement, balanced stance
        if ('advancing forward' in ' '.join(observations) and 
                angles.get('right_leg', 180) > 150 and 
                angles.get('left_leg', 180) > 150):
            return "advance"
        
        # Retreat detection - backward movement, balanced stance
        if ('retreating' in ' '.join(observations) and 
                angles.get('right_leg', 180) > 150 and 
                angles.get('left_leg', 180) > 150):
            return "retreat"
        
        # Parry detection - bent arm, stable stance, minimal torso movement
        if (angles.get('right_arm', 0) < 130 and 
                abs(angles.get('torso_vertical', 0)) < 10):
            return "parry"
        
        # Attack detection - extended arm, forward movement
        if (angles.get('right_arm', 0) > 150 and 
                'advancing' in ' '.join(observations)):
            return "attack"
        
        return None

def generate_descriptive_sentence(observations):
    """
    Generate a descriptive sentence from pose observations
    to enhance VideoMAE predictions
    """
    if not observations:
        return "Neutral fencing stance with minimal movement."
    
    # Join observations into a coherent sentence
    sentence = "Fencer is " + ", ".join(observations) + "."
    return sentence

def process_video_with_pose(video_path, output_path=None, save_frames=False):
    """
    Process a video with pose estimation and return frame-by-frame analysis
    """
    # Initialize pose estimator
    pose_estimator = PoseEstimator(use_mmpose=MMPOSE_AVAILABLE)
    
    # Open video file
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video file: {video_path}")
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Initialize output video writer if output path is specified
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Process video frame by frame
    prev_keypoints = None
    frame_analyses = []
    processed_frames = []
    
    print(f"Processing video: {video_path}")
    print(f"Total frames: {frame_count}")
    
    frame_idx = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process every 3rd frame to improve speed
        if frame_idx % 3 == 0:
            print(f"Processing frame {frame_idx}/{frame_count}")
            
            # Estimate pose
            keypoints = pose_estimator.estimate_pose(frame)
            
            if keypoints is not None:
                # Calculate angles
                angles = pose_estimator.calculate_angles(keypoints, prev_keypoints)
                
                # Analyze movement
                observations = pose_estimator.analyze_fencing_movement(angles)
                
                # Generate descriptive sentence
                sentence = generate_descriptive_sentence(observations)
                
                # Draw pose on frame
                annotated_frame = pose_estimator.draw_pose(frame, keypoints)
                
                # Add text with observations
                y_offset = 30
                for obs in observations:
                    cv2.putText(annotated_frame, obs, (10, y_offset), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    y_offset += 25
                
                # Store analysis results
                frame_analyses.append({
                    'frame_idx': frame_idx,
                    'angles': angles,
                    'observations': observations,
                    'sentence': sentence
                })
                
                # Save processed frame
                if save_frames:
                    processed_frames.append(annotated_frame)
                
                # Update previous keypoints
                prev_keypoints = keypoints
                
                # Write frame to output video
                if output_path:
                    out.write(annotated_frame)
            else:
                print(f"No pose detected in frame {frame_idx}")
                if output_path:
                    out.write(frame)  # Write original frame if no pose detected
        
        frame_idx += 1
    
    # Release resources
    cap.release()
    if output_path:
        out.release()
    
    print(f"Processed {len(frame_analyses)} frames with pose estimation")
    
    return frame_analyses, processed_frames

def extract_pose_descriptions(frame_analyses, sampling_rate=16):
    """
    Extract pose descriptions for VideoMAE model, sampling at regular intervals
    to match the expected number of frames (typically 16 for VideoMAE)
    """
    # If we don't have enough frames, duplicate the last ones
    if len(frame_analyses) < sampling_rate:
        # Duplicate the last frame analysis to fill up to sampling_rate
        last_analysis = frame_analyses[-1] if frame_analyses else {'sentence': ''}
        frame_analyses.extend([last_analysis] * (sampling_rate - len(frame_analyses)))
    
    # Sample frame analyses at regular intervals
    indices = np.linspace(0, len(frame_analyses) - 1, sampling_rate, dtype=int)
    sampled_analyses = [frame_analyses[i] for i in indices]
    
    # Extract sentences
    descriptions = [analysis['sentence'] for analysis in sampled_analyses]
    
    return descriptions

def enhance_videomae_with_pose(video_path, videomae_model, processor, pose_descriptions=None):
    """
    Enhance VideoMAE predictions with pose estimation descriptions
    """
    if pose_descriptions is None:
        # Process video with pose estimation
        frame_analyses, _ = process_video_with_pose(video_path, save_frames=False)
        pose_descriptions = extract_pose_descriptions(frame_analyses)
    
    # Combine pose descriptions with VideoMAE processing
    # This would typically involve adding the descriptions as text prompts or additional context
    # However, current VideoMAE models don't accept text input directly
    # Instead, we'll use this information to adjust/contextualize the model predictions
    
    print("Enhancing VideoMAE predictions with pose information:")
    for desc in pose_descriptions:
        print(f"- {desc}")
    
    # Count occurrences of fencing techniques in descriptions
    technique_counts = {
        'attack': 0, 'defense': 0, 'parry': 0, 'riposte': 0, 
        'lunge': 0, 'advance': 0, 'retreat': 0, 'feint': 0
    }
    
    for desc in pose_descriptions:
        desc_lower = desc.lower()
        for technique in technique_counts.keys():
            if technique in desc_lower:
                technique_counts[technique] += 1
    
    # Normalize counts
    total = sum(technique_counts.values()) + 1e-6  # Avoid division by zero
    pose_probs = {k: v/total for k, v in technique_counts.items()}
    
    # Return the pose probabilities that can be combined with VideoMAE predictions
    return pose_probs

if __name__ == "__main__":
    # Example usage
    video_path = "evenevenmorecropped (1).mp4"
    output_path = "pose_analysis.mp4"
    
    # Process video with pose estimation
    frame_analyses, frames = process_video_with_pose(video_path, output_path, save_frames=True)
    
    # Display some sample frames with pose estimation
    if frames:
        plt.figure(figsize=(15, 10))
        for i in range(min(4, len(frames))):
            idx = i * len(frames) // 4
            plt.subplot(2, 2, i+1)
            plt.imshow(cv2.cvtColor(frames[idx], cv2.COLOR_BGR2RGB))
            plt.title(f"Frame {frame_analyses[idx]['frame_idx']}")
            plt.axis('off')
        
        plt.suptitle("Fencing Pose Analysis")
        plt.tight_layout()
        plt.savefig("pose_analysis_frames.png")
        plt.close()
    
    # Extract pose descriptions
    pose_descriptions = extract_pose_descriptions(frame_analyses)
    print("\nPose descriptions for VideoMAE:")
    for desc in pose_descriptions:
        print(f"- {desc}") 