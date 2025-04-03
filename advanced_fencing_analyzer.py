import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json
import datetime
from collections import Counter
import seaborn as sns
import matplotlib.patches as patches

# Import pose estimation helper directly
from pose_estimation_helper import PoseEstimator, process_video_with_pose, extract_pose_descriptions, generate_descriptive_sentence
# Import fencer detector
from fencer_detector import FencerDetector
# Import our enhanced fencer detector with CNN and sword detection
from enhanced_fencer_detector import EnhancedFencerDetector, SwordDetector

class SimplifiedFencingAnalyzer:
    """
    Simplified fencing analysis system that focuses on pose estimation
    """
    def __init__(self, pose_model_path=None, sword_model_path=None, use_enhanced_detector=True):
        """Initialize the analyzer with pose estimation capability"""
        print("Initializing Simplified Fencing Analyzer...")
        
        # Check if we should use the enhanced detector
        self.use_enhanced_detector = use_enhanced_detector
        self.pose_model_path = pose_model_path
        self.sword_model_path = sword_model_path
        
        # Initialize pose estimator
        print("Loading pose estimator...")
        try:
            from pose_estimation_helper import MMPOSE_AVAILABLE
            self.pose_estimator = PoseEstimator(use_mmpose=MMPOSE_AVAILABLE)
            print(f"Pose estimator initialized (using MMPose: {MMPOSE_AVAILABLE})")
        except ImportError as e:
            print(f"Error initializing pose estimator: {e}")
            self.pose_estimator = None
        
        # Initialize fencer detector
        print("Loading fencer detector...")
        try:
            if self.use_enhanced_detector:
                self.fencer_detector = EnhancedFencerDetector(
                    pose_model_path=self.pose_model_path,
                    sword_model_path=self.sword_model_path
                )
                print("Enhanced fencer detector initialized with CNN pose classifier and sword detector")
            else:
                self.fencer_detector = FencerDetector()
                print("Basic fencer detector initialized")
        except ImportError as e:
            print(f"Error initializing fencer detector: {e}")
            self.fencer_detector = None
        
        # Initialize manual fencer selection state
        self.manual_fencer_boxes = []
        self.manual_fencer_ids = []
        
        print("Initialization complete")
    
    def show_first_frame_for_selection(self, video_path, output_dir):
        """
        Display the first frame of the video and save it for potential manual selection
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save the first frame
            
        Returns:
            path to saved first frame image
        """
        # Open video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {video_path}")
            return None
        
        # Read first frame
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Could not read first frame from {video_path}")
            cap.release()
            return None
        
        # Save first frame
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        first_frame_path = os.path.join(output_dir, f"{video_name}_first_frame.png")
        cv2.imwrite(first_frame_path, frame)
        
        # Detect fencers in the first frame
        detections = []
        if self.fencer_detector:
            if self.use_enhanced_detector:
                tracked_items, sword_detections = self.fencer_detector.track_and_classify(frame)
                
                # Convert tracked items to detections format expected by the visualization code
                detections = []
                for item in tracked_items:
                    fencer_id = item['fencer_id']
                    box = item['box']
                    pose_class = item.get('pose_class', 'neutral')
                    
                    detections.append({
                        'box': box,
                        'fencer_id': fencer_id,
                        'pose_class': pose_class
                    })
                
                # Store sword detections for reference
                self.sword_detections = sword_detections
            else:
                detections = self.fencer_detector.detect_fencers(frame)
        
        # Create a larger figure for better visibility
        plt.figure(figsize=(12, 8))
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        plt.imshow(frame_rgb)
        
        # Draw bounding boxes for detected fencers
        ax = plt.gca()
        for i, detection in enumerate(detections):
            box = detection['box']
            x1, y1, x2, y2 = box
            fencer_id = detection.get('fencer_id', i)
            pose_class = detection.get('pose_class', 'neutral')
            
            # Use different colors for different pose classes
            color = 'g'  # Default green
            if pose_class == 'attack':
                color = 'r'
            elif pose_class == 'defense':
                color = 'b'
            elif pose_class == 'lunge':
                color = 'm'
            
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor=color, facecolor='none')
            ax.add_patch(rect)
            plt.text(x1, y1-10, f"Fencer {fencer_id} ({pose_class})", color=color, fontsize=12, weight='bold')
        
        # Draw sword detections if available
        if hasattr(self, 'sword_detections') and self.sword_detections:
            for det in self.sword_detections:
                box = det['box']
                x1, y1, x2, y2 = box
                class_name = det['class_name']
                
                # Use orange for sword parts
                rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='orange', facecolor='none')
                ax.add_patch(rect)
                plt.text(x1, y1-10, class_name, color='orange', fontsize=10)
        
        plt.title("First Frame - Detected Fencers & Swords")
        plt.axis('off')
        
        # Save the annotated frame
        annotated_frame_path = os.path.join(output_dir, f"{video_name}_first_frame_annotated.png")
        plt.savefig(annotated_frame_path, bbox_inches='tight')
        plt.close()
        
        # Release video resources
        cap.release()
        
        print(f"First frame saved to: {first_frame_path}")
        print(f"Annotated frame saved to: {annotated_frame_path}")
        print(f"Detected {len(detections)} potential fencers in the first frame")
        print("You can use the --manual_select option with fencer IDs to manually select which fencers to analyze")
        
        return annotated_frame_path
    
    def analyze_video(self, video_path, output_dir="results", num_frames=None, 
                     save_visualization=True, manual_select=None):
        """
        Perform pose estimation analysis on a fencing video
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save results
            num_frames: Maximum number of frames to process (None for all)
            save_visualization: Whether to save visualization outputs
            manual_select: Optional comma-separated list of fencer IDs to manually select
            
        Returns:
            results: Dict containing analysis results
        """
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        
        print(f"Analyzing video: {video_path}")
        
        # Process manual selection if specified
        if manual_select:
            print(f"Manual fencer selection: {manual_select}")
            try:
                selected_ids = [int(id.strip()) for id in manual_select.split(',')]
                self.manual_fencer_ids = selected_ids
                print(f"Will focus analysis on fencer IDs: {selected_ids}")
            except ValueError:
                print("Warning: Invalid manual selection format. Expected comma-separated integers.")
                self.manual_fencer_ids = []
        
        # Check if we should detect and isolate fencers first
        if self.fencer_detector is not None:
            print(f"Running fencer detection and isolation...")
            
            # Process video with fencer detection first, then do pose analysis per detected fencer
            # Setup video output
            pose_analysis_path = os.path.join(output_dir, f"{video_name}_pose_analysis.mp4") if save_visualization else None
            
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error opening video file: {video_path}")
                return None
            
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if num_frames:
                total_frames = min(total_frames, num_frames)
            
            # Initialize video writer if output path is provided
            writer = None
            if pose_analysis_path:
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(pose_analysis_path, fourcc, fps, (frame_width, frame_height))
            
            # Process frames
            frame_analyses = []
            processed_frames = []
            
            frame_count = 0
            print(f"Processing video with fencer detection and pose estimation...")
            print(f"Total frames to process: {total_frames}")
            
            # Store initial detections to help with tracking consistency
            manual_boxes = {}
            fencer_detection_counts = {fencer_id: 0 for fencer_id in self.manual_fencer_ids} if self.manual_fencer_ids else {}
            last_known_boxes = {}  # Store last known location of each fencer
            
            # Track sword detections for statistics
            sword_detection_counts = {"blade-guard": 0, "blade-tip": 0, "fencing-blade": 0}
            
            while cap.isOpened() and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process every 3rd frame to improve speed
                if frame_count % 3 == 0:
                    print(f"Processing frame {frame_count}/{total_frames}")
                    
                    # Detect fencers using YOLOv8 - different approach for enhanced detector
                    if self.use_enhanced_detector:
                        # Use the enhanced detector with CNN classifier and sword detection
                        tracked_items, sword_detections = self.fencer_detector.track_and_classify(frame)
                        
                        # Filter tracked items if manual selection is active
                        if self.manual_fencer_ids:
                            tracked_items = [item for item in tracked_items 
                                            if item['fencer_id'] in self.manual_fencer_ids]
                        
                        # Create a copy of the frame for visualization
                        annotated_frame = self.fencer_detector.draw_enhanced_detections(
                            frame.copy(), tracked_items, sword_detections
                        )
                        
                        # Add frame number to the annotated frame
                        cv2.putText(annotated_frame, f"Frame: {frame_count}", (20, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        # Update sword detection counts
                        if sword_detections:
                            for det in sword_detections:
                                class_name = det['class_name']
                                if class_name in sword_detection_counts:
                                    sword_detection_counts[class_name] += 1
                        
                        # Process each detected fencer - now using tracked_items directly
                        for item in tracked_items:
                            fencer_id = item['fencer_id']
                            box = item['box']
                            pose_class = item.get('pose_class', 'neutral')
                            
                            # Update statistics
                            if fencer_id in fencer_detection_counts:
                                fencer_detection_counts[fencer_id] += 1
                            
                            # Always update the last known box for this fencer
                            last_known_boxes[fencer_id] = box
                            
                            # Estimate pose specifically for this fencer's bounding box
                            keypoints = self.pose_estimator.estimate_pose(frame, box)
                            
                            if keypoints is not None:
                                # Calculate angles
                                angles = self.pose_estimator.calculate_angles(keypoints)
                                
                                # Analyze movement
                                observations = self.pose_estimator.analyze_fencing_movement(angles)
                                
                                # Generate descriptive sentence - include CNN pose classification
                                base_sentence = generate_descriptive_sentence(observations)
                                sentence = f"{base_sentence} CNN classifies as: {pose_class}"
                                
                                # Draw pose on frame
                                annotated_frame = self.pose_estimator.draw_pose(annotated_frame, keypoints)
                                
                                # Store analysis results
                                frame_analyses.append({
                                    'frame_idx': frame_count,
                                    'fencer_id': fencer_id,
                                    'angles': angles,
                                    'observations': observations,
                                    'sentence': sentence,
                                    'box': [float(x) for x in box],
                                    'pose_class': pose_class
                                })
                    else:
                        # Use the original detection approach
                        detections = self.fencer_detector.detect_fencers(frame)
                        
                        # Track fencers to maintain identity
                        tracked_boxes = self.fencer_detector.track_fencers(frame, detections)
                        
                        # Filter tracked boxes if manual selection is active
                        if self.manual_fencer_ids:
                            # First apply the filter based on IDs
                            tracked_boxes = [(fencer_id, box) for fencer_id, box in tracked_boxes 
                                            if fencer_id in self.manual_fencer_ids]
                            
                            # Check if we're missing any selected fencers
                            current_ids = [fid for fid, _ in tracked_boxes]
                            missing_ids = [fid for fid in self.manual_fencer_ids if fid not in current_ids]
                            
                            # For missing fencers, use their last known location if available
                            for missing_id in missing_ids:
                                if missing_id in last_known_boxes:
                                    print(f"Using last known location for fencer {missing_id} in frame {frame_count}")
                                    tracked_boxes.append((missing_id, last_known_boxes[missing_id]))
                        
                        # Create a copy of the frame for visualization
                        annotated_frame = frame.copy()
                        
                        # Add frame number to the annotated frame
                        cv2.putText(annotated_frame, f"Frame: {frame_count}", (20, 30), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                        
                        if tracked_boxes:
                            # Update detection counts and last known locations
                            for fencer_id, box in tracked_boxes:
                                if fencer_id in fencer_detection_counts:
                                    fencer_detection_counts[fencer_id] += 1
                                
                                # Always update the last known box for this fencer
                                last_known_boxes[fencer_id] = box
                            
                            # Process each detected fencer
                            for fencer_id, box in tracked_boxes:
                                # Save box for first frame if it's a manually selected fencer
                                if frame_count == 0 and fencer_id in self.manual_fencer_ids:
                                    manual_boxes[fencer_id] = box
                                
                                # Estimate pose specifically for this fencer's bounding box
                                keypoints = self.pose_estimator.estimate_pose(frame, box)
                                
                                if keypoints is not None:
                                    # Calculate angles
                                    angles = self.pose_estimator.calculate_angles(keypoints)
                                    
                                    # Analyze movement
                                    observations = self.pose_estimator.analyze_fencing_movement(angles)
                                    
                                    # Generate descriptive sentence
                                    sentence = generate_descriptive_sentence(observations)
                                    
                                    # Draw pose on frame
                                    annotated_frame = self.pose_estimator.draw_pose(annotated_frame, keypoints)
                                    
                                    # Draw bounding box
                                    x1, y1, x2, y2 = box
                                    cv2.rectangle(annotated_frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                                    cv2.putText(annotated_frame, f"Fencer {fencer_id}", (int(x1), int(y1)-10), 
                                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                    
                                    # Add text with observations
                                    y_offset = int(y1) + 30
                                    for obs in observations:
                                        cv2.putText(annotated_frame, obs, (int(x1), y_offset), 
                                                  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                                        y_offset += 25
                                    
                                    # Store analysis results
                                    frame_analyses.append({
                                        'frame_idx': frame_count,
                                        'fencer_id': fencer_id,
                                        'angles': angles,
                                        'observations': observations,
                                        'sentence': sentence,
                                        'box': [float(x) for x in box]
                                    })
                        else:
                            print(f"No fencers detected in frame {frame_count}")
                    
                    # Save for visualization
                    if frame_count % 15 == 0 and save_visualization:  # Sample frames for visualization
                        processed_frames.append(annotated_frame)
                    
                    # Write frame to output video
                    if writer:
                        writer.write(annotated_frame)
                
                frame_count += 1
            
            # Release resources
            cap.release()
            if writer:
                writer.release()
            
            print(f"Processed {len(frame_analyses)} frames with pose estimation")
            if fencer_detection_counts:
                print("\nFencer detection statistics:")
                for fencer_id, count in fencer_detection_counts.items():
                    print(f"Fencer {fencer_id}: detected in {count} frames")
            
            # Print sword detection statistics if available
            if self.use_enhanced_detector:
                print("\nSword detection statistics:")
                if sum(sword_detection_counts.values()) > 0:
                    for part, count in sword_detection_counts.items():
                        print(f"  {part}: {count} detections")
                else:
                    print("  No sword parts detected")
        else:
            # Fall back to standard pose estimation without fencer detection
            print(f"Fencer detector not available. Running general pose estimation...")
            pose_analysis_path = os.path.join(output_dir, f"{video_name}_pose_analysis.mp4") if save_visualization else None
            
            print(f"Running pose estimation on video...")
            frame_analyses, processed_frames = process_video_with_pose(
                video_path, 
                pose_analysis_path,
                save_frames=save_visualization
            )
            
            print(f"Processed {len(frame_analyses)} frames with pose estimation")
        
        # Save sample frames
        if save_visualization and processed_frames:
            print("Saving sample frames with pose estimation...")
            plt.figure(figsize=(15, 10))
            for i in range(min(4, len(processed_frames))):
                idx = i * len(processed_frames) // 4
                plt.subplot(2, 2, i+1)
                plt.imshow(cv2.cvtColor(processed_frames[idx], cv2.COLOR_BGR2RGB))
                plt.title(f"Frame {i*len(processed_frames)//4}")
                plt.axis('off')
            
            plt.suptitle("Fencing Pose Analysis")
            plt.tight_layout()
            frames_path = os.path.join(output_dir, f"{video_name}_pose_frames.png")
            plt.savefig(frames_path)
            plt.close()
            print(f"Saved sample frames to {frames_path}")
        
        # Extract pose descriptions
        pose_descriptions = extract_pose_descriptions(frame_analyses)
        
        # Analyze pose descriptions to identify techniques
        technique_counts = {
            'attack': 0, 'defense': 0, 'parry': 0, 'riposte': 0, 
            'lunge': 0, 'advance': 0, 'retreat': 0, 'feint': 0,
            'movement': 0  # Add movement as an explicit category
        }
        
        # Track techniques per fencer
        fencer_technique_counts = {}
        
        # Extract frame-by-frame timeline data
        timeline_data = []
        
        for analysis in frame_analyses:
            frame_idx = analysis['frame_idx']
            fencer_id = analysis['fencer_id']
            desc_lower = analysis['sentence'].lower()
            
            # Get CNN classification if available
            cnn_pose_class = analysis.get('pose_class', 'unknown')
            
            # Initialize counter for this fencer if not exists
            if fencer_id not in fencer_technique_counts:
                fencer_technique_counts[fencer_id] = {
                    'attack': 0, 'defense': 0, 'parry': 0, 'riposte': 0, 
                    'lunge': 0, 'advance': 0, 'retreat': 0, 'feint': 0,
                    'movement': 0, 
                    # Add CNN classifications
                    'cnn_neutral': 0, 'cnn_attack': 0, 'cnn_defense': 0, 'cnn_lunge': 0
                }
            
            # Check if any technique was detected in this frame
            technique_detected = False
            for technique in [t for t in technique_counts.keys() if t != 'movement']:
                if technique in desc_lower:
                    timeline_data.append((frame_idx, technique, fencer_id))
                    technique_counts[technique] += 1
                    fencer_technique_counts[fencer_id][technique] += 1
                    technique_detected = True
                    break
            
            # Add generic "movement" if no specific technique detected
            if not technique_detected:
                timeline_data.append((frame_idx, "movement", fencer_id))
                technique_counts['movement'] += 1
                fencer_technique_counts[fencer_id]['movement'] += 1
            
            # Add CNN classification counts
            if cnn_pose_class != 'unknown':
                cnn_key = f'cnn_{cnn_pose_class}'
                if cnn_key in fencer_technique_counts[fencer_id]:
                    fencer_technique_counts[fencer_id][cnn_key] += 1
        
        # Generate visualizations
        if save_visualization:
            print("\nGenerating visualizations...")
            vis_paths = self.create_visualizations(
                video_name=video_name,
                technique_counts=technique_counts,
                fencer_technique_counts=fencer_technique_counts,
                timeline_data=timeline_data,
                frame_analyses=frame_analyses,
                processed_frames=processed_frames,
                output_dir=output_dir,
                use_enhanced_detector=self.use_enhanced_detector
            )
            print("Generated visualizations:")
            for name, path in vis_paths.items():
                print(f"  {name}: {path}")
        
        # Compile all results
        results = {
            'video_path': video_path,
            'pose_descriptions': pose_descriptions,
            'technique_counts': technique_counts,
            'fencer_technique_counts': fencer_technique_counts,
            'timeline_data': timeline_data,
            'frame_analyses': frame_analyses
        }
        
        # Save results to JSON
        results_path = os.path.join(output_dir, f"{video_name}_pose_analysis.json")
        with open(results_path, 'w') as f:
            # Convert non-serializable objects to strings
            serializable_results = {
                'video_path': results['video_path'],
                'pose_descriptions': results['pose_descriptions'],
                'technique_counts': results['technique_counts'],
                'fencer_technique_counts': {str(k): v for k, v in results['fencer_technique_counts'].items()},
                'timeline_data': [(idx, tech, fid) for idx, tech, fid in results['timeline_data']],
                'frame_analyses': [
                    {
                        'frame_idx': analysis['frame_idx'],
                        'fencer_id': analysis['fencer_id'],
                        'observations': analysis['observations'],
                        'sentence': analysis['sentence'],
                        'angles': {k: float(v) for k, v in analysis['angles'].items()},
                        'box': analysis.get('box', []),
                        'pose_class': analysis.get('pose_class', 'unknown')
                    }
                    for analysis in results['frame_analyses']
                ]
            }
            json.dump(serializable_results, f, indent=2)
        
        print(f"Analysis results saved to {results_path}")
        
        # Print technique summary by fencer
        print("\nTechnique Detection Summary:")
        for fencer_id, counts in sorted(fencer_technique_counts.items()):
            print(f"\nFencer {fencer_id}:")
            
            # Print CNN classification counts first if enhanced detector was used
            if self.use_enhanced_detector:
                print("  CNN Classifications:")
                for key in ['cnn_neutral', 'cnn_attack', 'cnn_defense', 'cnn_lunge']:
                    if key in counts and counts[key] > 0:
                        pose_name = key.replace('cnn_', '')
                        print(f"    {pose_name}: {counts[key]} frames")
            
            # Print traditional technique detection counts
            print("  Detected Techniques:")
            for technique, count in sorted([(k, v) for k, v in counts.items() if v > 0 and k != 'movement' and not k.startswith('cnn_')], 
                                         key=lambda x: x[1], reverse=True):
                print(f"    {technique}: {count} instances")
            if counts['movement'] > 0:
                print(f"    general movement: {counts['movement']} instances")
        
        return results
    
    def create_visualizations(self, video_name, technique_counts, fencer_technique_counts, timeline_data, frame_analyses, processed_frames, output_dir, use_enhanced_detector):
        """
        Create visualizations for the analysis
        
        Args:
            video_name: Name of the video
            technique_counts: Dictionary of technique counts
            fencer_technique_counts: Dictionary of technique counts per fencer
            timeline_data: List of (frame_idx, technique, fencer_id) tuples
            frame_analyses: List of frame analysis dictionaries
            processed_frames: List of processed frames with pose visualization
            output_dir: Directory to save visualizations
            use_enhanced_detector: Boolean indicating whether to use the enhanced detector
            
        Returns:
            vis_paths: Dictionary of visualization paths
        """
        vis_paths = {}
        
        # Define a consistent color map for techniques
        all_techniques = ['attack', 'defense', 'parry', 'riposte', 'lunge', 'advance', 'retreat', 'feint', 'movement']
        technique_to_color = {
            tech: plt.cm.tab10(i % 10) for i, tech in enumerate(all_techniques)
        }
        
        # 1. Pie charts of technique distribution (overall and per fencer)
        # Overall pie chart
        plt.figure(figsize=(10, 8))
        # Filter out techniques with zero count
        filtered_counts = {k: v for k, v in technique_counts.items() if v > 0 and k != 'movement'}
        
        if filtered_counts:
            plt.pie(
                filtered_counts.values(),
                labels=filtered_counts.keys(),
                autopct='%1.1f%%',
                startangle=90,
                shadow=True,
                colors=[technique_to_color[t] for t in filtered_counts.keys()]
            )
            plt.axis('equal')
            plt.title('Overall Technique Distribution')
        else:
            plt.text(0.5, 0.5, 'No specific techniques detected', 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
        
        pie_path = os.path.join(output_dir, f"{video_name}_pie.png")
        plt.savefig(pie_path)
        plt.close()
        vis_paths['pie_chart'] = pie_path
        
        # Pie charts per fencer
        for fencer_id, counts in fencer_technique_counts.items():
            plt.figure(figsize=(10, 8))
            filtered_counts = {k: v for k, v in counts.items() if v > 0 and k != 'movement'}
            
            if filtered_counts:
                plt.pie(
                    filtered_counts.values(),
                    labels=filtered_counts.keys(),
                    autopct='%1.1f%%',
                    startangle=90,
                    shadow=True,
                    colors=[technique_to_color[t] for t in filtered_counts.keys()]
                )
                plt.axis('equal')
                plt.title(f'Fencer {fencer_id} Technique Distribution')
            else:
                plt.text(0.5, 0.5, 'No specific techniques detected', 
                        horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
            
            fencer_pie_path = os.path.join(output_dir, f"{video_name}_fencer{fencer_id}_pie.png")
            plt.savefig(fencer_pie_path)
            plt.close()
            vis_paths[f'fencer{fencer_id}_pie'] = fencer_pie_path
        
        # 2. Bar chart of technique counts
        plt.figure(figsize=(12, 6))
        filtered_items = [(k, v) for k, v in technique_counts.items() if v > 0 and k != 'movement']
        
        if filtered_items:
            techniques, counts = zip(*sorted(filtered_items, key=lambda x: x[1], reverse=True))
            plt.bar(techniques, counts, color=[technique_to_color[t] for t in techniques])
            plt.xlabel('Techniques')
            plt.ylabel('Count')
            plt.title('Technique Frequency')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'No specific techniques detected', 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
        
        plt.tight_layout()
        bar_path = os.path.join(output_dir, f"{video_name}_bar.png")
        plt.savefig(bar_path)
        plt.close()
        vis_paths['bar_chart'] = bar_path
        
        # Bar charts per fencer
        for fencer_id, counts in fencer_technique_counts.items():
            plt.figure(figsize=(12, 6))
            filtered_items = [(k, v) for k, v in counts.items() if v > 0 and k != 'movement']
            
            if filtered_items:
                techniques, counts = zip(*sorted(filtered_items, key=lambda x: x[1], reverse=True))
                plt.bar(techniques, counts, color=[technique_to_color[t] for t in techniques])
                plt.xlabel('Techniques')
                plt.ylabel('Count')
                plt.title(f'Fencer {fencer_id} Technique Frequency')
                plt.xticks(rotation=45)
            else:
                plt.text(0.5, 0.5, 'No specific techniques detected', 
                        horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
            
            plt.tight_layout()
            fencer_bar_path = os.path.join(output_dir, f"{video_name}_fencer{fencer_id}_bar.png")
            plt.savefig(fencer_bar_path)
            plt.close()
            vis_paths[f'fencer{fencer_id}_bar'] = fencer_bar_path
        
        # 3. Timeline visualization (overall and per fencer)
        if timeline_data:
            # Overall timeline
            plt.figure(figsize=(15, 6))
            frames, techniques, _ = zip(*timeline_data)
            
            # Plot points with color-coded techniques
            unique_techniques = sorted(set(techniques))
            
            for technique in unique_techniques:
                indices = [i for i, (_, tech, _) in enumerate(timeline_data) if tech == technique]
                if indices:  # Only plot if there are instances
                    plt.scatter(
                        [frames[i] for i in indices],
                        [1 for _ in indices],
                        c=[technique_to_color[technique] for _ in indices],
                        label=technique,
                        s=50
                    )
            
            plt.yticks([])
            plt.xlabel('Frame')
            plt.title('Overall Technique Timeline')
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
            plt.tight_layout()
            timeline_path = os.path.join(output_dir, f"{video_name}_timeline.png")
            plt.savefig(timeline_path)
            plt.close()
            vis_paths['timeline'] = timeline_path
            
            # Timeline per fencer
            for fencer_id in fencer_technique_counts.keys():
                plt.figure(figsize=(15, 6))
                
                # Filter timeline data for this fencer
                fencer_timeline = [(f, t) for f, t, fid in timeline_data if fid == fencer_id]
                
                if fencer_timeline:
                    frames, techniques = zip(*fencer_timeline)
                    
                    # Plot points with color-coded techniques
                    unique_techniques = sorted(set(techniques))
                    
                    for technique in unique_techniques:
                        indices = [i for i, (_, tech) in enumerate(fencer_timeline) if tech == technique]
                        if indices:  # Only plot if there are instances
                            plt.scatter(
                                [frames[i] for i in indices],
                                [1 for _ in indices],
                                c=[technique_to_color[technique] for _ in indices],
                                label=technique,
                                s=50
                            )
                    
                    plt.yticks([])
                    plt.xlabel('Frame')
                    plt.title(f'Fencer {fencer_id} Technique Timeline')
                    plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=5)
                else:
                    plt.text(0.5, 0.5, 'No data for this fencer', 
                            horizontalalignment='center', verticalalignment='center')
                    plt.axis('off')
                
                plt.tight_layout()
                fencer_timeline_path = os.path.join(output_dir, f"{video_name}_fencer{fencer_id}_timeline.png")
                plt.savefig(fencer_timeline_path)
                plt.close()
                vis_paths[f'fencer{fencer_id}_timeline'] = fencer_timeline_path
        
        # 4. Joint angle analysis
        # Extract a few key joint angles over time
        angle_data = {}
        for analysis in frame_analyses:
            frame_idx = analysis['frame_idx']
            fencer_id = analysis['fencer_id']
            
            if fencer_id not in angle_data:
                angle_data[fencer_id] = {'frames': [], 'right_arm': [], 'left_arm': [], 'torso': [], 'right_leg': [], 'left_leg': []}
            
            angle_data[fencer_id]['frames'].append(frame_idx)
            
            angles = analysis['angles']
            angle_data[fencer_id]['right_arm'].append(angles.get('right_arm', float('nan')))
            angle_data[fencer_id]['left_arm'].append(angles.get('left_arm', float('nan')))
            angle_data[fencer_id]['torso'].append(angles.get('torso_vertical', float('nan')))
            angle_data[fencer_id]['right_leg'].append(angles.get('right_leg', float('nan')))
            angle_data[fencer_id]['left_leg'].append(angles.get('left_leg', float('nan')))
        
        # Plot angles for all fencers, with each fencer in a separate file
        if angle_data:
            # Overall angle plot
            plt.figure(figsize=(15, 10))
            
            for i, (fencer_id, data) in enumerate(list(angle_data.items())[:3]):
                plt.subplot(3, 1, i+1)
                
                # Plot arm angles
                if data['right_arm']:
                    plt.plot(data['frames'], data['right_arm'], 'r-', label='Right Arm Angle')
                
                if data['left_arm']:
                    plt.plot(data['frames'], data['left_arm'], 'g-', label='Left Arm Angle')
                
                # Plot leg angles
                if data['right_leg']:
                    plt.plot(data['frames'], data['right_leg'], 'b--', label='Right Leg Angle')
                
                if data['left_leg']:
                    plt.plot(data['frames'], data['left_leg'], 'm--', label='Left Leg Angle')
                
                plt.title(f'Fencer {fencer_id} Joint Angles')
                plt.xlabel('Frame')
                plt.ylabel('Angle (degrees)')
                plt.legend()
            
            plt.tight_layout()
            angles_path = os.path.join(output_dir, f"{video_name}_angles.png")
            plt.savefig(angles_path)
            plt.close()
            vis_paths['angles'] = angles_path
            
            # Individual angle plots per fencer
            for fencer_id, data in angle_data.items():
                plt.figure(figsize=(15, 8))
                
                # Plot arm angles
                if data['right_arm']:
                    plt.plot(data['frames'], data['right_arm'], 'r-', label='Right Arm Angle')
                
                if data['left_arm']:
                    plt.plot(data['frames'], data['left_arm'], 'g-', label='Left Arm Angle')
                
                # Plot leg angles
                if data['right_leg']:
                    plt.plot(data['frames'], data['right_leg'], 'b--', label='Right Leg Angle')
                
                if data['left_leg']:
                    plt.plot(data['frames'], data['left_leg'], 'm--', label='Left Leg Angle')
                
                plt.title(f'Fencer {fencer_id} Joint Angles')
                plt.xlabel('Frame')
                plt.ylabel('Angle (degrees)')
                plt.legend()
                
                plt.tight_layout()
                fencer_angles_path = os.path.join(output_dir, f"{video_name}_fencer{fencer_id}_angles.png")
                plt.savefig(fencer_angles_path)
                plt.close()
                vis_paths[f'fencer{fencer_id}_angles'] = fencer_angles_path
        
        # 5. Summary visualization
        plt.figure(figsize=(15, 10))
        
        # Add detection statistics to the title
        detection_stats = []
        if hasattr(self, 'manual_fencer_ids') and self.manual_fencer_ids:
            for fencer_id in sorted(fencer_technique_counts.keys()):
                frame_count = len([1 for analysis in frame_analyses if analysis['fencer_id'] == fencer_id])
                detection_stats.append(f"Fencer {fencer_id}: {frame_count} frames")
        
        detection_stats_text = " | ".join(detection_stats) if detection_stats else ""
        
        # Technique pie chart (top left)
        plt.subplot(2, 2, 1)
        filtered_counts = {k: v for k, v in technique_counts.items() if v > 0 and k != 'movement'}
        if filtered_counts:
            plt.pie(
                filtered_counts.values(),
                labels=filtered_counts.keys(),
                autopct='%1.1f%%',
                startangle=90,
                colors=[technique_to_color[t] for t in filtered_counts.keys()]
            )
            plt.axis('equal')
            plt.title('Technique Distribution')
        else:
            plt.text(0.5, 0.5, 'No specific techniques detected', 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
        
        # Timeline snippet (top right)
        plt.subplot(2, 2, 2)
        if timeline_data:
            frames_subset = [f for f, t, _ in timeline_data if t != 'movement'][:100]  # Focus on non-movement techniques
            if frames_subset:
                # Filter to get subset with actual techniques
                subset_data = [(f, t, fid) for f, t, fid in timeline_data if t != 'movement'][:100]
                frames, techniques, fencer_ids = zip(*subset_data)
                
                unique_techniques = sorted(set(techniques))
                
                for technique in unique_techniques:
                    indices = [i for i, (_, tech, _) in enumerate(subset_data) if tech == technique]
                    if indices:
                        plt.scatter(
                            [frames[i] for i in indices],
                            [1 for _ in indices],
                            c=[technique_to_color[technique] for _ in indices],
                            label=technique,
                            s=30
                        )
            else:
                # If no specific techniques, show movement timeline but color-coded by fencer
                frames, techniques, fencer_ids = zip(*timeline_data[:100])
                
                # Create a scatter plot with points colored by fencer ID
                unique_fencers = sorted(set(fencer_ids))
                for fencer_id in unique_fencers:
                    indices = [i for i, (_, _, fid) in enumerate(timeline_data[:100]) if fid == fencer_id]
                    if indices:
                        plt.scatter(
                            [frames[i] for i in indices],
                            [1 for _ in indices],
                            label=f"Fencer {fencer_id}",
                            s=30
                        )
                
                plt.yticks([])
                plt.xlabel('Frame')
                plt.title('Movement Timeline by Fencer (First 100 Frames)')
                plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=len(unique_fencers), fontsize='small')
        
        # Frame sample (bottom left)
        plt.subplot(2, 2, 3)
        if processed_frames:
            plt.imshow(cv2.cvtColor(processed_frames[0], cv2.COLOR_BGR2RGB))
            plt.title(f"Sample Frame")
            plt.axis('off')
        
        # Add detection statistics as text annotation
        if detection_stats:
            y_pos = 0.05
            for stat in detection_stats:
                plt.figtext(0.5, y_pos, stat, ha='center', fontsize=10, 
                           bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.5'))
                y_pos += 0.04
        
        # Technique bar chart (bottom right)
        plt.subplot(2, 2, 4)
        filtered_items = [(k, v) for k, v in technique_counts.items() if v > 0 and k != 'movement']
        if filtered_items:
            techniques, counts = zip(*sorted(filtered_items, key=lambda x: x[1], reverse=True))
            plt.bar(techniques, counts, color=[technique_to_color[t] for t in techniques])
            plt.xlabel('Techniques')
            plt.ylabel('Count')
            plt.title('Technique Frequency')
            plt.xticks(rotation=45, fontsize='small')
        else:
            # If no specific techniques, show movement count by fencer
            if fencer_technique_counts:
                fencer_ids = sorted(fencer_technique_counts.keys())
                movement_counts = [fencer_technique_counts[fid].get('movement', 0) for fid in fencer_ids]
                fencer_labels = [f"Fencer {fid}" for fid in fencer_ids]
                
                if any(movement_counts):
                    plt.bar(fencer_labels, movement_counts)
                    plt.xlabel('Fencer')
                    plt.ylabel('Count')
                    plt.title('Movement Count by Fencer')
                else:
                    plt.text(0.5, 0.5, 'No data available', 
                            horizontalalignment='center', verticalalignment='center')
                    plt.axis('off')
            else:
                movement_count = technique_counts.get('movement', 0)
                if movement_count > 0:
                    plt.bar(['movement'], [movement_count], color=technique_to_color['movement'])
                    plt.xlabel('Types')
                    plt.ylabel('Count')
                    plt.title('Movement Count')
                else:
                    plt.text(0.5, 0.5, 'No data available', 
                            horizontalalignment='center', verticalalignment='center')
                    plt.axis('off')
        
        # Add subtitle with detection statistics if available
        subtitle = f"Analysis of {len(frame_analyses)} frames"
        if detection_stats_text:
            subtitle += f" | {detection_stats_text}"
        
        plt.suptitle(f"Fencing Analysis Summary: {video_name}\n{subtitle}", fontsize=16)
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        
        summary_path = os.path.join(output_dir, f"{video_name}_summary.png")
        plt.savefig(summary_path)
        plt.close()
        vis_paths['summary'] = summary_path
        
        # Create per-fencer summary visualizations
        for fencer_id, counts in fencer_technique_counts.items():
            plt.figure(figsize=(15, 10))
            
            # Calculate detection stats for this fencer
            frame_count = len([1 for analysis in frame_analyses if analysis['fencer_id'] == fencer_id])
            frame_percent = (frame_count / len(timeline_data) * 100) if timeline_data else 0
            detection_stat = f"Detected in {frame_count} frames ({frame_percent:.1f}%)"
            
            # Technique pie chart (top left)
            plt.subplot(2, 2, 1)
            filtered_counts = {k: v for k, v in counts.items() if v > 0 and k != 'movement'}
            if filtered_counts:
                plt.pie(
                    filtered_counts.values(),
                    labels=filtered_counts.keys(),
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=[technique_to_color[t] for t in filtered_counts.keys()]
                )
                plt.axis('equal')
                plt.title(f'Fencer {fencer_id} Technique Distribution')
            else:
                movement_count = counts.get('movement', 0)
                if movement_count > 0:
                    plt.pie([movement_count], labels=['movement'], autopct='%1.1f%%', startangle=90,
                          colors=[technique_to_color['movement']])
                    plt.axis('equal')
                    plt.title(f'Fencer {fencer_id} - Only Movement Detected')
                else:
                    plt.text(0.5, 0.5, 'No techniques detected for this fencer', 
                            horizontalalignment='center', verticalalignment='center')
                    plt.axis('off')
            
            # Timeline snippet (top right)
            plt.subplot(2, 2, 2)
            fencer_timeline = [(f, t) for f, t, fid in timeline_data if fid == fencer_id]
            if fencer_timeline:
                frames, techniques = zip(*fencer_timeline)
                
                # Plot the frames on a timeline to visualize coverage
                plt.eventplot([frames], colors=['green'], lineoffsets=[1], linelengths=[0.5])
                
                # Annotate with frame indices for key moments (first, last, and some in between)
                if len(frames) > 0:
                    plt.annotate(f"Start: {frames[0]}", xy=(frames[0], 1), 
                                xytext=(frames[0], 1.2), fontsize=9,
                                arrowprops=dict(arrowstyle='->'))
                    
                    if len(frames) > 1:
                        plt.annotate(f"End: {frames[-1]}", xy=(frames[-1], 1), 
                                    xytext=(frames[-1], 1.2), fontsize=9,
                                    arrowprops=dict(arrowstyle='->'))
                    
                    # Add annotations for some middle frames if there are enough
                    if len(frames) > 10:
                        middle_idx = len(frames) // 2
                        plt.annotate(f"Mid: {frames[middle_idx]}", xy=(frames[middle_idx], 1), 
                                    xytext=(frames[middle_idx], 1.2), fontsize=9,
                                    arrowprops=dict(arrowstyle='->'))
                
                plt.yticks([])
                plt.xlabel('Frame')
                plt.title(f'Fencer {fencer_id} Timeline Coverage')
                
                # Add a second axis with technique distribution if techniques are detected
                unique_techniques = sorted(set(techniques))
                non_movement_techniques = [t for t in unique_techniques if t != 'movement']
                
                if non_movement_techniques:
                    # If specific techniques exist, show them color-coded
                    ax2 = plt.twinx()
                    ax2.set_yticks([])
                    
                    for technique in non_movement_techniques:
                        indices = [i for i, (_, tech) in enumerate(fencer_timeline) if tech == technique]
                        if indices:
                            ax2.scatter(
                                [frames[i] for i in indices],
                                [1.5 for _ in indices],  # Position above the timeline
                                c=[technique_to_color[technique] for _ in indices],
                                label=technique,
                                s=30
                            )
                
                    ax2.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=min(3, len(non_movement_techniques)), fontsize='small')
            else:
                plt.text(0.5, 0.5, 'No timeline data for this fencer', 
                        horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
            
            # Angle data (bottom left)
            plt.subplot(2, 2, 3)
            if fencer_id in angle_data and angle_data[fencer_id]['frames']:
                data = angle_data[fencer_id]
                
                # Plot arm angles
                if data['right_arm']:
                    plt.plot(data['frames'], data['right_arm'], 'r-', label='Right Arm Angle')
                
                if data['left_arm']:
                    plt.plot(data['frames'], data['left_arm'], 'g-', label='Left Arm Angle')
                
                # Add annotations about data coverage
                if data['frames']:
                    plt.annotate(f"Data points: {len(data['frames'])}", 
                               xy=(0.05, 0.95), xycoords='axes fraction',
                               fontsize=9, backgroundcolor='white')
                
                plt.xlabel('Frame')
                plt.ylabel('Angle (degrees)')
                plt.title(f'Fencer {fencer_id} Arm Angles')
                plt.legend(fontsize='small')
            else:
                plt.text(0.5, 0.5, 'No angle data for this fencer', 
                        horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
            
            # Technique bar chart or Frame detection chart (bottom right)
            plt.subplot(2, 2, 4)
            filtered_items = [(k, v) for k, v in counts.items() if v > 0 and k != 'movement']
            if filtered_items:
                techniques, counts = zip(*sorted(filtered_items, key=lambda x: x[1], reverse=True))
                plt.bar(techniques, counts, color=[technique_to_color[t] for t in techniques])
                plt.xlabel('Techniques')
                plt.ylabel('Count')
                plt.title(f'Fencer {fencer_id} Technique Frequency')
                plt.xticks(rotation=45, fontsize='small')
            else:
                # If no specific techniques, create a frame detection visualization
                # Extract frame numbers when this fencer was detected
                fencer_frames = [analysis['frame_idx'] for analysis in frame_analyses if analysis['fencer_id'] == fencer_id]
                
                if fencer_frames:
                    # Create a timeline with markers for when the fencer was detected
                    total_frames = timeline_data[-1][0] if timeline_data else frame_count  # Use max frame idx
                    frame_presence = np.zeros(total_frames + 1)
                    for f in fencer_frames:
                        if f <= total_frames:  # Safety check
                            frame_presence[f] = 1
                    
                    # Plot detection pattern
                    plt.step(range(len(frame_presence)), frame_presence, where='mid')
                    plt.fill_between(range(len(frame_presence)), frame_presence, step="mid", alpha=0.4)
                    plt.xlabel('Frame')
                    plt.ylabel('Detected')
                    plt.title(f'Fencer {fencer_id} Detection Pattern')
                    plt.yticks([0, 1], ['No', 'Yes'])
                else:
                    movement_count = counts.get('movement', 0)
                    if movement_count > 0:
                        plt.bar(['movement'], [movement_count], color=technique_to_color['movement'])
                        plt.xlabel('Types')
                        plt.ylabel('Count')
                        plt.title('Movement Count')
                    else:
                        plt.text(0.5, 0.5, 'No data available', 
                                horizontalalignment='center', verticalalignment='center')
                        plt.axis('off')
            
            plt.suptitle(f"Fencer {fencer_id} Analysis Summary\n{detection_stat}", fontsize=16)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            fencer_summary_path = os.path.join(output_dir, f"{video_name}_fencer{fencer_id}_summary.png")
            plt.savefig(fencer_summary_path)
            plt.close()
            vis_paths[f'fencer{fencer_id}_summary'] = fencer_summary_path
        
        return vis_paths
    
    def generate_report(self, results, output_path):
        """
        Generate a simple HTML report from pose analysis results
        
        Args:
            results: Analysis results dictionary
            output_path: Path to save the HTML report
        """
        # Get video name
        video_name = os.path.splitext(os.path.basename(results['video_path']))[0]
        output_dir = os.path.dirname(output_path)
        
        # Create HTML content
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fencing Pose Analysis: {video_name}</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    line-height: 1.6;
                    max-width: 1200px;
                    margin: 0 auto;
                    padding: 20px;
                }}
                .header {{
                    background-color: #f5f5f5;
                    padding: 20px;
                    border-radius: 5px;
                    margin-bottom: 20px;
                }}
                .section {{
                    margin-bottom: 30px;
                    border-bottom: 1px solid #ddd;
                    padding-bottom: 20px;
                }}
                table {{
                    border-collapse: collapse;
                    width: 100%;
                }}
                th, td {{
                    border: 1px solid #ddd;
                    padding: 8px;
                    text-align: left;
                }}
                th {{
                    background-color: #f2f2f2;
                }}
                .visualization {{
                    margin: 20px 0;
                    max-width: 100%;
                }}
                .visualization img {{
                    max-width: 100%;
                    height: auto;
                }}
                .two-column {{
                    display: flex;
                    flex-wrap: wrap;
                    gap: 20px;
                }}
                .two-column > div {{
                    flex: 1;
                    min-width: 300px;
                }}
                .fencer-section {{
                    background-color: #f9f9f9;
                    border-radius: 5px;
                    padding: 15px;
                    margin-bottom: 20px;
                }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Fencing Pose Analysis Report</h1>
                <p>Video: {results['video_path']}</p>
                <p>Analysis Date: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
            </div>
            
            <div class="section">
                <h2>Summary</h2>
                <p>This report presents a pose analysis of fencing techniques and movements.</p>
                <p>Number of analyzed frames: {len(results['frame_analyses'])}</p>
                
                <div class="visualization">
                    <h3>Analysis Summary</h3>
                    <img src="{video_name}_summary.png" alt="Analysis Summary">
                </div>
            </div>
            
            <div class="section">
                <h2>Technique Detection</h2>
                <div class="two-column">
                    <div>
                <table>
                    <tr>
                        <th>Technique</th>
                        <th>Count</th>
                    </tr>
        """
        
        # Add technique counts
        for technique, count in sorted(results['technique_counts'].items(), key=lambda x: x[1], reverse=True):
            if count > 0:
                html_content += f"""
                    <tr>
                        <td>{technique}</td>
                        <td>{count}</td>
                    </tr>
            """
        
        html_content += """
                </table>
            </div>
                    <div>
        """
        
        # Add technique distribution visualization if it exists
        if os.path.exists(os.path.join(output_dir, f"{video_name}_pie.png")):
                html_content += f"""
                <div class="visualization">
                            <h3>Technique Distribution</h3>
                            <img src="{video_name}_pie.png" alt="Technique Distribution">
                </div>
                """
        
        html_content += """
            </div>
                </div>
            </div>
            
            <div class="section">
                <h2>Per-Fencer Analysis</h2>
        """
        
        # Add per-fencer sections
        fencer_ids = sorted(results.get('fencer_technique_counts', {}).keys())
        for fencer_id in fencer_ids:
            html_content += f"""
                <div class="fencer-section">
                <h3>Fencer {fencer_id}</h3>
                    
                    <div class="visualization">
                        <img src="{video_name}_fencer{fencer_id}_summary.png" alt="Fencer {fencer_id} Summary">
            </div>
            
                    <h4>Technique Distribution</h4>
                <table>
                    <tr>
                            <th>Technique</th>
                            <th>Count</th>
                    </tr>
        """
        
            # Add technique counts for this fencer
            counts = results['fencer_technique_counts'].get(str(fencer_id), {})
            if isinstance(counts, dict):  # Just to be safe
                for technique, count in sorted([(k, v) for k, v in counts.items() if v > 0], 
                                            key=lambda x: x[1], reverse=True):
                    html_content += f"""
                    <tr>
                                <td>{technique}</td>
                                <td>{count}</td>
                    </tr>
                """
        
        html_content += """
                </table>
                </div>
            """
        
        html_content += """
            </div>
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_path, 'w') as f:
            f.write(html_content)
        
        print(f"Report generated at {output_path}")

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Simplified Fencing Pose Analysis")
    parser.add_argument("video_path", help="Path to the fencing video")
    parser.add_argument("--output_dir", default="results", help="Directory to save results")
    parser.add_argument("--max_frames", type=int, default=None, help="Maximum number of frames to process")
    parser.add_argument("--no_viz", action="store_true", help="Disable visualization outputs")
    parser.add_argument("--manual_select", help="Optional comma-separated list of fencer IDs to manually select")
    parser.add_argument("--first_only", action="store_true", help="Only show the first frame with detected fencers and exit")
    
    args = parser.parse_args()
    
    # Check if video exists
    if not os.path.exists(args.video_path):
        print(f"Error: Video file {args.video_path} not found!")
        return
    
    # Initialize analyzer
    analyzer = SimplifiedFencingAnalyzer()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Show first frame for analysis
    if args.first_only:
        annotated_frame_path = analyzer.show_first_frame_for_selection(args.video_path, args.output_dir)
        print("\nFirst frame analysis complete!")
        print(f"Annotated frame saved to: {annotated_frame_path}")
        print("\nTo analyze specific fencers, run the command again with --manual_select followed by comma-separated fencer IDs")
        print("Example: python advanced_fencing_analyzer.py video.mp4 --manual_select 0,1")
        return
    
    # Show first frame for selection if manual select is specified
    if args.manual_select:
        analyzer.show_first_frame_for_selection(args.video_path, args.output_dir)
    
    # Analyze video with pose estimation
    results = analyzer.analyze_video(
        video_path=args.video_path,
        output_dir=args.output_dir,
        num_frames=args.max_frames,
        save_visualization=not args.no_viz,
        manual_select=args.manual_select
    )
    
    # Generate report
    video_name = os.path.splitext(os.path.basename(args.video_path))[0]
    report_path = os.path.join(args.output_dir, f"{video_name}_pose_report.html")
    analyzer.generate_report(results, report_path)
    
    print("\nAnalysis complete!")
    print(f"Results saved to {args.output_dir}")
    print(f"Report generated at {report_path}")

if __name__ == "__main__":
    main()