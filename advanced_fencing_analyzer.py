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
# Import our enhanced fencer detector with CNN
from enhanced_fencer_detector import EnhancedFencerDetector

class SimplifiedFencingAnalyzer:
    """
    Simplified fencing analyzer that combines pose estimation with fencer detection
    """
    def __init__(self, pose_model_path=None, use_enhanced_detector=False):
        """
        Initialize the analyzer
        
        Args:
            pose_model_path: Path to trained CNN pose classifier model
            use_enhanced_detector: Whether to use enhanced detector with CNN
        """
        print("Initializing Simplified Fencing Analyzer...")
        
        # Initialize pose estimator
        print("Loading pose estimator...")
        try:
            from pose_estimation_helper import MMPOSE_AVAILABLE
            self.pose_estimator = PoseEstimator(use_mmpose=MMPOSE_AVAILABLE)
            print(f"Pose estimator initialized (using MMPose: {MMPOSE_AVAILABLE})")
        except ImportError as e:
            print(f"Warning: mmpose not available. Using mediapipe as fallback.")
            self.pose_estimator = PoseEstimator(use_mmpose=False)
        
        # Store configuration
        self.use_enhanced_detector = use_enhanced_detector
        
        # Initialize fencer detector - use enhanced one if specified
        if self.use_enhanced_detector:
            print("Using device: cpu")
            self.fencer_detector = EnhancedFencerDetector(
                pose_model_path=pose_model_path
            )
            print("Enhanced fencer detector initialized with CNN pose classifier")
        else:
            self.fencer_detector = FencerDetector()
            print("Basic fencer detector initialized")
        
        # Store manual fencer selection IDs
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
                tracked_items = self.fencer_detector.track_and_classify(frame)
                
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
        
        plt.title("First Frame - Detected Fencers")
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
    
    def analyze_video(self, video_path, output_dir=None, save_visualization=True, max_frames=None):
        """
        Analyze a fencing video and generate a report
        
        Args:
            video_path: Path to the video file
            output_dir: Directory to save outputs. If None, use video name.
            save_visualization: Whether to save visualizations
            max_frames: Maximum number of frames to process. If None, process all.
            
        Returns:
            Dictionary of analysis results
        """
        # Set up output directory
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        if output_dir is None:
            output_dir = f"{video_name}_analysis"
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Run analysis
        frame_analyses, processed_frames = self.process_video(
            video_path, output_dir, max_frames=max_frames
        )
        
        # Extract pose descriptions and analyze the sequence
        pose_descriptions = extract_pose_descriptions(frame_analyses)
        
        # Analyze fencing techniques
        technique_counts, fencer_technique_counts, hit_events = self.analyze_fencing_techniques(frame_analyses)
        
        # Convert tuple results to dictionaries if needed
        technique_counts_dict = {}
        if isinstance(technique_counts, dict):
            technique_counts_dict = technique_counts
        else:
            print(f"Warning: technique_counts is not a dictionary, converting from {type(technique_counts)}")
            technique_counts_dict = {
                'attack': 0, 'defense': 0, 'parry': 0, 'riposte': 0, 
                'lunge': 0, 'advance': 0, 'retreat': 0, 'feint': 0, 
                'hit': 0, 'movement': 0
            }
        
        fencer_technique_counts_dict = {}
        if isinstance(fencer_technique_counts, dict):
            fencer_technique_counts_dict = fencer_technique_counts
        else:
            print(f"Warning: fencer_technique_counts is not a dictionary, converting from {type(fencer_technique_counts)}")
            for analysis in frame_analyses:
                fencer_id = analysis.get('fencer_id')
                if fencer_id is not None and fencer_id not in fencer_technique_counts_dict:
                    fencer_technique_counts_dict[fencer_id] = {
                        'attack': 0, 'defense': 0, 'parry': 0, 'riposte': 0, 
                        'lunge': 0, 'advance': 0, 'retreat': 0, 'feint': 0,
                        'hit': 0, 'movement': 0, 
                        'cnn_neutral': 0, 'cnn_attack': 0, 'cnn_defense': 0, 'cnn_lunge': 0
                    }
        
        hit_events_list = []
        if isinstance(hit_events, list):
            hit_events_list = hit_events
        else:
            print(f"Warning: hit_events is not a list, converting from {type(hit_events)}")
        
        # Generate visualizations
        if save_visualization:
            # Create a type-safe wrapper for create_visualizations
            def create_visualizations_safe():
                try:
                    return self.create_visualizations(
                        video_name,
                        technique_counts_dict,
                        fencer_technique_counts_dict,
                        hit_events_list,
                        frame_analyses,
                        processed_frames,
                        output_dir,
                        self.use_enhanced_detector
                    )
                except Exception as e:
                    print(f"Error during visualization: {e}")
                    return {}
            
            vis_paths = create_visualizations_safe()
        
        # Generate report
        report_path = os.path.join(output_dir, f"{video_name}_report.html")
        self.generate_report({
            'video_name': video_name,
            'pose_descriptions': pose_descriptions,
            'technique_counts': technique_counts,
            'fencer_technique_counts': fencer_technique_counts,
            'hit_events': hit_events,
            'frame_analyses': frame_analyses
        }, report_path)
        
        # Return results
        return {
            'video_name': video_name,
            'frame_analyses': frame_analyses,
            'pose_descriptions': pose_descriptions,
            'technique_counts': technique_counts,
            'fencer_technique_counts': fencer_technique_counts,
            'hit_events': hit_events
        }
    
    def analyze_fencing_techniques(self, frame_analyses):
        """
        Analyze fencing techniques based on pose sequences and sword positions
        
        Args:
            frame_analyses: List of frame analysis results
            
        Returns:
            technique_counts: Dictionary of technique counts
            fencer_technique_counts: Dictionary of technique counts per fencer
            hit_events: List of detected hit events
        """
        # Initialize counters
        technique_counts = {
            'attack': 0, 'defense': 0, 'parry': 0, 'riposte': 0, 
            'lunge': 0, 'advance': 0, 'retreat': 0, 'feint': 0, 
            'hit': 0, 'movement': 0
        }
        
        fencer_technique_counts = {}
        hit_events = []
        
        # Group analyses by fencer and sort by frame
        fencer_frames = {}
        for analysis in frame_analyses:
            fencer_id = analysis['fencer_id']
            frame_idx = analysis['frame_idx']
            
            if fencer_id not in fencer_frames:
                fencer_frames[fencer_id] = []
            
            fencer_frames[fencer_id].append(analysis)
        
        # Sort each fencer's frames by index
        for fencer_id in fencer_frames:
            fencer_frames[fencer_id].sort(key=lambda x: x['frame_idx'])
        
        # Initialize state tracking for each fencer
        fencer_states = {}
        
        # Process each fencer's frames to detect techniques
        for fencer_id, frames in fencer_frames.items():
            # Initialize fencer technique counter if needed
            if fencer_id not in fencer_technique_counts:
                fencer_technique_counts[fencer_id] = {
                    'attack': 0, 'defense': 0, 'parry': 0, 'riposte': 0,
                    'lunge': 0, 'advance': 0, 'retreat': 0, 'feint': 0,
                    'hit': 0, 'movement': 0,
                    # Add CNN classifications
                    'cnn_neutral': 0, 'cnn_attack': 0, 'cnn_defense': 0, 'cnn_lunge': 0
                }
            # Ensure fencer_technique_counts[fencer_id] is a dictionary
            elif not isinstance(fencer_technique_counts[fencer_id], dict):
                print(f"Warning: Fencer {fencer_id} counts was not a dict ({type(fencer_technique_counts[fencer_id])}). Re-initializing.")
                fencer_technique_counts[fencer_id] = {
                    'attack': 0, 'defense': 0, 'parry': 0, 'riposte': 0,
                    'lunge': 0, 'advance': 0, 'retreat': 0, 'feint': 0,
                    'hit': 0, 'movement': 0,
                    'cnn_neutral': 0, 'cnn_attack': 0, 'cnn_defense': 0, 'cnn_lunge': 0
                }
            
            # Initialize this fencer's state
            if fencer_id not in fencer_states:
                fencer_states[fencer_id] = {
                    'last_pose': 'neutral',
                    'attack_start_frame': None,
                    'potential_hit': False,
                    'consecutive_attack_frames': 0
                }
            
            # Track attack sequences
            for i, analysis in enumerate(frames):
                frame_idx = analysis['frame_idx']
                pose_class = analysis.get('pose_class', 'neutral')
                observations = analysis.get('observations', [])
                angles = analysis.get('angles', {})
                
                # Convert observations list to indicators
                has_arm_extended = False
                has_arm_defensive = False
                has_forward_motion = False
                has_backward_motion = False
                
                # Look through observations list for relevant patterns
                for obs in observations:
                    if isinstance(obs, str):
                        if 'arm extended' in obs.lower():
                            has_arm_extended = True
                        if 'defensive position' in obs.lower():
                            has_arm_defensive = True
                        if 'moving forward' in obs.lower():
                            has_forward_motion = True
                        if 'moving backward' in obs.lower():
                            has_backward_motion = True
                
                # Add to CNN classification count
                cnn_key = f'cnn_{pose_class}'
                if cnn_key in fencer_technique_counts[fencer_id]:
                    fencer_technique_counts[fencer_id][cnn_key] += 1
                
                # Check for specific techniques based on observations and pose
                techniques_detected = []
                
                # Look for lunges in body angle changes
                torso_angle = angles.get('torso_vertical', 90)
                knee_angle = angles.get('right_leg', 180)
                
                # Detect lunge - arm extended with forward leaning and bent knee
                if has_arm_extended and torso_angle < 75 and knee_angle < 130:
                    techniques_detected.append('lunge')
                    
                # Detect attack - arm extended and moving forward
                if has_arm_extended and pose_class == 'attack':
                    techniques_detected.append('attack')
                    # Track consecutive attack frames
                    if fencer_states[fencer_id]['consecutive_attack_frames'] == 0:
                        fencer_states[fencer_id]['attack_start_frame'] = frame_idx
                    fencer_states[fencer_id]['consecutive_attack_frames'] += 1
                else:
                    # Reset if not in attack
                    fencer_states[fencer_id]['consecutive_attack_frames'] = 0
                
                # Attack must be sustained for several frames to be counted
                if fencer_states[fencer_id]['consecutive_attack_frames'] >= 3:
                    if 'attack' not in techniques_detected:
                        techniques_detected.append('attack')
                
                # Look for defensive positions
                if has_arm_defensive and pose_class == 'defense':
                    techniques_detected.append('defense')
                    
                    # Check if this is a parry after opponent attack
                    for other_id, other_state in fencer_states.items():
                        if other_id != fencer_id and other_state.get('consecutive_attack_frames', 0) > 0:
                            # This could be a parry in response to attack
                            techniques_detected.append('parry')
                            break
                
                # Detect retreat based on backward motion
                if has_backward_motion:
                    techniques_detected.append('retreat')
                
                # Detect advance based on forward motion
                if has_forward_motion and 'attack' not in techniques_detected:
                    techniques_detected.append('advance')
                
                # Add generic movement if nothing specific detected
                if not techniques_detected:
                    techniques_detected.append('movement')
                
                # Update the technique counts
                for technique in techniques_detected:
                    technique_counts[technique] += 1
                    fencer_technique_counts[fencer_id][technique] += 1
                
                # Update fencer state
                fencer_states[fencer_id]['last_pose'] = pose_class
        
        # Look for potential hits across fencers
        # We consider a hit when one fencer is attacking and their sword tip
        # comes close to the other fencer's body
        if len(fencer_frames) >= 2 and any(frames for frames in fencer_frames.values() if frames): 
            min_frame = min([frames[0]['frame_idx'] for frames in fencer_frames.values() if frames])
            max_frame = max([frames[-1]['frame_idx'] for frames in fencer_frames.values() if frames])
            
            for frame_idx in range(min_frame, max_frame + 1):
                attacking_fencers = []
                defending_fencers = []
                
                # Find fencers attacking or defending in this frame
                for fencer_id, frames in fencer_frames.items():
                    matching_frames = [f for f in frames if f['frame_idx'] == frame_idx]
                    
                    if matching_frames:
                        analysis = matching_frames[0]
                        pose_class = analysis.get('pose_class', 'neutral')
                        
                        if pose_class == 'attack' or fencer_states[fencer_id]['consecutive_attack_frames'] > 0:
                            attacking_fencers.append((fencer_id, analysis))
                        else:
                            defending_fencers.append((fencer_id, analysis))
                
                # Check for potential hits
                for attacker_id, attacker_analysis in attacking_fencers:
                    for defender_id, defender_analysis in defending_fencers:
                        # In a real implementation, we would check if the attacker's sword tip
                        # is close to the defender's body using computer vision
                        # For now, we'll use a simplified approach using the bounding boxes
                        
                        if 'box' in attacker_analysis and 'box' in defender_analysis:
                            attacker_box = attacker_analysis['box']
                            defender_box = defender_analysis['box']
                            
                            # Calculate overlap or proximity
                            iou = self.calculate_iou(attacker_box, defender_box)
                            
                            # If boxes overlap and attacker is in attacking pose, consider it a potential hit
                            if iou > 0 and attacker_analysis.get('pose_class') == 'attack':
                                # Create a hit event
                                hit_event = {
                                    'frame_idx': frame_idx,
                                    'attacker_id': attacker_id,
                                    'defender_id': defender_id,
                                    'confidence': iou * 100  # Convert to percentage
                                }
                                hit_events.append(hit_event)
                                
                                # Update technique counts
                                technique_counts['hit'] += 1
                                fencer_technique_counts[attacker_id]['hit'] += 1
        
        # Ensure we're explicitly returning a dictionary and a list
        return dict(technique_counts), dict(fencer_technique_counts), list(hit_events)
    
    def create_visualizations(self, video_name, technique_counts, fencer_technique_counts, hit_events, frame_analyses, processed_frames, output_dir, use_enhanced_detector):
        """
        Create visualizations for the analysis
        
        Args:
            video_name: Name of the video file
            technique_counts: Dictionary of technique counts
            fencer_technique_counts: Dictionary of technique counts per fencer
            hit_events: List of detected hit events
            frame_analyses: List of frame analysis dictionaries
            processed_frames: List of processed frames with pose visualization
            output_dir: Output directory for visualizations
            use_enhanced_detector: Whether enhanced detector was used
        """
        # Debug
        print(f"Type of technique_counts: {type(technique_counts)}")
        print(f"Type of fencer_technique_counts: {type(fencer_technique_counts)}")
        print(f"Type of hit_events: {type(hit_events)}")
        
        # Force technique_counts to be a dictionary
        if not isinstance(technique_counts, dict):
            print(f"Converting technique_counts from {type(technique_counts)} to dict")
            # Create a new dictionary
            technique_counts = {
                'attack': 0, 'defense': 0, 'parry': 0, 'riposte': 0, 
                'lunge': 0, 'advance': 0, 'retreat': 0, 'feint': 0,
                'hit': 0, 'movement': 0
            }
        
        # Force fencer_technique_counts to be a dictionary
        if not isinstance(fencer_technique_counts, dict):
            print(f"Converting fencer_technique_counts from {type(fencer_technique_counts)} to dict")
            # Create a new dictionary
            fencer_technique_counts = {}
            for analysis in frame_analyses:
                fencer_id = analysis.get('fencer_id')
                if fencer_id is not None and fencer_id not in fencer_technique_counts:
                    fencer_technique_counts[fencer_id] = {
                        'attack': 0, 'defense': 0, 'parry': 0, 'riposte': 0, 
                        'lunge': 0, 'advance': 0, 'retreat': 0, 'feint': 0,
                        'hit': 0, 'movement': 0, 
                        'cnn_neutral': 0, 'cnn_attack': 0, 'cnn_defense': 0, 'cnn_lunge': 0
                    }
        
        # Force hit_events to be a list
        if not isinstance(hit_events, list):
            print(f"Converting hit_events from {type(hit_events)} to list")
            hit_events = []
        
        # Define colors for techniques
        technique_to_color = {
            'attack': 'red',
            'defense': 'blue',
            'parry': 'green',
            'riposte': 'purple',
            'lunge': 'orange',
            'advance': 'cyan',
            'retreat': 'brown',
            'feint': 'pink',
            'hit': 'darkred',
            'movement': 'gray'
        }
        
        # 1. Pie chart of technique distribution
        plt.figure(figsize=(10, 10))
        
        # Filter out movement for cleaner visualization
        filtered_counts = {}
        for k, v in technique_counts.items():
            if v > 0 and k != 'movement':
                filtered_counts[k] = v
        
        if filtered_counts:
            plt.pie(
                filtered_counts.values(),
                labels=filtered_counts.keys(),
                autopct='%1.1f%%',
                startangle=90,
                colors=[technique_to_color[t] for t in filtered_counts.keys()]
            )
            plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
            plt.title('Technique Distribution')
        else:
            plt.text(0.5, 0.5, 'No specific techniques detected', 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
        
        # Save the pie chart
        pie_path = os.path.join(output_dir, f"{video_name}_pie.png")
        plt.savefig(pie_path)
        plt.close()
        visualizations = {'pie_chart': pie_path}
        
        # --- BEGIN DEBUG PRINT ---
        print("\n--- Debugging fencer_technique_counts before loops ---")
        print(f"Type of fencer_technique_counts: {type(fencer_technique_counts)}")
        if isinstance(fencer_technique_counts, dict):
            for f_id, f_counts in fencer_technique_counts.items():
                print(f"  Fencer ID: {f_id}, Type of counts: {type(f_counts)}, Value: {f_counts}")
        else:
            print(f"fencer_technique_counts is not a dict! Value: {fencer_technique_counts}")
        print("--- End Debugging ---\n")
        # --- END DEBUG PRINT ---

        # Create per-fencer pie charts
        for fencer_id, counts_data in fencer_technique_counts.items():
            plt.figure(figsize=(8, 8))

            # --- BEGIN ROBUST TRY-EXCEPT ---
            try:
                # Attempt to iterate, assuming dict-like structure
                filtered_counts = {}
                # Try calling .items() and iterate
                for k, v in counts_data.items():
                    if v > 0 and k != 'movement' and not k.startswith('cnn_'):
                        filtered_counts[k] = v

            except AttributeError:
                # Catch error if .items() fails (e.g., it's a tuple)
                print(f"Warning: Expected dictionary-like object for fencer {fencer_id} counts, but got {type(counts_data)}. Skipping pie chart.")
                plt.close() # Close the empty figure
                continue # Skip to next fencer
            except Exception as e:
                 # Catch any other unexpected errors during iteration
                 print(f"Warning: Error processing counts for fencer {fencer_id} pie chart ({type(counts_data)}): {e}. Skipping.")
                 plt.close()
                 continue
            # --- END ROBUST TRY-EXCEPT ---

            # --- Plotting logic using filtered_counts ---
            if filtered_counts:
                plt.pie(
                    filtered_counts.values(),
                    labels=filtered_counts.keys(),
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=[technique_to_color.get(t, 'gray') for t in filtered_counts.keys()] # Use .get for safety
                )
                plt.axis('equal')
                plt.title(f'Fencer {fencer_id} Technique Distribution')
            else:
                plt.text(0.5, 0.5, 'No specific techniques detected',
                        horizontalalignment='center', verticalalignment='center')
                plt.axis('off')

            # Save the fencer pie chart
            fencer_pie_path = os.path.join(output_dir, f"{video_name}_fencer{fencer_id}_pie.png")
            plt.savefig(fencer_pie_path)
            plt.close()
            visualizations[f'fencer{fencer_id}_pie'] = fencer_pie_path
        
        # 2. Bar chart of technique counts
        plt.figure(figsize=(15, 8))
        
        # Exclude movement from bar chart for clarity
        filtered_counts = {}
        for k, v in technique_counts.items():
            if k != 'movement':
                filtered_counts[k] = v
        
        # Sort by count descending
        sorted_techniques = []
        for k, v in filtered_counts.items():
            sorted_techniques.append((k, v))
        
        sorted_techniques.sort(key=lambda x: x[1], reverse=True)
        
        techniques, counts = zip(*sorted_techniques) if sorted_techniques else ([], [])
        
        if techniques:
            bars = plt.bar(techniques, counts, color=[technique_to_color[t] for t in techniques])
            
            # Add count labels on top of bars
            for bar in bars:
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                        f'{height:.0f}',
                        ha='center', va='bottom')
            
            plt.title('Technique Counts')
            plt.ylabel('Count')
            plt.xticks(rotation=45)
        else:
            plt.text(0.5, 0.5, 'No specific techniques detected', 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
        
        # Save the bar chart
        bar_path = os.path.join(output_dir, f"{video_name}_bar.png")
        plt.savefig(bar_path)
        plt.close()
        visualizations['bar_chart'] = bar_path
        
        # Create per-fencer bar charts
        for fencer_id, counts_data in fencer_technique_counts.items():
            plt.figure(figsize=(12, 6))

            # --- BEGIN ROBUST TRY-EXCEPT ---
            try:
                # Attempt to iterate, assuming dict-like structure
                filtered_counts = {}
                # Try calling .items() and iterate
                for k, v in counts_data.items():
                     if k != 'movement' and not k.startswith('cnn_'):
                        filtered_counts[k] = v

            except AttributeError:
                # Catch error if .items() fails (e.g., it's a tuple)
                print(f"Warning: Expected dictionary-like object for fencer {fencer_id} counts, but got {type(counts_data)}. Skipping bar chart.")
                plt.close() # Close the empty figure
                continue # Skip to next fencer
            except Exception as e:
                 # Catch any other unexpected errors during iteration
                 print(f"Warning: Error processing counts for fencer {fencer_id} bar chart ({type(counts_data)}): {e}. Skipping.")
                 plt.close()
                 continue
            # --- END ROBUST TRY-EXCEPT ---

            # --- Processing logic using filtered_counts ---
            # Sort by count descending
            sorted_techniques = []
            # Iterate over the filtered_counts dictionary created in the try block
            for k, v in filtered_counts.items():
                sorted_techniques.append((k, v))

            sorted_techniques.sort(key=lambda x: x[1], reverse=True)

            techniques, technique_counts_values = zip(*sorted_techniques) if sorted_techniques else ([], []) # Renamed variable

            # --- Plotting logic ---
            if techniques:
                # Use technique_counts_values here
                bars = plt.bar(techniques, technique_counts_values, color=[technique_to_color.get(t, 'gray') for t in techniques]) # Use .get

                # Add count labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{height:.0f}',
                            ha='center', va='bottom')

                plt.title(f'Fencer {fencer_id} Technique Counts')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
            else:
                plt.text(0.5, 0.5, 'No specific techniques detected', 
                        horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
            
            # Save the fencer bar chart
            fencer_bar_path = os.path.join(output_dir, f"{video_name}_fencer{fencer_id}_bar.png")
            plt.savefig(fencer_bar_path)
            plt.close()
            visualizations[f'fencer{fencer_id}_bar'] = fencer_bar_path
        
        # 3. Timeline visualization for hit events
        hit_timeline_path = os.path.join(output_dir, f"{video_name}_hit_timeline.png")
        
        if hit_events:
            plt.figure(figsize=(15, 6))
            # Extract data from hit events
            frame_indices = [event['frame_idx'] for event in hit_events]
            attack_fencer_ids = [event['attacker_id'] for event in hit_events]
            confidence_values = [event['confidence'] for event in hit_events]
            
            # Create a scatter plot of hit events
            scatter = plt.scatter(
                frame_indices, 
                confidence_values,
                c=np.array(attack_fencer_ids) % 10,  # Cycle through 10 colors
                cmap='tab10',
                s=100,  # Point size
                alpha=0.7,
                edgecolors='black'
            )
            
            # Add a legend for fencer IDs
            legend1 = plt.legend(*scatter.legend_elements(),
                                loc="upper right", title="Attacker ID")
            plt.gca().add_artist(legend1)
            
            plt.title('Hit Events Timeline')
            plt.xlabel('Frame')
            plt.ylabel('Hit Confidence (%)')
            plt.ylim(0, 100)
            plt.grid(True, alpha=0.3)
            
            # Add annotations for major hits (high confidence)
            for i, event in enumerate(hit_events):
                if event['confidence'] > 50:  # Only annotate high confidence hits
                    plt.annotate(
                        f"{event['attacker_id']} → {event['defender_id']}",
                        (event['frame_idx'], event['confidence']),
                        xytext=(10, 10),
                        textcoords='offset points',
                        arrowprops=dict(arrowstyle='->', color='black')
                    )
            
            plt.tight_layout()
            plt.savefig(hit_timeline_path)
            plt.close()
            visualizations['hit_timeline'] = hit_timeline_path
        
        # 4. Create joint angle visualization
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
        
        # Overall angle visualization
        plt.figure(figsize=(15, 10))
        
        # Plot angles for each fencer with different colors
        for fencer_id, data in angle_data.items():
            frames = data['frames']
                
            if not frames:
                continue
                
            # Get a color for this fencer
            fencer_color = plt.cm.tab10(fencer_id % 10)
                
            # Plot different angles with transparency
            plt.plot(frames, data['right_arm'], 'o-', alpha=0.7, markersize=3, linewidth=1, label=f'Fencer {fencer_id} - Right Arm', color=fencer_color)
                
        plt.title('Right Arm Angle Variation')
        plt.xlabel('Frame')
        plt.ylabel('Angle (degrees)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # Save overall angle visualization
        angles_path = os.path.join(output_dir, f"{video_name}_angles.png")
        plt.savefig(angles_path)
        plt.close()
        visualizations['angles'] = angles_path
        
        # Create per-fencer angle visualizations
        for fencer_id, data in angle_data.items():
            plt.figure(figsize=(15, 10))
            
            frames = data['frames']
            if not frames:
                continue
            
            # Plot multiple angles for this fencer
            plt.plot(frames, data['right_arm'], 'r-', label='Right Arm', linewidth=2)
            plt.plot(frames, data['left_arm'], 'b-', label='Left Arm', linewidth=2)
            plt.plot(frames, data['torso'], 'g-', label='Torso Vertical', linewidth=2)
            plt.plot(frames, data['right_leg'], 'm-', label='Right Leg', linewidth=2)
            plt.plot(frames, data['left_leg'], 'c-', label='Left Leg', linewidth=2)
            
            plt.title(f'Fencer {fencer_id} - Joint Angles')
            plt.xlabel('Frame')
            plt.ylabel('Angle (degrees)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Add annotations for hit events involving this fencer
            for event in hit_events:
                if event['attacker_id'] == fencer_id or event['defender_id'] == fencer_id:
                    frame_idx = event['frame_idx']
                    # Find closest frame in data
                    if frame_idx in frames:
                        idx = frames.index(frame_idx)
                        angle = data['right_arm'][idx]  # Use right arm angle for annotation
                        role = "Attacker" if event['attacker_id'] == fencer_id else "Defender"
                        plt.annotate(
                            f"{role}: vs Fencer {event['defender_id'] if role == 'Attacker' else event['attacker_id']}",
                            (frame_idx, angle),
                            xytext=(5, 5),
                            textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', color='red' if role == 'Attacker' else 'blue')
                        )
            
            # Save per-fencer angle visualization
            fencer_angles_path = os.path.join(output_dir, f"{video_name}_fencer{fencer_id}_angles.png")
            plt.savefig(fencer_angles_path)
            plt.close()
            visualizations[f'fencer{fencer_id}_angles'] = fencer_angles_path
        
        # 5. Create a summary visualization
        plt.figure(figsize=(15, 12))
        
        # Pie chart (top left)
        plt.subplot(2, 2, 1)
        filtered_counts = {}
        for k, v in technique_counts.items():
            if v > 0 and k != 'movement':
                filtered_counts[k] = v
        
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
        
        # Hit events timeline (top right)
        plt.subplot(2, 2, 2)
        if hit_events:
            # Extract data from hit events
            frame_indices = [event['frame_idx'] for event in hit_events]
            attack_fencer_ids = [event['attacker_id'] for event in hit_events]
            confidence_values = [event['confidence'] for event in hit_events]
            
            # Create a scatter plot of hit events
            scatter = plt.scatter(
                frame_indices, 
                confidence_values,
                c=np.array(attack_fencer_ids) % 10,  # Cycle through 10 colors
                cmap='tab10',
                s=50,  # Point size
                alpha=0.7
            )
            
            plt.title('Hit Events')
            plt.xlabel('Frame')
            plt.ylabel('Confidence (%)')
            plt.ylim(0, 100)
        else:
            plt.text(0.5, 0.5, 'No hit events detected', 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
        
        # Sample frames montage (bottom half)
        plt.subplot(2, 1, 2)
        if processed_frames:
            # Create a montage of sample frames
            frame_height, frame_width = processed_frames[0].shape[:2]
            aspect_ratio = frame_width / frame_height
            
            # Determine grid size based on aspect ratio
            num_frames = min(6, len(processed_frames))
            grid_cols = min(3, num_frames)
            grid_rows = (num_frames + grid_cols - 1) // grid_cols
            
            # Create subgrid for frames
            grid = np.ones((grid_rows * frame_height, grid_cols * frame_width, 3), dtype=np.uint8) * 255
            
            # Fill grid with frames
            for i in range(num_frames):
                row = i // grid_cols
                col = i % grid_cols
                frame = processed_frames[i].copy()
                
                # Add hit information if applicable
                frame_idx = i * 15  # Since we sample every 15 frames
                hit_in_frame = [event for event in hit_events if abs(event['frame_idx'] - frame_idx) < 5]
                
                if hit_in_frame:
                    for event in hit_in_frame:
                        text = f"Hit: {event['attacker_id']} → {event['defender_id']}"
                        cv2.putText(frame, text, (50, 50), 
                                  cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
                
                # Add frame to grid
                y_start = row * frame_height
                y_end = y_start + frame_height
                x_start = col * frame_width
                x_end = x_start + frame_width
                
                # Handle different frame sizes
                h, w = frame.shape[:2]
                if h != frame_height or w != frame_width:
                    frame = cv2.resize(frame, (frame_width, frame_height))
                
                grid[y_start:y_end, x_start:x_end] = frame
            
            # Convert from BGR to RGB for matplotlib
            grid_rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
            
            # Display the grid
            plt.imshow(grid_rgb)
            plt.title('Sample Frames with Pose Estimation')
            plt.axis('off')
        else:
            plt.text(0.5, 0.5, 'No sample frames available', 
                    horizontalalignment='center', verticalalignment='center')
            plt.axis('off')
        
        plt.tight_layout()
        
        # Save summary visualization
        summary_path = os.path.join(output_dir, f"{video_name}_summary.png")
        plt.savefig(summary_path)
        plt.close()
        visualizations['summary'] = summary_path
        
        # Create per-fencer summary visualizations
        for fencer_id, counts in fencer_technique_counts.items():
            plt.figure(figsize=(15, 10))
            
            # Calculate detection stats for this fencer
            frame_count = len([1 for analysis in frame_analyses if analysis['fencer_id'] == fencer_id])
            
            # Technique pie chart (top left)
            plt.subplot(2, 2, 1)
            filtered_counts = {k: v for k, v in counts.items() if v > 0 and k != 'movement' and not k.startswith('cnn_')}
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
                plt.text(0.5, 0.5, 'No techniques detected for this fencer', 
                        horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
            
            # Hit events involving this fencer (top right)
            plt.subplot(2, 2, 2)
            fencer_hits = [event for event in hit_events if event['attacker_id'] == fencer_id or event['defender_id'] == fencer_id]
            
            if fencer_hits:
                # Create a timeline showing when this fencer was involved in hits
                frame_indices = [event['frame_idx'] for event in fencer_hits]
                roles = ['Attacker' if event['attacker_id'] == fencer_id else 'Defender' for event in fencer_hits]
                confidences = [event['confidence'] for event in fencer_hits]
                
                # Create a scatter plot with different colors for attacker vs defender
                for role in ['Attacker', 'Defender']:
                    indices = [i for i, r in enumerate(roles) if r == role]
                    if indices:
                        plt.scatter(
                            [frame_indices[i] for i in indices],
                            [confidences[i] for i in indices],
                            label=role,
                            color='red' if role == 'Attacker' else 'blue',
                            s=100,
                            alpha=0.7
                        )
            
                plt.title(f'Fencer {fencer_id} Hit Involvement')
                plt.xlabel('Frame')
                plt.ylabel('Confidence (%)')
                plt.ylim(0, 100)
                plt.legend()
                plt.grid(True, alpha=0.3)
            else:
                plt.text(0.5, 0.5, 'No hit events for this fencer', 
                        horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
            
            # Joint angles (bottom left)
            plt.subplot(2, 2, 3)
            if fencer_id in angle_data and angle_data[fencer_id]['frames']:
                data = angle_data[fencer_id]
                frames = data['frames']
                
                # Plot only right arm and leg for clarity
                plt.plot(frames, data['right_arm'], 'r-', label='Right Arm', linewidth=2)
                plt.plot(frames, data['right_leg'], 'm-', label='Right Leg', linewidth=2)
                
                plt.title(f'Fencer {fencer_id} - Joint Angles')
                plt.xlabel('Frame')
                plt.ylabel('Angle (degrees)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                # Add annotations for hit events
                for event in fencer_hits:
                    frame_idx = event['frame_idx']
                    # Find closest frame in data
                    if frame_idx in frames:
                        idx = frames.index(frame_idx)
                        angle = data['right_arm'][idx]  # Use right arm angle for annotation
                        role = "Attacker" if event['attacker_id'] == fencer_id else "Defender"
                        plt.annotate(
                            f"{role}",
                            (frame_idx, angle),
                            xytext=(5, 5),
                            textcoords='offset points',
                            arrowprops=dict(arrowstyle='->', color='red' if role == 'Attacker' else 'blue')
                        )
            else:
                plt.text(0.5, 0.5, 'No angle data for this fencer', 
                        horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
            
            # CNN classifications (bottom right)
            plt.subplot(2, 2, 4)
            cnn_counts = {k.replace('cnn_', ''): v for k, v in counts.items() if k.startswith('cnn_') and v > 0}
            
            if cnn_counts:
                # Sort by count descending
                sorted_poses = sorted(cnn_counts.items(), key=lambda x: x[1], reverse=True)
                poses, pose_counts = zip(*sorted_poses)
                
                # Define colors for CNN poses
                cnn_colors = {
                    'neutral': 'gray',
                    'attack': 'red',
                    'defense': 'blue',
                    'lunge': 'orange'
                }
                
                bars = plt.bar(poses, pose_counts, color=[cnn_colors.get(p, 'gray') for p in poses])
                
                # Add count labels on top of bars
                for bar in bars:
                    height = bar.get_height()
                    plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                            f'{height:.0f}',
                            ha='center', va='bottom')
                
                plt.title(f'Fencer {fencer_id} CNN Classifications')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
            else:
                plt.text(0.5, 0.5, 'No CNN classifications for this fencer', 
                        horizontalalignment='center', verticalalignment='center')
                plt.axis('off')
            
            plt.tight_layout()
            
            # Save per-fencer summary
            fencer_summary_path = os.path.join(output_dir, f"{video_name}_fencer{fencer_id}_summary.png")
            plt.savefig(fencer_summary_path)
            plt.close()
            visualizations[f'fencer{fencer_id}_summary'] = fencer_summary_path
        
        # Return paths to all visualizations
        print(f"Generated visualizations:")
        for name, path in visualizations.items():
            print(f"  {name}: {path}")
        
        return visualizations
    
    def generate_report(self, results, output_path):
        """
        Generate an HTML report from the analysis results
        
        Args:
            results: Dictionary containing analysis results
            output_path: Path where the HTML report will be saved
        """
        # Create basic HTML report
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Fencing Analysis Report</title>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                h1, h2, h3 { color: #333; }
                .section { margin-bottom: 30px; }
                .fencer-section { margin-bottom: 20px; padding: 10px; border: 1px solid #ccc; }
                .technique-count { margin-right: 20px; }
                img { max-width: 100%; height: auto; margin: 10px 0; }
                table { border-collapse: collapse; width: 100%; }
                th, td { border: 1px solid #ddd; padding: 8px; text-align: left; }
                th { background-color: #f2f2f2; }
                tr:nth-child(even) { background-color: #f9f9f9; }
            </style>
        </head>
        <body>
            <h1>Fencing Video Analysis Report</h1>
        """
        
        # Get video name
        video_name = results.get('video_name', 'Unknown Video')
        
        # Add summary section
        html += f"""
            <div class="section">
            <h2>Analysis Summary for {video_name}</h2>
            <p>Total frames analyzed: {len(results.get('frame_analyses', []))}</p>
        """
        
        # Add technique counts if available
        technique_counts = results.get('technique_counts', {})
        if technique_counts:
            html += "<h3>Overall Technique Distribution</h3><div style='display: flex; flex-wrap: wrap;'>"
            for technique, count in sorted(technique_counts.items(), key=lambda x: x[1], reverse=True):
                if count > 0 and technique != 'movement':
                    html += f"<div class='technique-count'><strong>{technique}:</strong> {count}</div>"
            html += "</div>"
        
        html += "</div>"
        
        # Add per-fencer analysis
        fencer_technique_counts = results.get('fencer_technique_counts', {})
        if fencer_technique_counts:
            html += "<div class='section'><h2>Fencer Analysis</h2>"
            
            for fencer_id, counts in fencer_technique_counts.items():
                html += f"""
                <div class='fencer-section'>
                    <h3>Fencer {fencer_id}</h3>
                    <h4>Technique Distribution</h4>
                    <div style='display: flex; flex-wrap: wrap;'>
        """
        
        # Add technique counts
                for technique, count in sorted([(k, v) for k, v in counts.items() 
                                              if k != 'movement' and not k.startswith('cnn_') and v > 0], 
                                           key=lambda x: x[1], reverse=True):
                    html += f"<div class='technique-count'><strong>{technique}:</strong> {count}</div>"
                
                html += "</div>"
                
                # Add CNN classification counts if available
                cnn_counts = {k.replace('cnn_', ''): v for k, v in counts.items() 
                             if k.startswith('cnn_') and v > 0}
                if cnn_counts:
                    html += "<h4>CNN Classification Counts</h4><div style='display: flex; flex-wrap: wrap;'>"
                    for pose, count in sorted(cnn_counts.items(), key=lambda x: x[1], reverse=True):
                        html += f"<div class='technique-count'><strong>{pose}:</strong> {count}</div>"
                    html += "</div>"
                
                html += "</div>"
            
            html += "</div>"
        
        # Add hit events if available
        hit_events = results.get('hit_events', [])
        if hit_events:
            html += """
            <div class='section'>
                <h2>Hit Events</h2>
                <table>
                    <tr>
                        <th>Frame</th>
                        <th>Attacker</th>
                        <th>Defender</th>
                        <th>Confidence</th>
                    </tr>
            """
            
            for event in sorted(hit_events, key=lambda x: x.get('frame_idx', 0)):
                html += f"""
                <tr>
                    <td>{event.get('frame_idx', 'N/A')}</td>
                    <td>Fencer {event.get('attacker_id', 'N/A')}</td>
                    <td>Fencer {event.get('defender_id', 'N/A')}</td>
                    <td>{event.get('confidence', 0):.1f}%</td>
                </tr>
                """
            
            html += "</table></div>"
        
        # Add pose descriptions if available
        pose_descriptions = results.get('pose_descriptions', [])
        if pose_descriptions:
            html += """
            <div class='section'>
                <h2>Pose Descriptions</h2>
                <table>
                    <tr>
                        <th>Frame</th>
                        <th>Fencer</th>
                        <th>Description</th>
                    </tr>
        """
        
            for i, desc in enumerate(pose_descriptions[:100]):  # Limit to first 100 descriptions
                html += f"""
                <tr>
                    <td>{desc[0]}</td>
                    <td>Fencer {desc[1]}</td>
                    <td>{desc[2]}</td>
                    </tr>
                """
        
            html += "</table></div>"
        
        # Close HTML
        html += """
        </body>
        </html>
        """
        
        # Write HTML to file
        with open(output_path, 'w') as f:
            f.write(html)
        
        print(f"Report generated at {output_path}")

    def calculate_iou(self, box1, box2):
        """
        Calculate the Intersection over Union (IoU) between two bounding boxes
        
        Args:
            box1: First box in format [x1, y1, x2, y2]
            box2: Second box in format [x1, y1, x2, y2]
            
        Returns:
            iou: IoU value
        """
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate area of each box
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        # Check if boxes intersect
        if x1_i >= x2_i or y1_i >= y2_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate IoU
        union = area1 + area2 - intersection
        
        # Handle edge case of zero union
        if union <= 0:
            return 0.0
        
        return intersection / union

    def process_video(self, video_path, output_dir, max_frames=None):
        """
        Process video with fencer detection and pose estimation
        
        Args:
            video_path: Path to video file
            output_dir: Output directory for processed frames
            max_frames: Maximum number of frames to process (None for all)
            
        Returns:
            frame_analyses: List of analyses for each processed frame
            processed_frames: List of processed frames with visualizations
        """
        print(f"Analyzing video: {video_path}")
        
        # Initialize return values
        frame_analyses = []
        processed_frames = []
        
        # Process manual selection if specified
        if hasattr(self, 'manual_fencer_ids') and self.manual_fencer_ids:
            # Set target fencers in the enhanced detector if available
            if self.use_enhanced_detector and hasattr(self.fencer_detector, 'set_target_fencers'):
                self.fencer_detector.set_target_fencers(self.manual_fencer_ids)
                
                # Disable grid lines in visualization
                if hasattr(self.fencer_detector, 'draw_enhanced_detections'):
                    original_draw_method = self.fencer_detector.draw_enhanced_detections
                    def modified_draw_method(frame, tracked_items, sword_detections=None):
                        annotated_frame = frame.copy()
                        
                        # NO GRID LINES - Draw just the boxes and directly use the sword detection output
                        
                        # First, use the model's sword detector to directly visualize swords if available
                        if sword_detections and len(sword_detections) > 0:
                            # Use the optimized sword detector's drawing function directly
                            # This will draw large, visible dots at sword tips with connecting lines
                            annotated_frame = self.fencer_detector.sword_detector.draw_detections(annotated_frame, sword_detections)
                        
                        # Draw fencer detections (boxes and labels only)
                        for item in tracked_items:
                            # Extract information
                            fencer_id = item.get('fencer_id', item.get('id', -1))
                            box = item.get('box', item.get('bbox', None))
                            pose_class = item.get('pose_class', 'neutral')
                            
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
                                
                                # Draw fencer ID and pose information - just minimal info
                                fencer_label = f"Fencer {fencer_id}"
                                pose_label = f"Pose: {pose_class}"
                                
                                # Position labels at top of box
                                cv2.putText(annotated_frame, fencer_label, (x1+5, y1+20),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                                cv2.putText(annotated_frame, pose_label, (x1+5, y1+45),
                                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                        
                        return annotated_frame
                    
                    # Replace the draw method with our modified version
                    self.fencer_detector.draw_enhanced_detections = modified_draw_method
                
        # Check if we should detect and isolate fencers first
        if self.fencer_detector is not None:
            print(f"Running fencer detection and pose classification...")
            
            # Process video with fencer detection first, then do pose analysis per detected fencer
            # Setup video output
            pose_analysis_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_pose_analysis.mp4")
            
            # Initialize video capture
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error opening video file: {video_path}")
                return [], []
            
            # Get video properties
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if max_frames:
                total_frames = min(total_frames, max_frames)
            
            # Initialize video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(pose_analysis_path, fourcc, fps, (frame_width, frame_height))
            
            # Storage for results
            frame_analyses = []
            processed_frames = []
            
            # Process frames
            frame_count = 0
            print(f"Processing video with fencer detection and pose estimation...")
            print(f"Total frames to process: {total_frames}")
            
            # Store initial detections to help with tracking consistency
            fencer_detection_counts = {fencer_id: 0 for fencer_id in self.manual_fencer_ids} if hasattr(self, 'manual_fencer_ids') else {}
            last_known_boxes = {}  # Store last known location of each fencer
            
            while cap.isOpened() and frame_count < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Only process every 3rd frame to improve speed
                if frame_count % 3 == 0:
                    print(f"Processing frame {frame_count}/{total_frames}")
                    
                    # Detect fencers and classify pose using our enhanced detector
                    tracked_items = self.fencer_detector.track_and_classify(frame)
                    
                    # Filter tracked items if manual selection is active
                    if hasattr(self, 'manual_fencer_ids') and self.manual_fencer_ids:
                        tracked_items = [item for item in tracked_items
                                        if item['fencer_id'] in self.manual_fencer_ids]
                    
                    # Create a clean copy of the frame for visualization
                    annotated_frame = frame.copy()
                    
                    # Draw enhanced detections (fencer boxes, pose labels)
                    if self.use_enhanced_detector:
                         # Use the enhanced drawing method (no swords)
                         annotated_frame = self.fencer_detector.draw_enhanced_detections(annotated_frame, tracked_items)
                    
                    # Add frame number to the annotated frame
                    cv2.putText(annotated_frame, f"Frame: {frame_count}", (20, 30), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                    
                    # Process each tracked fencer for pose estimation (using separate estimator)
                    for item in tracked_items:
                        fencer_id = item['fencer_id']
                        box = item['box']
                        pose_class = item.get('pose_class', 'neutral')
                        
                        x1, y1, x2, y2 = map(int, box)
                        if fencer_id in fencer_detection_counts:
                            fencer_detection_counts[fencer_id] += 1
                        last_known_boxes[fencer_id] = box

                        # Estimate detailed pose using PoseEstimator
                        keypoints = self.pose_estimator.estimate_pose(frame, box)

                        if keypoints is not None:
                            # Draw detailed pose on top
                            annotated_frame = self.pose_estimator.draw_pose(annotated_frame, keypoints, confined_to_box=box)
                            angles = self.pose_estimator.calculate_angles(keypoints)
                            observations = self.pose_estimator.analyze_fencing_movement(angles)
                            base_sentence = generate_descriptive_sentence(observations)
                            # Include CNN pose in sentence
                            sentence = f"{base_sentence} CNN classifies as: {pose_class}"

                            frame_analyses.append({
                                'frame_idx': frame_count,
                                'fencer_id': fencer_id,
                                'angles': angles,
                                'observations': observations,
                                'sentence': sentence,
                                'box': [float(x) for x in box],
                                'pose_class': pose_class
                            })
                    
                    # Save for visualization
                    if frame_count % 15 == 0:  # Sample frames for visualization
                        processed_frames.append(annotated_frame)
                    
                    # Write frame to output video
                    writer.write(annotated_frame)
                
                frame_count += 1
            
            # Release resources
            cap.release()
            writer.release()
            
            print(f"Processed {len(frame_analyses)} frames with pose estimation")
            if fencer_detection_counts:
                print("\nFencer detection statistics:")
                for fencer_id, count in fencer_detection_counts.items():
                    print(f"Fencer {fencer_id}: detected in {count} frames")
        else:
            # Fall back to standard pose estimation
            print(f"Fencer detector not available. Running general pose estimation...")
            pose_analysis_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_pose_analysis.mp4")
            
            print(f"Running pose estimation on video...")
            frame_analyses, processed_frames = process_video_with_pose(
                video_path, 
                pose_analysis_path
            )
            
            print(f"Processed {len(frame_analyses)} frames with pose estimation")
        
        # Save sample frames
        if processed_frames:
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
            frames_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_pose_frames.png")
            plt.savefig(frames_path)
            plt.close()
            print(f"Saved sample frames to {frames_path}")
        
        return frame_analyses, processed_frames

def force_dict(obj):
    """Helper function to force an object to be a dictionary"""
    if isinstance(obj, dict):
        return obj
    
    # If it's not a dict, create a new empty one
    print(f"Converting {type(obj)} to dict")
    return {}

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Advanced Fencing Analyzer")
    parser.add_argument("video", help="Path to video file")
    parser.add_argument("--output_dir", help="Output directory", default=None)
    parser.add_argument("--max_frames", type=int, help="Maximum number of frames to process", default=None)
    parser.add_argument("--use_enhanced", action="store_true", help="Use enhanced detector with CNN pose classification")
    parser.add_argument("--pose_model", help="Path to pose classifier model", default=None)
    parser.add_argument("--manual_select", help="Comma-separated list of fencer IDs to manually select", default=None)
    
    args = parser.parse_args()
    
    # Create analyzer
    analyzer = SimplifiedFencingAnalyzer(
        use_enhanced_detector=args.use_enhanced,
        pose_model_path=args.pose_model
    )
    
    # Set manual selection if provided
    if args.manual_select:
        print(f"Manual fencer selection: {args.manual_select}")
        try:
            selected_ids = [int(id.strip()) for id in args.manual_select.split(',')]
            analyzer.manual_fencer_ids = selected_ids
            print(f"Will focus analysis on fencer IDs: {selected_ids}")
        except ValueError:
            print("Warning: Invalid manual selection format. Expected comma-separated integers.")
    
    # Analyze video
    results = analyzer.analyze_video(
        video_path=args.video,
        output_dir=args.output_dir,
        max_frames=args.max_frames
    )
    
    print("Analysis complete!")

if __name__ == "__main__":
    main()