import os
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from neo4j import GraphDatabase
from dotenv import load_dotenv
import torchvision.transforms as transforms
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

# Import pose estimation helper if available
try:
    from pose_estimation_helper import process_video_with_pose, extract_pose_descriptions, enhance_videomae_with_pose
    POSE_HELPER_AVAILABLE = True
    print("Pose estimation helper is available for enhanced fencing analysis")
except ImportError:
    POSE_HELPER_AVAILABLE = False
    print("Pose estimation helper not found. Installing required dependencies:")
    print("pip install mediapipe matplotlib")

# Safe import for VideoMAE
try:
    from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
    VIDEOMAE_AVAILABLE = True
except ImportError:
    print("Warning: transformers library not available or incompatible. Falling back to basic CNN.")
    VIDEOMAE_AVAILABLE = False

load_dotenv()

NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class FencingKnowledgeGraph:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password=NEO4J_PASSWORD):
        self.technique_classes = [
            'attack', 'defense', 'parry', 'riposte', 'lunge',
            'advance', 'retreat', 'feint'
        ]
        self.technique_patterns = {
            'attack': ['forward motion', 'blade extension'],
            'defense': ['backward motion', 'blade block'],
            'parry': ['blade deflection', 'lateral motion'],
            'riposte': ['parry', 'counter attack'],
            'lunge': ['forward step', 'blade extension'],
            'advance': ['forward step'],
            'retreat': ['backward step'],
            'feint': ['fake attack', 'deception']
        }
        
        # Try to connect to Neo4j if available
        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.neo4j_available = True
            print("Successfully connected to Neo4j database")
        except Exception as e:
            print(f"Neo4j connection failed: {str(e)}")
            print("Using default fencing patterns instead")
            self.neo4j_available = False
        
    def extract_technique_patterns(self):
        """Extract technique patterns from the knowledge graph"""
        if not self.neo4j_available:
            print("Using default technique patterns")
            return
            
        try:
            with self.driver.session() as session:
                # Check if there are any entities in the database
                result = session.run("MATCH (n:Entity) RETURN count(n) as count")
                entity_count = result.single()["count"]
                print(f"Found {entity_count} entities in the knowledge graph")
                
                if entity_count > 0:
                    # Query for techniques and their associated movements/positions
                    result = session.run("""
                        MATCH (t:Entity)-[r:RELATED_TO]->(m:Entity)
                        WHERE EXISTS((t)-[:MENTIONED_IN]->(:Book))
                        RETURN t.name as technique, collect(m.name) as movements,
                               count(DISTINCT r) as frequency
                    """)
                    
                    for record in result:
                        technique = record["technique"]
                        movements = record["movements"]
                        frequency = record["frequency"]
                        print(f"Found technique: {technique} with {len(movements)} movements")
        except Exception as e:
            print(f"Error querying Neo4j: {str(e)}")
            print("Using default fencing patterns")
        
    def get_technique_classes(self):
        return self.technique_classes
        
    def close(self):
        if hasattr(self, 'driver') and self.neo4j_available:
            self.driver.close()

# Fallback CNN in case VideoMAE isn't available
class SimpleFencingCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        self.classifier = nn.Sequential(
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(512, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class FencingAnalyzer:
    def __init__(self):
        global VIDEOMAE_AVAILABLE
        print("Initializing FencingAnalyzer...")
        # Initialize knowledge graph
        self.knowledge_graph = FencingKnowledgeGraph()
        print("Extracting technique patterns from knowledge graph...")
        self.knowledge_graph.extract_technique_patterns()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Get technique classes
        num_classes = len(self.knowledge_graph.get_technique_classes())
        
        # Initialize model - try VideoMAE first, fallback to SimpleCNN
        if VIDEOMAE_AVAILABLE:
            print("Using VideoMAE model for classification")
            try:
                # Create label mapping for VideoMAE
                labels = self.knowledge_graph.get_technique_classes()
                self.id2label = {i: label for i, label in enumerate(labels)}
                self.label2id = {label: i for i, label in self.id2label.items()}
                
                # Load VideoMAE model
                model_ckpt = "MCG-NJU/videomae-base-finetuned-kinetics"
                self.processor = VideoMAEImageProcessor.from_pretrained(model_ckpt)
                self.model = VideoMAEForVideoClassification.from_pretrained(
                    model_ckpt,
                    num_labels=num_classes,
                    label2id=self.label2id,
                    id2label=self.id2label,
                    ignore_mismatched_sizes=True
                )
                
                # Get model parameters
                self.num_frames = self.model.config.num_frames
                self.image_size = (
                    self.processor.size["height"] 
                    if "height" in self.processor.size 
                    else self.processor.size["shortest_edge"]
                )
                self.mean = self.processor.image_mean
                self.std = self.processor.image_std
                print(f"VideoMAE model initialized: {self.num_frames} frames, {self.image_size}x{self.image_size} resolution")
            except Exception as e:
                print(f"Error initializing VideoMAE: {str(e)}")
                print("Falling back to basic CNN")
                VIDEOMAE_AVAILABLE = False
                
        if not VIDEOMAE_AVAILABLE:
            print("Using simple CNN model for classification")
            self.model = SimpleFencingCNN(num_classes).to(self.device)
            
        self.model = self.model.to(self.device)
        print(f"Model initialized with {num_classes} classes")
    
    def analyze_video(self, video_path):
        """Analyze a fencing video"""
        print(f"Analyzing video: {video_path}")
        
        # Load video
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError("Could not open video file")
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video properties: {frame_count} frames, {fps} fps, {width}x{height}")
        
        # First, perform pose estimation if available
        pose_probs = None
        pose_descriptions = None
        technique_counts = {}
        detected_techniques = set()
        
        # Try to perform pose estimation
        try:
            if POSE_HELPER_AVAILABLE:
                try:
                    print("Performing enhanced analysis with pose estimation first...")
                    # Process video with pose estimation
                    print("Starting pose analysis of video...")
                    frame_analyses, _ = process_video_with_pose(video_path, save_frames=False)
                    pose_descriptions = extract_pose_descriptions(frame_analyses)
                    
                    # Extract the main fencing techniques detected by pose analysis
                    for desc in pose_descriptions:
                        desc_lower = desc.lower()
                        if "lunge" in desc_lower:
                            detected_techniques.add("lunge")
                        if "advance" in desc_lower:
                            detected_techniques.add("advance")
                        if "attack" in desc_lower:
                            detected_techniques.add("attack")
                        if "retreat" in desc_lower:
                            detected_techniques.add("retreat")
                        if "parry" in desc_lower:
                            detected_techniques.add("parry")
                    
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
                    
                    # Print the techniques detected by pose analysis
                    if detected_techniques:
                        print(f"Primary techniques detected by pose analysis: {', '.join(detected_techniques)}")
                        for technique, count in technique_counts.items():
                            if count > 0:
                                print(f"  - {technique}: {count} occurrences")
                    
                except Exception as e:
                    print(f"Error performing pose analysis: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    print("Continuing with basic VideoMAE analysis only")
        except Exception as e:
            print(f"Import error with pose estimation: {str(e)}")
            print("Continuing without pose estimation")
        
        # Extract frames
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # Convert to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        cap.release()
        
        print(f"Extracted {len(frames)} frames")
        
        # Process video with VideoMAE, informed by pose analysis if available
        videomae_probs = None
        try:
            if VIDEOMAE_AVAILABLE:
                # Use pose information to guide VideoMAE processing if available
                if pose_descriptions:
                    print("Using pose information to guide VideoMAE analysis...")
                    videomae_probs = self._analyze_with_videomae_and_pose(frames, pose_descriptions, detected_techniques)
                else:
                    videomae_probs = self._analyze_with_videomae(frames)
            else:
                videomae_probs = self._analyze_with_simplecnn(frames)
        except Exception as e:
            print(f"Error during video classification: {str(e)}")
            import traceback
            traceback.print_exc()
            # Fall back to simple CNN
            try:
                print("Falling back to simple CNN classification...")
                videomae_probs = self._analyze_with_simplecnn(frames)
            except Exception as e2:
                print(f"Error during simple CNN classification: {str(e2)}")
                videomae_probs = None
        
        if videomae_probs is None:
            print("WARNING: Classification failed. Returning uniform probabilities.")
            # Return uniform probabilities as fallback
            techniques = self.knowledge_graph.get_technique_classes()
            videomae_probs = np.ones(len(techniques)) / len(techniques)
        
        # Compute pose-based probabilities if pose analysis was performed
        if technique_counts:
            try:
                # Normalize counts to get probabilities
                total = sum(technique_counts.values()) + 1e-6  # Avoid division by zero
                pose_probs = {k: v/total for k, v in technique_counts.items()}
                
                print("\nProbabilities based on pose analysis:")
                techniques = self.knowledge_graph.get_technique_classes()
                for technique in techniques:
                    if technique in pose_probs:
                        print(f"  {technique}: {pose_probs.get(technique, 0.0):.4f}")
            except Exception as e:
                print(f"Error computing pose probabilities: {str(e)}")
                pose_probs = None
        
        # Now combine the results, using knowledge graph to refine
        final_scores = videomae_probs
        if pose_probs:
            try:
                print("\nRefining results using knowledge graph...")
                
                # Get the technique patterns from the knowledge graph
                technique_patterns = {}
                for technique in self.knowledge_graph.get_technique_classes():
                    technique_patterns[technique] = self.knowledge_graph.technique_patterns.get(technique, [])
                
                # Determine confidence scores from knowledge graph
                knowledge_confidence = self._evaluate_consistency_with_knowledge_graph(
                    videomae_probs, 
                    pose_probs, 
                    technique_patterns, 
                    detected_techniques
                )
                
                # Compute final scores with knowledge graph weighting
                techniques = self.knowledge_graph.get_technique_classes()
                final_scores = np.zeros(len(techniques))
                
                for i, technique in enumerate(techniques):
                    # Get VideoMAE probability
                    videomae_prob = videomae_probs[i]
                    # Get pose probability
                    pose_prob = pose_probs.get(technique, 0.0)
                    # Get knowledge graph confidence
                    kg_confidence = knowledge_confidence.get(technique, 0.5)
                    
                    # Weighted combination based on knowledge graph confidence
                    # Higher confidence in pose estimation gets more weight for pose predictions
                    if technique in detected_techniques:
                        final_scores[i] = 0.3 * videomae_prob + 0.7 * pose_prob
                    else:
                        final_scores[i] = 0.7 * videomae_prob + 0.3 * pose_prob
                
                # Re-normalize
                final_scores = final_scores / np.sum(final_scores)
                
                # Print final knowledge graph confidence
                print("\nKnowledge graph confidence scores:")
                for technique, conf in knowledge_confidence.items():
                    print(f"  {technique}: {conf:.2f}")
                
            except Exception as e:
                print(f"Error refining results with knowledge graph: {str(e)}")
                final_scores = videomae_probs
        
        techniques = self.knowledge_graph.get_technique_classes()
        return final_scores, techniques
    
    def _analyze_with_videomae(self, frames):
        """Analyze video with VideoMAE model"""
        try:
            # Process frames using VideoMAE's processor
            print("Processing frames with VideoMAE...")
            
            # Uniform sampling to get num_frames
            frame_indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            sampled_frames = [frames[i] for i in frame_indices]
            
            # Convert to PIL for processor
            pil_frames = [Image.fromarray(frame) for frame in sampled_frames]
            
            # Process frames
            inputs = self.processor(
                images=pil_frames,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            print("Running inference with VideoMAE...")
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
            # Get probabilities
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            return probs
            
        except Exception as e:
            print(f"Error during VideoMAE analysis: {str(e)}")
            print("Falling back to simple CNN")
            return self._analyze_with_simplecnn(frames)
    
    def _analyze_with_simplecnn(self, frames):
        """Analyze video with simple CNN"""
        print("Analyzing with simple CNN...")
        # Resize frames
        resized_frames = [cv2.resize(frame, (224, 224)) for frame in frames]
        
        # Analyze frames at regular intervals
        frame_step = max(1, len(frames) // 10)  # Analyze about 10 frames
        results = []
        
        for i in range(0, len(frames), frame_step):
            frame = resized_frames[i]
            frame_tensor = torch.from_numpy(frame.transpose(2, 0, 1)).float().unsqueeze(0) / 255.0
            frame_tensor = frame_tensor.to(self.device)
            
            # Get predictions
            with torch.no_grad():
                output = self.model(frame_tensor)
                probabilities = F.softmax(output, dim=1)[0]
            
            # Store results
            frame_result = {
                'frame_idx': i,
                'probabilities': probabilities.cpu().numpy()
            }
            results.append(frame_result)
        
        # Rule-based analysis using knowledge patterns
        techniques = self.knowledge_graph.get_technique_classes()
        final_scores = np.zeros(len(techniques))
        
        # Simple algorithm: use motion detection to weight predictions
        for i in range(1, len(results)):
            prev_frame = resized_frames[results[i-1]['frame_idx']]
            curr_frame = resized_frames[results[i]['frame_idx']]
            
            # Simple motion detection
            motion = np.mean(np.abs(curr_frame.astype(np.float32) - prev_frame.astype(np.float32)))
            motion_weight = min(1.0, motion / 30.0)  # Normalize motion
            
            # Weight current prediction by motion
            weighted_probs = results[i]['probabilities'] * motion_weight
            final_scores += weighted_probs
        
        # Normalize final scores
        if np.sum(final_scores) > 0:
            final_scores = final_scores / np.sum(final_scores)
        
        return final_scores

    def _analyze_with_videomae_and_pose(self, frames, pose_descriptions, detected_techniques):
        """
        Analyze video with VideoMAE model, guided by pose information
        """
        try:
            print("Processing frames with VideoMAE (guided by pose)...")
            
            # Uniform sampling to get num_frames
            frame_indices = np.linspace(0, len(frames) - 1, self.num_frames, dtype=int)
            sampled_frames = [frames[i] for i in frame_indices]
            
            # Convert to PIL for processor
            pil_frames = [Image.fromarray(frame) for frame in sampled_frames]
            
            # Process frames
            inputs = self.processor(
                images=pil_frames,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            
            # Get predictions
            print("Running inference with VideoMAE...")
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                
            # Get initial probabilities
            probs = F.softmax(logits, dim=1).cpu().numpy()[0]
            
            # Adjust based on detected techniques from pose
            if detected_techniques:
                print("Adjusting VideoMAE predictions based on pose-detected techniques...")
                techniques = self.knowledge_graph.get_technique_classes()
                adjusted_probs = probs.copy()
                
                # Boost probabilities for detected techniques
                for i, technique in enumerate(techniques):
                    if technique in detected_techniques:
                        # Boost probability
                        adjusted_probs[i] *= 1.2
                
                # Re-normalize
                adjusted_probs = adjusted_probs / np.sum(adjusted_probs)
                return adjusted_probs
            
            return probs
            
        except Exception as e:
            print(f"Error during VideoMAE analysis with pose guidance: {str(e)}")
            print("Falling back to standard VideoMAE analysis")
            return self._analyze_with_videomae(frames)

    def _evaluate_consistency_with_knowledge_graph(self, videomae_probs, pose_probs, technique_patterns, detected_techniques):
        """
        Evaluate consistency between VideoMAE predictions, pose predictions, and knowledge graph
        Returns confidence scores for each technique
        """
        techniques = self.knowledge_graph.get_technique_classes()
        confidence = {}
        
        # Compare top VideoMAE predictions with detected pose techniques
        videomae_rankings = np.argsort(-videomae_probs)
        videomae_top_techniques = [techniques[i] for i in videomae_rankings[:3]]
        
        # Calculate confidence based on agreement between different sources
        for technique in techniques:
            # Base confidence
            conf = 0.5
            
            # If technique is in both top VideoMAE predictions and detected by pose
            if technique in videomae_top_techniques and technique in detected_techniques:
                conf += 0.3
            
            # If technique is only in one of them, smaller boost
            elif technique in videomae_top_techniques or technique in detected_techniques:
                conf += 0.1
                
            # Check if the technique patterns from knowledge graph are present in pose descriptions
            pattern_words = ' '.join(technique_patterns.get(technique, []))
            pattern_match_score = 0.0
            
            if pattern_words:
                # Check if pattern words are in pose descriptions
                for word in pattern_words.split():
                    for desc in detected_techniques:
                        if word.lower() in desc.lower():
                            pattern_match_score += 0.1
                            break
                
                conf += min(0.2, pattern_match_score)  # Cap at 0.2
            
            confidence[technique] = min(1.0, conf)  # Cap at 1.0
        
        return confidence

if __name__ == "__main__":
    try:
        print("Starting fencing video analysis...")
        # Get video filename from command line argument
        import sys
        if len(sys.argv) > 1:
            video_path = sys.argv[1]
        else:
            video_path = "fencing_demo_video.mp4"  # Default
            
        print(f"Using video: {video_path}")
        
        # Initialize analyzer
        analyzer = FencingAnalyzer()
        
        print(f"Checking if video exists: {os.path.exists(video_path)}")
        if not os.path.exists(video_path):
            print(f"ERROR: Video file {video_path} not found!")
            exit(1)
        
        # Analyze video
        print("\n=== Analyzing Fencing Video ===")
        scores, techniques = analyzer.analyze_video(video_path)
        
        # Print results
        print("\n=== Fencing Analysis Results ===")
        print("Technique scores:")
        for i, technique in enumerate(techniques):
            if scores is not None:
                print(f"{technique}: {scores[i]:.4f}")
            else:
                print(f"{technique}: N/A (analysis failed)")
        
        # Print top predicted technique
        if scores is not None:
            top_idx = np.argmax(scores)
            print(f"\nTop predicted technique: {techniques[top_idx]} ({scores[top_idx]:.4f})")
        else:
            print("\nNo technique predicted (analysis failed)")
        
        # Create visualization
        try:
            plt.figure(figsize=(10, 6))
            plt.bar(techniques, scores)
            plt.xlabel('Technique')
            plt.ylabel('Score')
            plt.title(f'Fencing Technique Analysis - {os.path.basename(video_path)}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            # Save visualization
            output_file = f"fencing_analysis_{os.path.splitext(os.path.basename(video_path))[0]}.png"
            plt.savefig(output_file)
            print(f"\nVisualization saved to {output_file}")
            
            # Display a few frames from the video with the prediction
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                # Extract frames at regular intervals for visualization
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                intervals = 4  # Number of frames to extract
                frames = []
                
                for i in range(intervals):
                    # Set position to evenly spaced frames
                    position = int(i * frame_count / intervals)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, position)
                    ret, frame = cap.read()
                    if ret:
                        frames.append(frame)
                cap.release()
                
                if frames:
                    # Create a grid of frames
                    plt.figure(figsize=(12, 8))
                    for i, frame in enumerate(frames):
                        plt.subplot(2, 2, i+1)
                        plt.imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                        plt.axis('off')
                        plt.title(f"Frame {i+1}")
                    
                    plt.suptitle(f"Video: {os.path.basename(video_path)}\nTop prediction: {techniques[top_idx]} ({scores[top_idx]:.4f})")
                    plt.tight_layout()
                    
                    # Save visualization
                    output_file_frames = f"frames_{os.path.splitext(os.path.basename(video_path))[0]}.png"
                    plt.savefig(output_file_frames)
                    print(f"Frame visualization saved to {output_file_frames}")
                    
                    # If pose analysis is available, run it on the video to create a visualization
                    if POSE_HELPER_AVAILABLE:
                        try:
                            print("Creating pose analysis video...")
                            pose_output = f"pose_analysis_{os.path.splitext(os.path.basename(video_path))[0]}.mp4"
                            process_video_with_pose(video_path, pose_output, save_frames=True)
                            print(f"Pose analysis video saved to {pose_output}")
                        except Exception as e:
                            print(f"Error creating pose analysis video: {str(e)}")
                    
        except Exception as e:
            print(f"Could not create visualization: {str(e)}")
            
    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc() 