import os
import torch
import numpy as np
import cv2
from neo4j import GraphDatabase
import fitz
import spacy
import networkx as nx
from torch import nn
import torch.nn.functional as F
from transformers import VideoMAEImageProcessor
from tqdm import tqdm

class FencingKnowledgeGraph:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="your_password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.nlp = spacy.load("en_core_web_sm")
        self.technique_embeddings = {}
        self.movement_patterns = {}
        
    def extract_technique_patterns(self):
        """Extract technique patterns from the knowledge graph"""
        with self.driver.session() as session:
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
                
                if self.is_fencing_technique(technique):
                    self.technique_embeddings[technique] = self.create_technique_embedding(
                        technique, movements, frequency
                    )
                    self.movement_patterns[technique] = self.extract_movement_pattern(
                        technique, movements
                    )
    
    def is_fencing_technique(self, text):
        """Check if the text describes a fencing technique"""
        technique_terms = {
            'attack', 'parry', 'riposte', 'lunge', 'thrust',
            'cut', 'disengage', 'feint', 'bind', 'beat'
        }
        return any(term in text.lower() for term in technique_terms)
    
    def create_technique_embedding(self, technique, movements, frequency):
        """Create an embedding for a fencing technique"""
        # Use spaCy to create embeddings for the technique and its movements
        technique_doc = self.nlp(technique)
        movement_docs = [self.nlp(m) for m in movements]
        
        # Combine embeddings
        technique_vector = technique_doc.vector
        movement_vectors = np.mean([doc.vector for doc in movement_docs], axis=0)
        
        # Weight by frequency
        combined_vector = (technique_vector + movement_vectors) * np.log1p(frequency)
        return torch.FloatTensor(combined_vector)
    
    def extract_movement_pattern(self, technique, movements):
        """Extract expected movement patterns for a technique"""
        pattern = {
            'body_position': [],
            'blade_position': [],
            'movement_sequence': []
        }
        
        # Extract positions and movements from text
        for movement in movements:
            doc = self.nlp(movement)
            for token in doc:
                if token.pos_ == 'VERB':
                    pattern['movement_sequence'].append(token.text)
                elif token.pos_ == 'NOUN':
                    if any(term in token.text.lower() for term in ['arm', 'leg', 'foot', 'body']):
                        pattern['body_position'].append(token.text)
                    elif any(term in token.text.lower() for term in ['blade', 'sword', 'point']):
                        pattern['blade_position'].append(token.text)
        
        return pattern

class StrokeDetector(nn.Module):
    def __init__(self, knowledge_graph):
        super().__init__()
        self.knowledge_graph = knowledge_graph
        
        # Convolutional layers for motion detection
        self.motion_conv = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(128)
        )
        
        # Spatial feature extraction
        self.spatial_features = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256)
        )
        
        # Knowledge integration
        embedding_dim = 300  # spaCy embedding dimension
        self.knowledge_projection = nn.Linear(embedding_dim, 256)
        
        # Stroke detection head
        self.stroke_detector = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Input shape: [batch, time, channels, height, width]
        batch_size = x.size(0)
        
        # Motion feature extraction
        motion_features = self.motion_conv(x.transpose(1, 2))  # [batch, 128, time, height, width]
        
        # Get features for each timestep
        timesteps = motion_features.size(2)
        stroke_probs = []
        
        for t in range(timesteps):
            # Spatial features for current timestep
            spatial_feat = self.spatial_features(motion_features[:, :, t])  # [batch, 256, height, width]
            
            # Global average pooling
            spatial_feat = spatial_feat.mean(dim=[2, 3])  # [batch, 256]
            
            # Detect stroke at current timestep
            stroke_prob = self.stroke_detector(spatial_feat)
            stroke_probs.append(stroke_prob)
        
        return torch.stack(stroke_probs, dim=1)  # [batch, time, 1]

class TechniqueClassifier(nn.Module):
    def __init__(self, knowledge_graph):
        super().__init__()
        self.knowledge_graph = knowledge_graph
        
        # Create technique embeddings matrix
        self.num_techniques = len(knowledge_graph.technique_embeddings)
        self.technique_list = list(knowledge_graph.technique_embeddings.keys())
        
        technique_embeds = []
        for technique in self.technique_list:
            technique_embeds.append(knowledge_graph.technique_embeddings[technique])
        self.technique_embeds = torch.stack(technique_embeds)  # [num_techniques, embed_dim]
        
        # Feature extraction for stroke segments
        self.feature_extractor = nn.Sequential(
            nn.Conv3d(3, 64, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(64),
            nn.Conv3d(64, 128, kernel_size=(3, 3, 3), padding=(1, 1, 1)),
            nn.ReLU(),
            nn.BatchNorm3d(128),
            nn.AdaptiveAvgPool3d((1, 1, 1))
        )
        
        # Project video features to embedding space
        self.projection = nn.Linear(128, self.technique_embeds.size(1))
    
    def forward(self, x, stroke_segments):
        """
        x: Video tensor [batch, time, channels, height, width]
        stroke_segments: List of (start, end) indices for detected strokes
        """
        batch_size = x.size(0)
        classifications = []
        
        for b in range(batch_size):
            segment_preds = []
            for start, end in stroke_segments[b]:
                # Extract features for the stroke segment
                segment = x[b:b+1, start:end+1]  # [1, segment_length, C, H, W]
                features = self.feature_extractor(segment.transpose(1, 2))  # [1, 128, 1, 1, 1]
                features = features.squeeze()  # [128]
                
                # Project to embedding space
                projected = self.projection(features)  # [embed_dim]
                
                # Compare with technique embeddings
                similarities = F.cosine_similarity(
                    projected.unsqueeze(0),
                    self.technique_embeds,
                    dim=1
                )
                
                # Get top techniques
                segment_preds.append(similarities)
            
            if segment_preds:
                # Aggregate predictions for all segments
                segment_preds = torch.stack(segment_preds)
                classifications.append(segment_preds.mean(dim=0))
            else:
                # No strokes detected
                classifications.append(torch.zeros(self.num_techniques))
        
        return torch.stack(classifications)  # [batch, num_techniques]

class FencingAnalyzer:
    def __init__(self):
        # Initialize knowledge graph
        self.knowledge_graph = FencingKnowledgeGraph()
        print("Extracting technique patterns from knowledge graph...")
        self.knowledge_graph.extract_technique_patterns()
        
        # Initialize models
        self.stroke_detector = StrokeDetector(self.knowledge_graph)
        self.technique_classifier = TechniqueClassifier(self.knowledge_graph)
        
        # Initialize video processor
        self.processor = VideoMAEImageProcessor()
    
    def analyze_video(self, video_path, confidence_threshold=0.5):
        """Analyze a fencing video using the knowledge graph"""
        # Load video frames
        frames = self.load_video(video_path)
        frame_tensor = torch.FloatTensor(frames).unsqueeze(0)  # Add batch dimension
        
        # Detect strokes
        stroke_probs = self.stroke_detector(frame_tensor)
        stroke_segments = self.get_stroke_segments(stroke_probs[0], confidence_threshold)
        
        # Classify detected strokes
        if stroke_segments:
            technique_probs = self.technique_classifier(frame_tensor, [stroke_segments])
            predictions = self.get_predictions(technique_probs[0])
        else:
            predictions = []
        
        return predictions, stroke_segments
    
    def load_video(self, video_path, target_size=(224, 224)):
        """Load and preprocess video frames"""
        cap = cv2.VideoCapture(video_path)
        frames = []
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame = cv2.resize(frame, target_size)
            frames.append(frame)
        
        cap.release()
        return np.array(frames) / 255.0
    
    def get_stroke_segments(self, stroke_probs, threshold):
        """Get start and end frames of detected strokes"""
        stroke_probs = stroke_probs.squeeze().cpu().numpy()
        segments = []
        in_stroke = False
        start_frame = 0
        
        for i, prob in enumerate(stroke_probs):
            if prob > threshold and not in_stroke:
                in_stroke = True
                start_frame = i
            elif prob <= threshold and in_stroke:
                in_stroke = False
                segments.append((start_frame, i-1))
        
        if in_stroke:
            segments.append((start_frame, len(stroke_probs)-1))
        
        return segments
    
    def get_predictions(self, technique_probs):
        """Get predicted techniques with probabilities"""
        probs = F.softmax(technique_probs, dim=0).cpu().numpy()
        predictions = []
        
        for i, prob in enumerate(probs):
            technique = self.technique_classifier.technique_list[i]
            predictions.append((technique, prob))
        
        return sorted(predictions, key=lambda x: x[1], reverse=True)

if __name__ == "__main__":
    # Example usage
    analyzer = FencingAnalyzer()
    video_path = "fencing_demo_video.mp4"
    
    print(f"Analyzing video: {video_path}")
    predictions, stroke_segments = analyzer.analyze_video(video_path)
    
    print("\nDetected strokes:")
    for start, end in stroke_segments:
        print(f"Stroke from frame {start} to {end}")
    
    print("\nTop predicted techniques:")
    for technique, prob in predictions[:5]:
        print(f"{technique}: {prob:.4f}") 