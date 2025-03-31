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
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
from tqdm import tqdm
from dotenv import load_dotenv

load_dotenv()

NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

class FencingKnowledgeGraph:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password=NEO4J_PASSWORD):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.nlp = spacy.load("en_core_web_sm")
        self.technique_embeddings = {}
        self.movement_patterns = {}
        self.technique_classes = [
            'attack', 'defense', 'parry', 'riposte', 'lunge',
            'advance', 'retreat', 'feint'
        ]
        self.technique_patterns = {}
        
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
    
    def get_technique_classes(self):
        return self.technique_classes
    
    def get_technique_pattern(self, technique):
        return self.technique_patterns.get(technique, [])

class StrokeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        # Load pre-trained ResNet3D backbone
        self.backbone = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Replace classification head
        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        
        # Initialize classifier weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights using Xavier initialization"""
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # x shape: [batch, time, height, width, channels]
        # Get backbone features
        outputs = self.backbone(x, output_hidden_states=True)
        features = outputs.hidden_states[-1][:, 0]  # Use [CLS] token features
        
        # Classification
        logits = self.classifier(features)
        return logits

class TechniqueClassifier(nn.Module):
    def __init__(self, knowledge_graph):
        super().__init__()
        self.knowledge_graph = knowledge_graph
        
        # Load pre-trained backbone
        self.backbone = VideoMAEForVideoClassification.from_pretrained(
            "MCG-NJU/videomae-base-finetuned-kinetics"
        )
        
        # Freeze backbone parameters
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # Knowledge-aware attention
        hidden_size = self.backbone.config.hidden_size
        self.attention = nn.MultiheadAttention(embed_dim=hidden_size, num_heads=8)
        
        # Technique classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, len(self.knowledge_graph.get_technique_classes()))
        )
        
        # Initialize new parameters
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize the weights using Xavier initialization"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        # Get backbone features
        outputs = self.backbone(x, output_hidden_states=True)
        features = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
        
        # Apply knowledge-aware attention
        attended_features, _ = self.attention(
            features.transpose(0, 1),
            features.transpose(0, 1),
            features.transpose(0, 1)
        )
        
        # Global pooling
        features = attended_features.transpose(0, 1).mean(dim=1)  # [batch, hidden_size]
        
        # Classify technique
        logits = self.classifier(features)
        return logits

class FencingAnalyzer:
    def __init__(self, video_path=None):
        # Initialize Neo4j connection
        load_dotenv()
        uri = "bolt://localhost:7687"
        user = "neo4j"
        password = os.getenv("NEO4J_PASSWORD")
        
        print("Extracting technique patterns from knowledge graph...")
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        
        # Initialize knowledge graph
        self.knowledge_graph = FencingKnowledgeGraph()
        self.knowledge_graph.extract_technique_patterns()
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize models
        self.stroke_detector = StrokeDetector().to(self.device)
        self.technique_classifier = TechniqueClassifier(self.knowledge_graph).to(self.device)
        
        # Initialize video processor
        self.processor = VideoMAEImageProcessor()
    
    def analyze_video(self, video_path):
        """Analyze a fencing video and detect techniques"""
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
        
        # Process frames using VideoMAE processor
        inputs = self.processor(
            frames,
            return_tensors="pt",
            sampling_rate=4  # Sample every 4th frame
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        print("Detecting strokes...")
        # Detect strokes
        with torch.no_grad():
            stroke_probs = self.stroke_detector(inputs["pixel_values"])
        
        # Get stroke segments
        stroke_segments = self.get_stroke_segments(stroke_probs, fps)
        
        # Classify techniques for each segment
        predictions = []
        print("Classifying techniques...")
        for i, segment in enumerate(stroke_segments):
            start_frame, end_frame = segment
            # Get frames for the segment
            segment_frames = frames[start_frame:end_frame]
            # Process frames
            segment_inputs = self.processor(
                segment_frames,
                return_tensors="pt",
                sampling_rate=4
            )
            # Move to device
            segment_inputs = {k: v.to(self.device) for k, v in segment_inputs.items()}
            # Get predictions
            with torch.no_grad():
                technique_probs = self.technique_classifier(segment_inputs["pixel_values"])
            predictions.append({
                'segment': segment,
                'technique': technique_probs
            })
            print(f"Segment {i+1}: frames {start_frame}-{end_frame}")
        
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
    
    def get_stroke_segments(self, stroke_probs, fps):
        """Convert stroke probabilities to time segments"""
        print("Converting stroke probabilities to segments...")
        
        # Convert to numpy array
        stroke_probs = stroke_probs.cpu().detach().numpy()
        
        # Print probability statistics
        print(f"Probability range: {stroke_probs.min():.3f} - {stroke_probs.max():.3f}")
        print(f"Mean probability: {stroke_probs.mean():.3f}")
        
        # Lower threshold for stroke detection
        threshold = 0.3
        
        # Find segments where probability exceeds threshold
        segments = []
        start_frame = None
        
        for i, prob in enumerate(stroke_probs[0]):
            if prob > threshold and start_frame is None:
                start_frame = i
            elif prob <= threshold and start_frame is not None:
                # Only add segments longer than 0.1 seconds
                if (i - start_frame) / fps >= 0.1:
                    segments.append((start_frame, i))
                    print(f"Found segment: frames {start_frame}-{i} (probability: {prob:.3f})")
                start_frame = None
        
        # Handle case where stroke continues until end
        if start_frame is not None and (len(stroke_probs[0]) - start_frame) / fps >= 0.1:
            segments.append((start_frame, len(stroke_probs[0])))
            print(f"Found final segment: frames {start_frame}-{len(stroke_probs[0])}")
        
        print(f"Found {len(segments)} stroke segments")
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