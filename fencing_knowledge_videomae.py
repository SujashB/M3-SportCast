import os
import torch
import numpy as np
from transformers import VideoMAEForVideoClassification, VideoMAEImageProcessor
from neo4j import GraphDatabase
import fitz  # PyMuPDF for PDF processing
import spacy
from tqdm import tqdm
import cv2
from torch import nn
import networkx as nx

class FencingKnowledgeExtractor:
    def __init__(self, uri="bolt://localhost:7687", user="neo4j", password="your_password"):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        self.nlp = spacy.load("en_core_web_sm")
        
    def close(self):
        self.driver.close()
        
    def clear_database(self):
        with self.driver.session() as session:
            session.run("MATCH (n) DETACH DELETE n")
    
    def extract_text_from_pdf(self, pdf_path):
        """Extract text from PDF file"""
        doc = fitz.open(pdf_path)
        text = ""
        for page in doc:
            text += page.get_text()
        return text
    
    def create_knowledge_nodes(self, text, book_title):
        """Create knowledge nodes from text"""
        doc = self.nlp(text)
        
        # Extract fencing-specific entities and relationships
        with self.driver.session() as session:
            # Create book node
            session.run(
                "CREATE (b:Book {title: $title})",
                title=book_title
            )
            
            # Process sentences and create relationships
            for sent in doc.sents:
                # Extract fencing techniques, movements, and concepts
                entities = []
                for ent in sent.ents:
                    if self.is_fencing_related(ent.text):
                        entities.append(ent)
                
                # Create nodes and relationships
                for i, ent in enumerate(entities):
                    # Create or merge entity node
                    session.run(
                        """
                        MERGE (e:Entity {name: $name})
                        ON CREATE SET e.type = $type
                        WITH e
                        MATCH (b:Book {title: $book})
                        MERGE (e)-[:MENTIONED_IN]->(b)
                        """,
                        name=ent.text,
                        type=ent.label_,
                        book=book_title
                    )
                    
                    # Create relationships between consecutive entities
                    if i > 0:
                        session.run(
                            """
                            MATCH (e1:Entity {name: $name1})
                            MATCH (e2:Entity {name: $name2})
                            MERGE (e1)-[:RELATED_TO]->(e2)
                            """,
                            name1=entities[i-1].text,
                            name2=ent.text
                        )
    
    def is_fencing_related(self, text):
        """Check if text is related to fencing"""
        fencing_terms = {
            'attack', 'defense', 'parry', 'riposte', 'lunge', 'advance',
            'retreat', 'blade', 'point', 'guard', 'stance', 'footwork',
            'distance', 'tempo', 'bout', 'target', 'line', 'engagement'
        }
        return any(term in text.lower() for term in fencing_terms)
    
    def get_knowledge_graph(self):
        """Return the knowledge graph as a NetworkX object"""
        G = nx.DiGraph()
        
        with self.driver.session() as session:
            # Get all nodes
            result = session.run("MATCH (n) RETURN n")
            for record in result:
                node = record['n']
                G.add_node(node.id, **dict(node))
            
            # Get all relationships
            result = session.run("MATCH (n)-[r]->(m) RETURN n, r, m")
            for record in result:
                G.add_edge(
                    record['n'].id,
                    record['m'].id,
                    type=type(record['r']).__name__
                )
        
        return G

class KnowledgeEnhancedVideoMAE(nn.Module):
    def __init__(self, base_model, knowledge_graph, num_classes):
        super().__init__()
        self.base_model = base_model
        self.knowledge_graph = knowledge_graph
        
        # Knowledge graph embedding
        num_nodes = len(knowledge_graph.nodes())
        self.node_embeddings = nn.Embedding(num_nodes, 256)
        
        # Additional layers for knowledge integration
        self.knowledge_attention = nn.MultiheadAttention(
            embed_dim=768,  # VideoMAE hidden size
            num_heads=8,
            dropout=0.1
        )
        
        # Spatial feature modeling (3D ConvNet as shown in the image)
        self.spatial_modeling = nn.Sequential(
            nn.Conv3d(768, 196, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(196)
        )
        
        # Temporal feature modeling (Transformer as shown in the image)
        self.temporal_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=196,
                nhead=8,
                dim_feedforward=2048
            ),
            num_layers=2
        )
        
        # Final classifier
        self.classifier = nn.Linear(196, num_classes)
    
    def forward(self, pixel_values):
        # Get base VideoMAE features
        outputs = self.base_model(pixel_values, output_hidden_states=True)
        hidden_states = outputs.hidden_states[-1]  # [batch, seq_len, hidden_size]
        
        # Get knowledge graph embeddings
        node_embeds = self.node_embeddings(
            torch.arange(len(self.knowledge_graph.nodes())).to(pixel_values.device)
        )
        
        # Apply knowledge attention
        enhanced_features, _ = self.knowledge_attention(
            hidden_states.transpose(0, 1),
            node_embeds.unsqueeze(0).expand(hidden_states.size(1), -1, -1),
            node_embeds.unsqueeze(0).expand(hidden_states.size(1), -1, -1)
        )
        enhanced_features = enhanced_features.transpose(0, 1)
        
        # Reshape for 3D convolution
        batch_size = enhanced_features.size(0)
        enhanced_features = enhanced_features.view(
            batch_size, -1, 16, 14, 14  # Assuming 16 frames and 14x14 spatial size
        )
        
        # Spatial feature modeling
        spatial_features = self.spatial_modeling(enhanced_features)
        
        # Temporal feature modeling
        temporal_features = spatial_features.permute(0, 2, 1, 3, 4)
        temporal_features = temporal_features.reshape(batch_size, 16, -1)
        temporal_features = self.temporal_transformer(temporal_features)
        
        # Global average pooling
        pooled_features = temporal_features.mean(dim=[1, 2])
        
        # Classification
        logits = self.classifier(pooled_features)
        
        return logits

def process_fencing_books(books_dir, knowledge_extractor):
    """Process all fencing books and create knowledge graph"""
    print("Processing fencing books...")
    for book_file in os.listdir(books_dir):
        if book_file.endswith('.pdf'):
            print(f"Processing {book_file}...")
            book_path = os.path.join(books_dir, book_file)
            text = knowledge_extractor.extract_text_from_pdf(book_path)
            knowledge_extractor.create_knowledge_nodes(text, book_file)
    print("Knowledge graph creation complete!")

def create_enhanced_model(pretrained_model="MCG-NJU/videomae-base-finetuned-kinetics", num_classes=8):
    """Create knowledge-enhanced VideoMAE model"""
    # Load base model
    base_model = VideoMAEForVideoClassification.from_pretrained(pretrained_model)
    
    # Create Neo4j connection and knowledge graph
    knowledge_extractor = FencingKnowledgeExtractor()
    
    # Process books and create knowledge graph
    process_fencing_books("fencing_books", knowledge_extractor)
    
    # Get NetworkX graph
    knowledge_graph = knowledge_extractor.get_knowledge_graph()
    
    # Create enhanced model
    model = KnowledgeEnhancedVideoMAE(base_model, knowledge_graph, num_classes)
    
    return model, knowledge_extractor

if __name__ == "__main__":
    # Example usage
    model, knowledge_extractor = create_enhanced_model()
    print("Enhanced VideoMAE model created successfully!")
    knowledge_extractor.close() 