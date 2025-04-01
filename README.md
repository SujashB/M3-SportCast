# M3-SportCast Fencing Analysis

This project combines computer vision and knowledge graph technologies to analyze fencing videos. It can classify fencing techniques and actions by leveraging both visual cues and structured knowledge from fencing literature.

## Components

### 1. Knowledge Graph Database (Neo4j)

The system uses a Neo4j graph database to store and query fencing knowledge extracted from fencing books and manuals. The knowledge graph contains:

- Fencing techniques (attack, defense, parry, riposte, etc.)
- Movement patterns associated with each technique
- Relationships between techniques
- Knowledge extracted from fencing literature

### 2. Video Analysis

The system analyzes fencing videos to detect and classify fencing techniques:

- Frame extraction and preprocessing
- Motion detection between frames
- Classification of techniques using a combination of visual cues and knowledge patterns

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements_knowledge.txt
   ```

2. Configure Neo4j:
   - Create a `.env` file with your Neo4j password:
     ```
     NEO4J_PASSWORD=your_password
     ```
   - Make sure Neo4j is running on localhost:7687
   - Run `create_fencing_db.py` to populate the database with fencing knowledge

3. Analyze a fencing video:
   ```
   python simple_fencing_analyzer.py
   ```

## Results

When analyzing `fencing_demo_video.mp4`, the system produces classification probabilities for various fencing techniques. The knowledge graph enhances the analysis by providing structured information about fencing techniques and their patterns.

## Implementation

The system is implemented with:
- PyTorch for neural network models
- OpenCV for video processing
- Neo4j for knowledge graph storage and querying
- Python's data processing libraries

## Future Work

1. Train the model on a dataset of labeled fencing videos
2. Improve the motion analysis component
3. Enhance the knowledge graph with more detailed technique descriptions
4. Add temporal analysis to identify sequences of techniques
5. Develop a user interface for interactive analysis
