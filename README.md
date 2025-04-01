# M3-SportCast Advanced Fencing Analysis

This project combines computer vision, deep learning, and knowledge graph technologies to create a comprehensive fencing analysis system. It can classify fencing techniques, analyze movement sequences, track fencers, detect hits, and provide rich visualizations.

## Components

### 1. Core Video Analysis
- **VideoMAE Classification**: Identifies fencing techniques using a video transformer model
- **Pose Estimation**: Analyzes fencer body positions using MediaPipe
- **Integrated Analysis**: Combines both approaches for accurate technique identification

### 2. Sequence Analysis
- **LSTM Model**: Deep learning model for temporal sequence understanding
- **Sequence Extraction**: Identifies common sequences of techniques
- **Pattern Recognition**: Detects signature patterns for individual fencers

### 3. Knowledge Graph and Logical Rules
- **Neo4j Database**: Stores structured fencing knowledge
- **Logical Inference**: Applies first-order logic to refine classifications
- **Knowledge Integration**: Enhances analysis with expert fencing knowledge

### 4. Fencer Segmentation and Tracking
- **SAMURAI Model**: Advanced segmentation for precise fencer isolation (optional)
- **OpenCV Tracking**: Follows fencers throughout the video
- **Hit Detection**: Identifies potential hit moments based on proximity

### 5. Comprehensive Visualization
- **Technique Distribution**: Pie charts and bar graphs of techniques
- **Timeline Analysis**: Temporal visualization of technique sequences
- **Transition Matrices**: Shows flows between techniques
- **Movement Heatmaps**: Visualizes fencer positioning patterns
- **Interactive Reports**: HTML reports with integrated visualizations

## Setup

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Configure Neo4j (optional for knowledge graph features):
   - Create a `.env` file with your Neo4j password:
     ```
     NEO4J_PASSWORD=your_password
     ```
   - Make sure Neo4j is running on localhost:7687
   - Run `create_fencing_db.py` to populate the database with fencing knowledge

3. For segmentation with the SAMURAI model (optional):
   - Download the SAM model checkpoint from [the SAM repository](https://github.com/facebookresearch/segment-anything)
   - Save it to a location accessible to the script

## Usage

### Basic Analysis

```bash
python advanced_fencing_analyzer.py path_to_video.mp4
```

### Advanced Options

```bash
python advanced_fencing_analyzer.py path_to_video.mp4 \
  --output_dir results_folder \
  --sam_checkpoint path/to/sam_model.pth \
  --max_frames 500 \
  --no_viz  # Disable visualizations
  --no_seg  # Disable segmentation video output
```

## Output

The system generates:

1. **Analysis JSON**: Comprehensive analysis data in JSON format
2. **Visualizations**: Multiple visualization types (pie charts, timelines, heatmaps, etc.)
3. **HTML Report**: Interactive report summarizing all findings
4. **Segmentation Video**: Video showing tracked fencers (optional)
5. **Sequence Model**: Trained model for sequence prediction

## Implementation

The system integrates multiple Python modules:
- `advanced_fencing_analyzer.py`: Main integration script
- `simple_fencing_analyzer.py`: Core VideoMAE and pose analysis
- `sequence_analysis.py`: LSTM-based sequence processing
- `knowledge_graph_rules.py`: Logical rules and reasoning
- `fencer_segmentation.py`: Segmentation and tracking
- `data_visualization.py`: Comprehensive visualization tools
- `pose_estimation_helper.py`: MediaPipe-based pose analysis

## Results

When analyzing fencing videos, the system identifies:
- Individual techniques (attack, defense, parry, riposte, etc.)
- Sequences and patterns of techniques
- Fencer positions and movements
- Potential hit moments
- Signature sequences used by each fencer

## Future Work

1. Real-time analysis for live fencing matches
2. Training on a larger dataset of labeled fencing videos
3. Integration with competition scoring systems
4. Multi-camera support for 3D analysis
5. Tactical pattern recognition and suggestion system
