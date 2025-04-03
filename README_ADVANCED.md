# Advanced Fencing Analysis System

A comprehensive pipeline for analyzing fencing videos using computer vision, deep learning, and tactical knowledge modeling techniques. This system combines 3D ConvNet and transformer architectures, temporal segmentation, and a fencing knowledge graph to provide detailed analysis and coaching feedback.

## Key Components

### 1. 3D ConvNet + Transformer Architecture (`videomae_model.py`)
- Implements a VideoMAE-inspired approach for temporal understanding
- Uses 3D patch embedding for spatial-temporal analysis
- Includes transformer blocks with self-attention mechanism
- Provides a fencing-specific temporal model for technique classification

### 2. Temporal Segmentation (`temporal_segmentation.py`)
- Detects fencing actions using motion analysis
- Segments videos into meaningful technique clips
- Analyzes motion patterns to identify strokes and actions
- Handles frame buffering and segment extraction

### 3. Fencing Knowledge Graph (`fencing_knowledge_graph.py`)
- Models fencing tactical knowledge as a directed graph
- Encodes techniques, transitions, and probabilities
- Provides tactical analysis and technique prediction
- Generates coaching feedback based on observed sequences

### 4. Integrated Analyzer (`integrated_fencing_analyzer.py`)
- Combines all components into a unified pipeline
- Processes videos frame-by-frame with temporal analysis
- Adds visualization with coaching feedback and technique detection
- Provides detailed analysis summaries and statistics

## Usage

```bash
python integrated_fencing_analyzer.py video_path [options]
```

### Options
- `--output_dir`: Directory to save results (default: "results")
- `--max_frames`: Maximum number of frames to process
- `--fencer_ids`: Comma-separated list of fencer IDs to track
- `--temporal_model`: Path to trained temporal model
- `--pose_model`: Path to trained pose classifier model (default: "models/pose_classifier.pth")
- `--sword_model`: Path to trained sword detector model (default: "models/yolov8n_blade.pt")
- `--knowledge_graph`: Path to fencing knowledge graph
- `--no_vis`: Disable visualization output
- `--bout_mode`: Optimize for two fencers (bouts)

## Example

```bash
python integrated_fencing_analyzer.py fencing_videos/bout_1.mp4 --output_dir results --bout_mode
```

## Output

The system generates:
- Annotated video with fencer tracking, technique detection, and coaching feedback
- JSON analysis file with detailed information on detected techniques and segments
- Visualization of the fencing knowledge graph
- Summary statistics on technique usage and patterns

## Requirements

- Python 3.8+
- PyTorch 1.8+
- OpenCV 4.5+
- NetworkX 2.5+
- NumPy, Matplotlib
- CUDA-enabled GPU recommended for optimal performance

## Model Training

For optimal performance, train the temporal model on your specific fencing dataset:

```bash
python train_temporal_model.py --data_path fencing_dataset --epochs 25 --batch_size 16
```

## Advanced Features

- **Tactical Analysis**: Evaluates technique sequences against the knowledge graph
- **Coaching Feedback**: Provides actionable coaching tips based on detected patterns
- **Temporal Segmentation**: Automatically identifies and extracts technique segments
- **Technique Classification**: Identifies specific fencing techniques using temporal cues
- **Visualization**: Renders analysis results directly on the video with feedback

## Extending the System

The modular architecture makes it easy to extend:
- Add new techniques to the knowledge graph
- Train the temporal model on additional datasets
- Customize coaching feedback templates
- Integrate with scoring systems or additional sensors 