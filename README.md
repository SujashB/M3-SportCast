# M3-SportCast: Fencing Analytics

A computer vision system for analyzing fencing movements and techniques using pose estimation and tracking.

## Features

- Detect and track fencers in videos
- Analyze pose and movement patterns
- Identify fencing techniques and movements
- Generate visualizations and reports
- Support for multiple fencers in the same frame

## Installation

1. Clone the repository:
```bash
git clone https://github.com/SujashB/M3-SportCast.git
cd M3-SportCast
```

2. Create a virtual environment and activate it:
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Run the fencing analyzer on a video file:

```bash
python advanced_fencing_analyzer.py path/to/your/video.mp4
```

This will process the video and generate visualization results in the `results` directory.

### Manual Fencer Selection

To first see the detected fencers in the first frame:

```bash
python advanced_fencing_analyzer.py path/to/your/video.mp4 --first_only
```

Then select specific fencers to analyze (using their IDs from the first frame):

```bash
python advanced_fencing_analyzer.py path/to/your/video.mp4 --manual_select 0,1
```

### Additional Options

- `--output_dir`: Specify a custom output directory (default: "results")
- `--max_frames`: Limit the number of frames to process
- `--no_viz`: Disable visualization outputs
- `--first_only`: Only show the first frame with detected fencers and exit

## Output

The analyzer generates the following files:

- JSON analysis data with pose information and technique detection
- Visualization images (pie charts, bar charts, timelines)
- Summary visualizations for each detected fencer
- HTML report with all analysis results
- Video with pose visualization and bounding boxes

## Examples

```bash
# Analyze only the first 100 frames
python advanced_fencing_analyzer.py fencing_video.mp4 --max_frames 100

# Save results to a custom directory
python advanced_fencing_analyzer.py fencing_video.mp4 --output_dir my_results

# First check which fencers are detected
python advanced_fencing_analyzer.py fencing_video.mp4 --first_only

# Then analyze specific fencers
python advanced_fencing_analyzer.py fencing_video.mp4 --manual_select 0,1
```

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
