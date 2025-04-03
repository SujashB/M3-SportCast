# M3-SportCast Fencing Analysis System

This project provides an AI-powered fencing bout analysis system that:
1. Detects fencers using YOLOv8
2. Tracks fencers across frames
3. Classifies fencer poses (neutral, attack, defense, lunge) using a CNN
4. Detects fencing blades and sword parts
5. Generates comprehensive analysis and visualization

## Key Features

- **Fencer Detection**: Automatically identifies fencers in videos using YOLOv8
- **Multi-Fencer Tracking**: Maintains fencer identity across frames using SORT tracking
- **Pose Classification**: CNN-based classification of fencing poses (neutral, attack, defense, lunge)
- **Enhanced Lunge Detection**: Special highlighting and effects for lunge detection
- **Blade Tracking**: Identification of fencing blades and their parts
- **Comprehensive Analysis**: Statistics on techniques, poses, and movements
- **Bout Mode**: Optimized detection for two-fencer competitive bouts

## Components

The system consists of the following main components:

- **Advanced Fencing Analyzer**: Comprehensive analysis with fencer tracking, pose estimation, and sword detection
- **Enhanced Fencer Detector**: YOLOv8-based detector with CNN pose classification and sword detection
- **Fencing CNN**: CNN-based pose classifier for fencing poses
- **Pose Estimation Helper**: Provides skeletal keypoint detection for traditional pose analysis
- **SORT Tracking**: Maintains fencer identity across frames

## Installation

1. Install required packages:

```bash
pip install -r requirements.txt
```

2. Make sure you have the following models in the `models` directory:
   - `pose_classifier.pth`: CNN model for pose classification
   - `yolov8n_blade.pt`: YOLOv8 model for sword detection
   - `yolov8n.pt`: Base YOLOv8 model (will be downloaded automatically if missing)

## Quick Start

The system includes two easy-to-use scripts:

1. First, identify fencers in your video:
```bash
./identify_fencers.sh fencing_demo_video.mp4
```

2. Then analyze the video with specific fencer IDs:
```bash
./analyze_fencing.sh fencing_demo_video.mp4 0,1 150
```

Where:
- `fencing_demo_video.mp4` is your video file
- `0,1` are the fencer IDs to track (from the identification step)
- `150` is the maximum number of frames to process (optional)

## Detailed Usage

### Using the Identify Fencers Script

```bash
./identify_fencers.sh <video_file>
```

This script will:
1. Process only the first frame of the video
2. Show the detected fencers with their ID numbers
3. Save an annotated frame to the results directory
4. Help you determine which fencer IDs to use for full analysis

Example:
```bash
./identify_fencers.sh fencing_demo_video.mp4
```

### Using the Analysis Script

```bash
./analyze_fencing.sh <video_file> [fencer_ids] [max_frames]
```

Parameters:
- `video_file`: Path to the fencing video (required)
- `fencer_ids`: Comma-separated list of fencer IDs to track (optional)
- `max_frames`: Maximum number of frames to process (optional)

Example:
```bash
./analyze_fencing.sh cropped_bout.mp4 0,1 200
```

### Using the Full-Featured Script

For more advanced options, use the full-featured script:

```bash
./run_fencing_analyzer.sh -v <video_file> [options]
```

Options:
- `-v, --video`: Video file to analyze (required)
- `-f, --fencers`: Comma-separated list of fencer IDs to track (e.g. '0,1')
- `-m, --max-frames`: Maximum number of frames to process (default: all)
- `-i, --identify`: Only show the first frame to identify fencers and exit
- `-b, --bout-mode`: Optimize for detecting exactly 2 fencers (for bouts)
- `-p, --pose-model`: Path to CNN pose classifier model
- `-s, --sword-model`: Path to sword detector model
- `-o, --output-dir`: Directory to save results
- `-e, --no-enhanced`: Disable enhanced detection (use basic detection only)
- `-h, --help`: Show help message

Example:
```bash
./run_fencing_analyzer.sh -v fencing_demo_video.mp4 -f 0,1 -m 100 -b -o my_results
```

### Using the Python API

You can also use the Python API directly for more programmatic control:

```python
from advanced_fencing_analyzer import SimplifiedFencingAnalyzer

# Initialize the analyzer
analyzer = SimplifiedFencingAnalyzer(
    pose_model_path="models/pose_classifier.pth",
    sword_model_path="models/yolov8n_blade.pt",
    use_enhanced_detector=True,
    bout_mode=True
)

# Analyze a video
results = analyzer.analyze_video(
    video_path="fencing_demo_video.mp4",
    manual_select="0,1",  # Optional: Specify fencer IDs to track
    max_frames=100        # Optional: Limit number of frames to process
)
```

## Output Files and Results

The system generates the following outputs in the `results` directory:

1. **Annotated Video**: `<video_name>_pose_analysis.mp4`
   - Shows fencers with bounding boxes, pose classifications, and sword detections
   - Enhanced visual effects for lunges (glow, thicker borders, directional arrows)

2. **HTML Report**: `<video_name>_pose_report.html`
   - Comprehensive analysis with visualizations and statistics
   - Interactive analysis of the fencing performance

3. **JSON Data**: `<video_name>_pose_analysis.json`
   - Raw analysis data for further processing
   - Contains frame-by-frame pose and technique information

4. **Visualizations**:
   - Pose distribution pie charts
   - Technique frequency bar charts
   - Timeline visualizations
   - Joint angle analysis

5. **First Frame**: `<video_name>_first_frame_annotated.png`
   - When using identification mode, shows detected fencers with IDs

## Advanced Features

### Enhanced Lunge Detection

The system combines two approaches for accurate lunge detection:
1. **Traditional Pose Analysis**: Using skeletal keypoints to identify lunges based on joint angles and position changes
2. **CNN Classification**: Using a trained CNN model to classify fencer poses

When a lunge is detected, the system:
- Updates the bounding box with "LUNGE" text
- Adds a glowing effect around the fencer
- Displays thicker borders
- Shows directional arrows indicating movement

### Bout Mode

For analyzing competitive bouts with two fencers, the `--bout_mode` option:
- Optimizes detection for exactly two fencers
- Improves tracking of crossed fencers
- Enhances visualization for better bout analysis

## Training Your Own Models

### Training Pose Classifier

```bash
python train_fencing_cnn.py --mode train_pose
```

This trains a CNN to classify fencing poses (neutral, attack, defense, lunge) using your own dataset.

### Training Sword Detector

```bash
python train_fencing_cnn.py --mode train_sword
```

This trains a YOLOv8 model to detect fencing blades and their parts.

## Example Videos

The repository includes sample videos for testing:
- `fencing_demo_video.mp4`: Sample fencing demo
- `evenevenmorecropped (1).mp4`: Single fencer footage
- `cropped_bout.mp4`: Two-fencer bout

## System Requirements

- Python 3.8 or higher
- CUDA-compatible GPU (recommended for faster processing)
- At least 8GB RAM
- 500MB disk space for models

## Credits

This project uses:
- YOLOv8 for object detection
- MediaPipe for pose estimation
- PyTorch for CNN implementation
- OpenCV for image processing
- SORT algorithm for object tracking
