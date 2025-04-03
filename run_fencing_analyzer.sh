#!/bin/bash

# Run Fencing Analysis with CNN and Sword Detection

# Default values
VIDEO=""
MAX_FRAMES=0
FENCERS=""
FIRST_ONLY=0
POSE_MODEL="models/pose_classifier.pth"
SWORD_MODEL="models/yolov8n_blade.pt"
OUTPUT_DIR="results"
BOUT_MODE=0

# Display help
function show_help {
    echo "Fencing Video Analysis Script"
    echo "Usage: ./run_fencing_analyzer.sh -v <video_file> [options]"
    echo ""
    echo "Options:"
    echo "  -v, --video        Video file to analyze (required)"
    echo "  -f, --fencers      Comma-separated list of fencer IDs to track (e.g. '0,1')"
    echo "  -m, --max-frames   Maximum number of frames to process (default: all)"
    echo "  -p, --pose-model   Path to CNN pose classifier model (default: models/pose_classifier.pth)"
    echo "  -s, --sword-model  Path to sword detector model (default: models/yolov8n_blade.pt)"
    echo "  -o, --output       Output directory (default: results)"
    echo "  -i, --identify     Only show the first frame to identify fencers and exit"
    echo "  -b, --bout-mode    Enable bout mode to optimize for 2 fencers (for bouts)"
    echo "  -h, --help         Show this help message"
    echo ""
    echo "Example:"
    echo "  ./run_fencing_analyzer.sh -v fencing_demo_video.mp4 -f 0,1 -m 100 -b"
    exit 1
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    key="$1"
    case $key in
        -v|--video)
            VIDEO="$2"
            shift
            shift
            ;;
        -f|--fencers)
            FENCERS="$2"
            shift
            shift
            ;;
        -m|--max-frames)
            MAX_FRAMES="$2"
            shift
            shift
            ;;
        -p|--pose-model)
            POSE_MODEL="$2"
            shift
            shift
            ;;
        -s|--sword-model)
            SWORD_MODEL="$2"
            shift
            shift
            ;;
        -o|--output)
            OUTPUT_DIR="$2"
            shift
            shift
            ;;
        -i|--identify)
            FIRST_ONLY=1
            shift
            ;;
        -b|--bout-mode)
            BOUT_MODE=1
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            show_help
            ;;
    esac
done

# Check if video file is provided
if [ -z "$VIDEO" ]; then
    echo "Error: Video file is required"
    show_help
fi

# Check if video file exists
if [ ! -f "$VIDEO" ]; then
    echo "Error: Video file '$VIDEO' not found"
    exit 1
fi

# Build command
ESCAPED_VIDEO=$(printf %q "$VIDEO")
CMD="python advanced_fencing_analyzer.py $ESCAPED_VIDEO --use_enhanced --pose_model $POSE_MODEL --sword_model $SWORD_MODEL --output_dir $OUTPUT_DIR"

# Add max frames if specified
if [ $MAX_FRAMES -gt 0 ]; then
    CMD="$CMD --max_frames $MAX_FRAMES"
fi

# Add fencer selection if specified
if [ ! -z "$FENCERS" ]; then
    CMD="$CMD --manual_select $FENCERS"
fi

# Add first frame only option if specified
if [ $FIRST_ONLY -eq 1 ]; then
    CMD="$CMD --first_only"
fi

# Add bout mode option if specified
if [ $BOUT_MODE -eq 1 ]; then
    CMD="$CMD --bout_mode"
fi

# Print command and run
echo "Running command: $CMD"
# Use quotes around the command to properly handle spaces and special characters
eval "$CMD"

# Check if analysis was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "==============================================================="
    echo "Analysis completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    
    # Get video name without extension
    VIDEO_NAME=$(basename "$VIDEO" | cut -f 1 -d '.')
    
    echo "Output files:"
    echo "  - Analysis video: $OUTPUT_DIR/${VIDEO_NAME}_pose_analysis.mp4"
    echo "  - HTML report: $OUTPUT_DIR/${VIDEO_NAME}_pose_report.html"
    echo "  - Results JSON: $OUTPUT_DIR/${VIDEO_NAME}_pose_analysis.json"
    echo "  - Visualizations: $OUTPUT_DIR/${VIDEO_NAME}_*.png"
    echo ""
    echo "To view the analysis, open the HTML report in a web browser."
    echo "==============================================================="
else
    echo "Analysis failed. Please check the error messages above."
fi 