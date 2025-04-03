#!/bin/bash

# Simple script to run fencing analysis

if [ $# -lt 1 ]; then
    echo "Usage: ./analyze_fencing.sh video_file [fencer_ids] [max_frames]"
    echo "Example: ./analyze_fencing.sh fencing_demo_video.mp4 0,1 100"
    exit 1
fi

VIDEO="$1"
FENCERS="${2:-}"
MAX_FRAMES="${3:-0}"

# Set defaults
POSE_MODEL="models/pose_classifier.pth"
SWORD_MODEL="models/yolov8n_blade.pt"
OUTPUT_DIR="results"

# Construct the command
CMD="python advanced_fencing_analyzer.py \"$VIDEO\" --use_enhanced --bout_mode"
CMD="$CMD --pose_model $POSE_MODEL --sword_model $SWORD_MODEL --output_dir $OUTPUT_DIR"

# Add max frames if provided
if [ $MAX_FRAMES -gt 0 ]; then
    CMD="$CMD --max_frames $MAX_FRAMES"
fi

# Add fencer selection if provided
if [ ! -z "$FENCERS" ]; then
    CMD="$CMD --manual_select $FENCERS"
fi

# Print the command
echo "Running: $CMD"

# Execute the command
eval "$CMD"

# Check if analysis was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "===================================================="
    echo "Analysis completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    
    # Get video name without extension
    VIDEO_NAME=$(basename "$VIDEO" | cut -f 1 -d '.')
    
    echo "Key output files:"
    echo "  - Video: $OUTPUT_DIR/${VIDEO_NAME}_pose_analysis.mp4"
    echo "  - Report: $OUTPUT_DIR/${VIDEO_NAME}_pose_report.html"
    echo ""
    echo "To view the analysis, open the HTML report in a web browser."
    echo "===================================================="
else
    echo "Analysis failed. Please check the error messages above."
fi 