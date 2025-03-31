import cv2
import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import os

print("Starting fencing video analysis...")

# Path to the video
video_path = "fencing_demo_video.mp4"

if not os.path.exists(video_path):
    print(f"Error: File {video_path} not found!")
    exit(1)

# Get basic video info
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
duration = frame_count / fps
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.release()

print(f"Video properties:")
print(f"- Resolution: {width}x{height}")
print(f"- FPS: {fps}")
print(f"- Duration: {duration:.2f} seconds ({frame_count} frames)")

# Load the model
model_name = "MCG-NJU/videomae-base-finetuned-kinetics"
print(f"Loading model: {model_name}")
processor = VideoMAEImageProcessor.from_pretrained(model_name)
model = VideoMAEForVideoClassification.from_pretrained(model_name)

# Get the number of frames the model expects
num_frames = model.config.num_frames
image_size = model.config.image_size
print(f"Model expects {num_frames} frames with size {image_size}x{image_size}")

# Extract frames
print(f"Extracting {num_frames} frames from video...")
cap = cv2.VideoCapture(video_path)
indices = np.linspace(0, frame_count-1, num=num_frames, dtype=int)
frames = []

for i in indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, i)
    ret, frame = cap.read()
    if not ret:
        print(f"Warning: Could not read frame {i}")
        frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
    
    # Convert BGR to RGB
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Resize
    frame = cv2.resize(frame, (image_size, image_size))
    frames.append(frame)

cap.release()
print(f"Extracted {len(frames)} frames")

# Process and predict
print("Running inference...")
inputs = processor(frames, return_tensors="pt")
with torch.no_grad():
    outputs = model(**inputs)
    logits = outputs.logits

# Get predicted class
predicted_class_idx = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_class_idx]

print(f"\nPredicted class: {predicted_label}")

# Print top 5 predictions
scores = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
top_k = 5
top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

print("\nTop 5 predictions:")
for idx in top_indices:
    print(f"{model.config.id2label[idx]}: {scores[idx]:.4f}")

print("\nAnalysis complete!") 