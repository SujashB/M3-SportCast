import torch
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import cv2
import os

def load_video_opencv(video_path, num_frames=16, target_size=(224, 224)):
    """
    Load video using OpenCV and return frames in the format expected by VideoMAE
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Calculate indices of frames to extract (evenly spaced)
    indices = np.linspace(0, frame_count-1, num_frames, dtype=int)
    
    frames = []
    for i in indices:
        # Set position to the specific frame
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if not ret:
            print(f"Warning: Could not read frame {i}")
            # Use a black frame if we can't read the actual frame
            frame = np.zeros((target_size[0], target_size[1], 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize frame
        frame = cv2.resize(frame, target_size)
        
        frames.append(frame)
    
    cap.release()
    
    return frames

def predict_video_class(video_path, model_name="MCG-NJU/videomae-base-finetuned-kinetics", num_frames=16):
    """
    Predict the class of a video using VideoMAE
    """
    # Load the model and processor
    processor = VideoMAEImageProcessor.from_pretrained(model_name)
    model = VideoMAEForVideoClassification.from_pretrained(model_name)
    
    # Get model configuration to check required input shapes
    config = model.config
    print(f"Model expects: {config.num_frames} frames, Image size: {config.image_size}")
    
    # Load video frames with correct size and number of frames
    frames = load_video_opencv(
        video_path, 
        num_frames=config.num_frames, 
        target_size=(config.image_size, config.image_size)
    )
    
    # Process frames - the processor will handle normalization and other transformations
    inputs = processor(frames, return_tensors="pt")
    
    # Forward pass through the model
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get predicted class
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    
    return predicted_label, logits, model.config.id2label

if __name__ == "__main__":
    print("Starting video analysis...")
    
    # Use local fencing video file instead of downloading
    video_path = "fencing_demo_video.mp4"
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: File {video_path} not found.")
        exit(1)
    
    print(f"Analyzing video: {video_path}")
    
    # Predict the class
    predicted_label, logits, id2label = predict_video_class(video_path)
    print(f"Predicted class: {predicted_label}")
    
    # Print top 5 predictions
    scores = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    top_k = 5
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    print("\nTop 5 predictions:")
    for idx in top_indices:
        print(f"{id2label[idx]}: {scores[idx]:.4f}")
        
    print("Analysis complete!")
