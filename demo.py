import matplotlib.pyplot as plt
import numpy as np
import torch
from main import predict_video_class, load_video_opencv
import os

def display_video_frames(frames, num_frames=8):
    """Display a sample of frames from the video"""
    # Select evenly spaced frames to display
    total_frames = len(frames)
    sample_indices = np.linspace(0, total_frames-1, num=num_frames, dtype=int)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(sample_indices):
        axes[i].imshow(frames[idx])
        axes[i].set_title(f"Frame sample {i+1}")
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig("video_frames.png")
    print(f"Video frames saved to 'video_frames.png'")

def visualize_predictions(scores, id2label, top_k=10):
    """Visualize top k predictions with a bar chart"""
    # Get top k predictions
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_labels = [id2label[idx] for idx in top_indices]
    top_scores = [scores[idx] for idx in top_indices]
    
    # Create bar chart
    plt.figure(figsize=(12, 6))
    plt.barh(range(len(top_labels)), top_scores, color='skyblue')
    plt.yticks(range(len(top_labels)), top_labels)
    plt.xlabel('Confidence Score')
    plt.ylabel('Action Category')
    plt.title('Top Predictions')
    plt.gca().invert_yaxis()  # Display highest confidence at the top
    plt.tight_layout()
    plt.savefig("prediction_scores.png")
    print(f"Prediction chart saved to 'prediction_scores.png'")

if __name__ == "__main__":
    # Use local fencing video file instead of downloading
    video_path = "fencing_demo_video.mp4"
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: File {video_path} not found.")
        exit(1)
    
    print(f"Analyzing video: {video_path}")
    
    # Load video frames for visualization (32 frames is enough for display)
    frames = load_video_opencv(video_path, num_frames=32, target_size=(224, 224))
    print(f"Loaded {len(frames)} frames for visualization")
    
    # Display sample frames
    display_video_frames(frames)
    
    # Get predictions
    predicted_label, logits, id2label = predict_video_class(video_path)
    print(f"Predicted class: {predicted_label}")
    
    # Compute softmax scores
    scores = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    # Visualize predictions
    visualize_predictions(scores, id2label)
    
    print("\nTop 5 predictions:")
    top_k = 5
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    for idx in top_indices:
        print(f"{id2label[idx]}: {scores[idx]:.4f}") 