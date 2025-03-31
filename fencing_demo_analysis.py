import matplotlib.pyplot as plt
import numpy as np
import torch
from main import predict_video_class, load_video_opencv
import os
import cv2
from fencing_analyzer import FencingAnalyzer

def display_video_frames_with_timestamps(frames, fps=30, num_frames=8):
    """Display a sample of frames from the video with timestamps"""
    # Select evenly spaced frames to display
    total_frames = len(frames)
    sample_indices = np.linspace(0, total_frames-1, num=num_frames, dtype=int)
    
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, idx in enumerate(sample_indices):
        axes[i].imshow(frames[idx])
        time_in_seconds = idx / fps * (total_frames / num_frames)
        minutes = int(time_in_seconds // 60)
        seconds = time_in_seconds % 60
        timestamp = f"{minutes:02d}:{seconds:05.2f}"
        axes[i].set_title(f"Frame sample {i+1} (Est. Time: {timestamp})")
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig("fencing_frames.png")
    print(f"Fencing video frames saved to 'fencing_frames.png'")

def visualize_fencing_predictions(scores, id2label, top_k=15):
    """Visualize top k predictions for the fencing video with a bar chart"""
    # Get top k predictions
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    top_labels = [id2label[idx] for idx in top_indices]
    top_scores = [scores[idx] for idx in top_indices]
    
    # Create bar chart
    plt.figure(figsize=(14, 8))
    bars = plt.barh(range(len(top_labels)), top_scores, color='skyblue')
    plt.yticks(range(len(top_labels)), top_labels)
    plt.xlabel('Confidence Score')
    plt.ylabel('Action Category')
    plt.title('Top Predictions for Fencing Video')
    plt.gca().invert_yaxis()  # Display highest confidence at the top
    
    # Add score labels to the right of each bar
    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f"{score:.4f}", va='center')
    
    plt.tight_layout()
    plt.savefig("fencing_predictions.png")
    print(f"Fencing prediction chart saved to 'fencing_predictions.png'")

def analyze_fencing_video():
    # Define path to fencing video
    video_path = "fencing_demo_video.mp4"
    
    # Check if file exists
    if not os.path.exists(video_path):
        print(f"Error: File {video_path} not found.")
        return
    
    print(f"Analyzing fencing video: {video_path}")
    
    # Get video metadata using OpenCV
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
    
    # Load some frames for visualization (32 is enough for display)
    display_frames = load_video_opencv(video_path, num_frames=32, target_size=(224, 224))
    print(f"Loaded {len(display_frames)} frames for visualization")
    
    # Display sample frames with timestamps
    display_video_frames_with_timestamps(display_frames, fps)
    
    # Get predictions using VideoMAE
    print("\nRunning VideoMAE model for action classification...")
    predicted_label, logits, id2label = predict_video_class(video_path)
    
    # Apply softmax to get probabilities
    scores = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    # Print the top prediction
    print(f"\nPrimary prediction: {predicted_label} (confidence: {scores[logits.argmax(-1).item()]:.4f})")
    
    # Print top predictions
    print("\nTop 10 predictions:")
    top_k = 10
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
    
    for idx in top_indices:
        print(f"{id2label[idx]}: {scores[idx]:.4f}")
    
    # Check if any fencing-related categories are in the top predictions
    fencing_keywords = ["fencing", "sword", "duel", "fight", "sport"]
    fencing_related = []
    
    for idx, label in id2label.items():
        if any(keyword in label.lower() for keyword in fencing_keywords):
            fencing_related.append((idx, label, scores[idx]))
    
    if fencing_related:
        print("\nFencing-related categories found:")
        for idx, label, score in sorted(fencing_related, key=lambda x: x[2], reverse=True):
            print(f"{label}: {score:.4f}")
    else:
        print("\nNo explicit fencing-related categories found in the top predictions")
    
    # Visualize the predictions
    visualize_fencing_predictions(scores, id2label)

    # Use the new FencingAnalyzer
    analyzer = FencingAnalyzer()
    predictions, segments = analyzer.analyze_video(video_path)

if __name__ == "__main__":
    analyze_fencing_video() 