import cv2
import torch
import matplotlib.pyplot as plt
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import numpy as np
import os

def extract_frames(video_path, num_frames, image_size):
    """Extract frames from a video file"""
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
    return frames

def display_video_frames(frames, num_frames_to_show=8):
    """Display frames from the video"""
    # Select evenly spaced frames
    indices = np.linspace(0, len(frames)-1, num=num_frames_to_show, dtype=int)
    selected_frames = [frames[i] for i in indices]
    
    # Create a grid of images
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for i, frame in enumerate(selected_frames):
        axes[i].imshow(frame)
        axes[i].set_title(f"Frame {i+1}")
        axes[i].axis("off")
    
    plt.tight_layout()
    plt.savefig("fencing_frames.png")
    print(f"Saved frames visualization to 'fencing_frames.png'")
    
def visualize_predictions(scores, labels, title="Top Predictions", filename="predictions.png"):
    """Create a bar chart of top predictions"""
    # Get top predictions
    top_indices = np.argsort(scores)[::-1][:15]
    top_scores = [scores[i] for i in top_indices]
    top_labels = [labels[i] for i in top_indices]
    
    # Create horizontal bar chart
    plt.figure(figsize=(12, 8))
    bars = plt.barh(range(len(top_labels)), top_scores, color='skyblue')
    plt.yticks(range(len(top_labels)), top_labels)
    plt.xlabel('Confidence Score')
    plt.ylabel('Action Category')
    plt.title(title)
    plt.gca().invert_yaxis()  # Display highest confidence at the top
    
    # Add score values to the right of each bar
    for i, (bar, score) in enumerate(zip(bars, top_scores)):
        plt.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2, 
                 f"{score:.4f}", va='center')
    
    plt.tight_layout()
    plt.savefig(filename)
    print(f"Saved prediction visualization to '{filename}'")

def find_sport_related_categories(scores, labels, sport_keywords):
    """Find and extract sport-related categories from predictions"""
    sport_related = []
    
    for i, label in enumerate(labels):
        if any(keyword in label.lower() for keyword in sport_keywords):
            sport_related.append((i, label, scores[i]))
    
    return sorted(sport_related, key=lambda x: x[2], reverse=True)

def main():
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
    
    # Get expected dimensions
    num_frames = model.config.num_frames
    image_size = model.config.image_size
    print(f"Model expects {num_frames} frames with size {image_size}x{image_size}")
    
    # Extract frames
    print(f"Extracting {num_frames} frames from video...")
    frames = extract_frames(video_path, num_frames, image_size)
    print(f"Extracted {len(frames)} frames")
    
    # Display sample frames
    display_video_frames(frames)
    
    # Process and predict
    print("Running inference...")
    inputs = processor(frames, return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Convert to probabilities
    scores = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
    
    # Get predicted class
    predicted_class_idx = logits.argmax(-1).item()
    predicted_label = model.config.id2label[predicted_class_idx]
    
    print(f"\nPredicted class: {predicted_label} with confidence {scores[predicted_class_idx]:.4f}")
    
    # Convert id2label to a list for easier indexing
    labels = [model.config.id2label[i] for i in range(len(scores))]
    
    # Print top 5 predictions
    top_indices = np.argsort(scores)[::-1][:5]
    print("\nTop 5 predictions:")
    for idx in top_indices:
        print(f"{labels[idx]}: {scores[idx]:.4f}")
    
    # Visualize predictions
    visualize_predictions(scores, labels, "Fencing Video: Top Predictions", "fencing_predictions.png")
    
    # Find sport and combat related categories
    keywords = ["sword", "fencing", "fight", "combat", "martial", "sport", "duel"]
    sport_predictions = find_sport_related_categories(scores, labels, keywords)
    
    if sport_predictions:
        print("\nSport and combat related predictions:")
        for _, label, score in sport_predictions:
            print(f"{label}: {score:.4f}")
        
        # Create visualization for sport-related predictions
        sport_scores = [score for _, _, score in sport_predictions]
        sport_labels = [label for _, label, _ in sport_predictions]
        visualize_predictions(
            sport_scores, sport_labels, 
            "Fencing Video: Sport & Combat Related Predictions", 
            "fencing_sport_predictions.png"
        )
    
    print("\nAnalysis complete!")

if __name__ == "__main__":
    main() 