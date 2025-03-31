import argparse
import torch
from main import predict_video_class
import matplotlib.pyplot as plt

def visualize_top_predictions(scores, id2label, top_k=10):
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
    plt.savefig("custom_video_prediction.png")
    print(f"Prediction chart saved to 'custom_video_prediction.png'")

def main():
    parser = argparse.ArgumentParser(description="Classify a video using VideoMAE")
    parser.add_argument("--video_path", type=str, required=True, 
                        help="Path to the video file")
    parser.add_argument("--model", type=str, 
                        default="MCG-NJU/videomae-base-finetuned-kinetics",
                        help="HuggingFace model ID for VideoMAE")
    parser.add_argument("--top_k", type=int, default=10,
                        help="Number of top predictions to display")
    
    args = parser.parse_args()
    
    print(f"Classifying video: {args.video_path}")
    print(f"Using model: {args.model}")
    
    try:
        # Get predictions
        predicted_label, logits = predict_video_class(args.video_path, args.model)
        print(f"\nPredicted class: {predicted_label}")
        
        # Get model's id2label mapping
        from transformers import VideoMAEForVideoClassification
        model = VideoMAEForVideoClassification.from_pretrained(args.model)
        id2label = model.config.id2label
        
        # Compute softmax scores
        scores = torch.nn.functional.softmax(logits, dim=1).squeeze().tolist()
        
        # Print top k predictions
        print(f"\nTop {args.top_k} predictions:")
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:args.top_k]
        
        for idx in top_indices:
            print(f"{id2label[idx]}: {scores[idx]:.4f}")
            
        # Visualize predictions
        visualize_top_predictions(scores, id2label, args.top_k)
        
    except Exception as e:
        print(f"Error processing video: {e}")

if __name__ == "__main__":
    main() 