import os
import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from transformers import VideoMAEImageProcessor, VideoMAEForVideoClassification
import argparse
import json
from glob import glob
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
import seaborn as sns
from tqdm import tqdm

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
            print(f"Warning: Could not read frame {i} from {video_path}")
            frame = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        
        # Convert BGR to RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Resize
        frame = cv2.resize(frame, (image_size, image_size))
        frames.append(frame)
    
    cap.release()
    return frames

def predict_video(video_path, model, processor, device='cpu'):
    """Predict the class of a video using the fine-tuned model"""
    # Get model configuration
    num_frames = model.config.num_frames
    image_size = model.config.image_size
    
    # Extract frames
    frames = extract_frames(video_path, num_frames, image_size)
    
    # Process frames
    inputs = processor(frames, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Run inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
    
    # Get predicted class
    scores = torch.softmax(logits, dim=1).squeeze().cpu().numpy()
    predicted_class_idx = np.argmax(scores)
    predicted_label = model.config.id2label[predicted_class_idx]
    
    return predicted_label, scores

def evaluate_on_test_set(test_dir, model, processor, device='cpu'):
    """Evaluate the model on a test set"""
    # Get all subdirectories (categories)
    categories = [d for d in os.listdir(test_dir) if os.path.isdir(os.path.join(test_dir, d))]
    
    all_predictions = []
    all_ground_truth = []
    all_videos = []
    
    for category in categories:
        category_dir = os.path.join(test_dir, category)
        
        # Get all video files
        video_files = []
        for ext in ['mp4', 'avi', 'mov', 'mkv']:
            video_files.extend(glob(os.path.join(category_dir, f"*.{ext}")))
        
        print(f"Processing {len(video_files)} videos in category: {category}")
        
        for video_path in tqdm(video_files, desc=f"Evaluating {category}"):
            predicted_label, _ = predict_video(video_path, model, processor, device)
            
            all_predictions.append(predicted_label)
            all_ground_truth.append(category)
            all_videos.append(video_path)
    
    # Calculate metrics
    accuracy = accuracy_score(all_ground_truth, all_predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        all_ground_truth, all_predictions, average='weighted'
    )
    
    # Create confusion matrix
    class_names = sorted(list(set(all_ground_truth)))
    cm = confusion_matrix(all_ground_truth, all_predictions, labels=class_names)
    
    # Create a DataFrame with the results
    results = {
        'video': all_videos,
        'true_label': all_ground_truth,
        'predicted_label': all_predictions,
        'correct': [p == g for p, g in zip(all_predictions, all_ground_truth)]
    }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'confusion_matrix': cm,
        'class_names': class_names,
        'results': results
    }

def plot_confusion_matrix(cm, class_names, output_path):
    """Plot and save confusion matrix"""
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(output_path)
    print(f"Confusion matrix saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="Evaluate a fine-tuned VideoMAE model on fencing videos")
    parser.add_argument("--model_dir", type=str, required=True,
                        help="Directory containing the fine-tuned model")
    parser.add_argument("--test_dir", type=str, required=True,
                        help="Directory containing test videos organized in class subdirectories")
    parser.add_argument("--output_dir", type=str, default="./evaluation_results",
                        help="Directory to save evaluation results")
    parser.add_argument("--single_video", type=str, default=None,
                        help="Path to a single video for inference (optional)")
    parser.add_argument("--use_gpu", action="store_true",
                        help="Use GPU for inference if available")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() and args.use_gpu else 'cpu'
    print(f"Using device: {device}")
    
    # Load model and processor
    model = VideoMAEForVideoClassification.from_pretrained(args.model_dir)
    processor = VideoMAEImageProcessor.from_pretrained(args.model_dir)
    model = model.to(device)
    model.eval()
    
    # Print model details
    num_labels = model.config.num_labels
    id2label = model.config.id2label
    print(f"Model loaded with {num_labels} classes: {list(id2label.values())}")
    
    # Single video inference mode
    if args.single_video:
        if not os.path.exists(args.single_video):
            print(f"Error: Video file not found: {args.single_video}")
            return
        
        print(f"Running inference on video: {args.single_video}")
        predicted_label, scores = predict_video(args.single_video, model, processor, device)
        
        print(f"Predicted class: {predicted_label}")
        print("\nTop predictions:")
        top_indices = np.argsort(scores)[::-1][:5]
        for idx in top_indices:
            print(f"{id2label[idx]}: {scores[idx]:.4f}")
        
        # Visualize the prediction
        plt.figure(figsize=(10, 6))
        top_k = min(5, num_labels)
        top_indices = np.argsort(scores)[::-1][:top_k]
        top_labels = [id2label[idx] for idx in top_indices]
        top_scores = [scores[idx] for idx in top_indices]
        
        plt.barh(range(len(top_labels)), top_scores, color='skyblue')
        plt.yticks(range(len(top_labels)), top_labels)
        plt.xlabel('Confidence Score')
        plt.ylabel('Class')
        plt.title(f'Prediction: {predicted_label}')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        output_image = os.path.join(args.output_dir, "single_prediction.png")
        plt.savefig(output_image)
        print(f"Prediction visualization saved to {output_image}")
        
        return
    
    # Evaluate on test set
    print("Evaluating model on test set...")
    eval_results = evaluate_on_test_set(args.test_dir, model, processor, device)
    
    # Print evaluation metrics
    print("\nEvaluation Results:")
    print(f"Accuracy: {eval_results['accuracy']:.4f}")
    print(f"Precision: {eval_results['precision']:.4f}")
    print(f"Recall: {eval_results['recall']:.4f}")
    print(f"F1 Score: {eval_results['f1']:.4f}")
    
    # Save confusion matrix
    cm_path = os.path.join(args.output_dir, "confusion_matrix.png")
    plot_confusion_matrix(eval_results['confusion_matrix'], eval_results['class_names'], cm_path)
    
    # Save detailed results to JSON
    results_path = os.path.join(args.output_dir, "evaluation_results.json")
    with open(results_path, 'w') as f:
        # Convert numpy arrays to lists for JSON serialization
        results_json = {
            'accuracy': float(eval_results['accuracy']),
            'precision': float(eval_results['precision']),
            'recall': float(eval_results['recall']),
            'f1': float(eval_results['f1']),
            'class_names': eval_results['class_names'],
            'results': {
                'video': eval_results['results']['video'],
                'true_label': eval_results['results']['true_label'],
                'predicted_label': eval_results['results']['predicted_label'],
                'correct': eval_results['results']['correct']
            }
        }
        json.dump(results_json, f, indent=2)
    
    print(f"Detailed evaluation results saved to {results_path}")

if __name__ == "__main__":
    main() 