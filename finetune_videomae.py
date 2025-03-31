import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import cv2
from transformers import (
    VideoMAEForVideoClassification, 
    VideoMAEImageProcessor,
    TrainingArguments, 
    Trainer
)
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm
import json
import argparse
from dataclasses import dataclass
from typing import Dict, List, Optional, Union

# Custom dataset class for fencing videos
class FencingVideoDataset(Dataset):
    def __init__(self, video_paths, labels, label2id, id2label, processor, num_frames=16):
        self.video_paths = video_paths
        self.labels = labels
        self.label2id = label2id
        self.id2label = id2label
        self.processor = processor
        self.num_frames = num_frames
        
    def __len__(self):
        return len(self.video_paths)
    
    def extract_frames(self, video_path):
        """Extract evenly spaced frames from a video"""
        cap = cv2.VideoCapture(video_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate indices of frames to extract (evenly spaced)
        indices = np.linspace(0, frame_count-1, num=self.num_frames, dtype=int)
        
        frames = []
        for i in indices:
            # Set position to the specific frame
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
            
            if not ret:
                print(f"Warning: Could not read frame {i} from {video_path}")
                # Use a black frame if we can't read the actual frame
                frame = np.zeros((224, 224, 3), dtype=np.uint8)
            
            # Convert BGR to RGB
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        
        cap.release()
        return frames
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Extract frames from video
        frames = self.extract_frames(video_path)
        
        # Process frames using the VideoMAE processor
        pixel_values = self.processor(frames, return_tensors="pt").pixel_values.squeeze()
        
        # Convert label to numerical ID
        label_id = self.label2id[label]
        
        return {
            "pixel_values": pixel_values,
            "labels": torch.tensor(label_id)
        }

@dataclass
class DataCollatorForVideoMAE:
    """
    Data collator for VideoMAE. Stacks the processed frames and labels from multiple samples.
    """
    processor: VideoMAEImageProcessor
    
    def __call__(self, examples):
        pixel_values = torch.stack([example["pixel_values"] for example in examples])
        labels = torch.tensor([example["labels"] for example in examples])
        
        return {
            "pixel_values": pixel_values,
            "labels": labels
        }

def create_label_mappings(data_dir):
    """Create label to ID mappings based on the directory structure"""
    # Assuming each subdirectory in data_dir is a class
    class_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
    
    id2label = {i: label for i, label in enumerate(sorted(class_dirs))}
    label2id = {label: i for i, label in id2label.items()}
    
    return label2id, id2label

def get_video_paths_and_labels(data_dir):
    """Get all video paths and their corresponding labels"""
    video_paths = []
    labels = []
    
    # Assuming each subdirectory in data_dir is a class
    for class_name in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue
        
        # Get all video files in the class directory
        for ext in ['*.mp4', '*.avi', '*.mov', '*.mkv']:
            video_files = glob.glob(os.path.join(class_dir, ext))
            video_paths.extend(video_files)
            labels.extend([class_name] * len(video_files))
    
    return video_paths, labels

def finetune_videomae(
    data_dir,
    output_dir="./finetuned_videomae_fencing",
    pretrained_model="MCG-NJU/videomae-base-finetuned-kinetics",
    num_frames=16,
    batch_size=8,
    learning_rate=5e-5,
    num_epochs=10,
    warmup_ratio=0.1,
    save_steps=100,
):
    """Fine-tune VideoMAE on a custom fencing dataset"""
    # Create label mappings
    label2id, id2label = create_label_mappings(data_dir)
    num_labels = len(label2id)
    
    print(f"Found {num_labels} classes: {list(label2id.keys())}")
    
    # Save the label mappings
    with open(os.path.join(output_dir, "label_mappings.json"), "w") as f:
        json.dump({"label2id": label2id, "id2label": id2label}, f, indent=2)
    
    # Get video paths and labels
    video_paths, labels = get_video_paths_and_labels(data_dir)
    
    # Split into train and validation sets
    train_videos, val_videos, train_labels, val_labels = train_test_split(
        video_paths, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    print(f"Training on {len(train_videos)} videos, validating on {len(val_videos)} videos")
    
    # Load model and processor
    processor = VideoMAEImageProcessor.from_pretrained(pretrained_model)
    
    # Load the model and reset the classification head for our number of classes
    model = VideoMAEForVideoClassification.from_pretrained(
        pretrained_model,
        num_labels=num_labels,
        label2id=label2id,
        id2label=id2label,
        ignore_mismatched_sizes=True,  # Important when changing the size of the classification head
    )
    
    # Create datasets
    train_dataset = FencingVideoDataset(
        train_videos, train_labels, label2id, id2label, processor, num_frames
    )
    val_dataset = FencingVideoDataset(
        val_videos, val_labels, label2id, id2label, processor, num_frames
    )
    
    # Data collator
    data_collator = DataCollatorForVideoMAE(processor=processor)
    
    # Training arguments
    training_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        learning_rate=learning_rate,
        num_train_epochs=num_epochs,
        warmup_ratio=warmup_ratio,
        save_steps=save_steps,
        evaluation_strategy="steps",
        save_total_limit=2,
        load_best_model_at_end=True,
        logging_dir=f"{output_dir}/logs",
        remove_unused_columns=False,  # Important for custom datasets
    )
    
    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
    )
    
    # Start training
    print("Starting fine-tuning...")
    trainer.train()
    
    # Save the final model
    trainer.save_model(output_dir)
    print(f"Model saved to {output_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune VideoMAE on a custom fencing dataset")
    parser.add_argument("--data_dir", type=str, required=True, 
                        help="Directory containing video data organized in class subdirectories")
    parser.add_argument("--output_dir", type=str, default="./finetuned_videomae_fencing",
                        help="Directory to save the fine-tuned model")
    parser.add_argument("--pretrained_model", type=str, 
                        default="MCG-NJU/videomae-base-finetuned-kinetics",
                        help="Pretrained model to start from")
    parser.add_argument("--num_frames", type=int, default=16,
                        help="Number of frames to extract from each video")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs for training")
    
    args = parser.parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    finetune_videomae(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        pretrained_model=args.pretrained_model,
        num_frames=args.num_frames,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
    ) 