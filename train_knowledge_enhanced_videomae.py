import os
import torch
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from fencing_knowledge_videomae import create_enhanced_model
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import logging
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FencingVideoDataset(torch.utils.data.Dataset):
    def __init__(self, video_dir, processor, num_frames=16):
        self.video_paths = []
        self.labels = []
        self.processor = processor
        self.num_frames = num_frames
        
        # Collect video paths and labels
        for label in os.listdir(video_dir):
            label_dir = os.path.join(video_dir, label)
            if os.path.isdir(label_dir):
                for video_file in os.listdir(label_dir):
                    if video_file.endswith(('.mp4', '.avi', '.mov')):
                        self.video_paths.append(os.path.join(label_dir, video_file))
                        self.labels.append(label)
        
        # Create label mapping
        self.unique_labels = sorted(set(self.labels))
        self.label2id = {label: i for i, label in enumerate(self.unique_labels)}
        self.id2label = {i: label for label, i in self.label2id.items()}
    
    def __len__(self):
        return len(self.video_paths)
    
    def __getitem__(self, idx):
        video_path = self.video_paths[idx]
        label = self.labels[idx]
        
        # Load and process video frames
        frames = self.load_video(video_path)
        inputs = self.processor(frames, return_tensors="pt")
        
        # Convert label to tensor
        label_id = self.label2id[label]
        label_tensor = torch.tensor(label_id)
        
        return {
            "pixel_values": inputs.pixel_values.squeeze(),
            "labels": label_tensor
        }
    
    def load_video(self, video_path):
        """Load video frames using OpenCV"""
        import cv2
        import numpy as np
        
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate frame indices to sample
        indices = np.linspace(0, frame_count-1, self.num_frames, dtype=int)
        
        for idx in indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ret, frame = cap.read()
            if ret:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (224, 224))
                frames.append(frame)
            else:
                # If frame reading fails, add a black frame
                frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
        
        cap.release()
        return frames

def train(args):
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Set up tensorboard
    writer = SummaryWriter(os.path.join(args.output_dir, 'runs', datetime.now().strftime('%Y%m%d-%H%M%S')))
    
    # Create model and knowledge graph
    logger.info("Creating knowledge-enhanced VideoMAE model...")
    model, knowledge_extractor = create_enhanced_model(
        pretrained_model=args.pretrained_model,
        num_classes=len(os.listdir(args.data_dir))
    )
    
    # Move model to device
    device = torch.device("cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu")
    model = model.to(device)
    
    # Create datasets
    from transformers import VideoMAEImageProcessor
    processor = VideoMAEImageProcessor.from_pretrained(args.pretrained_model)
    
    train_dataset = FencingVideoDataset(
        os.path.join(args.data_dir, 'train'),
        processor,
        num_frames=16
    )
    
    val_dataset = FencingVideoDataset(
        os.path.join(args.data_dir, 'val'),
        processor,
        num_frames=16
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers
    )
    
    # Optimizer and scheduler
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Training loop
    best_val_acc = 0.0
    for epoch in range(args.num_epochs):
        logger.info(f"Starting epoch {epoch+1}/{args.num_epochs}")
        
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for batch in tqdm(train_loader, desc="Training"):
            pixel_values = batch["pixel_values"].to(device)
            labels = batch["labels"].to(device)
            
            optimizer.zero_grad()
            outputs = model(pixel_values)
            
            loss = torch.nn.functional.cross_entropy(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += labels.size(0)
            train_correct += predicted.eq(labels).sum().item()
        
        train_loss = train_loss / len(train_loader)
        train_acc = 100. * train_correct / train_total
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for batch in tqdm(val_loader, desc="Validation"):
                pixel_values = batch["pixel_values"].to(device)
                labels = batch["labels"].to(device)
                
                outputs = model(pixel_values)
                loss = torch.nn.functional.cross_entropy(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += labels.size(0)
                val_correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * val_correct / val_total
        
        # Log metrics
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/train', train_acc, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)
        
        logger.info(f"Epoch {epoch+1}/{args.num_epochs}")
        logger.info(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        logger.info(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, os.path.join(args.output_dir, 'best_model.pth'))
        
        scheduler.step()
    
    # Close knowledge graph connection
    knowledge_extractor.close()
    writer.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train knowledge-enhanced VideoMAE for fencing")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="Directory containing train and val subdirectories")
    parser.add_argument("--output_dir", type=str, default="./knowledge_enhanced_videomae",
                        help="Directory to save the model")
    parser.add_argument("--pretrained_model", type=str,
                        default="MCG-NJU/videomae-base-finetuned-kinetics",
                        help="Pretrained VideoMAE model to start from")
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate for training")
    parser.add_argument("--num_epochs", type=int, default=10,
                        help="Number of epochs for training")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--no_cuda", action="store_true",
                        help="Disable CUDA even if available")
    
    args = parser.parse_args()
    train(args) 