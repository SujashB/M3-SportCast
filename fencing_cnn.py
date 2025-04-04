import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os
from torchvision import models

class FencingPoseCNN(nn.Module):
    """
    CNN model for fencing pose classification
    """
    def __init__(self, num_classes=4):
        """
        Initialize the CNN model
        
        Args:
            num_classes: Number of output classes
        """
        super(FencingPoseCNN, self).__init__()
        
        # Use MobileNetV2 as base model
        self.features = models.mobilenet_v2(pretrained=True).features
        
        # Add classifier head
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(1280, num_classes)
        )
        
        # Freeze base model layers
        for param in self.features.parameters():
            param.requires_grad = False
        
        # Unfreeze the last two layers
        for param in self.features[-2:].parameters():
            param.requires_grad = True
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x: Input tensor
            
        Returns:
            logits: Output logits
        """
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        logits = self.classifier(x)
        return logits
    
    def predict(self, x):
        """
        Make a prediction
        
        Args:
            x: Input tensor
            
        Returns:
            class_id: Predicted class ID
            class_name: Predicted class name
            confidence: Prediction confidence
        """
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
            confidence, predicted_class = torch.max(probabilities, 1)
            
            class_id = predicted_class.item()
            confidence = confidence.item()
            
            class_names = ['neutral', 'attack', 'defense', 'lunge']
            class_name = class_names[class_id] if 0 <= class_id < len(class_names) else 'unknown'
            
            return class_id, class_name, confidence


class FencingPoseClassifier:
    """
    Fencing pose classifier using a CNN model
    """
    def __init__(self, model_path=None, device=None):
        """
        Initialize fencing pose classifier
        
        Args:
            model_path: Path to pretrained model weights
            device: torch device (cpu or cuda)
        """
        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
            
        # Initialize model
        self.model = FencingPoseCNN(num_classes=4)
        self.model.to(self.device)
        
        # Load pretrained weights if provided
        if model_path and os.path.exists(model_path):
            self.model.load_state_dict(torch.load(model_path, map_location=self.device))
            print(f"Loaded pose classifier weights from {model_path}")
        else:
            print("Initializing pose classifier with random weights")
        
        # Define image transforms
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    def preprocess_image(self, image):
        """
        Preprocess an image for the CNN
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            tensor: Preprocessed image tensor
        """
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Apply transforms
        tensor = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        return tensor
    
    def classify_pose(self, image, return_features=False):
        """
        Classify a fencer pose
        
        Args:
            image: Input image (BGR)
            return_features: Whether to return extracted features (default: False)
            
        Returns:
            pose_class: Predicted pose class
            confidence: Prediction confidence score (0-1)
            features (optional): Extracted features if return_features=True
        """
        # Convert BGR to RGB and preprocess
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_tensor = self.preprocess_image(image_rgb)
        
        # Extract features using MobileNet
        with torch.no_grad():
            features = self.model.features(image_tensor)
            features = F.adaptive_avg_pool2d(features, (1, 1))
            features = torch.flatten(features, 1)
            
            # Get class prediction
            output = self.model.classifier(features)
            
            # Apply softmax to get probabilities
            probabilities = F.softmax(output, dim=1)
            
            # Get predicted class and confidence
            confidence, predicted_class = torch.max(probabilities, 1)
            
            # Convert to numpy
            predicted_class = predicted_class.cpu().numpy()[0]
            confidence = confidence.cpu().numpy()[0]
            
            # Get class name
            class_names = ['neutral', 'attack', 'defense', 'lunge']
            if 0 <= predicted_class < len(class_names):
                pose_class = class_names[predicted_class]
            else:
                pose_class = 'unknown'
        
        if return_features:
            return pose_class, float(confidence), features.cpu().numpy()
        else:
            return pose_class, float(confidence)
    
    def train(self, train_loader, val_loader, epochs=10, lr=0.001, save_path=None):
        """
        Train the pose classifier
        
        Args:
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
            epochs: Number of training epochs
            lr: Learning rate
            save_path: Path to save trained model
            
        Returns:
            model: Trained model
        """
        # Set model to training mode
        self.model.train()
        
        # Define loss function and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        
        # Training loop
        best_val_acc = 0.0
        for epoch in range(epochs):
            # Training phase
            running_loss = 0.0
            correct = 0
            total = 0
            
            for inputs, labels in train_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                
                # Zero the parameter gradients
                optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(inputs)
                loss = criterion(outputs, labels)
                
                # Backward pass and optimize
                loss.backward()
                optimizer.step()
                
                # Statistics
                running_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
            
            train_loss = running_loss / total
            train_acc = correct / total
            
            # Validation phase
            self.model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0
            
            with torch.no_grad():
                for inputs, labels in val_loader:
                    inputs, labels = inputs.to(self.device), labels.to(self.device)
                    
                    outputs = self.model(inputs)
                    loss = criterion(outputs, labels)
                    
                    val_loss += loss.item() * inputs.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()
            
            val_loss = val_loss / val_total
            val_acc = val_correct / val_total
            
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            
            # Save best model
            if val_acc > best_val_acc and save_path:
                best_val_acc = val_acc
                torch.save(self.model.state_dict(), save_path)
                print(f"Saved model with validation accuracy: {val_acc:.4f}")
            
            # Set model back to training mode
            self.model.train()
        
        # Load best model if saved
        if save_path and os.path.exists(save_path):
            self.model.load_state_dict(torch.load(save_path, map_location=self.device))
            print(f"Loaded best model with validation accuracy: {best_val_acc:.4f}")
        
        return self.model


def extract_fencer_crops(frame, bounding_boxes, padding=10):
    """
    Extract cropped images of fencers from bounding boxes
    
    Args:
        frame: Input video frame
        bounding_boxes: List of bounding boxes (x1, y1, x2, y2)
        padding: Optional padding around bounding boxes
        
    Returns:
        crops: List of cropped images
    """
    crops = []
    frame_height, frame_width = frame.shape[:2]
    
    for box in bounding_boxes:
        x1, y1, x2, y2 = box
        
        # Add padding
        x1 = max(0, x1 - padding)
        y1 = max(0, y1 - padding)
        x2 = min(frame_width, x2 + padding)
        y2 = min(frame_height, y2 + padding)
        
        # Crop image
        crop = frame[y1:y2, x1:x2].copy()
        crops.append(crop)
    
    return crops


def create_dataset_from_yolo_dataset(dataset_path, output_path, split='train'):
    """
    Create a dataset for training the CNN from YOLOv8 dataset
    
    Args:
        dataset_path: Path to YOLOv8 dataset
        output_path: Path to save extracted crops
        split: Dataset split ('train', 'valid', 'test')
    """
    import shutil
    import random
    from tqdm import tqdm
    
    # Create output directories
    os.makedirs(output_path, exist_ok=True)
    classes = ['neutral', 'attack', 'defense', 'lunge']
    for cls in classes:
        os.makedirs(os.path.join(output_path, cls), exist_ok=True)
    
    # Get image paths
    image_dir = os.path.join(dataset_path, split, 'images')
    image_paths = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    # Process images
    for img_path in tqdm(image_paths, desc=f"Processing {split} set"):
        # Load image
        img = cv2.imread(img_path)
        if img is None:
            continue
        
        # Get corresponding label file
        label_path = img_path.replace('images', 'labels').rsplit('.', 1)[0] + '.txt'
        if not os.path.exists(label_path):
            continue
        
        # Read labels
        with open(label_path, 'r') as f:
            lines = f.readlines()
        
        # Extract fencer bounding boxes
        boxes = []
        for line in lines:
            parts = line.strip().split()
            class_id = int(parts[0])
            
            # Only process fencer class (assuming class 1 is fencer in the data.yaml)
            if class_id == 1:  # Fencer class (adjust if needed)
                # For YOLO format, we need to convert from center coordinates
                # For OBB format, we'll need to adapt this
                
                # For standard YOLO format: class x_center y_center width height
                if len(parts) == 5:
                    x_center, y_center, width, height = map(float, parts[1:5])
                    img_height, img_width = img.shape[:2]
                    
                    # Convert to pixel coordinates (x1, y1, x2, y2)
                    x1 = int((x_center - width/2) * img_width)
                    y1 = int((y_center - height/2) * img_height)
                    x2 = int((x_center + width/2) * img_width)
                    y2 = int((y_center + height/2) * img_height)
                    
                    boxes.append((x1, y1, x2, y2))
                
                # For OBB format: class x1 y1 x2 y2 x3 y3 x4 y4
                elif len(parts) == 9:
                    coords = list(map(float, parts[1:9]))
                    img_height, img_width = img.shape[:2]
                    
                    # Convert normalized coordinates to pixel coordinates
                    x_coords = [int(coords[i] * img_width) for i in range(0, 8, 2)]
                    y_coords = [int(coords[i] * img_height) for i in range(1, 8, 2)]
                    
                    # Get bounding box from min/max coordinates
                    x1, y1 = min(x_coords), min(y_coords)
                    x2, y2 = max(x_coords), max(y_coords)
                    
                    boxes.append((x1, y1, x2, y2))
        
        # Extract crops
        if boxes:
            crops = extract_fencer_crops(img, boxes)
            
            # Save crops to random pose classes for initial dataset creation
            # In a real implementation, you'd want to label these properly
            for i, crop in enumerate(crops):
                # Randomly assign to a class - in practice, you'd use labels
                cls = random.choice(classes)
                output_file = f"{os.path.basename(img_path).split('.')[0]}_crop_{i}.jpg"
                output_path_full = os.path.join(output_path, cls, output_file)
                cv2.imwrite(output_path_full, crop)
    
    print(f"Dataset creation complete. Output saved to {output_path}")


def create_dataloader_from_directory(data_dir, batch_size=32, train_ratio=0.8):
    """
    Create DataLoader objects from a directory of labeled images
    
    Args:
        data_dir: Path to directory containing class subdirectories
        batch_size: Batch size for DataLoader
        train_ratio: Ratio of training to validation data
        
    Returns:
        train_loader, val_loader: DataLoader objects for training and validation
    """
    from torch.utils.data import DataLoader, random_split
    from torchvision.datasets import ImageFolder
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Load dataset
    dataset = ImageFolder(root=data_dir, transform=transform)
    
    # Split into train and validation sets
    train_size = int(train_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader 