import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2
import os

class FencingPoseCNN(nn.Module):
    """
    CNN for classifying fencing poses from cropped bounding boxes
    """
    def __init__(self, num_classes=4):
        super(FencingPoseCNN, self).__init__()
        # Convolutional layers
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        
        # Pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Fully connected layers
        self.fc1 = nn.Linear(256 * 8 * 8, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)
        
        # Class names
        self.class_names = ['neutral', 'attack', 'defense', 'lunge']
    
    def forward(self, x):
        # Conv layers with ReLU and pooling
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = self.pool(F.relu(self.bn4(self.conv4(x))))
        
        # Flatten
        x = x.view(-1, 256 * 8 * 8)
        
        # Fully connected with dropout
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.fc2(x)
        
        return x
    
    def predict(self, image_tensor):
        """
        Make a prediction on a single image tensor
        
        Args:
            image_tensor: Preprocessed image tensor of shape [1, 3, 128, 128]
            
        Returns:
            class_id: Predicted class ID
            class_name: Name of predicted class
            confidence: Confidence score
        """
        self.eval()
        with torch.no_grad():
            outputs = self(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            
            return predicted.item(), self.class_names[predicted.item()], confidence.item()


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
    
    def classify_pose(self, image):
        """
        Classify fencing pose in an image
        
        Args:
            image: Input image (cropped fencer, BGR format)
            
        Returns:
            class_id: ID of predicted class
            class_name: Name of predicted class
            confidence: Confidence score
        """
        # Preprocess image
        tensor = self.preprocess_image(image)
        
        # Get prediction
        class_id, class_name, confidence = self.model.predict(tensor)
        
        return class_id, class_name, confidence
    
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