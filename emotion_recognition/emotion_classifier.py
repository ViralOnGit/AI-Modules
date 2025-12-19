"""
Emotion Recognition Module
Multi-class classification for emotions: Angry, Happy, Neutral, Sad
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from collections import Counter


class EmotionDataset(Dataset):
    """Custom Dataset for emotion recognition"""
    def __init__(self, root_dir, transform=None, use_face_detection=False):
        self.root_dir = root_dir
        self.transform = transform
        self.use_face_detection = use_face_detection
        self.classes = ['Angry', 'Happy', 'Neutral', 'Sad']
        self.class_to_idx = {cls_name: i for i, cls_name in enumerate(self.classes)}
        self.samples = []
        
        print(f"Scanning emotion dataset in {root_dir}")
        for class_name in self.classes:
            class_dir = os.path.join(root_dir, class_name)
            if os.path.isdir(class_dir):
                for img_name in os.listdir(class_dir):
                    if img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                        self.samples.append((os.path.join(class_dir, img_name), 
                                           self.class_to_idx[class_name]))
        print(f"Found {len(self.samples)} emotion samples in {root_dir}")
        
        if use_face_detection:
            try:
                from mtcnn import MTCNN
                self.detector = MTCNN()
            except ImportError:
                print("MTCNN not installed. Run `pip install mtcnn` or set use_face_detection=False.")
                self.use_face_detection = False
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.use_face_detection:
            try:
                img_array = np.array(image)
                faces = self.detector.detect_faces(img_array)
                if faces:
                    x, y, w, h = faces[0]['box']
                    x, y = max(0, x), max(0, y)
                    image = image.crop((x, y, x+w, y+h))
            except Exception as e:
                print(f"Face detection failed for {img_path}: {e}")
        
        if self.transform:
            image = self.transform(image)
        return image, label


class EmotionCNN(nn.Module):
    """Custom CNN for emotion recognition"""
    def __init__(self, num_classes=4):
        super(EmotionCNN, self).__init__()
        
        self.features = nn.Sequential(
            # Block 1
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Block 2
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
            
            # Block 3
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(256 * 28 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def get_resnet_model(num_classes=4, pretrained=True):
    """
    Get ResNet model for emotion recognition
    
    Args:
        num_classes: Number of emotion classes (default: 4)
        pretrained: Use ImageNet pretrained weights
    
    Returns:
        Modified ResNet model
    """
    model = models.resnet50(pretrained=pretrained)
    
    # Freeze early layers for transfer learning
    for param in list(model.parameters())[:-20]:
        param.requires_grad = False
    
    # Modify final layer for emotion classification
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.ReLU(),
        nn.Dropout(0.5),
        nn.Linear(512, num_classes)
    )
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
    
    epoch_loss = running_loss / len(train_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def validate(model, val_loader, criterion, device):
    """Validate the model"""
    model.eval()
    running_loss = 0.0
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    epoch_loss = running_loss / len(val_loader)
    epoch_acc = accuracy_score(all_labels, all_preds)
    
    return epoch_loss, epoch_acc


def evaluate_model(model, test_loader, device, class_names=['Angry', 'Happy', 'Neutral', 'Sad']):
    """Evaluate model and return comprehensive metrics"""
    model.eval()
    all_labels = []
    all_preds = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
    
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    cm = confusion_matrix(all_labels, all_preds)
    
    # Per-class metrics
    per_class_metrics = {}
    for i, class_name in enumerate(class_names):
        class_precision = precision_score(all_labels, all_preds, labels=[i], average='micro')
        class_recall = recall_score(all_labels, all_preds, labels=[i], average='micro')
        class_f1 = f1_score(all_labels, all_preds, labels=[i], average='micro')
        per_class_metrics[class_name] = {
            'precision': class_precision,
            'recall': class_recall,
            'f1_score': class_f1
        }
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'confusion_matrix': cm,
        'per_class_metrics': per_class_metrics
    }


def get_transforms():
    """Get data augmentation transforms"""
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.RandomErasing(p=0.5, scale=(0.02, 0.33)),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, test_transform


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(10, 8))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix')
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], 'd'),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black")
    
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    plt.show()


if __name__ == "__main__":
    # Example usage
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create model
    model = get_resnet_model(num_classes=4, pretrained=True)
    model = model.to(device)
    
    print("Emotion recognition model ready!")
    print(f"Total parameters: {sum(p.numel() for p in model.parameters())}")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
