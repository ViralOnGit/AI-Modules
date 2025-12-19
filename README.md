# AI Modules - Deep Learning Projects

A collection of production-ready AI modules for computer vision tasks, focusing on face recognition and emotion detection using transfer learning and deep neural networks.

## 📁 Project Structure

```
AI-Modules/
├── face_recognition/
│   └── face_verification.py    # VGGFace-based face verification module
├── emotion_recognition/
│   └── emotion_classifier.py   # Multi-class emotion recognition
├── requirements.txt
└── README.md
```

## 🚀 Modules

### 1. Face Verification (VGGFace Transfer Learning)
Binary classification module for face verification using pretrained VGGFace architecture.

**Features:**
- Transfer learning with VGGFace (pretrained on VGGFace2 dataset)
- Data augmentation for robust training
- Binary classification (your_face vs not_your_face)
- Comprehensive evaluation metrics

**Key Components:**
- Custom `FaceDataset` class for data loading
- VGGFace architecture implementation
- Training, validation, and evaluation functions
- Configurable data augmentation pipeline

### 2. Emotion Recognition
Multi-class emotion classifier for detecting 4 emotions: Angry, Happy, Neutral, Sad.

**Features:**
- ResNet50 transfer learning with ImageNet weights
- Custom CNN architecture option
- Optional MTCNN face detection integration
- Advanced data augmentation techniques
- Per-class and overall performance metrics
- Confusion matrix visualization

**Key Components:**
- `EmotionDataset` with optional face detection
- Custom EmotionCNN architecture
- ResNet-based transfer learning model
- Comprehensive evaluation and visualization tools

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/your-username/AI-Modules.git
cd AI-Modules

# Install dependencies
pip install -r requirements.txt
```

## 📦 Requirements

- Python 3.8+
- PyTorch 1.10+
- torchvision
- numpy
- Pillow
- scikit-learn
- matplotlib
- opencv-python (optional, for face detection)
- mtcnn (optional, for face detection in emotion module)

## 💻 Usage

### Face Verification

```python
from face_recognition.face_verification import (
    get_vggface_model, FaceDataset, get_transforms,
    train_epoch, validate, evaluate_model
)
import torch
from torch.utils.data import DataLoader

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_transform, test_transform = get_transforms()

# Load data
train_dataset = FaceDataset('path/to/train', transform=train_transform)
test_dataset = FaceDataset('path/to/test', transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Create model
model = get_vggface_model(num_classes=2, pretrained_path='vgg_face_dag.pth')
model = model.to(device)

# Training
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")

# Evaluation
metrics = evaluate_model(model, test_loader, device)
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"F1 Score: {metrics['f1_score']:.4f}")
```

### Emotion Recognition

```python
from emotion_recognition.emotion_classifier import (
    get_resnet_model, EmotionDataset, get_transforms,
    train_epoch, validate, evaluate_model, plot_confusion_matrix
)
import torch
from torch.utils.data import DataLoader

# Setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
train_transform, test_transform = get_transforms()

# Load data
train_dataset = EmotionDataset('path/to/train', transform=train_transform)
test_dataset = EmotionDataset('path/to/test', transform=test_transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# Create model
model = get_resnet_model(num_classes=4, pretrained=True)
model = model.to(device)

# Training
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(20):
    train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
    print(f"Epoch {epoch+1}: Loss={train_loss:.4f}, Acc={train_acc:.4f}")

# Evaluation
metrics = evaluate_model(model, test_loader, device)
print(f"Test Accuracy: {metrics['accuracy']:.4f}")
print(f"Weighted F1 Score: {metrics['f1_score']:.4f}")

# Plot confusion matrix
plot_confusion_matrix(metrics['confusion_matrix'], 
                     ['Angry', 'Happy', 'Neutral', 'Sad'])
```

## 📊 Dataset Structure

### Face Verification
```
dataset/
├── train/
│   ├── your_face/
│   │   ├── img1.jpg
│   │   └── ...
│   └── not_your_face/
│       ├── img1.jpg
│       └── ...
└── test/
    ├── your_face/
    └── not_your_face/
```

### Emotion Recognition
```
dataset/
├── train/
│   ├── Angry/
│   ├── Happy/
│   ├── Neutral/
│   └── Sad/
└── test/
    ├── Angry/
    ├── Happy/
    ├── Neutral/
    └── Sad/
```

## 🎯 Model Performance

### Face Verification
- Architecture: VGGFace with transfer learning
- Binary Classification
- Data Augmentation: RandomFlip, Rotation, ColorJitter, GaussianBlur, RandomErasing

### Emotion Recognition
- Architecture: ResNet50 / Custom CNN
- 4-class Classification (Angry, Happy, Neutral, Sad)
- Advanced Data Augmentation Pipeline
- Optional MTCNN face detection preprocessing

## 🔧 Key Features

- **Transfer Learning**: Leveraging pretrained models for better performance
- **Data Augmentation**: Comprehensive augmentation for robust training
- **Modular Design**: Easy to integrate and extend
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **GPU Support**: CUDA-enabled for faster training
- **Production Ready**: Clean, documented, and reusable code

## 📝 License

This project is available for educational and research purposes.

## 🤝 Contributing

Feel free to open issues or submit pull requests for improvements.

## 📧 Contact

For questions or collaboration, please open an issue on GitHub.

---

**Built with PyTorch** 🔥
