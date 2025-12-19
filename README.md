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
### Emotion Recognition
- Architecture: ResNet50 / Custom CNN
- 4-class Classification (Angry, Happy, Neutral, Sad)

## 🔧 Key Features

- **Transfer Learning**: Leveraging pretrained models for better performance
- **Data Augmentation**: Comprehensive augmentation for robust training
- **Modular Design**: Easy to integrate and extend
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix

