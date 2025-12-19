# AI Modules - Deep Learning & Machine Learning Projects

A collection of production-ready AI modules for computer vision, natural language processing, and video compression tasks using deep learning, transfer learning, and classical ML techniques.

##  Project Structure

\\\
AI-Modules/
 face_recognition/
    face_verification.py    # VGGFace-based face verification module
 emotion_recognition/
    emotion_classifier.py   # Multi-class emotion recognition
 ngram_language_model/
    ngram.py                # N-gram language model implementation
    user_interface.py       # Interactive terminal UI
    reports/
        Report_2.pdf        # Project report
 eigenface_video_compression/
    reports/
        Report_1.pdf        # Eigenface PCA compression report
 requirements.txt
 README.md
\\\

##  Modules

### 1. Face Verification (VGGFace Transfer Learning)
Binary classification module for face verification using pretrained VGGFace architecture.

**Features:**
- Transfer learning with VGGFace (pretrained on VGGFace2 dataset)
- Data augmentation for robust training
- Binary classification (your_face vs not_your_face)
- Comprehensive evaluation metrics

**Key Components:**
- Custom \FaceDataset\ class for data loading
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
- \EmotionDataset\ with optional face detection
- Custom EmotionCNN architecture
- ResNet-based transfer learning model
- Comprehensive evaluation and visualization tools

### 3. N-gram Language Model
Character-level n-gram language model for text prediction and word completion.

**Features:**
- Character-level n-gram modeling with configurable n-gram size
- Word prediction and auto-completion
- Terminal-based user interface with real-time suggestions
- Training on custom text corpora

**Key Components:**
- Core n-gram probability calculations
- Interactive terminal UI with auto-suggestions
- Real-time word completion
- Probability-based word ranking

**Usage:**
\\\ash
python user_interface.py <path_to_training_corpus> [--auto]
\\\

### 4. Eigenface-based Video Compression (PCA)
Video compression implementation using Principal Component Analysis and eigenface techniques.

**Features:**
- PCA-based dimensionality reduction
- Eigenface method for video compression
- Visual quality preservation with reduced data size

##  Installation

\\\ash
# Clone the repository
git clone https://github.com/ViralOnGit/AI-Modules.git
cd AI-Modules

# Install dependencies
pip install -r requirements.txt
\\\

##  Dataset Structure

### Face Verification
\\\
dataset/
 train/
    your_face/
       img1.jpg
       ...
    not_your_face/
        img1.jpg
        ...
 test/
     your_face/
     not_your_face/
\\\

### Emotion Recognition
\\\
dataset/
 train/
    Angry/
    Happy/
    Neutral/
    Sad/
 test/
     Angry/
     Happy/
     Neutral/
     Sad/
\\\

##  Model Performance

### Face Verification
- Architecture: VGGFace with transfer learning
- Binary Classification

### Emotion Recognition
- Architecture: ResNet50 / Custom CNN
- 4-class Classification (Angry, Happy, Neutral, Sad)

### N-gram Language Model
- Character-level probability modeling
- Real-time text prediction and auto-completion

### Eigenface Video Compression
- PCA-based dimensionality reduction
- Efficient video compression with quality preservation

##  Key Features

- **Transfer Learning**: Leveraging pretrained models for better performance
- **Data Augmentation**: Comprehensive augmentation for robust training
- **Modular Design**: Easy to integrate and extend
- **Comprehensive Metrics**: Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **NLP Capabilities**: N-gram language modeling and text prediction
- **Compression Techniques**: PCA-based video compression using eigenface methods
