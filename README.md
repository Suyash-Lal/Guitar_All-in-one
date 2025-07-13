# Guitar All-in-One: Advanced Audio Analysis System

A comprehensive audio analysis system implementing multiple approaches to guitar note detection, chord recognition, and chord progression prediction. The project demonstrates three complementary methodologies: rule-based signal processing, CNN-based classification, and LSTM-Attention sequence modeling.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-1.9+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)

## Project Overview

This system provides a solution for guitar audio analysis, implementing three distinct approaches that demonstrate the evolution from interpretable signal processing to deep learning techniques. The system handles note detection, chord recognition, and musical progression prediction with high accuracy across all modules.

### Key Achievements
- **Rule-based Note Detection**: 100% accuracy using spectral analysis and music theory
- **Rule-based Chord Detection**: 100% accuracy with transparent music theory-based identification
- **CNN Chord Recognition**: 93.64% accuracy across 24 chord types with robust generalization
- **LSTM-Attention Progression Prediction**: 91.67% accuracy in predicting musical sequences

## System Architecture

### 1. Rule-Based Note and Chord Detection
A transparent system using digital signal processing and music theory principles for both individual note detection and chord recognition.

**Note Detection Features:**
- Fundamental frequency extraction using FFT analysis
- Equal temperament tuning calculations
- Real-time frequency-to-note mapping
- Multiple note detection capability
- Deployable note detection pipeline

**Chord Detection Features:**
- Spectral flux onset detection
- High-resolution FFT analysis (sub-Hz precision)
- Music theory-based chord template matching
- 1-3-5 harmonic pattern recognition
- Comprehensive visualization output

**Technical Highlights:**
- Zero-padded FFT for enhanced frequency resolution
- Adaptive thresholding for robust onset detection
- Perfect accuracy through deterministic music theory algorithms
- Transparent decision-making process

### 2. CNN-Based Recognition System
Deep learning model optimized for spectral pattern recognition.

**Architecture:**
- Specialized CNN for audio spectral data
- 36-bin chromagram feature extraction
- Harmonic-percussive separation
- Tonnetz harmonic relationship encoding
- Adaptive pooling for variable-length inputs

**Performance:**
- 93.64% test accuracy across 24 chord types
- Robust data augmentation (7x dataset expansion)
- Excellent generalization with minimal overfitting
- Strong per-chord F1 scores (>0.90 for most chords)

### 3. LSTM-Attention Progression Predictor
Sequential model capturing musical grammar and harmonic relationships.

**Architecture:**
- Bidirectional LSTM with attention mechanism
- Multi-head attention for diverse pattern recognition
- Enhanced feature extraction (chromagram, tonnetz, spectral contrast, MFCCs)
- Residual connections for improved gradient flow

**Capabilities:**
- Learns common cadential patterns (V-I, IV-I)
- Understands circle of fifths relationships
- Predicts next chord with 91.67% accuracy
- Provides confidence scoring for predictions

## Performance Metrics

| Model | Accuracy | F1 Score | Key Strengths |
|-------|----------|----------|---------------|
| Rule-Based Note Detection | 100% | 1.0 | Deterministic, transparent, real-time capable |
| Rule-Based Chord Detection | 100% | 1.0 | Music theory-based, interpretable, no training required |
| CNN Chord Recognition | 93.64% | 0.937 | High accuracy, robust to noise, fast inference |
| LSTM-Attention Progression | 91.67% | 0.889 | Sequential understanding, musical grammar |

## Quick Start

### Prerequisites
```bash
pip install torch torchvision torchaudio
pip install librosa numpy matplotlib scipy
pip install jupyter pandas scikit-learn
```

### Running the Models

1. **Note Detection**
   ```bash
   cd Final_Project/NoteDetection/
   jupyter notebook Note_Detection.ipynb
   # For multiple note detection
   jupyter notebook Multiple_Note_Detection.ipynb
   # For deployable version
   python Deployable-Note_Detection.py
   ```

2. **Rule-Based Chord Detection**
   ```bash
   cd Final_Project/ChordDetection/
   jupyter notebook RuleBased-Chord_Detection.ipynb
   ```

3. **CNN Chord Recognition**
   ```bash
   cd Final_Project/ChordDetection/
   jupyter notebook AlgBased-Chord_Detection.ipynb
   ```

4. **LSTM Chord Progression Prediction**
   ```bash
   cd Final_Project/NextChordPrediction/
   jupyter notebook ChordPred.ipynb
   ```

### Dataset Structure
```
Final_Project/
├── ChordDetection/
│   ├── Chords_WAV/           # Base audio dataset
│   ├── Augmented_Chords_WAV/ # Generated augmented data
│   └── *.ipynb               # Model implementations
├── NextChordPrediction/
│   ├── ChordMapping.json     # Chord label mappings
│   ├── ChordProgression.json # Progression sequences
│   └── ChordPred.ipynb       # LSTM model
└── NoteDetection/
    └── *.ipynb               # Additional note detection tools
```

## Technical Deep Dive

### Feature Engineering
- **High-resolution Chromagram**: 36 bins per octave for precise pitch detection
- **Harmonic-Percussive Separation**: Isolates chord content from transients
- **Tonnetz Representation**: Encodes triadic harmonic relationships
- **Spectral Contrast**: Captures tonal vs. noise energy ratios
- **Data Augmentation**: Pitch shifting, time stretching, noise addition

### Model Innovations
- **Adaptive Pooling**: Handles variable-length audio inputs
- **Multi-head Attention**: Captures diverse musical pattern types
- **Bidirectional Processing**: Considers both past and future musical context
- **Class Weighting**: Addresses dataset imbalance across chord types

## Results & Analysis

### Confusion Matrix Insights
- Primary confusion occurs between relative major/minor pairs
- Strong performance on tonally distant chords
- Model confidence correlates well with prediction accuracy

### Musical Theory Validation
- Successfully learns V-I and IV-I cadential patterns
- Recognizes deceptive cadences (V-vi progressions)
- Demonstrates understanding of circle of fifths relationships

## Applications

This project serves as an excellent resource for:
- **Music Theory Education**: Visualizing frequency relationships in chords
- **Audio Signal Processing**: Demonstrating spectral analysis techniques
- **Deep Learning for Audio**: Showcasing CNN and LSTM architectures
- **Research Methodology**: Comparing rule-based vs. learning-based approaches

## References & Documentation

- **Complete Technical Report**: [Final_Project/Final/SuyashLal-Final_Report.pdf](Final_Project/Final/SuyashLal-Final_Report.pdf)
- **Detailed Methodology**: Individual workflow documentation in each module
- **Architecture Diagrams**: Visual representations of model structures
- **Performance Analysis**: Comprehensive evaluation metrics and confusion matrices

## Contributing

This project demonstrates advanced techniques in audio analysis and can be extended for:
- Multi-instrument chord recognition
- Real-time audio processing
- Integration with Digital Audio Workstations (DAWs)
- Style-specific musical analysis

---

*A comprehensive audio analysis system demonstrating the intersection of signal processing, music theory, and deep learning techniques.*