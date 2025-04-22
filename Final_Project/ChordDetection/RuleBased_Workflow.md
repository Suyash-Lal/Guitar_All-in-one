# Guitar Chord Detection: Rule-Based Approach

## Project Overview
This project implements a rule-based system for identifying guitar chords from audio recordings by analyzing the frequency content and mapping detected notes to standard chord structures. Unlike machine learning approaches, this system uses music theory principles and spectral analysis techniques to identify chords.

## Technical Pipeline

### 1. Audio Preprocessing
The system starts with several critical preprocessing steps:
- **Conversion to Mono**: Simplifies analysis by eliminating channel differences
- **High-pass Filtering**: Removes low-frequency noise below 80Hz
- **Waveform Visualization**: Displays the time-domain representation of the audio signal

### 2. Onset Detection with Spectral Flux
The system identifies when a chord is struck using spectral flux analysis:
- **Frame-by-Frame Analysis**: Divides audio into overlapping frames (1024 samples)
- **Hann Window Application**: Reduces spectral leakage in frequency analysis
- **FFT Magnitude Calculation**: Computes spectral content of each frame
- **Flux Measurement**: Calculates changes between consecutive spectral frames
- **Peak Detection**: Identifies sudden changes marking chord onsets

### 3. Audio Segmentation
Once onsets are detected, the system:
- **Extracts Segments**: Captures 0.5 seconds following each detected onset
- **Creates WAV Files**: Saves individual chord segments for analysis
- **Handles Edge Cases**: Analyzes the entire audio if no clear onsets are detected

### 4. Frequency Analysis with FFT
For each chord segment, the system performs detailed frequency analysis:
- **Extended FFT**: Uses increased zero-padding for better frequency resolution
- **Windowing**: Applies Hann window to minimize spectral leakage
- **Peak Detection**: Identifies significant frequency components in 80-1200Hz range
- **Magnitude Sorting**: Ranks frequency peaks by their strength
- **Visualization**: Creates detailed spectral plots highlighting detected peaks

### 5. Note Identification
The system converts detected frequencies to musical notes:
- **Equal Temperament Formula**: Uses the standard tuning reference (A4 = 440Hz)
- **Frequency-to-Note Mapping**: Identifies the closest musical note for each peak
- **Top Note Selection**: Focuses on the 8 strongest frequency components

### 6. Chord Recognition with Music Theory
The chord identification process follows standard music theory principles:
- **1-3-5 Pattern Analysis**: Maps detected notes to the root-third-fifth pattern
- **Major vs. Minor**: Differentiates based on the third (4 vs. 3 semitones above root)
- **Template Matching**: Scores potential chords against theoretical patterns
- **Function Identification**: Maps notes to their harmonic function (root, third, fifth)
- **Best Match Selection**: Chooses the chord with the highest matching score

### 7. Visualization and Reporting
The system provides comprehensive visualizations:
- **Waveform with Onset Markers**: Shows detected chord starting points
- **Spectrum Analysis**: Displays frequency content with note annotations
- **Chord Structure Diagrams**: Illustrates the 1-3-5 relationship of detected notes
- **Detailed Results**: Presents chord name, type, and constituent notes

## Implementation Strengths
The rule-based approach offers several advantages:
- **Transparency**: Clear understanding of how each chord is identified
- **Music Theory Alignment**: Recognition based on established chord theory
- **No Training Data Required**: Works without machine learning training
- **Educational Value**: Demonstrates the frequency relationships in musical chords

This project demonstrates how digital signal processing techniques combined with music theory can effectively identify chords from audio recordings, providing both analytical results and educational insights into harmonic structures.