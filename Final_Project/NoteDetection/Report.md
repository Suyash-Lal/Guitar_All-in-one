# Mid-Semester Project - Guitar Pitch Detection 

**Student Name**: Suyash Lal <br>
**Student ID**: 210477

This document provides an explanation for detecting pitch in 8-string electric guitar recordings. The code processes audio files, detects onsets, segments notes, and identifies the fundamental frequency to map it to a musical note. The onset detection and pitch analysis in this code is primarily a **heuristic approach**, but it incorporates algorithmic elements for specific tasks.


## Methodology

### **Why a Heuristic Approach?**
This project uses heuristic methods for the following reasons:
- **Fixed Constraints**: The problem involves a known set of notes with predefined frequency ranges (e.g., guitar tuning tables).
- **Efficiency**: Machine learning would require large datasets and computational resources, while heuristic rules leverage domain-specific knowledge (e.g., guitar note harmonics, sharp attacks).
- **Interpretability**: Heuristics allow precise tuning for guitar-specific challenges (e.g., low-frequency fundamentals, pick noise).


## 1. **Library Imports**
The code begins by importing necessary libraries:
- **`subprocess`**: Used to run system commands, specifically for `ffplay` audio playback.
- **`os`**: Handles file operations, such as creating directories for saving processed files.
- **`BytesIO`**: Enables in-memory streaming of audio data to avoid temporary file creation.
- **`pydub`**: A powerful library for audio processing, including loading, manipulating, and exporting audio files.
- **`matplotlib.pyplot`**: Used for visualizing audio waveforms and frequency spectra.
- **`numpy`**: Essential for numerical operations and signal processing.
- **`scipy.signal.find_peaks`**: Detects peaks in signals, pre-requisite for onset detection.


## 2. **Playback Fix (Critical for Windows)**
### Problem Solved:
The default `pydub.playback.play()` function creates temporary files for playback, which can cause permission errors on Windows. This fix avoids temporary files by streaming audio directly from memory using `ffplay`.

### How It Works:
1. **Audio Conversion**: The audio segment is exported to a WAV format and stored in a `BytesIO` buffer in memory.
2. **Streaming to `ffplay`**: The buffer is piped to `ffplay` via stdin, allowing playback without writing to disk.
3. **Flags**:
   - `-nodisp`: Disables the GUI for `ffplay`.
   - `-autoexit`: Automatically closes `ffplay` after playback.
   - `-hide_banner`: Suppresses version information for cleaner output.
4. **Monkey-Patching**: The custom playback function replaces `pydub`'s default `_play_with_ffplay` method.

This fix ensures seamless playback on any machine, regardless of file system permissions.


## 3. **Audio Loading & Preprocessing**
The audio file is loaded using `pydub`:
- **File Path**: The path to the WAV file is specified (e.g., `"Audio_Files\\E_8thString.wav"`).
- **Loading**: The file is loaded into an `AudioSegment` object using `AudioSegment.from_file()`.
- **Mono Conversion**: The audio is converted to mono using `file.set_channels(1)`. This simplifies processing and analysis, as stereo audio is not required for pitch detection.


## 4. **Onset Detection Pipeline**
This is where the approach is algorithmic.
### a. High-Pass Filtering
A high-pass filter is applied to remove low-frequency noise (e.g., rumble below 80Hz). This is particularly important for guitar recordings, as low-frequency noise can interfere with onset detection. The cutoff frequency is set to 80Hz, which is suitable for isolating the fundamental frequencies of guitar notes.

### b. Spectral Flux Calculation
**Definition**: Spectral flux measures how the power spectrum of an audio signal changes over time. It helps detect sudden changes in frequency content, which often indicate the beginning of a new note or sound event.

**What is spectral leakage?**: Spectral leakage occurs (in the context of Fast Fourier Transform (FFT)) when a signal is not perfectly periodic within the chosen window length. This happens because the FFT assumes the signal repeats continuously, and if the window does not capture an integer number of periods, discontinuities arise at the boundaries.

**How It Works**:
1. **Frame Splitting**: The audio signal is split into overlapping frames (1024 samples per frame, with a hop size of 512 samples).
2. **Windowing**: Each frame is multiplied by a **Hann window** to reduce spectral leakage.  
   - The Hann window is defined as:  
     $$ w(n) = 0.5 \left( 1 - \cos \left( \frac{2\pi n}{N-1} \right) \right) $$  
     where \( N \) is the frame size and \( n \) is the sample index.  
   - **Why Use a Hann Window?**  
     - Reduces spectral leakage caused by abrupt frame cuts.  
     - Tapers the frame’s edges to zero, minimizing discontinuities.  
3. **FFT Processing**: The Fast Fourier Transform (FFT) is applied to each frame to compute the magnitude spectrum.
4. **Flux Calculation**: The spectral flux is calculated as the squared difference between consecutive frames' magnitude spectra.  
5. **Normalization**:  
   The flux values are scaled between 0 and 1 using:  
   $$ \text{flux}_\text{normalized} = \frac{\text{flux} - \min(\text{flux})}{\max(\text{flux}) - \min(\text{flux})} $$  
   - **Purpose**:  
     - Standardizes the range for reliable peak detection.  
     - Reduces sensitivity to loudness variations.  

### c. Peak Picking
The `detect_onsets()` function identifies peaks in the spectral flux signal using `scipy.signal.find_peaks`. Key parameters:
- **Threshold**: Peaks must exceed a height of 0.3 (normalized flux value).
- **Distance**: Peaks must be at least 15 frames apart to avoid double-detections.

The detected peaks correspond to the onset times of notes in the audio.


## 5. **Audio Segmentation**
The `segment_peak_audio()` function extracts short segments of audio centered around the detected onsets. Each segment focuses on the attack portion of the note, which is critical for pitch perception. The function:
1. Converts onset times to sample indices.
2. Extracts a 100ms segment starting at each onset.
3. Creates a new `AudioSegment` object for each segment.

This segmentation isolates individual notes for further analysis.


## 6. **Frequency Analysis**
### a. FFT Configuration
The `analyze_guitar_frequency()` function performs frequency analysis on each segmented note:
1. **Zero-Padding**: The signal is padded to improve frequency resolution. The padding size is calculated as the next power of 2 multiplied by 4.
2. **Windowing**: A Hanning window is applied to the signal to reduce spectral leakage.
3. **FFT Processing**: The FFT is computed, and the magnitude spectrum is extracted.
4. **Frequency Range**: The analysis is limited to the range of 80Hz to 1200Hz, which covers the fundamental and harmonic frequencies of guitar notes.

### b. Fundamental Frequency Detection
The fundamental frequency is identified as the lowest strong peak in the magnitude spectrum. Peaks are detected using `find_peaks`, and the fundamental is selected as the lowest frequency among the significant peaks.

- The fundamental frequency (F0) - this is the lowest frequency and represents the actual note you're playing (guitar context).
- Harmonics or overtones - these are integer multiples of the fundamental (2×F0, 3×F0, 4×F0, etc.)


## 7. **Note Mapping**
The `frequency_to_note()` function maps the detected fundamental frequency to a musical note. It uses a reference table of guitar note frequencies, including the extended range of an 8-string guitar (e.g., E1=41Hz). The function:
1. Compares the detected frequency to the reference frequencies.
2. Finds the closest match using nearest-neighbor matching.
3. Returns the corresponding note name (e.g., "E").


## 8. **Visualization**
The code includes several visualization steps:
1. **Waveform Plot**: The audio waveform is plotted with detected onsets marked by vertical red lines.
2. **Segmented Notes**: Each segmented note is plotted individually to visualize its waveform.
3. **Frequency Spectrum**: The magnitude spectrum of a note is plotted, with detected peaks and the fundamental frequency highlighted.

These visualizations help verify the accuracy of the onset detection and frequency analysis.


## 9. **Exporting Processed Files**
The segmented notes are exported as individual WAV files:
1. An output directory (`processed_files`) is created if it doesn't already exist.
2. Each segmented note is saved as a separate file (e.g., `peak_note_1.wav`).


## Key Parameters to Tune
1. **High-Pass Cutoff**: Adjust based on the lowest expected note (e.g., 40Hz for 8-string guitars).
2. **Spectral Flux Threshold**: Controls sensitivity to onsets (higher values reduce false positives).
3. **Peak Duration**: The length of the segmented note (100ms works well for fast decays).
4. **FFT Resolution**: Increase zero-padding for better frequency resolution, especially for low notes.


## Why This Works for Guitars
1. **Onset Focus**: Guitar notes have sharp attacks, making them ideal for onset detection.
2. **Harmonic Handling**: The FFT effectively captures the strong harmonics of guitar notes.
3. **Noise Rejection**: The combination of high-pass filtering and spectral flux reduces interference from pick noise and other artifacts.


## Future Improvements
1. **MIDI Integration**: Convert detected notes to MIDI for use in music production.
2. **GUI**: Building an interface for direct interaction with the project would enhance user experience.
3. **Expansion of the project 1**: Next note prediction - Provide audio samples of 2-3 notes, and run a prediction model for all the notes possible to play after these notes.
4. **Expansion of the project 2**: Chord detection and feedback - Due to the thousands of chords that exist (and are still being named or created), I believe integrating Generative AI to the project in terms of naming the chord would future-proof the project. 

---