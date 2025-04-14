# DTSC422: Natural Language Processing Final Project

## Methodology

### Using a heuristic approach:
Since this in the area of pitch detection and audio processing, heuristic methods would be more viable.
As this project has a fixed set of constraints (number of notes, frequency already given per note, matchability
of notes to frequency ranges, etc.), using a machine learning algorithm would be less efficient and heuristical algorithms would be utilized better.

## Onset Detection

### **1. Spectral Flux**

**Definition:** Spectral flux measures how the power spectrum of an audio signal changes over time. It helps detect sudden changes in frequency content, which often indicate the beginning of a new note or sound event.

**How It Works:**
- The magnitude spectrum (Fourier Transform without phase) is computed for each short frame of the audio.
- The squared difference between consecutive magnitude spectra is calculated.
- A large change (or peak) in spectral flux suggests an onset, such as a note being played.

Since musical notes often create significant changes in the frequency spectrum, spectral flux is an effective feature for onset detection.

### **2. Hann Window**

**Definition:** The Hann window is a smoothing function applied to each audio frame before computing the Fourier Transform. It is defined as:

$$ w(n) = 0.5 \left( 1 - \cos \left( \frac{2\pi n}{N-1} \right) \right) $$

where:
- \( N \) is the frame size.
- \( n \) is the sample index.

**Why Use a Hann Window?**
- **Reduces Spectral Leakage:** Fourier Transform assumes a periodic signal. If the frame is cut abruptly, it creates sharp discontinuities, leading to false frequencies in the spectrum.
- **Smooth Transitions:** The Hann window tapers the frame’s edges to zero, reducing these discontinuities and improving frequency resolution.

By applying a Hann window to each frame, we ensure that the onset detection algorithm focuses on actual spectral changes rather than artifacts caused by windowing.

### **3. Normalization of Spectral Flux**

**Definition:** Normalization scales the spectral flux values between 0 and 1 using the formula:

$$ \text{flux}_\text{normalized} = \frac{\text{flux} - \min(\text{flux})}{\max(\text{flux}) - \min(\text{flux})} $$

**Purpose of Normalization:**
- **Standardizes the Range:** Ensures all values fall within a consistent scale (0 to 1), making peak detection more reliable.
- **Improves Comparability:** Helps in setting a meaningful onset threshold, as values become independent of the absolute magnitude of the original signal.
- **Reduces Sensitivity to Loudness:** Without normalization, louder recordings would produce larger spectral flux values, making onset detection inconsistent across different audio files.
