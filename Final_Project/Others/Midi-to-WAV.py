import sys
from midi2audio import FluidSynth

def convert_midi_to_wav(midi_path, wav_path, soundfont_path=None):
    """
    Convert a MIDI file to WAV format
    
    Args:
        midi_path (str): Path to the input MIDI file
        wav_path (str): Path to save the output WAV file
        soundfont_path (str, optional): Path to a soundfont file. If None, uses the default soundfont.
    """
    # Create a FluidSynth instance
    # If soundfont_path is provided, use it; otherwise FluidSynth will try to use the system default
    fs = FluidSynth(sound_font=soundfont_path) if soundfont_path else FluidSynth()
    
    # Convert the MIDI file to WAV
    fs.midi_to_audio(midi_path, wav_path)
    print(f"Successfully converted {midi_path} to {wav_path}")

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python midi_to_wav.py input.mid output.wav [soundfont.sf2]")
        sys.exit(1)
    
    midi_file = sys.argv[1]
    wav_file = sys.argv[2]
    soundfont = sys.argv[3] if len(sys.argv) > 3 else None
    
    convert_midi_to_wav(midi_file, wav_file, soundfont)