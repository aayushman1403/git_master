import numpy as np
import matplotlib.pyplot as plt
import wave
def compute_dft(signal):
    """Compute the Discrete Fourier Transform (DFT) of the signal."""
    N = len(signal)
    X = np.zeros(N, dtype=complex)
    
    for k in range(N):
        sum_value = 0
        for n in range(N):
            angle = 2 * np.pi * k * n / N
            sum_value += signal[n] * (np.cos(angle) - 1j * np.sin(angle))
        X[k] = sum_value

    return X

def load_and_preprocess_audio(file, max_samples=None):
    """Load audio from WAV file and preprocess it."""
    with wave.open(file, 'rb') as wav_file:
        n_channels = wav_file.getnchannels()
        fs = wav_file.getframerate()
        n_frames = wav_file.getnframes()
        audio_data = wav_file.readframes(n_frames)

        # Convert audio data to numpy array
        samples = np.frombuffer(audio_data, dtype=np.int16)

        # If stereo, take only one channel (mono)
        if n_channels == 2:
            samples = samples[::2]

        # Limit the number of samples if specified
        if max_samples and len(samples) > max_samples:
            samples = samples[:max_samples]

    return samples, fs

def normalize_signal(signal):
    """Normalize the audio signal to the range [-1, 1]."""
    max_val = np.max(np.abs(signal))
    return signal / max_val if max_val != 0 else signal

def plot_signal(signal, fs, title="Time-Domain Signal"):
    """Plot the signal in the time domain."""
    plt.figure(figsize=(10, 4))
    plt.plot(np.arange(len(signal)) / fs, signal)
    plt.title(title)
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid()
    plt.show()

def plot_magnitude_spectrum(signal, fs):
    """Plot the magnitude spectrum of the signal."""
    dft_result = compute_dft(signal)
    magnitude_spectrum = np.abs(dft_result)

    # Get corresponding frequencies for the DFT result
    frequencies = np.fft.fftfreq(len(dft_result), 1 / fs)

    plt.figure(figsize=(10, 4))
    plt.plot(frequencies[:len(frequencies) // 2],
             magnitude_spectrum[:len(magnitude_spectrum) // 2])
    plt.title('Magnitude Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.grid()
    plt.show()

def spectral_analysis():
    """Perform spectral analysis on an audio file."""
    audio_file_path = input("Enter the path to the audio WAV file: ")
    max_samples = 2048  # Limit to first 2048 samples for analysis

    # Load and normalize audio signal
    signal, fs = load_and_preprocess_audio(audio_file_path, max_samples)
    normalized_signal = normalize_signal(signal)

    # Plot time-domain signal and magnitude spectrum
    plot_signal(normalized_signal, fs, "Normalized Time-Domain Signal")
    plot_magnitude_spectrum(normalized_signal, fs)

# Run the spectral analysis
spectral_analysis()
