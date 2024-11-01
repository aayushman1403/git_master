import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal
import soundfile as sf
from google.colab import files
from IPython.display import Audio, display

def design_fir_filter(Fpass, Fstop, Fs, N=101):
    nyquist = 0.5 * Fs
    low = Fpass / nyquist
    high = Fstop / nyquist
    fir_coeff = signal.firwin(N, cutoff=[low, high], pass_zero=False)
    return fir_coeff

def filter_audio(audio_signal, fir_coeff):
    filtered_signal = signal.lfilter(fir_coeff, 1.0, audio_signal)
    return filtered_signal

# Upload the file
uploaded = files.upload()
filename = list(uploaded.keys())[0]
wav_filename = 'noise.wav'

# Convert uploaded file to .wav format using ffmpeg
!ffmpeg -i "{filename}" "{wav_filename}"

# Read the audio file
audio_signal, sample_rate = sf.read(wav_filename)

# Filter specifications
Fpass = 4000  # Passband frequency (Hz)
Fstop = 6000  # Stopband frequency (Hz)
Fs = 44100    # Sampling rate (Hz)
N = 101       # Filter order

# Design the FIR filter
fir_coeff = design_fir_filter(Fpass, Fstop, Fs, N)

# Apply the filter to the audio signal
filtered_signal = filter_audio(audio_signal, fir_coeff)

# Save the filtered output
output_filename = 'filtered_output.wav'
sf.write(output_filename, filtered_signal, sample_rate)

# Download the filtered audio file
files.download(output_filename)

# Display original audio
print("Original audio:")
display(Audio(filename=wav_filename))

# Display filtered audio
print("Filtered audio:")
display(Audio(filename=output_filename))

# Plot the original and filtered signals
plt.figure(figsize=(12, 6))

plt.subplot(2, 1, 1)
plt.plot(audio_signal)
plt.title('Original Audio Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(filtered_signal)
plt.title('Filtered Audio Signal')
plt.xlabel('Sample')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

