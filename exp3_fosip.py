import numpy as np
import matplotlib.pyplot as plt

x = [1, 2, 3, 4]

def compute_dft(signal):
    """Compute the Discrete Fourier Transform (DFT) of a signal."""
    N = len(signal)
    X = np.zeros(N, dtype=complex)
    
    # Compute the DFT
    for k in range(N):
        sum_value = 0
        for n in range(N):
            angle = 2 * np.pi * k * n / N
            sum_value += signal[n] * (np.cos(angle) - 1j * np.sin(angle))
        X[k] = sum_value
    
    return X

def plot_magnitude_spectrum(signal, title):
    """Plot the magnitude spectrum of the signal."""
    X = compute_dft(signal)
    magnitude = np.abs(X)
    magnitude = np.append(magnitude, magnitude[0])
    
    
    plt.figure(figsize=(8, 4))
    plt.plot(magnitude, 'o-', color='orange')
    plt.title(title)
    plt.xlabel('Frequency (k)')
    plt.ylabel('Magnitude')

    for i, mag in enumerate(magnitude):
        plt.text(i, mag + 0.5, f'{mag:.2f}', ha='center', va='bottom')

    plt.ylim(0, 12)
    plt.xlim(0, len(signal) + 2)
    plt.grid()
    plt.show()

plot_magnitude_spectrum(x, 'Magnitude Spectrum of Original Signal')

x_padded = np.append(x, [0, 0, 0, 0])
plot_magnitude_spectrum(x_padded, 'Magnitude Spectrum after Zero Padding')

x_expanded = np.zeros(len(x) * 2)
x_expanded[::2] = x
plot_magnitude_spectrum(x_expanded, 'Magnitude Spectrum after Alternate Zero Insertion')

