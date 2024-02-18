import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, periodogram
from scipy.integrate import simpson

eeg_data = np.loadtxt('eeg_data.txt')

sampling_freq = 100

frequencies_welch, psd_welch = welch(eeg_data, fs=sampling_freq, nperseg=sampling_freq*2)

plt.figure(figsize=(10, 4))
plt.plot(frequencies_welch, psd_welch)
plt.xlabel('Frequency (Hz)')
plt.ylabel('PSD (V^2/Hz)')
plt.title('Power Spectral Density')
plt.grid(True)
plt.show()

frequency_bands = {
    'Delta': (1, 4),
    'Theta': (4, 8),
    'Alpha': (8, 13),
    'Beta': (13, 30)
}

absolute_band_powers = {}

for band, (low_freq, high_freq) in frequency_bands.items():
    band_indices = np.where((frequencies_welch >= low_freq) & (frequencies_welch <= high_freq))[0]
    band_power = simpson(y=psd_welch[band_indices], x=frequencies_welch[band_indices])
    absolute_band_powers[band] = band_power

total_power = simpson(y=psd_welch, x=frequencies_welch)

relative_band_powers = {band: band_power / total_power for band, band_power in absolute_band_powers.items()}

print("Absolute band powers (Welch):")
for band, band_power in absolute_band_powers.items():
    print(f"{band}: {band_power}")

print("\nRelative band powers (Welch):")
for band, band_power in relative_band_powers.items():
    print(f"{band}: {band_power * 100:.2f}%")

_, psd_mt = periodogram(eeg_data, fs=sampling_freq, window='hamming', nfft=sampling_freq*2)

absolute_band_powers_mt = {}

for band, (low_freq, high_freq) in frequency_bands.items():
    band_indices = np.where((frequencies_welch >= low_freq) & (frequencies_welch <= high_freq))[0]
    band_power = simpson(y=psd_mt[band_indices], x=frequencies_welch[band_indices])
    absolute_band_powers_mt[band] = band_power

total_power_mt = simpson(y=psd_mt, x=frequencies_welch)

relative_band_powers_mt = {band: band_power / total_power_mt for band, band_power in absolute_band_powers_mt.items()}

print("\nAbsolute band powers (Multitaper):")
for band, band_power in absolute_band_powers_mt.items():
    print(f"{band}: {band_power}")

print("\nRelative band powers (Multitaper):")
for band, band_power in relative_band_powers_mt.items():
    print(f"{band}: {band_power * 100:.2f}%")