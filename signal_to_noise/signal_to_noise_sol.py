import os
import sys
import numpy as np
from sigmf import SigMFFile, sigmffile
from scipy.signal import decimate
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter

script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)
DECIMATION_RATE = 4
NFFT = 1024 
# NFFT = 8192

# Read data
filename = 'signal_to_noise'
signal = sigmffile.fromfile(filename)
samples = signal.read_samples().view(np.complex64).flatten()
sample_rate = signal.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
sample_count = signal.sample_count
signal_duration = sample_count / sample_rate

# decimate less interesting parts
decimated_signal = decimate(samples, DECIMATION_RATE, ftype='fir')
plt.figure("spectogram")
all_samples_specgram, orig_f, orig_t, _ = plt.specgram(decimated_signal, NFFT=NFFT, Fs=sample_rate/DECIMATION_RATE, mode="magnitude")
plt.figure("spectogram flipped filtered")
plt.pcolormesh(orig_t[::-1], orig_f[::-1], gaussian_filter(np.fliplr(all_samples_specgram), sigma=5))
plt.xlabel("Time (s)")
plt.ylabel("Frequency (MHz)")
plt.show()

# Main Takes
# 1. The decimation was crucial for better analysis performance, knowing that the information is not stored
#    across the whole sampled spectrum. 
# 2. Using histogram equalization over segments in time didn't work and just amplified the noise.
# 3. The FFT size was the main key in the solution. For the first seconds, we needed a smaller NFFT
#    so the resolution in time is better  and the signal doesn't get lost in the noise. Because there,
#    the signal was mostly spread in frequency and very concentrated in time.
#    From a certain point the information started being concentrated in a shorter frequency band and thus
#    we needed a better frequency resolution -> bigger fft size.