from matplotlib import pyplot as plt
import numpy as np
from scipy import fft
from scipy.signal import decimate

def read_iq_file(filename, dtype=np.complex64):
    iq_data = np.fromfile(filename, dtype=dtype)
    return iq_data

data = read_iq_file("digicode/digicode_0.iq", np.complex64)
t = np.linspace(0, len(data), len(data))
plt.figure("spectogram")
plt.title("spectogram")
spectogram, _, _, _ = plt.specgram(data, NFFT=2**8, mode="magnitude")
plt.xlabel("Time")
plt.ylabel("Frequency")
data_moved = data * np.exp(1j * 2 * np.pi * -0.425/2 * t)

decimated = decimate(data_moved, 5)

plt.figure("spectogram decimated")
plt.title("spectogram decimated")
spectogram, _, _, _ = plt.specgram(decimated, NFFT=2**10, mode="magnitude")
plt.xlabel("Time")
plt.ylabel("Frequency")

plt.figure("time decimated")
plt.title("time decimated")
plt.plot(np.real(decimated))
plt.plot(np.imag(decimated))

# plt.figure("constellation")
# plt.title("constellation")
# data_for_constellation = data[11520:19240]

# plt.title("spectogram")
# spectogram, _, _, _ = plt.specgram(data_for_constellation, NFFT=2**8, mode="magnitude")
# plt.xlabel("Time")
# plt.ylabel("Frequency")

# plt.scatter(np.real(data_for_constellation), np.imag(data_for_constellation))
# plt.figure(" fft")
# plt.title(" fft")
# data_for_fft = data[11520:19240]
# data_fft = fft.fft(data_for_fft, n=1024)
# data_fft = fft.fftshift(data_fft)
# plt.plot(np.abs(data_fft)**2)
plt.show()
