import os
from typing import Any
from attr import define
import numpy as np
from scipy import signal
from sigmf import SigMFFile, sigmffile
from scipy.signal import decimate
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy import fft
import numpy.typing as npt
from qpsk_demod import qpsk_demod

script_directory = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_directory)

@define
class ChannelSignal:
    samples: npt.NDArray
    fs: float

def extract_channel_by_center_and_dec_rate(samples: npt.NDArray[np.complex64],
                                           center_freq: float,
                                           dec_rate: int,
                                           fs: float,
                                           signal_duration: float,
                                           ftype: str) -> ChannelSignal:
    freq_to_shift = -center_freq
    t = np.arange(0, signal_duration, 1/fs)
    data_shifted = samples * np.exp(1j * 2 * np.pi * freq_to_shift * t)
    channelized_signal = signal.decimate(data_shifted, dec_rate, ftype=ftype)

    return ChannelSignal(channelized_signal, fs/dec_rate)

def read_iq_file(filename, dtype=np.complex64):
    iq_data = np.fromfile(filename, dtype=dtype)
    return iq_data

def save_iq_file(data: npt.NDArray[np.complex64], file_name: str):
    data.tofile(file_name)

def extract_and_save_channel_signals(samples: npt.NDArray[np.complex64], sample_rate: float, signal_duration: float):
    """Save to iq files the signals across the channels.
       The parameters of the center freq and decimation rate were extracted visually by looking at the spectogram.
    """
    sig0 = extract_channel_by_center_and_dec_rate(samples=samples,
                                                  center_freq=102.5e3,
                                                  dec_rate=5,
                                                  fs=sample_rate,
                                                  signal_duration = signal_duration,
                                                  ftype="fir")
    save_iq_file(sig0.samples.astype(np.complex64), "sig0_fs_" + str(sig0.fs) + ".iq")
    sig1 = extract_channel_by_center_and_dec_rate(samples=samples,
                                                  center_freq=300e3,
                                                  dec_rate=20,
                                                  fs=sample_rate,
                                                  signal_duration=signal_duration,
                                                  ftype="fir")
    save_iq_file(sig1.samples.astype(np.complex64), "sig1_fs_" + str(sig1.fs) + ".iq")

    sig2 = extract_channel_by_center_and_dec_rate(samples=samples,
                                                  center_freq=400e3,
                                                  dec_rate=12,
                                                  fs=sample_rate,
                                                  signal_duration=signal_duration,
                                                  ftype="fir")
    save_iq_file(sig2.samples.astype(np.complex64), "sig2_fs_" + str(sig2.fs) + ".iq")
    sig3 = extract_channel_by_center_and_dec_rate(samples=samples,
                                                  center_freq=-400e3,
                                                  dec_rate=24,
                                                  fs=sample_rate,
                                                  signal_duration=signal_duration,
                                                  ftype="fir")
    save_iq_file(sig3.samples.astype(np.complex64), "sig3_fs_" + str(sig3.fs) + ".iq")
    sig4 = extract_channel_by_center_and_dec_rate(samples=samples,
                                                  center_freq=-300e3,
                                                  dec_rate=48,  # TODO check if do 24
                                                  fs=sample_rate,
                                                  signal_duration=signal_duration,
                                                  ftype="fir")
    save_iq_file(sig4.samples.astype(np.complex64), "sig4_fs_" + str(sig4.fs) + ".iq")
    sig5 = extract_channel_by_center_and_dec_rate(samples=samples,
                                                  center_freq=-200e3,
                                                  dec_rate=48,  # TODO check if do 24
                                                  fs=sample_rate,
                                                  signal_duration=signal_duration,
                                                  ftype="fir")
    save_iq_file(sig5.samples.astype(np.complex64), "sig5_fs_" + str(sig5.fs) + ".iq")
    sig6 = extract_channel_by_center_and_dec_rate(samples=samples,
                                                  center_freq=-100e3,
                                                  dec_rate=48,  # TODO check if do 24
                                                  fs=sample_rate,
                                                  signal_duration=signal_duration,
                                                  ftype="fir")
    save_iq_file(sig6.samples.astype(np.complex64), "sig6_fs_" + str(sig6.fs) + ".iq")

def analyze_sig0():
    sig0 = ChannelSignal(samples=read_iq_file("sig0_fs_192000.0.iq"),
                         fs=192e3)
    freq_shift = 2500
    samples, fs = sig0.samples, sig0.fs
    samples = samples
    t = np.arange(0, len(samples)/fs, 1/fs)
    data_shifted = samples * np.exp(1j * 2 * np.pi * freq_shift * t)

    # plt.figure()
    # plt.title("Amp vs Time")
    # plt.plot(t, np.real(data_shifted))
    # plt.plot(t, np.imag(data_shifted))
    # plt.xlabel("Time (sec)")
    # plt.ylabel("Amp")

    # plt.figure()
    # plt.title("phase")
    # plt.scatter(t, np.angle(data_shifted))
    # plt.xlabel("Time (sec)")
    # plt.ylabel("phase")

    # plt.figure()
    # plt.title("Spectogram of shifted signal (Mag vs freq vs time)")
    # NFFT = 256
    # plt.specgram(data_shifted, NFFT=NFFT, mode="magnitude", Fs=fs)
    # plt.xlabel("Time (sec)")
    # plt.ylabel("Frequency (Hz)")

    # plt.figure()
    # plt.title("Constellation")
    # plt.scatter(np.real(samples), np.imag(samples))
    # plt.xlabel("real")
    # plt.ylabel("imag")

    plt.show()

def analyze_sig1():
    sig1 = ChannelSignal(samples=read_iq_file("sig1_fs_48000.0.iq"),
                         fs=48e3)
    # freq_shift = 2500
    samples, fs = sig1.samples, sig1.fs
    t = np.arange(0, len(samples)/fs, 1/fs)

    a_hat, x, y = qpsk_demod(samples, 5000, 40)

    plt.figure()
    plt.title("Spectogram of signal (Mag vs freq vs time)")
    NFFT = 8192
    plt.specgram(samples, NFFT=NFFT, mode="magnitude", Fs=fs)
    plt.xlabel("Time (sec)")
    plt.ylabel("Frequency (Hz)")

    plt.figure()
    plt.title("QPSK demod")
    plt.plot(a_hat)
    plt.xlabel("Time (sec)")
    plt.ylabel("Symbol")

    plt.figure()
    plt.title("Signal in Time")
    plt.plot(t, np.real(samples))
    plt.plot(t, np.imag(samples))
    plt.xlabel("Time (sec)")
    plt.ylabel("Mag")
    # TODO: figure out how to make a good QPSK demodulator and figure out the symbols and bits
    plt.figure()
    plt.title("Phase")
    plt.plot(t, np.arctan2(np.imag(samples), np.real(samples)))
    plt.xlabel("Time (sec)")
    plt.ylabel("Phase")

    plt.figure()
    plt.title("FFT")
    plt.plot(np.abs(fft.fftshift(fft.fft(samples[int(fs*29.9):int(fs*30.5)])))**2)
    plt.xlabel("Freq (bin)")
    plt.ylabel("Amp")

    # plt.figure()
    # plt.title("Constellation Diagram")
    # plt.scatter(np.real(samples), np.imag(samples))
    # plt.xlabel("Real")
    # plt.ylabel("Imag")

    plt.show()

def analyze_sig2():
    sig2 = ChannelSignal(samples=read_iq_file("sig2_fs_80000.0.iq", dtype=np.complex64),
                         fs=80e3)
    # freq_shift = 2500
    samples, fs = sig2.samples, sig2.fs
    t = np.arange(0, len(samples)/fs, 1/fs)

    plt.figure()
    plt.title("Spectogram of signal (Mag vs freq vs time)")
    NFFT = 32
    plt.specgram(samples, NFFT=NFFT, mode="magnitude", Fs=fs, noverlap=30)
    plt.xlabel("Time (sec)")
    plt.ylabel("Frequency (Hz)")
    plt.show()

def analyze_sig3():
    sig3 = ChannelSignal(samples=read_iq_file("sig3_fs_40000.0.iq", dtype=np.complex64),
                         fs=40e3)
    # freq_shift = 2500
    samples, fs = sig3.samples, sig3.fs
    t = np.arange(0, len(samples)/fs, 1/fs)
    # data_shifted = samples * np.exp(1j * 2 * np.pi * freq_shift * t)

    # plt.figure()
    # plt.title("Amp vs Time")
    # plt.plot(t, np.real(data_shifted))
    # plt.plot(t, np.imag(data_shifted))
    # plt.xlabel("Time (sec)")
    # plt.ylabel("Amp")

    plt.figure()
    plt.title("Spectogram of signal (Mag vs freq vs time)")
    NFFT = 256
    plt.specgram(samples, NFFT=NFFT, mode="magnitude", Fs=fs)
    plt.xlabel("Time (sec)")
    plt.ylabel("Frequency (Hz)")
    plt.show()

def quad_demod(iq_signal: npt.NDArray, fs: float, fsk_freq: float):
    demodulated_phase = np.arctan2(np.real(iq_signal), np.imag(iq_signal))

    # Differentiate phase to get the instantaneous frequency (frequency demodulation)
    demodulated_signal = np.diff(demodulated_phase) * fs / (2 * np.pi) / fsk_freq
    return demodulated_signal

def main():

    ### Read the sigMF original data. Uncomment this section to load it

    # filename = 'sigid'
    # sig = sigmffile.fromfile(filename)
    # samples = sig.read_samples().view(np.complex64).flatten()
    # sample_rate = sig.get_global_field(SigMFFile.SAMPLE_RATE_KEY)
    # sample_count = sig.sample_count
    # signal_duration = sample_count / sample_rate

    ### Uncomment this part to perform extraction and save the signals across the different channels (like a channelizer)

    # extract_and_save_channel_signals(samples=samples,
    #                                  sample_rate=sample_rate,
    #                                  signal_duration=signal_duration)

    ### Implement and call analyze_sigx() to try demodulating the signals after channelizing.
    
    # analyze_sig0()
    # analyze_sig1()  # fsk suspected
    analyze_sig2()  # morse code (challenge 8)
    # analyze_sig3()


    ### plot wide band spectogram
    # plt.figure()
    # plt.title("")
    # NFFT=256
    # plt.specgram(samples, NFFT=NFFT, mode="magnitude", Fs=sample_rate)
    # plt.xlabel("Time")
    # plt.ylabel("Frequency")
    # plt.show()

if __name__ == "__main__":
    main()