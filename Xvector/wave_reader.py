import librosa
import numpy as np
from scipy.signal import lfilter, butter
import copy

from Xvector import sigproc
#import constants as c

SAMPLE_RATE = 16000
NUM_FFT = 512
FRAME_STEP = 0.01
FRAME_LEN = 0.025
PREEMPHASIS_ALPHA = 0.97


def load_wav(filename, sample_rate):
    audio, sr = librosa.load(filename, sr=sample_rate, mono=True)
    audio = audio.flatten()
    return audio


def normalize_frames(m, epsilon=1e-12):
    b = (m - np.mean(m, axis=0)) / np.maximum(np.std(m, axis=0), epsilon)
    return b

# https://github.com/christianvazquez7/ivector/blob/master/MSRIT/rm_dc_n_dither.m


def remove_dc_and_dither(sin, sample_rate):
    alpha = None
    if sample_rate == 16e3:
        alpha = 0.99
    elif sample_rate == 8e3:
        alpha = 0.999
    else:
        print("Sample rate must be 16kHz or 8kHz only")
        exit(1)
    sin = lfilter([1, -1], [1, -alpha], sin)
    dither = np.random.random_sample(len(sin)) + np.random.random_sample(len(sin)) - 1
    spow = np.std(dither)
    sout = sin + 1e-6 * spow * dither
    return sout

#my funcation


def get_fft_spectrum(data):
    if isinstance(data, str):
        signal = load_wav(data, SAMPLE_RATE)
    else:
        signal = data.copy()
    signal *= 2**15
    # get FFT spectrum
    signal = remove_dc_and_dither(signal, SAMPLE_RATE)
    signal = sigproc.preemphasis(signal, coeff=PREEMPHASIS_ALPHA)
    frames = sigproc.framesig(signal, frame_len=FRAME_LEN*SAMPLE_RATE, frame_step=FRAME_STEP*SAMPLE_RATE, winfunc=np.hamming)
    fft = abs(np.fft.fft(frames, n=NUM_FFT))
    fft_norm = normalize_frames(fft).astype(np.float16).T
    return fft_norm


if __name__ == '__main__':
    a = get_fft_spectrum("D:\\dataset\\woxceleb\\train_split\\id10001\\0_00001_3.wav")
