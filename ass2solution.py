import os, glob
import numpy as np
from scipy import signal, stats
# import matplotlib
# matplotlib.use("Qt5Agg")
# from matplotlib import pyplot as plt
# TODO Change to scipy wavread
import librosa
# from tqdm import tqdm

eps = 1e-6


def hann(L):
    return 0.5 - (0.5 * np.cos(2 * np.pi / L * np.arange(L)))


def stft(xb, fs):
    X = []
    times = []
    ind = 0
    K = len(xb[1])
    for n in range(len(xb)):
        f, t, Zxx = signal.stft(xb[n], fs=fs, nperseg=K, noverlap=0, boundary=None, window=hann(K))
        abs_Z = np.abs(Zxx.flatten())
        X.append(abs_Z)
        times.append(t[0] + ind)
        ind += len(xb[1])
    return np.array(X), times


def block_audio(x, blockSize, hopSize, fs):
    numBlocks = int(np.ceil(x.size / hopSize))
    xb = np.zeros([numBlocks, blockSize])
    t = (np.arange(numBlocks) * hopSize) / fs
    x = np.concatenate((x, np.zeros(blockSize)), axis=0)
    for n in range(numBlocks):
        i_start = n * hopSize
        i_stop = np.min([x.size - 1, i_start + blockSize - 1])
        xb[n][np.arange(0, blockSize)] = x[np.arange(i_start, i_stop + 1)]
    return xb, t


def extract_spectral_centroid(xb, fs):
    K = len(xb[1]) // 2
    X, t = stft(xb, fs)
    v_sc = np.sum(np.arange(K + 1).reshape(1, -1) * X, axis=1) / (np.sum(X, axis=1) + eps) / (K - 1)
    return v_sc * fs / 2


def extract_rms(xb):
    v_rms = np.sqrt((1 / len(xb[1])) * np.sum(np.square(xb), axis=1))
    return np.maximum(20 * np.log10(v_rms + eps), -100)


def extract_zerocrossingrate(xb):
    return np.sum(np.abs(np.diff(np.sign(xb), axis=1)), axis=1) / (2 * len(xb[1]))


def extract_spectral_crest(xb):
    X, _ = stft(xb, 1)
    return np.max(X, axis=1) / np.sum(X, axis=1)


def extract_spectral_flux(xb):
    X, _ = stft(xb, 1)
    v_sf = np.sqrt(np.sum(np.square(np.diff(X, axis=0)), axis=1)) / (len(xb[1]) / 2)
    v_sf = np.insert(v_sf, 0, 0)
    return v_sf


# A2
def extract_features(x, blockSize, hopSize, fs):
    xb, t = block_audio(x, blockSize, hopSize, fs)
    v_sc = extract_spectral_centroid(xb, fs)
    v_rms = extract_rms(xb)
    v_zc = extract_zerocrossingrate(xb)
    v_scf = extract_spectral_crest(xb)
    v_sf = extract_spectral_flux(xb)
    return np.vstack((v_sc, v_rms, v_zc, v_scf, v_sf))


# A3
def aggregate_feature_per_file(features):
    mean = np.mean(features, axis=1)
    std = np.std(features, axis=1)
    return np.dstack((mean, std)).flatten()


# A4
def get_feature_data(path, blockSize, hopSize):
    files = glob.glob(os.path.join(path, "*.wav"))
    sr = 44100
    ft_data = np.empty((10, len(files)))
    for i, f in enumerate(files):
        x, _ = librosa.core.load(f, sr=sr, mono=True)
        ft = extract_features(x, blockSize, hopSize, sr)
        agg_ft = aggregate_feature_per_file(ft)
        ft_data[:, i] = agg_ft.flatten()
    return ft_data


# # B1
def normalize_zscore(featureData):
    return stats.zscore(featureData)


# C1
def visualize_features(path_to_musicspeech, blockSize=1024, hopSize=256):
    folders = ["speech_wav", "music_wav"]
    index = []
    ft_matrix = []
    for folder in folders:
        path = os.path.join(path_to_musicspeech, folder)
        ft_data = get_feature_data(path, blockSize, hopSize)
        ft_matrix.append(ft_data)
        index.append(ft_data)
    ft_matrix = np.vstack((ft_matrix[0], ft_matrix[1]))
    print(ft_matrix.shape)

# visualize_features("music_speech")
