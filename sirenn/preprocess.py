import os

import librosa
import numpy as np
import torch
from scipy.signal import butter, lfilter

MODEL_SR = 16_000
BANDPASS_LOW = 500
BANDPASS_HIGH = 2000
N_MFCC = 80


def bandpass_filter(
    data,
    sr,
    low=BANDPASS_LOW,
    high=BANDPASS_HIGH,
    order=4,
):
    nyq = sr / 2
    b, a = butter(order, [low / nyq, high / nyq], btype="band")
    return lfilter(b, a, data)


def load_mono_16k(source, sr=None):
    if isinstance(source, (str, np.str_, os.PathLike)):
        audio, sr = librosa.load(str(source), sr=MODEL_SR, mono=True)
    else:
        audio = np.asarray(source, dtype=np.float32)
        if sr is None:
            sr = MODEL_SR
        if sr != MODEL_SR:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=MODEL_SR)
    return audio.astype(np.float32), MODEL_SR


def extract_features_from_path(path, n_mfcc=N_MFCC):
    audio, sr = load_mono_16k(path)
    return _mfcc(audio, sr, n_mfcc)


def extract_features_from_array(audio, sr, n_mfcc=N_MFCC):
    audio, sr = load_mono_16k(audio, sr)
    return _mfcc(audio, sr, n_mfcc)


def _mfcc(audio, sr, n_mfcc):
    audio = bandpass_filter(audio, sr)
    mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    return mfccs.T.astype(np.float32)


def to_model_input(features):
    features = np.asarray(features, dtype=np.float32)
    return torch.FloatTensor(features).reshape(1, -1, features.shape[-1])
