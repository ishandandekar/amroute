from pathlib import Path

import keras as K

import torch
import librosa
import numpy as np


def features_extractor(file_name: Path):
    audio, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)

    return mfccs_scaled_features


def create_sireNN():
    K.backend.clear_session()
    model = K.models.Sequential()
    model.add(K.layers.LSTM(128, input_shape=(1, 80)))
    model.add(K.layers.Dropout(0.5))
    model.add(K.layers.Dense(64, activation="relu"))
    model.add(K.layers.Dense(1, activation="softmax"))
    model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["acc"])
    return model


if torch.cuda.is_available():
    x = torch.rand(10000, 10000).to("cuda")
    y = torch.mm(x, x)
    print("Running on:", y.device)
else:
    print("CUDA not available")
