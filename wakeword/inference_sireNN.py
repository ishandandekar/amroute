from scipy.signal import butter, lfilter
import torch
import torch.nn as nn
import librosa
import numpy as np


class SireNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=2, dropout=0.5):
        super().__init__()
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            batch_first=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        _, (hidden, _) = self.lstm(x)
        x = self.dropout(hidden.squeeze(0))
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def bandpass_filter(data, low=500, high=2000, sr=16000):
    nyq = sr / 2
    b, a = butter(4, [low / nyq, high / nyq], btype="band")
    return lfilter(b, a, data)


def preprocess_audio(filepath, n_mfcc=80):
    audio, sr = librosa.load(filepath, res_type="kaiser_fast")
    audio = bandpass_filter(audio, low=500, high=2000, sr=sr)
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    tensor = torch.FloatTensor(mfccs_scaled_features).reshape(1, -1, n_mfcc)
    return tensor


def load_model(model_path="sireNN.pt", device=None):
    if device is None:
        device = get_device()
    model = SireNN(input_dim=80, hidden_dim=128, num_classes=2, dropout=0.5)
    state_dict = torch.load(model_path, weights_only=True, map_location=device)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith("_orig_mod."):
            new_state_dict[k[10:]] = v
        else:
            new_state_dict[k] = v
    model.load_state_dict(new_state_dict)
    model = torch.compile(model)
    model.to(device)
    model.eval()
    return model


def predict(model, audio_tensor, device=None):
    if device is None:
        device = get_device()

    with torch.no_grad():
        logits = model(audio_tensor.to(device))
        probabilities = torch.softmax(logits, dim=1)
        confidence, predicted_class = probabilities.max(dim=1)

    class_labels = {0: "traffic", 1: "ambulance"}
    pred_id = predicted_class.item()
    conf = confidence.item()

    return {
        "prediction": class_labels[pred_id],
        "class_id": pred_id,
        "confidence": round(conf, 4),
        "probabilities": {
            "traffic": round(probabilities[0, 0].item(), 4),
            "ambulance": round(probabilities[0, 1].item(), 4),
        },
    }


def main():
    model_path = "sireNN.pt"
    test_audio = "./sounds/ambulance_and_traffic/sound_5.wav"

    device = get_device()
    print(f"Using device: {device}")

    print(f"Loading model from {model_path}...")
    model = load_model(model_path, device)
    print("Model loaded and compiled successfully.\n")

    print(f"Processing audio: {test_audio}")
    audio_tensor = preprocess_audio(test_audio)
    print(f"Audio tensor shape: {audio_tensor.shape}\n")

    result = predict(model, audio_tensor, device)

    print("=" * 40)
    print("PREDICTION RESULTS")
    print("=" * 40)
    print(f"Predicted Class: {result['prediction'].upper()}")
    print(f"Class ID:        {result['class_id']}")
    print(f"Confidence:      {result['confidence']:.2%}")
    print(
        f"Probabilities:   Traffic={result['probabilities']['traffic']:.2%}, "
        f"Ambulance={result['probabilities']['ambulance']:.2%}"
    )
    print("=" * 40)


if __name__ == "__main__":
    main()
