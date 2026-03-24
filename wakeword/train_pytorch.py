import librosa
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import torch.optim as optim


def features_extractor(file_name):
    audio, sample_rate = librosa.load(file_name, res_type="kaiser_fast")
    mfccs_features = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=80)
    mfccs_scaled_features = np.mean(mfccs_features.T, axis=0)
    return mfccs_scaled_features


def extract_features(audio_dataset_path, pkl_path="./Extracted_Features.pkl"):
    if os.path.exists(pkl_path):
        print("Loading cached features...")
        with open(pkl_path, "rb") as f:
            return pickle.load(f)

    extracted_features = []
    for path in os.listdir(audio_dataset_path):
        print(path)
        if path == "firetruck" or path == "ambulance_and_traffic":
            continue
        for file in tqdm(
            os.listdir(audio_dataset_path + path + "/"), desc="Extracting " + str(path)
        ):
            if file.lower().endswith(".wav"):
                file_name = audio_dataset_path + path + "/" + file
                data = features_extractor(file_name)
                extracted_features.append([data, path])

    with open(pkl_path, "wb") as f:
        pickle.dump(extracted_features, f)
    return extracted_features


def prepare_data(extracted_features):
    import pandas as pd

    df = pd.DataFrame(extracted_features, columns=["feature", "class"])
    X = np.array(df["feature"].tolist())
    y = np.array(df["class"].map({"ambulance": 1, "traffic": 0}).tolist())

    print("X shape: " + str(X.shape))
    print("y shape: " + str(y.shape))
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0, stratify=y, shuffle=True
    )

    return X_train, X_test, y_train, y_test


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


def train_model(
    model,
    train_loader,
    val_loader,
    device,
    epochs=100,
    lr=0.001,
    patience=10,
    model_path="best_sireNN_model.pt",
):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    best_val_loss = float("inf")
    patience_counter = 0
    best_model_state = None

    model = torch.compile(model)
    model.to(device)

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0

        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_x.size(0)
            _, predicted = outputs.max(1)
            train_total += batch_y.size(0)
            train_correct += predicted.eq(batch_y).sum().item()

        train_loss /= train_total
        train_acc = train_correct / train_total

        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                loss = criterion(outputs, batch_y)

                val_loss += loss.item() * batch_x.size(0)
                _, predicted = outputs.max(1)
                val_total += batch_y.size(0)
                val_correct += predicted.eq(batch_y).sum().item()

        val_loss /= val_total
        val_acc = val_correct / val_total

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = {
                k: v.cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
            print(f"  -> New best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_model_state)
    torch.save(best_model_state, model_path)
    print(f"Best model saved to {model_path}")

    uncompiled_model = SireNN(input_dim=80, hidden_dim=128, num_classes=2, dropout=0.5)
    uncompiled_model.load_state_dict(best_model_state)
    uncompiled_path = model_path.replace(".pt", "_uncompiled.pt")
    torch.save(uncompiled_model.state_dict(), uncompiled_path)
    print(f"Uncompiled model saved to {uncompiled_path}")

    return uncompiled_model


def evaluate_model(model, X_test, y_test, device):
    model.eval()
    model.to(device)

    x_test_tensor = torch.FloatTensor(X_test.reshape(len(X_test), -1, 80)).to(device)

    with torch.no_grad():
        outputs = model(x_test_tensor)
        _, y_pred = outputs.max(1)

    y_pred = y_pred.cpu().numpy()
    y_test_labels = y_test

    accuracy = (y_pred == y_test_labels).mean()
    conf_mat = confusion_matrix(y_test_labels, y_pred)

    return accuracy, conf_mat, y_pred


def main():
    audio_dataset_path = "./sounds/"
    pkl_path = "./Extracted_Features.pkl"

    device = get_device()
    print(f"Using device: {device}")

    extracted_features = extract_features(audio_dataset_path, pkl_path)
    X_train, X_test, y_train, y_test = prepare_data(extracted_features)

    x_train_features = X_train.reshape(len(X_train), -1, 80)
    x_test_features = X_test.reshape(len(X_test), -1, 80)

    print(f"Training samples: {X_train.shape[0]}")
    print(f"Test samples: {X_test.shape[0]}")

    train_dataset = TensorDataset(
        torch.FloatTensor(x_train_features), torch.LongTensor(y_train)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(x_test_features), torch.LongTensor(y_test)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
    )

    model = SireNN(input_dim=80, hidden_dim=128, num_classes=2, dropout=0.5)

    model = train_model(
        model,
        train_loader,
        test_loader,
        device,
        epochs=100,
        lr=0.001,
        patience=10,
        model_path="best_lstm.pt",
    )

    accuracy, conf_mat, y_pred = evaluate_model(model, X_test, y_test, device)

    print(f"\nTest Accuracy: {accuracy:.4f}")
    print("Confusion Matrix:")
    print(conf_mat)

    torch.save(model.state_dict(), "./sireNN.pt")
    print("Model saved to ./sireNN.pt")


if __name__ == "__main__":
    main()
