import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from tqdm import tqdm

import preprocess

CLASS_LABELS = {0: "traffic", 1: "siren", 2: "noise"}
CLASS_TO_ID = {"traffic": 0, "siren": 1, "noise": 2}

SIRENNET_ROOT = (
    "sireNNet-Emergency Vehicle Siren Classification Dataset For Urban Applications/sireNNet"
)
CACHE_DIR = Path("cache")
FEATURES_CACHE = Path("Extracted_Features_v2.pkl")
TEMPERATURE_FILE = Path("temperature.json")

CACHE_RMS_SIREN_MIN = 0.30
CACHE_RMS_NOISE_MAX = 0.10

CLASS_DIRS = {
    "siren": [
        f"{SIRENNET_ROOT}/ambulance",
        f"{SIRENNET_ROOT}/firetruck",
        f"{SIRENNET_ROOT}/police",
        "sounds/ambulance",
        "sounds/firetruck",
        "sounds/ambulance_and_traffic",
    ],
    "traffic": [
        f"{SIRENNET_ROOT}/traffic",
        "sounds/traffic",
    ],
    "noise": [
        "sounds/noise",
    ],
}


class SireNN(nn.Module):
    def __init__(self, input_dim, hidden_dim=128, num_classes=3, dropout=0.5):
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

    def forward(self, x, lengths=None):
        if lengths is not None:
            x = nn.utils.rnn.pack_padded_sequence(
                x, lengths.cpu(), batch_first=True, enforce_sorted=False
            )
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


def cache_windows_by_rms(label):
    pattern = re.compile(r"_rms_([0-9.]+)\.wav$")
    if label == "siren":
        pred = lambda r: r >= CACHE_RMS_SIREN_MIN
    elif label == "noise":
        pred = lambda r: r < CACHE_RMS_NOISE_MAX
    else:
        return []
    files = []
    for path in sorted(CACHE_DIR.glob("*.wav")):
        m = pattern.search(path.name)
        if m and pred(float(m.group(1))):
            files.append(str(path))
    return files


def discover_files():
    files_by_class = {}
    for cls, dirs in CLASS_DIRS.items():
        paths = []
        for d in dirs:
            p = Path(d)
            if p.exists():
                paths.extend(str(x) for x in sorted(p.glob("*.wav")))
        paths.extend(cache_windows_by_rms(cls))
        files_by_class[cls] = paths
        print(f"{cls}: {len(paths)} files")
    return files_by_class


def extract_all_features(files_by_class, force=False):
    if FEATURES_CACHE.exists() and not force:
        print(f"Loading cached features from {FEATURES_CACHE}...")
        with open(FEATURES_CACHE, "rb") as f:
            return pickle.load(f)

    extracted = []
    for cls, files in files_by_class.items():
        label = CLASS_TO_ID[cls]
        for path in tqdm(files, desc=f"Extracting {cls}", unit="file"):
            feats = preprocess.extract_features_from_path(path)
            extracted.append((feats, label))

    with open(FEATURES_CACHE, "wb") as f:
        pickle.dump(extracted, f)
    print(f"Cached {len(extracted)} samples -> {FEATURES_CACHE}")
    return extracted


def collate(batch):
    feats, labels = zip(*batch)
    lengths = torch.LongTensor([f.shape[0] for f in feats])
    max_len = int(lengths.max())
    padded = torch.zeros((len(feats), max_len, feats[0].shape[1]))
    for i, f in enumerate(feats):
        padded[i, : f.shape[0]] = torch.from_numpy(f)
    return padded, torch.LongTensor(labels), lengths


def class_weights(labels):
    counts = np.bincount(labels, minlength=len(CLASS_LABELS)).astype(np.float64)
    weights = len(labels) / (len(CLASS_LABELS) * counts)
    return torch.FloatTensor(weights)


def train_model(model, train_loader, val_loader, device, epochs, lr, patience, weights):
    criterion = nn.CrossEntropyLoss(weight=weights.to(device))
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

        for batch_x, batch_y, lengths in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            optimizer.zero_grad()
            outputs = model(batch_x, lengths)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            train_loss += loss.item() * batch_y.size(0)
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
            for batch_x, batch_y, lengths in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x, lengths)
                loss = criterion(outputs, batch_y)
                val_loss += loss.item() * batch_y.size(0)
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
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
            patience_counter = 0
            print(f"  -> New best model saved (val_loss: {val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

    model.load_state_dict(best_model_state)
    return model


def predict_logits(model, loader, device):
    model.eval()
    all_logits, all_labels = [], []
    with torch.no_grad():
        for batch_x, batch_y, lengths in loader:
            batch_x = batch_x.to(device)
            logits = model(batch_x, lengths)
            all_logits.append(logits.detach().cpu())
            all_labels.append(batch_y)
    return torch.cat(all_logits), torch.cat(all_labels)


def fit_temperature(logits, labels):
    logits = logits.double()
    labels = labels.long()

    def nll(t):
        return nn.functional.cross_entropy(logits / t, labels)

    t = torch.tensor(1.0, dtype=torch.double, requires_grad=True)
    optimizer = optim.LBFGS([t], lr=0.05, max_iter=200, tolerance_grad=1e-8)

    def closure():
        optimizer.zero_grad()
        loss = nll(t)
        loss.backward()
        return loss

    optimizer.step(closure)
    return float(max(0.2, min(10.0, t.item())))


def ece(probs, labels, n_bins=15):
    conf, pred = probs.max(1)
    acc = (pred == labels).double()
    edges = torch.linspace(0, 1, n_bins + 1)
    total = 0.0
    for i in range(n_bins):
        mask = (conf > edges[i]) & (conf <= edges[i + 1])
        if mask.sum() == 0:
            continue
        weight = mask.sum().item() / len(labels)
        total += weight * abs(
            acc[mask].mean().item() - conf[mask].mean().item()
        )
    return total


def report_calibration(logits, labels, temperature, tag):
    probs = torch.softmax(logits / temperature, dim=1)
    _, pred = probs.max(1)
    acc = (pred == labels).double().mean().item()
    conf_mat = confusion_matrix(labels.numpy(), pred.numpy())
    print(f"\n[{tag}] temperature={temperature:.4f}")
    print(f"[{tag}] accuracy={acc:.4f} ece={ece(probs, labels):.4f}")
    print(f"[{tag}] confusion matrix (rows=truth {list(CLASS_LABELS.values())}, cols=pred):")
    print(conf_mat)


def main():
    parser = argparse.ArgumentParser(
        description="Train sireNN 3-class (traffic/siren/noise) LSTM with "
        "per-frame MFCCs + temperature calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-path", default="sireNN.pt")
    parser.add_argument(
        "--force-extract", action="store_true", help="rebuild the feature cache"
    )
    args = parser.parse_args()

    device = get_device()
    print(f"Using device: {device}")

    files_by_class = discover_files()
    extracted = extract_all_features(files_by_class, force=args.force_extract)

    feats = [f for f, _ in extracted]
    labels = np.array([l for _, l in extracted])
    print(f"Total samples: {len(extracted)}")
    print(f"Label distribution: {dict(zip(CLASS_LABELS.values(), np.bincount(labels)))}")

    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=0.2,
        random_state=0,
        stratify=labels,
        shuffle=True,
    )

    train_ds = [(feats[i], labels[i]) for i in train_idx]
    val_ds = [(feats[i], labels[i]) for i in val_idx]

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate
    )

    model = SireNN(input_dim=preprocess.N_MFCC, hidden_dim=128, num_classes=3)
    weights = class_weights(labels[train_idx])
    print(f"Class weights: {weights.tolist()}")

    model = train_model(
        model,
        train_loader,
        val_loader,
        device,
        epochs=args.epochs,
        lr=args.lr,
        patience=args.patience,
        weights=weights,
    )

    val_logits, val_labels = predict_logits(model, val_loader, device)
    report_calibration(val_logits, val_labels, 1.0, "before temperature")

    temperature = fit_temperature(val_logits, val_labels)
    report_calibration(val_logits, val_labels, temperature, "after temperature")

    uncompiled_model = SireNN(input_dim=preprocess.N_MFCC, hidden_dim=128, num_classes=3)
    fixed_state_dict = {
        k.removeprefix("_orig_mod."): v for k, v in model.state_dict().items()
    }
    uncompiled_model.load_state_dict(fixed_state_dict)
    torch.save(uncompiled_model.state_dict(), args.model_path)
    print(f"Model saved to {args.model_path}")

    with open(TEMPERATURE_FILE, "w") as f:
        json.dump({"temperature": temperature}, f, indent=2)
    print(f"Temperature saved to {TEMPERATURE_FILE}")


if __name__ == "__main__":
    main()
