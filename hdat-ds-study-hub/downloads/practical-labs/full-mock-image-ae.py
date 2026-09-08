"""29강 이미지 분류와 Autoencoder 이상탐지를 오프라인 합성 이미지로 연습한다."""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def draw_pattern(label: int, rng: np.random.Generator, size: int = 32) -> np.ndarray:
    image = rng.normal(0.02, 0.04, (size, size)).astype(np.float32)
    offset = int(rng.integers(-4, 5))
    thickness = int(rng.integers(2, 4))
    if label == 0:
        center = size // 2 + offset
        image[:, max(0, center - thickness) : min(size, center + thickness)] += 0.9
    elif label == 1:
        center = size // 2 + offset
        image[max(0, center - thickness) : min(size, center + thickness), :] += 0.9
    elif label == 2:
        for row in range(size):
            column = row + offset
            if 0 <= column < size:
                image[row, max(0, column - thickness) : min(size, column + thickness)] += 0.9
    else:
        raise ValueError("label은 0,1,2")
    return np.clip(image, 0.0, 1.0)


def make_classification_data(
    samples_per_class: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if samples_per_class < 20:
        raise ValueError("class당 20개 이상 필요")
    rng = np.random.default_rng(seed)
    images, labels = [], []
    for label in range(3):
        for _ in range(samples_per_class):
            images.append(draw_pattern(label, rng))
            labels.append(label)
    order = rng.permutation(len(labels))
    x = np.asarray(images, dtype=np.float32)[order, None, :, :]
    y = np.asarray(labels, dtype=np.int64)[order]
    return x, y


def make_single_class(count: int, label: int, seed: int) -> np.ndarray:
    if count < 1:
        raise ValueError("count는 1 이상")
    rng = np.random.default_rng(seed)
    images = [draw_pattern(label, rng) for _ in range(count)]
    return np.asarray(images, dtype=np.float32)[:, None, :, :]


def stratified_three_way_split(
    x: np.ndarray,
    y: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    indices = np.arange(len(y))
    train_index, remainder_index = train_test_split(
        indices,
        train_size=0.65,
        random_state=seed,
        shuffle=True,
        stratify=y,
    )
    valid_share_of_remainder = 0.17 / (1.0 - 0.65)
    valid_index, test_index = train_test_split(
        remainder_index,
        train_size=valid_share_of_remainder,
        random_state=seed + 1,
        shuffle=True,
        stratify=y[remainder_index],
    )
    expected_classes = np.unique(y)
    for name, split_index in (
        ("train", train_index),
        ("valid", valid_index),
        ("test", test_index),
    ):
        split_classes = np.unique(y[split_index])
        if not np.array_equal(split_classes, expected_classes):
            raise RuntimeError(
                f"{name} class 누락: {split_classes.tolist()} != "
                f"{expected_classes.tolist()}"
            )
    return (
        x[train_index],
        y[train_index],
        x[valid_index],
        y[valid_index],
        x[test_index],
        y[test_index],
    )


class SmallImageCNN(nn.Module):
    def __init__(self, n_classes: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(32, n_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x).flatten(1))


class ConvAutoencoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.decoder(self.encoder(x))


def fit_classifier(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> nn.Module:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), labels)
            if not torch.isfinite(loss):
                raise FloatingPointError("classifier non-finite loss")
            loss.backward()
            optimizer.step()
    return model


def predict_class(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch[0].to(device)
            chunks.append(model(images).argmax(1).cpu().numpy())
    return np.concatenate(chunks)


def fit_autoencoder(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    epochs: int,
) -> nn.Module:
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    for _ in range(epochs):
        model.train()
        for batch in loader:
            images = batch[0].to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction = model(images)
            loss = torch.nn.functional.mse_loss(reconstruction, images)
            if not torch.isfinite(loss):
                raise FloatingPointError("AE non-finite loss")
            loss.backward()
            optimizer.step()
    return model


def reconstruction_errors(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            images = batch[0].to(device)
            reconstruction = model(images)
            error = (reconstruction - images).pow(2).flatten(1).mean(1)
            chunks.append(error.cpu().numpy())
    return np.concatenate(chunks)


def checked_save(path: Path, prediction: np.ndarray, expected_rows: int) -> None:
    prediction = np.asarray(prediction)
    if prediction.shape != (expected_rows,) or prediction.dtype == object:
        raise ValueError((prediction.shape, prediction.dtype))
    if not np.isfinite(prediction).all():
        raise ValueError("prediction NaN/Inf")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, prediction, allow_pickle=False)
    loaded = np.load(path, allow_pickle=False)
    if loaded.shape != prediction.shape or not np.array_equal(loaded, prediction):
        raise IOError("NPY reload 불일치")
    print("saved:", path, loaded.shape, loaded.dtype)


def run(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x, y = make_classification_data(args.samples_per_class, args.seed)
    x_train, y_train, x_valid, y_valid, x_test, y_test_hidden = (
        stratified_three_way_split(x, y, args.seed)
    )
    print(
        "classification split class counts:",
        np.bincount(y_train).tolist(),
        np.bincount(y_valid).tolist(),
        np.bincount(y_test_hidden).tolist(),
    )

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_valid)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    test_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_test)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    classifier = SmallImageCNN().to(device)
    if classifier(torch.zeros(2, 1, 32, 32, device=device)).shape != (2, 3):
        raise RuntimeError("classifier dummy shape")
    classifier = fit_classifier(classifier, train_loader, device, args.epochs)
    valid_prediction = predict_class(classifier, valid_loader, device)
    test_prediction = predict_class(classifier, test_loader, device)
    print("classification valid accuracy:", accuracy_score(y_valid, valid_prediction))
    checked_save(args.class_output, test_prediction.astype(np.int64), len(x_test))
    print("classification hidden test accuracy:", accuracy_score(y_test_hidden, test_prediction))

    normal_train = make_single_class(args.samples_per_class, 0, args.seed + 10)
    normal_valid = make_single_class(
        max(20, args.samples_per_class // 3), 0, args.seed + 11
    )
    anomaly_mix, anomaly_labels = make_classification_data(
        max(20, args.samples_per_class // 2), args.seed + 12
    )
    anomaly_target = (anomaly_labels != 0).astype(np.int64)

    ae_train_loader = DataLoader(
        TensorDataset(torch.from_numpy(normal_train)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    ae_valid_loader = DataLoader(
        TensorDataset(torch.from_numpy(normal_valid)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    ae_test_loader = DataLoader(
        TensorDataset(torch.from_numpy(anomaly_mix)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    autoencoder = ConvAutoencoder().to(device)
    if autoencoder(torch.zeros(2, 1, 32, 32, device=device)).shape != (2, 1, 32, 32):
        raise RuntimeError("AE dummy shape")
    autoencoder = fit_autoencoder(autoencoder, ae_train_loader, device, args.epochs)
    normal_error = reconstruction_errors(autoencoder, ae_valid_loader, device)
    test_error = reconstruction_errors(autoencoder, ae_test_loader, device)
    threshold = float(np.quantile(normal_error, 0.99))
    anomaly_prediction = (test_error > threshold).astype(np.int64)
    print("AE threshold:", threshold)
    checked_save(args.ae_output, anomaly_prediction, len(anomaly_mix))
    print(
        "AE hidden test macro-F1:",
        f1_score(anomaly_target, anomaly_prediction, average="macro", zero_division=0),
    )


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-class", type=int, default=800)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--class-output",
        type=Path,
        default=root / "Submission_problem_image_mock.npy",
    )
    parser.add_argument(
        "--ae-output",
        type=Path,
        default=root / "Submission_problem_ae_mock.npy",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
