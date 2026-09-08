"""33강의 차량 센서 다중출력 Problem 모의를 끝까지 실행하는 연습 스크립트."""

from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def generate_mock_series(
    n: int,
    n_features: int,
    horizon: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    if horizon < 1:
        raise ValueError("horizon은 1 이상의 미래 간격이어야 합니다")
    if n <= horizon + 100 or n_features < 7:
        raise ValueError("n은 horizon+100보다 크고 n_features는 7 이상이어야 합니다")
    rng = np.random.default_rng(seed)
    features = rng.normal(0, 0.2, (n, n_features)).astype(np.float32)
    seasonal = np.sin(np.arange(n, dtype=np.float32) / 80.0)
    for index in range(1, n):
        features[index] += 0.90 * features[index - 1]
        features[index, 6] += 0.15 * seasonal[index]

    target = np.full((n, 3), np.nan, dtype=np.float32)
    source = features[:-horizon]
    noise = rng.normal(0, 0.03, (n - horizon, 3)).astype(np.float32)
    target[horizon:, 0] = 0.8 * source[:, 0] + 0.2 * source[:, 3] + noise[:, 0]
    target[horizon:, 1] = -0.5 * source[:, 1] + 0.3 * source[:, 4] ** 2 + noise[:, 1]
    target[horizon:, 2] = (
        0.6 * source[:, 2] - 0.2 * source[:, 5] + 0.1 * source[:, 6] + noise[:, 2]
    )
    return features, target


def split_end_indices(
    n_rows: int,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if lookback < 1 or horizon < 1 or n_rows <= lookback + horizon:
        raise ValueError("lookback과 horizon은 1 이상이고 유효 window가 있어야 합니다")
    train_target_cut = int(n_rows * 0.70)
    valid_target_cut = int(n_rows * 0.85)
    ends = np.arange(lookback - 1, n_rows - horizon, dtype=np.int64)
    target_indices = ends + horizon
    train = ends[target_indices < train_target_cut]
    # Each forecast can use only labels already available at its input endpoint.
    valid = ends[(ends >= train_target_cut) & (target_indices < valid_target_cut)]
    test = ends[ends >= valid_target_cut]
    if min(map(len, (train, valid, test))) == 0:
        raise ValueError("train/valid/test window 중 빈 구간이 있습니다")
    return train, valid, test


def materialize(
    features: np.ndarray,
    target: np.ndarray,
    end_indices: np.ndarray,
    lookback: int,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    windows = np.stack(
        [features[end - lookback + 1 : end + 1] for end in end_indices]
    ).astype(np.float32)
    labels = np.stack([target[end + horizon] for end in end_indices]).astype(np.float32)
    expected_x = (len(end_indices), lookback, features.shape[1])
    if windows.shape != expected_x or labels.shape != (len(end_indices), 3):
        raise RuntimeError((windows.shape, labels.shape, expected_x))
    if not np.isfinite(windows).all() or not np.isfinite(labels).all():
        raise ValueError("window 또는 target에 NaN/Inf가 있습니다")
    return windows, labels


class SensorCNN1D(nn.Module):
    def __init__(self, n_features: int, out_dim: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(n_features, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x.permute(0, 2, 1)))


def evaluate_original_mse(
    model: nn.Module,
    loader: DataLoader,
    target_scaler: StandardScaler,
    device: torch.device,
) -> float:
    if len(loader) == 0:
        raise ValueError("빈 validation loader")
    model.eval()
    squared_error, count = 0.0, 0
    with torch.inference_mode():
        for features, target in loader:
            features = features.to(device)
            prediction_scaled = model(features)
            if not torch.isfinite(prediction_scaled).all():
                raise FloatingPointError("non-finite validation prediction")
            prediction = target_scaler.inverse_transform(
                prediction_scaled.cpu().numpy()
            )
            target_original = target_scaler.inverse_transform(target.numpy())
            difference = prediction.astype(np.float64) - target_original.astype(np.float64)
            squared_error += float(np.square(difference).sum())
            count += difference.size
    return squared_error / count


def fit(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    target_scaler: StandardScaler,
    device: torch.device,
    max_epochs: int,
    max_seconds: float,
) -> nn.Module:
    criterion = nn.MSELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_value, best_state, wait = float("inf"), None, 0
    start = time.monotonic()

    for epoch in range(1, max_epochs + 1):
        model.train()
        total, count = 0.0, 0
        for features, target in train_loader:
            features = features.to(device)
            target = target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite train loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += loss.detach().item() * len(features)
            count += len(features)

        valid_value = evaluate_original_mse(
            model, valid_loader, target_scaler, device
        )
        print(
            f"epoch={epoch:03d} train_scaled_mse={total / count:.6f} "
            f"valid_original_mse={valid_value:.6f}"
        )
        if valid_value < best_value - 1e-6:
            best_value = valid_value
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
            wait = 0
        else:
            wait += 1
        if wait >= 5 or time.monotonic() - start >= max_seconds:
            break

    if best_state is None:
        raise RuntimeError("유효한 best checkpoint가 없습니다")
    model.load_state_dict(best_state)
    return model.to(device)


def predict(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            features = batch[0].to(device)
            chunks.append(model(features).cpu().numpy())
    if not chunks:
        raise ValueError("빈 prediction loader")
    return np.concatenate(chunks, axis=0)


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.shape != y_pred.shape:
        raise ValueError((y_true.shape, y_pred.shape))
    return float(np.mean((y_true - y_pred) ** 2))


def run(args: argparse.Namespace) -> None:
    problem_start = time.monotonic()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    features, target = generate_mock_series(
        args.n, args.features, args.horizon, args.seed
    )
    train_end, valid_end, test_end = split_end_indices(
        len(features), args.lookback, args.horizon
    )
    x_train, y_train = materialize(
        features, target, train_end, args.lookback, args.horizon
    )
    x_valid, y_valid = materialize(
        features, target, valid_end, args.lookback, args.horizon
    )
    x_test, y_test_hidden = materialize(
        features, target, test_end, args.lookback, args.horizon
    )
    print("windows:", x_train.shape, x_valid.shape, x_test.shape)

    n_features = x_train.shape[-1]
    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train.reshape(-1, n_features)).reshape(x_train.shape)
    x_valid = x_scaler.transform(x_valid.reshape(-1, n_features)).reshape(x_valid.shape)
    x_test = x_scaler.transform(x_test.reshape(-1, n_features)).reshape(x_test.shape)
    x_train = x_train.astype(np.float32)
    x_valid = x_valid.astype(np.float32)
    x_test = x_test.astype(np.float32)

    y_scaler = StandardScaler()
    y_train_scaled = y_scaler.fit_transform(y_train).astype(np.float32)
    y_valid_scaled = y_scaler.transform(y_valid).astype(np.float32)
    mean_prediction = np.repeat(y_train.mean(0, keepdims=True), len(y_valid), axis=0)
    baseline_valid_mse = mse(y_valid, mean_prediction)
    print("mean baseline valid MSE:", baseline_valid_mse)
    baseline_test_prediction = np.repeat(
        y_train.mean(0, keepdims=True), len(x_test), axis=0
    ).astype(np.float32)
    if baseline_test_prediction.shape != (len(x_test), 3):
        raise RuntimeError("baseline test shape")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, baseline_test_prediction, allow_pickle=False)
    baseline_reloaded = np.load(args.output, allow_pickle=False)
    if not np.array_equal(baseline_reloaded, baseline_test_prediction):
        raise IOError("baseline NPY reload 불일치")
    print("first valid submission seconds:", time.monotonic() - problem_start)

    train_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train_scaled)),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=0,
    )
    valid_loader = DataLoader(
        TensorDataset(torch.from_numpy(x_valid), torch.from_numpy(y_valid_scaled)),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    valid_predict_loader = DataLoader(
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

    model = SensorCNN1D(n_features).to(device)
    dummy = model(torch.zeros(2, args.lookback, n_features, device=device))
    if dummy.shape != (2, 3):
        raise RuntimeError(f"dummy output 오류: {dummy.shape}")
    model = fit(
        model,
        train_loader,
        valid_loader,
        y_scaler,
        device,
        args.epochs,
        args.max_seconds,
    )

    valid_prediction = y_scaler.inverse_transform(
        predict(model, valid_predict_loader, device)
    )
    test_prediction = y_scaler.inverse_transform(predict(model, test_loader, device))
    cnn_valid_mse = mse(y_valid, valid_prediction)
    print("CNN valid MSE:", cnn_valid_mse)
    print(
        "valid RMSE by output:",
        np.sqrt(np.mean((y_valid - valid_prediction) ** 2, axis=0)),
    )

    if cnn_valid_mse < baseline_valid_mse:
        selected_name = "CNN"
        selected_valid_mse = cnn_valid_mse
        final_prediction = test_prediction
    else:
        selected_name = "mean baseline"
        selected_valid_mse = baseline_valid_mse
        final_prediction = baseline_test_prediction
    print("selected model/valid MSE:", selected_name, selected_valid_mse)

    expected_shape = (len(x_test), 3)
    if final_prediction.shape != expected_shape or not np.isfinite(final_prediction).all():
        raise ValueError("test prediction 계약 오류")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, final_prediction, allow_pickle=False)
    reloaded = np.load(args.output, allow_pickle=False)
    if reloaded.shape != expected_shape or not np.array_equal(reloaded, final_prediction):
        raise IOError("NPY reload 불일치")
    print("saved:", args.output, reloaded.shape, reloaded.dtype)
    print("hidden test MSE (연습 채점용):", mse(y_test_hidden, final_prediction))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=30_000)
    parser.add_argument("--features", type=int, default=12)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=40)
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-seconds", type=float, default=120.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "Submission_problem.npy",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
