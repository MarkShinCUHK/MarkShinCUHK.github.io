"""33.16 통합 모의의 새 차량 고장 분류 Problem 생성기와 모범 풀이."""

from __future__ import annotations

import argparse
import copy
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30.0, 30.0)))


def generate_data(
    n_groups: int,
    rows_per_group: int,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    if n_groups < 20 or rows_per_group < 5:
        raise ValueError("n_groups>=20, rows_per_group>=5가 필요합니다")
    rng = np.random.default_rng(seed)
    group_ids = np.repeat(np.arange(n_groups), rows_per_group)
    n_rows = len(group_ids)
    group_risk = rng.normal(0, 0.45, n_groups)
    vehicle_type_by_group = rng.choice(["sedan", "suv", "truck"], n_groups, p=[0.5, 0.3, 0.2])
    region_by_group = rng.choice(["north", "south", "east", "west"], n_groups)
    mode = rng.choice(["eco", "normal", "sport"], n_rows, p=[0.25, 0.6, 0.15])

    temp = rng.normal(0, 1, n_rows)
    vibration = rng.normal(0, 1, n_rows)
    pressure = rng.normal(0, 1, n_rows)
    speed = rng.normal(0, 1, n_rows)
    age = rng.uniform(0, 1, n_rows)
    sensor_noise = rng.normal(0, 1, (n_rows, 5))
    vehicle_type = vehicle_type_by_group[group_ids].astype(object)
    region = region_by_group[group_ids].astype(object)

    logit = (
        -3.2
        + 1.25 * vibration
        + 0.75 * (temp > 1.0)
        + 0.55 * (mode == "sport")
        + 0.80 * age
        + 0.50 * (vehicle_type == "truck")
        + group_risk[group_ids]
    )
    target = (rng.random(n_rows) < sigmoid(logit)).astype(np.int64)

    frame = pd.DataFrame(
        {
            "vehicle_id": [f"V{value:04d}" for value in group_ids],
            "temp": temp,
            "vibration": vibration,
            "pressure": pressure,
            "speed": speed,
            "age": age,
            "vehicle_type": vehicle_type,
            "region": region,
            "mode": mode.astype(object),
            **{f"sensor_{index}": sensor_noise[:, index] for index in range(5)},
            "fault": target,
        }
    )

    numeric = [
        "temp",
        "vibration",
        "pressure",
        "speed",
        "age",
        *[f"sensor_{index}" for index in range(5)],
    ]
    categorical = ["vehicle_type", "region", "mode"]
    for column in numeric:
        mask = rng.random(n_rows) < 0.04
        frame.loc[mask, column] = np.nan
    for column in categorical:
        mask = rng.random(n_rows) < 0.03
        frame.loc[mask, column] = None

    all_groups = np.arange(n_groups)
    rng.shuffle(all_groups)
    test_group_count = max(1, int(n_groups * 0.20))
    test_groups = set(all_groups[:test_group_count])
    is_test = np.fromiter((value in test_groups for value in group_ids), dtype=bool)
    train = frame.loc[~is_test].reset_index(drop=True)
    test_full = frame.loc[is_test].reset_index(drop=True)
    test_target = test_full.pop("fault").to_numpy(dtype=np.int64)
    if set(train["vehicle_id"]) & set(test_full["vehicle_id"]):
        raise RuntimeError("train/test group이 겹칩니다")
    return train, test_full, test_target


def normalize_categories(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    out = frame.copy()
    for column in columns:
        values = out[column].astype("string").astype(object)
        out[column] = values.where(pd.notna(values), np.nan)
    return out


def make_preprocessor(
    numeric_columns: list[str],
    categorical_columns: list[str],
) -> ColumnTransformer:
    numeric_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    category_pipe = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric_pipe, numeric_columns),
            ("category", category_pipe, categorical_columns),
        ]
    )


class BinaryMLP(nn.Module):
    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def make_loader(
    features: np.ndarray,
    target: np.ndarray | None,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    x_tensor = torch.from_numpy(np.asarray(features, dtype=np.float32))
    if target is None:
        dataset = TensorDataset(x_tensor)
    else:
        y_tensor = torch.from_numpy(np.asarray(target, dtype=np.float32).reshape(-1, 1))
        dataset = TensorDataset(x_tensor, y_tensor)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


def train_with_validation(
    model: nn.Module,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    positive_weight: float,
    device: torch.device,
    epochs: int,
) -> tuple[nn.Module, int, float, float]:
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best_state = None
    best_score, best_epoch, best_threshold, wait = -1.0, 0, 0.5, 0

    for epoch in range(1, epochs + 1):
        model.train()
        for features, target in train_loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite train loss")
            loss.backward()
            optimizer.step()

        model.eval()
        total, count = 0.0, 0
        probability_chunks: list[np.ndarray] = []
        target_chunks: list[np.ndarray] = []
        with torch.inference_mode():
            for features, target in valid_loader:
                features, target = features.to(device), target.to(device)
                logits = model(features)
                if not torch.isfinite(logits).all():
                    raise FloatingPointError("non-finite validation logits")
                loss = criterion(logits, target)
                total += loss.item() * len(features)
                count += len(features)
                probability_chunks.append(
                    torch.sigmoid(logits).cpu().numpy().reshape(-1)
                )
                target_chunks.append(target.cpu().numpy().reshape(-1))
        valid_loss = total / count
        valid_probability = np.concatenate(probability_chunks)
        valid_target = np.concatenate(target_chunks).astype(np.int64)
        epoch_threshold, epoch_score = choose_threshold(
            valid_target,
            valid_probability,
        )
        print(
            f"epoch={epoch:03d} valid_bce={valid_loss:.6f} "
            f"threshold={epoch_threshold:.2f} macro_f1={epoch_score:.6f}"
        )
        if epoch_score > best_score + 1e-12:
            best_score = epoch_score
            best_epoch = epoch
            best_threshold = epoch_threshold
            best_state = copy.deepcopy(
                {name: value.detach().cpu() for name, value in model.state_dict().items()}
            )
            wait = 0
        else:
            wait += 1
        if wait >= 5:
            break

    if best_state is None:
        raise RuntimeError("best checkpoint 없음")
    model.load_state_dict(best_state)
    return model.to(device), best_epoch, best_threshold, best_score


def train_fixed_epochs(
    model: nn.Module,
    loader: DataLoader,
    positive_weight: float,
    device: torch.device,
    epochs: int,
) -> nn.Module:
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([positive_weight], dtype=torch.float32, device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    for _ in range(max(1, epochs)):
        model.train()
        for features, target in loader:
            features, target = features.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(features), target)
            if not torch.isfinite(loss):
                raise FloatingPointError("non-finite full-train loss")
            loss.backward()
            optimizer.step()
    return model


def predict_probability(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    with torch.inference_mode():
        for batch in loader:
            features = batch[0].to(device)
            chunks.append(torch.sigmoid(model(features)).cpu().numpy().reshape(-1))
    if not chunks:
        raise ValueError("빈 prediction loader")
    return np.concatenate(chunks)


def positive_weight(target: np.ndarray) -> float:
    positives = int(np.sum(target == 1))
    negatives = int(np.sum(target == 0))
    if positives == 0 or negatives == 0:
        raise ValueError("양 class가 모두 필요합니다")
    return negatives / positives


def choose_threshold(target: np.ndarray, probability: np.ndarray) -> tuple[float, float]:
    best_threshold, best_score = 0.5, -1.0
    for threshold in np.linspace(0.05, 0.95, 91):
        score = f1_score(
            target,
            probability >= threshold,
            average="macro",
            zero_division=0,
        )
        if score > best_score:
            best_threshold, best_score = float(threshold), float(score)
    return best_threshold, best_score


def split_groups_with_all_classes(
    target: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    seed: int,
    max_attempts: int = 200,
) -> tuple[np.ndarray, np.ndarray]:
    classes = np.unique(target)
    if len(classes) < 2:
        raise ValueError("group split 전에 양 class가 모두 필요합니다")
    splitter = GroupShuffleSplit(
        n_splits=max_attempts,
        test_size=test_size,
        random_state=seed,
    )
    placeholder = np.empty(len(target), dtype=np.uint8)
    for attempt, (train_index, valid_index) in enumerate(
        splitter.split(placeholder, target, groups),
        start=1,
    ):
        train_classes = np.unique(target[train_index])
        valid_classes = np.unique(target[valid_index])
        if np.array_equal(train_classes, classes) and np.array_equal(valid_classes, classes):
            if set(groups[train_index]) & set(groups[valid_index]):
                raise RuntimeError("validation group 누수")
            print("class-safe group split attempt:", attempt)
            return train_index, valid_index
    raise ValueError(
        f"{max_attempts}회 시도했지만 train/valid 모두에 "
        f"class {classes.tolist()}를 보존하는 group split을 찾지 못했습니다"
    )


def checked_save(path: Path, prediction: np.ndarray, expected_rows: int, label: str) -> None:
    prediction = np.asarray(prediction)
    if prediction.shape != (expected_rows,) or prediction.dtype == object:
        raise ValueError((prediction.shape, prediction.dtype))
    if not np.isfinite(prediction).all():
        raise ValueError(f"{label} prediction NaN/Inf")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, prediction, allow_pickle=False)
    reloaded = np.load(path, allow_pickle=False)
    if reloaded.shape != prediction.shape or not np.array_equal(reloaded, prediction):
        raise IOError(f"{label} NPY reload 불일치")
    print(f"{label} saved:", path, reloaded.shape, reloaded.dtype)


def write_generated_data(
    train: pd.DataFrame,
    test: pd.DataFrame,
    data_dir: Path,
) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    train.to_csv(data_dir / "train.csv", index=False)
    test.to_csv(data_dir / "test.csv", index=False)
    print("generated:", data_dir / "train.csv", train.shape)
    print("generated:", data_dir / "test.csv", test.shape)
    print("required output shape:", (len(test),))


def run_solution(args: argparse.Namespace) -> None:
    problem_start = time.monotonic()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train, test, hidden_test_target = generate_data(
        args.groups, args.rows_per_group, args.seed
    )
    if args.generate_only:
        write_generated_data(train, test, args.data_dir)
        return

    target = train["fault"].to_numpy(dtype=np.int64)
    groups = train["vehicle_id"].to_numpy()
    feature_columns = [column for column in test.columns if column != "vehicle_id"]
    numeric_columns = [
        column for column in feature_columns if pd.api.types.is_numeric_dtype(train[column])
    ]
    categorical_columns = [column for column in feature_columns if column not in numeric_columns]
    train = normalize_categories(train, categorical_columns)
    test = normalize_categories(test, categorical_columns)

    train_index, valid_index = split_groups_with_all_classes(
        target,
        groups,
        test_size=0.20,
        seed=args.seed,
    )
    classes = np.unique(target)
    if not np.array_equal(np.unique(target[train_index]), classes):
        raise RuntimeError("train class 누락")
    if not np.array_equal(np.unique(target[valid_index]), classes):
        raise RuntimeError("valid class 누락")

    preprocessor = make_preprocessor(numeric_columns, categorical_columns)
    x_train = preprocessor.fit_transform(train.iloc[train_index][feature_columns]).astype(np.float32)
    x_valid = preprocessor.transform(train.iloc[valid_index][feature_columns]).astype(np.float32)
    x_test = preprocessor.transform(test[feature_columns]).astype(np.float32)
    y_train, y_valid = target[train_index], target[valid_index]
    if not all(np.isfinite(array).all() for array in (x_train, x_valid, x_test)):
        raise ValueError("preprocessing NaN/Inf")

    baseline = LogisticRegression(
        max_iter=500,
        class_weight="balanced",
        random_state=args.seed,
    )
    baseline.fit(x_train, y_train)
    base_probability = baseline.predict_proba(x_valid)[:, 1]
    base_threshold, base_score = choose_threshold(y_valid, base_probability)
    print("baseline threshold/macro-F1:", base_threshold, base_score)
    baseline_test_probability = baseline.predict_proba(x_test)[:, 1]
    baseline_prediction = (
        baseline_test_probability >= base_threshold
    ).astype(np.int64)
    checked_save(args.output, baseline_prediction, len(test), "baseline")
    print("first valid submission seconds:", time.monotonic() - problem_start)

    train_loader = make_loader(x_train, y_train, args.batch_size, True)
    valid_loader = make_loader(x_valid, y_valid, args.batch_size, False)
    model = BinaryMLP(x_train.shape[1]).to(device)
    model, best_epoch, threshold, score = train_with_validation(
        model,
        train_loader,
        valid_loader,
        positive_weight(y_train),
        device,
        args.epochs,
    )
    print("MLP best_epoch/threshold/macro-F1:", best_epoch, threshold, score)

    full_preprocessor = make_preprocessor(numeric_columns, categorical_columns)
    x_full = full_preprocessor.fit_transform(train[feature_columns]).astype(np.float32)
    x_test_full = full_preprocessor.transform(test[feature_columns]).astype(np.float32)
    if base_score >= score:
        selected_name = "logistic baseline"
        selected_score = base_score
        final_baseline = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            random_state=args.seed,
        )
        final_baseline.fit(x_full, target)
        prediction = (
            final_baseline.predict_proba(x_test_full)[:, 1] >= base_threshold
        ).astype(np.int64)
    else:
        selected_name = "PyTorch MLP"
        selected_score = score
        full_loader = make_loader(x_full, target, args.batch_size, True)
        test_loader = make_loader(x_test_full, None, args.batch_size, False)
        final_model = BinaryMLP(x_full.shape[1]).to(device)
        final_model = train_fixed_epochs(
            final_model,
            full_loader,
            positive_weight(target),
            device,
            best_epoch,
        )
        test_probability = predict_probability(final_model, test_loader, device)
        prediction = (test_probability >= threshold).astype(np.int64)
    print("selected model/validation macro-F1:", selected_name, selected_score)
    checked_save(args.output, prediction, len(test), "final")
    print(
        "hidden test macro-F1 (연습 채점용):",
        f1_score(hidden_test_target, prediction, average="macro", zero_division=0),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--groups", type=int, default=500)
    parser.add_argument("--rows-per-group", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--generate-only", action="store_true")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "integrated_mock_data",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).resolve().parent / "Submission_problem.npy",
    )
    return parser.parse_args()


if __name__ == "__main__":
    run_solution(parse_args())
