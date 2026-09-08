"""독립 문제 P01–P08 참고 구현. 먼저 starter를 직접 구현하세요."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image


def _selected(frame, columns):
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("DataFrame 필요")
    if not frame.columns.is_unique or len(set(columns)) != len(columns):
        raise ValueError("열 이름은 중복될 수 없습니다")
    for col in columns:
        if col not in frame:
            raise KeyError(col)
        if not pd.api.types.is_numeric_dtype(frame[col]):
            raise TypeError("선택 열은 수치형이어야 합니다")
        if np.isinf(frame[col].to_numpy(dtype=float)).any():
            raise ValueError("Inf 입력은 허용하지 않습니다")
    return frame.copy(deep=True)


def scale_sensor_columns(frame, columns):
    out = _selected(frame, columns)
    for col in columns:
        s = frame[col].astype("float64")
        lo, hi = s.min(), s.max()
        out[col] = s.where(s.isna(), 0.0) if hi == lo else (s - lo) / (hi - lo)
    return out


def robust_sensor_columns(frame, columns):
    out = _selected(frame, columns)
    for col in columns:
        s = frame[col].astype("float64")
        if s.dropna().empty:
            out[col] = s
            continue
        iqr = s.quantile(.75, interpolation="linear") - s.quantile(.25, interpolation="linear")
        out[col] = s.where(s.isna(), 0.0) if iqr == 0 else (s - s.median()) / iqr
    return out


def add_prior_average(frame, group_col, time_col, value_col, window=2):
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window는 양의 정수")
    out = _selected(frame, [value_col])
    name = f"{value_col}_prior_mean"
    if name in frame:
        raise ValueError("출력 열이 이미 존재합니다")
    for col in (group_col, time_col):
        if col not in frame:
            raise KeyError(col)
    if frame[group_col].isna().any() or frame[time_col].isna().any():
        raise ValueError("group/time 결측 불가")
    # 원본 index는 중복 가능. 위치 전용 임시 표로 안전하게 계산한다.
    work = pd.DataFrame({"g": frame[group_col].to_numpy(), "t": frame[time_col].to_numpy(),
        "v": frame[value_col].to_numpy(dtype=float), "pos": np.arange(len(frame))})
    work = work.sort_values(["g", "t", "pos"], kind="stable")
    value = work.groupby("g", sort=False)["v"].transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    result = np.full(len(frame), np.nan, dtype=np.float64)
    result[work["pos"].to_numpy()] = value.to_numpy()
    out[name] = result
    return out


def mean_logit_loss(logits, labels):
    x, y = np.asarray(logits), np.asarray(labels)
    if x.ndim != 2 or x.shape[0] < 1 or x.shape[1] < 1:
        raise ValueError("logits는 비어 있지 않은 (N,C)")
    if y.shape != (x.shape[0],) or not np.issubdtype(y.dtype, np.integer):
        raise ValueError("labels는 정수 (N,)")
    if not np.issubdtype(x.dtype, np.number) or np.iscomplexobj(x) or not np.isfinite(x).all():
        raise ValueError("유한한 실수 logits 필요")
    if np.any(y < 0) or np.any(y >= x.shape[1]):
        raise ValueError("label 범위 오류")
    z = x.astype(np.float64) - x.max(axis=1, keepdims=True)
    return float(np.mean(np.log(np.exp(z).sum(axis=1)) - z[np.arange(len(y)), y]))


def center_sensor_crop(image, crop_width, crop_height):
    if not isinstance(image, Image.Image):
        raise TypeError("PIL.Image 필요")
    if image.mode not in {"L", "RGB", "RGBA"}:
        raise ValueError("L/RGB/RGBA만 지원")
    for val, size in ((crop_width, image.width), (crop_height, image.height)):
        if isinstance(val, (bool, np.bool_)) or not isinstance(val, (int, np.integer)) or not 1 <= val <= size:
            raise ValueError("crop 크기는 이미지 이내의 양의 정수")
    left = (image.width - crop_width) // 2
    top = (image.height - crop_height) // 2
    return np.array(image.crop((left, top, left + crop_width, top + crop_height)), copy=True)


class SensorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(nn.Linear(12, 20), nn.BatchNorm1d(20), nn.ReLU(), nn.Dropout(.2), nn.Linear(20, 4))

    def forward(self, x):
        return self.network(x)


class SurfaceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(nn.Conv2d(1, 6, 3, padding=1, bias=False), nn.BatchNorm2d(6), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Conv2d(6, 10, 3, stride=2, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(10, 2)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))


class TripGRU(nn.Module):
    def __init__(self, n_features, hidden_size, n_outputs):
        super().__init__()
        for value in (n_features, hidden_size, n_outputs):
            if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
                raise ValueError("모델 차원은 양의 정수")
        self.gru = nn.GRU(n_features, hidden_size, num_layers=2, batch_first=True, bidirectional=True, dropout=.1)
        self.head = nn.Linear(2 * hidden_size, n_outputs)

    def forward(self, x):
        _, hidden = self.gru(x)
        return self.head(torch.cat([hidden[-2], hidden[-1]], dim=1))
