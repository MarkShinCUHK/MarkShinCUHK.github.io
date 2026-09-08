"""P01–P08: 함수 이름과 인수를 보존하고 TODO만 구현하세요."""
import numpy as np
import pandas as pd
import torch
from torch import nn
from PIL import Image


def scale_sensor_columns(frame, columns):
    raise NotImplementedError("P01: 선택 열 Min-Max")


def robust_sensor_columns(frame, columns):
    raise NotImplementedError("P02: median/IQR")


def add_prior_average(frame, group_col, time_col, value_col, window=2):
    raise NotImplementedError("P03: 현재 행을 제외한 과거 평균")


def mean_logit_loss(logits, labels):
    raise NotImplementedError("P04: 수치적으로 안정적인 CE")


def center_sensor_crop(image, crop_width, crop_height):
    raise NotImplementedError("P05: PIL 중앙 crop → 독립 NumPy 배열")


class SensorMLP(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("P06: 정확한 Sequential")

    def forward(self, x):
        raise NotImplementedError


class SurfaceCNN(nn.Module):
    def __init__(self):
        super().__init__()
        raise NotImplementedError("P07: features + head")

    def forward(self, x):
        raise NotImplementedError


class TripGRU(nn.Module):
    def __init__(self, n_features, hidden_size, n_outputs):
        super().__init__()
        raise NotImplementedError("P08: 양방향 2층 GRU")

    def forward(self, x):
        raise NotImplementedError
