"""HDAT-DS 실기용 복사·수정 템플릿.

사용법
------
1. 이 파일 전체를 실행하려 하지 말고, 문제 유형에 맞는 ``[TAG]`` 블록을 복사한다.
2. ``EDIT`` 주석이 붙은 값만 먼저 바꾼다.
3. 문제에서 지정한 함수명, 변수명, 반환형, 파일명을 항상 우선한다.

시험 중 이 파일 전체를 업로드·import하거나 end-to-end 함수를 무수정 호출하지 않는다.
필요한 작은 블록만 제공 skeleton에 옮겨 실제 열·split·metric·출력에 맞게 수정한다.
``fit_tabular_baseline``과 저장 helper는 모의시험·연습 검증용이다.

기준 환경: NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2,
PyTorch 2.7.0. 추가 라이브러리에 의존하지 않는다.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Literal, Sequence

import numpy as np
import pandas as pd


# %% [COMMON-SEED] 재현성 -----------------------------------------------------
def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # 완전 결정론은 느려질 수 있으므로 시험에서는 아래 두 줄을 선택적으로 사용한다.
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# %% [COMMON-EDA] 2분 점검 -----------------------------------------------------
def quick_audit(df: pd.DataFrame, target_cols: Sequence[str] = ()) -> dict:
    """출력 폭발을 피하면서 필수 정보만 반환한다."""
    targets = [c for c in target_cols if c in df.columns]
    return {
        "shape": df.shape,
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_columns": df.columns[df.columns.duplicated()].tolist(),
        "dtype_counts": df.dtypes.astype(str).value_counts().to_dict(),
        "missing_top20": df.isna().sum().sort_values(ascending=False).head(20),
        "nunique_top20": df.nunique(dropna=False).sort_values().head(20),
        "target_summary": {
            c: {
                "dtype": str(df[c].dtype),
                "missing": int(df[c].isna().sum()),
                "nunique": int(df[c].nunique(dropna=False)),
                "head_counts": df[c].value_counts(dropna=False).head(10).to_dict(),
            }
            for c in targets
        },
    }


def assert_frame_contract(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target_cols: Sequence[str],
    ignore_feature_cols: Sequence[str] = (),
) -> None:
    """학습/테스트 열과 중복 열을 빠르게 검사한다."""
    target_cols = list(target_cols)
    ignored = list(ignore_feature_cols)
    assert not train.columns.duplicated().any(), "train에 중복 column이 있습니다."
    assert not test.columns.duplicated().any(), "test에 중복 column이 있습니다."
    assert set(target_cols).issubset(train.columns), "target이 train에 없습니다."
    unknown_ignored = set(ignored) - set(train.columns)
    if unknown_ignored:
        raise KeyError(f"drop/ignore 열 오타: {sorted(unknown_ignored)}")
    leaked = set(target_cols).intersection(test.columns)
    assert not leaked, f"test에 target이 존재합니다: {sorted(leaked)}"
    expected = [c for c in train.columns if c not in target_cols and c not in ignored]
    missing = set(expected) - set(test.columns)
    extra = set(test.columns) - set(expected) - set(ignored)
    assert not missing, f"test에 필요한 feature가 없습니다: {sorted(missing)}"
    if extra:
        print("주의: test에만 있는 열을 무시합니다:", sorted(extra))


# %% [PROCESS-MINMAX] 선택 열 Min-Max -----------------------------------------
def minmax_selected(
    df: pd.DataFrame,
    columns: Sequence[str],
) -> pd.DataFrame:
    """원본/index/비대상 열을 보존하고 상수 열의 유효값을 0으로 만든다."""
    out = df.copy(deep=True)
    missing = [c for c in columns if c not in out.columns]
    if missing:
        raise KeyError(f"없는 열: {missing}")

    for col in columns:
        values = pd.to_numeric(out[col], errors="raise").astype(float)
        lo, hi = values.min(skipna=True), values.max(skipna=True)
        if pd.isna(lo) or pd.isna(hi):  # 전부 NaN
            out[col] = values
        elif math.isclose(float(hi), float(lo)):
            out[col] = values.where(values.isna(), 0.0)
        else:
            out[col] = (values - lo) / (hi - lo)
    return out


# %% [PROCESS-CLIP] train에서 경계를 학습한 이상치 clipping ------------------
def fit_iqr_bounds(
    train: pd.DataFrame,
    columns: Sequence[str],
    whisker: float = 1.5,
) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for col in columns:
        q1, q3 = train[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        bounds[col] = (float(q1 - whisker * iqr), float(q3 + whisker * iqr))
    return bounds


def apply_clip_bounds(
    df: pd.DataFrame,
    bounds: dict[str, tuple[float, float]],
) -> pd.DataFrame:
    out = df.copy(deep=True)
    for col, (lo, hi) in bounds.items():
        out[col] = out[col].clip(lo, hi)
    return out


# %% [PROCESS-DATETIME] 날짜 feature ------------------------------------------
def add_datetime_features(
    df: pd.DataFrame,
    columns: Sequence[str],
    drop_original: bool = True,
) -> pd.DataFrame:
    out = df.copy(deep=True)
    for col in columns:
        dt = pd.to_datetime(out[col], errors="coerce")
        out[f"{col}__year"] = dt.dt.year
        out[f"{col}__month"] = dt.dt.month
        out[f"{col}__day"] = dt.dt.day
        out[f"{col}__dow"] = dt.dt.dayofweek
        out[f"{col}__hour"] = dt.dt.hour
        out[f"{col}__minute"] = dt.dt.minute
        if drop_original:
            out = out.drop(columns=col)
    return out


def clean_tabular_values(df: pd.DataFrame) -> pd.DataFrame:
    """수치 Inf를 NaN으로, 범주형 혼합 타입을 일관된 문자열로 만든다."""
    out = df.copy(deep=True)
    numeric_cols = out.select_dtypes(include=["number", "bool"]).columns
    out.loc[:, numeric_cols] = out.loc[:, numeric_cols].replace([np.inf, -np.inf], np.nan)
    categorical_cols = [c for c in out.columns if c not in numeric_cols]
    for col in categorical_cols:
        s = out[col].astype("string").astype(object)
        out[col] = s.where(pd.notna(s), np.nan)
    return out


# %% [PROCESS-IMAGE] PIL crop --------------------------------------------------
def crop_to_numpy(image, box: tuple[int, int, int, int]) -> np.ndarray:
    """box=(left, upper, right, lower), 반환 shape=(H,W,C) 또는 (H,W)."""
    arr = np.asarray(image.crop(box)).copy()
    return arr


def conv_output_size(
    input_size: int,
    kernel_size: int,
    stride: int = 1,
    padding: int = 0,
    dilation: int = 1,
) -> int:
    return math.floor(
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1) / stride
        + 1
    )


# %% [TABULAR-CONFIG] 여기부터 EDIT -------------------------------------------
@dataclass
class TabularConfig:
    task: Literal["regression", "binary", "multiclass", "multilabel"] = "regression"  # EDIT
    target_cols: tuple[str, ...] = ("target",)  # EDIT
    drop_cols: tuple[str, ...] = ()  # ID, 누수 열. EDIT
    datetime_cols: tuple[str, ...] = ()  # EDIT
    time_col: str | None = None  # 시간순 validation이면 EDIT
    group_col: str | None = None  # 동일 개체 분리면 EDIT
    metric: str = "rmse"  # EDIT
    model_name: Literal["linear", "extra_trees"] = "extra_trees"  # EDIT
    encoding: Literal["onehot", "ordinal"] = "onehot"  # 고유값 많으면 ordinal
    valid_size: float = 0.2
    seed: int = 42


# %% [TABULAR-SPLIT] validation ------------------------------------------------
def safe_train_valid_indices(
    X: pd.DataFrame,
    y,
    task: str,
    valid_size: float = 0.2,
    seed: int = 42,
    time_col: str | None = None,
    time_values=None,
    groups=None,
) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import GroupShuffleSplit, train_test_split

    n = len(X)
    all_idx = np.arange(n)
    if n < 2:
        raise ValueError("split에는 최소 2개 행이 필요합니다.")
    if len(np.asarray(y)) != n:
        raise ValueError("y와 X 길이가 다릅니다.")
    if not 0 < valid_size < 1:
        raise ValueError("valid_size는 0과 1 사이여야 합니다.")

    has_time = time_values is not None or time_col is not None
    if has_time and groups is not None:
        raise ValueError("time/group 중 실제 평가 목표에 맞는 split 하나만 명시하세요.")

    if has_time:
        raw_time = np.asarray(time_values if time_values is not None else X[time_col])
        if len(raw_time) != n:
            raise ValueError("time과 X 길이가 다릅니다.")
        parsed_time = pd.Series(pd.to_datetime(raw_time, errors="coerce"))
        if parsed_time.isna().any():
            raise ValueError("time parse 실패/결측 행을 먼저 처리하세요.")
        order = np.lexsort((all_idx, parsed_time.to_numpy()))
        cut = max(1, min(n - 1, int(n * (1 - valid_size))))
        return order[:cut], order[cut:]

    if groups is not None:
        groups = np.asarray(groups)
        if len(groups) != n:
            raise ValueError("groups와 X 길이가 다릅니다.")
        if pd.isna(groups).any():
            raise ValueError("group 결측 행을 먼저 처리하세요.")
        if len(pd.unique(groups)) < 2:
            raise ValueError("group split에는 최소 2개 group이 필요합니다.")
        splitter = GroupShuffleSplit(n_splits=1, test_size=valid_size, random_state=seed)
        tr, va = next(splitter.split(X, y, groups=groups))
        return np.asarray(tr), np.asarray(va)

    stratify = None
    y_arr = np.asarray(y)
    if task in {"binary", "multiclass"} and y_arr.ndim == 1:
        counts = pd.Series(y_arr).value_counts(dropna=False)
        n_valid = int(math.ceil(n * valid_size))
        n_train = n - n_valid
        if (
            len(counts) > 1
            and counts.min() >= 2
            and n_valid >= len(counts)
            and n_train >= len(counts)
        ):
            stratify = y_arr

    tr, va = train_test_split(
        all_idx,
        test_size=valid_size,
        random_state=seed,
        shuffle=True,
        stratify=stratify,
    )
    return np.asarray(tr), np.asarray(va)


# %% [TABULAR-PREPROCESS] 결측·범주형 -----------------------------------------
def make_preprocessor(
    X: pd.DataFrame,
    encoding: Literal["onehot", "ordinal"] = "onehot",
    scale_numeric: bool = False,
):
    from sklearn.compose import ColumnTransformer
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler

    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_steps: list[tuple[str, object]] = [
        (
            "imputer",
            SimpleImputer(
                strategy="median",
                add_indicator=True,
                keep_empty_features=True,
            ),
        )
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    if encoding == "onehot":
        encoder = OneHotEncoder(
            handle_unknown="ignore", sparse_output=True, dtype=np.float32
        )
    else:
        encoder = OrdinalEncoder(
            handle_unknown="use_encoded_value",
            unknown_value=-1,
            encoded_missing_value=-2,
        )

    transformers = []
    if numeric_cols:
        transformers.append(("num", Pipeline(numeric_steps), numeric_cols))
    if categorical_cols:
        transformers.append(
            (
                "cat",
                Pipeline(
                    [
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="constant",
                                fill_value="__MISSING__",
                                keep_empty_features=True,
                            ),
                        ),
                        ("encoder", encoder),
                    ]
                ),
                categorical_cols,
            )
        )
    if not transformers:
        raise ValueError("사용할 feature가 없습니다.")

    return ColumnTransformer(transformers, remainder="drop")


def make_tabular_estimator(
    task: str,
    model_name: str = "extra_trees",
    seed: int = 42,
    multioutput: bool = False,
):
    if model_name == "linear":
        if task == "regression":
            from sklearn.linear_model import Ridge

            model = Ridge(alpha=1.0)
        else:
            from sklearn.linear_model import LogisticRegression

            model = LogisticRegression(
                max_iter=1000,
                class_weight=None,
                n_jobs=1 if multioutput else -1,
            )
    elif model_name == "extra_trees":
        if task == "regression":
            from sklearn.ensemble import ExtraTreesRegressor

            model = ExtraTreesRegressor(
                n_estimators=100,
                min_samples_leaf=5,
                max_features=1.0,
                n_jobs=-1,
                random_state=seed,
            )
        else:
            from sklearn.ensemble import ExtraTreesClassifier

            model = ExtraTreesClassifier(
                n_estimators=100,
                min_samples_leaf=5,
                max_features="sqrt",
                class_weight=None,
                n_jobs=1 if multioutput else -1,
                random_state=seed,
            )
    else:
        raise ValueError(f"지원하지 않는 model_name: {model_name}")

    if multioutput and task in {"binary", "multiclass", "multilabel"}:
        from sklearn.multioutput import MultiOutputClassifier

        model = MultiOutputClassifier(model, n_jobs=-1)
    return model


# %% [METRIC] -----------------------------------------------------------------
def evaluate_predictions(
    y_true,
    y_pred,
    metric: str,
    y_score=None,
) -> float:
    """대표 metric 계산.

    다중출력 분류의 accuracy/F1은 열별 점수 평균이다. 실제 문제의 공식 산식이
    다르면 이 helper를 쓰지 말고 명시된 산식을 그대로 구현한다.
    """
    from sklearn.metrics import (
        accuracy_score,
        f1_score,
        mean_absolute_error,
        mean_squared_error,
        mean_squared_log_error,
        roc_auc_score,
    )

    metric = metric.lower()
    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)
    if metric == "mse":
        return float(mean_squared_error(y_true, y_pred))
    if metric == "rmse":
        return float(mean_squared_error(y_true, y_pred) ** 0.5)
    if metric == "mae":
        return float(mean_absolute_error(y_true, y_pred))
    if metric == "rmsle":
        y_true_arr = np.asarray(y_true)
        if np.nanmin(y_true_arr) < 0:
            raise ValueError("RMSLE는 음수 target에 사용할 수 없습니다.")
        clipped = np.clip(np.asarray(y_pred), 0, None)
        return float(mean_squared_log_error(y_true_arr, clipped) ** 0.5)
    if metric == "accuracy":
        if y_true_arr.ndim == 2 and y_true_arr.shape[1] > 1:
            return float(
                np.mean(
                    [accuracy_score(y_true_arr[:, j], y_pred_arr[:, j])
                     for j in range(y_true_arr.shape[1])]
                )
            )
        return float(accuracy_score(y_true, y_pred))
    if metric in {"f1", "f1_macro"}:
        if y_true_arr.ndim == 2 and y_true_arr.shape[1] > 1:
            return float(
                np.mean(
                    [f1_score(y_true_arr[:, j], y_pred_arr[:, j], average="macro")
                     for j in range(y_true_arr.shape[1])]
                )
            )
        return float(f1_score(y_true, y_pred, average="macro"))
    if metric in {"auc", "roc_auc"}:
        score = y_score if y_score is not None else y_pred
        score_arr = np.asarray(score)
        if y_true_arr.ndim == 2:
            if score_arr.shape != y_true_arr.shape:
                raise ValueError(
                    f"multilabel AUC score shape 오류: {score_arr.shape} vs {y_true_arr.shape}"
                )
            return float(roc_auc_score(y_true_arr, score_arr, average="macro"))
        if score_arr.ndim == 2 and score_arr.shape[1] > 2:
            return float(roc_auc_score(y_true, score_arr, multi_class="ovr"))
        if score_arr.ndim == 2:
            score_arr = score_arr[:, 1]
        return float(roc_auc_score(y_true, score_arr))
    raise ValueError(f"지원하지 않는 metric: {metric}")


def assert_class_fold_coverage(
    y,
    train_indices,
    valid_indices,
    task: str,
    metric: str,
) -> None:
    """time/group split에서 validation-only class와 AUC 불능 fold를 차단한다."""
    if task not in {"binary", "multiclass", "multilabel"}:
        return
    y_arr = np.asarray(y)
    y_2d = y_arr[:, None] if y_arr.ndim == 1 else y_arr
    for column in range(y_2d.shape[1]):
        train_classes = set(pd.unique(y_2d[np.asarray(train_indices), column]))
        valid_classes = set(pd.unique(y_2d[np.asarray(valid_indices), column]))
        if not valid_classes.issubset(train_classes):
            unseen = valid_classes - train_classes
            raise ValueError(f"validation에 train에 없는 class가 있습니다: {unseen}")
        if task in {"binary", "multiclass"} and len(train_classes) < 2:
            raise ValueError("classification train fold에 class가 하나뿐입니다.")
        if metric.lower() in {"auc", "roc_auc"}:
            if len(train_classes) < 2 or len(valid_classes) < 2:
                raise ValueError("ROC-AUC는 train/validation 양쪽에 두 class 이상 필요합니다.")
            if task == "multiclass" and valid_classes != train_classes:
                raise ValueError("multiclass ROC-AUC는 양 fold의 class 집합을 맞추세요.")


# %% [TABULAR-FIT] 표형 baseline 전체 흐름 ------------------------------------
def fit_tabular_baseline(
    train: pd.DataFrame,
    test: pd.DataFrame,
    cfg: TabularConfig,
):
    """모의시험·연습 검증용 end-to-end wrapper.

    실제 시험에서는 이 함수를 무수정 호출하지 말고 split/preprocess/model/metric
    블록을 제공 skeleton과 문제 계약에 맞춰 수정한다.
    """
    from sklearn.pipeline import Pipeline

    seed_everything(cfg.seed)
    if set(cfg.target_cols) & (set(cfg.drop_cols) | set(cfg.datetime_cols)):
        raise ValueError("target을 drop_cols/datetime_cols에 넣지 마세요.")
    if set(cfg.drop_cols) & set(cfg.datetime_cols):
        raise ValueError("같은 열을 drop_cols/datetime_cols에 동시에 넣지 마세요.")
    assert_frame_contract(train, test, cfg.target_cols, cfg.drop_cols)

    target_cols = list(cfg.target_cols)
    y = train[target_cols].copy()
    if len(target_cols) == 1:
        y = y.iloc[:, 0]
    if pd.isna(np.asarray(y)).any():
        raise ValueError("target에 결측치가 있습니다. 문제 지시에 따라 먼저 처리하세요.")

    X = train.drop(columns=target_cols).copy()
    X_test = test.copy()

    groups = X[cfg.group_col].copy() if cfg.group_col else None
    time_values = X[cfg.time_col].copy() if cfg.time_col else None
    X = add_datetime_features(X, cfg.datetime_cols)
    X_test = add_datetime_features(X_test, cfg.datetime_cols)

    missing_drops = [
        c for c in cfg.drop_cols if c not in X.columns and c not in cfg.datetime_cols
    ]
    if missing_drops:
        raise KeyError(f"전처리 후 drop 열 오타: {missing_drops}")
    drops = [c for c in cfg.drop_cols if c in X.columns]
    X = X.drop(columns=drops)
    X_test = X_test.drop(columns=[c for c in drops if c in X_test.columns])
    missing_features = set(X.columns) - set(X_test.columns)
    if missing_features:
        raise KeyError(f"test에 필요한 feature가 없습니다: {sorted(missing_features)}")
    X_test = X_test.reindex(columns=X.columns)
    X = clean_tabular_values(X)
    X_test = clean_tabular_values(X_test)

    tr_idx, va_idx = safe_train_valid_indices(
        X,
        y,
        cfg.task,
        valid_size=cfg.valid_size,
        seed=cfg.seed,
        time_values=time_values,
        groups=groups,
    )
    assert_class_fold_coverage(y, tr_idx, va_idx, cfg.task, cfg.metric)

    multioutput = np.asarray(y).ndim == 2 and np.asarray(y).shape[1] > 1
    scale_numeric = cfg.model_name == "linear"
    pipe = Pipeline(
        [
            ("prep", make_preprocessor(X.iloc[tr_idx], cfg.encoding, scale_numeric)),
            (
                "model",
                make_tabular_estimator(
                    cfg.task,
                    cfg.model_name,
                    cfg.seed,
                    multioutput=multioutput,
                ),
            ),
        ]
    )
    pipe.fit(X.iloc[tr_idx], np.asarray(y)[tr_idx])
    valid_pred = pipe.predict(X.iloc[va_idx])

    valid_score = None
    if cfg.metric.lower() in {"auc", "roc_auc"} and hasattr(pipe, "predict_proba"):
        probability = pipe.predict_proba(X.iloc[va_idx])
        if multioutput:
            if cfg.task != "multilabel":
                raise ValueError("다중출력 AUC는 task='multilabel' 0/1 indicator만 지원합니다.")
            classes_per_target = pipe.named_steps["model"].classes_
            valid_score = np.column_stack([
                probability[j][:, list(classes_per_target[j]).index(1)]
                for j in range(len(classes_per_target))
            ])
        else:
            valid_score = probability
    metric_value = evaluate_predictions(
        np.asarray(y)[va_idx],
        valid_pred,
        cfg.metric,
        y_score=valid_score,
    )
    print(f"validation {cfg.metric} = {metric_value:.6f}")

    # validation 확인 후 전체 train으로 재학습
    pipe.fit(X, y)
    test_pred = pipe.predict(X_test)
    if cfg.metric.lower() == "rmsle":
        test_pred = np.clip(test_pred, 0, None)
    return pipe, np.asarray(test_pred), metric_value


# %% [TIME-WINDOW] 긴 시계열을 (N,T,F) window로 -------------------------------
def make_point_forecast_windows(
    features: np.ndarray,
    targets: np.ndarray,
    lookback: int,
    horizon: int = 1,
    stride: int = 1,
    assume_sorted: bool = False,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """과거 lookback개로 마지막 관측점에서 horizon칸 뒤의 target을 예측한다.

    반환: X=(N, lookback, F), y=(N, D), target_indices=(N,)
    예: 100Hz에서 1초 뒤면 horizon=100. 시간 열을 stable sort하고 행 대응을
    보존한 뒤에만 assume_sorted=True로 호출한다.
    """
    if not assume_sorted:
        raise ValueError("시간순 stable sort를 확인한 뒤 assume_sorted=True로 호출하세요.")
    Xv = np.asarray(features)
    yv = np.asarray(targets)
    if Xv.ndim == 1:
        Xv = Xv[:, None]
    elif Xv.ndim != 2:
        raise ValueError("features는 (T,F)여야 합니다.")
    if yv.ndim == 1:
        yv = yv[:, None]
    elif yv.ndim != 2:
        raise ValueError("targets는 (T,) 또는 (T,D)여야 합니다.")
    if len(Xv) != len(yv):
        raise ValueError("features와 targets 길이가 다릅니다.")
    if lookback < 1 or horizon < 0 or stride < 1:
        raise ValueError("lookback/stride는 1 이상, horizon은 0 이상이어야 합니다.")

    last_start = len(Xv) - lookback - horizon
    if last_start < 0:
        return (
            np.empty((0, lookback, Xv.shape[1]), dtype=Xv.dtype),
            np.empty((0, yv.shape[1]), dtype=yv.dtype),
            np.empty((0,), dtype=int),
        )

    starts = np.arange(0, last_start + 1, stride)
    target_idx = starts + lookback - 1 + horizon
    Xw = np.stack([Xv[s : s + lookback] for s in starts])
    yw = yv[target_idx]
    return Xw, yw, target_idx


def split_raw_time_then_window(
    features: np.ndarray,
    targets: np.ndarray,
    train_ratio: float,
    lookback: int,
    horizon: int,
    stride: int = 1,
    gap: int = 0,
    assume_sorted: bool = False,
):
    """target 시점으로 시간 분할하되 validation이 직전 과거 context를 쓸 수 있게 한다.

    gap>0이면 경계 양쪽 target을 제외해 overlapping window의 낙관성을 줄인다.
    """
    if not assume_sorted:
        raise ValueError("시간순 stable sort를 확인한 뒤 assume_sorted=True로 호출하세요.")
    n = len(features)
    if n < 2 or not 0 < train_ratio < 1 or gap < 0:
        raise ValueError("행 수/train_ratio/gap을 확인하세요.")
    cut = max(1, min(n - 1, int(n * train_ratio)))
    Xw, yw, target_idx = make_point_forecast_windows(
        features, targets, lookback, horizon, stride, assume_sorted=True
    )
    train_mask = target_idx < max(cut - gap, 0)
    valid_mask = target_idx >= min(cut + gap, n)
    train = (Xw[train_mask], yw[train_mask], target_idx[train_mask])
    valid = (Xw[valid_mask], yw[valid_mask], target_idx[valid_mask])
    if len(train[0]) == 0 or len(valid[0]) == 0:
        raise ValueError("train/validation window가 비었습니다. ratio/lookback/horizon/gap을 확인하세요.")
    return train, valid, cut


def make_grouped_point_forecast_windows(
    features: np.ndarray,
    targets: np.ndarray,
    groups,
    lookback: int,
    horizon: int = 1,
    stride: int = 1,
    times=None,
    assume_sorted: bool = False,
):
    """group별로 정렬·windowing하여 차량/주행 경계를 넘지 않게 한다.

    반환 target_original_indices로 원래 행과의 대응을 확인할 수 있다. times=None이면
    각 group 내부 입력 순서가 이미 시간순일 때만 assume_sorted=True로 호출한다.
    """
    Xv, yv, gv = np.asarray(features), np.asarray(targets), np.asarray(groups)
    if len(Xv) != len(yv) or len(Xv) != len(gv):
        raise ValueError("features/targets/groups 길이가 다릅니다.")
    if pd.isna(gv).any():
        raise ValueError("group 결측 행을 먼저 처리하세요.")
    if times is not None and len(times) != len(Xv):
        raise ValueError("times와 features 길이가 다릅니다.")
    if times is None and not assume_sorted:
        raise ValueError("times를 주거나 group 내부 정렬 확인 후 assume_sorted=True로 호출하세요.")
    tv = None if times is None else pd.to_datetime(np.asarray(times), errors="coerce")
    if tv is not None and pd.isna(tv).any():
        raise ValueError("group time parse 실패/결측 행을 먼저 처리하세요.")

    X_parts, y_parts, index_parts, group_parts = [], [], [], []
    for group in pd.unique(gv):
        original_idx = np.flatnonzero(gv == group)
        if tv is not None:
            original_idx = original_idx[np.argsort(tv[original_idx], kind="stable")]
        Xg, yg, local_target_idx = make_point_forecast_windows(
            Xv[original_idx], yv[original_idx], lookback, horizon, stride,
            assume_sorted=True,
        )
        if len(Xg) == 0:
            continue
        X_parts.append(Xg)
        y_parts.append(yg)
        index_parts.append(original_idx[local_target_idx])
        group_parts.append(np.repeat(group, len(Xg)))

    if not X_parts:
        raise ValueError("어느 group에서도 window를 만들 수 없습니다.")
    return (
        np.concatenate(X_parts),
        np.concatenate(y_parts),
        np.concatenate(index_parts),
        np.concatenate(group_parts),
    )


def fit_scale_3d(X_train: np.ndarray):
    from sklearn.preprocessing import StandardScaler

    if X_train.ndim != 3:
        raise ValueError("X_train shape은 (N,T,F)여야 합니다.")
    scaler = StandardScaler()
    scaler.fit(X_train.reshape(-1, X_train.shape[-1]))
    return scaler


def transform_scale_3d(X: np.ndarray, scaler) -> np.ndarray:
    shape = X.shape
    return scaler.transform(X.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)


# %% [SUBMISSION] npy/csv ------------------------------------------------------
def validate_prediction_array(
    pred,
    expected_shape: tuple[int, ...],
    allow_nan: bool = False,
    require_numeric: bool = True,
) -> np.ndarray:
    arr = np.asarray(pred)
    if arr.shape != tuple(expected_shape):
        raise ValueError(f"출력 shape 오류: expected={expected_shape}, actual={arr.shape}")
    is_numeric = np.issubdtype(arr.dtype, np.number)
    if require_numeric and not is_numeric:
        raise TypeError(f"수치 제출이 필요한데 dtype={arr.dtype}입니다.")
    if not allow_nan:
        if is_numeric:
            bad_mask = ~np.isfinite(arr)
        else:
            bad_mask = pd.isna(arr)
        if np.any(bad_mask):
            raise ValueError(f"NaN/Inf가 {int(np.sum(bad_mask))}개 있습니다.")
    return arr


def save_npy_submission(
    pred,
    path: str | Path,
    expected_shape: tuple[int, ...],
    dtype=None,
) -> np.ndarray:
    """연습용 helper. 시험에 제공 저장 셀이 있으면 그 셀을 그대로 사용한다."""
    arr = validate_prediction_array(pred, expected_shape)
    if dtype is not None:
        with np.errstate(over="ignore", invalid="ignore"):
            arr = arr.astype(dtype)
        arr = validate_prediction_array(arr, expected_shape)
    actual_path = Path(path)
    if actual_path.suffix != ".npy":
        actual_path = Path(f"{actual_path}.npy")
    np.save(actual_path, arr, allow_pickle=False)
    loaded = np.load(actual_path, allow_pickle=False)
    if loaded.shape != arr.shape or not np.allclose(loaded, arr):
        raise IOError("저장 후 재검증에 실패했습니다.")
    print(f"saved: {actual_path}, shape={arr.shape}, dtype={arr.dtype}")
    return arr


def save_csv_submission(
    sample_submission: pd.DataFrame,
    pred,
    target_cols: Sequence[str],
    path: str | Path,
) -> pd.DataFrame:
    out = sample_submission.copy(deep=True)
    target_cols = list(target_cols)
    if not out.columns.is_unique:
        raise ValueError("sample_submission에 중복 column이 있습니다.")
    missing_targets = [c for c in target_cols if c not in out.columns]
    if missing_targets:
        raise KeyError(f"sample_submission에 target 열이 없습니다: {missing_targets}")
    arr = np.asarray(pred)
    if len(target_cols) == 1:
        allowed_shapes = {(len(out),), (len(out), 1)}
        if arr.shape not in allowed_shapes:
            raise ValueError(
                f"단일 target prediction shape은 {allowed_shapes} 중 하나여야 합니다: {arr.shape}"
            )
        arr = arr.reshape(-1)
        out[target_cols[0]] = arr
    else:
        arr = validate_prediction_array(
            arr,
            (len(out), len(target_cols)),
            require_numeric=False,
        )
        out.loc[:, target_cols] = arr
    if out[target_cols].isna().any().any():
        raise ValueError("submission target에 NaN이 있습니다.")
    numeric_targets = out[target_cols].select_dtypes(include=["number"])
    if numeric_targets.shape[1] and not np.isfinite(numeric_targets.to_numpy()).all():
        raise ValueError("submission target에 Inf가 있습니다.")
    out.to_csv(path, index=False)
    loaded = pd.read_csv(path)
    if list(loaded.columns) != list(out.columns) or len(loaded) != len(out):
        raise IOError("CSV 저장 후 column/row 검증 실패")
    if loaded[target_cols].isna().any().any():
        raise IOError("CSV 재로드 후 target에 NaN이 있습니다.")
    loaded_numeric = loaded[target_cols].select_dtypes(include=["number"])
    if loaded_numeric.shape[1] and not np.isfinite(loaded_numeric.to_numpy()).all():
        raise IOError("CSV 재로드 후 target에 NaN/Inf가 있습니다.")
    print(f"saved: {path}, shape={out.shape}")
    return out


# %% [TORCH-OPTIONAL] PyTorch가 있을 때만 정의 --------------------------------
try:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # 이 파일을 sklearn 전용으로 열 때도 import 가능하게 함
    torch = None
    nn = None
    DataLoader = None
    TensorDataset = None


if torch is not None:

    # %% [TORCH-MLP] X=(N,F)
    class MLP(nn.Module):
        def __init__(
            self,
            n_features: int,
            out_dim: int,
            hidden: Sequence[int] = (128, 64),
            dropout: float = 0.1,
        ):
            super().__init__()
            layers: list[nn.Module] = []
            in_dim = n_features
            for h in hidden:
                layers.extend(
                    [
                        nn.Linear(in_dim, h),
                        nn.ReLU(),
                        nn.Dropout(dropout),
                    ]
                )
                in_dim = h
            layers.append(nn.Linear(in_dim, out_dim))
            self.net = nn.Sequential(*layers)

        def forward(self, x):
            return self.net(x.flatten(1) if x.ndim > 2 else x)


    # %% [TORCH-CNN1D] 입력 X=(N,T,F), 내부에서 (N,F,T)로 변환
    class CNN1D(nn.Module):
        def __init__(
            self,
            n_features: int,
            out_dim: int,
            channels: Sequence[int] = (64, 128),
            dropout: float = 0.1,
        ):
            super().__init__()
            if len(channels) != 2:
                raise ValueError("기본 CNN1D는 channels=(c1,c2) 두 값을 받습니다.")
            c1, c2 = channels
            self.features = nn.Sequential(
                nn.Conv1d(n_features, c1, kernel_size=5, padding=2),
                nn.GroupNorm(1, c1),
                nn.ReLU(),
                nn.Conv1d(c1, c2, kernel_size=3, padding=1),
                nn.GroupNorm(1, c2),
                nn.ReLU(),
                nn.AdaptiveAvgPool1d(1),
            )
            self.head = nn.Sequential(
                nn.Flatten(),
                nn.Dropout(dropout),
                nn.Linear(c2, out_dim),
            )

        def forward(self, x):
            x = x.transpose(1, 2)  # (N,T,F) -> (N,F,T)
            return self.head(self.features(x))


    # %% [TORCH-RNN] 입력 X=(N,T,F)
    class SequenceRNN(nn.Module):
        def __init__(
            self,
            n_features: int,
            out_dim: int,
            hidden_size: int = 64,
            num_layers: int = 1,
            kind: Literal["gru", "lstm"] = "gru",
            bidirectional: bool = False,
            dropout: float = 0.0,
        ):
            super().__init__()
            if kind not in {"gru", "lstm"}:
                raise ValueError("kind는 'gru' 또는 'lstm'이어야 합니다.")
            rnn_cls = nn.GRU if kind == "gru" else nn.LSTM
            self.rnn = rnn_cls(
                input_size=n_features,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            direction = 2 if bidirectional else 1
            self.num_layers = num_layers
            self.num_directions = direction
            self.head = nn.Linear(hidden_size * direction, out_dim)

        def forward(self, x):
            _, state = self.rnn(x)
            h_n = state[0] if isinstance(state, tuple) else state
            batch_size = x.shape[0]
            h_n = h_n.reshape(
                self.num_layers,
                self.num_directions,
                batch_size,
                -1,
            )[-1]
            h = h_n.transpose(0, 1).reshape(batch_size, -1)
            return self.head(h)


    # %% [TORCH-AE] 표형/flatten feature 이상탐지
    class Autoencoder(nn.Module):
        def __init__(
            self,
            input_dim: int,
            latent_dim: int = 16,
            hidden_dim: int = 64,
        ):
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, latent_dim),
            )
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def forward(self, x):
            return self.decoder(self.encoder(x))


    # %% [TORCH-VAE] 표형/flatten 입력 VAE
    class VariationalAutoencoder(nn.Module):
        def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dim: int = 64):
            super().__init__()
            self.encoder = nn.Sequential(nn.Linear(input_dim, hidden_dim), nn.ReLU())
            self.mu = nn.Linear(hidden_dim, latent_dim)
            self.logvar = nn.Linear(hidden_dim, latent_dim)
            self.decoder = nn.Sequential(
                nn.Linear(latent_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, input_dim),
            )

        def reparameterize(self, mu, logvar):
            if self.training:
                std = torch.exp(0.5 * logvar)
                return mu + torch.randn_like(std) * std
            return mu

        def forward(self, x):
            hidden = self.encoder(x)
            mu, logvar = self.mu(hidden), self.logvar(hidden)
            z = self.reparameterize(mu, logvar)
            return self.decoder(z), mu, logvar


    def vae_loss(reconstruction, x, mu, logvar, beta: float = 1.0):
        reconstruction_loss = nn.functional.mse_loss(reconstruction, x, reduction="mean")
        kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        return reconstruction_loss + beta * kl, reconstruction_loss, kl


    # %% [TORCH-GAN] 표형/flatten 입력 최소 GAN block
    class MLPGenerator(nn.Module):
        def __init__(self, noise_dim: int, output_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(noise_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, output_dim),
            )

        def forward(self, z):
            return self.net(z)


    class MLPDiscriminator(nn.Module):
        def __init__(self, input_dim: int, hidden_dim: int = 128):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(input_dim, hidden_dim),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.LeakyReLU(0.2),
                nn.Linear(hidden_dim // 2, 1),
            )

        def forward(self, x):
            return self.net(x)  # raw logit; BCEWithLogitsLoss 사용


    # %% [TIME-LAZY-DATASET] 큰 시계열용: window 전체를 메모리에 만들지 않는다.
    class LazyWindowDataset(torch.utils.data.Dataset):
        def __init__(
            self,
            features,
            targets=None,
            lookback: int = 20,
            horizon: int = 1,
            stride: int = 1,
            starts=None,
            expected_n: int | None = None,
            assume_sorted: bool = False,
        ):
            if not assume_sorted:
                raise ValueError(
                    "시간순 stable sort를 확인한 뒤 assume_sorted=True로 호출하세요."
                )
            X_arr = np.asarray(features)
            if X_arr.ndim == 1:
                X_arr = X_arr[:, None]
            if X_arr.ndim != 2:
                raise ValueError("features는 (T,F)여야 합니다.")
            y_arr = None if targets is None else np.asarray(targets)
            if y_arr is not None and len(y_arr) != len(X_arr):
                raise ValueError("features와 targets 길이가 다릅니다.")
            if y_arr is not None and y_arr.ndim not in {1, 2}:
                raise ValueError("targets는 (T,) 또는 (T,D)여야 합니다.")
            self.X = torch.as_tensor(X_arr, dtype=torch.float32)
            self.y = None if y_arr is None else torch.as_tensor(
                y_arr, dtype=torch.float32
            )
            if self.y is not None and self.y.ndim == 1:
                self.y = self.y[:, None]
            self.lookback = int(lookback)
            self.horizon = int(horizon)
            if self.lookback < 1 or self.horizon < 0 or stride < 1:
                raise ValueError("lookback/stride는 1 이상, horizon은 0 이상이어야 합니다.")
            inference_mode = self.y is None
            last_start = len(self.X) - self.lookback
            if not inference_mode:
                last_start -= self.horizon
            if starts is None:
                if inference_mode:
                    raise ValueError(
                        "targets=None인 inference는 submission 행과 대응하는 starts를 명시하세요."
                    )
                self.starts = np.arange(0, max(last_start + 1, 0), stride, dtype=int)
            else:
                self.starts = np.asarray(starts, dtype=int).reshape(-1)
            if len(self.starts):
                if self.starts.min() < 0 or self.starts.max() > last_start:
                    raise IndexError("window start가 유효 범위를 벗어났습니다.")
            if expected_n is not None and len(self.starts) != expected_n:
                raise ValueError(
                    f"window 행 수 오류: expected={expected_n}, actual={len(self.starts)}"
                )

        def __len__(self):
            return len(self.starts)

        def __getitem__(self, idx):
            start = int(self.starts[idx])
            x = self.X[start : start + self.lookback]
            if self.y is None:
                return x
            target_idx = start + self.lookback - 1 + self.horizon
            return x, self.y[target_idx]


    # %% [TORCH-IMAGE] 입력 X=(N,C,H,W), 크기는 adaptive pooling으로 자유
    class SmallImageCNN(nn.Module):
        def __init__(self, in_channels: int, out_dim: int):
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(in_channels, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d(1),
            )
            self.head = nn.Linear(128, out_dim)

        def forward(self, x):
            return self.head(self.features(x).flatten(1))


    # %% [TORCH-RESIDUAL] ResNet형 2D residual block
    class ResidualBlock2D(nn.Module):
        def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
            super().__init__()
            self.main = nn.Sequential(
                nn.Conv2d(
                    in_channels,
                    out_channels,
                    kernel_size=3,
                    stride=stride,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(
                    out_channels,
                    out_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(out_channels),
            )
            if stride != 1 or in_channels != out_channels:
                self.skip = nn.Sequential(
                    nn.Conv2d(
                        in_channels,
                        out_channels,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm2d(out_channels),
                )
            else:
                self.skip = nn.Identity()
            self.activation = nn.ReLU()

        def forward(self, x):
            return self.activation(self.main(x) + self.skip(x))


    # %% [TORCH-TRANSFORMER] 입력 X=(N,T,F)
    class TransformerSequenceModel(nn.Module):
        def __init__(
            self,
            n_features: int,
            out_dim: int,
            d_model: int = 64,
            nhead: int = 4,
            num_layers: int = 2,
            dim_feedforward: int = 128,
            dropout: float = 0.1,
            max_len: int = 512,
        ):
            super().__init__()
            if d_model % nhead != 0:
                raise ValueError("d_model은 nhead로 나누어져야 합니다.")
            self.max_len = max_len
            self.input_projection = nn.Linear(n_features, d_model)
            self.position = nn.Parameter(torch.zeros(1, max_len, d_model))
            layer = nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True,
                norm_first=False,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
            self.norm = nn.LayerNorm(d_model)
            self.head = nn.Linear(d_model, out_dim)

        def forward(self, x, padding_mask=None):
            if x.ndim != 3:
                raise ValueError(f"expected (N,T,F), got {tuple(x.shape)}")
            length = x.shape[1]
            if length > self.max_len:
                raise ValueError("sequence가 max_len보다 깁니다.")
            h = self.input_projection(x) + self.position[:, :length]
            h = self.encoder(h, src_key_padding_mask=padding_mask)
            if padding_mask is None:
                pooled = h.mean(dim=1)
            else:
                valid = (~padding_mask).unsqueeze(-1).to(h.dtype)
                pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1)
            return self.head(self.norm(pooled))


    def make_tensor_loader(
        X,
        y=None,
        task: Literal["regression", "binary", "multiclass", "multilabel"] = "regression",
        batch_size: int = 128,
        shuffle: bool = False,
    ):
        if hasattr(X, "tocsr"):
            raise TypeError(
                "scipy sparse를 PyTorch로 직접 넘길 수 없습니다. 작은 경우만 dense로, "
                "큰 경우 sklearn 선형모델 또는 dense/ordinal 전처리를 사용하세요."
            )
        X_tensor = torch.as_tensor(np.asarray(X), dtype=torch.float32)
        if y is None:
            dataset = TensorDataset(X_tensor)
        else:
            y_arr = np.asarray(y)
            if len(y_arr) != len(X_tensor):
                raise ValueError("X와 y 길이가 다릅니다.")
            if task == "multiclass":
                y_tensor = torch.as_tensor(y_arr.reshape(-1), dtype=torch.long)
            else:
                if task == "binary" and not set(np.unique(y_arr)).issubset({0, 1}):
                    raise ValueError("binary target은 먼저 0/1로 mapping하세요.")
                if task == "multilabel" and not set(np.unique(y_arr)).issubset({0, 1}):
                    raise ValueError("multilabel target은 0/1 indicator여야 합니다.")
                y_tensor = torch.as_tensor(y_arr, dtype=torch.float32)
                if y_tensor.ndim == 1:
                    y_tensor = y_tensor[:, None]
            dataset = TensorDataset(X_tensor, y_tensor)
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)


    def make_torch_loss(task: str, pos_weight=None, class_weight=None):
        if task == "regression":
            return nn.MSELoss()
        if task in {"binary", "multilabel"}:
            weight = None
            if pos_weight is not None:
                weight = torch.as_tensor(pos_weight, dtype=torch.float32)
                if weight.ndim == 0:
                    weight = weight.reshape(1)
            return nn.BCEWithLogitsLoss(pos_weight=weight)
        if task == "multiclass":
            weight = None if class_weight is None else torch.as_tensor(
                class_weight, dtype=torch.float32
            )
            return nn.CrossEntropyLoss(weight=weight)
        raise ValueError(f"지원하지 않는 task: {task}")


    def _batch_loss(model, xb, yb, criterion, task, device):
        xb, yb = xb.to(device), yb.to(device)
        logits = model(xb)
        if task == "multiclass":
            yb = yb.long().reshape(-1)
        else:
            yb = yb.float()
            if logits.ndim == 2 and yb.ndim == 1:
                yb = yb[:, None]
            if logits.shape != yb.shape:
                raise ValueError(
                    f"output/target shape mismatch: {tuple(logits.shape)} vs {tuple(yb.shape)}"
                )
        return criterion(logits, yb), len(xb)


    # %% [TORCH-TRAIN] 공통 train/eval/early stopping
    def train_torch_model(
        model,
        train_loader,
        valid_loader,
        task: Literal["regression", "binary", "multiclass", "multilabel"],
        epochs: int = 30,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 5,
        max_seconds: float | None = None,
        pos_weight=None,
        class_weight=None,
        grad_clip: float = 1.0,
        device: str | None = None,
    ):
        if len(train_loader.dataset) == 0 or len(valid_loader.dataset) == 0:
            raise ValueError("train/validation Dataset이 비었습니다.")
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        criterion = make_torch_loss(task, pos_weight, class_weight)
        if hasattr(criterion, "to"):
            criterion = criterion.to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_loss = float("inf")
        best_state = None
        wait = 0
        history = []
        started = time.monotonic()

        for epoch in range(1, epochs + 1):
            model.train()
            train_sum = 0.0
            train_n = 0
            for xb, yb in train_loader:
                optimizer.zero_grad(set_to_none=True)
                loss, batch_n = _batch_loss(model, xb, yb, criterion, task, device)
                if not torch.isfinite(loss):
                    raise RuntimeError(f"train loss가 NaN/Inf입니다: {loss.item()}")
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                train_sum += loss.item() * batch_n
                train_n += batch_n
                if max_seconds is not None and time.monotonic() - started >= max_seconds:
                    break

            model.eval()
            valid_sum = 0.0
            valid_n = 0
            with torch.inference_mode():
                for xb, yb in valid_loader:
                    loss, batch_n = _batch_loss(model, xb, yb, criterion, task, device)
                    if not torch.isfinite(loss):
                        raise RuntimeError(f"valid loss가 NaN/Inf입니다: {loss.item()}")
                    valid_sum += loss.item() * batch_n
                    valid_n += batch_n

            train_loss = train_sum / max(train_n, 1)
            valid_loss = valid_sum / max(valid_n, 1)
            history.append({"epoch": epoch, "train_loss": train_loss, "valid_loss": valid_loss})
            print(f"epoch={epoch:03d} train={train_loss:.6f} valid={valid_loss:.6f}")

            if valid_loss < best_loss - 1e-8:
                best_loss = valid_loss
                best_state = {
                    key: value.detach().cpu().clone()
                    for key, value in model.state_dict().items()
                }
                wait = 0
            else:
                wait += 1
                if wait >= patience:
                    print("early stopping")
                    break

            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                print("time budget reached")
                break

        if best_state is None:
            raise RuntimeError("정상 validation checkpoint가 없습니다.")
        model.load_state_dict(best_state)
        model.eval()
        return model, pd.DataFrame(history)


    # %% [TORCH-PREDICT]
    def predict_torch(
        model,
        loader,
        task: Literal["regression", "binary", "multiclass", "multilabel"],
        return_proba: bool = False,
        threshold: float = 0.5,
        device: str | None = None,
    ) -> np.ndarray:
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device).eval()
        outputs = []
        with torch.inference_mode():
            for batch in loader:
                xb = batch[0] if isinstance(batch, (tuple, list)) else batch
                xb = xb.to(device)
                logits = model(xb)
                if task in {"binary", "multilabel"}:
                    prob = torch.sigmoid(logits)
                    out = prob if return_proba else (prob >= threshold).long()
                elif task == "multiclass":
                    prob = torch.softmax(logits, dim=1)
                    out = prob if return_proba else prob.argmax(dim=1)
                else:
                    out = logits
                outputs.append(out.detach().cpu().numpy())
        if not outputs:
            raise ValueError("prediction Dataset이 비었습니다.")
        return np.concatenate(outputs, axis=0)


# %% [IMAGE-NUMPY] 이미지 배열 shape/scale ------------------------------------
def prepare_numpy_images(
    X: np.ndarray,
    layout: Literal["NHWC", "NCHW"],
    divide_255: bool,
) -> np.ndarray:
    """명시한 layout을 PyTorch NCHW float32로 바꾼다."""
    arr = np.asarray(X)
    if arr.ndim == 3:  # grayscale batch
        arr = arr[..., None] if layout == "NHWC" else arr[:, None, :, :]
    if arr.ndim != 4:
        raise ValueError("이미지는 3D grayscale batch 또는 4D batch여야 합니다.")
    if layout == "NHWC":
        arr = np.transpose(arr, (0, 3, 1, 2))
    elif layout != "NCHW":
        raise ValueError(f"지원하지 않는 layout: {layout}")
    if arr.shape[1] not in (1, 3, 4):
        raise ValueError(f"channel 수를 확인하세요: shape={arr.shape}")
    arr = arr.astype(np.float32)
    if divide_255:
        arr /= 255.0
    if not np.isfinite(arr).all():
        raise ValueError("image에 NaN/Inf가 있습니다.")
    return arr


# %% [DEBUG-SHAPES] 제출 직전 shape 표시 --------------------------------------
def print_shapes(**arrays) -> None:
    for name, value in arrays.items():
        arr = np.asarray(value)
        print(f"{name:20s} shape={arr.shape!s:18s} dtype={arr.dtype}")


__all__ = [
    "TabularConfig",
    "seed_everything",
    "quick_audit",
    "assert_frame_contract",
    "minmax_selected",
    "fit_iqr_bounds",
    "apply_clip_bounds",
    "add_datetime_features",
    "clean_tabular_values",
    "crop_to_numpy",
    "conv_output_size",
    "safe_train_valid_indices",
    "make_preprocessor",
    "make_tabular_estimator",
    "evaluate_predictions",
    "assert_class_fold_coverage",
    "fit_tabular_baseline",
    "make_point_forecast_windows",
    "make_grouped_point_forecast_windows",
    "split_raw_time_then_window",
    "fit_scale_3d",
    "transform_scale_3d",
    "validate_prediction_array",
    "save_npy_submission",
    "save_csv_submission",
    "prepare_numpy_images",
    "print_shapes",
]

if torch is not None:
    __all__ += [
        "MLP",
        "CNN1D",
        "SequenceRNN",
        "SmallImageCNN",
        "ResidualBlock2D",
        "TransformerSequenceModel",
        "Autoencoder",
        "VariationalAutoencoder",
        "vae_loss",
        "MLPGenerator",
        "MLPDiscriminator",
        "LazyWindowDataset",
        "make_tensor_loader",
        "make_torch_loss",
        "train_torch_model",
        "predict_torch",
    ]
