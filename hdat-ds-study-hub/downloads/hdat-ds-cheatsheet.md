# HDAT-DS 실기 오픈북 치트시트 — PyTorch 전용

버전: 2026-09-08 · 배열 크기·브로드캐스팅 기초 보강 (2026-09-07 Astra 감사 유지)<br>
검색 키워드: `EDIT`, `binary`, `multiclass`, `multioutput`, `RMSLE`, `group`, `timeseries`, `window`, `CNN1D`, `GRU`, `LSTM`, `image`, `autoencoder`, `submission`, `hidden test`, `OOM`, `NaN`

> 목표는 만능 모델이 아니라 **문제 유형을 2분 안에 분류하고, 기준 모델로 유효한 제출 파일을 먼저 만든 뒤, 한 가지 개선 모델만 시도하는 것**이다. 실제 문제의 기본 코드(skeleton)·함수명·변수명·파일명·출력 크기(shape)가 이 문서보다 항상 우선한다.

> 이 자료는 개인 연습·검색용이다. 파일 반입·업로드·외부 접속 및 코드 사용 허용 범위는 해당 회차 규정으로 확인한다. 생성형 AI 활용은 공식 안내상 금지다. 허용되는 범위 안에서 필요한 블록을 제공된 기본 코드의 명세에 맞춘다. 이 문서가 시험장 사용 허가를 의미하지 않는다.

**어디서 시작할지 모르겠다면 [12유형 풀이 가이드](./playbook/)부터 보세요.** 새 가이드는 다운로드 소스 `hdat_templates.py`의 API로 통일했다. 이 문서의 독립 예제(`fit`, `ImageCNN`)와 소스 함수(`train_torch_model`, `SmallImageCNN`)를 섞지 않는다. 딥러닝은 PyTorch이며 5·7절의 sklearn 모델은 고전 머신러닝 선택 참고다. PyTorch로 표 데이터 문제를 처음 풀 때는 10→11(MLP)→12→17절 순으로 참고한다.

## 0. 시험장에서 가장 먼저 할 일

다음 10줄을 문제 상단에 직접 적고 빈칸을 채운 뒤 코딩한다.

```text
TASK        = regression / binary / multiclass / multilabel / anomaly
METRIC      = MSE / RMSE / MAE / RMSLE / Accuracy / Macro-F1 / AUC
TRAIN_X     = shape, dtype
TRAIN_Y     = shape, dtype, label 범위
TEST_X      = shape, dtype
OUTPUT      = label / probability / continuous value
OUT_SHAPE   = (N,) / (N,1) / (N,K) / (N,C)
SPLIT       = random / stratified / group / chronological
LEAKAGE     = 미래 정보·사후 정보·동일 group 중복 여부
FILE        = 정확한 파일명, 대소문자, 저장 위치
```

### 60초 문제 유형 결정표

| 문제 신호 | 검증 방법 | 첫 기준 모델 | PyTorch 개선 모델 | 대표적인 함정 |
|---|---|---|---|---|
| 독립 행 + 연속 정답 | 무작위 분할/KFold | Ridge 또는 ExtraTrees 회귀 | MLP | `(N,)`와 `(N,1)`, 정답 단위의 역변환 |
| 독립 행 + 클래스 2개 | 계층화 분할 | Logistic/ExtraTrees 분류 | MLP + BCE | 제출값이 라벨인지 양성 확률인지 확인 |
| 독립 행 + 클래스 3개 이상 | 계층화 분할 | Logistic/ExtraTrees 분류 | MLP + CE | 클래스 인덱스 0…C-1, 확률 열 순서 |
| 연속 정답 여러 개 | 데이터 구조에 맞춤 | ExtraTrees 다중출력 | MLP/CNN1D | 출력 `(N,K)`와 열 순서 |
| 같은 차량·설비 ID 반복 | 그룹 분할 | 표 데이터 기준 모델 | MLP/시퀀스 모델 | 같은 ID가 훈련·검증 양쪽에 존재 |
| 미래 예측·시간 순서 | 시간순 분할 | 마지막 값 예측/Ridge/ExtraTrees | CNN1D → GRU | 무작위 분할, 윈도 인덱스가 한 칸 어긋남 |
| 이미지 `(N,H,W,C)` | 계층화·그룹 분할 | 작은 CNN | 이미지 CNN | NCHW, RGB/흑백, 훈련자료에만 증강 |
| 이상 탐지 | 정상 훈련자료 분리 | IsolationForest/PCA | Autoencoder | 점수의 방향과 임계값 결정 |
| 수요·건수 + RMSLE | 구조에 맞춤 | `log1p(y)` 회귀 | MLP | `expm1` 누락, 음수 예측 |
| 심한 불균형 | 계층화·그룹·시간 분할 | 클래스 가중치 | BCE `pos_weight` | 정확도 맹신, 테스트 점수로 임계값 선택 |

### 데이터 분할 기준의 우선순위

1. 미래를 예측하면 **시간순으로 분할**한다.
2. 처음 보는 차량·설비·사용자를 예측하면 **그룹별로 분할**한다.
3. 일반적인 분류 문제이면 **계층화 분할**을 사용한다.
4. 그 외 일반적인 회귀 문제이면 무작위 단일 검증 분할 또는 KFold를 사용한다.

시간과 그룹 정보가 함께 있으면 실제로 예측할 상황을 먼저 해석한다. “같은 차량의 미래”와 “처음 보는 차량”은 서로 다른 검증 방법이 필요하다.

## 1. 절대 규칙

- Problem과 Process는 각각 저장·제출 상태를 확인한다.
- 의미 있는 셀을 실행할 때마다 저장하고, 문항 이동 전에도 `Ctrl+S`를 누른다.
- 별도 파일보다 제공된 기본 코드 안의 지정 셀·변수·저장 코드를 우선한다.
- 문제에 제출/저장 셀이 제공되면 그 셀은 바꾸지 않는다. 직전에 예측값만 검증한 뒤 제공 셀을 그대로 실행한다.
- 실기 170분에는 학습·실행 시간도 포함된다.
- 테스트 행 순서를 바꾸지 않는다. 정렬했다면 원래 순서로 복원한다.
- 스케일러·결측 대체기·인코더·PCA·특성 선택기는 훈련자료에만 `fit`한다.
- 검증 결과를 확인하기 전에 훈련자료 전체로 재학습하지 않는다.
- 처음부터 큰 LSTM·Transformer·대규모 탐색을 돌리지 않는다.
- 20분 남으면 성능 개선을 멈추고 저장 여부·배열 크기·NaN·제출 상태를 확인한다.
- 실제 시험 중에는 생성형 AI, GitHub, Notion, Colab, Kaggle 등 공식 금지 대상을 열지 않는다.

## 2. PyTorch 배열 크기·손실 요약표

| 과제 | `y` | 변환 전 모델 출력 | 손실함수 | 제출용 변환 |
|---|---|---|---|---|
| 단일 회귀 | `[B,1]` 실수형 | `[B,1]` | `MSELoss` | 연속 예측값 |
| 다중 회귀 | `[B,K]` 실수형 | `[B,K]` | `MSELoss` | 연속 예측값 |
| 이진분류 | `[B,1]` 실수형 0/1 | `[B,1]` 로짓(logits) | `BCEWithLogitsLoss` | 시그모이드 확률 또는 임계값 판정 |
| 다중분류 | `[B]` long 정수형, 0…C-1 | `[B,C]` 로짓 | `CrossEntropyLoss` | 소프트맥스 또는 argmax |
| 다중라벨 분류 | `[B,K]` 실수형 0/1 | `[B,K]` 로짓 | `BCEWithLogitsLoss` | 열별 시그모이드 확률 또는 임계값 판정 |

반드시 지킬 것:

- `CrossEntropyLoss` 앞에 소프트맥스를 붙이지 않는다.
- `BCEWithLogitsLoss` 앞에 시그모이드를 붙이지 않는다.
- 회귀·BCE에서 예측과 정답 배열의 크기를 완전히 같게 만든다. MSELoss는 `[B]`와 `[B,1]`을 브로드캐스팅해 의도와 다른 비교를 할 수 있다. BCEWithLogitsLoss는 서로 다른 배열 크기를 허용하지 않고 오류를 낸다.
- 표형 배치는 `[B,F]`, 윈도(window)로 묶은 시퀀스 배치는 `[B,T,F]`, PyTorch 이미지 배치는 `[B,C,H,W]`. 전체 샘플 수 N과 현재 배치 크기 B를 구분한다.
- RNN/LSTM/GRU 입력 `[B,T,F]`는 `batch_first=True`일 때다. 기본값은 `[T,B,F]`이며 마지막 은닉 상태의 축 순서는 별도다.
- `Conv1d`는 `[B,C,L]`이다. 특성 F를 채널, 시간 T를 길이로 쓰면 `x.permute(0,2,1)`로 `[B,T,F] → [B,F,T]`로 바꾼다. 모델을 감싼 코드가 내부에서 축을 바꾸면 중복 변환하지 않는다.
- 검증·테스트 로더는 `shuffle=False`로 설정한다.
- 추론은 `model.eval()`과 `torch.inference_mode()`.

### 배열 크기·축 교환·브로드캐스팅 빠른 복습

처음 보는 기호라면 [입문 6강의 쉬운 설명과 13문제·해설](https://markshincuhk.github.io/hdat-ds-study-hub/start/06/)부터 읽는다. 이 링크는 온라인 강의이며, 이 절 자체의 요약은 내려받은 치트시트에서도 읽을 수 있다.

| 구분 | 의미 | 실기에서 확인할 것 |
|---|---|---|
| 배열 크기(shape) | 숫자의 값이 아니라 각 축의 크기 | B=배치 샘플 수, F=특성 수, T=시점 수, C=채널, H/W=높이/너비 |
| `[B,K]` 로짓 | 샘플마다 클래스 점수 K개 | 아직 확률 아님; `argmax(dim=1)` 결과는 `[B]` |
| `permute(0,2,1)` | 기존 축 0·2·1 순서로 배치 | `[B,T,F] → [B,F,T]`; reshape로 대체하지 않기 |
| 브로드캐스팅(broadcasting) | 계산할 때 같은 값을 여러 위치에 적용 | 오른쪽부터 크기가 같거나 한쪽이 1, 없는 왼쪽 축은 1 |
| `[B,F] + [F]` | 특성별 값을 배치 전체에 적용 | `[1,F]`로 맞춰 읽기 |
| `[B,T,F] + [1,T,1]` | 시간별 값을 모든 배치·특성에 적용 | `[T]`만 쓰면 마지막 F축과 비교됨 |
| `[B,C,H,W] - [1,C,1,1]` | 이미지 채널별 평균 빼기 | `[C]`만 쓰면 마지막 W축과 비교됨 |
| 회귀 `[B,1] - [B]` | `[B,B]`로 모든 짝을 비교하는 함정 | 한 출력이면 둘 다 `[B,1]` 또는 둘 다 `[B]`로 맞추기 |

원본 시계열 CSV가 항상 `[B,T,F]`인 것은 아니다. `[전체 시점 수,F]`에서 윈도를 만든 뒤 배치로 묶은 형태인지 확인한다. 브로드캐스팅이 성공해도 의도한 축에 적용되었는지는 별도 확인해야 한다.

```python
import torch

# 독립 실행 가능한 shape 검사 예제. 실제 문제에서는 데이터·출력 계약을 확인한다.
pred = torch.tensor([[2.], [4.]])
target = torch.tensor([3., 5.])
assert (pred - target).shape == (2, 2)  # 에러 없이 잘못된 전체 짝 비교
target = target.unsqueeze(1)           # 한 출력 회귀: [B] -> [B,1]
assert pred.shape == target.shape
assert (pred - target).square().mean().item() == 1.0
assert pred[:1].squeeze(1).shape == (1,) # B=1에서도 배치 축 보존
```

다중출력 회귀라면 둘 다 `[B,K]`로 맞춘다. 정수 클래스 인덱스를 쓰는 CE는 로짓 `[B,K]`와 정답 `[B]`가 올바르므로 무조건 unsqueeze하지 않는다. 기호 C·K 등의 뜻은 각 절의 정의를 따른다.

## 3. Process: 비공개 테스트까지 대비하는 법

### 입출력 조건 점검표

- 요구 함수명, 인자 순서, 기본값, 전역 변수명, 모델명을 그대로 썼는가?
- 반환형이 DataFrame/Series/NumPy/Tensor 중 무엇인지 확인했는가?
- 입력을 수정하라는 말이 없으면 `copy()`했는가?
- 지정한 열만 바꾸고 나머지 열·열 순서·인덱스를 보존했는가?
- 샘플 수, 열 이름, 이미지 크기, 배치 크기를 코드에 고정된 값으로 넣지 않았는가?
- 임의로 `squeeze()`하지 않았는가?
- 예시 입력으로 모델을 실행해 출력 크기를 확인했는가?

### 최소 입력으로 테스트하기

```python
# 함수명은 실제 문제에 맞게 바꾼다.
tests = [
    pd.DataFrame({"x": [], "keep": []}),
    pd.DataFrame({"x": [5.0], "keep": [9]}, index=[100]),
    pd.DataFrame({"x": [3.0, 3.0], "keep": [1, 2]}, index=[2, 7]),
    pd.DataFrame({"x": [np.nan, 1.0], "keep": [1, 2]}),
]

for case in tests:
    before = case.copy(deep=True)
    result = your_function(case, ["x"])  # <<< EDIT
    assert case.equals(before), "원본이 변경됨"
    assert result.index.equals(before.index), "index 손실"
    assert list(result.columns) == list(before.columns), "column 순서 변경"
```

### 선택 열 Min-Max

```python
def minmax_selected(df, columns):
    out = df.copy(deep=True)
    for col in columns:
        x = pd.to_numeric(out[col], errors="raise").astype(float)
        lo, hi = x.min(skipna=True), x.max(skipna=True)
        if pd.isna(lo) or pd.isna(hi):
            out[col] = x                  # 전부 NaN
        elif hi == lo:
            out[col] = x.where(x.isna(), 0.0)
        else:
            out[col] = (x - lo) / (hi - lo)
    return out
```

### PIL로 이미지 자르기

```python
def crop_to_numpy(image, box):
    # box = (left, upper, right, lower)
    return np.asarray(image.crop(box)).copy()
```

### 합성곱의 출력 크기

```python
def conv_out(n, kernel, stride=1, padding=0, dilation=1):
    return (n + 2*padding - dilation*(kernel-1) - 1) // stride + 1
```

Process에서 구조를 정확히 지정하면 범용 모델로 바꾸지 말고 층 순서, 커널, 보폭, 패딩, 활성함수, 풀링을 **문제 문장 그대로** 구현한다.

## 4. 2분 EDA·입출력 조건 검사

```python
print("train/test:", train.shape, test.shape)             # <<< 변수명 EDIT
print("duplicate columns:", train.columns[train.columns.duplicated()].tolist())
print("dtypes:\n", train.dtypes.value_counts())
print("missing top:\n", train.isna().mean().sort_values(ascending=False).head(20))
print("nunique low:\n", train.nunique(dropna=False).sort_values().head(20))
print("duplicate rows:", train.duplicated().sum())

TARGETS = ["target"]                                      # <<< EDIT
print(train[TARGETS].describe(include="all").T)
for col in TARGETS:
    print(col, train[col].value_counts(dropna=False).head(20))

assert train.columns.is_unique and test.columns.is_unique
assert set(TARGETS).issubset(train.columns)
```

누수 의심 열:

- 정답을 다른 형식으로 나타내거나 정답에서 만든 파생 열
- 사고·고장·배송 완료 이후에만 알 수 있는 정보
- 미래 시점 값
- 동일 객체의 전체 기간 요약값
- 훈련자료에만 있으며 의미를 알 수 없는 열
- 기록 순서가 사실상 정답 순서와 같은 ID

ID를 자동으로 삭제하지 말고 그룹·시간 정보인지 먼저 확인한다.

## 5. 표 형태의 데이터 기준 모델 — 먼저 제출 파일 확보

### 언제 사용하나요?

- 행마다 독립적인 표형 분류·회귀.
- 시퀀스라도 각 행이 이미 하나의 윈도 또는 특성 묶음이라면 표 데이터 기준 모델부터 시도한다.

### 이 부분을 수정하세요

`TARGETS`, `DROP_COLS`, `DATE_COLS`, `TASK`, `METRIC`, `model`.

### 범용 코드

```python
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import (
    accuracy_score, f1_score, mean_absolute_error, mean_squared_error,
    roc_auc_score,
)

SEED = 42
TARGETS = ["target"]                 # <<< EDIT: 다중출력은 여러 열
DROP_COLS = []                        # <<< EDIT: ID/누수 열
DATE_COLS = []                        # <<< EDIT
TASK = "regression"                  # <<< EDIT: regression/binary/multiclass/multilabel
IS_CLASSIFICATION = TASK in {"binary", "multiclass", "multilabel"}
METRIC = "rmse"                      # <<< EDIT

# train/test는 skeleton이 제공한 객체를 사용한다. 파일을 임의로 다시 읽지 않는다.
train_df = train.copy()               # <<< EDIT
test_df = test.copy()                 # <<< EDIT

if not train_df.columns.is_unique or not test_df.columns.is_unique:
    raise ValueError("중복 column을 먼저 처리하세요")
if not set(TARGETS).issubset(train_df.columns) or set(TARGETS) & set(test_df.columns):
    raise KeyError("target train/test 계약을 확인하세요")
if not set(DATE_COLS).issubset(train_df.columns) or not set(DATE_COLS).issubset(test_df.columns):
    raise KeyError("DATE_COLS 오타 또는 train/test 불일치")
if not set(DROP_COLS).issubset(train_df.columns):
    raise KeyError(f"DROP_COLS 오타: {set(DROP_COLS) - set(train_df.columns)}")
if set(TARGETS) & (set(DATE_COLS) | set(DROP_COLS)):
    raise ValueError("target을 DATE_COLS/DROP_COLS에 넣지 마세요")
if set(DROP_COLS) & set(DATE_COLS):
    raise ValueError("같은 열을 DATE_COLS와 DROP_COLS에 동시에 넣지 마세요")

def dateparts(df, columns):
    out = df.copy()
    for col in columns:
        dt = pd.to_datetime(out[col], errors="coerce")
        out[f"{col}__year"] = dt.dt.year
        out[f"{col}__month"] = dt.dt.month
        out[f"{col}__day"] = dt.dt.day
        out[f"{col}__dow"] = dt.dt.dayofweek
        out[f"{col}__hour"] = dt.dt.hour
        out = out.drop(columns=col)
    return out

train_df = dateparts(train_df, DATE_COLS)
test_df = dateparts(test_df, DATE_COLS)

y = train_df[TARGETS]
if len(TARGETS) == 1:
    y = y.iloc[:, 0]
if pd.isna(np.asarray(y)).any():
    raise ValueError("target에 결측치가 있습니다")

X = train_df.drop(columns=TARGETS + DROP_COLS)
X_test = test_df.drop(columns=[c for c in DROP_COLS if c in test_df.columns])
missing_features = set(X.columns) - set(X_test.columns)
if missing_features:
    raise KeyError(f"test에 필요한 feature가 없습니다: {missing_features}")
X_test = X_test.reindex(columns=X.columns)

# Inf 처리와 범주형 혼합 타입 정리
num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]
X[num_cols] = X[num_cols].replace([np.inf, -np.inf], np.nan)
X_test[num_cols] = X_test[num_cols].replace([np.inf, -np.inf], np.nan)
for col in cat_cols:
    s_train = X[col].astype("string").astype(object)
    s_test = X_test[col].astype("string").astype(object)
    X[col] = s_train.where(pd.notna(s_train), np.nan)
    X_test[col] = s_test.where(pd.notna(s_test), np.nan)

num_pipe = Pipeline([
    ("impute", SimpleImputer(
        strategy="median", add_indicator=True, keep_empty_features=True
    )),
    ("scale", StandardScaler()),
])
cat_pipe = Pipeline([
    ("impute", SimpleImputer(
        strategy="constant", fill_value="__MISSING__", keep_empty_features=True
    )),
    ("ohe", OneHotEncoder(
        handle_unknown="ignore", sparse_output=True, dtype=np.float32
    )),
])
prep = ColumnTransformer([
    ("num", num_pipe, num_cols),
    ("cat", cat_pipe, cat_cols),
])

if IS_CLASSIFICATION:
    model = ExtraTreesClassifier(
        n_estimators=100, min_samples_leaf=5, max_features="sqrt",
        class_weight=None, n_jobs=-1, random_state=SEED
    )
else:
    model = ExtraTreesRegressor(
        n_estimators=100, min_samples_leaf=5, max_features=1.0,
        n_jobs=-1, random_state=SEED
    )

pipe = Pipeline([("prep", prep), ("model", model)])

# 일반 split. time/group이면 다음 절의 코드로 교체한다.
stratify = None
if IS_CLASSIFICATION and np.asarray(y).ndim == 1:
    counts = pd.Series(y).value_counts()
    n_valid = int(np.ceil(len(y) * 0.2))
    if counts.min() >= 2 and n_valid >= len(counts):
        stratify = y

tr_idx, va_idx = train_test_split(
    np.arange(len(X)), test_size=0.2, random_state=SEED,
    stratify=stratify
)

# time/group split로 교체했을 때 특히 중요: validation-only class 차단
if IS_CLASSIFICATION:
    y_arr = np.asarray(y)
    y_2d = y_arr[:, None] if y_arr.ndim == 1 else y_arr
    for j in range(y_2d.shape[1]):
        train_classes = set(pd.unique(y_2d[tr_idx, j]))
        valid_classes = set(pd.unique(y_2d[va_idx, j]))
        if not valid_classes.issubset(train_classes):
            raise ValueError(f"validation-only class: {valid_classes - train_classes}")
        if TASK in {"binary", "multiclass"} and len(train_classes) < 2:
            raise ValueError("train fold에 class가 하나뿐입니다")
        if METRIC.lower() in {"auc", "roc_auc"} and len(valid_classes) < 2:
            raise ValueError("AUC validation fold에 class가 하나뿐입니다")
        if (METRIC.lower() in {"auc", "roc_auc"} and TASK == "multiclass"
                and valid_classes != train_classes):
            raise ValueError("multiclass AUC는 양 fold의 class 집합을 맞추세요")

pipe.fit(X.iloc[tr_idx], np.asarray(y)[tr_idx])
va_pred = pipe.predict(X.iloc[va_idx])

metric = METRIC.lower()
if IS_CLASSIFICATION:
    yt, yp = np.asarray(y)[va_idx], np.asarray(va_pred)
    if metric in {"auc", "roc_auc"}:
        prob = pipe.predict_proba(X.iloc[va_idx])
        if yt.ndim == 2 and yt.shape[1] > 1:
            # 0/1 multilabel만 지원. categorical multi-output은 별도 산식 필요.
            if not set(np.unique(yt)).issubset({0, 1}):
                raise ValueError("이 AUC 분기는 0/1 multilabel 전용입니다")
            classes = pipe.named_steps["model"].classes_
            score = np.mean([
                roc_auc_score(yt[:, j], prob[j][:, list(classes[j]).index(1)])
                for j in range(yt.shape[1])
            ])
        elif TASK == "binary":
            score = roc_auc_score(yt, prob[:, 1])
        else:
            classes = pipe.named_steps["model"].classes_
            score = roc_auc_score(
                yt, prob, labels=classes, multi_class="ovr", average="macro"
            )
    elif yt.ndim == 2 and yt.shape[1] > 1:
        # 예시: 열별 점수 평균. 공식 metric이 다르면 반드시 교체.
        acc = np.mean([accuracy_score(yt[:, j], yp[:, j])
                       for j in range(yt.shape[1])])
        f1 = np.mean([f1_score(yt[:, j], yp[:, j], average="macro",
                               zero_division=0) for j in range(yt.shape[1])])
    else:
        acc = accuracy_score(yt, yp)
        f1 = f1_score(yt, yp, average="macro", zero_division=0)
    if metric == "accuracy":
        score = acc
    elif metric in {"f1", "f1_macro", "macro-f1"}:
        score = f1
    elif metric not in {"auc", "roc_auc"}:
        raise ValueError(f"classification metric을 구현하세요: {METRIC}")
else:
    yt = np.asarray(y)[va_idx]
    if metric == "mse":
        score = mean_squared_error(yt, va_pred)
    elif metric == "rmse":
        score = mean_squared_error(yt, va_pred) ** 0.5
    elif metric == "mae":
        score = mean_absolute_error(yt, va_pred)
    elif metric == "rmsle":
        if np.nanmin(yt) < 0:
            raise ValueError("RMSLE target에 음수가 있습니다")
        score = np.sqrt(np.mean(
            (np.log1p(yt) - np.log1p(np.clip(va_pred, 0, None))) ** 2
        ))
    else:
        raise ValueError(f"regression metric을 구현하세요: {METRIC}")
print("validation", METRIC, float(score))

# 검증 후 전체 train 재학습
pipe.fit(X, y)
OUTPUT_KIND = "label" if IS_CLASSIFICATION else "value"    # <<< EDIT: probability도 가능
if OUTPUT_KIND == "probability":
    prob = pipe.predict_proba(X_test)
    classes = pipe.named_steps["model"].classes_
    if TASK == "binary":
        POS_LABEL = 1                                     # <<< 문제에서 지정한 양성
        test_pred = prob[:, list(classes).index(POS_LABEL)]
    elif TASK == "multiclass":
        REQUIRED_LABEL_ORDER = list(classes)               # <<< 제출 열 순서
        assert len(REQUIRED_LABEL_ORDER) == len(classes)
        assert set(REQUIRED_LABEL_ORDER) == set(classes)
        test_pred = prob[:, [list(classes).index(v) for v in REQUIRED_LABEL_ORDER]]
    elif TASK == "multilabel":
        if not set(np.unique(y)).issubset({0, 1}):
            raise ValueError("0/1 multilabel만 지원합니다")
        test_pred = np.column_stack([
            p[:, list(c).index(1)] if 1 in c else np.zeros(len(X_test))
            for p, c in zip(prob, classes)
        ])
    else:
        raise ValueError("회귀의 제출은 value입니다")
elif OUTPUT_KIND == ("label" if IS_CLASSIFICATION else "value"):
    test_pred = pipe.predict(X_test)
else:
    raise ValueError("TASK와 OUTPUT_KIND를 확인하세요")
if metric == "rmsle":
    test_pred = np.clip(test_pred, 0, None)
print("test_pred", np.asarray(test_pred).shape)
assert len(test_pred) == len(X_test)
```

### 실행 후 확인할 내용

- 범주가 너무 많아 원-핫 인코딩(OHE)의 열 수가 지나치게 늘어나지 않는가?
- 테스트에 훈련 때 없던 범주가 있어도 `handle_unknown="ignore"`로 동작하는가?
- 다중출력 회귀이면 `test_pred.shape == (len(X_test), len(TARGETS))`인가?
- 위의 트리 100개짜리 기준 모델로 유효한 파일을 제출한 뒤 시간이 남으면 `n_estimators=300`, `min_samples_leaf=2`를 검증자료에서 비교한다.
- 다중출력·다중라벨 평가지표의 산식은 대회마다 다르다. 위 점수를 그대로 믿지 말고 공식 산식을 구현한다.

### 모델만 교체

```python
# 희소·고차원 선형 분류
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=1500, class_weight=None, C=1.0)
# multilabel/다중출력이면 MultiOutputClassifier(model, n_jobs=-1)로 감싼다.

# 선형 회귀
from sklearn.linear_model import Ridge
model = Ridge(alpha=1.0)
```

## 6. 검증 방법 바꾸기

### 그룹 분할

```python
from sklearn.model_selection import GroupShuffleSplit

groups = train_df["vehicle_id"].to_numpy()                 # <<< EDIT
splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, va_idx = next(splitter.split(X, y, groups=groups))
assert set(groups[tr_idx]).isdisjoint(set(groups[va_idx]))
```

“처음 보는 그룹”을 예측하는 문제라면 `vehicle_id` 같은 그룹 식별자는 보통 `DROP_COLS`에도 넣는다. 단, ID 자체가 의미 있는 특성이라고 명시한 경우에는 검증 결과로 판단한다.

### 시간순 분할

```python
# 날짜 feature 생성 전에 보존한 원본에서 가져온다.
time_key = pd.to_datetime(train["datetime"], errors="coerce")     # <<< EDIT
if time_key.isna().any():
    raise ValueError("시간 parse 실패/결측 행을 먼저 처리하세요")
order = np.argsort(time_key.to_numpy(), kind="stable")
cut = int(len(order) * 0.8)
if cut == 0 or cut == len(order):
    raise ValueError("train/validation 중 하나가 비었습니다")
tr_idx, va_idx = order[:cut], order[cut:]
assert time_key.iloc[tr_idx].max() <= time_key.iloc[va_idx].min()
```

“같은 차량의 미래”와 “처음 보는 차량”을 구분한다. 처음 보는 차량의 미래를 평가한다면 그룹과 시간 조건을 **함께** 만족해야 한다. 하나를 임의로 버리지 않는다.

### 이동 윈도가 겹칠 때

- 윈도를 만든 뒤 무작위로 분할하지 않는다.
- 실시간 미래 예측은 훈련 정답을 알 수 있는 시점이 첫 검증의 **예측 기준 시점**(입력 끝)보다 늦으면 안 된다. 정답 인덱스만 기준으로 나누면 horizon>1에서 이 조건을 어길 수 있다. 아래 기본 코드는 정답을 알 수 있는 시점과 예측 기준 시점을 함께 검사한다.
- 겹침으로 성능이 과도하게 낙관적이면 분할 지점 양쪽의 정답 사이에 간격(gap)을 둔다. 간격은 문제 구조에 맞게 정하고 검증 샘플이 남는지 확인한다.
- 그룹별 시퀀스에서는 한 윈도에 서로 다른 그룹의 관측을 섞지 않는다.

## 7. RMSLE·건수 문제

`y >= 0`일 때만 사용한다.

```python
from sklearn.compose import TransformedTargetRegressor

assert np.nanmin(np.asarray(y, dtype=float)) >= 0
log_model = TransformedTargetRegressor(
    regressor=pipe,
    func=np.log1p,
    inverse_func=np.expm1,
    check_inverse=False,
)
log_model.fit(X.iloc[tr_idx], np.asarray(y)[tr_idx])
va_pred = np.clip(log_model.predict(X.iloc[va_idx]), 0, None)
va_true = np.asarray(y)[va_idx]
rmsle = np.sqrt(np.mean((np.log1p(va_true) - np.log1p(va_pred)) ** 2))
print("rmsle", rmsle)
```

평가지표가 RMSE라면 정답에 로그를 취하는 것이 항상 좋지는 않다. 원래 정답을 학습하는 기준 모델과 검증자료에서 비교한다.

## 8. 불균형 분류

우선순위: 계층화 분할 → 클래스 가중치 → Macro-F1 확인 → 이진 판정 임계값 → 필요한 경우에만 재표집.

```python
# sklearn binary threshold
POS_LABEL, NEG_LABEL = 1, 0                              # <<< EDIT
pipe.fit(X.iloc[tr_idx], np.asarray(y)[tr_idx])
classes = list(pipe.named_steps["model"].classes_)
pos_idx = classes.index(POS_LABEL)
prob = pipe.predict_proba(X.iloc[va_idx])[:, pos_idx]

best_t, best_f1 = 0.5, -1
for t in np.linspace(0.05, 0.95, 181):
    pred = np.where(prob >= t, POS_LABEL, NEG_LABEL)
    score = f1_score(np.asarray(y)[va_idx], pred, average="macro")
    if score > best_f1:
        best_t, best_f1 = t, score
print(best_t, best_f1)
```

- `classes_[1]`이 항상 양성이라고 가정하지 않는다.
- 임계값은 검증자료로만 고른다.
- SMOTE를 전체 데이터에 적용한 뒤 분할하면 누수다.
- `imbalanced-learn`에 의존하지 않는 클래스 가중치는 시험 환경에서 추가 의존성 없이 적용하기에 안전한 선택이다.

PyTorch 이진분류:

```python
y_binary = np.asarray(y_train).reshape(-1)              # <<< 먼저 명시적 0/1 mapping
assert set(np.unique(y_binary)).issubset({0, 1})
n_pos = (y_binary == 1).sum()
n_neg = (y_binary == 0).sum()
POS_WEIGHT = n_neg / max(n_pos, 1)
print("POS_WEIGHT", POS_WEIGHT)
# 10절의 USE_CLASS_WEIGHTS=True가 같은 값을 계산해 공통 fit에 전달한다.
```

## 9. 시계열 윈도

### 인덱스를 먼저 정의

입력 마지막 관측 행을 `e`, 과거 입력 길이(lookback)를 `L`, 마지막 관측 이후 예측 간격(horizon)을 `H`라 하면:

```python
X_window = X[e-L+1:e+1]   # 길이 L
y_target = y[e+H]         # H=1은 바로 다음 행
```

100Hz에서 마지막 관측으로부터 1초 뒤라면 보통 `H=100`이지만, 문제의 기준 시점 정의를 반드시 읽는다.

### 작은 데이터용

```python
def make_windows(features, targets, lookback, horizon=1, stride=1,
                 assume_sorted=False):
    if not assume_sorted:
        raise ValueError("시간순 stable sort 확인 후 assume_sorted=True로 호출")
    Xv = np.asarray(features)
    yv = np.asarray(targets)
    if Xv.ndim == 1:
        Xv = Xv[:, None]
    if yv.ndim == 1:
        yv = yv[:, None]
    if Xv.ndim != 2 or yv.ndim != 2:
        raise ValueError("features=(T,F), targets=(T,) 또는 (T,D)여야 합니다")
    if len(Xv) != len(yv):
        raise ValueError("features와 targets 길이가 다릅니다")
    if lookback < 1 or horizon < 0 or stride < 1:
        raise ValueError("lookback/stride는 1 이상, horizon은 0 이상")

    last_start = len(Xv) - lookback - horizon
    starts = np.arange(0, max(last_start + 1, 0), stride)
    if len(starts) == 0:
        return (np.empty((0, lookback, Xv.shape[1]), dtype=Xv.dtype),
                np.empty((0, yv.shape[1]), dtype=yv.dtype),
                np.empty(0, dtype=int))
    target_idx = starts + lookback - 1 + horizon
    Xw = np.stack([Xv[s:s+lookback] for s in starts])
    yw = yv[target_idx]
    return Xw, yw, target_idx

# 원본 행이 섞였을 수 있으므로 시간으로 stable sort하고 원래 행 번호를 보존한다.
time_all = pd.to_datetime(TIME_ALL, errors="coerce")                     # <<< EDIT
if pd.isna(time_all).any():
    raise ValueError("time parse 실패/결측")
row_order = np.argsort(np.asarray(time_all), kind="stable")
X_sorted, y_sorted = np.asarray(X_all)[row_order], np.asarray(y_all)[row_order]

# 경계 전에 학습 정답을 확보하고, 경계 이후 관측 시점에서 예측한다.
Xw, yw, target_idx = make_windows(
    X_sorted, y_sorted, LOOKBACK, HORIZON, STRIDE, assume_sorted=True
)
target_original_rows = row_order[target_idx]
cut, gap = int(len(X_sorted) * 0.8), 0                                   # <<< EDIT
LABEL_DELAY = 0  # <<< 정답 확정 지연, 여기서는 raw 행 단위
origin_idx = target_idx - HORIZON
train_mask = target_idx + LABEL_DELAY < cut - gap
valid_mask = origin_idx >= cut + gap
X_train, y_train = Xw[train_mask], yw[train_mask]
X_valid, y_valid = Xw[valid_mask], yw[valid_mask]
assert len(X_train) and len(X_valid)
assert (target_idx[train_mask] + LABEL_DELAY).max() <= origin_idx[valid_mask].min()
```

차량·주행별 시퀀스라면 각 그룹 안에서만 정렬하고 윈도를 만든다.

```python
def make_grouped_windows(features, targets, groups, times,
                         lookback, horizon=1, stride=1):
    Xv, yv, gv = map(np.asarray, (features, targets, groups))
    tv = pd.to_datetime(np.asarray(times), errors="coerce")
    if (len(Xv) != len(yv) or len(Xv) != len(gv) or len(Xv) != len(tv)
            or pd.isna(gv).any() or pd.isna(tv).any()):
        raise ValueError("길이 또는 time을 확인하세요")
    Xs, ys, target_rows, target_groups = [], [], [], []
    for g in pd.unique(gv):
        rows = np.flatnonzero(gv == g)
        rows = rows[np.argsort(tv[rows], kind="stable")]
        Xg, yg, local_idx = make_windows(
            Xv[rows], yv[rows], lookback, horizon, stride,
            assume_sorted=True
        )
        if len(Xg):
            Xs.append(Xg); ys.append(yg)
            target_rows.append(rows[local_idx])
            target_groups.append(np.repeat(g, len(Xg)))
    if not Xs:
        raise ValueError("어느 group에서도 window를 만들 수 없습니다")
    return (np.concatenate(Xs), np.concatenate(ys),
            np.concatenate(target_rows), np.concatenate(target_groups))
```

그룹별 윈도를 만든 뒤 무작위로 분할하지 않는다. 아래 두 평가 목표 중 하나를 고른다.

```python
Xg, yg, target_rows, target_groups = make_grouped_windows(
    X_all, y_all, GROUPS, TIME_ALL, LOOKBACK, HORIZON, STRIDE       # <<< EDIT
)
SPLIT_MODE = "unseen_group"                                       # <<< EDIT

if SPLIT_MODE == "unseen_group":
    # 처음 보는 차량/설비
    from sklearn.model_selection import GroupShuffleSplit
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_idx, va_idx = next(splitter.split(Xg, yg, groups=target_groups))
    assert set(target_groups[tr_idx]).isdisjoint(set(target_groups[va_idx]))
elif SPLIT_MODE == "future_within_group":
    # 공동 모델은 모든 train label이 첫 valid 예측 원점 전에 알려져야 한다.
    target_times = pd.to_datetime(np.asarray(TIME_ALL)[target_rows], errors="coerce")
    if pd.isna(target_times).any():
        raise ValueError("target time parse 실패")
    all_times = pd.to_datetime(np.asarray(TIME_ALL), errors="raise")
    all_groups = np.asarray(GROUPS)
    origin_for_row = np.full(len(all_times), -1, dtype=int)
    for group in pd.unique(all_groups):
        rows = np.flatnonzero(all_groups == group)
        rows = rows[np.argsort(all_times[rows], kind="stable")]
        positions = np.arange(HORIZON, len(rows))
        origin_for_row[rows[positions]] = rows[positions - HORIZON]
    assert (origin_for_row[target_rows] >= 0).all()
    origin_times = all_times[origin_for_row[target_rows]]
    ordered = all_times.sort_values()
    cutoff_time = ordered[int(len(ordered) * 0.8)]              # <<< 공통 시간 경계
    label_delay = pd.Timedelta(0)                              # <<< 정답 확정 지연
    train_mask = target_times + label_delay < cutoff_time
    valid_mask = origin_times >= cutoff_time
    tr_idx, va_idx = np.flatnonzero(train_mask), np.flatnonzero(valid_mask)
    if len(tr_idx) == 0 or len(va_idx) == 0:
        raise ValueError("group별 time split 결과가 비었습니다")
    assert (target_times[tr_idx] + label_delay).max() <= origin_times[va_idx].min()
else:
    raise ValueError(SPLIT_MODE)

X_train, y_train = Xg[tr_idx], yg[tr_idx]
X_valid, y_valid = Xg[va_idx], yg[va_idx]
```

### 메모리 계산

```python
gb = n_windows * lookback * n_features * 4 / 1e9  # float32
print("estimated GB", gb)
```

100만 × 20 × 23 × float32는 약 1.84GB이며 복사본까지 생기면 메모리 부족으로 커널이 종료될 수 있다.

### 큰 데이터용 지연 로딩 Dataset

```python
class LazyWindowDataset(torch.utils.data.Dataset):
    def __init__(self, X, y=None, lookback=20, horizon=1, stride=1,
                 starts=None, expected_n=None, assume_sorted=False):
        if not assume_sorted:
            raise ValueError("시간순 stable sort 확인 후 assume_sorted=True로 호출")
        X_arr = np.asarray(X)
        if X_arr.ndim == 1:
            X_arr = X_arr[:, None]
        if X_arr.ndim != 2:
            raise ValueError("X는 (T,F)여야 합니다")
        y_arr = None if y is None else np.asarray(y)
        if y_arr is not None and len(y_arr) != len(X_arr):
            raise ValueError("X와 y 길이가 다릅니다")
        if y_arr is not None and y_arr.ndim not in {1, 2}:
            raise ValueError("y는 (T,) 또는 (T,D)여야 합니다")
        self.X = torch.as_tensor(X_arr, dtype=torch.float32)
        self.y = None if y_arr is None else torch.as_tensor(y_arr, dtype=torch.float32)
        if self.y is not None and self.y.ndim == 1:
            self.y = self.y[:, None]
        self.L, self.H = lookback, horizon
        if lookback < 1 or horizon < 0 or stride < 1:
            raise ValueError("lookback/stride는 1 이상, horizon은 0 이상")
        last = len(self.X) - lookback - (0 if self.y is None else horizon)
        if starts is None:
            if self.y is None:
                raise ValueError("추론은 제출 행과 대응하는 starts를 명시하세요")
            self.starts = np.arange(0, max(last + 1, 0), stride, dtype=int)
        else:
            self.starts = np.asarray(starts, dtype=int).reshape(-1)
        if len(self.starts) and (self.starts.min() < 0 or self.starts.max() > last):
            raise IndexError("유효 범위를 벗어난 window start")
        if expected_n is not None and len(self.starts) != expected_n:
            raise ValueError((expected_n, len(self.starts)))

    def __len__(self):
        return len(self.starts)

    def __getitem__(self, i):
        s = int(self.starts[i])
        x = self.X[s:s+self.L]
        if self.y is None:
            return x
        target_idx = s + self.L - 1 + self.H
        return x, self.y[target_idx]
```

`y=None` 추론에서 자동으로 윈도 수를 정하면 제출 행과 어긋날 수 있다. 문제에서 정의한 테스트 예측 기준 시점의 `starts`를 직접 만들고 `expected_n=len(sample_submission)`처럼 검증한다. `assume_sorted=True`는 실제로 시간순 정렬과 원래 행 대응을 확인한 뒤에만 준다.

### 시퀀스 스케일링

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
scaler.fit(X_train.reshape(-1, X_train.shape[-1]))  # train만 fit

def scale_3d(X):
    shape = X.shape
    return scaler.transform(X.reshape(-1, shape[-1])).reshape(shape).astype(np.float32)

X_train = scale_3d(X_train)
X_valid = scale_3d(X_valid)
X_test = scale_3d(X_test)
```

앞선 값으로 결측을 채우는 전방 채우기(forward fill)는 과거 값만 사용한다. 뒤의 값으로 채우는 후방 채우기(backward fill)는 미래 정보를 사용해 누수가 생길 수 있다.

## 10. PyTorch 공통 설정·DataLoader

### 표 형태의 DataFrame → PyTorch 밀집 배열

표형 `X`를 MLP에 넣을 때는 5절처럼 Inf와 범주형 혼합 타입을 먼저 정리하고, 자료를 나눈 뒤 전처리기를 훈련 폴드에서만 학습한다. 아래 밀집 배열 형태의 OHE는 변환 후 열 수가 감당 가능한 경우에만 쓴다.

```python
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

num_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
cat_cols = [c for c in X.columns if c not in num_cols]

torch_prep = ColumnTransformer([
    ("num", Pipeline([
        ("impute", SimpleImputer(strategy="median", keep_empty_features=True)),
        ("scale", StandardScaler()),
    ]), num_cols),
    ("cat", Pipeline([
        ("impute", SimpleImputer(strategy="constant", fill_value="__MISSING__",
                                  keep_empty_features=True)),
        ("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False,
                              dtype=np.float32)),
    ]), cat_cols),
], sparse_threshold=0)

X_train = torch_prep.fit_transform(X.iloc[tr_idx]).astype(np.float32)
X_valid = torch_prep.transform(X.iloc[va_idx]).astype(np.float32)
X_test_torch = torch_prep.transform(X_test).astype(np.float32)
y_all = np.asarray(y)
y_train, y_valid = y_all[tr_idx], y_all[va_idx]
assert X_train.ndim == X_valid.ndim == X_test_torch.ndim == 2
assert X_train.shape[1] == X_valid.shape[1] == X_test_torch.shape[1]
print("dense features", X_train.shape[1],
      "estimated MB", (X_train.nbytes + X_valid.nbytes + X_test_torch.nbytes) / 1e6)
X_test = X_test_torch
```

OHE 결과가 수천~수만 열로 늘어나면 밀집 배열로 변환하지 않는다. sklearn의 희소 입력용 선형 기준 모델을 유지하거나, 문제 허용 범위 안에서 드문 범주를 묶고 `max_categories`를 제한한다.

### 공통 설정과 DataLoader

```python
import random, time, copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader

SEED = 42
TASK = "regression"              # <<< EDIT: regression/binary/multiclass/multilabel
METRIC = "rmse"                  # <<< EDIT: mse/rmse/mae/rmsle/accuracy/f1_macro/auc
MODEL_KIND = "mlp"               # <<< 표형 기본. [N,T,F]면 cnn1d/gru/lstm으로 EDIT
BATCH_SIZE = 128                  # OOM이면 64→32
MAX_EPOCHS = 30
PATIENCE = 5
LR = 1e-3
MAX_TRAIN_SECONDS = 600           # 한 모델 10분 제한
POS_LABEL = None                  # 0/1이면 None→양성 1; 그 외/양성 0이면 EDIT
USE_CLASS_WEIGHTS = False         # 심한 불균형 + validation 이득이 있을 때만 True

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(DEVICE)

# DataLoader를 만들기 전에 label을 모델용 값으로 바꾼다.
y_train_model, y_valid_model = np.asarray(y_train), np.asarray(y_valid)
label_encoder, NEG_LABEL = None, None
if TASK == "binary":
    classes = np.unique(y_train_model)
    if len(classes) != 2:
        raise ValueError(f"binary인데 train class={classes}")
    already_01 = set(classes).issubset({0, 1})
    if not already_01 or POS_LABEL is not None:
        if POS_LABEL is None or POS_LABEL not in classes:
            raise ValueError("POS_LABEL을 실제 양성 class로 지정하세요")
        if not set(np.unique(y_valid_model)).issubset(set(classes)):
            raise ValueError("validation에 train에 없던 binary label이 있습니다")
        NEG_LABEL = next(v for v in classes if v != POS_LABEL)
        y_train_model = (y_train_model == POS_LABEL).astype(np.float32)
        y_valid_model = (y_valid_model == POS_LABEL).astype(np.float32)
elif TASK == "multiclass":
    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder().fit(y_train_model.reshape(-1))
    y_train_model = label_encoder.transform(y_train_model.reshape(-1))
    y_valid_model = label_encoder.transform(y_valid_model.reshape(-1))

POS_WEIGHT, CLASS_WEIGHT = None, None
if USE_CLASS_WEIGHTS:
    if TASK == "binary":
        flat = y_train_model.reshape(-1)
        POS_WEIGHT = [float((flat == 0).sum() / max((flat == 1).sum(), 1))]
    elif TASK == "multilabel":
        labels = np.asarray(y_train_model, dtype=np.float32)
        if not set(np.unique(labels)).issubset({0, 1}):
            raise ValueError("multilabel은 0/1 indicator여야 합니다")
        pos = labels.sum(axis=0)
        POS_WEIGHT = ((len(labels) - pos) / np.maximum(pos, 1)).astype(np.float32)
    elif TASK == "multiclass":
        counts = np.bincount(np.asarray(y_train_model, dtype=int))
        CLASS_WEIGHT = (counts.sum() / (len(counts) * counts)).astype(np.float32)

def make_target(y, task):
    arr = np.asarray(y)
    if task == "multiclass":
        return torch.as_tensor(arr, dtype=torch.long).reshape(-1)
    if task == "binary" and not set(np.unique(arr)).issubset({0, 1}):
        raise ValueError("binary label을 먼저 0/1로 mapping하세요")
    if task == "multilabel" and not set(np.unique(arr)).issubset({0, 1}):
        raise ValueError("multilabel target은 0/1 indicator여야 합니다")
    t = torch.as_tensor(arr, dtype=torch.float32)
    if task in {"regression", "binary"} and t.ndim == 1:
        t = t[:, None]
    return t

def make_loader(X, y=None, task="regression", batch_size=128, shuffle=False):
    if hasattr(X, "tocsr"):
        raise TypeError("scipy sparse는 PyTorch Tensor로 직접 변환하지 마세요")
    xt = torch.as_tensor(np.asarray(X), dtype=torch.float32)
    ds = TensorDataset(xt) if y is None else TensorDataset(xt, make_target(y, task))
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      num_workers=0, pin_memory=(DEVICE.type == "cuda"))

train_loader = make_loader(
    X_train, y_train_model, task=TASK, batch_size=BATCH_SIZE, shuffle=True
)
valid_loader = make_loader(
    X_valid, y_valid_model, task=TASK, batch_size=BATCH_SIZE, shuffle=False
)
test_loader = make_loader(
    X_test, None, task=TASK, batch_size=BATCH_SIZE, shuffle=False
)
```

`OneHotEncoder(sparse_output=True)` 결과는 PyTorch로 바로 넘기지 않는다. 특성 수가 적고 메모리가 충분할 때만 `.toarray()`로 밀집 배열로 바꾸고, 크면 sklearn 선형 모델을 쓰거나 순서 인코딩 또는 밀집 배열용 전처리를 따로 설계한다.

원래 라벨 복원은 추론을 마친 뒤에만 한다. 이진분류 확률 제출이면 `POS_LABEL=None`인 0/1 데이터에서는 클래스 1의 확률, 명시적으로 지정했으면 그 `POS_LABEL`의 확률이며 복원하지 않는다.

## 11. PyTorch 모델 블록

### MLP — 표 형태의 데이터·펼친 특성

```python
class MLP(nn.Module):
    def __init__(self, input_dim, out_dim, hidden=(128, 64), dropout=0.1):
        super().__init__()
        layers, d = [], input_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x.flatten(1) if x.ndim > 2 else x)
```

### CNN1D — 센서·고정 윈도의 첫 시퀀스 모델

```python
class CNN1D(nn.Module):
    def __init__(self, n_features, out_dim, channels=(64, 128), dropout=0.1):
        super().__init__()
        c1, c2 = channels
        self.features = nn.Sequential(
            nn.Conv1d(n_features, c1, 5, padding=2),
            nn.GroupNorm(1, c1), nn.ReLU(),
            nn.Conv1d(c1, c2, 3, padding=1),
            nn.GroupNorm(1, c2), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(dropout), nn.Linear(c2, out_dim))

    def forward(self, x):
        assert x.ndim == 3, x.shape
        return self.head(self.features(x.transpose(1, 2)))
```

### GRU/LSTM — 더 긴 순서 의존성

```python
class SequenceRNN(nn.Module):
    def __init__(self, n_features, out_dim, kind="gru", hidden=64,
                 layers=1, bidirectional=False, dropout=0.0):
        super().__init__()
        if kind not in {"gru", "lstm"}:
            raise ValueError("kind는 gru 또는 lstm")
        cls = nn.GRU if kind == "gru" else nn.LSTM
        self.layers = layers
        self.dirs = 2 if bidirectional else 1
        self.rnn = cls(n_features, hidden, num_layers=layers, batch_first=True,
                       bidirectional=bidirectional,
                       dropout=dropout if layers > 1 else 0.0)
        self.head = nn.Linear(hidden * self.dirs, out_dim)

    def forward(self, x):
        _, state = self.rnn(x)
        h = state[0] if isinstance(state, tuple) else state
        b = x.shape[0]
        h = h.reshape(self.layers, self.dirs, b, -1)[-1]
        h = h.transpose(0, 1).reshape(b, -1)
        return self.head(h)
```

시간이 부족하면 은닉 크기 64인 단방향 1층 GRU를 선택한다.

### 이미지용 CNN

```python
class ImageCNN(nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, out_dim)

    def forward(self, x):
        return self.head(self.features(x).flatten(1))
```

### Autoencoder — 정상 데이터 기반 이상 탐지

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent=16, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(), nn.Linear(hidden, latent)
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, input_dim)
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
```

### 모델 선택

```python
if TASK == "binary":
    OUT_DIM = 1
elif TASK == "multiclass":
    OUT_DIM = len(np.unique(y_train))
else:
    OUT_DIM = np.asarray(y_train).shape[1] if np.asarray(y_train).ndim == 2 else 1

if MODEL_KIND == "mlp":
    model = MLP(int(np.prod(X_train.shape[1:])), OUT_DIM)
elif MODEL_KIND == "cnn1d":
    model = CNN1D(X_train.shape[2], OUT_DIM)
elif MODEL_KIND in {"gru", "lstm"}:
    model = SequenceRNN(X_train.shape[2], OUT_DIM, kind=MODEL_KIND)
elif MODEL_KIND == "image_cnn":
    model = ImageCNN(X_train.shape[1], OUT_DIM)
else:
    raise ValueError(f"지원하지 않는 MODEL_KIND: {MODEL_KIND}")

model = model.to(DEVICE)
with torch.no_grad():
    dummy = torch.as_tensor(X_train[:2], dtype=torch.float32, device=DEVICE)
    print("dummy output", model(dummy).shape)
```

## 12. PyTorch 공통 학습·예측

```python
def criterion_for(task, pos_weight=None, class_weight=None):
    if task == "regression":
        return nn.MSELoss()
    if task in {"binary", "multilabel"}:
        pw = None if pos_weight is None else torch.as_tensor(
            pos_weight, dtype=torch.float32, device=DEVICE
        )
        return nn.BCEWithLogitsLoss(pos_weight=pw)
    if task == "multiclass":
        cw = None if class_weight is None else torch.as_tensor(
            class_weight, dtype=torch.float32, device=DEVICE
        )
        return nn.CrossEntropyLoss(weight=cw)
    raise ValueError(task)

def checked_loss(output, target, criterion, task):
    if task == "multiclass":
        target = target.long().reshape(-1)
        assert output.ndim == 2 and len(output) == len(target)
    else:
        assert output.shape == target.shape, (output.shape, target.shape)
    return criterion(output, target)

def fit(model, train_loader, valid_loader, task, epochs=30, patience=5,
        lr=1e-3, max_seconds=600, pos_weight=None, class_weight=None):
    model = model.to(DEVICE)
    if len(train_loader.dataset) == 0 or len(valid_loader.dataset) == 0:
        raise ValueError("train/validation loader가 비었습니다")
    criterion = criterion_for(task, pos_weight, class_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_loss, best_state, bad = float("inf"), None, 0
    started = time.monotonic()

    def denominator(target):
        if isinstance(criterion, nn.CrossEntropyLoss):
            target = target.long().reshape(-1)
            keep = target != criterion.ignore_index
            n = (criterion.weight[target[keep]].sum().item()
                 if criterion.weight is not None else keep.sum().item())
        else:
            n = len(target)
        if n <= 0:
            raise ValueError("loss 평균 분모가 0입니다")
        return n

    for epoch in range(1, epochs + 1):
        model.train()
        train_sum, train_n = 0.0, 0
        for xb, yb in train_loader:
            if time.monotonic() - started >= max_seconds:
                break
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            output = model(xb)
            loss = checked_loss(output, yb, criterion, task)
            if not torch.isfinite(loss):
                raise RuntimeError("loss is NaN/Inf")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            n = denominator(yb)
            train_sum += loss.item() * n
            train_n += n

        model.eval()
        valid_sum, valid_n = 0.0, 0
        with torch.inference_mode():
            for xb, yb in valid_loader:
                xb, yb = xb.to(DEVICE), yb.to(DEVICE)
                loss = checked_loss(model(xb), yb, criterion, task)
                if not torch.isfinite(loss):
                    raise RuntimeError("valid loss is NaN/Inf")
                n = denominator(yb)
                valid_sum += loss.item() * n
                valid_n += n
        if train_n == 0 or valid_n == 0:
            raise RuntimeError("한 epoch에서 처리된 train/valid sample이 없습니다")
        va_loss = valid_sum / valid_n
        print(epoch, train_sum/train_n, va_loss)

        if va_loss < best_loss - 1e-8:
            best_loss = va_loss
            best_state = {k: v.detach().cpu().clone()
                          for k, v in model.state_dict().items()}
            bad = 0
        else:
            bad += 1
        if bad >= patience or time.monotonic() - started >= max_seconds:
            break

    if best_state is None:
        raise RuntimeError("valid checkpoint가 없습니다")
    model.load_state_dict(best_state)
    return model

def raw_predict(model, loader):
    model = model.to(DEVICE).eval()
    out = []
    with torch.inference_mode():
        for batch in loader:
            xb = batch[0] if isinstance(batch, (tuple, list)) else batch
            xb = xb.to(DEVICE)
            out.append(model(xb).cpu().numpy())
    if not out:
        raise ValueError("prediction loader가 비었습니다")
    return np.concatenate(out)

def decode(raw, task, probability=False, threshold=0.5):
    if task == "regression":
        return raw
    if task in {"binary", "multilabel"}:
        prob = 1 / (1 + np.exp(-np.clip(raw, -50, 50)))
        return prob if probability else (prob >= threshold).astype(int)
    if task == "multiclass":
        z = raw - raw.max(axis=1, keepdims=True)
        prob = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        return prob if probability else prob.argmax(axis=1)
    raise ValueError(task)

model = fit(
    model, train_loader, valid_loader, TASK,
    epochs=MAX_EPOCHS,
    patience=PATIENCE,
    lr=LR,
    max_seconds=MAX_TRAIN_SECONDS,
    pos_weight=POS_WEIGHT,
    class_weight=CLASS_WEIGHT,
)
valid_raw = raw_predict(model, valid_loader)
valid_pred = decode(valid_raw, TASK, probability=False)

# 반드시 문제의 공식 metric으로 별도 확인
metric = METRIC.lower()
if TASK == "regression":
    true = np.asarray(y_valid).reshape(valid_pred.shape)
    if metric == "mse":
        score = np.mean((true - valid_pred) ** 2)
    elif metric == "rmse":
        score = np.sqrt(np.mean((true - valid_pred) ** 2))
    elif metric == "mae":
        score = np.mean(np.abs(true - valid_pred))
    elif metric == "rmsle":
        if np.min(true) < 0:
            raise ValueError("RMSLE target에 음수가 있습니다")
        score = np.sqrt(np.mean(
            (np.log1p(true) - np.log1p(np.clip(valid_pred, 0, None))) ** 2
        ))
    else:
        raise ValueError(f"regression metric을 구현하세요: {METRIC}")
elif TASK in {"binary", "multiclass", "multilabel"}:
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
    true = np.asarray(y_valid_model)
    if TASK in {"binary", "multiclass"}:
        true, valid_pred = true.reshape(-1), valid_pred.reshape(-1)
    if metric == "accuracy":
        score = accuracy_score(true, valid_pred)
    elif metric in {"f1", "f1_macro", "macro-f1"}:
        score = f1_score(true, valid_pred, average="macro", zero_division=0)
    elif metric in {"auc", "roc_auc"}:
        prob = decode(valid_raw, TASK, probability=True)
        if TASK == "binary":
            score = roc_auc_score(true, prob.reshape(-1))
        elif TASK == "multiclass":
            score = roc_auc_score(true, prob, multi_class="ovr", average="macro")
        else:
            score = roc_auc_score(true, prob, average="macro")
    else:
        raise ValueError(f"classification metric을 구현하세요: {METRIC}")
else:
    raise ValueError(TASK)
print("valid", METRIC, float(score))

raw = raw_predict(model, test_loader)
OUTPUT_KIND = "value" if TASK == "regression" else "label"  # <<< EDIT
if OUTPUT_KIND not in ({"value"} if TASK == "regression" else {"label", "probability"}):
    raise ValueError("TASK와 OUTPUT_KIND가 맞지 않습니다")
test_pred = decode(raw, TASK, probability=(OUTPUT_KIND == "probability"))
```

다중라벨 분류의 정확도는 “모든 라벨이 맞아야 정답”인 부분집합 정확도(subset accuracy)일 수 있고 열별 평균일 수도 있다. 다중출력 평가지표은 반드시 문제의 공식 산식을 그대로 구현한다.

라벨을 제출해야 하고 위에서 라벨을 인코딩했다면 **추론 후** 원래 값으로 복원한다. 확률 제출이면 복원하지 않는다.

```python
if OUTPUT_KIND == "label" and TASK == "binary" and NEG_LABEL is not None:
    test_pred = np.where(np.asarray(test_pred).reshape(-1) == 1,
                         POS_LABEL, NEG_LABEL)
elif OUTPUT_KIND == "label" and TASK == "multiclass" and label_encoder is not None:
    test_pred = label_encoder.inverse_transform(np.asarray(test_pred).reshape(-1))
```

다중분류의 **확률 열**은 `label_encoder.classes_` 순서다. 제출 열 순서가 따로 주어지면 재정렬한다.

```python
if TASK == "multiclass" and OUTPUT_KIND == "probability":
    REQUIRED_LABEL_ORDER = list(label_encoder.classes_)  # <<< sample 제출 열 순서
    assert len(REQUIRED_LABEL_ORDER) == len(label_encoder.classes_)
    assert set(REQUIRED_LABEL_ORDER) == set(label_encoder.classes_)
    column_idx = [list(label_encoder.classes_).index(v) for v in REQUIRED_LABEL_ORDER]
    test_pred = np.asarray(test_pred)[:, column_idx]
```

**체크포인트 주의:** 위 독립 `fit`은 검증 **손실**이 최소인 모델을 복원한다. `METRIC`은 마지막 점수 계산용이며 체크포인트 선택을 바꾸지 않는다. 공식 MAE·RMSLE·F1·AUC로 매 학습 회차마다 모델을 선택하려면 새 유형별 가이드의 `hdat_templates.train_torch_model(score_fn=..., maximize=...)` 함수를 사용한다. 두 API를 혼용하지 않는다.

## 13. 이미지 처리

### NumPy 이미지를 NCHW로 변환하기

```python
def prepare_images(X, layout="NHWC", divide_255=True):
    arr = np.asarray(X)
    if layout not in {"NHWC", "NCHW"}:
        raise ValueError("layout은 NHWC 또는 NCHW")
    if arr.ndim == 3:                         # N,H,W gray
        arr = arr[:, None, :, :]
    elif arr.ndim != 4:
        raise ValueError(f"expected 3D/4D images, got {arr.shape}")
    elif layout == "NHWC":
        arr = arr.transpose(0, 3, 1, 2)
    elif layout != "NCHW":
        raise ValueError("layout은 NHWC 또는 NCHW")
    arr = arr.astype(np.float32)
    if divide_255:
        arr /= 255.0
    if not np.isfinite(arr).all():
        raise ValueError("image에 NaN/Inf가 있습니다")
    return arr
```

배열 크기만 보고 NHWC/NCHW를 자동 추정하지 않는다. 예를 들어 폭이 3인 NCHW 배열도 잘못 뒤집힐 수 있다. 원소 자료형이 `uint8`이고 값 범위가 0…255일 때만 `divide_255=True`; 이미 0…1이거나 표준화된 float이면 `False`다.

### 파일 경로 기반 Dataset

```python
from PIL import Image
from torchvision import transforms

train_tf = transforms.Compose([
    transforms.Resize((128, 128)),                  # <<< EDIT
    # transforms.RandomHorizontalFlip(),            # 방향이 label이면 사용 금지
    transforms.ToTensor(),
])
valid_tf = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

class ImagePathDataset(torch.utils.data.Dataset):
    def __init__(self, paths, y=None, transform=None, task="multiclass"):
        self.paths = list(paths)
        self.y = None if y is None else make_target(y, task)
        if self.y is not None and len(self.y) != len(self.paths):
            raise ValueError("paths와 y 길이가 다릅니다")
        self.transform = transform
    def __len__(self):
        return len(self.paths)
    def __getitem__(self, i):
        with Image.open(self.paths[i]) as opened:
            img = opened.convert("RGB").copy()
        img = self.transform(img) if self.transform else transforms.ToTensor()(img)
        return img if self.y is None else (img, self.y[i])
```

경로 문자열은 공통 `make_loader`에 넣지 않는다. 10절에서 라벨을 `y_train_model/y_valid_model`로 먼저 바꾼 뒤 전용 Dataset과 로더를 만든다.

```python
train_image_ds = ImagePathDataset(
    train_paths, y_train_model, transform=train_tf, task=TASK      # <<< EDIT
)
valid_image_ds = ImagePathDataset(
    valid_paths, y_valid_model, transform=valid_tf, task=TASK      # <<< EDIT
)
test_image_ds = ImagePathDataset(
    test_paths, None, transform=valid_tf, task=TASK                # <<< EDIT
)
train_loader = DataLoader(train_image_ds, batch_size=BATCH_SIZE,
                          shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_image_ds, batch_size=BATCH_SIZE,
                          shuffle=False, num_workers=0)
test_loader = DataLoader(test_image_ds, batch_size=BATCH_SIZE,
                         shuffle=False, num_workers=0)

model = ImageCNN(in_channels=3, out_dim=OUT_DIM).to(DEVICE)        # RGB convert 기준
xb, yb = next(iter(train_loader))
with torch.no_grad():
    assert model(xb[:2].to(DEVICE)).shape == (min(2, len(xb)), OUT_DIM)
# 이후 12절의 fit/raw_predict를 그대로 사용한다.
```

사전학습 가중치는 즉석 다운로드에 실패하거나 시간을 소모할 수 있다. 캐시·제공 여부를 확인하지 못했다면 작은 CNN을 먼저 사용한다. 사전학습 가중치 없이 특징 추출부의 가중치를 고정하지 않는다.

### 선택: ResNet18 전이학습

작은 CNN 제출 후, ImageNet 사전학습 가중치가 **이미 캐시되었거나 시험에서 제공됨을 확인했을 때만** `USE_PRETRAINED=True`로 바꾼다.

```python
from torchvision.models import resnet18, ResNet18_Weights

USE_PRETRAINED = False                                  # 기본은 다운로드 없는 scratch
weights = ResNet18_Weights.DEFAULT if USE_PRETRAINED else None
model = resnet18(weights=weights)
in_features = model.fc.in_features

if weights is not None:
    for parameter in model.parameters():
        parameter.requires_grad = False
    # ImageNet resize/crop/normalize. 안전한 첫 baseline은 train/valid에 동일 적용.
    train_tf = valid_tf = weights.transforms()
else:
    # weights=None인데 freeze하면 random feature라 학습되지 않는다.
    for parameter in model.parameters():
        parameter.requires_grad = True

model.fc = nn.Linear(in_features, OUT_DIM)
model = model.to(DEVICE)
# 기존 ImagePathDataset/loader를 바뀐 transform으로 다시 만든 뒤 12절 fit 사용.
```

사전학습 특징 추출부의 가중치를 고정한 첫 실험을 마친 뒤, 시간이 남고 검증 결과상 개선 여지가 있을 때만 마지막 블록 일부의 고정을 풀어 작은 학습률로 학습한다.

## 14. Autoencoder 이상 탐지

### 언제 사용하나요?

- 정상 데이터로 학습하고 복원 오차가 큰 샘플을 이상으로 본다.

```python
# X_train_normal만 scaler fit 후 사용
ae = Autoencoder(input_dim=X_train_normal.shape[1], latent=16).to(DEVICE)
train_loader = make_loader(
    X_train_normal, X_train_normal, task="regression", batch_size=256, shuffle=True
)
valid_loader = make_loader(
    X_valid_normal, X_valid_normal, task="regression", batch_size=256, shuffle=False
)
ae = fit(ae, train_loader, valid_loader, "regression", epochs=30, patience=5)

def reconstruction_error(model, X, batch=512):
    loader = make_loader(X, None, "regression", batch, False)
    recon = raw_predict(model, loader)
    return np.mean((np.asarray(X) - recon) ** 2, axis=1)

normal_error = reconstruction_error(ae, X_valid_normal)
threshold = np.quantile(normal_error, 0.99)              # <<< EDIT/validation으로 결정
test_error = reconstruction_error(ae, X_test)
pred_anomaly = (test_error > threshold).astype(int)
OUTPUT_KIND = "label"  # <<< EDIT: anomaly_score이면 error 자체를 제출
if OUTPUT_KIND == "label":
    test_pred = pred_anomaly  # 이상=1인 명세. 정상=1이면 반전
elif OUTPUT_KIND == "anomaly_score":
    test_pred = test_error
else:
    raise ValueError(OUTPUT_KIND)
assert np.asarray(test_pred).shape == (len(X_test),)
```

- 이상 라벨이 있는 검증자료가 있으면 F1 등 공식 평가지표로 임계값을 고른다.
- 이상 라벨이 없으면 정상 검증자료의 복원 오차에서 95번째·99번째 백분위수를 비교한다.
- 복원 오차가 클수록 이상에 가까운 점수로 사용한다.

빠른 기준 모델은 `IsolationForest`이며, Autoencoder 전에 비교한다.

## 14A. 명세가 직접 요구할 때만 쓰는 PyTorch 부록

VAE·GAN·Transformer는 170분 Problem의 첫 모델로 쓰지 않는다. Process가 구조를 직접 요구하거나, 빠른 기준 모델의 결과를 제출한 뒤 검증 결과상 근거가 있을 때만 사용한다.

VAE는 복원 손실+KL, GAN은 두 최적화 알고리즘을 이용한 교대 학습, 가변길이 Transformer는 패딩·마스크 처리가 필요하다. 구조 블록만 가져와 위 범용 `fit`에 연결하면 완성되지 않는다. 아래 코드는 구조 참고이며 문제 명세에 맞는 전용 루프를 별도로 작성한다.

### 잔차 블록

```python
class ResidualBlock2D(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.skip = (nn.Identity() if stride == 1 and in_ch == out_ch else
                     nn.Sequential(nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                                   nn.BatchNorm2d(out_ch)))
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.main(x) + self.skip(x))
```

### Transformer 인코더 — 입력 `[B,T,F]`

```python
class TransformerSequence(nn.Module):
    def __init__(self, n_features, out_dim, d_model=64, nhead=4,
                 layers=2, max_len=512):
        super().__init__()
        assert d_model % nhead == 0
        self.max_len = max_len
        self.proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model, nhead, dim_feedforward=128, dropout=0.1,
            batch_first=True, norm_first=False
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=layers)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x, padding_mask=None):
        T = x.shape[1]
        if T > self.max_len:
            raise ValueError("max_len을 늘리거나 sequence를 줄이세요")
        h = self.encoder(self.proj(x) + self.pos[:, :T],
                         src_key_padding_mask=padding_mask)
        if padding_mask is None:
            pooled = h.mean(dim=1)
        else:
            valid = (~padding_mask).unsqueeze(-1).to(h.dtype)
            pooled = (h * valid).sum(1) / valid.sum(1).clamp_min(1)
        return self.head(self.norm(pooled))
```

### VAE

```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent=8, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU())
        self.mu, self.logvar = nn.Linear(hidden, latent), nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, input_dim)
        )

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        std = torch.exp(0.5 * logvar)
        z = mu + torch.randn_like(std) * std if self.training else mu
        return self.dec(z), mu, logvar

def vae_loss(recon, x, mu, logvar, beta=1.0):
    rec = nn.functional.mse_loss(recon, x)
    kl = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    return rec + beta * kl, rec, kl
```

VAE는 `forward`가 `(recon, mu, logvar)`를 반환하므로 위의 공통 `fit`과 호환되지 않는다. 전용 학습 반복문에서 다음 순서로 계산한다.

```python
recon, mu, logvar = model(x)
total, rec, kl = vae_loss(recon, x, mu, logvar)
```

### 최소 GAN

```python
class Generator(nn.Module):
    def __init__(self, noise_dim, output_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, output_dim)
        )
    def forward(self, z): return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, input_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, hidden//2), nn.LeakyReLU(0.2),
            nn.Linear(hidden//2, 1)
        )
    def forward(self, x): return self.net(x)  # raw logit

# loss는 BCEWithLogitsLoss. Discriminator 출력에 sigmoid를 붙이지 않는다.
```

- 입력 데이터를 `[-1,1]`로 스케일하고 Generator 끝에 `Tanh`를 둘지 명세에 맞춰 결정한다.
- GAN도 생성기와 판별기를 각각의 최적화 알고리즘으로 번갈아 갱신하는 전용 학습 반복문가 필요하며 공통 `fit`으로 학습할 수 없다.
- GAN 학습은 불안정하고 제출용 예측 모델이 아니므로, 구조 구현 문제가 아니면 우선순위가 낮다.

## 15. 특성 선택·PCA·고유 범주

### 고유 범주가 너무 많을 때

```python
OneHotEncoder(
    handle_unknown="infrequent_if_exist",
    min_frequency=5,              # <<< EDIT
    max_categories=100,           # <<< EDIT
    sparse_output=True,
)
```

고유값 비율이 1에 가까운 열은 ID일 수 있지만 그룹·시간 정보를 담은 열인지 먼저 확인한다.

### 특성 선택

```python
from sklearn.feature_selection import SelectPercentile, f_classif, f_regression
IS_CLASSIFICATION = TASK in {"binary", "multiclass", "multilabel"}
score_func = f_classif if IS_CLASSIFICATION else f_regression
pipe = Pipeline([
    ("prep", prep),
    ("select", SelectPercentile(score_func, percentile=50)),  # <<< EDIT
    ("model", model),
])
```

위 `SelectPercentile` 예시는 단일 정답용이다. 다중출력/멀티라벨은 열별 선택 규칙을 따로 정하지 못했다면 생략한다. 특성 선택기·PCA는 반드시 Pipeline 안에서 훈련 폴드로만 학습한다.

### PCA·KMeans·IsolationForest 최소 코드

```python
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# 수치형 배열 X_num. imputer/scaler/PCA는 train에만 fit한다.
imputer = SimpleImputer(strategy="median", keep_empty_features=True)
Xtr_i = imputer.fit_transform(X_train_num)               # <<< EDIT
Xva_i = imputer.transform(X_valid_num)
Xte_i = imputer.transform(X_test_num)
scaler = StandardScaler()
Xtr_s = scaler.fit_transform(Xtr_i)
Xva_s = scaler.transform(Xva_i)
Xte_s = scaler.transform(Xte_i)

# Feature extraction
pca = PCA(n_components=0.95, random_state=42)
Xtr_p = pca.fit_transform(Xtr_s)
Xva_p = pca.transform(Xva_s)
Xte_p = pca.transform(Xte_s)
print("PCA dims", Xtr_s.shape[1], "->", Xtr_p.shape[1])

# Clustering: label 의미가 없으므로 cluster 번호를 정답 class처럼 해석하지 않는다.
kmeans = KMeans(n_clusters=3, n_init=10, random_state=42)  # <<< EDIT
train_cluster = kmeans.fit_predict(Xtr_p)
test_cluster = kmeans.predict(Xte_p)

# 이상탐지: predict는 정상=1, 이상=-1. 제출 규격이 이상=1이면 변환한다.
iso = IsolationForest(contamination=0.01, random_state=42, n_jobs=-1)  # <<< EDIT
iso.fit(Xtr_s)  # 가능하면 정상 train만
raw_label = iso.predict(Xte_s)
anomaly_label = (raw_label == -1).astype(int)
anomaly_score = -iso.score_samples(Xte_s)  # 값이 클수록 이상
```

- PCA 복원 오차가 필요하면 `inverse_transform(transform(X))`와 원본의 MSE를 계산한다.
- `contamination`과 임계값은 검증자료 또는 정상 검증 점수의 분위수로 결정한다.
- 군집 번호 0/1/2에는 순서나 의미가 없다.

## 16. 소규모 하이퍼파라미터 탐색

아래 하이퍼파라미터 탐색표는 **ExtraTrees 전용**이다. **첫 유효 제출을 완료했고 30분 이상 남을 때만** 시도한다. 아래 무작위 3폴드 분할은 독립·동일분포(IID) 행 데이터 전용이다. 시간·그룹 문제에서는 그 구조에 맞는 분할기를 `cv=`에 넣을 수 없으면 탐색을 생략한다.

```python
from sklearn.model_selection import RandomizedSearchCV

keys = pipe.get_params().keys()
prefix = "model__estimator__" if "model__estimator__n_estimators" in keys else "model__"
params = {
    f"{prefix}n_estimators": [100, 200],
    f"{prefix}max_depth": [None, 8, 16],
    f"{prefix}min_samples_leaf": [2, 5, 10],
    f"{prefix}max_features": ["sqrt", 0.5, 1.0],
}
# 바깥 search만 병렬화: 단일 ExtraTrees이면 내부 병렬을 끈다.
if f"{prefix}n_jobs" in keys:
    pipe.set_params(**{f"{prefix}n_jobs": 1})
search = RandomizedSearchCV(
    pipe, params, n_iter=4, cv=3,
    scoring="neg_root_mean_squared_error",              # <<< EDIT
    n_jobs=-1, random_state=42, refit=True,
)
search.fit(X, y)
print(search.best_params_, search.best_score_)
```

`MultiOutputClassifier`처럼 모델이 다른 모델을 감싼 구조라면 실제 하이퍼파라미터 경로가 `model__estimator__...`일 수 있다. `pipe.get_params().keys()`로 확인하며, 시간이 빠듯하면 손대지 않는다.

## 17. 제공 저장 셀 실행 전 검증 — 가장 중요한 블록

문제에 저장/제출 셀이 있으면 **그 셀을 아래 코드로 교체하지 않는다.** `test_pred`만 먼저 검증한 다음 제공 셀을 수정 없이 실행한다. 아래 직접 저장 예시는 제공 저장 코드가 전혀 없는 연습/예외 상황에서만 쓴다.

### 예측값의 제출 조건 검증

```python
EXPECTED_SHAPE = (len(X_test), 3)                       # <<< 문제 지시대로 EDIT
pred = np.asarray(test_pred)
assert pred.shape == EXPECTED_SHAPE, (pred.shape, EXPECTED_SHAPE)
is_numeric = np.issubdtype(pred.dtype, np.number)
if is_numeric:
    assert np.isfinite(pred).all()
else:
    assert not pd.isna(pred).any()                        # 문자열 label

# 여기까지 통과한 뒤 skeleton의 제공 저장/제출 셀을 그대로 실행한다.
```

### 저장 셀이 없는 연습용 NPY 저장

```python
OUT_PATH = "Submission_problem.npy"                     # <<< skeleton 우선
path = OUT_PATH if str(OUT_PATH).endswith(".npy") else f"{OUT_PATH}.npy"
# dtype은 문제 지시가 있을 때만 변환한다: pred = pred.astype(np.float32)
is_numeric = np.issubdtype(pred.dtype, np.number)
if pred.dtype == object:
    raise TypeError("object NPY 금지: 명세가 문자열이면 먼저 Unicode dtype으로 변환")
if np.issubdtype(pred.dtype, np.number) and not np.isfinite(pred).all():
    raise ValueError("dtype 변환 후 NaN/Inf가 생겼습니다")
np.save(path, pred, allow_pickle=False)
check = np.load(path, allow_pickle=False)
assert check.shape == EXPECTED_SHAPE
assert check.dtype == pred.dtype
assert np.isfinite(check).all() if is_numeric else not pd.isna(check).any()
print("SAVED", path, check.shape, check.dtype)
```

### 저장 셀이 없는 연습용 CSV 저장

```python
submission = sample_submission.copy()                    # <<< skeleton 변수 우선
PRED_COLS = ["target"]                                  # <<< EDIT
arr = np.asarray(test_pred)
assert submission.columns.is_unique
assert set(PRED_COLS).issubset(submission.columns), (PRED_COLS, list(submission.columns))

if len(PRED_COLS) == 1:
    assert arr.shape in {(len(submission),), (len(submission), 1)}, arr.shape
    submission[PRED_COLS[0]] = arr.reshape(-1)
else:
    assert arr.shape == (len(submission), len(PRED_COLS))
    submission[PRED_COLS] = arr

assert arr.ndim >= 1 and len(submission) == arr.shape[0]
assert not submission[PRED_COLS].isna().any().any()
numeric = submission[PRED_COLS].select_dtypes(include=["number"])
assert numeric.shape[1] == 0 or np.isfinite(numeric.to_numpy()).all()
submission.to_csv("submission.csv", index=False)         # <<< EDIT
check = pd.read_csv("submission.csv")
assert list(check.columns) == list(submission.columns)
assert len(check) == len(submission)
assert not check[PRED_COLS].isna().any().any()
numeric_check = check[PRED_COLS].select_dtypes(include=["number"])
assert numeric_check.shape[1] == 0 or np.isfinite(numeric_check.to_numpy()).all()
```

추가 확인:

- 라벨 인코더를 썼다면 원래 라벨로 역변환했는가?
- 확률 열 순서가 `model.classes_`와 제출 열 순서에 맞는가?
- 정답 스케일러를 썼다면 원래 단위로 역변환했는가?
- RMSLE면 음수 예측을 0 이상으로 처리했는가?
- 정렬·병합 후 테스트의 원래 순서를 복원했는가?
- `(N,1)`과 `(N,)` 중 정확히 무엇을 요구하는가?

## 18. 오류 → 즉시 조치

| 증상 | 가장 먼저 볼 것 | 조치 |
|---|---|---|
| `Input X contains NaN` | Inf/결측 | Inf→NaN, 결측 대체기가 Pipeline 안에 있는지 확인 |
| OHE의 혼합 자료형 오류 | 범주값에 숫자·문자 혼합 | 범주형 열을 pandas `string`으로 통일 |
| 훈련에 없던 범주 | 인코더 설정 | `handle_unknown="ignore"` |
| `mat1 and mat2 shapes` | `INPUT_DIM` | `print(X.shape)`와 첫 Linear 입력 확인 |
| Conv1d 채널 오류 | `[B,T,F]`와 `[B,F,T]` | 모델 안에서 `x.transpose(1,2)` |
| Conv2d 채널 오류 | NHWC와 NCHW | `transpose(0,3,1,2)` |
| CE 정답 자료형 오류 | 실수형·원-핫 정답 | `[B]` long 정수형 클래스 인덱스로 변환 |
| BCE 배열 크기 오류 | `[B]`와 `[B,1]` | 정답과 출력 배열의 크기를 같게 |
| 손실이 NaN | 입력 NaN/Inf, 학습률, 스케일링 | 입력 검사, LR `1e-3→3e-4`, 기울기 크기 제한 |
| CUDA OOM | 배치·모델·윈도 메모리 | 배치 크기를 절반으로, 채널 수·은닉 크기 축소, 지연 로딩 Dataset |
| 검증 점수만 비정상적으로 좋음 | 누수·분할 | 시간·그룹·중첩 윈도 분할 재검토 |
| 학습이 너무 느림 | 모델 크기·CPU | ExtraTrees 100개, GRU 은닉 크기 64, 학습 20회, 실행 시간 10분 제한 |
| 예측 행 수 불일치 | 테스트의 순서 섞기·정렬·윈도 생성 | 원래 테스트 인덱스와 1:1 대응 재검토 |
| `object dtype` NPY | 라벨·배열의 자료형 혼합 | 수치 자료형으로 명시 변환 |
| 커널 종료 | 윈도 메모리 | 지연 로딩 Dataset, float32, 불필요한 배열 해제 |

CUDA OOM 복구:

```python
del model, optimizer
import gc; gc.collect()
if torch.cuda.is_available():
    torch.cuda.empty_cache()
```

## 19. 시간 부족 축소 순서

1. 검증을 1회 수행하고 첫 제출 파일을 확보한다.
2. 교차검증과 하이퍼파라미터 탐색을 생략한다.
3. ExtraTrees 개선안 `300→100`, `min_samples_leaf=5` 기준 모델로 복귀한다.
4. 학습 반복 횟수 `30→15`, 조기 종료 대기 횟수 `5→3`.
5. LSTM 대신 GRU 1층, 은닉 크기 `128→64`.
6. CNN 채널 수 `(64,128)→(32,64)`.
7. 이미지 `224→128→96`.
8. 배치 때문에 메모리가 부족하면 `256→128→64→32`.
9. 앙상블은 검증자료에서 실제 성능 향상이 있을 때만 사용한다.
10. 20분 남으면 모델 개선을 즉시 중단한다.

## 20. 170분 권장 운영

시간 배분 연습용 예시이며 화면 이동 규칙이 아니다. 공식 “One-way only”는 오픈북 규정 문맥이다. 문항 이동·영역 전환·재진입 가능 여부는 해당 회차 안내로 확인하고, 재방문 가능성을 전제로 시간을 짜지 않는다.

| 시간 | 행동 |
|---:|---|
| 0~5분 | 기본 코드, 변수, 출력 크기, 평가지표, 파일명 확인 |
| 5~55분 | 허용된 순서로 Process 풀이. 이동 전 가능한 검사와 저장 완료 |
| 55~60분 | Process 소규모 테스트, 저장 상태 확인 |
| 60~65분 | **Process 첫 제출** |
| 65~78분 | Problem 데이터 점검, 분할·누수·출력 조건 확정 |
| 78~100분 | 빠른 기준 모델과 제출 파일 검증 |
| 100~105분 | **Problem 기준 모델 결과 첫 제출** |
| 105~140분 | 한 가지 PyTorch 개선 모델 |
| 140~150분 | 검증 성능 비교, 필요한 경우에만 단순 평균 앙상블 |
| 150~158분 | 최종 예측 생성·재검증·Problem 재제출 |
| 158~165분 | 최종 파일을 다시 읽어 테스트 행·배열 크기·원소 자료형·유한성 재확인 |
| 165~170분 | 두 영역 제출 완료 상태와 최종 저장 확인 |

## 21. 필요한 코드 찾기

동일 폴더의 `hdat_templates.py`에서 아래 태그를 Ctrl+F한다. 시험에서는 파일 전체를 무작정 붙이지 말고 필요한 블록만 실제 기본 코드에 맞게 수정한다.

> **본문 코드와 `hdat_templates.py` API를 한 흐름 안에서 섞지 않는다.** 예를 들어 본문은 `fit`/`ImageCNN`/`VAE`, 소스는 `train_torch_model`/`SmallImageCNN`/`VariationalAutoencoder`라는 이름을 쓴다. 소스를 택했다면 로더→모델→손실→학습→예측를 소스 태그 기준으로 일관되게 복사하고, 함수 호출은 위치 인자 대신 `epochs=...`, `lr=...`, `patience=...`처럼 키워드 인자로 쓴다.

| 필요한 것 | 검색 태그 |
|---|---|
| EDA | `[COMMON-EDA]` |
| Min-Max·IQR·날짜·crop | `[PROCESS-MINMAX]`, `[PROCESS-CLIP]`, `[PROCESS-DATETIME]`, `[PROCESS-IMAGE]` |
| 표형 설정·분할·파이프라인 | `[TABULAR-CONFIG]`, `[TABULAR-SPLIT]`, `[TABULAR-PREPROCESS]`, `[TABULAR-FIT]` |
| 평가지표 | `[METRIC]` |
| 시계열 윈도 | `[TIME-WINDOW]`, `[TIME-LAZY-DATASET]` |
| MLP | `[TORCH-MLP]` |
| 1D CNN | `[TORCH-CNN1D]` |
| GRU/LSTM | `[TORCH-RNN]` |
| 이미지 CNN | `[TORCH-IMAGE]` |
| Autoencoder | `[TORCH-AE]` |
| Residual/Transformer | `[TORCH-RESIDUAL]`, `[TORCH-TRANSFORMER]` |
| VAE/GAN | `[TORCH-VAE]`, `[TORCH-GAN]` |
| PyTorch 학습·예측 | `[TORCH-TRAIN]`, `[TORCH-PREDICT]` |
| NPY/CSV 저장 | `[SUBMISSION]` |

## 22. 최종 30초 체크

```text
[ ] 문제의 함수명·변수명·파일명을 그대로 사용했다.
[ ] train에만 fit했고 test 순서를 보존했다.
[ ] validation이 time/group 구조를 반영한다.
[ ] output shape와 dtype이 정확하다.
[ ] label인지 probability인지 확인했다.
[ ] NaN/Inf가 없다.
[ ] 저장한 파일을 다시 load해 검사했다.
[ ] Ctrl+S를 눌렀다.
[ ] Process와 Problem을 각각 제출했다.
[ ] 제출 완료 상태를 확인했다.
```

---

이 문서는 빠른 검색·수정을 위한 개인 학습자료다. 실제 시험 규정과 문제 지시가 항상 우선한다.
