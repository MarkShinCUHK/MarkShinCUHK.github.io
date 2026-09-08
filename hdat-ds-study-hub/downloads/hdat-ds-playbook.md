# HDAT-DS 실기 · 문제에서 코드로

2026-09-07 · Astra 감사 반영 · 비공식 독립 학습자료. 아래 유형은 공개 범위를 학습하기 위한 분류이지 출제 예측이나 실제 기출 복제가 아닙니다. 딥러닝 모델과 학습은 PyTorch로 통일했습니다. NumPy/pandas 및 sklearn 전처리·지표는 보조 도구입니다.

## 00. 먼저 읽기 · 문제 계약과 소스 사용법

문제에서 중요한 것은 모델 이름보다 **받는 값과 돌려줄 값의 약속**입니다. Process는 정해진 함수·클래스의 동작을 구현하고, Problem은 데이터를 나누고 모델을 학습해서 예측을 제출하는 흐름입니다. 같은 MLP라도 정답이 연속값인지 class 번호인지에 따라 마지막 차원과 loss가 달라집니다.

### 1. 코딩 전에 여섯 칸을 채우세요

| 확인 | 적을 내용 | 잘못 고르면 생기는 일 |
|---|---|---|
| 입력 | DataFrame/배열/경로, shape, dtype, 한 행의 의미 | 이미 window인 입력에 다시 window를 만듦 |
| 정답 | 연속값/단일 class/여러 0·1, 양성 label, 열 순서 | 다중회귀를 multilabel로 오해 |
| 검증 | 독립 행/같은 개체/미래/새 개체의 미래 | 누수로 좋은 점수만 보고 모델 선택 |
| 학습 | 모델 구조와 loss | CE 앞에 softmax를 중복 적용 |
| 선택 | 공식 metric과 높을수록/낮을수록 | MAE 시험에서 MSE 최저 epoch 선택 |
| 제출 | label/확률/값/이상 score, 정확한 shape·파일명 | AUC라고 무조건 확률 파일이라고 추측 |

**손실(loss), 모델 선택 기준(metric), 제출 변환은 세 가지 별도 설정**입니다. MSELoss로 학습하면서 MAE로 checkpoint를 고를 수도 있습니다. metric 이름만 바꿔도 이 세 설정이 자동으로 맞춰지지 않습니다.

### 2. 두 파일을 같은 폴더에 저장하세요

[hdat_templates.py](./hdat_templates.py)는 함수·모델 모음이고, [pytorch-problem-starter.py](./pytorch-problem-starter.py)는 이를 연결한 작은 합성 실행 예제입니다. 기존 하이픈 이름 `hdat-templates.py`도 같은 내용으로 유지하지만 import에는 **밑줄 이름**을 쓰세요. 로컬 학습 환경에서 NumPy·pandas·scikit-learn·PyTorch가 필요합니다. 이미지 Process에는 Pillow가 필요합니다.

```bash
python pytorch-problem-starter.py --case regression --epochs 3
python pytorch-problem-starter.py --case binary --epochs 3
```

회귀는 `(20, 1)`, binary 확률도 `(20, 1)`이 출력되며 마지막에 `finite OK`가 보여야 합니다. 합성 데이터의 점수는 합격 가능성이나 실제 문제 성능을 뜻하지 않습니다. `--output practice.npy`를 추가하면 새 파일을 저장하고 다시 읽어 동일성을 검사합니다. 이미 있는 파일은 덮어쓰지 않습니다.

### 3. 코드를 바꾸는 순서

1. 먼저 수정 없이 실행해 환경과 import를 확인합니다. 파일 이름 뒤에 `.txt`가 붙지 않았는지 확인하세요.
2. `CHANGE DATA`: 합성 배열을 실제 X·y·test로 교체합니다. ID와 target이 feature에 들어가지 않게 합니다.
3. `CHANGE SPLIT`: 독립 합성 예제의 80/20 슬라이싱을 실제 time/group/stratified split으로 바꿉니다.
4. train 전용 전처리를 적용합니다. 변환 **후** feature 수로 모델을 만듭니다.
5. `CHANGE METRIC`, `CHANGE LOSS`, `CHANGE OUTPUT`, `CHANGE CONTRACT`를 문제 명세로 바꿉니다.
6. 1 epoch·작은 batch로 학습→검증→추론→저장 전체를 통과한 뒤 시간을 늘립니다.

### 4. 의존 블록을 빠뜨리지 마세요

로컬에서는 `import hdat_templates as h` 한 줄로 한 API 계열을 씁니다. 블록을 따로 옮기는 것이 허용된 상황에서는 파일 상단 import와 함께 모델 정의, `make_tensor_loader`, `make_torch_loss`, `_batch_loss`, `train_torch_model`, `predict_torch`를 확인합니다. `[TORCH-TRAIN]` 함수 하나만 가져오면 앞선 정의가 없어 NameError가 납니다. 예측만 한다면 optimizer는 필요하지 않습니다.

기존 긴 치트시트의 `fit(...) → model`과 다운로드 소스의 `train_torch_model(...) → (model, history)`는 다른 예제입니다. 새 가이드는 후자를 사용합니다. 기본 `score_fn=None`이면 validation loss 최소 모델, `score_fn`을 주면 전체 validation의 공식 지표로 선택합니다. AUC/F1/Accuracy는 `maximize=True`, RMSE/MAE/RMSLE는 `False`입니다.

### 5. 시험 규정은 별도로 확인하세요

이 파일의 다운로드·import가 시험에서 허용된다는 의미가 아닙니다. [NGV 공식 안내](https://exam.hyundai-ngv.com/practice/13567)는 생성형 AI 활용을 금지합니다. “One-way only”는 오픈북 안내 문맥이며, 이것만으로 화면 이동·재진입 제한을 단정할 수 없습니다. 개인 파일·사이트 사용과 문항 이동은 해당 회차 안내를 확인하세요. 제공 skeleton·저장 셀이 이 문서보다 우선합니다.

## 01. Process · 선택 열 변환·결측·이상치

### 문제 신호와 선택할 코드

“지정 열만 0~1”, “중앙값과 IQR”, “clip/제거/flag”가 보이면 모델을 학습하는 문제가 아니라 DataFrame 계약을 구현하는 문제입니다. Min-Max는 `minmax_selected`; clipping은 `fit_iqr_bounds`와 `apply_clip_bounds`를 사용합니다. **clip은 값을 경계로 자르고, remove는 행을 지우고, flag는 표시 열을 만드는 것**이므로 서로 대체할 수 없습니다.

### 바꿀 부분

`columns`, 반환 함수명, 상수열·NaN 정책, IQR 계수 `whisker`, 원본 수정 여부를 먼저 적습니다. 표준화면 평균/표준편차와 `ddof`, robust scaling이면 `(x−median)/IQR`을 요구대로 구현합니다. Process 내부 통계를 구하는 명세와, Problem의 train 통계를 valid/test에 재사용하는 절차를 구분하세요.

### 작은 실행 검사

```python
import numpy as np
import pandas as pd
import hdat_templates as h

df = pd.DataFrame({"x": [1e9, 1e9 + .5, np.nan], "id": [7, 8, 9]}, index=[4, 2, 8])
before = df.copy(deep=True)
result = h.minmax_selected(df, ["x"])
np.testing.assert_allclose(result.x, [0, 1, np.nan], equal_nan=True)
pd.testing.assert_frame_equal(df, before)
pd.testing.assert_series_equal(result.id, before.id)
```

두 값이 아주 가까워도 같지 않으면 이 예제의 양 끝점은 0과 1입니다. `np.isclose(max,min)`로 상수열을 판정하면 정상 범위도 0으로 지울 수 있습니다. 상수 판정에 허용 오차를 쓰라는 별도 명세가 있을 때만 바꾸세요.

### 자가 연습과 통과 기준

`x=[3,NaN,3]`으로 바꾸면 `[0,NaN,0]`; 전부 NaN이면 그대로 NaN; `columns=[]`이면 값이 같은 복사본이어야 합니다. 열 순서·index·비대상 열·원본까지 검사해야 통과입니다. 없는 열은 이 템플릿에서 KeyError입니다. 실제 문제가 다른 정책을 요구하면 수정하세요.

더 읽기: [26강 Process 구현](../learn/26/), [32강 robust scaling 독립 모의](../learn/32/).

## 02. Process · group별 lag·rolling·시간 파생

### 문제 신호와 선택할 코드

“각 차량별 이전 n개 평균”, “현재 행 제외”, “원래 행 순서 유지”는 **정렬 → group → shift → rolling → 원순서 복원**입니다. 날짜의 연/월/요일만 필요하면 소스 `add_datetime_features`; 과거 평균은 [32강의 add_past_mean](../learn/32/)을 선택합니다.

### 바꿀 부분

`group_col`, `time_col`, `value_col`, `window`, `min_periods`, 같은 시간의 순서 정책을 적습니다. 현재 값을 제외하려면 rolling 전에 `shift(1)`합니다. 아래는 window=2, 최소 1개, 동률은 입력 순서라는 **연습용 계약**입니다.

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({"g": ["A", "B", "A", "A"], "t": [3, 1, 1, 2], "v": [30, 99, 10, 20]})
work = df.assign(_row=np.arange(len(df))).sort_values(["g", "t", "_row"], kind="stable")
work["past_mean"] = work.groupby("g", sort=False)["v"].transform(
    lambda s: s.shift(1).rolling(2, min_periods=1).mean()
)
result = work.sort_values("_row").drop(columns="_row")
np.testing.assert_allclose(result.past_mean, [15, np.nan, np.nan, 10], equal_nan=True)
assert result.index.equals(df.index)
```

### 함정과 연습

group A의 첫 관측은 B의 99를 가져오면 안 됩니다. `shift` 없이 계산하면 현재값을 과거 feature에 넣습니다. 실제 예측 시점에 아직 모르는 target을 rolling에 쓰거나 미래에서 `bfill`하면 누수입니다. 위 예제는 고유 RangeIndex·예약 열 `_row`가 없는 작은 입력용입니다. 함수화할 때는 중복 index와 예약 열 충돌까지 명세대로 처리하세요.

window를 3으로 바꾸고 A 관측을 한 개 추가해 손계산과 비교하세요. `min_periods=2`면 과거 관측이 하나인 행도 NaN이어야 합니다. Problem에서는 valid/test 시점에 이용 가능한 과거만 사용합니다.

## 03. Process · NumPy 수치 함수·평가 지표

### 문제 신호와 선택할 코드

“NumPy만 사용”, “logits를 받아 평균 CE 반환”, “F1 직접 구현”이면 라이브러리 호출로 대체하지 말고 수식을 구현합니다. [26강 stable_softmax](../learn/26/)와 [32강 numpy_cross_entropy](../learn/32/)가 시작점입니다.

### 바꿀 부분

입력이 확률인지 logits인지, class 축, reduction(`none/sum/mean`), label dtype, 빈 입력·0 분모 정책을 확인합니다. 한 샘플당 class 하나라면 `y=[B]` 정수, logits=`[B,C]`입니다. multilabel BCE와 다른 문제입니다.

```python
import numpy as np

logits = np.array([[1000., 1000., 1000., 1000.]])
y = np.array([2], dtype=np.int64)
z = logits - logits.max(axis=1, keepdims=True)
log_prob = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
loss = float(-log_prob[np.arange(len(y)), y].mean())
assert np.isclose(loss, np.log(4))
```

### 손계산으로 확인하기

binary F1은 `2TP/(2TP+FP+FN)`입니다. TP=2, FP=1, FN=3이면 0.5입니다. Macro-F1은 **class마다 F1을 계산한 뒤 평균**이므로 양성 F1 하나와 다릅니다. 클래스 부재·정답 부재 시 정책은 명세대로 정하세요.

같은 logits 네 개의 CE는 `log(4)≈1.38629`; 모든 logits에 10000을 더해도 같은 값이어야 합니다. `y`가 범위를 벗어났거나 `(B,1)`이면 허용 여부를 검사하고 명시적으로 처리합니다. 위 짧은 예제는 입력 유효성 검사를 생략한 수식 확인용이며 제출 함수는 32강 계약 검사를 함께 구현하세요.

## 04. Process · crop·channel·이미지 정규화

### 문제 신호와 선택할 코드

“PIL crop”, “원 mode 유지”, “uint8 HWC를 float32 CHW로”를 구분합니다. 영역 자르기는 `crop_to_numpy(image, box)`. batch 배열을 PyTorch로 바꾸는 것은 `prepare_numpy_images(X, layout, divide_255)`입니다. 중앙 crop 함수는 [32강 center_crop_array](../learn/32/)에 있습니다.

### 바꿀 부분

PIL box는 `(left, upper, right, lower)`이고 right/lower는 제외 경계입니다. 이미지 크기는 PIL에서 `(W,H)`, 배열에서 `(H,W,C)`입니다. gray=1, RGB=3, RGBA=4 채널을 명세대로 유지하세요. 이미 0~1인 float를 다시 255로 나누지 않습니다.

```python
import numpy as np
from PIL import Image
import hdat_templates as h

raw = np.arange(5 * 7 * 3, dtype=np.uint8).reshape(5, 7, 3)
crop = h.crop_to_numpy(Image.fromarray(raw), (1, 1, 5, 4))
np.testing.assert_array_equal(crop, raw[1:4, 1:5])
batch = h.prepare_numpy_images(crop[None], layout="NHWC", divide_255=True)
assert batch.shape == (1, 3, 3, 4)
assert batch.dtype == np.float32
```

### 함정과 연습

일반 PIL crop은 범위 밖 box에서 padding 동작을 할 수 있습니다. 문제가 범위 밖을 오류로 처리하라고 하면 호출 전에 검사해야 합니다. mode를 지키라는 문제에서 `convert("RGB")`를 임의로 추가하지 마세요. 중앙 위치를 정할 때 홀수 차이의 내림/올림도 명세입니다.

gray 입력으로 바꾸면 단일 crop은 `(H,W)`, 모델 batch는 `(N,1,H,W)`여야 합니다. 정규화 `(x−mean)/std`는 채널당 mean/std를 `(C,1,1)`로 펼쳐 broadcast합니다. std=0 정책도 확인하세요.

## 05. Process · 지정 모델 구조·Conv 출력 계산

### 문제 신호와 선택할 코드

“다음 순서로 레이어 구성”, “마지막 시점의 LSTM 출력”, “parameter 수 반환”이면 정확한 명세 구현입니다. 범용 MLP/CNN을 가져와 비슷하게 만드는 문제가 아닙니다. [26강 SpecCNN](../learn/26/), [32강 ExamMLP·ExamLSTM·conv2d_info](../learn/32/)를 선택하세요.

### 바꿀 부분

Conv의 kernel/stride/padding/dilation/groups/bias, Linear의 in/out, activation 위치, hidden_size/num_layers, return shape를 표로 적습니다. 추가 BatchNorm·Dropout·softmax는 요구가 있을 때만 넣습니다.

```python
import torch
import hdat_templates as h

layer = torch.nn.Conv2d(3, 8, kernel_size=3, stride=2, padding=1, bias=True)
x = torch.zeros(2, 3, 11, 15)
y = layer(x)
assert y.shape == (2, 8, 6, 8)
assert sum(p.numel() for p in layer.parameters()) == 8 * (3 * 3 * 3 + 1)
assert h.conv_output_size(11, 3, stride=2, padding=1) == 6
```

### 통과 기준과 연습

출력 길이는 `floor((L+2P−D(K−1)−1)/S+1)`입니다. Conv2d weight 수는 `Cout×(Cin/groups)×Kh×Kw`, bias가 있으면 Cout를 더합니다. B=2뿐 아니라 `eval()`에서 B=1·H≠W도 검사하세요. `squeeze()`가 batch 축을 없애지 않게 합니다.

LSTM의 `output[:, -1, :]`와 `h_n`은 특히 양방향에서 같은 선택이 아닙니다. 문제의 단방향 “마지막 output”을 임의의 양방향 hidden 결합으로 바꾸지 마세요. stride=1·bias=False로 바꿔 출력과 parameter 수를 다시 계산하는 것이 다음 연습입니다. 고급 구조는 별도 부록으로 연결합니다.

## 06. Problem · 표형 회귀·다중 target·RMSLE

### 첫 코드와 계약

연속 정답 1개면 `task="regression"`, `MLP(n_features=F,out_dim=1)`입니다. target K개면 `out_dim=K`, y와 output 모두 `[B,K]`입니다. `MODEL_KIND="mlp"`가 표형 출발점입니다. `[N,F]`에 CNN1D를 바로 연결하면 차원 오류가 납니다.

실행: `python pytorch-problem-starter.py --case regression`. RMSLE 변형은 `--case rmsle`. 두 경우 모두 완성된 **합성 실행 경로**이며 실제 데이터 전처리는 아래처럼 교체합니다.

### 실제 DataFrame을 연결하는 위치

다음 블록은 `train_df`, `test_df`, 올바른 `tr_idx`, `va_idx`가 이미 있다는 전제의 **연결용 코드**입니다. 독립 실행 예제가 아닙니다. ID·미래 정보 열은 `DROP_COLS`로 빼고 test 열을 train feature 순서로 맞춥니다.

```python
import numpy as np
import hdat_templates as h
import torch  # loss_fn=torch.nn.L1Loss()로 바꿀 때 사용

TARGETS = ["target"]  # EDIT: 다중회귀면 정답 열을 지정 순서대로
DROP_COLS = ["id"]    # EDIT: 존재하는 ID/누수 열
y = train_df[TARGETS].to_numpy(dtype=np.float32)
X = h.clean_tabular_values(train_df.drop(columns=TARGETS + DROP_COLS))
Xt = h.clean_tabular_values(test_df[X.columns])
prep = h.make_preprocessor(X.iloc[tr_idx], encoding="onehot", scale_numeric=True)
a = prep.fit_transform(X.iloc[tr_idx])
b, c = prep.transform(X.iloc[va_idx]), prep.transform(Xt)

def dense_small(value):
    # EDIT: 환경에 맞는 한도. 세 배열·모델·복사본의 합은 별도 계산한다.
    if value.shape[0] * value.shape[1] * 4 > 200_000_000:
        raise MemoryError("OHE dense 한 배열이 200MB를 넘습니다. 범주/인코딩을 재설계하세요")
    return np.asarray(value.toarray() if hasattr(value, "toarray") else value, dtype=np.float32)

Xtr, Xva, Xte = map(dense_small, (a, b, c))
assert np.isfinite(Xtr).all() and np.isfinite(y).all()
model = h.MLP(n_features=Xtr.shape[1], out_dim=len(TARGETS))
tr = h.make_tensor_loader(Xtr, y[tr_idx], "regression", shuffle=True)
va = h.make_tensor_loader(Xva, y[va_idx], "regression")
model, history = h.train_torch_model(
    model, tr, va, "regression",
    score_fn=lambda truth, raw: float(np.mean(np.abs(truth - raw))),
    maximize=False,  # EDIT: 위 예시는 원 단위 MAE로 checkpoint 선택
)
test_pred = h.predict_torch(model, h.make_tensor_loader(Xte), "regression")
```

### loss / 선택 / 제출을 나누세요

기본 학습은 MSELoss, 위 선택 기준은 MAE, 제출은 연속값입니다. MAE 자체로 학습하려면 `loss_fn=torch.nn.L1Loss()`를 별도로 넘깁니다. 단일 output에서 MSE와 RMSE는 같은 checkpoint 순위를 주지만 MAE와는 보장되지 않습니다. 다중 output의 합산·평균 방식은 공식 산식과 맞추세요.

RMSLE는 raw y≥0을 확인한 후 `y_train_log=np.log1p(y_train)`으로 학습하고, validation·test raw output에 `expm1` 후 0 clipping을 적용합니다. callback의 truth도 log 공간이면 함께 복원합니다. `METRIC="rmsle"`만 바꿔서는 log 학습이 되지 않습니다. 큰 raw output의 exp overflow와 복원 후 finite를 검사합니다.

### 연습과 최종 검사

정답을 2개로 바꿀 때 y·out_dim·metric·EXPECTED_SHAPE 네 곳을 함께 바꾸세요. 타깃 표준화를 추가하면 train y로만 fit하고 callback과 제출 양쪽에서 inverse합니다. 단일 제출 `(N,)`는 마지막에만 `[:,0]`으로 바꿉니다. 첫 모델보다 train 평균 예측이 더 좋은지도 비교하세요.

더 읽기: [27강 표형 전체 흐름](../learn/27/), [33강 원 단위 checkpoint](../learn/33/).

## 07. Problem · 이진분류·불균형·확률 제출

### 첫 코드와 계약

정상/고장처럼 두 class입니다. `MLP(n_features=F,out_dim=1)`, `task="binary"`, y=`[B,1]` float 0/1, loss=`BCEWithLogitsLoss`. 모델 마지막에 sigmoid를 넣지 않습니다. 실행은 `python pytorch-problem-starter.py --case binary`입니다.

### 변경 지점

원 label이 2/5이고 고장=2라면 `POS_LABEL=2`, `NEG_LABEL=5`, `y_model=(y==2).astype("float32")`입니다. 먼저 train·valid에 허용 label만 있고 train에 둘 다 있는지 확인합니다. AUC는 valid에도 둘 다 있어야 정의됩니다. train에서만 `pos_weight=n_negative/n_positive`를 계산하고 callback에서 sigmoid를 씁니다.

### 확률과 label은 한 분기 안에서 처리

아래는 학습된 `model`, 순서가 고정된 `test_loader`에 연결합니다. 이때 모델은 위의 명시적 0/1 mapping으로 학습했어야 합니다.

```python
import numpy as np
import hdat_templates as h

POS_LABEL, NEG_LABEL = 2, 5          # EDIT: 문제의 의미
OUTPUT_KIND = "probability"          # EDIT: label 또는 probability
THRESHOLD = 0.5                      # EDIT: valid에서만 선택
prob = h.predict_torch(model, test_loader, "binary", return_proba=True).reshape(-1)
if OUTPUT_KIND == "probability":
    test_pred = prob                 # P(original label == 2)
elif OUTPUT_KIND == "label":
    test_pred = np.where(prob >= THRESHOLD, POS_LABEL, NEG_LABEL)
else:
    raise ValueError(OUTPUT_KIND)
if OUTPUT_KIND == "probability":
    assert np.isfinite(test_pred).all() and ((test_pred >= 0) & (test_pred <= 1)).all()
else:
    assert set(np.unique(test_pred)).issubset({POS_LABEL, NEG_LABEL})
```

### metric·threshold의 함정

AUC는 확률의 순위를 평가하므로 label로 자르지 않습니다. AUC가 평가 지표라고 제출 형식도 자동으로 확률이라고 단정하지 않습니다. Macro-F1이면 validation에서 threshold를 비교하고 test에는 고정합니다. threshold까지 반복 튜닝한 valid 점수는 낙관적일 수 있으므로 탐색을 제한하세요.

가중 BCE로 만든 sigmoid는 확률 보정이 보장되지 않습니다. 확률 품질이 중요한 지표라면 가중치 유무를 validation으로 비교합니다. 불균형이라고 accuracy만 보고 판단하거나 test label을 가정해 threshold를 조정하지 마세요.

연습: 양성을 0으로 뒤집어 mapping·확률 의미·label 복원이 모두 뒤집히는지 확인하세요. 확률 제출에서 `np.where(prob == 1, POS_LABEL, NEG_LABEL)`을 실행하면 거의 전부 음성으로 망가집니다.

## 08. Problem · 다중분류와 multilabel 구분

### 먼저 질문 하나

한 샘플의 정답이 정확히 하나인가요? 그렇다면 multiclass입니다. 여러 고장 종류가 동시에 참일 수 있으면 multilabel입니다. 연속값 열이 여러 개인 경우는 둘 다 아니라 다중회귀입니다.

| 구분 | y | raw output | 학습 | 제출 확률 |
|---|---|---|---|---|
| multiclass | `[B]` long, 0…C−1 | `[B,C]` | CrossEntropyLoss | softmax, 행합≈1 |
| multilabel | `[B,K]` float 0/1 | `[B,K]` | BCEWithLogitsLoss | 열별 sigmoid, 행합 제한 없음 |

실행은 `--case multiclass` 또는 `--case multilabel`입니다. 모델은 둘 다 MLP이지만 task·정답 형태·out_dim·metric이 다릅니다.

### class mapping과 확률 열

multiclass label encoder는 train fold로 fit하고 valid에 미지 class가 있으면 split을 다시 검토합니다. 시험에 전체 class 목록이 지정되면 그 목록을 우선합니다. label 제출은 argmax index를 원 label로 복원하고, 확률 제출은 복원 함수에 넣지 말고 **열 순서**를 맞춥니다.

```python
import numpy as np

classes = ["normal", "scratch", "dent"]    # 실제 학습 index 0,1,2의 의미
required = ["dent", "normal", "scratch"]  # EDIT: 제출 지정 열 순서
prob = np.array([[.1, .2, .7], [.8, .1, .1]])
assert len(required) == len(classes) and set(required) == set(classes)
test_pred = prob[:, [classes.index(c) for c in required]]
np.testing.assert_allclose(test_pred.sum(axis=1), 1)
np.testing.assert_allclose(test_pred[0], [.7, .1, .2])
```

### 가중 loss와 검사

가중 CE의 mean은 batch 크기가 아니라 정답 class weight 합으로 나뉩니다. 다운로드 공통 루프는 이 분모를 반영해 epoch loss를 합산합니다. 모델 학습 자체는 batch 크기에 영향을 받지만, **고정된 logits의 전체 loss 집계**는 batch를 나누는 방법과 같아야 합니다.

multilabel은 각 열이 0/1인지 검사합니다. AUC는 label별 양/음 부재 처리와 평균 방식을 확인하고, accuracy는 subset accuracy인지 열별 평균인지 읽으세요. class가 세 가지인 target 열이 여러 개 있는 categorical multi-output은 이 간단한 multilabel 경로 밖입니다. 각각의 head·loss·metric을 명세대로 설계해야 합니다.

연습: 확률 제출을 label 제출로 바꾸고 예상 shape가 `(N,C)`에서 `(N,)`으로 변하는 것을 설명해 보세요. multilabel에는 이 변환을 적용하지 않습니다.

## 09. Problem · 이미 만들어진 시계열 window

### 첫 코드와 계약

처음 받은 X가 `[N,L,F]`이면 N개의 sample window가 이미 있습니다. **window를 다시 만들지 않습니다.** `CNN1D(n_features=F,out_dim=K)`가 첫 후보이며 내부에서 `[B,L,F]→[B,F,L]`로 바꿉니다. 외부에서 한 번 더 transpose하지 마세요.

실행은 `--case window`입니다. 합성 예제는 서로 독립인 window를 만든 것이므로, 실제 겹치는 시계열에 예제 split을 복사하지 않습니다. 원 timestamp·group·window 시작/끝 metadata로 검증을 설계합니다.

### 바꿀 부분과 연결

```python
import hdat_templates as h

# X_train/X_valid/X_test는 이미 올바르게 나눈 [N,L,F] 배열
scaler = h.fit_scale_3d(X_train)
X_train = h.transform_scale_3d(X_train, scaler)
X_valid = h.transform_scale_3d(X_valid, scaler)
X_test = h.transform_scale_3d(X_test, scaler)
model = h.CNN1D(n_features=X_train.shape[2], out_dim=2, channels=(32, 64))
# EDIT: 위 out_dim=2는 target 2개 회귀 예. 06~08의 task/loss 계약을 적용
```

기존 L은 문제에서 정한 sample 계약일 수 있으므로 임의로 줄이지 않습니다. 메모리를 줄일 때는 batch/channel부터 줄입니다. scaler는 train의 N·L을 펼쳐 feature별로 fit한 뒤 같은 F로 transform합니다.

### 모델 하나만 비교하기

RNN을 비교하려면 `SequenceRNN(n_features=F,out_dim=K,hidden_size=64,num_layers=1,kind="gru")`로 모델 생성만 교체합니다. LSTM은 `kind="lstm"`. DataLoader와 loss·metric·제출은 같은 계약을 유지합니다. `MLP`는 window를 펼치는 대안이지만 입력 feature 차원이 `L×F`가 됩니다.

검사: 원 sample마다 예측 한 행, test N 유지, 마지막 batch B=1, 원 target 단위 metric. 같은 raw 시점이 겹친 sample을 random split하면 누수가 생길 수 있습니다. 단순히 time순 정렬만으로 모든 중복·label 지연 문제가 해결되지 않으므로 다음 raw-time 가이드의 예측 원점 조건을 확인하세요.

## 10. Problem · raw 시계열에서 미래 한 시점 예측

### 인덱스부터 정의

입력이 `[T,F]`이고 “과거 L개로 마지막 관측 H칸 뒤를 예측”한다면 `make_point_forecast_windows`를 씁니다. 마지막 관측 e의 입력은 `X[e−L+1:e+1]`, 정답은 `y[e+H]`입니다. H=1은 다음 행입니다. 초와 sample 수를 혼동하지 말고 sampling rate와 실제 시간 간격을 확인하세요.

### 작게 만들어 첫·마지막을 검산

```python
import numpy as np
import hdat_templates as h

raw = np.arange(100)[:, None]
train, valid, cut = h.split_raw_time_then_window(
    raw, raw[:, 0], train_ratio=.7, lookback=5, horizon=10,
    assume_sorted=True, split_mode="forecast_origin", label_delay=0,
)
Xtr, ytr, train_target = train
Xva, yva, valid_target = valid
assert train_target.max() == 69
assert Xva[0, -1, 0] == 70 and valid_target[0] == 80
assert train_target.max() < (valid_target - 10).min()
```

### horizon이 길 때의 누수

target<70을 train, target≥70을 valid로만 나누면, target=70을 예측하는 시점은 e=60입니다. 그런데 train은 y[69]를 이미 배운 상태라 e=60에 알 수 없는 미래 정답을 학습에 쓴 셈입니다. 기본 함수는 train label이 경계 전에 알려지고 valid의 입력 끝이 경계 이후인 sample만 선택합니다.

`label_delay`는 정답 확정 지연의 **행 수**, `gap`도 raw 행 단위입니다. 실제 timestamp·불규칙 간격이면 시간으로 구현합니다. `split_mode="target"`은 별도의 배치평가 명세에 명시적으로 맞출 때만 쓰며 실시간 예측의 기본값이 아닙니다.

### group·test·메모리 분기

차량별 raw는 `make_grouped_point_forecast_windows`로 group 경계를 지킵니다. 반환 target row는 원 데이터 행입니다. 섞여 있는 global row 번호에서 H를 빼면 같은 group의 예측 원점을 찾을 수 없습니다. group 안의 정렬 위치에서 H를 빼고 실제 timestamp로 변환해야 합니다. 공동 모델이면 모든 train label이 첫 valid 원점 전에 알려져야 하며, 새 차량 평가까지 요구되면 group도 분리합니다.

test는 target이 없어 자동 window 개수 추정이 위험합니다. 주어진 starts·예측 대상 행과 expected_n을 읽고 `LazyWindowDataset`의 test 계약을 적용합니다. 균일한 단일 raw 시계열에서 window 수는 `max(0,floor((T−L−H)/stride)+1)`입니다. 다단계 전체 trajectory 예측은 이 단일 target-time 함수와 다른 계약입니다.

연습: H=1→10, stride=1→3, label_delay=0→4로 바꾸고 모든 train 정답 확정 시점이 첫 valid 원점보다 빠른지 검사하세요. 전체 실행 흐름은 [시계열 모의 코드](./full-mock-timeseries.py)에 있습니다.

## 11. Problem · 이미지 분류·회귀

### 첫 코드와 계약

배열이면 `prepare_numpy_images` → `SmallImageCNN(in_channels=C,out_dim=D)` → 공통 학습입니다. 실행은 `--case image`. 파일 경로 목록이면 긴 치트시트 13절의 `ImagePathDataset` 경로를 **별도 예제 계열로** 선택하거나 [이미지 통합 모의](./full-mock-image-ae.py)를 사용하세요. 경로 문자열을 `make_tensor_loader`에 넣지 않습니다.

### 바꿀 부분

layout=`NHWC/NCHW`, divide_255, resize 높이·너비, in_channels, task, out_dim, label mapping을 정합니다. 현재 작은 CNN은 두 번 pooling하므로 최소 H/W≥4가 필요합니다. 분류는 07/08의 loss, 회귀는 06의 원 단위 metric·출력 계약을 적용합니다.

### 학습·평가 전처리를 구분

train에만 안전한 augmentation을 넣고 valid/test는 deterministic하게 합니다. 좌우 방향이 target인 문제에 horizontal flip, 좌표 회귀인데 target 좌표를 함께 바꾸지 않은 crop/resize는 정답을 망가뜨립니다. path·label·ID는 같은 행으로 움직여야 합니다.

pretrained weight는 제공·캐시·사용 허용을 확인한 뒤 사용합니다. 인터넷 다운로드가 가능하다고 가정하지 않습니다. random weight의 backbone을 freeze하면 특징 추출기는 학습되지 않아 사전학습 전이 효과를 기대할 수 없습니다. head는 학습할 수 있지만 좋은 특징이 보장되는 것은 아닙니다. 먼저 작은 CNN으로 저장까지 통과하세요.

### 연습과 통과 기준

합성 이미지의 RGB를 grayscale로 바꾸고 C=1, batch shape, 첫 Conv를 함께 바꿉니다. H≠W·B=1에서 output은 `[B,D]`를 유지해야 합니다. float 0~1 입력의 `divide_255=False`도 검사하세요. 회귀라면 마지막에 softmax 없이 연속값을 반환하고, 표준화한 y는 metric·제출 양쪽에서 원 단위로 복원합니다.

## 12. Problem · 정상-only Autoencoder 이상 탐지

### 첫 코드와 계약

정상 데이터만으로 학습해서 샘플별 이상 정도를 구하는 문제입니다. 표형은 `Autoencoder(input_dim=F,latent_dim=...,hidden_dim=...)`, 입력 X 자체를 정답으로 주는 회귀 학습입니다. 실행은 `--case anomaly`. 이미지에는 [full-mock-image-ae.py](./full-mock-image-ae.py)의 `ConvAutoencoder`와 `reconstruction_errors`를 선택합니다. 표형 AE에 4D 배열을 그대로 넣지 않습니다.

### 바꿀 부분

정상 train mask, 정상 validation, train-only scaler, latent_dim, sample별 error reduction 축, score 방향, 제출 종류를 정합니다. 정상/이상 label이 충분하면 지도학습 이진분류도 비교할 수 있습니다.

### score와 label을 확실하게 연결

표형 재구성 오차는 `mean((X−recon)^2, axis=1)`로 sample당 하나입니다. 이미지라면 `(1,2,3)`을 평균해야 `[N]`이 됩니다. batch 전체 loss scalar 하나나 `[N,H,W]`를 제출하지 마세요.

```python
import numpy as np

X = np.array([[0., 0.], [3., 3.]])
recon = np.zeros_like(X)
test_error = ((X - recon) ** 2).mean(axis=1)
threshold = 1.0  # EDIT: 정상 valid 분위수 또는 labeled valid에서 결정
OUTPUT_KIND = "anomaly_score"
if OUTPUT_KIND == "anomaly_score":
    test_pred = test_error
elif OUTPUT_KIND == "label":
    test_pred = (test_error > threshold).astype(int)  # 이상=1
else:
    raise ValueError(OUTPUT_KIND)
np.testing.assert_allclose(test_error, [0, 9])
assert test_pred.shape == (2,)
```

### threshold의 의미와 연습

정상 validation의 99% 분위수는 연습용 출발점이지 최적 기준이 아닙니다. label이 있으면 공식 F1 등으로 threshold를 비교하고 고정합니다. label이 없으면 “정상 valid 중 약 1%를 이상으로 잡는 기준”이지 미래의 오탐율 1% 보장이 아닙니다. 정상=1이라면 label을 뒤집지만 error score 방향은 명세를 따릅니다.

기존 코드의 `pred_anomaly`를 계산하고 다른 `test_pred`를 저장하는 실수를 막기 위해 위처럼 **공통 제출 변수에 직접 할당**합니다. 연습에서는 score/label을 번갈아 선택하고 shape·값 범위·저장 변수가 모두 맞는지 확인하세요.

## 13. 공통 마무리 · checkpoint·제출·오류 복구

### checkpoint를 고른 뒤

`train_torch_model`은 최상의 monitor 시점 가중치를 복원하고 `(model, history)`를 반환합니다. score_fn은 전체 validation truth와 **raw output** NumPy 배열을 받습니다. classification이면 sigmoid/softmax/argmax, 회귀 변환이면 원 단위 복원을 callback 안에서 수행해야 합니다. 합성 starter의 `official_score`를 기준으로 고칩니다.

최종 전체 train 재학습은 선택 단계입니다. 진행한다면 best epoch 수를 고정하고, 새 model·새 optimizer·전체 train 전용 새 전처리로 학습합니다. held-out valid를 없앤 상태에서 같은 valid 점수로 early stopping했다고 주장하지 않습니다. 제한 시간이 부족하면 검증된 checkpoint를 그대로 사용하세요.

### 저장 전에 독립적으로 정한 계약과 대조

```python
import numpy as np

# EDIT: 문제에 적힌 값을 읽어 지정. test_pred.shape를 베껴 넣지 않는다.
N_TEST, K = 20, 1
EXPECTED_SHAPE = (N_TEST, K)
p = np.asarray(test_pred)
assert p.shape == EXPECTED_SHAPE, (p.shape, EXPECTED_SHAPE)
assert np.issubdtype(p.dtype, np.number)
assert np.isfinite(p).all()
# 확률이면 0~1, multiclass 확률이면 행합≈1, label이면 허용 class 집합 검사.
# 이후 제공 저장 셀을 그대로 실행. 제공 셀이 없는 개인 연습에서만:
np.save("practice_submission.npy", p)
reloaded = np.load("practice_submission.npy", allow_pickle=False)
np.testing.assert_array_equal(reloaded, p)
```

문자 label을 CSV로 제출하는 별도 계약이면 숫자 assert 대신 label 집합·열명·ID 순서를 검사합니다. 무조건 float32로 바꾸면 원 label이 손상될 수 있습니다. 위 예제의 파일명은 공식 제출명이 아닙니다.

### 오류를 좁히는 순서

| 증상 | 첫 검사 | 조치 |
|---|---|---|
| NameError/import 실패 | 파일 이름·앞 정의·같은 폴더 | `hdat_templates.py` 및 00번 의존 블록 확인 |
| mat1/mat2 오류 | 전처리 후 F와 Linear in_features | 모델을 변환 후 feature 수로 생성 |
| Conv channel 오류 | `[B,L,F]`/`[B,F,L]`, NHWC/NCHW | source 내부 transpose와 중복하지 않기 |
| BCE/MSE shape 오류 | output과 y의 정확한 tuple | `[B,1]`로 일치, B=1까지 검사 |
| CE 오류 | y long, 0…C−1 | mapping·범위·마지막 softmax 제거 |
| NaN/Inf | X/y/복원 값, loss, lr | 입력 유한값부터 검사하고 LR 낮추기 |
| CUDA OOM | batch·모델·window 복사 | batch 절반, channel 축소, lazy window |
| AUC 계산 불가 | valid class 수·probability shape | split 재검토, 임의 0점으로 숨기지 않기 |
| 점수만 비정상적으로 좋음 | time/group·사후 feature | 모델 튜닝 전에 누수부터 제거 |
| 예측 N 불일치 | test shuffle·정렬·window target 대응 | sample ID 기준 원순서 복원 |

### 마지막 30초

함수명·변수명·파일명, Process/Problem 각각 저장, test 행 순서, 출력 종류·shape·dtype, NaN/Inf, 저장 후 reload, 제출 완료 표시를 확인합니다. 문항 재방문을 가정한 운영은 피하고 당일 이동 규칙을 먼저 확인하세요.

## 14. 부록 · ResNet·Transformer·VAE·GAN의 지원 경계

이 구조들은 공개 학습 범위에 포함되지만 **이름이 범위에 있다고 모든 것이 특정 실기 유형으로 출제된다는 뜻은 아닙니다.** 구조가 직접 지정되면 정확히 구현하고, 자유 모델 선택 문제에서는 작은 baseline 이후 validation 근거가 있을 때 비교하세요.

| 명세 신호 | 다운로드 소스 | 반드시 추가로 확인 |
|---|---|---|
| residual/skip connection | `ResidualBlock2D(in_channels,out_channels,stride)` | skip과 main shape, projection, activation 위치 |
| sequence attention | `TransformerSequenceModel(n_features,out_dim,d_model,nhead,...)` | d_model % nhead=0, 위치·padding·causal mask |
| 잠재 평균·분산·KL | `VariationalAutoencoder` + `vae_loss` | `(recon,mu,logvar)` unpack, KL reduction·beta |
| 생성자·판별자 교대 | `MLPGenerator`, `MLPDiscriminator` | 두 optimizer, D용 fake.detach(), G용 gradient 경로 |

### 공통 루프를 그대로 쓰면 안 되는 경우

VAE는 모델이 tuple을 반환하므로 단일 output MSE 공통 루프가 맞지 않습니다. GAN은 두 네트워크를 교대로 업데이트하고 하나의 supervised y가 없는 구조라 전용 루프가 필요합니다. [긴 치트시트 14A 부록](../cheatsheet.html)에서 구조를 찾고 [23강](../learn/23/)·[24강](../learn/24/)·[25강](../learn/25/)에서 loss와 학습 원리를 대조하세요.

현재 Transformer는 고정 길이 window의 단일 예측용 encoder+pooling입니다. padding_mask를 받는 모델 함수가 있어도 기본 DataLoader/공통 루프는 `(X,y)`만 전달하므로 가변길이는 collate와 학습 루프도 바꿔야 합니다. 모든 토큰이 padding인 sample은 막아야 합니다. autoregressive 다음 토큰 예측, segmentation, 여러 미래 시점 전체 trajectory, categorical multi-output은 별도 출력·loss·mask 설계가 필요합니다.

### 구조 smoke test

```python
import torch
import hdat_templates as h

block = h.ResidualBlock2D(3, 8, stride=2).eval()
assert block(torch.zeros(1, 3, 12, 16)).shape == (1, 8, 6, 8)
model = h.TransformerSequenceModel(3, 2, d_model=8, nhead=2, num_layers=1).eval()
assert model(torch.zeros(1, 5, 3)).shape == (1, 2)
```

이 검사는 실행 가능한 shape 계약만 확인하며, 학습이 올바르거나 성능이 좋다는 보장은 아닙니다. 실제 지정 구조와 parameter 수·loss·gradient를 따로 확인하세요.

공식 API 대조: [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.7/generated/torch.nn.CrossEntropyLoss.html), [BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.7/generated/torch.nn.BCEWithLogitsLoss.html), [TransformerEncoder](https://docs.pytorch.org/docs/2.7/generated/torch.nn.TransformerEncoder.html).
