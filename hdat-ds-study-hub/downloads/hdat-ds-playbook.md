# HDAT-DS 실기 · 문제에서 코드로

2026-09-07 · Astra 감사 반영 · 비공식 자체 제작 학습자료. 아래 유형은 공개 범위를 학습하기 위한 분류이지 출제 예측이나 실제 기출 복제가 아닙니다. 딥러닝 모델과 학습은 PyTorch로 통일했습니다. NumPy/pandas 및 sklearn 전처리·지표는 보조 도구입니다.

## 00. 먼저 읽기 · 요구사항과 코드 사용법

문제에서 중요한 것은 모델 이름보다 **받는 값과 돌려줄 값의 약속**입니다. Process는 정해진 함수·클래스의 동작을 구현하고, Problem은 데이터를 나누고 모델을 학습해서 예측을 제출하는 흐름입니다. 같은 MLP라도 정답이 연속값인지 클래스 번호인지에 따라 출력 차원과 손실함수가 달라집니다.

### 1. 코딩 전에 여섯 칸을 채우세요

| 확인 | 적을 내용 | 잘못 고르면 생기는 일 |
|---|---|---|
| 입력 | DataFrame·배열·경로, 배열 크기(shape), 원소 자료형(dtype), 한 행의 의미 | 이미 윈도로 제공된 입력을 다시 윈도로 묶음 |
| 정답 | 연속값·단일 클래스·여러 0·1, 양성 라벨, 열 순서 | 다중회귀를 다중라벨 분류로 오해 |
| 검증 | 독립 행/같은 개체/미래/새 개체의 미래 | 누수로 좋은 점수만 보고 모델 선택 |
| 학습 | 모델 구조와 손실함수 | CE 앞에 소프트맥스를 중복 적용 |
| 선택 | 공식 평가지표와 최적화 방향(높을수록·낮을수록 좋음) | MAE로 평가하는 시험에서 MSE가 최소인 모델 선택 |
| 제출 | 라벨·확률·연속값·이상 점수, 정확한 배열 크기·파일명 | AUC로 평가한다는 이유만으로 확률 파일을 제출한다고 추측 |

**손실(loss), 모델 선택 기준(metric), 제출 변환은 세 가지 별도 설정**입니다. MSELoss로 학습하면서 MAE로 체크포인트를 고를 수도 있습니다. 지표 이름만 바꿔도 이 세 설정이 자동으로 맞춰지지 않습니다.

### 2. 두 파일을 같은 폴더에 저장하세요

함수의 인자 이름은 알겠는데 실제로 무엇을 넣을지 막힌다면 [템플릿 호출 매뉴얼](../templates/)부터 보세요. 작은 데이터를 직접 만드는 단계부터 설명하고, 모든 함수·모델의 인자를 값의 출처와 연결합니다. 인터넷 없이 읽을 때는 [오프라인 호출 매뉴얼](./hdat-template-guide.html)을 저장하세요.

[hdat_templates.py](./hdat_templates.py)는 함수·모델 모음이고, [pytorch-problem-starter.py](./pytorch-problem-starter.py)는 이를 연결한 합성 데이터로 실행해 볼 수 있는 작은 예제입니다. 기존 하이픈 이름 `hdat-templates.py`도 같은 내용으로 유지하지만 불러올 때는 **밑줄 이름**을 쓰세요. 로컬 학습 환경에서 NumPy·pandas·scikit-learn·PyTorch가 필요합니다. 이미지 Process에는 Pillow가 필요합니다.

```bash
python pytorch-problem-starter.py --case regression --epochs 3
python pytorch-problem-starter.py --case binary --epochs 3
```

회귀는 `(20, 1)`, 이진분류 확률도 `(20, 1)`이 출력되며 마지막에 `finite OK`가 보여야 합니다. 합성 데이터의 점수는 합격 가능성이나 실제 문제 성능을 뜻하지 않습니다. `--output practice.npy`를 추가하면 새 파일을 저장하고 다시 읽어 동일성을 검사합니다. 이미 있는 파일은 덮어쓰지 않습니다.

### 3. 코드를 바꾸는 순서

1. 먼저 수정 없이 실행해 실행 환경과 모듈을 제대로 불러오는지 확인합니다. 파일 이름 뒤에 `.txt`가 붙지 않았는지 확인하세요.
2. `CHANGE DATA`: 합성 배열을 실제 입력 X·정답 y·테스트 자료로 교체합니다. ID와 정답이 입력 특성에 들어가지 않게 합니다.
3. `CHANGE SPLIT`: 서로 독립인 합성 예제의 80/20 슬라이싱을 실제 문제에 맞는 시간·그룹·계층화 분할로 바꿉니다.
4. 훈련자료로만 전처리를 학습합니다. 변환 **후** 특성 수에 맞춰 모델을 만듭니다.
5. `CHANGE METRIC`, `CHANGE LOSS`, `CHANGE OUTPUT`, `CHANGE CONTRACT`를 문제 명세로 바꿉니다.
6. 작은 배치로 1회 학습해 학습→검증→추론→저장을 끝까지 실행한 뒤 학습 시간을 늘립니다.

### 4. 의존 블록을 빠뜨리지 마세요

로컬에서는 `import hdat_templates as h` 한 줄로 같은 모듈의 함수와 모델을 사용합니다. 블록을 따로 옮기는 것이 허용된 상황에서는 파일 상단의 import 문과 함께 모델 정의, `make_tensor_loader`, `make_torch_loss`, `_batch_loss`, `train_torch_model`, `predict_torch`를 확인합니다. `[TORCH-TRAIN]` 함수 하나만 가져오면 앞선 정의가 없어 NameError가 납니다. 예측만 한다면 최적화 알고리즘은 필요하지 않습니다.

기존 긴 치트시트의 `fit(...) → model`과 다운로드 소스의 `train_torch_model(...) → (model, history)`는 다른 예제입니다. 새 가이드는 후자를 사용합니다. 기본 `score_fn=None`이면 검증 손실이 최소인 모델을 고르고, `score_fn`을 주면 전체 검증자료의 공식 지표로 선택합니다. AUC/F1/Accuracy는 `maximize=True`, RMSE/MAE/RMSLE는 `False`입니다.

### 5. 시험 규정은 별도로 확인하세요

이 파일의 다운로드·불러오기가 시험에서 허용된다는 의미가 아닙니다. [NGV 공식 안내](https://exam.hyundai-ngv.com/practice/13567)는 생성형 AI 활용을 금지합니다. “One-way only”는 오픈북 안내 문맥이며, 이것만으로 화면 이동·재진입 제한을 단정할 수 없습니다. 개인 파일·사이트 사용과 문항 이동은 해당 회차 안내를 확인하세요. 제공된 기본 코드·저장 셀이 이 문서보다 우선합니다.

### 6. B·T·F나 broadcasting이 낯설다면

모델 코드를 바꾸기 전에 [입문 6강: shape·permute·broadcasting과 13문제 해설](https://markshincuhk.github.io/hdat-ds-study-hub/start/06/)을 읽으세요. `[B,T,F]`는 배치 안의 샘플 수·샘플당 시점 수·시점당 특성 수입니다. RNN은 `batch_first=True`일 때 이 순서를 쓰며, Conv1d는 보통 F와 T를 교환합니다. 단, 이 가이드의 CNN1D 구현은 내부에서 축을 교환하므로 입력을 또 바꾸지 않습니다.

한 출력 회귀의 `[B,1]-[B]`는 `[B,B]`로 계산되는 함정이 있습니다. 두 배열의 크기를 먼저 적고 한 샘플씩 대응하는지 확인하세요. 축을 교환하는 permute, 원소를 다른 크기로 묶는 reshape, 계산할 때 값을 반복 적용하는 브로드캐스팅(broadcasting)은 서로 다릅니다. 기초 강의는 온라인 링크이며 요약은 내려받은 치트시트의 배열 크기·손실 절에도 들어 있습니다.

## 01. Process · 선택 열 변환·결측·이상치

### 문제 신호와 선택할 코드

“지정 열만 0~1”, “중앙값과 IQR”, “clip/제거/flag”가 보이면 모델을 학습하는 문제가 아니라 DataFrame의 입출력 조건을 구현하는 문제입니다. 최소·최대 정규화(Min-Max)는 `minmax_selected`; 값의 범위 제한(clipping)은 `fit_iqr_bounds`와 `apply_clip_bounds`를 사용합니다. **clip은 범위 밖 값을 경계값으로 바꾸고, remove는 행을 지우고, flag는 표시 열을 만드는 작업**이므로 서로 대체할 수 없습니다.

### 바꿀 부분

`columns`, 반환 함수명, 상수열·NaN 정책, IQR 계수 `whisker`, 원본 수정 여부를 먼저 적습니다. 표준화라면 평균·표준편차와 `ddof`, 로버스트 스케일링(robust scaling)이라면 `(x−median)/IQR`을 요구대로 구현합니다. Process 함수 안에서 통계를 구하는 명세와, Problem에서 훈련 통계를 검증·테스트 자료에 재사용하는 절차를 구분하세요.

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

두 값이 아주 가까워도 같지 않으면 이 예제의 양 끝점은 0과 1입니다. `np.isclose(max,min)`로 상수열을 판정하면 서로 다른 값이 있는 열도 모두 0으로 바꿀 수 있습니다. 상수 판정에 허용 오차를 쓰라는 별도 명세가 있을 때만 바꾸세요.

### 자가 연습과 통과 기준

`x=[3,NaN,3]`으로 바꾸면 `[0,NaN,0]`; 전부 NaN이면 그대로 NaN; `columns=[]`이면 값이 같은 복사본이어야 합니다. 열 순서·인덱스·지정하지 않은 열·원본까지 검사해야 통과입니다. 없는 열은 이 템플릿에서 KeyError입니다. 실제 문제가 다른 정책을 요구하면 수정하세요.

더 읽기: [26강 Process 구현](../learn/26/), [32강 robust scaling 독립 모의](../learn/32/).

## 02. Process · 그룹별 시차·이동 통계·시간 특성

### 문제 신호와 선택할 코드

“각 차량별 이전 n개 평균”, “현재 행 제외”, “원래 행 순서 유지”는 **정렬 → 그룹별 처리 → 이전 값 이동 → 이동 통계 → 원래 순서 복원**입니다. 날짜의 연/월/요일만 필요하면 소스 `add_datetime_features`; 과거 평균은 [32강의 add_past_mean](../learn/32/)을 선택합니다.

### 바꿀 부분

`group_col`, `time_col`, `value_col`, `window`, `min_periods`, 같은 시간의 순서 정책을 적습니다. 현재 값을 제외하려면 이동 통계를 구하기 전에 `shift(1)`합니다. 아래 예제는 window=2, 최소 유효값 1개, 같은 시각이면 입력 순서를 유지한다는 **연습용 조건**을 따릅니다.

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

그룹 A의 첫 관측은 B의 99를 가져오면 안 됩니다. `shift` 없이 계산하면 현재값을 과거 특성에 넣습니다. 실제 예측 시점에 아직 모르는 정답으로 이동 통계를 계산하거나 미래 값으로 `bfill`하면 누수입니다. 위 예제는 고유 RangeIndex·예약 열 `_row`가 없는 작은 입력용입니다. 함수화할 때는 중복 인덱스와 예약 열 충돌까지 명세대로 처리하세요.

이동평균 구간의 길이를 3으로 바꾸고 A 관측을 한 개 추가해 손계산과 비교하세요. `min_periods=2`면 과거 관측이 하나인 행도 NaN이어야 합니다. Problem에서는 검증·테스트 예측 시점에 이용 가능한 과거만 사용합니다.

## 03. Process · NumPy 수치 함수·평가 지표

### 문제 신호와 선택할 코드

“NumPy만 사용”, “logits를 받아 평균 CE 반환”, “F1 직접 구현”이면 라이브러리 호출로 대체하지 말고 수식을 구현합니다. [26강 stable_softmax](../learn/26/)와 [32강 numpy_cross_entropy](../learn/32/)가 시작점입니다.

### 바꿀 부분

입력이 확률인지 로짓(logits)인지, 클래스 축, 손실 집계 방식(reduction)(`none/sum/mean`), 정답의 원소 자료형, 빈 입력·0 분모 정책을 확인합니다. 한 샘플당 클래스가 하나라면 `y=[B]` 정수, logits=`[B,C]`입니다. 다중라벨 분류의 BCE와 다른 문제입니다.

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

이진분류 F1은 `2TP/(2TP+FP+FN)`입니다. TP=2, FP=1, FN=3이면 0.5입니다. Macro-F1은 **클래스마다 F1을 계산한 뒤 평균한 값**이므로 양성 F1 하나와 다릅니다. 클래스 부재·정답 부재 시 정책은 명세대로 정하세요.

값이 같은 로짓 네 개의 CE는 `log(4)≈1.38629`; 모든 로짓에 10000을 더해도 같은 값이어야 합니다. `y`가 범위를 벗어났거나 `(B,1)`이면 허용 여부를 검사하고 명시적으로 처리합니다. 위 짧은 예제는 입력 유효성 검사를 생략한 수식 확인용이며 제출 함수는 32강의 입력 조건 검사를 함께 구현하세요.

## 04. Process · 자르기·채널·이미지 정규화

### 문제 신호와 선택할 코드

“PIL crop”, “원 mode 유지”, “uint8 HWC를 float32 CHW로”를 구분합니다. 영역을 자를 때는 `crop_to_numpy(image, box)`. 배치 배열을 PyTorch로 바꿀 때는 `prepare_numpy_images(X, layout, divide_255)`입니다. 중앙 자르기 함수는 [32강 center_crop_array](../learn/32/)에 있습니다.

### 바꿀 부분

PIL 자르기 영역(box)는 `(left, upper, right, lower)`이고 right/lower 위치는 결과에 포함되지 않습니다. 이미지 크기는 PIL에서 `(W,H)`, 일반적인 컬러 배열에서 `(H,W,C)`입니다. 일반적인 흑백 PIL 배열은 `(H,W)`로 채널 축이 없을 수 있습니다. 흑백=1, RGB=3, RGBA=4 채널을 명세대로 유지하세요. 이미 0~1인 float를 다시 255로 나누지 않습니다.

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

일반적인 PIL 자르기는 이미지 범위를 벗어난 영역에 패딩을 채울 수 있습니다. 문제가 범위 밖을 오류로 처리하라고 하면 호출 전에 검사해야 합니다. 모드를 보존하라는 문제에서 `convert("RGB")`를 임의로 추가하지 마세요. 중앙 위치를 정할 때 홀수 차이의 내림/올림도 명세입니다.

흑백 입력으로 바꾸면 이미지 하나를 자른 결과은 `(H,W)`, 모델 배치는 `(N,1,H,W)`여야 합니다. 정규화 `(x−mean)/std`는 채널별 평균·표준편차를 `(C,1,1)`로 만들어 브로드캐스팅합니다. std=0 정책도 확인하세요.

## 05. Process · 지정 모델 구조·Conv 출력 계산

### 문제 신호와 선택할 코드

“다음 순서로 레이어 구성”, “마지막 시점의 LSTM 출력”, “매개변수 수 반환”이면 정확한 명세 구현입니다. 범용 MLP/CNN을 가져와 비슷하게 만드는 문제가 아닙니다. [26강 SpecCNN](../learn/26/), [32강 ExamMLP·ExamLSTM·conv2d_info](../learn/32/)를 선택하세요.

### 바꿀 부분

Conv의 커널·보폭·패딩·팽창률·그룹 수·편향 설정, Linear의 입력·출력 차원, 활성함수 위치, 은닉 크기·층 수, 반환 배열의 크기를 표로 적습니다. 추가 BatchNorm·Dropout·softmax는 요구가 있을 때만 넣습니다.

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

출력 길이는 `floor((L+2P−D(K−1)−1)/S+1)`입니다. Conv2d 가중치 수는 `Cout×(Cin/groups)×Kh×Kw`, 편향이 있으면 Cout를 더합니다. B=2뿐 아니라 `eval()`에서 B=1·H≠W도 검사하세요. `squeeze()`가 배치 축을 없애지 않게 합니다.

LSTM의 `output[:, -1, :]`와 `h_n`은 특히 양방향에서 같은 선택이 아닙니다. 문제의 단방향 “마지막 출력”을 임의의 양방향 은닉 상태 결합으로 바꾸지 마세요. stride=1·bias=False로 바꿔 출력과 매개변수 수를 다시 계산하는 것이 다음 연습입니다. 고급 구조는 별도 부록으로 연결합니다.

## 06. Problem · 표 데이터 회귀·다중 정답·RMSLE

### 첫 코드와 계약

연속 정답 1개면 `task="regression"`, `MLP(n_features=F,out_dim=1)`입니다. 정답이 K개면 `out_dim=K`, 정답 y와 출력 모두 `[B,K]`입니다. `MODEL_KIND="mlp"`가 표 데이터 문제의 첫 선택입니다. `[N,F]`에 CNN1D를 바로 연결하면 차원 오류가 납니다.

실행: `python pytorch-problem-starter.py --case regression`. RMSLE 변형은 `--case rmsle`. 두 경우 모두 합성 데이터로 끝까지 실행할 수 있는 **완성 예제**이며 실제 데이터 전처리는 아래처럼 교체합니다.

### 실제 DataFrame을 연결하는 위치

다음 블록은 `train_df`, `test_df`, 올바른 `tr_idx`, `va_idx`가 이미 있다는 전제의 **연결용 코드**입니다. 독립 실행 예제가 아닙니다. ID·미래 정보 열은 `DROP_COLS`로 빼고 테스트 열을 훈련 특성 순서로 맞춥니다.

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

기본 학습은 MSELoss, 위 선택 기준은 MAE, 제출은 연속값입니다. MAE 자체로 학습하려면 `loss_fn=torch.nn.L1Loss()`를 별도로 넘깁니다. 단일 출력에서는 MSE와 RMSE는 체크포인트의 순위가 같지만 MAE까지 같은 순위가 되지는 않습니다. 다중출력의 합산·평균 방식은 공식 산식과 맞추세요.

RMSLE는 원래 정답 y≥0을 확인한 후 `y_train_log=np.log1p(y_train)`으로 학습하고, 검증·테스트 모델 출력에 `expm1` 후 음수값을 0으로 제한합니다. 콜백(callback)에 전달된 정답도 로그값이면 함께 복원합니다. `METRIC="rmsle"`만 바꿔서는 정답에 로그를 취해 학습하는 것은 아닙니다. 모델 출력이 클 때 지수 연산의 오버플로와 복원 후 유한성을 검사합니다.

### 연습과 최종 검사

정답을 2개로 바꿀 때 y·out_dim·metric·EXPECTED_SHAPE 네 곳을 함께 바꾸세요. 타깃 표준화를 추가하면 훈련 정답 y로만 학습하고 콜백과 제출 양쪽에서 역변환합니다. 단일 제출 `(N,)`는 마지막에만 `[:,0]`으로 바꿉니다. 첫 모델보다 훈련 정답 평균을 이용한 예측이 더 좋은지도 비교하세요.

더 읽기: [27강 표형 전체 흐름](../learn/27/), [33강 원 단위 checkpoint](../learn/33/).

## 07. Problem · 이진분류·불균형·확률 제출

### 첫 코드와 계약

정상/고장처럼 두 클래스입니다. `MLP(n_features=F,out_dim=1)`, `task="binary"`, y=`[B,1]` float 0/1, loss=`BCEWithLogitsLoss`. 모델 마지막에 시그모이드를 넣지 않습니다. 실행은 `python pytorch-problem-starter.py --case binary`입니다.

### 변경 지점

원래 라벨이 2/5이고 고장=2라면 `POS_LABEL=2`, `NEG_LABEL=5`, `y_model=(y==2).astype("float32")`입니다. 먼저 훈련·검증자료에 허용된 라벨만 있고 훈련자료에 두 클래스가 모두 있는지 확인합니다. AUC는 검증자료에도 두 클래스가 모두 있어야 정의됩니다. 훈련자료에서만 `pos_weight=n_negative/n_positive`를 계산하고 콜백에서 시그모이드를 씁니다.

### 확률과 라벨 변환을 한곳에서 처리

아래는 학습된 `model`, 순서가 고정된 `test_loader`에 연결합니다. 이때 모델은 위의 명시적인 0/1 대응표으로 학습했어야 합니다.

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

AUC는 확률의 순위를 평가하므로 계산 전에 확정 라벨로 바꾸지 않습니다. AUC로 평가한다는 이유만으로 제출값도 확률이라고 단정하지 않습니다. Macro-F1이면 검증자료에서 임계값을 비교하고 테스트에는 선택한 값을 고정해 적용합니다. 임계값까지 반복 조정한 검증 점수는 낙관적일 수 있으므로 탐색 횟수를 제한하세요.

가중 BCE로 학습한 모델의 시그모이드 출력이 잘 보정된 확률이라는 보장은 없습니다. 확률의 정확성이 중요한 지표라면 가중치를 쓴 모델과 쓰지 않은 모델을 검증자료에서 비교합니다. 불균형 문제에서 정확도만으로 판단하거나, 테스트 라벨을 추측해 임계값을 조정하지 마세요.

연습: 양성을 0으로 뒤집어 라벨 대응표·확률의 의미·원래 라벨 복원이 모두 뒤집히는지 확인하세요. 확률 제출에서 `np.where(prob == 1, POS_LABEL, NEG_LABEL)`을 실행하면 거의 모든 샘플이 음성으로 잘못 바뀝니다.

## 08. Problem · 다중분류와 다중라벨 분류 구분

### 먼저 질문 하나

한 샘플의 정답이 정확히 하나인가요? 그렇다면 다중분류(multiclass)입니다. 여러 고장 종류가 동시에 참일 수 있으면 다중라벨 분류(multilabel)입니다. 연속값 열이 여러 개인 경우는 둘 다 아니라 다중회귀입니다.

| 구분 | 정답 y | 변환 전 출력 | 학습 손실 | 제출 확률 |
|---|---|---|---|---|
| 다중분류 | `[B]` long 정수형, 0…C−1 | `[B,C]` | CrossEntropyLoss | 소프트맥스, 각 행의 합≈1 |
| 다중라벨 분류 | `[B,K]` 실수형 0/1 | `[B,K]` | BCEWithLogitsLoss | 열별 시그모이드, 행합 제한 없음 |

실행은 `--case multiclass` 또는 `--case multilabel`입니다. 모델은 둘 다 MLP이지만 과제 유형·정답 형태·출력 차원·평가지표이 다릅니다.

### 클래스 대응표와 확률 열

다중분류 라벨 인코더는 훈련 폴드로 학습하고 검증자료에 처음 보는 클래스가 있으면 분할을 다시 검토합니다. 시험에서 전체 클래스 목록을 지정했다면 그 목록을 우선합니다. 라벨을 제출할 때는 argmax 인덱스를 원래 라벨로 복원합니다. 확률을 제출할 때는 라벨 복원 함수를 적용하지 말고 **열 순서**를 맞춥니다.

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

가중 CE의 평균 손실은 배치 크기가 아니라 정답 클래스 가중치의 합으로 나눕니다. 다운로드한 공통 학습 반복문은 이 분모를 반영해 학습 회차별 손실을 집계합니다. 모델 학습 자체는 배치 크기의 영향을 받지만, **고정된 로짓의 전체 손실을 집계한 값**은 배치를 나누는 방법과 무관하게 같아야 합니다.

다중라벨 분류는 각 열의 값이 0/1인지 검사합니다. AUC는 라벨별로 양성·음성이 없는 경우의 처리와 평균 방식을 확인합니다. 정확도는 모든 라벨이 맞아야 정답인 부분집합 정확도인지 열별 평균인지 확인하세요. 각 정답 열이 세 가지 클래스를 가지는 범주형 다중출력(categorical multi-output)은 이 단순한 다중라벨 구현으로 처리할 수 없습니다. 출력층·손실·평가지표를 각각 명세에 맞춰 설계해야 합니다.

연습: 확률 제출을 라벨 제출로 바꾸고 예상 배열 크기가 `(N,C)`에서 `(N,)`으로 변하는 것을 설명해 보세요. 다중라벨 분류에는 이 변환을 적용하지 않습니다.

## 09. Problem · 이미 만들어진 시계열 윈도

### 첫 코드와 계약

처음 받은 X가 `[N,L,F]`이면 샘플 윈도 N개가 이미 있습니다. **윈도를 다시 만들지 않습니다.** `CNN1D(n_features=F,out_dim=K)`가 첫 후보이며 내부에서 `[B,L,F]→[B,F,L]`로 바꿉니다. 외부에서 축을 한 번 더 교환하지 마세요.

실행은 `--case window`입니다. 합성 예제는 서로 독립인 윈도를 만든 것이므로, 실제 겹치는 시계열에 예제의 분할 코드을 복사하지 않습니다. 원래 기록 시각·그룹·윈도 시작과 끝 위치로 검증을 설계합니다.

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

기존 L은 문제에서 정한 샘플 구성 조건일 수 있으므로 임의로 줄이지 않습니다. 메모리를 줄일 때는 배치 크기·채널 수부터 줄입니다. 스케일러는 훈련자료의 N·L 축을 펼쳐 특성별로 학습한 뒤, 같은 특성 수 F를 가진 검증·테스트 자료를 변환합니다.

### 모델 하나만 비교하기

RNN을 비교하려면 `SequenceRNN(n_features=F,out_dim=K,hidden_size=64,num_layers=1,kind="gru")`로 모델 생성만 교체합니다. LSTM은 `kind="lstm"`. DataLoader와 손실·평가지표·제출 조건은 그대로 유지합니다. `MLP`는 윈도를 펼치는 대안이지만 입력 특성 차원이 `L×F`가 됩니다.

검사: 원래 샘플마다 예측 한 행인지, 테스트 샘플 수 N을 유지하는지, 마지막 배치 B=1에서도 동작하는지, 정답의 원래 단위로 지표를 계산하는지 확인합니다. 같은 원시 시점이 겹치는 샘플을 무작위로 나누면 누수가 생길 수 있습니다. 시간순으로 정렬하는 것만으로 중첩 관측이나 정답 확정 지연 문제가 모두 해결되지는 않습니다. 다음 원시 시계열 가이드의 예측 기준 시점 조건을 확인하세요.

## 10. Problem · 원시 시계열에서 미래 한 시점 예측

### 인덱스부터 정의

입력이 `[T,F]`이고 “과거 L개로 마지막 관측 H칸 뒤를 예측”한다면 `make_point_forecast_windows`를 씁니다. 마지막 관측 e의 입력은 `X[e−L+1:e+1]`, 정답은 `y[e+H]`입니다. H=1은 다음 행입니다. 초와 샘플 수를 혼동하지 말고 샘플링 주기와 실제 시간 간격을 확인하세요.

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

### 예측 간격이 길 때의 누수

target<70을 훈련, target≥70을 검증으로만 나누면 정답 시점 70을 예측하는 기준 시점은 e=60입니다. 그런데 훈련에서는 y[69]까지 이미 사용했으므로 e=60에 알 수 없는 미래 정답을 미리 학습한 셈입니다. 기본 함수는 훈련 정답이 분할 시점 전에 알려지고, 검증 입력의 끝이 분할 시점 이후인 샘플만 선택합니다.

`label_delay`는 정답 확정 지연의 **행 수**, `gap`도 원시 자료의 행 단위입니다. 실제 기록 시각을 쓰거나 간격이 불규칙하면 시간으로 구현합니다. `split_mode="target"`은 별도의 배치평가 명세에 명시적으로 맞출 때만 쓰며 실시간 예측의 기본값이 아닙니다.

### 그룹·테스트·메모리에 따른 처리

차량별 원시 자료는 `make_grouped_point_forecast_windows`로 그룹을 넘나들지 않게 처리합니다. 반환되는 정답 행은 원본 데이터의 행 위치입니다. 순서가 섞인 전체 행 번호에서 H를 빼면 같은 그룹의 예측 기준 시점을 찾을 수 없습니다. 그룹 안의 정렬된 위치에서 H를 뺀 뒤 실제 기록 시각으로 바꿔야 합니다. 모든 그룹에 공통 모델을 사용한다면 모든 훈련 정답이 첫 검증 예측 기준 시점 전에 알려져야 합니다. 새 차량의 평가까지 요구되면 그룹도 분리합니다.

테스트에는 정답이 없으므로 윈도 개수를 자동으로 추정하면 제출 행과 어긋날 수 있습니다. 주어진 starts·예측 대상 행과 expected_n을 읽고 `LazyWindowDataset`의 테스트 입력 조건을 적용합니다. 규칙적으로 관측한 단일 원시 시계열의 윈도 수는 `max(0,floor((T−L−H)/stride)+1)`입니다. 여러 미래 시점의 전체 궤적을 예측하는 문제는 이 단일 정답 시점 함수와 입출력 조건이 다릅니다.

연습: H=1→10, stride=1→3, label_delay=0→4로 바꾸고 모든 훈련 정답의 확정 시점이 첫 검증 예측 기준 시점보다 빠른지 검사하세요. 전체 실행 흐름은 [시계열 모의 코드](./full-mock-timeseries.py)에 있습니다.

## 11. Problem · 이미지 분류·회귀

### 첫 코드와 계약

배열이면 `prepare_numpy_images` → `SmallImageCNN(in_channels=C,out_dim=D)` → 공통 학습입니다. 실행은 `--case image`. 파일 경로 목록이면 긴 치트시트 13절의 `ImagePathDataset`을 사용하는 **별도 예제 계열**을 선택하거나 [이미지 통합 모의](./full-mock-image-ae.py)를 사용하세요. 경로 문자열을 `make_tensor_loader`에 넣지 않습니다.

### 바꿀 부분

layout=`NHWC/NCHW`, divide_255, resize 높이·너비, in_channels, task, out_dim, 라벨 대응표을 정합니다. 현재 작은 CNN은 풀링을 두 번 적용하므로 최소 H/W≥4가 필요합니다. 분류는 07/08절의 손실, 회귀는 06절의 원래 단위 평가지표·출력 조건을 적용합니다.

### 학습·평가 전처리를 구분

훈련자료에만 정답을 보존하는 증강을 적용하고, 검증·테스트에는 매번 같은 결과를 내는 고정 변환을 적용합니다. 좌우 방향이 정답인 문제에서의 좌우 반전, 좌표 회귀에서 정답 좌표를 함께 바꾸지 않는 자르기·크기 조절은 입력과 정답의 대응을 깨뜨립니다. 이미지 경로·라벨·ID는 같은 행에 묶어 처리해야 합니다.

사전학습 가중치는 제공·캐시·사용 허용을 확인한 뒤 사용합니다. 인터넷 다운로드가 가능하다고 가정하지 않습니다. 무작위 가중치 상태의 특징 추출부를 고정하면 특징 추출기는 학습되지 않아 사전학습 전이 효과를 기대할 수 없습니다. 출력층은 학습할 수 있지만 좋은 특징이 보장되는 것은 아닙니다. 먼저 작은 CNN으로 학습부터 저장까지 실행해 보세요.

### 연습과 통과 기준

합성 이미지의 RGB를 흑백으로 바꾸고 C=1, 배치 크기, 첫 Conv를 함께 바꿉니다. H≠W·B=1에서 출력은 `[B,D]`를 유지해야 합니다. float 0~1 입력의 `divide_255=False`도 검사하세요. 회귀라면 마지막에 softmax 없이 연속값을 반환하고, 표준화한 y는 지표 계산·제출 양쪽에서 원 단위로 복원합니다.

## 12. Problem · 정상 자료로 학습하는 오토인코더 이상탐지

### 첫 코드와 계약

정상 데이터만으로 학습해서 샘플별 이상 정도를 구하는 문제입니다. 표형은 `Autoencoder(input_dim=F,latent_dim=...,hidden_dim=...)`, 입력 X 자체를 정답으로 주는 회귀 학습입니다. 실행은 `--case anomaly`. 이미지에는 [full-mock-image-ae.py](./full-mock-image-ae.py)의 `ConvAutoencoder`와 `reconstruction_errors`를 선택합니다. 표형 AE에 4D 배열을 그대로 넣지 않습니다.

### 바꿀 부분

정상 훈련 샘플을 고를 마스크, 정상 검증자료, 훈련자료로만 학습할 스케일러, 잠재 차원(latent_dim), 샘플별 복원 오차를 평균할 축, 점수의 방향, 제출할 값의 종류를 정합니다. 정상·이상 라벨이 충분하면 지도학습 이진분류도 비교할 수 있습니다.

### 점수와 라벨을 정확히 연결

표형 재구성 오차는 `mean((X−recon)^2, axis=1)`로 샘플당 하나입니다. 이미지라면 `(1,2,3)`을 평균해야 `[N]`이 됩니다. 배치 전체 손실값 하나나 `[N,H,W]`를 제출하지 마세요.

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

정상 검증 점수의 99% 분위수는 연습에서 시작할 후보이지 최적 기준은 아닙니다. 라벨이 있으면 공식 F1 등으로 임계값을 비교한 뒤 고정합니다. 이상 라벨이 없으면 “정상 검증 샘플 중 약 1%를 이상으로 판정하는 기준”일 뿐, 미래 오탐률이 1%라고 보장하지는 않습니다. 정상=1이라면 라벨을 뒤집되 복원 오차 점수의 방향은 명세를 따릅니다.

기존 코드의 `pred_anomaly`를 계산하고 다른 `test_pred`를 저장하는 실수를 막기 위해 위처럼 **공통 제출 변수에 직접 할당**합니다. 연습에서는 점수·라벨을 번갈아 선택하고 배열 크기·값 범위·저장 변수가 모두 맞는지 확인하세요.

## 13. 공통 마무리 · 체크포인트·제출·오류 복구

### checkpoint를 고른 뒤

`train_torch_model`은 선택 기준이 가장 좋았던 시점의 가중치를 복원하고 `(model, history)`를 반환합니다. score_fn은 전체 검증 정답과 **변환 전 모델 출력** NumPy 배열을 받습니다. 분류라면 시그모이드·소프트맥스·argmax, 회귀값을 변환해 학습했다면 원래 단위로의 역변환을 콜백 안에서 수행해야 합니다. 합성 시작 예제의 `official_score`를 기준으로 고칩니다.

훈련자료 전체를 사용한 최종 재학습은 선택 사항입니다. 재학습한다면 검증 성능이 가장 좋았던 학습 반복 횟수를 고정하고, 새 모델·새 최적화 알고리즘·전체 훈련자료로 새로 학습한 전처리를 사용합니다. 따로 남겨 둔 검증자료까지 학습에 합친 상태에서 같은 자료의 점수로 독립적인 조기 종료를 했다고 보아서는 안 됩니다. 시간이 부족하면 이미 검증한 체크포인트를 그대로 사용하세요.

### 저장 전에 문제에서 확인한 조건과 대조

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

문자열 라벨을 CSV로 제출하라는 조건이라면 수치 검사 대신 라벨 집합·열 이름·ID 순서를 검사합니다. 무조건 float32로 바꾸면 원래 라벨이 손상될 수 있습니다. 위 예제의 파일명은 공식 제출명이 아닙니다.

### 오류를 좁히는 순서

| 증상 | 첫 검사 | 조치 |
|---|---|---|
| NameError/import 실패 | 파일 이름·앞 정의·같은 폴더 | `hdat_templates.py` 및 00번 의존 블록 확인 |
| mat1/mat2 오류 | 전처리 후 특성 수 F와 Linear의 in_features | 전처리 후 특성 수로 모델 생성 |
| Conv 채널 오류 | `[B,L,F]`/`[B,F,L]`, NHWC/NCHW | 소스 내부의 축 교환과 중복하지 않기 |
| BCE/MSE 배열 크기 문제 | 출력과 정답 y의 정확한 크기 튜플; MSE는 오류 없이 틀릴 수도 있음 | 한 출력 `[B,1]`, 다중출력 `[B,K]`처럼 정확히 일치; B=1도 검사 |
| CE 오류 | 정답 y의 long 정수형 여부, 범위 0…C−1 | 라벨 대응표·범위 확인, 마지막 소프트맥스 제거 |
| NaN/Inf | 입력 X·정답 y·복원값, 손실, 학습률 | 입력의 유한성부터 검사하고 학습률 낮추기 |
| CUDA OOM | 배치·모델·윈도 복사 | 배치 크기를 절반으로, 채널 수 축소, 지연 로딩 윈도 |
| AUC 계산 불가 | 검증자료의 클래스 수·확률 배열 크기 | 분할 재검토, 임의로 0점을 부여해 숨기지 않기 |
| 점수만 비정상적으로 좋음 | 시간·그룹 분할, 예측 뒤에야 알 수 있는 특성 | 모델 조정 전에 누수부터 제거 |
| 예측 N 불일치 | 테스트 순서 섞기·정렬·윈도와 정답의 대응 | 샘플 ID 기준으로 원래 순서 복원 |

### 마지막 30초

함수명·변수명·파일명, 테스트 행 순서, 출력 종류·배열 크기·원소 자료형, NaN/Inf를 먼저 검사합니다. **Ctrl+S → Autosaved 확인 → 기존 저장 셀 실행·예측 파일 확인 → 다시 저장 → Process/Problem 각각 제출 → 정상 제출 팝업·로그 확인**까지 마쳐야 합니다. 파일 생성과 노트북 저장만으로 제출이 끝난 것은 아닙니다. Autosave Failed·연결 오류는 감독관에게 알리고, 테스트 종료 후에는 다시 수정할 수 없습니다. [공식 저장 안내](https://hdat.gitbook.io/2026-hdat-ds/undefined/undefined/6.-2/1/undefined-1.md)

2026 안내에서는 170분 안에 문항 간 이동이 가능합니다. 이동 전에 저장하고 이전 런박스 연결 해제 → 새 문항 연결 절차를 따르세요. **세팅 중에는 이동·새로고침하지 않습니다.** 왼쪽 런박스 시간과 오른쪽 상단 시험 잔여시간을 구분하세요. One-way는 자료 참고 방식이지 문항 이동 금지라는 뜻이 아닙니다. [문항 이동 안내](https://hdat.gitbook.io/2026-hdat-ds/undefined/undefined/6.-2/3.md)

로컬 `import hdat_templates as h`를 시험 제출 방식으로 오해하지 마세요. 문제에서 허용한 작성영역에 필요한 import·정의·호출을 배치하고 기존 저장 셀이 참조하는 변수로 연결합니다. 추가 `.py` 파일이 자동 제출된다고 가정하지 않습니다. 패키지 설치 허용과 작성영역·제출 조건은 별개입니다. 필기는 자료·검색 불가이고 실기는 허용된 단방향 참고만 가능합니다. 생성형 AI·검색 AI 답변·GitHub·Colab·Kaggle·Notion·메일·메신저는 사용하지 말고 문제 데이터는 제공 IDE 밖에서 처리하지 마세요. [공식 FAQ](https://hdat.gitbook.io/2026-hdat-ds/faq/3..md)

## 14. 부록 · ResNet·Transformer·VAE·GAN의 지원 범위

이 구조들은 공개 학습 범위에 포함되지만 **이름이 범위에 있다고 모든 것이 특정 실기 유형으로 출제된다는 뜻은 아닙니다.** 구조가 직접 지정되면 정확히 구현하고, 자유 모델 선택 문제에서는 작은 기준 모델을 만든 뒤 검증 결과상 근거가 있을 때 비교하세요.

| 명세 신호 | 다운로드 소스 | 반드시 추가로 확인 |
|---|---|---|
| 잔차·지름길 연결 | `ResidualBlock2D(in_channels,out_channels,stride)` | 지름길·주 분기의 배열 크기, 투영, 활성함수 위치 |
| 시퀀스 어텐션 | `TransformerSequenceModel(n_features,out_dim,d_model,nhead,...)` | d_model % nhead=0, 위치 정보·패딩·인과 마스크 |
| 잠재 평균·분산·KL | `VariationalAutoencoder` + `vae_loss` | `(recon,mu,logvar)` 반환값 분리, KL 집계 방식·beta |
| 생성기·판별기 교대 | `MLPGenerator`, `MLPDiscriminator` | 두 최적화 알고리즘, D용 fake.detach(), G로의 기울기 전달 |

### 공통 루프를 그대로 쓰면 안 되는 경우

VAE는 모델이 튜플을 반환하므로 단일 출력을 전제로 한 MSE 공통 학습 반복문가 맞지 않습니다. GAN은 두 네트워크를 교대로 업데이트하고 하나의 지도학습 정답 y를 사용하는 구조가 아니라 전용 루프가 필요합니다. [긴 치트시트 14A 부록](../cheatsheet.html)에서 구조를 찾고 [23강](../learn/23/)·[24강](../learn/24/)·[25강](../learn/25/)에서 손실과 학습 원리를 대조하세요.

현재 Transformer는 고정 길이 윈도에서 예측 하나를 만드는 인코더+풀링 구조입니다. padding_mask를 받는 모델 함수가 있어도 기본 DataLoader/공통 루프는 `(X,y)`만 전달하므로 가변길이는 collate와 학습 루프도 바꿔야 합니다. 모든 토큰이 패딩인 샘플은 허용하지 않아야 합니다. 자기회귀 다음 토큰 예측, 분할(segmentation), 여러 미래 시점의 전체 궤적 예측, 범주형 다중출력은 출력·손실·마스크를 별도로 설계해야 합니다.

### 예시 입력으로 구조 확인하기

```python
import torch
import hdat_templates as h

block = h.ResidualBlock2D(3, 8, stride=2).eval()
assert block(torch.zeros(1, 3, 12, 16)).shape == (1, 8, 6, 8)
model = h.TransformerSequenceModel(3, 2, d_model=8, nhead=2, num_layers=1).eval()
assert model(torch.zeros(1, 5, 3)).shape == (1, 2)
```

이 검사는 요구한 배열 크기로 실행되는지만 확인하며, 학습이 올바르거나 성능이 좋다는 보장은 아닙니다. 실제 지정 구조와 매개변수 수·손실·기울기를 따로 확인하세요.

공식 API 대조: [PyTorch CrossEntropyLoss](https://docs.pytorch.org/docs/2.7/generated/torch.nn.CrossEntropyLoss.html), [BCEWithLogitsLoss](https://docs.pytorch.org/docs/2.7/generated/torch.nn.BCEWithLogitsLoss.html), [TransformerEncoder](https://docs.pytorch.org/docs/2.7/generated/torch.nn.TransformerEncoder.html).
