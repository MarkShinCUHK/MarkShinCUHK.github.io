# hdat_templates, 괄호 안에 무엇을 넣나요?

이 매뉴얼은 함수를 외우는 책이 아닙니다. 내 데이터에서 어떤 값을 꺼내 어느 자리에 넣는지 배우는 책입니다. 먼저 01~02를 순서대로 실행한 뒤, 문제에 맞는 03~06으로 가세요. 07~09는 함수 이름을 알고 있을 때 찾아보는 사전입니다. 모든 예제는 비공식 합성 학습자료이며, 시험의 지정 함수·반환 형식·파일 사용 규정이 우선합니다.

## 01. 함수의 괄호를 읽는 법부터

`n_features: int`는 “정수를 넣으세요”라는 설명일 뿐입니다. 정작 필요한 설명은 “전처리를 마친 입력 배열의 열 수를 넣으세요”입니다. 이 차이를 출발점으로 삼겠습니다.

### 파일을 준비하고, 불러온 파일부터 확인하기

[hdat_templates.py](./hdat_templates.py)를 내려받아 연습용 노트북이나 Python 파일과 같은 폴더에 놓으세요. 밑줄이 있는 파일을 사용합니다. 내 연습 파일의 이름은 `hdat_templates.py`, `torch.py`, `numpy.py`, `pandas.py`로 짓지 마세요. 라이브러리와 이름이 겹칠 수 있습니다. 이 파일을 불러오려면 NumPy·pandas가 필요하고, 모델 예제에는 PyTorch·scikit-learn도 필요합니다. PIL 이미지를 쓰는 예제에는 Pillow가 필요합니다.

<!-- run: basics -->
```python
import numpy as np
import hdat_templates as h

print(h.__file__)  # 지금 불러온 파일의 실제 위치
print(hasattr(h, "MLP"))  # PyTorch가 설치되어 있다면 True
```

`h`는 긴 이름을 짧게 부르는 별명입니다. `h.MLP`는 “hdat_templates 안에 있는 MLP”라는 뜻입니다. 다운로드한 파일이 따로 있고 예전에 불러온 파일이 다른 곳에 있으면 수정 내용이 보이지 않습니다. 노트북에서 파일을 바꿨다면 커널을 재시작하고 위 셀부터 다시 실행하세요. `ModuleNotFoundError`가 나면 아직 모델 인자 문제가 아니라 파일 위치 또는 실행 환경 문제입니다.

웹페이지의 코드 자체가 이 화면에서 실행되지는 않습니다. 노트북에서는 Python 코드를 셀에 넣고 위에서 아래로 실행하세요. 실행 파일을 받았다면 그 파일들이 있는 폴더의 터미널에서 `python template-call-regression.py`처럼 실행합니다. `python ...py`는 터미널 명령이고, `import ...`부터 시작하는 코드는 Python 코드입니다. 두 가지를 같은 셀에 그대로 섞지 마세요. 내려받은 파일은 이미 해당 장의 준비 셀을 모두 포함합니다.

### 왼쪽은 자리 이름, 오른쪽은 내가 넣는 값

<!-- run: basics -->
```python
X_train = np.array([[10.0, 1.0], [20.0, 2.0]], dtype=np.float32)
feature_count = X_train.shape[1]
model = h.MLP(n_features=feature_count, out_dim=1, hidden=(8, 4))
print(X_train.shape, feature_count)  # (2, 2) 2
```

`n_features=feature_count`를 “n_features라는 자리에 feature_count에 들어 있는 2를 전달한다”라고 읽으세요. **왼쪽 `n_features`는 함수가 정한 이름이라 바꾸면 안 됩니다. 오른쪽 `feature_count`는 내가 만든 변수 이름이라 바꿀 수 있습니다.** 오른쪽에 `X_train.shape[1]`을 직접 적어도 같습니다. 입력 열이 2개라서 2를 넣은 것이지, 모든 MLP에 2를 넣는 규칙은 아닙니다.

| 호출에 적힌 것 | 실제로 전달되는 것 | 어디서 결정하나요? |
|---|---|---|
| `n_features=X_train.shape[1]` | 전처리 후 입력 열의 개수 | 입력 배열 확인 |
| `out_dim=1` | 표본 하나당 출력 숫자 1개 | 이 예제의 단일 회귀 목표 |
| `hidden=(8, 4)` | 첫 은닉층 8개, 다음 은닉층 4개 | 연습자가 선택한 작은 구조 |
| `dropout=0.1`을 생략 | 기본값 0.1이 적용됨 | 함수 정의의 기본값 |
| `model=model` | 앞에서 만든 모델 객체 | `h.MLP(...)`의 반환값 |

숫자 인자도 출처가 다릅니다. `n_features`는 데이터에 맞춰야 하고, `out_dim`은 정답 구조에 맞춰야 합니다. `hidden`, `lr`, `epochs`는 문제에서 지정하지 않았다면 검증 결과와 시간에 맞춰 고르는 설정입니다. “예제에 쓰인 숫자”와 “반드시 지켜야 할 숫자”를 구별하세요.

### 따옴표, 괄호, 리스트가 서로 다른 이유

| 표현 | 의미 | 자주 하는 실수 |
|---|---|---|
| `task="regression"` | 회귀라는 선택지를 나타내는 문자열 | `task=y_train`처럼 정답 배열을 넣음 |
| `X=X_train` | 변수에 담긴 입력 데이터 | `X="X_train"`처럼 변수 이름만 문자열로 전달 |
| `target_cols=["price"]` | 열 이름을 담은 리스트 | `target_cols=train["price"]`처럼 열의 값을 전달 |
| `target_cols=("price",)` | 원소가 하나인 튜플 | `("price")`는 튜플이 아니라 문자열 |
| `loss_fn=torch.nn.L1Loss()` | 실제로 만든 손실함수 객체 | `"L1Loss"`라는 문자열을 전달 |
| `score_fn=my_rmse` | 나중에 호출할 함수 자체 | `my_rmse()`로 지금 실행해 숫자를 전달 |
| `y=None` | 정답을 주지 않음. 예측용 loader에서 사용 | `y="None"`이라는 글자를 전달 |

`h.MLP`는 모델을 만드는 클래스이고, `h.MLP(...)`는 그 클래스로 모델을 실제로 만드는 호출입니다. 만든 결과를 `model`이라는 변수에 받습니다. 학습 함수에는 `model=model`로 그 결과를 전달합니다. 이 템플릿의 `train_torch_model`에는 `model_factory`, `config`, `metric`이라는 인자가 없습니다. 다른 자료의 같은 역할을 하는 코드와 인자 이름을 섞지 마세요.

### 반환값이 다음 호출의 재료가 된다

다음 흐름에서 화살표는 “왼쪽 호출의 결과를 오른쪽 호출에 넣는다”는 뜻입니다.

<ol class="template-flow"><li><strong>입력과 정답</strong><span>X_train, y_train을 만든다</span></li><li><strong>묶음 공급기</strong><span>make_tensor_loader → train_loader</span></li><li><strong>모델</strong><span>MLP → model</span></li><li><strong>학습</strong><span>train_torch_model → model, history</span></li><li><strong>예측</strong><span>predict_torch → pred</span></li></ol>

`model, history = ...`는 결과가 두 개라서 두 변수에 나누어 받는 표현입니다. 학습 함수의 첫 결과는 검증 기준이 가장 좋았던 가중치를 복원한 모델, 둘째는 에포크별 기록을 담은 DataFrame입니다. `history`를 예측 함수의 `model` 자리에 넣으면 안 됩니다. 다음 장에서 이 흐름의 모든 변수를 직접 만듭니다.

<details><summary>확인문제: 입력 배열이 (80, 7)이면 n_features에 80과 7 중 무엇을 넣나요?</summary><p>7입니다. 80은 표본 수, 7은 한 표본의 입력 열 수입니다. 미니배치가 16개씩 들어와도 각 표본이 가진 입력 7개는 변하지 않습니다. 다만 범주형 전처리로 열 수가 달라졌다면 원본 표가 아닌 전처리 후 배열의 shape을 확인해야 합니다.</p></details>

기본 문법을 더 확인하고 싶다면 [Python 공식 문서의 키워드 인자 설명](https://docs.python.org/3/tutorial/controlflow.html#keyword-arguments)을 참고하세요. 여기의 `h.*` 함수는 Python이나 PyTorch에 원래 있는 함수가 아니라 이 사이트가 제공하는 보조 코드입니다.

## 02. 표 데이터 하나로 처음부터 끝까지 호출하기

중고 물품의 가격을 예측한다고 합시다. 사용 기간 `age`와 주행량 `distance`가 입력이고, 가격 `price`가 정답입니다. 아래 셀은 **위에서 아래로 이어서 실행**합니다. 다른 장의 변수를 미리 만들 필요는 없습니다. 데이터 12행과 3에포크는 호출을 이해하기 위한 크기일 뿐, 성능 평가에 충분한 설정은 아닙니다.

### 1단계: 파일명 대신 실제 표를 준비한다

<!-- run: regression -->
```python
import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import hdat_templates as h

h.seed_everything(seed=42)
torch.set_num_threads(1)
train = pd.DataFrame({
    "age": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "distance": [12, 25, 31, 48, 55, 63, 74, 82, 93, 108, 117, 130],
    "price": [28, 27, 25, 24, 22, 21, 19, 18, 16, 14, 13, 11],
})
test = pd.DataFrame({"age": [3, 8], "distance": [35, 86]})
feature_cols = ["age", "distance"]
target_cols = ["price"]
h.assert_frame_contract(train=train, test=test, target_cols=target_cols)
X = train[feature_cols]
y = train[target_cols].to_numpy(dtype=np.float32)
print(X.shape, y.shape, test.shape)  # (12, 2) (12, 1) (2, 2)
```

실제 연습 파일이 있다면 `train = pd.read_csv("train.csv")`, `test = pd.read_csv("test.csv")`로 표를 읽는 부분을 바꿉니다. 파일 이름은 예시이며 문제에서 준 경로를 사용하세요. `train[feature_cols]`에는 모델이 보고 판단할 정보만 들어갑니다. 정답 `price`를 입력에 넣으면 정답을 미리 보여 주는 셈입니다. ID가 있어도 예측에 쓰라는 지시나 타당한 이유가 없다면 입력 열 목록에서 제외합니다.

`train["price"]`는 보통 `(12,)`, `train[["price"]]`는 `(12, 1)`입니다. 여기서는 출력 숫자가 하나라는 축을 남기기 위해 두 번째 방식을 썼습니다. `target_cols`가 이미 리스트이므로 `train[target_cols]`가 그 역할을 합니다.

### 2단계: 먼저 나누고, 학습 부분에서만 전처리 기준을 구한다

<!-- run: regression -->
```python
train_idx, valid_idx = h.safe_train_valid_indices(
    X=X, y=y, task="regression", valid_size=0.25, seed=42,
)
prep = h.make_preprocessor(
    X=X.iloc[train_idx], encoding="ordinal", scale_numeric=True,
)
X_train = prep.fit_transform(X.iloc[train_idx]).astype(np.float32)
X_valid = prep.transform(X.iloc[valid_idx]).astype(np.float32)
X_test = prep.transform(test[feature_cols]).astype(np.float32)
y_train = y[train_idx]
y_valid = y[valid_idx]
h.print_shapes(X_train=X_train, y_train=y_train, X_valid=X_valid, X_test=X_test)
assert X_train.shape == (9, 2)
assert y_train.shape == (9, 1)
```

`X=`에는 표, `y=`에는 그 표와 같은 순서의 정답을 넣습니다. 이 예제의 각 행은 독립 표본이라고 가정하므로 무작위 분할을 사용합니다. 시간순 예측이나 같은 차량의 반복 측정이라면 이 가정이 성립하지 않을 수 있습니다. 04장과 07장의 시간·그룹 분할 설명을 먼저 보세요.

`train_idx`, `valid_idx`는 행의 **위치 번호**입니다. pandas의 행 이름이 `[100, 200, ...]`이어도 이 번호는 0부터 셉니다. 따라서 `.loc`가 아니라 `.iloc`로 고릅니다. 전처리기 `prep`는 만들었을 때 아직 평균이나 중앙값을 모릅니다. `fit_transform`으로 학습 부분에서 그 기준을 구하고, 검증·테스트에는 같은 `prep.transform`만 적용합니다.

이 예제는 수치 열뿐입니다. 문자열 범주가 섞인 데이터라면 먼저 `clean_tabular_values` 등으로 자료형을 확인하세요. `encoding="ordinal"`은 범주를 숫자로 바꾸는 작은 연습의 출발점이지 항상 최선이라는 뜻이 아닙니다. 기본값인 `"onehot"`은 결과가 희소행렬일 수 있어 이 loader에 곧바로 넣을 수 없습니다. 작은 데이터에서만 메모리 크기를 확인한 뒤 `.toarray()`로 바꾸세요.

### 3단계: 배열을 loader로 바꾸고, 한 묶음을 열어 본다

<!-- run: regression -->
```python
train_loader = h.make_tensor_loader(
    X=X_train, y=y_train, task="regression", batch_size=4, shuffle=True,
)
valid_loader = h.make_tensor_loader(
    X=X_valid, y=y_valid, task="regression", batch_size=4, shuffle=False,
)
test_loader = h.make_tensor_loader(
    X=X_test, y=None, task="regression", batch_size=4, shuffle=False,
)
xb, yb = next(iter(train_loader))
print(xb.shape, yb.shape)  # torch.Size([4, 2]) torch.Size([4, 1])
print(xb.dtype, yb.dtype)  # torch.float32 torch.float32
```

loader는 새로운 데이터를 만드는 모델이 아닙니다. 준비한 데이터를 몇 개씩 꺼내 주는 도구입니다. `batch_size=4`이므로 학습 표본 9개는 4개·4개·1개로 나옵니다. **배치 크기는 입력 열 수가 아닙니다.** `shuffle=True`는 학습 표본의 순서를 섞되 각 입력과 정답의 짝은 유지합니다. 검증과 제출용 예측에서는 `False`로 두어 원래 행 순서를 유지하세요. 테스트에는 정답이 없으므로 `y=None`입니다.

PyTorch의 [DataLoader 공식 설명](https://docs.pytorch.org/docs/2.7/data.html)도 데이터 묶음과 순서를 구분합니다. 이 템플릿의 `make_tensor_loader`는 배열을 TensorDataset과 DataLoader로 감싸 주는 작은 보조 함수입니다.

### 4단계: 데이터에서 모델 크기를 구한 뒤 학습한다

<!-- run: regression -->
```python
model = h.MLP(
    n_features=X_train.shape[1],
    out_dim=y_train.shape[1],
    hidden=(16, 8),
    dropout=0.0,
)
model, history = h.train_torch_model(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    task="regression",
    epochs=3,
    lr=0.01,
    patience=2,
    device="cpu",
)
print(history[["epoch", "train_loss", "valid_loss", "monitor"]])
```

여기서 `n_features`는 2, `out_dim`은 1입니다. `model=model`의 오른쪽은 바로 앞에서 만든 MLP입니다. `train_loader=train_loader`의 오른쪽은 3단계에서 만든 학습용 loader입니다. 왼쪽과 오른쪽 이름이 우연히 같은 것이지, 빈칸에 그대로 적으면 자동으로 생기는 변수가 아닙니다.

`epochs=3`은 학습 데이터 전체를 최대 세 번 보겠다는 뜻입니다. `lr=0.01`은 한 번 업데이트할 때 움직이는 정도를 조절하는 학습률입니다. `patience=2`는 검증 기준이 두 에포크 연속 좋아지지 않으면 멈추라는 뜻입니다. 작은 예제용 선택일 뿐 실제 문제의 권장 정답은 아닙니다. `device="cpu"`는 CPU에서 실행하라는 뜻이며, 생략하면 이 helper는 CUDA가 있으면 CUDA, 아니면 CPU를 고릅니다. Apple MPS를 자동 선택하지는 않습니다.

`score_fn`을 생략했으므로 `monitor`는 검증 MSE입니다. “평가 지표가 RMSE인데 `metric="rmse"`를 추가하면 되나요?”라고 묻기 쉽지만 그 인자는 없습니다. 06장에서 `score_fn`을 직접 전달하는 방법을 배웁니다. 현재 예제의 MSE와 RMSE는 같은 정답 단위에서 계산하므로 에포크 순위는 같지만, 모든 손실과 지표가 그렇게 일치하는 것은 아닙니다.

### 5단계: 예측 모양을 확인하고, 제출 규격에 맞춰 저장한다

<!-- run: regression -->
```python
pred = h.predict_torch(
    model=model, loader=test_loader, task="regression", device="cpu",
)
print(pred.shape)  # (2, 1): 테스트 2행 × 가격 1개
h.validate_prediction_array(pred=pred, expected_shape=(len(test), 1))

# 연습 파일은 임시 폴더에 저장했다가 자동 정리한다.
with tempfile.TemporaryDirectory(prefix="hdat-call-") as folder:
    path = Path(folder) / "practice-pred.npy"
    saved = h.save_npy_submission(
        pred=pred, path=path, expected_shape=(len(test), 1), dtype=np.float32,
    )
    assert np.load(path, allow_pickle=False).shape == (2, 1)
```

예측 함수에는 학습 함수와 달리 `train_loader=`가 아니라 `loader=`라는 자리가 있습니다. 여기에 정답 없는 테스트용 loader를 넣습니다. 반환값 `pred`는 NumPy 배열이며 가격처럼 예측한 숫자가 담겨 있습니다. 학습한 모델 객체와 예측 결과 배열은 서로 다른 물건입니다.

이 연습은 제출 규격을 `(테스트 행 수, 1)`로 정했습니다. 실제 문제가 `(테스트 행 수,)`를 요구하면 단일 출력인지 확인한 뒤 `pred[:, 0]`을 사용합니다. `expected_shape=pred.shape`라고 쓰면 틀린 모양도 그대로 합격시킬 수 있으므로 **문제에 적힌 규격에서 기대 모양을 정해야 합니다.** 실제 파일을 보관하려면 `path`에 내가 정한 연습 경로를 넣되, 기존 파일이 없는지 먼저 확인하세요. 저장 helper는 같은 경로의 파일을 덮어쓸 수 있습니다.

<details><summary>확인문제: 입력이 20열, 정답이 온도·압력 두 열이면 무엇을 바꾸나요?</summary><p>전처리 후 입력이 정말 20열인지 먼저 확인합니다. 그때 n_features는 20, y_train이 (N, 2)이면 out_dim은 2입니다. 두 출력의 열 순서를 target_cols와 제출 규격에 맞춥니다. batch_size를 20이나 2로 바꿔야 하는 것은 아닙니다. 두 정답의 단위 차이가 크면 손실 가중치나 목표 스케일링도 별도로 검토해야 합니다.</p></details>

## 03. 회귀를 분류로 바꿀 때 함께 바꿀 것

모델의 마지막 숫자만 바꾸면 분류 문제가 되지 않습니다. 정답 표현, 출력 개수, task, 점수 계산, 제출 형태를 함께 맞춰야 합니다. 아래 세 예제는 각각 독립 실행할 수 있습니다.

### 먼저 정답 하나의 뜻을 구분한다

| 문제 | 정답 예 | loader에 넣는 y | out_dim | 기본 손실 | 확률을 요청하면 |
|---|---|---|---|---|---|
| 이진분류 | 정상 또는 고장 | 0/1, `(N,)` 또는 `(N, 1)` | 1 | BCEWithLogitsLoss | `(N, 1)` 양성 확률 |
| 다중분류 | 정상·마모·파손 중 하나 | 0~K−1 정수, `(N,)` | K | CrossEntropyLoss | `(N, K)` 전체 확률 |
| 다중라벨 | 누유와 진동이 동시에 가능 | 0/1 표, `(N, L)` | L | BCEWithLogitsLoss | `(N, L)` 라벨별 확률 |

`N`은 표본 수, `K`는 서로 배타적인 종류 수, `L`은 동시에 켜질 수 있는 표시의 개수입니다. 예를 들어 정답이 100행이어도 종류가 3개면 다중분류의 `out_dim`은 100이 아니라 3입니다. 각 라벨이 동시에 참일 수 있는 다중라벨에서는 열별 확률 합이 1일 필요가 없습니다.

### 이진분류: “고장”을 1로 정하고 확률을 받는다

<!-- run: binary -->
```python
import numpy as np
import torch
import hdat_templates as h

h.seed_everything(42)
torch.set_num_threads(1)
X = np.arange(48, dtype=np.float32).reshape(24, 2) / 48
original_y = np.array(["정상", "고장"] * 12)
y = (original_y == "고장").astype(np.float32)  # 양성의 뜻을 먼저 결정
train_loader = h.make_tensor_loader(X=X[:16], y=y[:16], task="binary", batch_size=8, shuffle=True)
valid_loader = h.make_tensor_loader(X=X[16:20], y=y[16:20], task="binary", batch_size=8)
test_loader = h.make_tensor_loader(X=X[20:], task="binary", batch_size=8)
model = h.MLP(n_features=X.shape[1], out_dim=1, hidden=(8,), dropout=0)
model, history = h.train_torch_model(
    model=model, train_loader=train_loader, valid_loader=valid_loader,
    task="binary", epochs=2, device="cpu",
)
prob = h.predict_torch(model=model, loader=test_loader, task="binary", return_proba=True, device="cpu")
label = (prob[:, 0] >= 0.5).astype(np.int64)
print(prob.shape, label.shape)  # (4, 1) (4,)
assert ((prob >= 0) & (prob <= 1)).all()
```

현재 표본의 순서는 문법 연습을 위해 만든 것입니다. 실제 데이터에는 적절한 분할과 전처리가 필요합니다. `y`는 고장이면 1, 아니면 0입니다. loader가 float32 텐서 `(B, 1)`로 바꿉니다. MLP는 확률이 아닌 점수인 **logit**을 출력하고, 기본 손실함수가 내부적으로 sigmoid를 처리합니다. 모델 끝에 sigmoid를 또 붙이지 마세요. 예측 함수는 `return_proba=True`일 때 sigmoid를 적용한 확률을 반환합니다. [BCEWithLogitsLoss 공식 문서](https://docs.pytorch.org/docs/2.7/generated/torch.nn.BCEWithLogitsLoss.html)에서 입력 조건을 확인할 수 있습니다.

확률 제출이면 `prob`를, 정수 라벨 제출이면 `label`을 사용하되 shape을 명세에 맞추세요. 0.5는 출발 임곗값일 뿐 F1에 최적인 값은 아닙니다. 원래 문자열이 필요한 경우 `np.where(label == 1, "고장", "정상")`처럼 복원합니다. **AUC를 평가한다고 무조건 확률을 제출하는 것은 아닙니다. 평가 지표와 제출 규격은 따로 확인합니다.**

### 다중분류: 문자 정답을 번호로 바꾸고 다시 복원한다

<!-- run: multiclass -->
```python
import numpy as np
import torch
import hdat_templates as h

h.seed_everything(42)
torch.set_num_threads(1)
X = np.arange(72, dtype=np.float32).reshape(24, 3) / 72
classes = np.array(["정상", "마모", "파손"])  # 이 예제에서 정한 고정 순서
label_to_index = {name: i for i, name in enumerate(classes)}
original_y = np.tile(classes, 8)
y = np.array([label_to_index[name] for name in original_y], dtype=np.int64)
train_loader = h.make_tensor_loader(X=X[:15], y=y[:15], task="multiclass", batch_size=5, shuffle=True)
valid_loader = h.make_tensor_loader(X=X[15:21], y=y[15:21], task="multiclass", batch_size=6)
test_loader = h.make_tensor_loader(X=X[21:], task="multiclass")
model = h.MLP(n_features=X.shape[1], out_dim=len(classes), hidden=(8,), dropout=0)
model, history = h.train_torch_model(
    model=model, train_loader=train_loader, valid_loader=valid_loader,
    task="multiclass", epochs=2, device="cpu",
)
pred_index = h.predict_torch(model=model, loader=test_loader, task="multiclass", device="cpu")
pred_name = classes[pred_index]
prob = h.predict_torch(model=model, loader=test_loader, task="multiclass", return_proba=True, device="cpu")
print(pred_index.shape, prob.shape, pred_name)  # (3,) (3, 3), 원래 문자열로 복원한 정답
assert np.allclose(prob.sum(axis=1), 1)
```

`classes`의 순서는 정답 번호와 확률 열의 뜻을 동시에 정합니다. 이 예제에서 `prob[:, 1]`은 마모 확률입니다. 실제 문제에서 클래스 순서를 지정했다면 그 순서를 사용하고, 지정이 없다면 학습 데이터에서 만든 매핑을 저장해 검증·테스트에 똑같이 적용하세요. 각 fold마다 따로 번호를 붙이면 같은 1이 다른 종류를 뜻할 수 있습니다.

이 공통 loader와 학습 루프는 정수 클래스 번호를 받습니다. `[0, 1, 0]` 같은 one-hot 행렬을 `task="multiclass"`에 넣지 마세요. 출력은 `(B, K)`지만 정답은 `(B,)`입니다. PyTorch 자체의 CrossEntropyLoss는 다른 target 형식도 지원하지만 **이 helper의 지원 범위는 더 좁습니다.** [CrossEntropyLoss 공식 문서](https://docs.pytorch.org/docs/2.7/generated/torch.nn.CrossEntropyLoss.html)와 이 사이트 함수의 조건을 구별하세요.

### 다중라벨: 열마다 정답을 하나씩 둔다

<!-- run: multilabel -->
```python
import numpy as np
import torch
import hdat_templates as h

h.seed_everything(42)
torch.set_num_threads(1)
rng = np.random.default_rng(42)
X = rng.normal(size=(24, 4)).astype(np.float32)
target_cols = ["leak", "vibration"]
y = np.column_stack([X[:, 0] > 0, X[:, 1] > 0]).astype(np.float32)
tr = h.make_tensor_loader(X=X[:16], y=y[:16], task="multilabel", batch_size=8, shuffle=True)
va = h.make_tensor_loader(X=X[16:20], y=y[16:20], task="multilabel")
te = h.make_tensor_loader(X=X[20:], task="multilabel")
model = h.MLP(n_features=X.shape[1], out_dim=len(target_cols), hidden=(8,), dropout=0)
model, history = h.train_torch_model(
    model=model, train_loader=tr, valid_loader=va,
    task="multilabel", epochs=2, device="cpu",
)
pred = h.predict_torch(model=model, loader=te, task="multilabel", threshold=0.5, device="cpu")
assert pred.shape == (4, 2)
assert set(np.unique(pred)).issubset({0, 1})
```

`target_cols`는 첫 열이 누유, 둘째 열이 진동이라는 약속입니다. 출력 2개는 둘 중 하나를 고르는 선택지가 아니라 각각 독립적인 경보입니다. 다중분류처럼 `argmax`를 쓰면 동시에 발생한 두 이상을 표현할 수 없습니다.

<details><summary>확인문제: out_dim=3만 보고 task를 결정할 수 있나요?</summary><p>아니요. 온도 세 개를 예측하면 회귀, 세 종류 중 하나를 고르면 다중분류, 세 경보가 동시에 켜질 수 있으면 다중라벨입니다. 출력 개수보다 먼저 정답의 의미를 읽어야 합니다.</p></details>

## 04. 시계열에서 T와 F를 실제 값으로 바꾸기

원본 센서 기록 `(전체 시점 수, 센서 수)`와 모델에 넣는 여러 구간 `(구간 수, 구간 길이, 센서 수)`는 서로 다릅니다. 아래 예제에서는 1초마다 측정한 센서 두 개를 가정합니다. 최근 4행을 보고 그 마지막 행에서 1행 뒤의 값을 예측합니다.

### lookback, horizon, stride를 행 번호로 읽는다

| 인자 | 이 예제의 값 | 첫 번째 구간에서 뜻하는 것 |
|---|---|---|
| `lookback=4` | 과거 4행 | 입력 행 0, 1, 2, 3 |
| `horizon=1` | 입력 마지막 행에서 1행 뒤 | 정답 행 4 |
| `stride=1` | 구간 시작을 1행씩 이동 | 다음 입력 행 1, 2, 3, 4 |
| `n_features=2` | 센서 2개 | 시간 길이 4와 무관 |
| `out_dim=1` | 예측할 값 1개 | horizon과 별개의 개수 |

`horizon=3`으로 바꾸면 마지막 관측점에서 3행 뒤 **한 시점**을 예측합니다. 앞으로 세 시점을 모두 예측하는 기능이 아닙니다. 100Hz로 일정하게 측정했다면 horizon 100은 1초에 해당하지만, 측정 간격이 불규칙하면 행 수를 초로 해석할 수 없습니다.

### 원본 정렬 → 누수 없는 구간 분할 → 같은 scaler 재사용

<!-- run: timeseries -->
```python
import numpy as np
import pandas as pd
import torch
import hdat_templates as h

h.seed_everything(42)
torch.set_num_threads(1)
df = pd.DataFrame({
    "time": pd.date_range("2026-01-01", periods=40, freq="s"),
    "temperature": np.linspace(10, 20, 40),
    "pressure": np.sin(np.arange(40) / 5),
    "next_value": np.linspace(20, 30, 40),
})
df = df.sort_values("time", kind="stable").reset_index(drop=True)
features = df[["temperature", "pressure"]].to_numpy(dtype=np.float32)
targets = df[["next_value"]].to_numpy(dtype=np.float32)
train_part, valid_part, cut = h.split_raw_time_then_window(
    features=features, targets=targets, train_ratio=0.7,
    lookback=4, horizon=1, stride=1, gap=0,
    assume_sorted=True, split_mode="forecast_origin", label_delay=0,
)
X_train_raw, y_train, train_target_rows = train_part
X_valid_raw, y_valid, valid_target_rows = valid_part
scaler = h.fit_scale_3d(X_train=X_train_raw)
X_train = h.transform_scale_3d(X=X_train_raw, scaler=scaler)
X_valid = h.transform_scale_3d(X=X_valid_raw, scaler=scaler)
print(cut, X_train.shape, X_valid.shape)  # 28 (24, 4, 2) (11, 4, 2)
assert train_target_rows.max() < cut
assert (valid_target_rows - 1).min() >= cut
```

`features`와 `targets`는 **같은 표를 함께 정렬한 뒤** 꺼냈습니다. 각자 정렬하면 입력과 정답의 짝이 깨집니다. `assume_sorted=True`는 함수에게 정렬을 시키는 명령이 아니라 “내가 이미 확인했다”는 선언입니다. 아직 정렬하지 않았다면 True만 적어 오류를 없애서는 안 됩니다.

`train_part`와 `valid_part`는 각각 세 배열을 담은 튜플입니다. 순서대로 입력 구간, 구간별 정답, 그 정답의 원본 행 위치입니다. `cut=28`은 원본에서 분할 기준으로 삼은 위치입니다. 기본 `forecast_origin` 방식은 검증의 예측 기준 시각에 학습 정답이 이미 알려져 있도록 구분합니다. 이때 검증 입력이 일부 과거 관측을 포함할 수 있습니다. 문제에서 완전히 분리된 구간이나 추가 간격을 요구한다면 `gap`과 실제 시각 조건을 다시 설계해야 합니다.

`fit_scale_3d`는 훈련 구간의 모든 시점에서 센서별 평균과 표준편차를 구합니다. 반환된 `scaler` 객체를 그대로 검증·테스트 변환에 넣으세요. 겹치는 구간의 관측값은 여러 번 집계됩니다. 원본 관측을 한 번씩만 세야 하는 명세라면 원본 훈련 구간에 scaler를 fit하는 별도 처리가 필요합니다.

### CNN1D에는 (N, T, F)를 그대로 넣는다

<!-- run: timeseries -->
```python
tr = h.make_tensor_loader(X=X_train, y=y_train, task="regression", batch_size=8, shuffle=True)
va = h.make_tensor_loader(X=X_valid, y=y_valid, task="regression", batch_size=8)
model = h.CNN1D(n_features=X_train.shape[2], out_dim=y_train.shape[1], channels=(8, 16), dropout=0)
model, history = h.train_torch_model(
    model=model, train_loader=tr, valid_loader=va,
    task="regression", epochs=2, device="cpu",
)
valid_pred = h.predict_torch(model=model, loader=va, task="regression", device="cpu")
assert valid_pred.shape == y_valid.shape == (11, 1)

# 모델을 새로 만드는 예: 아래 모델은 아직 학습되지 않았다.
rnn = h.SequenceRNN(n_features=2, out_dim=1, hidden_size=8, kind="lstm")
with torch.no_grad():
    assert rnn(torch.from_numpy(X_valid[:2])).shape == (2, 1)
```

일반적인 `torch.nn.Conv1d`는 `(B, F, T)`를 받지만 **이 파일의 `h.CNN1D`는 내부에서 축을 바꿉니다.** 앞에서 다시 `permute`하면 오히려 시간과 센서가 뒤집힙니다. `SequenceRNN`도 `(B, T, F)`를 받습니다. `n_features=X_train.shape[2]`는 마지막 축의 센서 수 2입니다. RNN의 `hidden_size=8`은 내부 기억 벡터의 크기이지 입력 센서 수가 아닙니다.

이 예제는 검증 예측까지입니다. 실제 제출에서 정답 없는 긴 기록을 구간으로 만들 때는 **제출의 각 행이 어느 시작 위치와 대응하는지** 알아야 합니다. 임의로 `np.arange(...)`를 만들고 제출 행 수만 맞추면 시점이 어긋날 수 있습니다. 07장의 `LazyWindowDataset` 예제에서 `starts`의 의미를 확인하세요. 여러 차량이 섞여 있으면 그룹을 넘는 구간도 막아야 합니다. 그룹별 구간 생성은 그룹별 검증 분할까지 자동으로 보장하지는 않습니다.

## 05. 이미지와 오토인코더는 어디가 다른가요?

이미지 모델은 채널 축을 보고, 오토인코더는 입력 자체를 정답으로 봅니다. 공통 학습 함수는 같아도 데이터를 준비하는 방식과 예측값의 해석이 다릅니다. 아래 두 예제는 각각 독립 실행입니다.

### 이미지: 원본 layout과 값 범위를 직접 확인한다

<!-- run: image -->
```python
import numpy as np
import torch
import hdat_templates as h

h.seed_everything(42)
torch.set_num_threads(1)
rng = np.random.default_rng(42)
raw = rng.integers(0, 256, size=(18, 12, 16, 3), dtype=np.uint8)
y = np.array([0, 1, 2] * 6, dtype=np.int64)
X = h.prepare_numpy_images(X=raw, layout="NHWC", divide_255=True)
print(raw.shape, X.shape)  # (18, 12, 16, 3) → (18, 3, 12, 16)
tr = h.make_tensor_loader(X=X[:12], y=y[:12], task="multiclass", batch_size=6, shuffle=True)
va = h.make_tensor_loader(X=X[12:15], y=y[12:15], task="multiclass")
te = h.make_tensor_loader(X=X[15:], task="multiclass")
model = h.SmallImageCNN(in_channels=X.shape[1], out_dim=3)
model, history = h.train_torch_model(
    model=model, train_loader=tr, valid_loader=va,
    task="multiclass", epochs=2, device="cpu",
)
pred = h.predict_torch(model=model, loader=te, task="multiclass", device="cpu")
assert pred.shape == (3,)
```

`layout="NHWC"`는 **현재 배열의 축 순서**가 이미지 수·높이·너비·채널이라는 뜻입니다. 원하는 결과 모양을 써 넣는 자리가 아닙니다. 함수의 결과는 항상 NCHW float32입니다. 원본이 이미 `(N, C, H, W)`면 `layout="NCHW"`로 적으세요. uint8의 0~255 값을 가져왔으므로 `divide_255=True`입니다. 이미 0~1인 float 배열에 다시 True를 주면 값이 또 255로 나뉩니다. 함수가 범위를 보고 대신 선택해 주지 않습니다.

`in_channels=X.shape[1]`은 이 예제에서 3입니다. RGB의 채널 수가 3이고 분류 종류도 3이라 숫자만 같을 뿐, 둘은 다른 개념입니다. 흑백 사진으로 10종을 분류한다면 `in_channels=1`, `out_dim=10`입니다. `SmallImageCNN`은 두 번 크기를 절반으로 줄이므로 높이와 너비가 적어도 4여야 합니다. 서로 다른 크기의 사진들을 한 배열로 묶으려면 먼저 같은 크기로 맞춰야 합니다.

실제 PIL 이미지라면 `Image.open(path).convert("RGB")`로 읽고 필요 시 크기를 맞춘 뒤 `np.asarray(image)`로 바꿉니다. 이때 한 장은 `(H, W, C)`이므로 여러 장을 `np.stack`으로 모아 `(N, H, W, C)`를 만드세요. 한 장짜리 RGB 배열을 배치라고 착각해 그대로 넘기지 마세요. 아래의 이미지 처리 API 예제에서는 PIL 이미지 객체와 경로 문자열도 구별합니다.

### 오토인코더: y에는 X를 넣고, 반환된 복원값에서 오차를 구한다

<!-- run: autoencoder -->
```python
import numpy as np
import torch
import hdat_templates as h

h.seed_everything(42)
torch.set_num_threads(1)
rng = np.random.default_rng(42)
X_normal = rng.normal(size=(24, 4)).astype(np.float32)
X_test = rng.normal(size=(5, 4)).astype(np.float32)
X_test[-1] += 5  # 합성 이상 표본. 실제 성능을 보장하는 데이터가 아님
tr = h.make_tensor_loader(X=X_normal[:16], y=X_normal[:16], task="regression", batch_size=8, shuffle=True)
va = h.make_tensor_loader(X=X_normal[16:], y=X_normal[16:], task="regression")
te = h.make_tensor_loader(X=X_test, task="regression")
model = h.Autoencoder(input_dim=4, latent_dim=2, hidden_dim=8)
model, history = h.train_torch_model(
    model=model, train_loader=tr, valid_loader=va,
    task="regression", epochs=2, device="cpu",
)
valid_reconstruction = h.predict_torch(model=model, loader=va, task="regression", device="cpu")
valid_error = ((X_normal[16:] - valid_reconstruction) ** 2).mean(axis=1)
threshold = np.quantile(valid_error, 0.95)
reconstruction = h.predict_torch(model=model, loader=te, task="regression", device="cpu")
anomaly_score = ((X_test - reconstruction) ** 2).mean(axis=1)
anomaly_label = (anomaly_score > threshold).astype(np.int64)
print(reconstruction.shape, anomaly_score.shape)  # (5, 4) (5,)
assert np.isfinite(anomaly_score).all()
```

입력 특징 4개를 2개의 잠재값으로 압축했다가 다시 4개로 복원합니다. 따라서 `input_dim=4`, `latent_dim=2`입니다. 학습할 때 정답은 고장 번호가 아니라 입력 자신이므로 `y=X_normal[:16]`입니다. `predict_torch`의 반환값 `(5, 4)`는 이상 점수가 아닌 **복원된 입력**입니다. 각 표본에서 네 특징의 제곱오차를 평균내면 `(5,)` 이상 점수가 됩니다.

0.95 분위수는 연습용 임곗값입니다. 정상 검증 표본 8개는 안정적인 임곗값 추정에 충분하지 않습니다. 실제로는 정상 훈련/검증의 출처, 스케일링, 검증 라벨과 평가 규칙을 확인하세요. 이상 점수가 필요한데 0/1 경보를 제출하거나 복원 배열을 그대로 제출하지 않도록 주의합니다. VAE는 모델이 세 값을 반환하고 GAN은 두 모델을 번갈아 학습하므로 이 공통 루프에 바로 넣지 않습니다. 08장의 별도 구조 예제를 참고하세요.

## 06. 지표·손실·저장 인자를 내 문제에 맞추기

02~05장은 우선 기본 손실이 작아지는 모델을 고르게 했습니다. 실전에서는 “학습할 때 줄이는 값”, “가장 좋은 에포크를 고르는 값”, “제출 파일에 담는 값”이 다를 수 있습니다. 세 가지를 각각 정해야 합니다.

### score_fn에는 숫자가 아니라 계산할 함수를 넣는다

<!-- run: metrics -->
```python
import numpy as np
import torch
import hdat_templates as h

def my_rmse(y_true, raw_output):
    return float(np.sqrt(np.mean((y_true - raw_output) ** 2)))

def my_binary_auc(y_true, raw_output):
    prob = torch.sigmoid(torch.as_tensor(raw_output)).numpy().reshape(-1)
    return h.evaluate_predictions(
        y_true=y_true.reshape(-1), y_pred=(prob >= 0.5).astype(int),
        metric="auc", y_score=prob, task="binary",
    )

truth = np.array([[1.0], [3.0]], dtype=np.float32)
raw = np.array([[2.0], [5.0]], dtype=np.float32)
print(my_rmse(truth, raw))  # sqrt((1 + 4) / 2) = 약 1.5811
assert np.isclose(my_rmse(truth, raw), np.sqrt(2.5))
assert my_binary_auc(np.array([[0], [1]]), np.array([[-2.0], [2.0]])) == 1.0
```

`my_rmse`의 두 인자 이름은 내가 정할 수 있지만, 순서는 이 helper의 약속입니다. 첫째는 검증 정답 전체, 둘째는 모델의 원출력 전체이며 둘 다 NumPy 배열입니다. 학습 함수가 매 에포크 끝에 이 함수를 호출합니다. 회귀는 보통 `(N, D)`이고, 이진분류 원출력은 `(N, 1)` logit입니다. 함수가 반환해야 하는 것은 Python 숫자 하나입니다.

다음은 **02장의 준비·학습 셀을 실행한 후**, 위에서 `my_rmse`도 정의한 상태에서 사용하는 연결 코드입니다. 새 모델로 다시 학습하려면 `model = h.MLP(...)`도 다시 실행해야 합니다. 이미 학습된 model을 넘기면 그 가중치에서 시작합니다. 다만 이 helper는 호출마다 optimizer를 새로 만들므로 정확한 optimizer 상태 복원까지 해 주는 재개 기능은 아닙니다.

```python
model, history = h.train_torch_model(
    model=model, train_loader=train_loader, valid_loader=valid_loader,
    task="regression", epochs=30, score_fn=my_rmse, maximize=False,
)
```

여기서 `score_fn=my_rmse` 뒤에 괄호가 없습니다. `score_fn=my_rmse(truth, raw)`라고 하면 지금 계산한 1.5811을 넘기는 셈이라 학습 함수가 나중에 호출할 수 없습니다. `score_fn="rmse"`도 함수가 아니라 글자입니다. 이진 AUC를 기준으로 고를 때는 **이진분류용 모델과 loader**를 준비하고 `task="binary", score_fn=my_binary_auc, maximize=True`를 전달합니다. 회귀 데이터를 그대로 두고 이름만 바꾸는 예제가 아닙니다. AUC는 검증 정답에 양성과 음성이 모두 있어야 합니다.

| 원하는 기준 | 넘길 인자 | 의미 |
|---|---|---|
| 기본 검증 손실 최소 | `score_fn=None, maximize=False` | task에 맞는 손실로 선택 |
| RMSE 최소 | `score_fn=my_rmse, maximize=False` | 작은 값이 좋음 |
| AUC 최대 | `score_fn=my_binary_auc, maximize=True` | 큰 값이 좋음 |
| MAE로 학습 | `loss_fn=torch.nn.L1Loss()` | 학습 손실을 바꿈. monitor는 별도 |
| log1p 목표의 원 단위 RMSE | callback 안에서 정답·출력 모두 `expm1` 복원 | 변환된 단위의 오차와 구별 |

### pos_weight와 class_weight는 훈련 정답에서 계산한다

이진분류에서 양성 오류를 더 크게 반영하고 싶다면 `pos_weight`를 검토할 수 있습니다. 데이터가 불균형하다는 이유만으로 무조건 켜는 정답 설정은 아닙니다. 검증 지표와 확률 보정에 미치는 영향도 확인해야 합니다. `class_weight`는 이진용 인자가 아니라 다중분류 CrossEntropyLoss용입니다.

<!-- run: metrics -->
```python
y_binary_train = np.array([0, 0, 0, 1], dtype=np.float32)
positive = (y_binary_train == 1).sum()
negative = (y_binary_train == 0).sum()
assert positive > 0 and negative > 0
positive_weight = float(negative / positive)  # 3.0
binary_loss = h.make_torch_loss(task="binary", pos_weight=positive_weight)

y_multi_train = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)
counts = np.bincount(y_multi_train, minlength=3)
assert (counts > 0).all()
class_weights = (len(y_multi_train) / (3 * counts)).astype(np.float32)
multi_loss = h.make_torch_loss(task="multiclass", class_weight=class_weights)
print(positive_weight, class_weights)  # 3.0 [1.0, 2.0, 약 0.6667]
```

이진분류 학습 호출에 `pos_weight=positive_weight`를 추가하거나, 이미 만든 손실을 `loss_fn=binary_loss`로 넘길 수 있습니다. 둘 다 주면 `loss_fn`이 우선해서 가중치 인자는 사용되지 않습니다. 다중라벨이면 라벨 열마다 음성 수/양성 수를 구한 길이 L의 배열을 넣습니다. 어떤 열에 양성이 0개라면 0으로 나누지 말고 데이터와 검증 설계를 먼저 확인하세요. 다중분류는 매핑된 번호 순서에 맞는 길이 K의 가중치 배열을 넣습니다.

### CSV 저장에는 빈 파일 이름이 아니라 샘플 표를 넣는다

<!-- run: metrics -->
```python
import tempfile
from pathlib import Path
import pandas as pd

sample_submission = pd.DataFrame({"id": ["T02", "T01"], "price": [0.0, 0.0]})
pred = np.array([[12.0], [25.0]], dtype=np.float32)
with tempfile.TemporaryDirectory(prefix="hdat-csv-") as folder:
    out = h.save_csv_submission(
        sample_submission=sample_submission,
        pred=pred,
        target_cols=["price"],
        path=Path(folder) / "practice-submission.csv",
    )
    assert out["id"].tolist() == ["T02", "T01"]
```

`sample_submission`은 `pd.read_csv(...)` 등으로 **이미 읽은 DataFrame**입니다. helper는 이 표를 복사하고 지정한 정답 열만 채웁니다. `target_cols`에는 예측 배열이 아니라 샘플 표의 열 이름을 넣습니다. `path`에는 저장할 파일의 경로를 넣습니다. **이 함수는 ID로 예측을 정렬하거나 결합하지 않습니다.** pred의 첫 행이 T02, 둘째 행이 T01에 해당하는지 먼저 확인해야 합니다. 행 수가 같다는 것만으로는 대응이 맞다는 뜻이 아닙니다.

### 오류 문구를 보고 되짚는 순서

| 오류 또는 증상 | 먼저 확인할 것 | 해결 방향 |
|---|---|---|
| `NameError: X_train` | 그 변수를 만든 셀을 실행했나요? | 02장의 전처리 셀부터 순서대로 실행 |
| `unexpected keyword argument 'metric'` | 지금 호출하는 함수 이름은 무엇인가요? | `evaluate_predictions`의 metric과 `train_torch_model`의 score_fn을 구별 |
| `str object ...` / `... has no attribute to` | model 자리에 문자열을 넣었나요? | `model = h.MLP(...)`로 만든 객체 전달 |
| `mat1 and mat2 shapes ...` | 전처리 후 열 수와 n_features가 같나요? | `X_train.shape`를 찍고 모델을 새로 생성 |
| `output/target shape mismatch` | 출력·정답이 `(B, 1)`과 `(B,)`로 어긋났나요? | task와 loader 변환, 직접 만든 모델 출력을 확인 |
| multiclass class index 오류 | y가 0~K−1이고 out_dim이 K인가요? | 번호 매핑과 마지막 층을 같이 확인 |
| sparse 입력 오류 | one-hot 결과가 scipy 희소행렬인가요? | 메모리와 인코딩 방식을 검토. 무조건 dense로 바꾸지 않기 |
| `X_test` 행 순서가 바뀜 | 테스트 loader를 섞었나요? | `shuffle=False`, 전처리와 저장 전 ID 대응 확인 |
| 점수는 나오는데 결과가 엉뚱함 | task·양성의 뜻·클래스 순서·복원 단위가 맞나요? | shape 검사 다음에 값의 의미까지 확인 |

<details><summary>마지막 확인문제: expected_shape=pred.shape가 왜 충분한 검사가 아닌가요?</summary><p>내가 만든 결과의 모양을 정답 모양이라고 가정하기 때문입니다. 문제가 테스트 30행의 단일 숫자를 (30,)으로 요구했는데 pred가 (30, 1)이어도 이 검사는 통과합니다. 기대 모양은 문제의 테스트 행 수·출력 개수·축 규격에서 따로 만들어야 합니다. 같은 이유로 제출 열 순서와 ID 대응도 별도로 확인해야 합니다.</p></details>

## 07. 입력·전처리·분할·저장 함수 사전

열 이름, 배열, 분할 위치, 설정 객체를 구별하는 사전입니다. 각 예제는 해당 항목의 준비 코드부터 실행하세요. TabularConfig와 sklearn 모델 보조 함수는 기존 파일의 API를 빠짐없이 설명하기 위한 참고이며, 앞의 실전 학습 예제는 모두 PyTorch입니다.

#### `LazyWindowDataset`

모든 윈도를 한 번에 복제하지 않고, 요청한 시작 위치에서 한 윈도씩 꺼낸다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `features` | 필수 | `series_df[['speed', 'temperature']].to_numpy(dtype=np.float32)` | 정답·ID를 제외한 시간순 원시 특성 행. (T,F), 1D (T,)도 한 특성으로 확장 가능. 시간 축 T는 전체 원시 행 수. 그룹 함수에서는 원래 행 대응 유지. |
| `targets` | `None` | `series_df[['fuel_use']].to_numpy(dtype=np.float32)` | 각 원시 행 시점의 정답 값 배열. (T,) 또는 (T,D), features와 같은 원시 행 길이. 미래 정답을 helper가 인덱스로 고르므로 이미 앞당긴 y를 다시 horizon만큼 이동하지 않는다. |
| `lookback` | `20` | `20` | 문제에서 사용할 과거 시간 길이를 샘플 개수로 환산. 양의 정수. 100Hz에서 0.2초를 20개로 해석하는 문제 관례. 초 0.2를 그대로 넣지 않는다. |
| `horizon` | `1` | `40` | 마지막 입력 관측에서 목표까지의 간격을 행 개수로 환산. 0 이상 정수. 100Hz에서 0.4초 뒤는 40칸. 입력 시작이 아니라 마지막 입력에서 센다. |
| `stride` | `1` | `1` | 윈도 시작 위치를 몇 행씩 이동할지 정한 값. 양의 정수. 윈도 helper에서는 초 단위가 아닌 시간순 행 간격. |
| `starts` | `None` | `np.array([0, 5, 10])` | 제출 행 또는 선택한 학습 표본에 대응하도록 계산한 입력 시작 행 위치. 1D 정수 위치. targets=None이면 필수. end_idx−lookback+1 또는 target_idx−horizon−lookback+1로 계산. 함수가 제출 순서를 추론하지 않는다. |
| `expected_n` | `None` | `len(submission_rows)` | 제출 명세/선택한 표본 목록에서 정한 예상 윈도 수. 정수 또는 None. None이면 행 수 검증 생략. 원시 행 수 T와 윈도 수 N은 다르다. |
| `assume_sorted` | `False` | `True` | 실제로 안정 시간순 정렬·행 대응을 확인한 뒤 명시하는 확인값. bool. True가 정렬을 실행해 주는 것은 아니다. 기본 False인 일반 윈도/분할/Lazy 호출은 오류를 낸다. |

**__len__ 호출 인자**

직접 전달할 인자는 없다.

반환: 윈도 개수 int

**__getitem__ 호출 인자**

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `idx` | 필수 | `0` | Dataset 안에서 꺼낼 윈도의 순서. Python 정수 위치 0..len(ds)-1. 원시 start 위치가 아니라 starts 배열의 인덱스. |

반환: (x,y) 또는 x; LazyWindowDataset 반환 계약 참조.

**반환값**

Dataset 객체. len(ds)=윈도 수. ds[i]는 targets가 있으면 (x:(L,F), y:(D,)), 없으면 x:(L,F) Tensor.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다. PyTorch가 설치되어 있어야 이 API가 정의된다.

```python
import hdat_templates as h
import numpy as np
X = np.arange(24, dtype=np.float32).reshape(12, 2)
y = np.arange(12, dtype=np.float32)
train_ds = h.LazyWindowDataset(X, y, lookback=3, horizon=2,
                             assume_sorted=True)
xb, yb = train_ds[0]  # X[0:3], y[4]
assert xb.shape == (3, 2) and yb.tolist() == [4.0]
# 여기서는 제출할 세 행이 입력 끝 2, 7, 11에 대응한다고 정했다.
end_indices = np.array([2, 7, 11])
starts = end_indices - 3 + 1
test_ds = h.LazyWindowDataset(X, targets=None, lookback=3,
                            starts=starts, expected_n=3, assume_sorted=True)
assert len(test_ds) == 3
```

**주의할 점**

- 원시 X/y는 CPU float32 Tensor로 보관한다. 전체 윈도 복제를 피하지만 원시 데이터 자체는 메모리에 필요하다.
- targets=None이면 starts 필수: 제출 각 행과 일치하는 입력 시작 위치를 명시한다. expected_n도 권장한다.
- horizon은 targets가 있을 때 y 선택에 사용. 추론에서는 입력이 실제 관측 범위에 있는지를 검사하며 미래 y 공간을 요구하지 않는다.
- 그룹 경계 검사가 없으므로 여러 차량을 이어 붙인 원시 배열에 무수정 적용하지 않는다.
- starts는 전달한 순서를 그대로 쓰며 int 변환 과정에서 소수가 잘릴 수 있으므로 호출자가 정수·중복·행 의미를 먼저 검사한다.
- 회귀 y는 적합하지만 다중분류 labels도 float32로 바뀐다. 공통 루프는 hard 정수 값이면 long으로 바꾸지만 class mapping은 별도다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `TabularConfig`

표형 sklearn baseline 흐름에 전달할 선택값을 한 객체에 묶는다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `task` | `'regression'` | `'binary'` | 정답이 연속값인지, 이진/다중분류/다중라벨인지 문제에서 판별. 'regression'\|'binary'\|'multiclass'\|'multilabel'의 정확한 문자열. 출력·정답·loss 계약이 함께 바뀐다. 모든 함수가 오타를 즉시 검사하는 것은 아니다. |
| `target_cols` | `('target',)` | `('fault',)` | 제공된 훈련 표의 실제 정답 열 이름과 출력 순서. tuple[str,...]. 기본 ('target',)은 예시이며 실제 열 이름으로 수정해야 한다. |
| `drop_cols` | `()` | `('vehicle_id', 'record_id')` | 정답 이외의 ID·누수 열 등 실제로 제외할 열. tuple[str,...]. target_cols/datetime_cols와 겹치면 wrapper가 거절. group_col을 지정해도 자동으로 빠지지 않는다. |
| `datetime_cols` | `()` | `('recorded_at',)` | 날짜 파생 특성을 만들 실제 원본 열. tuple[str,...]. train/test에 있어야 한다. 파싱 실패는 NaT, 파생 열은 NaN으로 처리된다. |
| `time_col` | `None` | `'recorded_at'` | 시간순 검증에 쓸 훈련 표의 실제 열 이름. 문자열 또는 None. 그룹 분할용 group_col과 동시에 설정하지 않는다. 이 설정 객체에는 time_values 필드가 없다. |
| `group_col` | `None` | `'vehicle_id'` | 차량·사람 등 같은 집단을 묶을 실제 열 이름. 문자열 또는 None. split 전에 원래 X에서 값을 확보. 테스트가 새 차량이면 적합한 후보이며 ID 자동 삭제 기능은 없다. |
| `metric` | `'rmse'` | `'f1_macro'` | 문제에서 정한 공식 평가 지표 이름. mse\|rmse\|mae\|rmsle\|accuracy\|f1\|f1_macro\|auc\|roc_auc. f1도 이 helper에서는 macro이며 양성 F1 별칭이 아니다. |
| `model_name` | `'extra_trees'` | `'linear'` | 문제 명세 또는 검증 예산으로 고른 sklearn 기준 모델 종류. 'linear'\|'extra_trees'. 'MLP'나 이미 생성된 객체가 아니다. |
| `encoding` | `'onehot'` | `'ordinal'` | 범주 열의 고유값 수·메모리와 모델에 맞춰 정한 표현. 'onehot'\|'ordinal'만 사용. 현 구현은 onehot 이외 문자열을 ordinal로 처리하므로 오타 주의. |
| `valid_size` | `0.2` | `0.2` | 검증에 남길 비율: 문제 조건 또는 사전에 고정한 계획. 0<값<1. 무작위/시간은 행 기준, 그룹 분할에서는 고유 그룹 기준 비율. |
| `seed` | `42` | `42` | 본인이 고정한 실험 재현용 정수. 문제에 지정값이 있으면 그것을 사용. 정수 하나. 같은 seed가 모든 GPU 연산의 완전한 결정론을 보장하지 않는다. |
| `output_kind` | `None` | `'label'` | 점수 이름이 아니라 제출 파일이 요구하는 값의 종류. 회귀 value, 분류 label/probability. None이면 일반적으로 회귀 value·분류 label; AUC에서는 명시해야 한다. |
| `positive_label` | `1` | `'fault'` | 이진 확률 열이 의미해야 할 원래 정답 라벨. 학습된 classes_ 안의 값. 제출 확률 열 선택에 쓰며 훈련 라벨을 자동 0/1 매핑하지 않는다. |
| `class_order` | `None` | `('normal', 'minor', 'major')` | 제출 확률 열의 정해진 클래스 순서. 모든 학습 클래스를 중복 없이 포함한 순서. None은 estimator classes_ 순서. |

**반환값**

TabularConfig 인스턴스. cfg.task, cfg.target_cols처럼 속성으로 읽는다.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
cfg = h.TabularConfig(
    task="binary",
    target_cols=("fault",),       # 제공된 훈련 표의 실제 정답 열
    drop_cols=("record_id", "vehicle_id"),
    group_col="vehicle_id",       # 새 차량을 검증 대상으로 분리
    metric="f1_macro",
    model_name="extra_trees",
    encoding="ordinal",
    valid_size=0.2,
    output_kind="label",          # 제출은 확률이 아니라 0/1 라벨
)
assert cfg.target_cols == ("fault",)
```

**주의할 점**

- 기술적으로 모든 필드에 기본값이 있지만 실제 문제의 정답 열·task·metric·출력 종류는 반드시 대조한다.
- target_cols=('target',)의 'target'은 실제 열 이름을 대신하는 기본 예시다. task를 binary로 바꿔도 metric='rmse'는 자동 변경되지 않는다.
- time_col과 group_col을 동시에 설정하면 wrapper의 split 단계가 거절한다.
- group_col은 분할 용도일 뿐 자동 drop하지 않는다. 새 차량 ID를 빼려면 drop_cols에 명시한다.
- PyTorch train_torch_model은 이 객체를 받지 않는다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `seed_everything`

Python·NumPy·가능한 PyTorch의 난수 시작점을 맞춘다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `seed` | `42` | `42` | 본인이 고정한 실험 재현용 정수. 문제에 지정값이 있으면 그것을 사용. 정수 하나. 같은 seed가 모든 GPU 연산의 완전한 결정론을 보장하지 않는다. |

**반환값**

None. 전역 난수 상태를 바꾼다.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import numpy as np
h.seed_everything(42)
first = np.random.rand(3)
h.seed_everything(42)
second = np.random.rand(3)
assert np.array_equal(first, second)
```

**주의할 점**

- 모델 생성·shuffle·무작위 분할 전에 호출한다.
- 여러 장치·라이브러리 버전·비결정적 연산까지 결과 동일성을 보장하지 않는다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `quick_audit`

표의 크기·중복·dtype·결측·고유값·정답 요약을 제한된 분량으로 돌려준다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `df` | 필수 | `train_df.copy()` | 현재 처리할 원본 표 또는 분할한 표. pandas DataFrame. 열 이름/순서·인덱스를 확인한다. |
| `target_cols` | `()` | `('fuel_use',)` | 문제 설명의 정답 열 이름과 출력 순서. 문자열의 list/tuple. 한 열 tuple에도 쉼표가 필요하다. ('fuel_use')는 문자열이다. |

**반환값**

dict. shape tuple, 개수, Series, target_summary dict 등이 섞여 있다.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import pandas as pd
df = pd.DataFrame({"speed": [10., None, 20.], "fault": [0, 1, 0]})
report = h.quick_audit(df, target_cols=("fault",))
print(report["shape"])
print(report["target_summary"]["fault"])
```

**주의할 점**

- target_cols에 없는 열 이름은 오류 없이 요약 대상에서 빠지므로 오타를 별도로 확인한다.
- 중복 열을 찾아 보여 주지만 정답 열 이름이 중복되면 target_summary 처리도 실패할 수 있다.
- nunique_top20은 고유값 수가 작은 순이다. 이름의 top이 큰 순이라는 뜻은 아니다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `assert_frame_contract`

훈련 정답과 테스트 특성의 열 계약을 검사한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `train` | 필수 | `train_df` | 제공된 훈련 파일을 읽은 표 또는 통계를 배울 훈련 폴드. DataFrame. wrapper에는 정답 열 포함, 통계 fit helper에는 훈련 부분만. |
| `test` | 필수 | `test_df` | 정답 없이 제공된 테스트 파일의 표. DataFrame. 제출 대상 행의 원래 순서를 보존한다. |
| `target_cols` | 필수 | `('fuel_use',)` | 문제 설명의 정답 열 이름과 출력 순서. 문자열의 list/tuple. 한 열 tuple에도 쉼표가 필요하다. ('fuel_use')는 문자열이다. |
| `ignore_feature_cols` | `()` | `('record_id',)` | ID·누수 열처럼 모델 입력에서 제외하기로 한 실제 열. 열 이름 목록. train에 없는 이름은 오류. 이 함수는 검사만 하고 실제 삭제하지 않는다. |

**반환값**

None. assert/KeyError로 실패를 알리고 일부 추가 열은 경고만 출력.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import pandas as pd
train = pd.DataFrame({"id": [1, 2], "speed": [10., 20.], "target": [1., 2.]})
test = pd.DataFrame({"id": [3], "speed": [15.]})
h.assert_frame_contract(train, test, target_cols=("target",),
                       ignore_feature_cols=("id",))
```

**주의할 점**

- 정답은 train에 있어야 하고 test에는 없어야 한다. 필요한 특성이 빠졌으면 실패한다.
- test의 추가 열은 즉시 실패가 아니라 경고다. 열 순서를 정렬하거나 dtype를 비교하지 않는다.
- ignore_feature_cols는 검사에서 제외할 뿐 표에서 열을 지우지 않는다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `minmax_selected`

지정 열을 현재 표의 최솟값·최댓값으로 0..1 스케일링한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `df` | 필수 | `train_df.copy()` | 현재 처리할 원본 표 또는 분할한 표. pandas DataFrame. 열 이름/순서·인덱스를 확인한다. |
| `columns` | 필수 | `['speed', 'temperature']` | 문제에서 처리하라고 지정한 실제 열 이름. 문자열 목록. df 전체나 열 값 자체가 아닌 열 이름을 넣는다. 대상 데이터는 해당 연산이 가능한 dtype이어야 한다. |

**반환값**

원래 인덱스·비대상 열을 보존한 복사 DataFrame. 선택 열은 float로 변환.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import pandas as pd
df = pd.DataFrame({"speed": [10., 20., 30.], "id": ["a", "b", "c"]})
scaled = h.minmax_selected(df, columns=["speed"])
assert scaled["speed"].tolist() == [0.0, 0.5, 1.0]
assert df["speed"].tolist() == [10.0, 20.0, 30.0]
```

**주의할 점**

- 같은 호출의 표에서 통계를 매번 계산한다. train/valid/test에 각각 호출하면 서로 다른 척도를 쓰므로 예측 전처리의 train-fit/valid-transform 대체품이 아니다.
- 상수 열은 유효값만 0, NaN은 보존. 전부 NaN인 열은 NaN 유지.
- 숫자로 변환 가능한 문자열은 변환하지만 Inf·중복 열·비수치 입력을 완전히 방어하는 함수는 아니다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `fit_iqr_bounds`

훈련 표에서만 IQR clipping 하한·상한을 계산한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `train` | 필수 | `X.iloc[tr_idx]` | 통계를 배워도 되는 훈련 폴드만 선택한 표. DataFrame. 지정 columns는 수치형이고 train 통계만 이후 valid/test에 적용. |
| `columns` | 필수 | `['speed', 'temperature']` | 문제에서 처리하라고 지정한 실제 열 이름. 문자열 목록. df 전체나 열 값 자체가 아닌 열 이름을 넣는다. 대상 데이터는 해당 연산이 가능한 dtype이어야 한다. |
| `whisker` | `1.5` | `1.5` | IQR 경계 식의 배수: 문제 지시 또는 검증에서 정한 설정. 실수 하나. 통상 0 이상으로 사용하며 helper 자체는 음수 등을 별도 검증하지 않는다. |

**반환값**

dict[str,tuple[float,float]]. 해당 열을 아직 변환하지 않는다.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import pandas as pd
train_fold = pd.DataFrame({"speed": [0., 2., 4., 6.]})
bounds = h.fit_iqr_bounds(train_fold, columns=["speed"], whisker=1.5)
assert bounds["speed"] == (-3.0, 9.0)
```

**주의할 점**

- split 후 train 폴드에만 fit하고 동일 bounds를 valid/test에 재사용한다.
- IQR=Q3−Q1, 하한 Q1−whisker×IQR, 상한 Q3+whisker×IQR.
- 전체 NaN·상수·Inf 등의 경계를 별도로 확인한다. 자동 삭제 함수가 아니다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `apply_clip_bounds`

이미 배운 열별 하한·상한으로 값을 경계에 눌러 붙인다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `df` | 필수 | `train_df.copy()` | 현재 처리할 원본 표 또는 분할한 표. pandas DataFrame. 열 이름/순서·인덱스를 확인한다. |
| `bounds` | 필수 | `bounds` | h.fit_iqr_bounds(train_fold, columns)의 반환값. {열 이름: (하한, 상한)} 딕셔너리. valid/test에서 새로 계산하지 않는다. |

**반환값**

복사 DataFrame. 대상 열의 초과값만 clip하며 행을 삭제하지 않는다.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import pandas as pd
train_fold = pd.DataFrame({"speed": [0., 2., 4., 6.]})
valid_fold = pd.DataFrame({"speed": [-10., 20.]})
bounds = h.fit_iqr_bounds(train_fold, ["speed"])
clipped = h.apply_clip_bounds(valid_fold, bounds)
assert clipped["speed"].tolist() == [-3.0, 9.0]
```

**주의할 점**

- bounds를 valid/test에서 다시 학습하지 않는다.
- 범위를 벗어난 행을 제거하거나 이상치 라벨을 만드는 함수가 아니다.
- 존재하지 않는 bounds 열은 오류. 양 끝·NaN 경계를 호출 전 점검한다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `add_datetime_features`

지정 날짜 열에서 연·월·일·요일·시·분을 만든다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `df` | 필수 | `train_df.copy()` | 현재 처리할 원본 표 또는 분할한 표. pandas DataFrame. 열 이름/순서·인덱스를 확인한다. |
| `columns` | 필수 | `['time']` | 날짜 파생변수를 만들 원본 날짜 열의 이름 목록. 예제의 df에는 time 열이 있으므로 ['time']을 넣는다. 날짜 값 자체나 수치 센서 열 이름을 넣는 자리가 아니다. |
| `drop_original` | `True` | `True` | 날짜 원래 열을 모델 입력에서 없앨지 결정. bool. True면 날짜 파생 열을 만들고 원래 열을 삭제한다. |

**반환값**

복사 DataFrame. 각 열에 __year/__month/__day/__dow/__hour/__minute를 추가.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import pandas as pd
df = pd.DataFrame({"time": ["2026-09-08 13:45", "bad-date"]})
result = h.add_datetime_features(df, columns=["time"], drop_original=True)
print(result[["time__year", "time__dow", "time__hour"]])
```

**주의할 점**

- 문자열 날짜를 errors='coerce'로 해석하므로 잘못된 날짜는 NaT/파생 NaN.
- dayofweek는 월요일 0부터 일요일 6. 원래 열은 drop_original=True일 때 삭제한다.
- 이 함수는 시간순으로 정렬하지 않는다. 같은 이름의 기존 파생 열이 있으면 덮어쓸 수 있다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `clean_tabular_values`

수치 Inf를 NaN으로 바꾸고 범주형 혼합값을 문자열 표현으로 통일한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `df` | 필수 | `train_df.copy()` | 현재 처리할 원본 표 또는 분할한 표. pandas DataFrame. 열 이름/순서·인덱스를 확인한다. |

**반환값**

복사 DataFrame. 숫자 열은 Inf→NaN, 다른 열은 문자열 또는 np.nan.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import numpy as np
import pandas as pd
df = pd.DataFrame({"value": [1., np.inf], "category": ["A", None]})
cleaned = h.clean_tabular_values(df)
assert pd.isna(cleaned.loc[1, "value"])
assert pd.isna(cleaned.loc[1, "category"])
```

**주의할 점**

- 결측 대체·범주 인코딩·표준화를 수행하지 않는다. 후속 전처리기의 입력을 정리한다.
- datetime·문자열 수치는 비수치 범주로 취급될 수 있어 날짜 분해/수치 변환 여부를 먼저 결정한다.
- target에 무작정 적용하지 말고 모델 입력 X를 정리하는 데 사용한다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `crop_to_numpy`

단일 PIL 이미지의 지정 좌표를 잘라 NumPy 배열로 돌려준다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `image` | 필수 | `image` | PIL로 이미 읽은 단일 이미지. PIL Image처럼 crop 메서드를 가진 객체. 파일 경로 문자열이나 배치 배열이 아니다. |
| `box` | 필수 | `(2, 1, 7, 5)` | 문제의 자르기 좌표 또는 이미지 폭·높이로 계산한 경계. (left, upper, right, lower)의 정수 4개. x=2..6, y=1..4 영역으로 폭 5·높이 4. 중앙 crop 크기 2개를 넣는 인자가 아니다. |

**반환값**

잘린 영역의 copy ndarray. RGB (H,W,3), 흑백 (H,W) 등 원래 모드의 dtype/채널 유지.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
from PIL import Image
image = Image.new("RGB", (10, 7), color=(255, 0, 0))  # PIL은 (W,H)
patch = h.crop_to_numpy(image, box=(2, 1, 7, 5))
assert patch.shape == (4, 5, 3)
```

**주의할 점**

- PIL Image가 필요하다. 경로 문자열이나 (x,y,width,height) 표현은 인자가 아니다.
- 오른쪽·아래 끝은 제외된다. 좌표 유효성·원본 범위·중앙 맞춤은 이 wrapper가 별도로 검사하지 않는다.
- 반환값을 PyTorch에 넣으려면 배치/채널 축과 값 척도 준비가 더 필요하다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `conv_output_size`

한 합성곱 공간 축의 출력 길이를 계산한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `input_size` | 필수 | `32` | 현재 합성곱에 들어갈 한 공간 축의 길이. 양의 정수. 높이와 너비는 각각 호출. 배치 크기·채널 수가 아니다. |
| `kernel_size` | 필수 | `3` | 문제에 지정된 합성곱 커널의 해당 축 길이. 양의 정수. 이 helper는 2D tuple이 아닌 한 축의 정수 계산. |
| `stride` | `1` | `2` | 문제 명세의 합성곱 이동 간격. 양의 정수. 커널 시작 위치가 한 공간 축에서 움직이는 간격이다. |
| `padding` | `0` | `1` | 합성곱 한쪽에 붙이는 padding 길이. 0 이상 정수. 양쪽 합이 아니라 한쪽 값이며 식에서 2P로 계산. |
| `dilation` | `1` | `1` | 합성곱 커널 원소 사이의 간격 설정. 양의 정수. 유효 커널 크기는 D(K−1)+1. |

**반환값**

Python int. floor((I+2P−D(K−1)−1)/S+1).

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
out_height = h.conv_output_size(input_size=32, kernel_size=3,
                               stride=2, padding=1, dilation=1)
out_width = h.conv_output_size(input_size=40, kernel_size=3,
                              stride=2, padding=1, dilation=1)
assert (out_height, out_width) == (16, 20)
```

**주의할 점**

- 한 축 계산기이므로 높이·너비가 다르면 두 번 호출한다.
- 유효한 양의 input/kernel/stride/dilation과 0 이상 padding을 호출자가 보장해야 한다.
- 채널 수·학습 파라미터 수를 반환하지 않는다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `safe_train_valid_indices`

행 무작위·시간·그룹 중 지정한 한 방법으로 train/valid 행 위치를 만든다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `X` | 필수 | `X` | 아직 train/valid를 나누지 않은 입력 표. DataFrame N행, N≥2. y/time/groups의 원래 행 순서와 일치. |
| `y` | 필수 | `y_train` | X의 각 행과 정확히 대응하는 정답. 회귀 (N,) 또는 (N,D); 이진 (N,) 또는 (N,1) 0/1; 다중분류 (N,) class index; 다중라벨 (N,K) 0/1. sklearn 함수는 원래 클래스 라벨도 가능. |
| `task` | 필수 | `'binary'` | 정답이 연속값인지, 이진/다중분류/다중라벨인지 문제에서 판별. 'regression'\|'binary'\|'multiclass'\|'multilabel'의 정확한 문자열. 출력·정답·loss 계약이 함께 바뀐다. 모든 함수가 오타를 즉시 검사하는 것은 아니다. |
| `valid_size` | `0.2` | `0.2` | 검증에 남길 비율: 문제 조건 또는 사전에 고정한 계획. 0<값<1. 무작위/시간은 행 기준, 그룹 분할에서는 고유 그룹 기준 비율. |
| `seed` | `42` | `42` | 본인이 고정한 실험 재현용 정수. 문제에 지정값이 있으면 그것을 사용. 정수 하나. 같은 seed가 모든 GPU 연산의 완전한 결정론을 보장하지 않는다. |
| `time_col` | `None` | `'recorded_at'` | 시간순 검증에 쓸 실제 열 이름. 문자열 또는 None. time_values가 제공되면 그것이 우선. group split과 동시에 쓰지 않는다. |
| `time_values` | `None` | `train_df['recorded_at'].to_numpy()` | X와 같은 행 순서로 보관한 실제 시간 열 값. 길이 N의 1D. time_col보다 우선하며 pd.to_datetime으로 변환할 수 있어야 한다. |
| `groups` | `None` | `train_df['vehicle_id'].to_numpy()` | X/features 각 행에 대응하는 차량·주행·개체 ID 값. 길이 N 또는 T의 1D, 결측 없음. 열 이름 문자열 자체를 넣지 않는다. |

**반환값**

(tr_idx,va_idx): 1D 정수 NumPy 위치 배열.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import numpy as np
import pandas as pd
X = pd.DataFrame({"speed": np.arange(6, dtype=float)})
y = np.arange(6, dtype=float)
groups = np.array(["A", "A", "B", "B", "C", "C"])
tr_idx, va_idx = h.safe_train_valid_indices(
    X, y, task="regression", valid_size=1/3, groups=groups)
assert not set(groups[tr_idx]) & set(groups[va_idx])
X_train, X_valid = X.iloc[tr_idx], X.iloc[va_idx]
```

**주의할 점**

- X/y는 같은 N행, N≥2. 반환값은 .iloc 또는 NumPy 위치 인덱싱에 쓴다. .loc와 혼동하지 않는다.
- time_values/time_col과 groups를 동시에 제공하면 오류. 두 목표를 결합한 검증은 별도 구현.
- time은 stable한 시간 정렬 뒤 앞/뒤를 나누지만 horizon·정답 공개 시점은 모른다. 시계열 윈도 전용 함수를 검토한다.
- group valid_size는 그룹 비율이며 행 비율이 정확히 같지 않을 수 있다.
- 무작위 분류의 stratify는 각 클래스 개수와 양쪽 분할 크기가 허용할 때만 켜진다. 클래스 보존은 별도 검사한다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `make_preprocessor`

입력 표의 수치/범주 열에 맞는 미학습 전처리기를 만든다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `X` | 필수 | `X.iloc[tr_idx]` | 분할한 훈련 폴드의 입력 표. pandas DataFrame (N_train,F_raw). 함수 생성 시 열/dtype만 읽으며 fit은 이후 train에만. |
| `encoding` | `'onehot'` | `'ordinal'` | 범주 열의 고유값 수·메모리와 모델에 맞춰 정한 표현. 'onehot'\|'ordinal'만 사용. 현 구현은 onehot 이외 문자열을 ordinal로 처리하므로 오타 주의. |
| `scale_numeric` | `False` | `True` | 거리/선형 모델 등에 수치 표준화를 적용할지 결정. bool. True면 중앙값 대체 후 StandardScaler; missing indicator도 수치 파이프라인 안에서 변환된다. |

**반환값**

미학습 ColumnTransformer. fit_transform 후 (N,F_out)의 수치 배열 또는 sparse matrix.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import pandas as pd
X_train = pd.DataFrame({"speed": [10., None, 30.], "road": ["city", "road", "city"]})
X_valid = pd.DataFrame({"speed": [20.], "road": ["new"]})
prep = h.make_preprocessor(X_train, encoding="ordinal", scale_numeric=True)
X_train_s = prep.fit_transform(X_train)
X_valid_s = prep.transform(X_valid)
assert X_train_s.shape[1] == X_valid_s.shape[1]
```

**주의할 점**

- 생성할 때 X는 열·dtype를 고르는 용도다. 통계를 배우는 실제 fit은 train 폴드에만 해야 한다.
- 수치: 중앙값 대체+결측 indicator, 선택적 표준화. 범주: 결측 문자열 대체+onehot/ordinal.
- OneHotEncoder를 썼다고 최종 출력이 항상 sparse나 float32라는 보장은 없다. 결합 결과의 실제 타입·dtype·폭을 확인한다.
- unseen 범주는 onehot 0벡터 또는 ordinal −1로 처리. 열 이름/순서는 후속 입력에도 동일하게 유지한다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `make_tabular_estimator`

과제·모델 이름에 맞는 미학습 sklearn estimator를 만든다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `task` | 필수 | `'binary'` | 정답이 연속값인지, 이진/다중분류/다중라벨인지 문제에서 판별. 'regression'\|'binary'\|'multiclass'\|'multilabel'의 정확한 문자열. 출력·정답·loss 계약이 함께 바뀐다. 모든 함수가 오타를 즉시 검사하는 것은 아니다. |
| `model_name` | `'extra_trees'` | `'linear'` | 문제 명세 또는 검증 예산으로 고른 sklearn 기준 모델 종류. 'linear'\|'extra_trees'. 'MLP'나 이미 생성된 객체가 아니다. |
| `seed` | `42` | `42` | 본인이 고정한 실험 재현용 정수. 문제에 지정값이 있으면 그것을 사용. 정수 하나. 같은 seed가 모든 GPU 연산의 완전한 결정론을 보장하지 않는다. |
| `multioutput` | `False` | `True` | 서로 다른 정답 열이 여러 개인지 shape로 판단. bool. 다중분류의 클래스 수 C가 3이라는 이유만으로 True가 되는 것은 아니다. |

**반환값**

Ridge/LogisticRegression/ExtraTreesRegressor/ExtraTreesClassifier 또는 MultiOutputClassifier.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import numpy as np
X = np.array([[0.], [1.], [2.], [3.]])
y = np.array([0., 2., 4., 6.])
estimator = h.make_tabular_estimator(task="regression", model_name="linear")
estimator.fit(X, y)  # 이 작은 예시는 API 동작 확인용
pred = estimator.predict(np.array([[1.5]]))
assert pred.shape == (1,)
```

**주의할 점**

- task='regression'인지에 따라 회귀/분류가 나뉜다. 허용 문자열을 정확히 쓰고 typo가 분류로 흘러가지 않게 한다.
- multioutput은 정답 열 여러 개를 뜻한다. 클래스 3종의 단일 정답은 multiclass,multioutput=False다.
- 입력 전처리를 수행하지 않으며 fit도 하지 않는다. 비수치·결측 입력은 먼저 처리한다.
- 모델 내부 하이퍼파라미터는 이 생성기의 고정값이며 문제의 정확한 모델 명세와 다른지 대조한다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `evaluate_predictions`

실제 정답·예측·필요한 확률로 지원 지표 하나를 계산한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `y_true` | 필수 | `y_valid` | 검증 폴드의 실제 정답, 공식 단위·라벨로 준비. 예측과 동일한 행 순서. 분류 지표의 라벨 정의와 다중출력 집계 정책도 맞춰야 한다. |
| `y_pred` | 필수 | `valid_pred` | 현재 모델의 검증 예측값 또는 확정 라벨. MSE 등은 연속값, accuracy/F1은 확정 라벨. AUC에서 y_score를 주면 y_pred 대신 그 점수를 사용. |
| `metric` | 필수 | `'f1_macro'` | 문제에서 정한 공식 평가 지표 이름. mse\|rmse\|mae\|rmsle\|accuracy\|f1\|f1_macro\|auc\|roc_auc. f1도 이 helper에서는 macro이며 양성 F1 별칭이 아니다. |
| `y_score` | `None` | `valid_probability` | ROC-AUC에 사용할 연속 점수 또는 확률. 이진 (N,) 또는 (N,1)/(N,2), 다중분류 (N,C) 확률, 다중라벨 (N,K). None이면 y_pred 사용. 라벨 점수 대신 연속값 권장. |
| `task` | `None` | `'multilabel'` | 특히 다중라벨 평가 집계를 명시할 때 사용하는 과제 구분. None 가능. multilabel이면 subset accuracy/label-macro F1. 명시하지 않으면 다른 다중출력 분류 집계가 적용될 수 있다. |

**반환값**

Python float. 학습/모델 선택/제출을 하지 않는다.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import numpy as np
y_true = np.array([0, 1, 1, 0])
prob = np.array([0.1, 0.8, 0.6, 0.7])
label = (prob >= 0.5).astype(int)
macro_f1 = h.evaluate_predictions(y_true, label, metric="f1_macro", task="binary")
auc = h.evaluate_predictions(y_true, label, metric="roc_auc",
                            y_score=prob, task="binary")
print(macro_f1, auc)
```

**주의할 점**

- metric='f1'와 'f1_macro' 모두 macro F1. 양성 클래스 하나의 binary F1과 다르다.
- task='multilabel'의 accuracy는 전체 라벨이 모두 맞아야 맞는 subset accuracy; F1은 라벨별 양성 F1 평균.
- 다른 다중출력 분류는 열별 accuracy/macro-F1 평균. 공식 산식이 다르면 별도 구현한다.
- RMSLE는 음수 y_true를 거절하지만 y_pred는 0으로 clip한다. 음수 예측을 오류로 처리하는 명세와 다르다.
- AUC는 연속 점수가 필요하다. 2열은 두 번째 열을, 1열은 첫 열을 사용한다. positive_label/class_order 인자가 없어 원래 라벨과 열 대응을 호출자가 맞춰야 한다.
- 다중분류 AUC 확률 열은 sklearn의 클래스 순서와 일치해야 하며 각 행 확률 합 등 조건도 맞아야 한다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `assert_class_fold_coverage`

분류 train/valid에 필요한 클래스가 들어 있는지 검사한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `y` | 필수 | `y_train` | X의 각 행과 정확히 대응하는 정답. 회귀 (N,) 또는 (N,D); 이진 (N,) 또는 (N,1) 0/1; 다중분류 (N,) class index; 다중라벨 (N,K) 0/1. sklearn 함수는 원래 클래스 라벨도 가능. |
| `train_indices` | 필수 | `tr_idx` | 분할 함수가 반환한 훈련 행 위치. 1D 정수 배열. DataFrame 인덱스 라벨이 아니라 0..N−1 위치. |
| `valid_indices` | 필수 | `va_idx` | 분할 함수가 반환한 검증 행 위치. 1D 정수 배열. y와 같은 원래 행 배열을 기준으로 한다. |
| `task` | 필수 | `'binary'` | 정답이 연속값인지, 이진/다중분류/다중라벨인지 문제에서 판별. 'regression'\|'binary'\|'multiclass'\|'multilabel'의 정확한 문자열. 출력·정답·loss 계약이 함께 바뀐다. 모든 함수가 오타를 즉시 검사하는 것은 아니다. |
| `metric` | 필수 | `'f1_macro'` | 문제에서 정한 공식 평가 지표 이름. mse\|rmse\|mae\|rmsle\|accuracy\|f1\|f1_macro\|auc\|roc_auc. f1도 이 helper에서는 macro이며 양성 F1 별칭이 아니다. |

**반환값**

None 또는 ValueError.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import numpy as np
y = np.array([0, 1, 0, 1])
tr_idx, va_idx = np.array([0, 1]), np.array([2, 3])
h.assert_class_fold_coverage(y, tr_idx, va_idx,
                           task="binary", metric="roc_auc")
```

**주의할 점**

- 분할 후 모델 fit 전에 호출. task가 회귀면 검사 없이 반환한다.
- 검증에만 존재하는 클래스를 거절하고 binary/multiclass train 클래스 1개도 거절한다.
- AUC는 양쪽에 2개 이상 클래스 필요; multiclass AUC는 양쪽 클래스 집합도 같아야 한다.
- 라벨 자체의 모든 dtype/0..C−1 계약이나 같은 그룹 중복까지 검사하는 함수는 아니다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `fit_tabular_baseline`

표 데이터의 분할·train-only 전처리·검증·전체 재학습·테스트 예측을 연결한 연습용 wrapper다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `train` | 필수 | `train_df` | 제공된 훈련 파일을 읽은 표 또는 통계를 배울 훈련 폴드. DataFrame. wrapper에는 정답 열 포함, 통계 fit helper에는 훈련 부분만. |
| `test` | 필수 | `test_df` | 정답 없이 제공된 테스트 파일의 표. DataFrame. 제출 대상 행의 원래 순서를 보존한다. |
| `cfg` | 필수 | `h.TabularConfig(task='regression', target_cols=('fuel_use',), metric='rmse')` | h.TabularConfig(...)를 호출해 만든 설정 객체. TabularConfig 인스턴스. 딕셔너리·클래스 자체·Torch config 객체가 아니다. |

**반환값**

(pipe,test_pred,metric_value). pipe는 마지막 전체 train 재학습 모델; metric_value는 앞선 holdout 모델의 검증 점수.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import pandas as pd
train = pd.DataFrame({"speed": [0.,1.,2.,3.,4.,5.,6.,7.],
                      "fuel": [0.,2.,4.,6.,8.,10.,12.,14.]})
test = pd.DataFrame({"speed": [2.5, 4.5]})
cfg = h.TabularConfig(task="regression", target_cols=("fuel",),
                     metric="rmse", model_name="linear",
                     output_kind="value", valid_size=0.25)
pipe, test_pred, valid_rmse = h.fit_tabular_baseline(train, test, cfg)
assert test_pred.shape == (2,)
# 아직 파일 저장이나 실제 제출은 이루어지지 않았다.
```

**주의할 점**

- 실제 시험의 제공 skeleton/함수명/저장 셀을 대신하는 무수정 end-to-end 호출로 안내하면 안 된다.
- cfg는 TabularConfig 객체. train은 정답 포함, test는 정답 미포함. binary/multiclass는 정답 열 하나만 지원.
- group/time 값은 특성 삭제 전에 확보한다. group_col은 자동 삭제되지 않음.
- valid 결과를 출력한 뒤 자동으로 전체 train 재학습. 반환 pipe가 holdout-only 모델이라고 생각하면 안 된다.
- AUC여도 제출 label/probability는 별도 명세. output_kind=None이면 오류가 늦게 발생하므로 시작 전에 명시한다.
- 저장/실제 제출/임계값 최적화는 하지 않는다. positive_label/class_order는 확률 제출에 반영된다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `predict_tabular_probabilities`

학습된 sklearn Pipeline에서 제출 라벨 의미·클래스 순서에 맞춘 확률을 고른다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `pipe` | 필수 | `pipe` | 전처리와 model 이름의 estimator가 들어 있고 fit을 마친 Pipeline. sklearn Pipeline. predict_proba 및 named_steps['model'].classes_가 있어야 한다. |
| `X` | 필수 | `X_test` | 학습된 Pipeline에 같은 열 순서로 전달할 검증/테스트 입력 표. DataFrame (N,F_raw). 이미 전처리한 배열을 Pipeline에 중복 전달하지 않는다. |
| `task` | 필수 | `'binary'` | 정답이 연속값인지, 이진/다중분류/다중라벨인지 문제에서 판별. 'regression'\|'binary'\|'multiclass'\|'multilabel'의 정확한 문자열. 출력·정답·loss 계약이 함께 바뀐다. 모든 함수가 오타를 즉시 검사하는 것은 아니다. |
| `positive_label` | `1` | `'fault'` | 이진 확률 열이 의미해야 할 원래 정답 라벨. 학습된 classes_ 안의 값. 제출 확률 열 선택에 쓰며 훈련 라벨을 자동 0/1 매핑하지 않는다. |
| `class_order` | `None` | `('normal', 'minor', 'major')` | 제출 확률 열의 정해진 클래스 순서. 모든 학습 클래스를 중복 없이 포함한 순서. None은 estimator classes_ 순서. |

**반환값**

binary (N,), multiclass (N,C), multilabel (N,K)의 NumPy 확률 배열.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import pandas as pd
from sklearn.pipeline import Pipeline
X = pd.DataFrame({"speed": [0., 1., 2., 3.]})
y = ["normal", "normal", "fault", "fault"]
pipe = Pipeline([("prep", h.make_preprocessor(X)),
                 ("model", h.make_tabular_estimator("binary", "linear"))])
pipe.fit(X, y)
prob_fault = h.predict_tabular_probabilities(pipe, X, task="binary",
                                            positive_label="fault")
assert prob_fault.shape == (4,)
```

**주의할 점**

- pipe에는 named_steps['model'].classes_가 있어야 한다. raw estimator만 넣으면 안 된다.
- binary positive_label은 실제 classes_에 있어야 한다. None/기본 1이 문자열 'fault'를 대신하지 않는다.
- multiclass class_order는 모든 학습 클래스를 중복 없이 포함해야 한다.
- multilabel은 0/1 indicator이고 학습에서 양성 1이 전혀 없는 열의 확률은 0으로 만든다. 회귀는 지원하지 않는다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `make_point_forecast_windows`

정렬된 한 시계열에서 입력 윈도와 마지막 관측 이후의 정답을 함께 꺼낸다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `features` | 필수 | `series_df[['speed', 'temperature']].to_numpy(dtype=np.float32)` | 정답·ID를 제외한 시간순 원시 특성 행. (T,F), 1D (T,)도 한 특성으로 확장 가능. 시간 축 T는 전체 원시 행 수. 그룹 함수에서는 원래 행 대응 유지. |
| `targets` | 필수 | `series_df[['fuel_use']].to_numpy(dtype=np.float32)` | 각 원시 행 시점의 정답 값 배열. (T,) 또는 (T,D), features와 같은 원시 행 길이. 미래 정답을 helper가 인덱스로 고르므로 이미 앞당긴 y를 다시 horizon만큼 이동하지 않는다. |
| `lookback` | 필수 | `20` | 문제에서 사용할 과거 시간 길이를 샘플 개수로 환산. 양의 정수. 100Hz에서 0.2초를 20개로 해석하는 문제 관례. 초 0.2를 그대로 넣지 않는다. |
| `horizon` | `1` | `40` | 마지막 입력 관측에서 목표까지의 간격을 행 개수로 환산. 0 이상 정수. 100Hz에서 0.4초 뒤는 40칸. 입력 시작이 아니라 마지막 입력에서 센다. |
| `stride` | `1` | `1` | 윈도 시작 위치를 몇 행씩 이동할지 정한 값. 양의 정수. 윈도 helper에서는 초 단위가 아닌 시간순 행 간격. |
| `assume_sorted` | `False` | `True` | 실제로 안정 시간순 정렬·행 대응을 확인한 뒤 명시하는 확인값. bool. True가 정렬을 실행해 주는 것은 아니다. 기본 False인 일반 윈도/분할/Lazy 호출은 오류를 낸다. |

**반환값**

(Xw,yw,target_idx). shape (N,L,F),(N,D),(N,). X/y dtype는 원래 배열을 유지하고 인덱스는 정수.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import numpy as np
X = np.arange(24, dtype=np.float32).reshape(12, 2)
y = np.arange(12, dtype=np.float32)
Xw, yw, target_idx = h.make_point_forecast_windows(
    X, y, lookback=3, horizon=2, stride=1, assume_sorted=True)
assert Xw.shape == (8, 3, 2) and yw.shape == (8, 1)
assert target_idx.tolist() == list(range(4, 12))
assert yw[0, 0] == 4.0
```

**주의할 점**

- 입력 start=s이면 x=features[s:s+L], 정답 위치=s+L−1+H다. target_idx는 정렬된 원시 배열의 위치이다.
- N=max(0,floor((T−L−H)/stride)+1). 길이가 부족하면 올바른 shape의 빈 배열 반환.
- assume_sorted=True는 확인표시이며 정렬 기능은 없다. 다른 그룹을 이어 붙이면 경계를 넘으므로 grouped helper 사용.
- 정답 NaN·Inf를 자동 제거하지 않는다. 사전 유효성 검사와 이후 mask 선택이 필요하다.
- 전체 윈도를 np.stack으로 복제하므로 N×L×F 메모리를 먼저 계산한다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `make_grouped_point_forecast_windows`

각 차량/주행 그룹 안에서만 정렬하고 윈도를 만들어 경계 횡단을 막는다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `features` | 필수 | `series_df[['speed', 'temperature']].to_numpy(dtype=np.float32)` | 정답·ID를 제외한 시간순 원시 특성 행. (T,F), 1D (T,)도 한 특성으로 확장 가능. 시간 축 T는 전체 원시 행 수. 그룹 함수에서는 원래 행 대응 유지. |
| `targets` | 필수 | `series_df[['fuel_use']].to_numpy(dtype=np.float32)` | 각 원시 행 시점의 정답 값 배열. (T,) 또는 (T,D), features와 같은 원시 행 길이. 미래 정답을 helper가 인덱스로 고르므로 이미 앞당긴 y를 다시 horizon만큼 이동하지 않는다. |
| `groups` | 필수 | `train_df['vehicle_id'].to_numpy()` | X/features 각 행에 대응하는 차량·주행·개체 ID 값. 길이 N 또는 T의 1D, 결측 없음. 열 이름 문자열 자체를 넣지 않는다. |
| `lookback` | 필수 | `20` | 문제에서 사용할 과거 시간 길이를 샘플 개수로 환산. 양의 정수. 100Hz에서 0.2초를 20개로 해석하는 문제 관례. 초 0.2를 그대로 넣지 않는다. |
| `horizon` | `1` | `40` | 마지막 입력 관측에서 목표까지의 간격을 행 개수로 환산. 0 이상 정수. 100Hz에서 0.4초 뒤는 40칸. 입력 시작이 아니라 마지막 입력에서 센다. |
| `stride` | `1` | `1` | 윈도 시작 위치를 몇 행씩 이동할지 정한 값. 양의 정수. 윈도 helper에서는 초 단위가 아닌 시간순 행 간격. |
| `times` | `None` | `series_df['time'].to_numpy()` | 각 원시 행에 대응하는 실제 시간 값. 길이 T의 1D, 결측/파싱 실패 없음. 주면 각 그룹 안에서 stable sort한다. |
| `assume_sorted` | `False` | `True` | 실제로 안정 시간순 정렬·행 대응을 확인한 뒤 명시하는 확인값. bool. True가 정렬을 실행해 주는 것은 아니다. 기본 False인 일반 윈도/분할/Lazy 호출은 오류를 낸다. |

**반환값**

(Xw,yw,target_original_indices,window_groups). shape (N,L,F),(N,D),(N,),(N,).

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import numpy as np
import pandas as pd
X = np.arange(24, dtype=np.float32).reshape(12, 2)
y = np.arange(12, dtype=np.float32)
groups = np.array(["A"] * 6 + ["B"] * 6)
times = pd.date_range("2026-01-01", periods=12, freq="s")
Xw, yw, original_target, window_groups = h.make_grouped_point_forecast_windows(
    X, y, groups, lookback=3, horizon=1, times=times)
assert Xw.shape == (6, 3, 2)
assert original_target.tolist() == [3, 4, 5, 9, 10, 11]
assert window_groups.tolist() == ["A", "A", "A", "B", "B", "B"]
```

**주의할 점**

- times를 주면 그룹 내부 stable sort; times가 없으면 실제 그룹 내 정렬을 확인한 assume_sorted=True 필요.
- 출력은 그룹 첫 등장 순서별로 묶인다. 원래 행 순서나 전체 시간순이 아닐 수 있다.
- target_original_indices는 원래 입력의 정답 행 위치이며 DataFrame 인덱스 라벨/입력 끝 위치가 아니다.
- 그룹별 윈도 생성은 train/valid의 그룹 분리를 자동 수행하지 않는다. 새 그룹 일반화라면 window_groups로 별도 분할.
- 짧아서 윈도 없는 그룹은 건너뛰고 모든 그룹이 짧으면 오류. NaN 정답·불규칙 샘플 간격은 별도 처리.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `split_raw_time_then_window`

원시 시계열의 경계와 정답 가용성 기준으로 윈도를 훈련/검증에 배정한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `features` | 필수 | `series_df[['speed', 'temperature']].to_numpy(dtype=np.float32)` | 정답·ID를 제외한 시간순 원시 특성 행. (T,F), 1D (T,)도 한 특성으로 확장 가능. 시간 축 T는 전체 원시 행 수. 그룹 함수에서는 원래 행 대응 유지. |
| `targets` | 필수 | `series_df[['fuel_use']].to_numpy(dtype=np.float32)` | 각 원시 행 시점의 정답 값 배열. (T,) 또는 (T,D), features와 같은 원시 행 길이. 미래 정답을 helper가 인덱스로 고르므로 이미 앞당긴 y를 다시 horizon만큼 이동하지 않는다. |
| `train_ratio` | 필수 | `0.7` | 원시 시계열에서 훈련 쪽 경계를 놓을 비율. 0<값<1. 최종 윈도 행 비율과 다를 수 있다. |
| `lookback` | 필수 | `20` | 문제에서 사용할 과거 시간 길이를 샘플 개수로 환산. 양의 정수. 100Hz에서 0.2초를 20개로 해석하는 문제 관례. 초 0.2를 그대로 넣지 않는다. |
| `horizon` | 필수 | `40` | 마지막 입력 관측에서 목표까지의 간격을 행 개수로 환산. 0 이상 정수. 100Hz에서 0.4초 뒤는 40칸. 입력 시작이 아니라 마지막 입력에서 센다. |
| `stride` | `1` | `1` | 윈도 시작 위치를 몇 행씩 이동할지 정한 값. 양의 정수. 윈도 helper에서는 초 단위가 아닌 시간순 행 간격. |
| `gap` | `0` | `0` | 경계 주변에서 추가로 제외할 원시 행 간격. 0 이상 정수. 학습 경계는 cut−gap, 검증 시작 기준은 cut+gap이므로 양쪽에 적용된다. |
| `assume_sorted` | `False` | `True` | 실제로 안정 시간순 정렬·행 대응을 확인한 뒤 명시하는 확인값. bool. True가 정렬을 실행해 주는 것은 아니다. 기본 False인 일반 윈도/분할/Lazy 호출은 오류를 낸다. |
| `split_mode` | `'forecast_origin'` | `'forecast_origin'` | 실제 평가의 예측 기준이 무엇인지 명세에서 선택. 기본은 미래 예측의 입력 끝 기준. 'target'은 별도 배치 평가 계약용이며 미래 예측 누수를 자동 해결하지 않는다. |
| `label_delay` | `0` | `2` | 정답 시점 이후 실제 정답이 알려지기까지의 추가 지연. 0 이상 정수. horizon과 별개의 원시 행 지연. 불규칙 시간은 timestamp로 별도 검사. |

**반환값**

(train,valid,cut). 각 train/valid는 (Xw,yw,target_idx) tuple, cut은 원시 행 경계 int.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import numpy as np
X = np.arange(40, dtype=np.float32).reshape(20, 2)
y = np.arange(20, dtype=np.float32)
train, valid, cut = h.split_raw_time_then_window(
    X, y, train_ratio=0.6, lookback=3, horizon=2,
    gap=0, label_delay=0, assume_sorted=True)
X_train, y_train, train_target = train
X_valid, y_valid, valid_target = valid
assert cut == 12
assert len(X_train) == 8 and len(X_valid) == 6
assert train_target.max() < (valid_target - 2).min()
```

**주의할 점**

- train 조건은 target_idx+label_delay<cut−gap, 기본 valid 조건은 target_idx−horizon≥cut+gap.
- target 모드는 valid target≥cut+gap이며, 미래 예측 기준에서 훈련 정답이 이미 알려져 있는지는 별도 보장해야 한다.
- 추가 gap=0이어도 horizon/label_delay 때문에 경계 부근 윈도가 제외될 수 있다.
- 그룹 인자를 받지 않는다. 한 시간순 시계열용이며 불규칙 시각은 실제 timestamp 조건이 필요하다.
- 이름과 달리 내부에서는 전체 윈도를 먼저 생성한 뒤 mask로 나눈다. 메모리 절약형이 아니다.
- train 또는 valid가 비면 오류. 전처리 통계를 아직 fit하지 않는다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `fit_scale_3d`

훈련 윈도의 모든 시점을 펼쳐 센서별 평균·표준편차를 학습한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `X_train` | 필수 | `X_train_windows` | 이미 train/valid를 나눈 뒤 훈련 윈도만. 수치 ndarray (N,T,F). 모델 입력에서 F가 마지막 축이어야 한다. |

**반환값**

훈련된 StandardScaler. X 배열 자체는 변환하지 않는다.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import numpy as np
X_train = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
scaler = h.fit_scale_3d(X_train)
assert np.allclose(scaler.mean_, [5.0, 6.0])
```

**주의할 점**

- 입력은 비어 있지 않은 (N,T,F)의 수치 ndarray. scaler는 마지막 F축별 통계를 배운다.
- 중첩 윈도의 같은 원시 관측이 반복 포함되면 통계도 반복 횟수로 가중된다. 원시 행을 한 번씩 fit한 통계와 다를 수 있다.
- 검증·테스트 윈도를 합쳐 fit하지 않는다. NaN/Inf 등 자료 유효성은 호출 전 점검.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `transform_scale_3d`

훈련에서 배운 센서별 기준을 3D 윈도에 적용하고 원래 shape로 돌린다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `X` | 필수 | `X_valid_windows` | 같은 센서 순서의 train/valid/test 윈도. NumPy (N,T,F). scaler가 학습한 F와 센서 순서가 일치해야 한다. |
| `scaler` | 필수 | `x_scaler` | h.fit_scale_3d(X_train)의 반환값. 훈련 특성 F개에 fit된 StandardScaler. 새로 만든 미학습 scaler가 아니다. |

**반환값**

입력과 같은 (N,T,F) shape의 float32 ndarray.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import numpy as np
X_train = np.arange(12, dtype=np.float32).reshape(2, 3, 2)
X_valid = np.array([[[5., 6.], [7., 8.], [9., 10.]]], dtype=np.float32)
scaler = h.fit_scale_3d(X_train)
X_train_s = h.transform_scale_3d(X_train, scaler)
X_valid_s = h.transform_scale_3d(X_valid, scaler)
assert X_valid_s.shape == (1, 3, 2)
assert X_valid_s.dtype == np.float32
```

**주의할 점**

- scaler는 fit_scale_3d로 훈련 F개 특성에 이미 fit되어 있어야 한다.
- 내부에서 3D 자체를 명시 검증하지 않으므로 호출자가 ndim=3과 같은 F축을 확인한다.
- 자료를 정렬/분할하거나 새 평균을 학습하지 않는다. feature 순서가 같아야 한다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `validate_prediction_array`

예측 배열의 정확한 크기·수치형 여부·결측/무한값을 검사한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `pred` | 필수 | `test_pred` | 행 순서·단위·클래스 대응을 복원한 최종 제출 예측. 제출 계약의 shape/dtype. 함수가 의미·순서를 알아서 고치지 않는다. |
| `expected_shape` | 필수 | `(len(test_df), 3)` | 문제의 제출 행 수와 출력 열 수에서 만든 tuple. 예: 한 값 (N,), 3개 값 (N,3). (N,)와 (N,1)은 엄격히 다르다. |
| `allow_nan` | `False` | `False` | 문제에서 결측 허용을 명시했을 때만 변경. bool. 현 코드에서 True이면 NaN뿐 아니라 Inf 검사도 건너뛰므로 일반 제출에는 False 유지. |
| `require_numeric` | `True` | `True` | 제출이 숫자만 허용하는지 확인. bool. False는 문자열 라벨을 허용하지만 열 의미·라벨 집합을 검증하지 않는다. |

**반환값**

np.asarray(pred)의 결과 ndarray. dtype/shape를 임의로 고치지 않는다.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import numpy as np
pred = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
checked = h.validate_prediction_array(pred, expected_shape=(2, 3))
assert checked.shape == (2, 3)
```

**주의할 점**

- 기본은 수치형과 유한성 요구. expected_shape=(N,)와 (N,1)은 다르다.
- allow_nan=True는 Inf까지 검사를 생략하므로 일반 제출에 사용하지 않는다.
- np.number는 complex도 포함하고 bool은 일반 수치형 검사에 통과하지 않는다. 실수·정수·라벨 집합 조건은 추가 검사한다.
- 통과해도 행 순서·열 의미·단위·허용 라벨이 맞다는 증거는 아니다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `save_npy_submission`

예측을 검사하고 NPY로 저장한 다음 다시 읽어 shape·근접한 값을 검사한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `pred` | 필수 | `test_pred` | 행 순서·단위·클래스 대응을 복원한 최종 제출 예측. 제출 계약의 shape/dtype. 함수가 의미·순서를 알아서 고치지 않는다. |
| `path` | 필수 | `'Submission_problem.npy'` | 문제에서 정한 실제 파일명과 저장 위치. str 또는 pathlib.Path. 부모 폴더가 존재해야 하며 기존 같은 경로는 덮어쓴다. |
| `expected_shape` | 필수 | `(len(test_df), 3)` | 문제의 제출 행 수와 출력 열 수에서 만든 tuple. 예: 한 값 (N,), 3개 값 (N,3). (N,)와 (N,1)은 엄격히 다르다. |
| `dtype` | `None` | `np.float32` | 제출 명세의 정확한 원소 자료형. NumPy dtype 또는 None. None은 현재 dtype 유지. float→int는 자르기이므로 라벨 결정 대용이 아니다. |

**반환값**

저장한 ndarray arr. 반환은 다시 읽은 loaded가 아니라 저장 전 검증·형변환 배열.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import numpy as np
from pathlib import Path
from tempfile import TemporaryDirectory
pred = np.array([0.2, 0.8], dtype=np.float64)
with TemporaryDirectory() as folder:  # 예시는 임시 파일만 만든다.
    path = Path(folder) / "Submission_problem.npy"
    saved = h.save_npy_submission(pred, path, expected_shape=(2,), dtype=np.float32)
    loaded = np.load(path, allow_pickle=False)
    assert loaded.dtype == np.float32
    assert np.array_equal(loaded, saved)
```

**주의할 점**

- 연습용 helper이며 시험에 제공한 저장 셀이 있으면 그것을 따른다.
- 명시 dtype 변환 전후 유한성을 검사하므로 float32 overflow를 잡는다. 그러나 실수→정수 자르기의 의미 오류는 못 잡는다.
- 확장자가 정확히 .npy가 아니면 .npy를 뒤에 붙인다. 기존 파일은 덮어쓰며 폴더는 만들지 않는다.
- 재읽기는 np.allclose 기반이고 dtype 동일성/정확한 bitwise equality를 별도 확인하지 않는다.
- NPY 저장이 노트북 저장·실제 제출 버튼 완료를 대신하지 않는다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `save_csv_submission`

제출 예시 표의 타깃 열에 예측을 넣고 CSV 저장·재읽기를 검사한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `sample_submission` | 필수 | `sample_submission_df` | 제공된 제출 예시 파일을 읽은 원래 표. DataFrame. ID·행 순서·출력 열이 기준이며 pred를 ID로 자동 join하지 않는다. |
| `pred` | 필수 | `test_pred` | 행 순서·단위·클래스 대응을 복원한 최종 제출 예측. 제출 계약의 shape/dtype. 함수가 의미·순서를 알아서 고치지 않는다. |
| `target_cols` | 필수 | `('fuel_use',)` | 문제 설명의 정답 열 이름과 출력 순서. 문자열의 list/tuple. 한 열 tuple에도 쉼표가 필요하다. ('fuel_use')는 문자열이다. |
| `path` | 필수 | `'Submission.csv'` | 문제의 CSV 제출 파일명과 저장 위치. str 또는 Path. 부모 폴더가 있어야 하며 같은 경로는 덮어쓴다. |

**반환값**

저장 전 작성한 DataFrame out. 원본 sample_submission은 복사하여 보존.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import numpy as np
import pandas as pd
from pathlib import Path
from tempfile import TemporaryDirectory
sample = pd.DataFrame({"id": [101, 102], "fault": [0, 0]})
pred = np.array([1, 0], dtype=np.int64)  # 101,102 행 순서
with TemporaryDirectory() as folder:
    path = Path(folder) / "Submission.csv"
    out = h.save_csv_submission(sample, pred, target_cols=["fault"], path=path)
    loaded = pd.read_csv(path)
    assert loaded["id"].tolist() == [101, 102]
    assert loaded["fault"].tolist() == [1, 0]
```

**주의할 점**

- pred는 sample_submission의 현재 행 순서와 이미 대응해야 한다. ID로 자동 정렬/join하지 않는다.
- 한 열이면 (N,) 또는 (N,1), 여러 열이면 정확히 (N,D). 문자열 라벨도 지원하지만 허용 라벨 집합 검사는 없다.
- NaN/숫자 Inf와 재읽은 열/행 수를 검사한다. 전체 ID 값·dtype·타깃 수치 일치까지 검증하지 않는다.
- CSV 재읽기에서 'NA' 같은 문자열이나 앞자리 0인 ID가 다르게 해석될 수 있으므로 명세에 맞춘 추가 대조 필요.
- index=False로 저장하며 기존 같은 파일을 덮어쓴다. 실제 제출 상태는 별도 확인한다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `prepare_numpy_images`

명시한 이미지 배열 layout을 NCHW float32로 변환하고 필요하면 255로 나눈다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `X` | 필수 | `images` | 파일/배열을 읽고 이미지 batch로 모은 실제 픽셀 배열. 3D (N,H,W) 흑백 batch 또는 layout에 맞는 4D batch. 단일 RGB HWC는 먼저 배치 축 추가. |
| `layout` | 필수 | `'NHWC'` | 이미지 배열의 실제 축 순서를 shape와 자료 설명으로 확인. 'NHWC'\|'NCHW' 필수. 3D 입력은 단일 RGB가 아니라 흑백 배치 (N,H,W)로 해석한다. |
| `divide_255` | 필수 | `True` | 입력 픽셀이 아직 0..255 척도인지 실제 값으로 확인. bool 필수. 이미 0..1 또는 표준화한 값이면 False. float dtype 자체는 이미 나눴다는 증거가 아니다. |

**반환값**

NumPy float32 배열 (N,C,H,W), C는 1·3·4 중 하나.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import numpy as np
images = np.zeros((2, 16, 20, 3), dtype=np.uint8)  # N,H,W,C
images[..., 0] = 255
prepared = h.prepare_numpy_images(images, layout="NHWC", divide_255=True)
assert prepared.shape == (2, 3, 16, 20)
assert prepared.dtype == np.float32
assert prepared[:, 0].max() == 1.0
```

**주의할 점**

- layout과 divide_255에 기본값이 없다. 실제 축 순서·픽셀 척도를 보고 둘 다 지정한다.
- 3D는 흑백 배치 (N,H,W)로 해석한다. 단일 RGB (H,W,3)에는 X[None,...]로 배치 축을 먼저 추가한다.
- RGB 변환·알파 제거·resize·채널별 mean/std 정규화·증강은 하지 않는다.
- 입력이 float라도 여전히 0..255일 수 있다. 이미 0..1인 배열을 다시 나누지 않는다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `print_shapes`

여러 배열을 이름과 함께 shape·dtype로 출력한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `**arrays` | `{}` | `X_train=X_train_s, y_train=y_train, pred=test_pred` | 이름을 붙여 점검할 현재 배열들. 키워드 인자 여러 개. NumPy로 변환할 수 있는 배열. GPU Tensor·dict/sparse 등은 따로 확인. |

**반환값**

None. 화면에 점검 문자열을 출력.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다.

```python
import hdat_templates as h
import numpy as np
X_train = np.zeros((8, 6), dtype=np.float32)
y_train = np.zeros((8, 1), dtype=np.float32)
pred = np.zeros((3, 1), dtype=np.float32)
h.print_shapes(X_train=X_train, y_train=y_train, pred=pred)
```

**주의할 점**

- 키워드 이름은 설명용으로 원하는 이름을 붙인다. h.print_shapes(X_train=X)는 문자열 'X_train'을 배열 이름으로 출력.
- 검사 실패를 raise하지 않고 유한성/축 의미/행 대응을 검증하지 않는다.
- GPU Tensor는 먼저 detach().cpu().numpy() 등으로 변환해야 한다.

구현 확인: [hdat_templates.py](./hdat_templates.py).

## 08. 모델을 만드는 인자 사전

모델은 이름만 고르는 것이 아니라 데이터에 맞는 크기로 만들어야 합니다. 아래 예제의 model(x)는 출력 모양만 확인합니다. 학습이 끝났다는 뜻은 아닙니다. 지정 구조 문제에서는 이 기본 모델과 문제의 층 구성이 같은지 별도로 대조하세요.

#### `MLP`

표 데이터 또는 평탄화한 한 샘플을 입력받는 신경망을 만든다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `n_features` | 필수 | `X_train_s.shape[1]` | 전처리된 2D 입력 폭. 3D 이상을 넣으면 한 샘플 flatten 후의 원소 수. 2D (N,F)면 F. (N,T,F)를 직접 넣으면 T×F로 지정해야 한다. |
| `out_dim` | 필수 | `1` | 문제의 출력 계약에서 결정한 출력 수. 회귀 D, 이진 1, 다중분류 C, 다중라벨 K. 정답의 원소 수 N이 아니다. |
| `hidden` | `(128, 64)` | `(128, 64)` | 문제 명세 또는 작은 검증에서 고른 MLP 은닉층 폭. 양의 정수 시퀀스. (64,)가 한 층이며 64 단독은 시퀀스가 아니다. 빈 tuple은 Linear 한 층. |
| `dropout` | `0.1` | `0.1` | 문제 명세 또는 검증에서 정한 drop 비율. 0≤p≤1 범위 실수. 보통 0.1은 10%이며 10을 넣지 않는다. |

**forward 호출 인자** — `model(x)`

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `x` | 필수 | `xb` | loader가 반환한 현재 배치의 입력 Tensor. float32 Tensor (B,F). 3D 이상이면 flatten(1) 후 폭이 n_features와 같아야 한다. |

반환: 생성: MLP 인스턴스. model(xb): float Tensor (B,out_dim).

**반환값**

생성: MLP 인스턴스. model(xb): float Tensor (B,out_dim).

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다. PyTorch가 설치되어 있어야 이 API가 정의된다.

```python
import hdat_templates as h
import numpy as np
import torch
X = np.zeros((4, 6), dtype=np.float32)  # 샘플 4개, 전처리된 특성 6개
model = h.MLP(n_features=X.shape[1], out_dim=1, hidden=(16, 8))
raw = model(torch.as_tensor(X))
assert raw.shape == (4, 1)
```

**주의할 점**

- n_features는 전처리 이후의 폭이다. 원-핫 표현 후에는 원본 열 개수와 다르다.
- 3D 이상 x는 flatten(1)하므로 시계열 MLP라면 n_features=T×F이어야 한다.
- 마지막에 sigmoid/softmax가 없다. 회귀 연속값·분류 raw logits로 해석한다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `CNN1D`

시간 윈도 (B,T,F)를 받아 시간축 합성곱으로 예측한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `n_features` | 필수 | `X_train_s.shape[-1]` | 전처리가 끝난 실제 모델 입력 배열의 특성 수. 양의 정수. MLP 2D면 shape[1]; CNN1D/RNN/Transformer (N,T,F)면 shape[2]. 원본 CSV 전체 열 수나 배치 수가 아니다. |
| `out_dim` | 필수 | `1` | 문제의 출력 계약에서 결정한 출력 수. 회귀 D, 이진 1, 다중분류 C, 다중라벨 K. 정답의 원소 수 N이 아니다. |
| `channels` | `(64, 128)` | `(64, 128)` | 문제 명세 또는 검증에서 정한 CNN1D 두 층 채널 수. 양의 정수 정확히 2개. 센서 개수 F와는 별개의 학습 특징 채널. |
| `dropout` | `0.1` | `0.1` | 문제 명세 또는 검증에서 정한 drop 비율. 0≤p≤1 범위 실수. 보통 0.1은 10%이며 10을 넣지 않는다. |

**forward 호출 인자** — `model(x)`

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `x` | 필수 | `xb` | loader가 반환한 현재 배치의 입력 Tensor. float32 Tensor (B,T,F). 내부에서 (B,F,T)로 전치하므로 미리 전치하지 않는다. |

반환: 생성: CNN1D 인스턴스. model(xb): (B,out_dim) 실수 Tensor.

**반환값**

생성: CNN1D 인스턴스. model(xb): (B,out_dim) 실수 Tensor.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다. PyTorch가 설치되어 있어야 이 API가 정의된다.

```python
import hdat_templates as h
import numpy as np
import torch
X = np.zeros((4, 20, 6), dtype=np.float32)
model = h.CNN1D(n_features=X.shape[-1], out_dim=3, channels=(8, 16))
raw = model(torch.as_tensor(X))
assert raw.shape == (4, 3)
```

**주의할 점**

- forward 안에서 (B,T,F)→(B,F,T)로 전치한다. 호출 전에 이미 전치하면 축을 다시 뒤집게 된다.
- channels는 정확히 두 값. kernel/normalization 구조는 고정되어 있어 명세형 Process 문제에 무수정 사용하면 안 된다.
- 전역 평균 풀링을 사용하므로 마지막 시점 정보를 특별히 강조하는 구조는 아니다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `SequenceRNN`

고정 길이 시계열을 GRU/LSTM으로 읽고 마지막 층 은닉 상태로 한 번 예측한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `n_features` | 필수 | `X_train_s.shape[-1]` | 전처리가 끝난 실제 모델 입력 배열의 특성 수. 양의 정수. MLP 2D면 shape[1]; CNN1D/RNN/Transformer (N,T,F)면 shape[2]. 원본 CSV 전체 열 수나 배치 수가 아니다. |
| `out_dim` | 필수 | `1` | 문제의 출력 계약에서 결정한 출력 수. 회귀 D, 이진 1, 다중분류 C, 다중라벨 K. 정답의 원소 수 N이 아니다. |
| `hidden_size` | `64` | `64` | RNN 한 방향의 은닉 상태 폭. 양의 정수. bidirectional=True면 최종 표현 폭은 2×hidden_size. |
| `num_layers` | `1` | `2` | 겹칠 순환층 또는 Transformer encoder 층 수. 양의 정수. 여러 시점 T와 다르다. |
| `kind` | `'gru'` | `'lstm'` | 문제 명세가 요구한 순환 셀 종류. 'gru'\|'lstm'만 가능. |
| `bidirectional` | `False` | `False` | 전체 입력 구간을 양방향으로 읽어도 되는 과제인지 선택. bool. 같은 입력 구간 안 과거·역방향을 읽는다. 실시간 시점별 미래 차단을 자동 보장하지 않는다. |
| `dropout` | `0.0` | `0.1` | 문제 명세 또는 검증에서 정한 drop 비율. 0≤p≤1 범위 실수. 보통 0.1은 10%이며 10을 넣지 않는다. |

**forward 호출 인자** — `model(x)`

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `x` | 필수 | `xb` | loader가 반환한 현재 배치의 입력 Tensor. float32 Tensor (B,T,F), 패딩 없는 고정 길이 입력. |

반환: 생성: SequenceRNN 인스턴스. model(xb): (B,out_dim) 실수 Tensor.

**반환값**

생성: SequenceRNN 인스턴스. model(xb): (B,out_dim) 실수 Tensor.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다. PyTorch가 설치되어 있어야 이 API가 정의된다.

```python
import hdat_templates as h
import numpy as np
import torch
X = np.zeros((4, 20, 6), dtype=np.float32)
model = h.SequenceRNN(n_features=X.shape[-1], out_dim=1,
                    hidden_size=8, num_layers=2, kind="lstm",
                    bidirectional=False, dropout=0.1)
raw = model(torch.as_tensor(X))
assert raw.shape == (4, 1)
```

**주의할 점**

- n_features=입력 F, hidden_size=한 방향 상태 폭. 양방향은 마지막 층의 두 방향 상태를 연결한다.
- num_layers=1이면 전달한 dropout도 내부에서 0으로 바뀐다.
- 패딩/가변 길이 packing을 지원하지 않는다. 마지막 패딩이 들어간 batch에 그대로 사용하지 않는다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `SmallImageCNN`

NCHW 이미지 배치를 읽어 샘플별 연속값 또는 클래스 점수를 출력한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `in_channels` | 필수 | `X_images.shape[1]` | NCHW 입력 텐서의 실제 채널 축 크기. 양의 정수. RGB 3, 흑백 1. H/W나 배치 수가 아니다. |
| `out_dim` | 필수 | `1` | 문제의 출력 계약에서 결정한 출력 수. 회귀 D, 이진 1, 다중분류 C, 다중라벨 K. 정답의 원소 수 N이 아니다. |

**forward 호출 인자** — `model(x)`

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `x` | 필수 | `xb` | loader가 반환한 현재 배치의 입력 Tensor. float32 Tensor (B,in_channels,H,W), H/W 각각 최소 4. |

반환: 생성: SmallImageCNN 인스턴스. model(xb): (B,out_dim) float Tensor.

**반환값**

생성: SmallImageCNN 인스턴스. model(xb): (B,out_dim) float Tensor.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다. PyTorch가 설치되어 있어야 이 API가 정의된다.

```python
import hdat_templates as h
import numpy as np
import torch
images = np.zeros((2, 3, 16, 20), dtype=np.float32)  # 이미 NCHW
model = h.SmallImageCNN(in_channels=images.shape[1], out_dim=4)
raw = model(torch.as_tensor(images))
assert raw.shape == (2, 4)
```

**주의할 점**

- in_channels는 준비된 배열 shape[1]에서 확인한다. RGB=3, 흑백=1.
- MaxPool2d(2)가 두 번 있어 H/W는 최소 4여야 한다. 'adaptive이므로 어떤 양의 크기든 가능'은 틀리다.
- 다른 크기의 이미지는 한 일반 batch로 stack하기 전에 크기를 맞춰야 한다.
- 마지막 sigmoid/softmax 없음. 강제된 CNN 명세라면 구조를 그대로 사용하지 않는다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `ResidualBlock2D`

주 분기와 지름길을 같은 크기로 맞춰 더하는 2D 특징 블록을 만든다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `in_channels` | 필수 | `X_images.shape[1]` | NCHW 입력 텐서의 실제 채널 축 크기. 양의 정수. RGB 3, 흑백 1. H/W나 배치 수가 아니다. |
| `out_channels` | 필수 | `64` | 다음 잔차 블록이 만들 특징 채널 수. 양의 정수. 분류 클래스 수가 아닌 내부 특징 채널. |
| `stride` | `1` | `2` | 공간 해상도를 유지할지 줄일지 정한 블록 설정. 양의 정수. 1이면 해상도 유지, 2이면 대략 절반이며 두 분기에 함께 적용된다. |

**forward 호출 인자** — `model(x)`

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `x` | 필수 | `xb` | loader가 반환한 현재 배치의 입력 Tensor. float32 Tensor (B,in_channels,H,W), BN 훈련의 충분한 표본/공간 크기 확인. |

반환: 생성: ResidualBlock2D. model(xb): (B,out_channels,ceil(H/stride),ceil(W/stride)).

**반환값**

생성: ResidualBlock2D. model(xb): (B,out_channels,ceil(H/stride),ceil(W/stride)).

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다. PyTorch가 설치되어 있어야 이 API가 정의된다.

```python
import hdat_templates as h
import torch
x = torch.zeros(2, 3, 16, 20)
block = h.ResidualBlock2D(in_channels=x.shape[1], out_channels=8, stride=2)
features = block(x)
assert features.shape == (2, 8, 8, 10)
```

**주의할 점**

- 최종 분류기가 아니라 중간 특징 블록이므로 out_channels를 클래스 수로 혼동하지 않는다.
- 채널 또는 stride가 바뀌면 1×1 projection이 생긴다. stride는 양의 정수.
- BN 훈련은 채널당 통계가 가능한 값 개수가 필요하다. 단일 B=1,H=W=1은 별도 eval 테스트가 필요하다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `TransformerSequenceModel`

시퀀스 특성을 내부 폭으로 투영하고 위치 정보·자기어텐션·평균 풀링으로 예측한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `n_features` | 필수 | `X_train_s.shape[-1]` | 전처리가 끝난 실제 모델 입력 배열의 특성 수. 양의 정수. MLP 2D면 shape[1]; CNN1D/RNN/Transformer (N,T,F)면 shape[2]. 원본 CSV 전체 열 수나 배치 수가 아니다. |
| `out_dim` | 필수 | `1` | 문제의 출력 계약에서 결정한 출력 수. 회귀 D, 이진 1, 다중분류 C, 다중라벨 K. 정답의 원소 수 N이 아니다. |
| `d_model` | `64` | `64` | 입력 특성을 투영할 Transformer 내부 폭. 양의 정수, nhead로 나누어떨어져야 한다. |
| `nhead` | `4` | `4` | 어텐션 head 개수. 양의 정수. d_model%nhead==0; 출력 클래스 수와 무관. |
| `num_layers` | `2` | `2` | 겹칠 순환층 또는 Transformer encoder 층 수. 양의 정수. 여러 시점 T와 다르다. |
| `dim_feedforward` | `128` | `128` | Transformer 내부 MLP의 은닉 폭. 양의 정수. 입력 F·시퀀스 T와 별도 설정. |
| `dropout` | `0.1` | `0.1` | 문제 명세 또는 검증에서 정한 drop 비율. 0≤p≤1 범위 실수. 보통 0.1은 10%이며 10을 넣지 않는다. |
| `max_len` | `512` | `512` | 허용할 가장 긴 입력 시퀀스 길이. 양의 정수이며 모든 배치 T≤max_len. 위치 파라미터 크기를 정한다. |

**forward 호출 인자** — `model(x, padding_mask=padding_mask)`

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `x` | 필수 | `xb` | loader가 반환한 현재 배치의 입력 Tensor. float32 Tensor (B,T,n_features), 1≤T≤max_len. |
| `padding_mask` | `None` | `torch.tensor([[False, False, True]])` | 실제 길이와 padding 위치로 만든 마스크. bool (B,T), True가 무시할 padding. x와 같은 device. 샘플마다 하나 이상 False 필요. |

반환: 생성: TransformerSequenceModel. model(x,padding_mask=None): (B,out_dim) 실수 Tensor.

**반환값**

생성: TransformerSequenceModel. model(x,padding_mask=None): (B,out_dim) 실수 Tensor.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다. PyTorch가 설치되어 있어야 이 API가 정의된다.

```python
import hdat_templates as h
import torch
x = torch.zeros(2, 5, 3)
padding_mask = torch.tensor([[False, False, False, False, False],
                             [False, False, False, True, True]])
model = h.TransformerSequenceModel(n_features=x.shape[-1], out_dim=1,
                                  d_model=8, nhead=2, num_layers=1,
                                  dim_feedforward=16, max_len=5)
raw = model(x, padding_mask=padding_mask)
assert raw.shape == (2, 1)
```

**주의할 점**

- forward x는 (B,T,F), padding_mask는 선택 bool (B,T): True가 무시할 패딩, False가 실제 관측.
- d_model은 nhead로 나누어져야 하고 T≤max_len. 각 샘플에 실제 관측이 적어도 하나는 있어야 한다.
- padding_mask는 인과 마스크가 아니다. 전체 입력 구간 내부를 양방향으로 보며 토큰별 미래 차단 기능은 없다.
- 공통 train_torch_model은 model(xb)만 호출하므로 mask가 필요한 가변 길이 배치에는 전용 루프를 사용한다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `Autoencoder`

입력을 작은 잠재 벡터로 줄인 뒤 원래 특성으로 복원한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `input_dim` | 필수 | `X_train_flat.shape[1]` | 평탄화·전처리한 한 샘플의 입력 특성 수. 양의 정수 F. AE/VAE/Discriminator는 기본 (B,F) 입력이며 자동 flatten하지 않는다. |
| `latent_dim` | `16` | `8` | 검증 또는 문제 명세로 정한 잠재 표현의 폭. 양의 정수. 데이터 클래스 수·원시 행 수에서 자동 계산하는 값이 아니다. |
| `hidden_dim` | `64` | `64` | 문제 명세 또는 작은 검증으로 고른 내부 은닉 폭. 양의 정수. Discriminator는 hidden_dim//2 층을 쓰므로 최소 2 권장. |

**forward 호출 인자** — `model(x)`

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `x` | 필수 | `xb` | loader가 반환한 현재 배치의 입력 Tensor. float32 Tensor (B,input_dim). 입력을 미리 평탄화한다. |

반환: 생성: Autoencoder 인스턴스. model(xb): xb와 같은 (B,input_dim) 복원 Tensor.

**반환값**

생성: Autoencoder 인스턴스. model(xb): xb와 같은 (B,input_dim) 복원 Tensor.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다. PyTorch가 설치되어 있어야 이 API가 정의된다.

```python
import hdat_templates as h
import numpy as np
import torch
X = np.array([[0., 1., 2.], [1., 2., 3.]], dtype=np.float32)
model = h.Autoencoder(input_dim=X.shape[1], latent_dim=2, hidden_dim=8)
reconstruction = model(torch.as_tensor(X))
assert reconstruction.shape == X.shape
```

**주의할 점**

- 기본 사용은 2D (B,F); 이미지/윈도는 호출 전에 한 샘플씩 평탄화한다.
- 공통 학습 루프를 쓰려면 task='regression', loader의 y=X로 같은 복원 목표를 만들어야 한다.
- h.predict_torch(task='regression')의 결과는 이상 라벨이 아니라 복원값이다. 샘플별 오차와 정상 검증 임계값은 별도 계산한다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `VariationalAutoencoder`

입력별 잠재 평균·로그분산을 만들고 재매개화한 잠재 변수로 복원한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `input_dim` | 필수 | `X_train_flat.shape[1]` | 평탄화·전처리한 한 샘플의 입력 특성 수. 양의 정수 F. AE/VAE/Discriminator는 기본 (B,F) 입력이며 자동 flatten하지 않는다. |
| `latent_dim` | `8` | `8` | 검증 또는 문제 명세로 정한 잠재 표현의 폭. 양의 정수. 데이터 클래스 수·원시 행 수에서 자동 계산하는 값이 아니다. |
| `hidden_dim` | `64` | `64` | 문제 명세 또는 작은 검증으로 고른 내부 은닉 폭. 양의 정수. Discriminator는 hidden_dim//2 층을 쓰므로 최소 2 권장. |

**reparameterize 호출 인자**

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `mu` | 필수 | `mu` | VAE forward가 반환한 잠재 평균. (B,Z) 실수 Tensor. 모델 객체·원시 입력 평균이 아니다. |
| `logvar` | 필수 | `logvar` | VAE forward가 반환한 잠재 로그분산. mu와 같은 (B,Z) 실수 Tensor. 분산/표준편차를 그대로 넣지 않는다. |

반환: (B,Z) Tensor. train은 잡음 표본, eval은 mu.

**forward 호출 인자** — `model(x)`

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `x` | 필수 | `xb` | loader가 반환한 현재 배치의 입력 Tensor. float32 Tensor (B,input_dim). 입력을 미리 평탄화한다. |

반환: 생성: VariationalAutoencoder 인스턴스. model(xb)는 (reconstruction,mu,logvar) tuple; shape는 (B,F),(B,Z),(B,Z).

**반환값**

생성: VariationalAutoencoder 인스턴스. model(xb)는 (reconstruction,mu,logvar) tuple; shape는 (B,F),(B,Z),(B,Z).

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다. PyTorch가 설치되어 있어야 이 API가 정의된다.

```python
import hdat_templates as h
import numpy as np
import torch
X = np.zeros((4, 6), dtype=np.float32)
model = h.VariationalAutoencoder(input_dim=X.shape[1], latent_dim=2, hidden_dim=8)
model.eval()
with torch.inference_mode():
    reconstruction, mu, logvar = model(torch.as_tensor(X))
assert reconstruction.shape == (4, 6)
assert mu.shape == logvar.shape == (4, 2)
```

**주의할 점**

- train 모드의 reparameterize(mu,logvar)는 mu+exp(logvar/2)×무작위 잡음, eval 모드는 mu를 반환한다.
- 공통 train_torch_model/predict_torch는 Tensor 하나를 기대하므로 tuple 출력 VAE에 그대로 쓸 수 없다.
- vae_loss와 전용 optimizer 루프를 쓴다. eval의 평균 잠재 복원은 Monte Carlo ELBO 평가와 다르다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `vae_loss`

VAE의 복원 오차와 KL 규제를 합친 학습 손실을 계산한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `reconstruction` | 필수 | `reconstruction` | VAE forward가 반환한 복원 텐서. x와 같은 (B,F), 실수 Tensor. tuple 전체를 넣지 않는다. |
| `x` | 필수 | `xb` | 현재 배치의 입력 텐서. 실수 Tensor. VAE loss에서는 reconstruction과 같은 (B,F), 같은 device. |
| `mu` | 필수 | `mu` | VAE forward가 반환한 잠재 평균. (B,Z) 실수 Tensor. 모델 객체·원시 입력 평균이 아니다. |
| `logvar` | 필수 | `logvar` | VAE forward가 반환한 잠재 로그분산. mu와 같은 (B,Z) 실수 Tensor. 분산/표준편차를 그대로 넣지 않는다. |
| `beta` | `1.0` | `1.0` | 복원 손실 대비 KL 항의 검증용 비중. 실수 스칼라. 이 helper는 두 항을 각각 모든 원소 평균하므로 교재의 잠재 차원 합 ELBO와 스케일이 다르다. |

**반환값**

(total_loss,reconstruction_loss,kl)의 scalar Tensor 3개. total_loss.backward() 대상으로 첫 항목을 사용.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다. PyTorch가 설치되어 있어야 이 API가 정의된다.

```python
import hdat_templates as h
import torch
x = torch.tensor([[0., 1.], [2., 3.]])
model = h.VariationalAutoencoder(input_dim=x.shape[1], latent_dim=1, hidden_dim=8)
reconstruction, mu, logvar = model(x)
total, reconstruction_loss, kl = h.vae_loss(reconstruction, x, mu, logvar, beta=1.0)
total.backward()
assert total.ndim == reconstruction_loss.ndim == kl.ndim == 0
```

**주의할 점**

- reconstruction/x는 같은 (B,F), mu/logvar는 같은 (B,Z), 모두 같은 장치의 실수 Tensor.
- 복원 MSE도 KL도 모든 원소에 대한 평균이다. KL을 Z축 합하고 B축 평균하는 교재식과 beta의 상대 스케일이 다르다.
- 반환 tuple 전체를 backward하거나 공통 loss_fn으로 그대로 전달하지 않는다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `MLPGenerator`

잡음 벡터에서 표형/평탄화 샘플을 생성한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `noise_dim` | 필수 | `16` | 생성기에 입력할 표준정규 잡음 벡터의 폭. 양의 정수 Z. z=torch.randn(B, noise_dim)의 마지막 축. |
| `output_dim` | 필수 | `X_train_flat.shape[1]` | 생성기가 만들 실제 샘플 한 개의 특성 수. 양의 정수 F. 원자료·Discriminator input_dim과 맞춰야 한다. |
| `hidden_dim` | `128` | `64` | 문제 명세 또는 작은 검증으로 고른 내부 은닉 폭. 양의 정수. Discriminator는 hidden_dim//2 층을 쓰므로 최소 2 권장. |

**forward 호출 인자** — `generator(z)`

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `z` | 필수 | `torch.randn(4,3)` | torch.randn(batch_size,noise_dim)로 만든 잡음. float32 Tensor (B,noise_dim), 모델과 같은 장치의 무작위 잡음. |

반환: 생성: MLPGenerator 인스턴스. G(z): (B,output_dim) 실수 Tensor.

**반환값**

생성: MLPGenerator 인스턴스. G(z): (B,output_dim) 실수 Tensor.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다. PyTorch가 설치되어 있어야 이 API가 정의된다.

```python
import hdat_templates as h
import numpy as np
import torch
real = np.zeros((4, 6), dtype=np.float32)
noise_dim = 3
generator = h.MLPGenerator(noise_dim=noise_dim, output_dim=real.shape[1], hidden_dim=8)
z = torch.randn(len(real), noise_dim)
fake = generator(z)
assert fake.shape == (4, 6)
```

**주의할 점**

- noise_dim은 설계한 잡음 폭이고 output_dim은 실제 한 샘플의 특성 수다.
- 마지막 활성함수가 없으므로 출력 범위를 0..1로 제한하지 않는다. 실제 자료 전처리와 생성 범위를 맞춰야 한다.
- GAN 교대 학습의 G/D 목적과 detach는 별도 루프로 작성한다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `MLPDiscriminator`

한 샘플이 진짜인지 가짜인지 판별할 로짓을 만든다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `input_dim` | 필수 | `X_train_flat.shape[1]` | 평탄화·전처리한 한 샘플의 입력 특성 수. 양의 정수 F. AE/VAE/Discriminator는 기본 (B,F) 입력이며 자동 flatten하지 않는다. |
| `hidden_dim` | `128` | `64` | 문제 명세 또는 작은 검증으로 고른 내부 은닉 폭. 양의 정수. Discriminator는 hidden_dim//2 층을 쓰므로 최소 2 권장. |

**forward 호출 인자** — `model(x)`

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `x` | 필수 | `xb` | loader가 반환한 현재 배치의 입력 Tensor. float32 Tensor (B,input_dim), 실제 자료 또는 생성기 출력. |

반환: 생성: MLPDiscriminator 인스턴스. D(xb): (B,1) raw logit Tensor.

**반환값**

생성: MLPDiscriminator 인스턴스. D(xb): (B,1) raw logit Tensor.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다. PyTorch가 설치되어 있어야 이 API가 정의된다.

```python
import hdat_templates as h
import numpy as np
import torch
real = np.zeros((4, 6), dtype=np.float32)
discriminator = h.MLPDiscriminator(input_dim=real.shape[1], hidden_dim=8)
logits = discriminator(torch.as_tensor(real))
assert logits.shape == (4, 1)
```

**주의할 점**

- input_dim은 실제 자료와 G 출력의 폭 F이다. 이진 판정이므로 input_dim=1이 되는 것은 아니다.
- BCEWithLogitsLoss 앞에서 sigmoid를 추가하지 않는다.
- hidden_dim//2 층이 있으므로 hidden_dim은 2 이상을 사용한다. 공통 지도학습 루프는 GAN의 교대 갱신을 대신하지 않는다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


## 09. loader·학습·예측의 모든 인자

가장 많이 쓰는 네 함수의 정확한 호출법입니다. 인자 표에서 필수 여부와 기본값을 확인한 뒤 실제 값을 만드는 예제로 내려가세요. 다른 코드의 model_factory·config·metric 인자를 이 파일의 학습 함수와 섞지 마세요.

#### `train_torch_model`

이미 만든 모델을 훈련하고 검증 기준이 가장 좋았던 가중치로 복원한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `model` | 필수 | `h.MLP(n_features=X_train_s.shape[1], out_dim=1)` | 모델 생성자를 호출해 만든 실제 신경망 인스턴스. torch.nn.Module 객체. 클래스 MLP 자체, 'MLP' 문자열, lambda/model_factory를 그대로 넣지 않는다. |
| `train_loader` | 필수 | `h.make_tensor_loader(X_train_s, y_train, task='regression', shuffle=True)` | 정답을 포함한 훈련 Dataset으로 만든 DataLoader. 각 배치 (xb,yb) 정확히 2개. 비어 있지 않음. 입력·정답 축·dtype가 task와 일치. |
| `valid_loader` | 필수 | `h.make_tensor_loader(X_valid_s, y_valid, task='regression', shuffle=False)` | 훈련에 포함되지 않은 검증 Dataset의 DataLoader. 각 배치 (xb,yb), 비어 있지 않음, drop_last=False. 마스크 등 3개 배치는 공통 루프가 지원하지 않는다. |
| `task` | 필수 | `'binary'` | 정답이 연속값인지, 이진/다중분류/다중라벨인지 문제에서 판별. 'regression'\|'binary'\|'multiclass'\|'multilabel'의 정확한 문자열. 출력·정답·loss 계약이 함께 바뀐다. 모든 함수가 오타를 즉시 검사하는 것은 아니다. |
| `epochs` | `30` | `30` | 최대 학습 반복 횟수: 문제 시간·검증으로 정한 상한. 1 이상 정수. 0이면 정상 checkpoint가 없어 오류. 조기 종료로 실제 횟수는 작을 수 있다. |
| `lr` | `0.001` | `1e-3` | 학습률: 작은 시험 실행과 검증에서 결정. 양의 실수. 1e-3은 0.001. 문자열이 아니며 값이 클수록 항상 좋은 것은 아니다. |
| `weight_decay` | `0.0001` | `1e-4` | AdamW의 가중치 감쇠 비중. 0 이상 실수. 학습률·Dropout 비율과 다른 설정. |
| `patience` | `5` | `5` | 검증 기준이 연속 몇 회 개선되지 않으면 멈출지. 1 이상 정수 권장. 시간 초 단위가 아니다. |
| `max_seconds` | `None` | `120.0` | 이번 함수 호출의 학습 부분에 배정할 초 단위 예산. 양수 또는 None. None은 시간 제한 없음. 데이터 준비·모델 생성은 제외하며 검증 완료까지 초과할 수 있다. |
| `pos_weight` | `None` | `[9.0]` | 훈련 폴드의 음성 수/양성 수로 정한 BCE 양성 항 가중치. 이진 길이 1 또는 스칼라, 다중라벨 길이 K. 양성 0개는 별도 처리. 테스트/검증 라벨로 계산하지 않는다. |
| `class_weight` | `None` | `[0.5, 1.0, 2.0]` | 훈련 폴드에서 클래스 0..C−1 순서로 계산한 CE 가중치. 길이 C의 유한한 실수, 통상 양수. BCE용 pos_weight와 다른 정의. |
| `grad_clip` | `1.0` | `1.0` | 한 배치 역전파 후 gradient 전체 norm의 상한. 양의 실수 권장. 0이면 기울기를 사실상 0으로 만들 수 있다. |
| `device` | `None` | `'cpu'` | 현재 실제로 사용 가능한 계산 장치. None은 CUDA 가능 시 cuda, 아니면 cpu. 'mps'는 지원 여부를 확인하고 명시해야 하며 자동 선택되지 않는다. |
| `score_fn` | `None` | `lambda yt, raw: h.evaluate_predictions(yt, raw, 'rmse')` | 공식 지표 계산법을 함수로 작성한 콜백. callable(y_true_numpy, raw_output_numpy)->유한 float. sigmoid/softmax·임계값·원단위 역변환을 직접 수행. 'rmse' 문자열이나 미리 계산한 점수가 아니다. |
| `maximize` | `False` | `True` | 공식 지표가 클수록 좋은지 확인. AUC/F1/accuracy True, MSE/RMSE/MAE False. score_fn=None이면 True가 거절된다. |
| `loss_fn` | `None` | `torch.nn.L1Loss(reduction='mean')` | 필요할 때 생성한 mean reduction 손실 객체. None이면 task로 기본 손실 생성. torch.nn.MSELoss 클래스나 'mse' 문자열이 아닌 torch.nn.MSELoss() 객체. VAE tuple loss 등은 불가. |

**반환값**

(model, history). model은 전달한 인스턴스가 학습·복원된 상태이며 eval 모드. history는 epoch/train_loss/valid_loss/monitor 열의 DataFrame.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다. PyTorch가 설치되어 있어야 이 API가 정의된다.

```python
import hdat_templates as h
import numpy as np
X_train = np.array([[0., 0.], [1., 0.], [0., 1.], [1., 1.]], dtype=np.float32)
y_train = np.array([0., 1., 1., 2.], dtype=np.float32)
X_valid = np.array([[0.5, 0.], [0., 0.5]], dtype=np.float32)
y_valid = np.array([0.5, 0.5], dtype=np.float32)
h.seed_everything(42)
train_loader = h.make_tensor_loader(X_train, y_train, task="regression",
                                  batch_size=2, shuffle=True)
valid_loader = h.make_tensor_loader(X_valid, y_valid, task="regression",
                                  batch_size=2, shuffle=False)
model = h.MLP(n_features=X_train.shape[1], out_dim=1, hidden=(8,))
model, history = h.train_torch_model(
    model=model, train_loader=train_loader, valid_loader=valid_loader,
    task="regression", epochs=2, lr=1e-3, patience=2, device="cpu",
    score_fn=lambda yt, raw: h.evaluate_predictions(yt, raw, "rmse"),
    maximize=False,
)
print(history[["epoch", "valid_loss", "monitor"]])
```

**주의할 점**

- 필수는 model,train_loader,valid_loader,task. model_factory/config/metric 인자는 없다. model_factory를 쓰고 싶다면 model=model_factory()처럼 먼저 객체를 만든다.
- 표용 TabularConfig는 받지 않는다. epochs/lr 등은 개별 키워드 인자로 전달한다.
- model과 loader/loss/score_fn의 task·출력 shape·dtype를 하나의 계약으로 맞춘다.
- score_fn=None이면 검증 loss 최소를 선택한다. 공식 F1/AUC/accuracy 선택에는 score_fn과 maximize=True를 함께 전달한다.
- score_fn에는 loader에 들어 있던 y와 모델의 raw 출력이 NumPy로 들어간다. 확률 변환·문턱·클래스 복원·y 스케일 역변환은 콜백 책임이다.
- loss_fn이 있으면 pos_weight/class_weight로 새 loss를 만들지 않는다. loss_fn은 mean reduction이며 가중 CE는 hard class-index만 지원한다.
- VAE/GAN·튜플 모델 출력·여러 입력·mask·은닉 상태를 배치 간에 유지하는 순환 학습은 전용 루프 필요.
- max_seconds는 training batch 종료 때 검사하고 그 뒤 전체 validation을 끝낸다. 준비·제출 시간도 포함한 엄격한 벽시계 제한은 아니다.
- 같은 model로 다시 호출하면 이전 가중치에서 이어진다. 독립 실험·fold·전체 재학습마다 새 모델을 만들고, 마지막 test 선택용 추가 검증으로 데이터를 재사용하지 않는다.
- optimizer는 내부 AdamW로 고정. history의 최고 monitor가 반환 모델이며 전체 train 재학습/파일 저장/실제 제출은 자동 수행하지 않는다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `make_tensor_loader`

현재 수치 배열을 TensorDataset과 DataLoader로 묶는다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `X` | 필수 | `X_train_s` | 전처리·정렬·분할을 마친 현재 단계의 수치 배열. dense CPU NumPy/배열 변환 가능한 객체. (N,F),(N,T,F),(N,C,H,W) 등 모델 계약. float32로 변환. |
| `y` | `None` | `y_train` | X와 같은 순서의 정답 배열. 회귀 (N,) 또는 (N,D); 이진 (N,) 또는 (N,1)의 0/1; 다중분류 (N,)의 정수 번호; 다중라벨 (N,K)의 0/1. 이 loader는 문자열 라벨을 받지 않으므로 먼저 번호로 바꾼다. |
| `task` | `'regression'` | `'binary'` | 정답이 연속값인지, 이진/다중분류/다중라벨인지 문제에서 판별. 'regression'\|'binary'\|'multiclass'\|'multilabel'의 정확한 문자열. 출력·정답·loss 계약이 함께 바뀐다. 모든 함수가 오타를 즉시 검사하는 것은 아니다. |
| `batch_size` | `128` | `128` | 한 번에 학습/예측할 샘플 수: 메모리와 작은 실행 실험으로 결정. 양의 정수. 윈도 길이 T나 전체 행 수 N이 아니다. |
| `shuffle` | `False` | `True` | 훈련 샘플 순서를 섞을지 선택. 훈련에서 보통 True, 검증·테스트는 False. 시계열은 윈도를 나누는 단계가 아니라 이미 안전하게 나눈 훈련 윈도의 배치 순서만 섞는다. |

**반환값**

DataLoader. y가 없으면 배치 (xb,), 있으면 (xb,yb). xb는 float32.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다. PyTorch가 설치되어 있어야 이 API가 정의된다.

```python
import hdat_templates as h
import numpy as np
X = np.array([[0., 1.], [1., 0.], [2., 1.], [1., 2.]], dtype=np.float32)
y = np.array([0, 1, 2, 1])  # 미리 정한 클래스 매핑 0,1,2
loader = h.make_tensor_loader(X, y, task="multiclass",
                            batch_size=2, shuffle=False)
xb, yb = next(iter(loader))
assert tuple(xb.shape) == (2, 2)
assert tuple(yb.shape) == (2,)
```

**주의할 점**

- 이 단계는 train/valid를 나누거나 표준화·라벨 인코딩을 하지 않는다. 먼저 끝낸 배열을 넣는다.
- 회귀/이진/다중라벨 y=(N,)는 (N,1) float32로 확장. 다중분류는 (N,) long으로 변환한다.
- multiclass는 0 이상 정수 값만 받고 범위 C는 학습 시 모델 출력과 추가 검사. raw 문자열 라벨을 먼저 0..C−1로 매핑해야 한다.
- sparse 입력을 거절한다. 전처리의 출력이 dense라고 추정하지 말고 실제 타입·shape를 확인한다.
- 반환 DataLoader는 num_workers=0, drop_last=False. test에는 shuffle=False를 사용한다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `make_torch_loss`

과제 종류에 맞는 기본 손실 객체를 만든다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `task` | 필수 | `'binary'` | 정답이 연속값인지, 이진/다중분류/다중라벨인지 문제에서 판별. 'regression'\|'binary'\|'multiclass'\|'multilabel'의 정확한 문자열. 출력·정답·loss 계약이 함께 바뀐다. 모든 함수가 오타를 즉시 검사하는 것은 아니다. |
| `pos_weight` | `None` | `[9.0]` | 훈련 폴드의 음성 수/양성 수로 정한 BCE 양성 항 가중치. 이진 길이 1 또는 스칼라, 다중라벨 길이 K. 양성 0개는 별도 처리. 테스트/검증 라벨로 계산하지 않는다. |
| `class_weight` | `None` | `[0.5, 1.0, 2.0]` | 훈련 폴드에서 클래스 0..C−1 순서로 계산한 CE 가중치. 길이 C의 유한한 실수, 통상 양수. BCE용 pos_weight와 다른 정의. |

**반환값**

regression→MSELoss, binary/multilabel→BCEWithLogitsLoss, multiclass→CrossEntropyLoss. 모두 기본 mean.

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다. PyTorch가 설치되어 있어야 이 API가 정의된다.

```python
import hdat_templates as h
import torch
criterion = h.make_torch_loss(task="binary", pos_weight=[3.0])
logits = torch.tensor([[0.0], [1.0]])
targets = torch.tensor([[0.0], [1.0]])
loss = criterion(logits, targets)
assert loss.ndim == 0
```

**주의할 점**

- 기본 회귀 손실은 MSE다. 평가 metric='mae'라고 자동 변경되지 않는다.
- pos_weight는 BCE 양성 항, class_weight는 CE 클래스별 전체 항이다. 다른 task에서는 해당 인자를 무시한다.
- 직접 GPU 계산에 사용하면 criterion.to(device)가 필요하다. train_torch_model은 내부에서 옮긴다.
- 반환값은 계산된 loss 숫자가 아니라 criterion(logits,y)로 호출할 객체다.

구현 확인: [hdat_templates.py](./hdat_templates.py).


#### `predict_torch`

모델을 eval 모드에서 호출하고 과제 종류에 맞게 값·라벨·확률로 변환한다.

| 인자 | 필수 여부·기본값 | 실제 값 예 | 값의 출처·shape·dtype·조건 |
|---|---|---|---|
| `model` | 필수 | `trained_model` | train_torch_model이 반환했거나 checkpoint를 복원한 실제 모델 객체. torch.nn.Module 인스턴스. 출력 Tensor와 task가 일치해야 한다. |
| `loader` | 필수 | `h.make_tensor_loader(X_test_s, task='regression', shuffle=False)` | 제출 행 순서대로 구성한 추론 DataLoader. xb 또는 (xb,) 또는 (xb,yb) 배치. 첫 항목을 입력으로 쓰며 모든 출력이 누락 없이 제출 순서여야 한다. |
| `task` | 필수 | `'binary'` | 정답이 연속값인지, 이진/다중분류/다중라벨인지 문제에서 판별. 'regression'\|'binary'\|'multiclass'\|'multilabel'의 정확한 문자열. 출력·정답·loss 계약이 함께 바뀐다. 모든 함수가 오타를 즉시 검사하는 것은 아니다. |
| `return_proba` | `False` | `True` | 제출 명세가 확률인지 라벨인지 확인. binary/multilabel: True면 sigmoid 확률; multiclass True면 softmax 확률. 회귀에는 영향 없음. |
| `threshold` | `0.5` | `0.4` | 검증 데이터에서 정하고 고정한 이진/다중라벨 판정 문턱. 확률 기준 실수, 통상 0..1. out=(prob>=threshold), 등호 포함. 테스트 라벨로 고르지 않는다. |
| `device` | `None` | `'cpu'` | 현재 실제로 사용 가능한 계산 장치. None은 CUDA 가능 시 cuda, 아니면 cpu. 'mps'는 지원 여부를 확인하고 명시해야 하며 자동 선택되지 않는다. |

**반환값**

NumPy 배열. 회귀 (N,D); 이진 (N,1); 다중라벨 (N,K); 다중분류 확률 (N,C) 또는 라벨 (N,).

**짧은 호출 예제**

아래 import부터 독립 실행합니다. hdat_templates.py를 같은 폴더에 두세요. 입력값은 예제 안에서 직접 만듭니다. PyTorch가 설치되어 있어야 이 API가 정의된다.

```python
import hdat_templates as h
import numpy as np
X_test = np.zeros((3, 2), dtype=np.float32)
model = h.MLP(n_features=X_test.shape[1], out_dim=1, hidden=(4,))
# 이 예시는 미학습 모델의 출력 형식만 확인한다.
test_loader = h.make_tensor_loader(X_test, task="binary", shuffle=False)
prob = h.predict_torch(model, test_loader, task="binary",
                      return_proba=True, device="cpu")
assert prob.shape == (3, 1)
one_probability_per_row = prob[:, 0]  # 제출 계약이 (N,)일 때만
assert one_probability_per_row.shape == (3,)
```

**주의할 점**

- model은 학습 완료 인스턴스다. 아래 짧은 예시는 shape 점검만 하며 무작위 모델의 값을 제출하라는 예시가 아니다.
- loader 순서 그대로 concatenate한다. shuffle=True/drop_last=True를 자동 차단하지 않으므로 테스트는 반드시 점검한다.
- 이진/다중라벨 threshold는 >= 비교. 회귀는 return_proba/threshold를 쓰지 않고 raw 출력 그대로 반환한다.
- 클래스 인덱스를 원래 문자열/번호 라벨로 복원하거나 회귀 역표준화하는 기능은 없다.
- 이진 (N,1)을 제출 (N,)로 요구하면 의미를 확인한 뒤 pred[:,0]처럼 명시한다. flatten으로 다중출력을 합치지 않는다.
- VAE tuple 반환·mask 전달은 지원하지 않는다.

구현 확인: [hdat_templates.py](./hdat_templates.py).

