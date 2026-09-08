# HDAT-DS 치트시트 실전 사용 실습

## 00. 시작 매뉴얼 · 읽기에서 직접 제출까지

Python 문법을 조금 아는 사람을 위한 독립 실습입니다. Process 8문제는 요구된 동작을 구현하고, Problem 7문제는 데이터에서 모델을 학습해 예측 파일을 만듭니다. 정답을 읽는 것과 혼자 실행하는 것을 구분하세요.

### 무엇을 연습하는가

공식 HDAT-DS 실기는 Process 8문항과 Problem 1문항으로 구성되며 합산 170분입니다. 이 연습팩의 Problem 7문제는 한 번에 풀 시험지가 아니라 여러 날 나눠 풀 **유형 확장 문제은행**입니다. 공식 원문·데이터를 복제하거나 변형하지 않고 새로 만든 합성 문제입니다. 기출 적중·출제 빈도·시험과 같은 난도는 보장하지 않습니다. [NGV 공식 안내](https://exam.hyundai-ngv.com/practice/13567)

2026-09-08 공식 GitBook을 재확인했습니다. 실기는 제공 Jupyter IDE에서 풀며 허용된 단방향 자료 참고·검색만 가능합니다. 필기는 자료·검색 불가입니다. 생성형 AI·검색의 AI 답변, GitHub·Colab·Kaggle·Notion·메일·메신저 등은 사용하지 않습니다. 이 사이트와 파일은 **시험 전 학습용**입니다. 개인 참고물은 사전에 허용된 로컬 형식으로 준비하고 문제 데이터는 제공 IDE 밖에서 처리하지 마세요. 시험 감독관의 공식 연락 절차는 별도로 따릅니다. [부정행위 FAQ](https://hdat.gitbook.io/2026-hdat-ds/faq/3..md)

로컬 연습의 `import hdat_templates as h`는 제출 규칙이 아닙니다. 실제 문제에서는 허용된 작성영역에 필요한 import·함수·모델 정의를 넣고 지정된 변수명으로 연결하세요. 별도 `.py` 파일이 자동 제출된다고 가정하지 마세요. 추가 패키지 설치 허용은 내 helper 파일의 제출 보장과 다릅니다. 문제별 라이브러리·작성영역 제한이 우선이며 설치·실행 시간도 시험시간에 포함됩니다. [환경 FAQ](https://hdat.gitbook.io/2026-hdat-ds/faq/1.-pc.md)

| 단계 | 먼저 할 일 | 스스로 확인할 질문 |
|---|---|---|
| 1. 문제만 읽기 | 명세를 메모하고 10분간 혼자 시도 | 함수인가? 학습 문제인가? 반환값의 shape는? |
| 2. 힌트 한 번 보기 | 막힌 지점만 확인하고 다시 구현 | 지금 필요한 개념 하나는 무엇인가? |
| 3. 수정 매뉴얼 보기 | 치트시트 항목을 열고 인수를 변경 | 그대로 쓰는 부분과 바꾸는 부분을 구분했는가? |
| 4. 로컬 검사 | 작은 입력 → 전체 입력 → 저장 후 reload | 실행 성공과 좋은 성능을 혼동하지 않는가? |
| 5. 해설 비교 | 오류 원인을 기록하고 새 파일로 재도전 | 왜 이 줄이 필요한지 설명할 수 있는가? |

### 내려받기와 첫 실행

[실전 연습팩 ZIP](/downloads/hdat-practical-labs.zip)을 받아 **압축을 먼저 해제**합니다. `practical-labs` 폴더의 `manual.html`을 더블클릭하면 인터넷 없이 이 매뉴얼을 읽을 수 있습니다. 웹페이지에는 Python 실행 기능이 없습니다. 실행은 내 컴퓨터의 Python 또는 Jupyter에서 합니다. 인터넷 없이 실행하려면 라이브러리도 미리 설치되어 있어야 합니다.

| 파일 | 역할 | 직접 수정? |
|---|---|---|
| `00-start.ipynb` | 처음 실행할 때의 폴더 확인·연습 명령 | 실행할 문제 번호만 수정 |
| `process_starter.py` | P01–P08의 빈 함수·클래스 | 예: 선택 문제의 TODO |
| `problem_starter.py` | B01–B07을 받는 `solve(case, data)` | 예: 선택 문제 분기 구현 |
| `hdat_templates.py` | 치트시트의 모델·전처리·학습 루프 | 처음에는 수정하지 않고 import |
| `check_process.py` | Process의 공개 예제·경계 검사 | 아니오 |
| `lab_runtime.py`, `check_submission.py` | Problem 저장·입출력 검사·자가채점 | 아니오 |
| `data/B01.npz` 등 | NumPy 배열 묶음, train/valid 정답 포함 | 아니오 |
| `solutions/` | 참고 구현과 별도 test 정답 키 | 먼저 풀고 나서 열기 |

터미널은 명령을 입력하는 창입니다. 해당 폴더에서 열거나 `cd` 뒤에 압축을 푼 **실제 경로**를 넣으세요. 아래 `python`은 설치에 따라 `python3`일 수 있습니다. 이미 Jupyter가 있으면 `00-start.ipynb`부터 열어도 됩니다.

```bash
python --version
python -m pip install -r requirements.txt
python check_process.py --case P01 --file process_starter.py
```

마지막 명령의 첫 결과는 `NotImplementedError`가 정상입니다. 환경 문제가 아니라 빈 함수라는 뜻입니다. P01을 직접 구현하고 같은 명령을 다시 실행하세요. `PASS`는 제공한 공개 검사 통과일 뿐, 모든 입력에서의 정답 증명이나 공식 시험 합격이 아닙니다.

이 팩은 NumPy·pandas·scikit-learn을 데이터 준비와 지표에 사용하고, **모든 신경망은 PyTorch**로 구현합니다. requirements는 로컬 학습용이며 실제 시험 버전을 선언하는 파일이 아닙니다. 사용법을 처음 배우면 [PyTorch 공식 입문](https://docs.pytorch.org/tutorials/beginner/basics/quickstart_tutorial.html)에서 Dataset/DataLoader → 모델 → 학습/평가 흐름을 함께 읽으세요.

### 치트시트를 코드에 연결하는 공통 순서

1. [문제 유형 결정표](/cheatsheet.html#60)를 열고 문제의 입력 형태를 찾습니다. 표면적인 ‘자동차’라는 소재보다 표·이미지·시계열인지가 중요합니다.
2. [유형별 가이드](/playbook/)에서 해당 유형을 엽니다. 문제별 매뉴얼에는 정확한 번호와 함수명이 적혀 있습니다.
3. Process는 비슷한 함수를 가져오되 **이 문제의 명세가 우선**입니다. 예를 들어 중앙 crop과 왼쪽 위 crop은 같은 답이 아닙니다.
4. Problem은 `import hdat_templates as h` 후 전처리, 모델, 학습 루프, 제출 변환의 네 부분을 연결합니다. `h.MLP`를 복사할 필요 없이 `h.MLP(입력열수, 출력개수)`처럼 호출합니다.
5. 아래 공통 RUN을 쓰려면 앞에서 `Xtr`, `Xva`, `Xte`, `ytr`, `yva`, `model`, `task`, `score_fn`, `maximize`를 모두 정의해야 합니다. 이 이름들은 자동으로 생기지 않습니다.

```python
import numpy as np
import hdat_templates as h

# 전처리와 모델 정의가 끝난 후 solve() 내부에 넣는 공통 블록입니다.
tr = h.make_tensor_loader(Xtr, ytr, task, batch_size=32, shuffle=True)
va = h.make_tensor_loader(Xva, yva, task, batch_size=64, shuffle=False)
te = h.make_tensor_loader(Xte, task=task, batch_size=64, shuffle=False)
model, history = h.train_torch_model(
    model, tr, va, task, epochs=15, patience=5, lr=0.003,
    device="cpu", score_fn=score_fn, maximize=maximize,
)
pred = h.predict_torch(model, te, task, device="cpu",
                       return_proba=case in {"B06", "B07"})
# 마지막에 문제별 라벨 복원·역변환·error 계산 후 return pred
```

`task`는 loss와 target dtype을 정합니다. `score_fn`은 epoch별로 어떤 모델을 남길지 정합니다. 분류는 loss를 줄이면서 Macro-F1 또는 AUC가 가장 높은 모델을 고를 수 있습니다. 이 둘은 서로 다른 목적입니다. 학습 루프는 최선 epoch의 가중치를 돌려주므로 마지막 epoch의 모델로 임의 교체하지 마세요.

### Problem 데이터를 열어 보는 법

NPZ는 여러 이름 붙은 배열을 한 파일에 담는 NumPy 형식입니다. 압축 해제된 **팩 폴더에서** 다음 셀을 실행합니다. `data`는 키로 배열을 꺼내는 Python 딕셔너리입니다.

```python
from lab_runtime import load_case
data = load_case("B01")
for name, value in data.items():
    print(name, value.shape, value.dtype)
print(data["X_train"][:3])
print(data["y_train"][:10])
```

`train`은 학습용, `valid`는 모델 비교용, `test`는 최종 예측용입니다. 이미 분리되어 있으므로 다시 무작위로 합치거나 나누지 않습니다. 이 팩의 분할은 문제별 계약입니다. 실제 문제는 그룹·시간·중복 조건을 보고 직접 검증 전략을 정해야 할 수 있습니다.

`*_ids`는 각 행의 식별자입니다. 숫자 크기에 예측 의미가 없으므로 모델 입력에 넣지 않습니다. 특히 test를 ID순 정렬하지 마세요. 제공된 test 배열 순서가 제출 순서입니다. `dataset_id`는 버전 혼동을 검사하는 값이지 특성이 아닙니다.

### 저장·형식 검사·자가채점을 구분하기

`problem_starter.py`의 선택 문제를 구현하면 다음 명령으로 실행합니다. CLI는 `solve`가 반환한 배열을 검사하고 ID와 함께 저장합니다. 기존 출력 파일을 보호하므로 재시도할 때는 `v2`, `v3`처럼 새 이름을 쓰세요.

```bash
python problem_starter.py --case B01 --output outputs/B01-v1.npz
python check_submission.py --case B01 --file outputs/B01-v1.npz
python check_submission.py --case B01 --file outputs/B01-v1.npz --score
```

두 번째 명령은 정답을 읽지 않습니다. shape·dtype·유한값·ID·버전·허용 범위만 검사합니다. 세 번째 명령은 `solutions/answer_key.npz`를 읽어 점수를 계산합니다. 이 키는 **보안상 숨긴 정답이 아닙니다**. 자가학습 편의를 위해 별도 폴더에 둔 것입니다. test 점수를 보며 계속 튜닝하면 test를 검증셋처럼 쓰게 됩니다. 튜닝은 valid로, 최종 `--score`는 결정을 마친 뒤 한 번만 하세요.

NPZ 제출은 이 학습팩 자체 형식이며 공식 시험의 제출 형식과 같다는 뜻이 아닙니다. 실제 시험에서는 제공된 저장 셀·파일명·변수명·shape를 따르세요. ID 검사는 잘못 선언한 순서를 찾아주지만, 사용자가 거짓 ID를 붙인 예측값의 의미까지 증명하지는 못합니다.

실제 시험 마무리는 **코드 Ctrl+S → Autosaved 확인 → 기존 저장 셀 실행·예측 파일 확인 → 다시 저장 → Process/Problem 각각 제출 → 정상 제출 팝업·로그 확인**입니다. 파일 생성, 노트북 저장, 문항 제출은 서로 다른 상태입니다. `Autosave Failed`나 연결 오류가 나면 즉시 감독관에게 알리세요. 테스트 종료 후에는 재입장·수정할 수 없습니다. [저장·작성 안내](https://hdat.gitbook.io/2026-hdat-ds/undefined/undefined/6.-2/1/undefined-1.md)

### 권장 공부 순서와 기록

처음이면 P01 → P02 → P03 → P04 → P05 → P06 → P07 → P08 → B01 → B02 → B03 → B06 → B07 → B04 → B05 순으로 풀어보세요. 개념이 막히면 기초 강의로 돌아가도 됩니다. 첫 회에는 시간 제한 없이, 두 번째 회에는 Process 전체 70분 + B03 100분을 권장합니다. 이 배분은 비공식 연습안입니다.

15문제를 마쳤다면 [13강 검증 전략](/learn/13/)과 [33강 통합 모의](/learn/33/)로 이동하세요. 이 팩은 valid를 제공하므로 **직접 분할을 설계하는 능력**까지 평가하지 않습니다. 제공 valid가 없다고 가정하고 stratified/group/time 중 선택 이유, train-only fit, test 행 대응을 글로 설명한 뒤 실행하세요. 큰 합성 입력으로 초/epoch·메모리·첫 유효 파일까지의 시간도 별도로 측정해야 합니다.

매번 ‘처음 유효 출력까지 걸린 시간 / 펼친 힌트 수 / 틀린 계약 / valid 지표 / 다시 시도할 한 가지’를 적으세요. 합성 데이터는 작고 비교적 단순합니다. 높은 점수를 받았더라도 실제 데이터의 노이즈·불균형·자원 제한에 대비한 검증과 시간 관리까지 익혔다고 단정하지 마세요.

## P01. 선택 센서 열만 Min-Max 변환

Process · 쉬움 · 권장 8분. 표 전체를 바꾸지 않고 지정된 열만 정확하게 바꾸는 문제입니다.

### 문제와 계약

`scale_sensor_columns(frame, columns)`를 구현하세요. `frame`은 열 이름이 유일한 pandas DataFrame, `columns`는 중복 없는 열 이름 리스트입니다. 선택 열은 수치형이며 NaN은 가능하지만 Inf는 입력 범위 밖입니다. 각 선택 열의 결측을 제외한 최솟값·최댓값으로 `(x-min)/(max-min)`을 계산하세요.

상수 열의 **유효값만 0.0**, 전부 NaN인 열은 NaN을 유지합니다. 선택 열 결과 dtype은 float64입니다. 비선택 열의 값·dtype, 행/열 순서, 중복 가능한 index는 유지합니다. 원본은 바꾸지 않고 새 DataFrame을 반환합니다. 빈 DataFrame·빈 선택 목록도 허용합니다. 없는 열은 KeyError, 중복 선택 목록은 ValueError입니다.

```python
import numpy as np
import pandas as pd
frame = pd.DataFrame({"sensor": [4., 10., 7., np.nan], "fixed": [8, 8, 8, 8],
                      "code": ["A", "B", "C", "D"]}, index=[9, 9, 2, 4])
# sensor -> [0.0, 1.0, 0.5, NaN], fixed -> [0.0]*4
# code와 index [9,9,2,4]는 그대로
```

### 힌트

`frame.copy(deep=True)`를 먼저 만드세요. 열마다 `s.min()`, `s.max()`를 구합니다. NaN을 0으로 채우라는 명세가 아니므로 `fillna(0)`는 여기서 오답입니다.

### 치트시트 수정 매뉴얼

1. [Process 선택 열 가이드](/playbook/01/)와 [Min-Max 블록](/cheatsheet.html#min-max)을 엽니다. `process_starter.py`의 `scale_sensor_columns` 안에 적용합니다.
2. 전체 숫자 열 자동 선택 대신 인수 `columns`만 반복합니다. 반환 직전 원본의 비선택 열을 새로 합치지 말고 복사본의 선택 열만 대입하세요.
3. `lo == hi`를 사용합니다. `np.isclose`는 1e9와 1e9+0.25를 사실상 같은 수로 볼 수 있어 이 문제에서는 틀립니다.
4. 상수 분기에서는 `s.where(s.isna(), 0.0)`로 NaN을 보존합니다. 그 외에는 표준식을 쓰면 all-NaN도 NaN으로 남습니다.
5. [입출력 계약](/cheatsheet.html#_1) 순서대로 원본 불변·index·dtype를 검사합니다.

```bash
python check_process.py --case P01 --file process_starter.py
```

### 해설과 재도전

‘0으로 나눈다’는 문제를 분기 처리하되, 결측 정책을 섞지 않는 것이 핵심입니다. 전처리를 학습 데이터에서 fit해 재사용하는 Problem과 달리, 이 Process 계약은 전달받은 표 자체의 min/max로 계산하라고 했습니다. 같은 Min-Max라도 문제에 따라 통계 범위가 달라집니다.

참고 구현은 `solutions/process_solution.py`의 `_selected`와 `scale_sensor_columns`입니다. 내 함수와 비교할 때 helper의 입력 검사도 같이 읽으세요. 재도전: `[5, NaN, 5]`, `[NaN, NaN]`, `[1e9, 1e9+.25]`, 빈 표를 직접 넣고 기대 출력을 먼저 종이에 쓰세요.

## P02. 중앙값과 IQR로 센서 안정화

Process · 쉬움–보통 · 권장 8분. P01의 뼈대에서 통계량과 0분모 정책만 정확히 바꾸는 연습입니다.

### 문제와 계약

`robust_sensor_columns(frame, columns)`를 구현하세요. 입력 보존·출력 dtype·NaN·빈 입력·없는 열·중복 열 정책은 P01과 같습니다. 각 열에서 `(x - median) / IQR`을 계산합니다. `IQR = Q(0.75)-Q(0.25)`이고 quantile의 보간은 **linear**입니다.

IQR이 정확히 0이면 그 열의 유효값은 모두 0.0으로 합니다. 이는 이 문제만의 명시적인 예외 정책입니다. 일반 라이브러리의 RobustScaler와 모든 경계 동작이 같다고 가정하지 마세요.

```text
입력 [0, 2, 4, 6]: median=3, Q1=1.5, Q3=4.5, IQR=3
반환 [-1, -1/3, 1/3, 1]
입력 [0,0,0,0,100]: IQR=0 → [0,0,0,0,0]
```

### 힌트

P01에서 복사본 생성·선택 열 순회는 그대로입니다. 최솟값을 중앙값으로, 범위를 IQR로 바꾸세요. ‘이상치가 있으니 100을 그대로 둔다’는 직관보다 명세가 우선입니다.

### 치트시트 수정 매뉴얼

1. [선택 열 변환 가이드](/playbook/01/)의 함수 계약을 연 뒤 P01 구현을 별도 함수로 복사합니다.
2. `s = frame[col].astype("float64")`는 유지합니다. `s.median()`과 `s.quantile(.75, interpolation="linear")`를 사용합니다.
3. 상수 판정 `max == min`을 `iqr == 0`으로 바꿉니다. 두 조건은 같지 않습니다. 예제의 100이 바로 차이를 드러냅니다.
4. division 결과를 원본에 대입하지 말고 `out[col]`에 넣고 `return out` 합니다.
5. P01의 검사만 통과했다고 끝내지 말고 quantile 보간 예제와 IQR=0 비상수 열을 검사하세요.

```bash
python check_process.py --case P02 --file process_starter.py
```

### 해설과 재도전

IQR은 가운데 50%의 폭입니다. 평균/표준편차보다 일부 극단값의 영향이 작을 수 있지만, IQR=0인 데이터를 어떻게 처리할지는 별도 선택입니다. 이 문제에서는 명세대로 0을 반환합니다. 실제 모델 전처리에서는 train의 median/IQR을 보관하고 valid/test에 재사용해야 합니다.

참고 구현의 `_selected`, `robust_sensor_columns`를 확인하세요. 재도전: linear 대신 nearest 보간을 썼을 때 `[0,2,4,6]`의 결과가 왜 달라지는지 계산하세요. 수정 후에도 비선택 문자열 열의 dtype이 바뀌지 않았는지 확인합니다.

## P03. 설비별 과거 두 행의 평균

Process · 보통 · 권장 12분. 정렬·그룹·현재값 제외·원래 순서 복원의 네 가지를 함께 다룹니다.

### 문제와 계약

`add_prior_average(frame, group_col, time_col, value_col, window=2)`를 구현하세요. group은 결측 없는 문자열, time은 결측 없는 정수, value는 수치형/NaN입니다. 열 이름은 유일하고 원본 index는 중복될 수 있습니다.

그룹별 time 오름차순으로 정렬하되 동시간이면 원래 행 위치가 앞선 것이 먼저입니다. 각 행에서 **현재 행을 제외한 직전 window개의 관측 행**의 평균을 계산합니다. NaN은 해당 범위 내 평균에서 제외하며, 사용할 유효값이 없으면 NaN입니다. ‘유효값 window개를 찾을 때까지 더 과거로 이동’하는 뜻이 아닙니다.

원본을 변경하지 않고 모든 기존 값·dtype·index·순서를 유지한 표를 반환하며 마지막에 `<value_col>_prior_mean` float64 열을 추가합니다. 그 열이 이미 있으면 ValueError, 없는 인수 열은 KeyError입니다. window는 bool이 아닌 양의 정수, 그렇지 않으면 ValueError입니다. 빈 표도 허용합니다.

```text
원래 행: group [A, B, A, A, B]
          time [3, 2, 1, 2, 1]
         value [30,100,10,20,80]
반환 새 열: [15, 80, NaN, 10, NaN]
```

### 힌트

group별로 `shift(1)`을 먼저 적용하고 `rolling(window, min_periods=1).mean()`을 합니다. 원본 index가 중복되므로 index로 되돌리려 하지 말고 0부터 시작하는 행 위치를 별도로 기록하세요.

### 치트시트 수정 매뉴얼

1. [group별 lag·rolling](/playbook/02/)을 엽니다. `group_col`, `time_col`, `value_col`은 문자열 인수로 받고 실제 열 이름을 하드코딩하지 마세요.
2. 임시 표에 group/time/value/행위치만 담습니다. 원본 표에 `_row`라는 임시 열을 직접 만들면 기존 동명 열을 덮을 수 있습니다.
3. group, time, 행위치의 세 키로 정렬합니다. 이어서 `groupby(...).transform(...)` 안에서 shift와 rolling을 함께 수행합니다.
4. 결과를 길이 N의 NaN 배열에 `result[정렬된 원래위치] = 계산값`으로 배치합니다. 마지막에 원본 복사본의 새 열로 넣습니다.
5. A그룹의 첫 행에 B그룹의 값이 섞이지 않는지, 현재 행을 99999로 바꿔도 그 행의 과거 평균은 그대로인지 검사합니다.

```bash
python check_process.py --case P03 --file process_starter.py
```

### 해설과 재도전

현재값을 제외하지 않으면 ‘현재값으로 현재를 예측’하는 누수가 생깁니다. 그룹 구분 없이 rolling하면 다른 설비의 기록이 섞입니다. 정렬 결과 그대로 반환하면 값 계산이 맞아도 제출 행이 틀립니다.

참고 함수 `add_prior_average`는 임시 표를 따로 만들어 원본과 충돌을 피합니다. 재도전: 동일 그룹 값 `[10, NaN, 30, 40]`, 순서 `[1,1,2,3]`, window=2의 결과는 `[NaN,10,10,30]`입니다. 마지막 평균이 20이 아닌 이유를 설명하세요. window=1, 중복 index, `_row` 원본 열도 검사합니다.

## P04. 큰 logits에서도 안정적인 교차엔트로피

Process · 보통 · 권장 10분. PyTorch loss를 호출하지 않고 NumPy로 수식을 구현합니다.

### 문제와 계약

`mean_logit_loss(logits, labels)`를 구현하세요. logits는 유한한 실수 `(N,C)`, N≥1, C≥1입니다. labels는 정수 dtype의 `(N,)` 배열이고 값 범위는 0부터 C-1입니다. 한 행마다 softmax의 정답 클래스 음의 로그를 계산한 뒤 평균하여 Python `float` 하나를 반환합니다. 내부 연산은 float64로 하며 입력은 바꾸지 않습니다.

잘못된 shape, 빈 입력, float/bool labels, 범위 밖 label, NaN/Inf logits는 ValueError입니다. 특히 `(N,1)` labels를 자동으로 펴서 받아주지 않습니다.

```text
logits=[[0,0],[log(3),0]], labels=[0,1]
정답=(log(2)+log(4))/2 ≈ 1.03972077
logits가 모두 1000인 3클래스 → log(3), overflow 없어야 함
C=1, labels가 모두 0 → 0.0
```

### 힌트

softmax에 같은 수를 더하거나 빼도 확률은 같습니다. 각 행의 최댓값을 빼고 exp를 계산하세요. 확률을 만든 뒤 `log(0)`을 피하려고 임의 epsilon을 넣는 것보다 logits에서 바로 log-sum-exp를 계산하는 편이 정확합니다.

### 치트시트 수정 매뉴얼

1. [NumPy 수치 함수](/playbook/03/)를 엽니다. 평균 MSE 예시라면 반복 뼈대만 참고하고 계산식은 반드시 바꿉니다.
2. 입력 계약을 먼저 확인합니다. labels를 무조건 `astype(int)` 하면 0.9 같은 잘못된 입력이 조용히 통과하므로 먼저 정수 dtype인지 검사합니다.
3. `z = x.astype(np.float64) - x.max(axis=1, keepdims=True)`를 만듭니다. keepdims가 있어야 행별 최댓값이 같은 행에 적용됩니다.
4. `np.log(np.exp(z).sum(axis=1)) - z[np.arange(N), y]`가 샘플별 loss입니다. 평균을 구해 `float(...)`로 반환합니다.
5. [shape/loss 표](/cheatsheet.html#2-pytorch-shapeloss)로 실제 모델 학습에서는 CrossEntropyLoss에 softmax 전 logits를 주는 이유까지 연결하세요.

```bash
python check_process.py --case P04 --file process_starter.py
```

### 해설과 재도전

최댓값을 빼면 exp의 입력 최댓값이 0이므로 큰 양수 exp overflow를 막습니다. log-sum-exp에서 같은 정답 logit을 빼므로 공통 이동량도 상쇄됩니다. 확률에 epsilon을 더하는 방식은 원래 수식의 값을 바꿀 수 있습니다.

참고 함수 `mean_logit_loss`와 비교하고 logits에 10000을 더한 결과가 같은지 검사하세요. 재도전: labels shape가 `(2,1)`인 경우, -1인 경우, C=1인 경우를 직접 실행하고 명세에 맞는 성공/실패를 예측하세요.

## P05. 채널을 보존하는 중앙 이미지 crop

Process · 쉬움–보통 · 권장 8분. PIL 좌표 순서와 NumPy 축 순서를 구분합니다.

### 문제와 계약

`center_sensor_crop(image, crop_width, crop_height)`를 구현하세요. 입력은 mode가 L, RGB 또는 RGBA인 Pillow 이미지입니다. 출력은 중앙 영역을 자른 **쓰기 가능한 독립 NumPy 배열**입니다. L은 `(Hc,Wc)`, RGB는 `(Hc,Wc,3)`, RGBA는 `(Hc,Wc,4)`이며 dtype을 유지합니다. 색상 변환·resize·padding은 하지 않습니다.

시작점은 `left=(W-Wc)//2`, `top=(H-Hc)//2`입니다. 남는 크기가 홀수이면 오른쪽/아래쪽에 한 픽셀이 더 남습니다. 크기는 bool이 아닌 양의 정수이고 원본 이하여야 합니다. 잘못된 크기·지원하지 않는 mode는 ValueError, PIL 아닌 입력은 TypeError입니다. 원본 이미지는 변경하지 않습니다.

```text
원본 NumPy RGB shape=(6,9,3), crop_width=4, crop_height=3
정답 raw[1:4, 2:6, :] / 출력 shape=(3,4,3)
```

### 힌트

PIL의 `size`는 `(width,height)`이지만 NumPy shape의 앞부분은 `(height,width)`입니다. crop box의 오른쪽·아래쪽 끝은 포함되지 않습니다.

### 치트시트 수정 매뉴얼

1. [crop 가이드](/playbook/04/)와 [PIL crop 코드](/cheatsheet.html#pil-crop)를 엽니다. 함수 이름과 입력 크기 인수를 이 문제대로 바꿉니다.
2. `image.convert("RGB")`가 있다면 사용하지 않습니다. RGBA의 alpha 또는 L의 차원이 사라지면 계약 위반입니다.
3. width/height를 검증한 뒤 `(left, top, left+crop_width, top+crop_height)` box를 만듭니다.
4. `np.array(cropped, copy=True)`로 반환합니다. `np.asarray`가 반환한 읽기 전용 view를 그대로 주지 않도록 합니다.
5. 결과 배열의 첫 픽셀을 바꾼 뒤 원본 픽셀은 변하지 않는지 검사하세요.

```bash
python check_process.py --case P05 --file process_starter.py
```

### 해설과 재도전

이 문제는 augmentation으로 무작위 crop을 만드는 문제가 아닙니다. 고정 좌표·채널·복사 여부가 답입니다. RGB에서만 검증하면 L/RGBA 입력의 숨은 차원을 놓치기 쉽습니다.

참고 함수 `center_sensor_crop`를 확인하세요. 재도전: 원본 전체 크기를 자르기, 1×1 crop, 홀수 남는 크기, 팔레트 P 이미지, bool 크기를 각각 테스트합니다. 배열 shape만 맞아도 잘못된 위치를 잘랐으면 오답이므로 픽셀 값까지 비교하세요.

## P06. 명세 그대로 구현하는 작은 MLP

Process · 보통 · 권장 8분. 좋은 모델을 설계하는 문제가 아니라 정확한 구조를 구현하는 문제입니다.

### 문제와 계약

`SensorMLP()`라는 `nn.Module`을 작성하세요. `self.network`는 다음 순서의 `nn.Sequential`입니다. Linear의 bias는 둘 다 True이고 BatchNorm은 PyTorch 기본 인수입니다.

```text
Linear(12,20) → BatchNorm1d(20) → ReLU → Dropout(p=0.2) → Linear(20,4)
입력: float32 (B,12), 출력: raw logits (B,4)
train 모드 B≥2, eval 모드 B≥1. 추가 Softmax/Sigmoid 없음.
```

forward는 위 순서를 그대로 통과시킵니다. 외부에서 train/eval 전환 및 loss.backward가 가능해야 합니다. 총 학습 파라미터 수는 384개입니다. 모델 안에서 입력 shape를 억지로 reshape하거나 임의 레이어를 추가하지 마세요.

### 힌트

`super().__init__()` 다음에 Sequential을 정의하고 forward는 `self.network(x)`를 반환합니다. 마지막 Linear는 확률이 아닌 네 개의 점수입니다.

### 치트시트 수정 매뉴얼

1. [지정 모델 구조](/playbook/05/)와 [MLP 블록](/cheatsheet.html#mlp-flatten-feature)을 엽니다. `h.MLP`를 그대로 생성하면 BatchNorm 유무·레이어 배치가 다를 수 있어 이 문제의 답이 아닙니다.
2. `process_starter.py`의 `SensorMLP`에 명세의 다섯 레이어만 직접 선언합니다. 범용 모델 템플릿은 문법 참고용입니다.
3. 레이어 차원 12→20→4, Dropout .2, bias 기본 True를 하나씩 대조합니다.
4. `net.train()`에서 B=3으로 backward를 확인하고 `net.eval()`에서는 B=1을 검사합니다. train B=1 BatchNorm 오류를 reshape로 감추지 마세요.
5. 파라미터 합은 첫 Linear 260 + BN 40 + 마지막 Linear 84 = 384입니다. BN의 running mean/variance는 학습 파라미터가 아닌 buffer입니다.

```bash
python check_process.py --case P06 --file process_starter.py
```

### 해설과 재도전

Process에서는 출력 shape만 같아도 내부 구조가 다르면 틀릴 수 있습니다. 학습 가능한 파라미터 수와 모듈 목록을 함께 검사하면 누락된 bias/BN을 찾기 쉽지만 이것만으로 완전한 동일성을 증명하지는 못합니다.

참고 클래스 `SensorMLP`를 비교하세요. 재도전: eval에서 같은 입력을 두 번 넣었을 때 같은 결과인지, train에서 Dropout이 작동하는 이유가 무엇인지 설명하세요. BN은 eval에서도 학습된 affine 파라미터를 사용하되 배치 통계 대신 저장된 통계를 사용합니다.

## P07. 해상도가 달라도 동작하는 CNN

Process · 보통 · 권장 10분. Conv 출력 계산과 flatten의 배치 축을 확인합니다.

### 문제와 계약

`SurfaceCNN()`을 구현하세요. float32 입력은 `(B,1,H,W)`, H/W≥4이며 raw logits `(B,2)`를 반환합니다. `self.features`는 아래 일곱 레이어의 Sequential, `self.head`는 `Linear(10,2,bias=True)`입니다.

```text
Conv2d(1,6,kernel_size=3,stride=1,padding=1,bias=False)
BatchNorm2d(6) [기본 인수]
ReLU
MaxPool2d(kernel_size=2,stride=2)
Conv2d(6,10,kernel_size=3,stride=2,padding=1,bias=True)
ReLU
AdaptiveAvgPool2d(1)
```

features 결과를 `flatten(1)` 한 뒤 head에 넣습니다. 추가 활성화 없음, 총 학습 파라미터 638개입니다. train/eval·backward가 가능해야 하며 원본 입력을 바꾸지 않습니다. 입력 예 `(2,1,13,17)`의 최종 출력은 `(2,2)`입니다.

### 힌트

`flatten()`은 배치까지 없앱니다. `flatten(1)`은 배치를 남깁니다. 마지막 adaptive pooling을 쓰면 고정 해상도를 Linear 입력에 하드코딩할 필요가 없습니다.

### 치트시트 수정 매뉴얼

1. [모델 구조 가이드](/playbook/05/)와 [convolution 출력식](/cheatsheet.html#convolution)을 엽니다. 출력식은 `floor((입력+2p-d(k-1)-1)/s+1)`입니다.
2. [Image CNN](/cheatsheet.html#image-cnn)은 구조 문법만 참고합니다. 기본 SmallImageCNN을 그대로 쓰지 말고 위의 채널·stride·bias를 직접 구현합니다.
3. 입력 13×17은 첫 Conv 후 13×17, Pool 후 6×8, 두 번째 Conv 후 3×4, AdaptivePool 후 1×1입니다.
4. 파라미터는 54 + BN12 + 두 번째 Conv550 + head22 = 638입니다. 첫 bias를 켜면 6개가 추가되므로 틀립니다.
5. 최소 해상도 `(1,1,4,7)`은 eval에서 검사합니다. 큰 입력만 통과하는 하드코딩을 제거하세요.

```bash
python check_process.py --case P07 --file process_starter.py
```

### 해설과 재도전

채널 수, 공간 크기, 배치 수는 서로 다른 축입니다. Conv는 채널을 바꾸고 Pool은 주로 공간을 줄이며 flatten(1)은 각 샘플의 특성만 펴 줍니다. BatchNorm2d(6)의 6은 해상도가 아닌 채널 수입니다.

참고 클래스 `SurfaceCNN`를 대조하세요. 재도전: 15×19 입력의 각 중간 크기를 먼저 계산한 뒤 hook 또는 임시 print로 확인합니다. Softmax를 마지막에 붙이지 않는 이유는 P04와 연결해서 설명하세요.

## P08. 양방향 GRU의 마지막 상태 결합

Process · 보통–어려움 · 권장 12분. ‘마지막 timestep 출력’과 ‘양방향 마지막 hidden state’를 구분합니다.

### 문제와 계약

`TripGRU(n_features, hidden_size, n_outputs)`를 구현하세요. 세 인수는 bool이 아닌 양의 정수이며 아니면 ValueError입니다. `self.gru`는 GRU, 2층, batch_first=True, bidirectional=True, dropout=0.1입니다. bias 등 나머지 인수는 기본값입니다.

입력은 float32 `(B,T,F)`, B/T≥1, F=n_features입니다. 마지막 층의 **정방향 최종 hidden, 역방향 최종 hidden**을 이 순서로 연결하고 `self.head = Linear(2*hidden_size,n_outputs)`로 raw 출력을 만듭니다. 출력 shape는 `(B,n_outputs)`입니다.

```text
TripGRU(4,5,3), 입력 (2,7,4) → 출력 (2,3)
총 파라미터 873개. eval 입력 (1,1,4)도 허용.
output[:, -1, :]를 head에 넣는 구현은 이 문제의 요구와 다릅니다.
```

### 힌트

GRU는 `output, h_n`을 반환합니다. h_n의 첫 축 순서는 층0정방향, 층0역방향, 층1정방향, 층1역방향입니다. 마지막 두 항목을 찾으세요.

### 치트시트 수정 매뉴얼

1. [지정 모델 가이드](/playbook/05/)와 [GRU/LSTM 블록](/cheatsheet.html#grulstm)을 엽니다. 범용 단방향 모델의 마지막 출력 코드를 그대로 쓰지 않습니다.
2. GRU 생성자의 num_layers=2, batch_first=True, bidirectional=True, dropout=.1을 대조합니다. 입력 feature 수는 생성자 인수를 사용합니다.
3. forward에서 `_, hidden = self.gru(x)`를 받고 `torch.cat([hidden[-2], hidden[-1]], dim=1)`을 head에 넣습니다.
4. bidirectional이므로 head 입력은 H가 아닌 2H입니다. eval에서 같은 GRU의 hidden을 따로 받아 만든 기대값과 비교하세요.
5. backward와 B=1/T=1을 검사합니다. 두 번째 방향 정보를 빼먹거나 층0의 hidden을 선택하면 shape가 맞아도 값이 틀립니다.

```bash
python check_process.py --case P08 --file process_starter.py
```

### 해설과 재도전

정방향은 마지막 시점에서 전체 순서를 읽었지만, 역방향은 반대쪽에서 읽습니다. `output[:, -1]`의 역방향 부분은 ‘역방향으로 전체를 읽은 최종 상태’가 아닙니다. h_n을 사용하면 명세가 원하는 두 방향의 최종 상태를 정확히 가져올 수 있습니다.

참고 클래스 `TripGRU`를 확인하세요. 재도전: hidden_size=3, n_outputs=1로 바꿔 B=1에서도 `(1,1)`이 유지되는지 검사합니다. 인수나 레이어를 바꾼 변형 문제는 원래 P08 채점의 파라미터 수와 일치할 필요가 없으므로 별도 파일에서 실험하세요.

## B01. 제조 품질 3분류 · Macro-F1

Problem · 입문 핵심 · 권장 45분. 표형 데이터 전처리부터 원래 라벨을 복원해 제출하는 첫 완주 문제입니다.

### 문제와 계약

독립 합성 센서 6개와 생산라인 범주로 품질 label 10/20/40을 예측하세요. `data/B01.npz`의 train 480행, valid 120행, test 97행은 이미 분리되어 있습니다. 각 X는 float32 `(N,6)`이고 NaN을 포함합니다. `category_*`는 `(N,)` 문자열이며 test에 train에서 못 본 D가 포함됩니다. y_train/y_valid는 `(N,)` 정수입니다.

반환은 test 배열 순서대로 `(97,)` 정수 원래 라벨입니다. 클래스 인덱스 0/1/2나 `(97,3)` 확률은 제출 답이 아닙니다. 평가 지표는 클래스 10/20/40을 모두 포함한 Macro-F1, 클수록 좋습니다. 신경망은 PyTorch를 사용하세요.

### 힌트

문제의 숫자 라벨이 10/20/40이라는 사실과 모델의 출력 노드가 3개라는 사실을 분리하세요. CrossEntropyLoss target은 0/1/2여야 합니다. test 예측 이후 원래 값으로 되돌립니다.

### 치트시트 수정 매뉴얼

1. [다중분류 가이드](/playbook/08/), [표형 전처리](/playbook/06/), [불균형 분류](/cheatsheet.html#8)를 순서대로 엽니다. `problem_starter.py`의 `solve`에서 `if case == "B01":` 분기를 만듭니다.
2. X 배열을 DataFrame으로 만들고 열 이름을 `pressure,current,temp,vibration,speed,load`로 부여한 뒤 `line`에 category를 붙입니다. ID는 입력에서 제외합니다.
3. `h.clean_tabular_values` → `h.make_preprocessor(train, encoding="onehot", scale_numeric=True)`를 사용합니다. train만 fit_transform, valid/test는 transform입니다. 출력이 sparse이면 `.toarray()` 후 float32로 바꿉니다. 범주 D는 학습 때 정의된 인코더의 unknown 처리에 맡기며 열을 임의 추가하지 않습니다.
4. 정렬된 `labels=np.array([10,20,40])`, `ytr=np.searchsorted(labels,data["y_train"])`로 인덱스를 만들고 valid도 같은 표를 사용합니다.
5. `model=h.MLP(Xtr.shape[1],3,hidden=(32,16),dropout=0)`, `task="multiclass"`로 합니다. 출력 수를 원본 열 개수 6이나 라벨 최댓값 40으로 설정하면 안 됩니다.
6. 아래 callback과 시작 매뉴얼의 공통 RUN을 연결합니다. callback의 truth는 변환 후 인덱스이므로 여기에서는 labels=[0,1,2]를 씁니다.

```python
from sklearn.metrics import f1_score
score_fn = lambda truth, raw: f1_score(
    truth, raw.argmax(1), labels=[0, 1, 2], average="macro", zero_division=0)
maximize = True
# 공통 RUN 이후 pred는 클래스 인덱스 (N,)
pred = labels[pred].astype(np.int64)
return pred
```

7. `python problem_starter.py --case B01 --output outputs/B01-v1.npz`로 실행하고 `check_submission.py`로 먼저 형식을 검사합니다. 마지막 소수 배치가 빠지지 않도록 test loader는 shuffle=False, drop_last=False를 유지합니다.

### 해설과 재도전

참고 풀이 `solutions/problem_solution.py`의 `tabular`, `CASE B01`, `COMMON RUN`, `SUBMISSION ADAPTER` 네 부분을 함께 읽으세요. CASE 부분만 복사하면 공통 변수·학습·반환이 빠집니다. ‘다수 클래스만 찍는’ 기준선의 valid Macro-F1을 계산하고 MLP가 이를 개선하는지 확인하세요. Accuracy만 좋아지고 희소 클래스 F1이 낮으면 클래스별 혼동을 봅니다.

첫 실행은 2 epoch로 연결 확인, 다음은 15 epoch입니다. hidden 크기 또는 learning rate 중 하나만 바꾸고 valid Macro-F1로 비교하세요. test 점수로 모델을 고르지 않습니다. 자주 나는 오류: CE에 40을 넣어 target out of bounds, one-hot train/test 열 불일치, 마지막에 0/1/2 제출, ID순 재정렬입니다.

변형 연습: 반환을 확률로 바꾸라는 가상의 문제라면 `(N,3)`와 열 순서 10/20/40을 명세에 추가해야 합니다. 원래 B01 검사기에 확률을 제출하면 실패하는 것이 맞습니다.

## B02. 양수 수요 예측 · RMSLE

Problem · 기본 · 권장 40분. log 변환으로 학습하되 제출은 원래 단위로 돌려주는 문제입니다.

### 문제와 계약

합성 운행 조건의 수치 6열과 범주 1열로 비음수 수요를 예측하세요. 데이터 구조·행 수는 B01과 같습니다. 단 y_train/y_valid는 비음수 float32 `(N,1)`이며 연속값도 가능합니다. 정수 반올림하라는 지시는 없습니다.

제출은 `(97,1)` 실수 비음수 원 단위 값입니다. 지표는 `sqrt(mean((log1p(y)-log1p(pred))**2))`, 작을수록 좋습니다. 표형 입력 결측은 train 통계로 처리하고 정답이나 test를 포함해 전처리를 fit하지 않습니다.

### 힌트

학습용 y를 log1p로 바꾸면 값의 크기 차이가 줄어듭니다. 그 상태의 모델 출력을 그대로 제출하면 단위가 틀립니다. 반대 함수는 exp가 아닌 expm1입니다.

### 치트시트 수정 매뉴얼

1. [표형 회귀·RMSLE](/playbook/06/)와 [RMSLE 코드](/cheatsheet.html#7-rmsle)를 엽니다. B01에서 데이터 전처리 부분만 가져오고 라벨 인코딩·argmax는 제거합니다.
2. `ytr=np.log1p(data["y_train"])`, valid도 동일하게 변환합니다. `(N,1)`을 유지합니다.
3. `h.MLP(Xtr.shape[1],1,hidden=(32,16),dropout=0)`, `task="regression"`, `maximize=False`로 설정합니다. 공통 루프는 MSELoss를 사용합니다.
4. callback의 truth는 이미 로그 단위입니다. 예측은 expm1 → 0 하한 → log1p 후 비교하면 제출 정책과 같은 RMSLE를 얻습니다.

```python
def score_fn(truth, raw):
    restored = np.clip(np.expm1(raw), 0, None)
    return float(np.sqrt(np.mean((truth - np.log1p(restored)) ** 2)))
# 공통 RUN 이후
pred = np.clip(np.expm1(pred), 0, None)
return pred  # shape (N,1), rounding/argmax 없음
```

5. [제출 검사](/playbook/13/)를 보고 `--case B02`로 실행합니다. `(97,)`이면 1열이 사라진 원인을 찾으세요. 모든 출력을 강제로 reshape해서 원인을 숨기지 않습니다.

### 해설과 재도전

참고 풀이의 `CASE B02`와 제출 어댑터를 확인하세요. ‘학습 loss를 계산하는 단위’와 ‘저장하는 단위’를 종이에 적는 습관이 중요합니다. clipping은 이 문제의 비음수 조건에 따른 예측 정책이지 모든 회귀에서 쓰는 만능 규칙이 아닙니다.

기준선은 train의 log1p target 평균을 모든 valid 행에 예측한 뒤 expm1로 복원해 계산합니다. MLP가 이 기준선을 개선하는지 확인하세요. 재도전: raw y MSE로 학습한 모델과 log1p 모델을 **동일한 valid RMSLE**로 비교합니다. 각각 다른 지표로 평가하면 결론을 비교할 수 없습니다.

자주 나는 오류: target을 StandardScaler와 log로 이중 변환 후 복원 한 번만 하기, exp 사용으로 1이 더해지기, 제출 y가 아닌 로그를 저장하기, 정답 `(N,1)`과 예측 `(N,)`를 빼서 NxN broadcasting하기입니다.

## B03. 센서 시계열 · 6시점 뒤 3개 값 예측

Problem · 실전 핵심 흐름 · 권장 80–100분. raw 시계열을 창으로 만들고 미래 정답 정렬·시간 분리·원 단위 평가까지 연결합니다.

### 문제와 계약

`data/B03.npz`의 `raw_X`는 시간순 `(1400,6)` 실수 센서입니다. 입력 끝 위치 e에서 **e-11부터 e까지 12행**을 사용해 **e+6 한 시점의 target 3개**를 예측합니다. 6개 미래 시점 전체를 예측하는 문제가 아닙니다. 각 origin에서는 그 시점까지의 관측만 사용할 수 있는 rolling-origin 설정입니다.

`ends_train`은 11…833의 823개, `ends_valid`는 840…1113의 274개, `ends_test`는 1120…1393의 274개가 섞인 순서입니다. y_train/y_valid는 각각 ends 순서에 대응하는 `(N,3)` 원 단위 정답입니다. 제공 분할을 유지하세요. raw_X의 뒷부분을 미리 읽어 특성으로 쓰면 안 됩니다.

반환은 test ends 순서대로 `(274,3)` 실수 원 단위 예측입니다. 지표는 **전체 N×3 원소의 MSE**입니다. 열별 MSE의 합, 표준화된 MSE, RMSE는 이 문제의 지표가 아닙니다. 과거 context는 분할 경계를 가로질러도 되지만 학습 정답 시점은 다음 분할의 첫 예측 origin보다 앞서야 합니다.

```text
e=11 → 입력 raw_X[0:12], 정답 시점 17
e=12 → 입력 raw_X[1:13], 정답 시점 18
train 마지막 정답 시점 833+6=839 < valid 첫 origin 840
valid 마지막 정답 시점 1113+6=1119 < test 첫 origin 1120
```

### 힌트

먼저 모델 없이 창 2개를 만들고 위 인덱스와 일치하는지 확인하세요. 창을 만든 뒤 무작위 split하면 인접 관측·미래 정답이 섞입니다. CNN1D 입력은 이 템플릿에서는 `(N,T,F)`를 받습니다.

### 치트시트 수정 매뉴얼

1. [raw 시계열 가이드](/playbook/10/) → [이미 만든 window](/playbook/09/) → [다중 target 회귀](/playbook/06/) 순서로 엽니다.
2. `L=12`, `H=6`, 출력 수 3을 메모합니다. 공통 코드의 변수 LOOKBACK/HORIZON이 있다면 이 값으로 바꾸세요. X 끝 위치와 target 위치를 혼동하지 않습니다.

```python
raw = data["raw_X"]
L, H = 12, 6
def window(ends):
    return np.stack([raw[e-L+1:e+1] for e in ends]).astype(np.float32)
Xtr, Xva, Xte = [window(data[f"ends_{s}"]) for s in ("train", "valid", "test")]
assert (data["ends_train"] + H).max() < data["ends_valid"].min()
assert (data["ends_valid"] + H).max() < data["ends_test"].min()
```

3. [sequence scaling](/cheatsheet.html#sequence-scaling)의 `h.fit_scale_3d(Xtr)`로 train 특성 통계를 얻고 세 split에 같은 `h.transform_scale_3d`를 적용합니다. 겹친 train 창의 값이 반복 반영되는 것은 이 참고 구현의 정규화 정책이며 미래 데이터는 사용하지 않습니다.
4. target은 `StandardScaler().fit(y_train)`을 별도로 보관합니다. train/valid를 transform한 `(N,3)` float32로 학습합니다. X scaler와 y scaler는 다른 물건입니다.
5. [CNN1D](/cheatsheet.html#cnn1d-window-sequence)의 `h.CNN1D(6,3,channels=(16,32),dropout=0)`, task regression으로 시작합니다. 템플릿 내부에서 채널 축으로 transpose하므로 외부에서 다시 `(N,F,T)`로 바꾸지 않습니다.
6. callback에서 truth와 raw를 **둘 다 target_scaler.inverse_transform**한 다음 전체 원소 MSE를 계산하고 maximize=False로 둡니다. 세 target의 크기가 달라 표준화 MSE와 원 단위 MSE의 모델 순위는 달라질 수 있습니다.
7. 공통 RUN의 pred를 inverse_transform하고 float32로 반환합니다. test ends를 정렬하거나 target 축을 평균해서 없애지 않습니다.
8. `--case B03`으로 저장·reload 검사 후, valid 기준선과 비교하고 최종 자가채점합니다.

### 해설과 재도전

참고 풀이의 `CASE B03`은 창 생성, 두 경계의 purge 검사, X/y scaling, 원 단위 callback을 한 블록에 보여줍니다. 정답은 제공된 `y_*`를 사용하며 raw_X의 미래행으로 직접 답을 만드는 코드가 아닙니다.

먼저 train y의 열별 평균을 valid 전체에 반복하는 `(N,3)` 기준선을 만드세요. 마지막 X의 첫 세 열을 곧바로 y로 반환하는 것은 이 문제에서 서로 단위·의미가 다른 센서라 올바른 persistence 기준선이 아닙니다.

재도전: window 전체 평균 특성을 이용한 MLP와 CNN1D를 동일 분할·원 단위 MSE로 비교합니다. LSTM/GRU로 바꾸려면 입력 `(N,12,6)`과 출력 `(N,3)` 계약은 유지합니다. 새 horizon으로 문제 자체를 바꾸려면 데이터와 정답도 다시 정의해야 하므로 H 숫자만 바꾸고 기존 y를 쓰면 틀립니다.

**실수 복구:** 점수가 비정상적으로 좋으면 미래행·test 통계·누수부터 확인합니다. 손실이 shape 경고 없이 계산되어도 `(N,1,3)`과 `(N,3)`가 broadcast될 수 있으니 정확한 동등 shape를 assert하세요. test 순서가 섞여 있는 것은 의도한 함정입니다.

## B04. RGB 표면 이미지 3분류

Problem · 기본 · 권장 45분. 파일 로딩 대신 작은 합성 이미지 배열로 NHWC/NCHW 변환과 분류를 익힙니다.

### 문제와 계약

독립 생성 RGB 패턴 이미지로 label 0/1/2를 예측하세요. train 240장, valid 90장, test 97장, X shape는 `(N,16,20,3)`, dtype uint8, 값 0…255입니다. y_train/y_valid는 `(N,)` 정수입니다. 분할 간 동일 원본을 재사용하지 않습니다. 실제 불량 사진 데이터는 아닙니다.

반환은 `(97,)` 정수 label이며 지표는 Accuracy입니다. 학습·추론은 PyTorch. 입력의 색상과 채널 정보는 보존하고 validation/test에는 무작위 증강을 적용하지 마세요.

### 힌트

Conv2d는 `(배치,채널,높이,너비)`를 받습니다. 입력 배열의 마지막 3을 두 번째 축으로 옮겨야 합니다. dtype만 float로 바꾸는 것과 255로 나누는 것은 다른 작업입니다.

### 치트시트 수정 매뉴얼

1. [이미지 Problem](/playbook/11/)와 [NumPy image를 NCHW로](/cheatsheet.html#numpy-image-nchw)를 엽니다.
2. `h.prepare_numpy_images(data["X_train"], "NHWC", True)`를 사용하고 valid/test에도 같은 결정적 변환을 적용합니다. 세 번째 True는 이 팩의 uint8 값을 255로 나누는 선택입니다. 이미 0~1인 데이터라면 같은 설정을 맹목적으로 쓰지 않습니다.
3. `h.SmallImageCNN(3,3)`, task multiclass로 둡니다. 첫 3은 RGB 입력 채널, 두 번째 3은 클래스 수입니다. 여기서는 우연히 같을 뿐입니다.
4. y는 `(N,)` long, logits는 `(N,3)`입니다. callback은 `mean(truth == raw.argmax(1))`, maximize=True입니다.
5. 공통 RUN은 return_proba=False로 label `(N,)`를 반환하므로 별도 sigmoid나 softmax를 붙이지 않습니다.
6. `--case B04`로 실행합니다. 마지막 33장도 남김없이 예측했는지 97개인지 확인합니다. 이 작은 문제는 CPU로도 완주할 수 있습니다.

### 해설과 재도전

참고 풀이 `CASE B04`는 외부 다운로드·사전학습 가중치가 필요 없습니다. 단순 색상 패턴을 의도적으로 사용했으므로 높은 점수가 실제 산업 이미지 문제의 준비 완료를 뜻하지 않습니다. 이 문제의 목표는 배열 계약과 학습 흐름입니다.

재도전: 학습에만 작은 밝기 변화 증강을 넣고 valid는 고정합니다. 데이터 분할 전에 이미지를 복제·증강해서 두 split에 같은 원본이 들어가면 안 됩니다. 세 클래스가 같은 수라고 가정하지 말고 valid의 다수 클래스 Accuracy도 기준선으로 계산하세요.

자주 나는 오류: 입력을 `(N,H,W,C)` 그대로 Conv에 넣기, transpose 두 번, /255 두 번, validation에서 random crop, 마지막에 확률 행렬 제출입니다.

## B05. 정상 데이터만으로 이상 점수 만들기

Problem · 응용 · 권장 50분. Autoencoder의 재구성 값과 최종 이상 점수가 다른 것을 익힙니다.

### 문제와 계약

센서 6차원의 정상 상태만 있는 train `(600,6)`과 정상 valid `(120,6)`, 정상/이상이 섞인 test `(127,6)`을 제공합니다. 모두 float32이고 결측은 없습니다. train/valid에는 y가 없습니다. 작은 latent 공간으로 복원하는 PyTorch Autoencoder를 학습하세요.

반환은 표준화된 test 각 샘플의 **6특성 평균 제곱 재구성 오차**, shape `(127,)`, 비음수 실수입니다. 큰 값일수록 이상이라고 정의합니다. 자가채점 지표는 이상=1을 양성으로 한 ROC-AUC입니다. 확률이나 threshold를 적용한 0/1 라벨을 요구하지 않습니다.

### 힌트

Autoencoder 학습 정답은 X 자신입니다. 정상 데이터를 잘 복원하는 작은 모델을 만든 뒤, test 입력과 복원값의 차이를 특성 축에서 평균합니다. 정상 valid만으로는 ROC-AUC를 계산할 수 없습니다.

### 치트시트 수정 매뉴얼

1. [정상-only AE 가이드](/playbook/12/)와 [Autoencoder 이상 탐지](/cheatsheet.html#14-autoencoder)를 엽니다.
2. `StandardScaler().fit(data["X_train"])` 후 train/valid/test를 transform합니다. 표준화된 공간의 error를 제출하는 계약이므로 뒤에서 inverse_transform하지 않습니다.
3. `ytr=Xtr.copy()`, `yva=Xva.copy()`, task regression으로 둡니다. label이 없다고 모두 0 정답을 만들어 주면 복원을 배우지 못합니다.
4. `h.Autoencoder(6,latent_dim=2,hidden_dim=16)`을 사용합니다. 정상 valid의 복원 MSE 최소화로 early stopping합니다. maximize=False이며 여기의 callback은 AUC가 아니라 MSE입니다.
5. 공통 RUN의 pred는 아직 `(N,6)` 재구성입니다. 최종 어댑터가 필요합니다.

```python
pred = np.mean((Xte - pred) ** 2, axis=1)
return pred  # (127,), 큰 값=이상. threshold와 sigmoid 없음
```

6. `--case B05`로 형식을 확인합니다. test 정답은 마지막 자가채점에서만 열립니다. 모든 score가 같으면 AUC가 보통 0.5이며 구분을 하지 못하는 상태입니다.

### 해설과 재도전

참고 풀이 `CASE B05`는 정상 valid의 복원오차로만 모델을 고릅니다. AUC를 최적화할 혼합 검증 라벨이 없으므로 ‘valid AUC가 최고인 모델’을 골랐다고 말할 수 없습니다. 모델 용량을 지나치게 키우면 이상 입력도 잘 복원할 수 있습니다.

재도전: latent_dim=1/2/4 중 두 개만 선택해 정상 valid MSE와 score 분포를 비교합니다. 낮은 정상 복원 MSE가 항상 높은 이상탐지 AUC를 보장하지 않는 이유를 설명하세요. threshold가 필요한 별도 문제라면 정상 valid 분위수 등으로 먼저 정하고 test에 적용합니다. **이 B05 제출은 threshold 없는 score**입니다.

자주 나는 오류: test도 train에 합치기, `y_train` 키를 찾다가 중단, error를 배치 축 axis=0으로 평균, 오차가 작을수록 이상이라고 부호 반전, 임의 sigmoid로 정의된 재구성 오차를 변경하기입니다.

## B06. 고장 확률 · 양성 라벨이 2인 이진분류

Problem · 기본 · 권장 40분. 숫자가 큰 클래스가 양성이라는 무의식적 가정을 깨는 문제입니다.

### 문제와 계약

B01과 같은 표형 구조·행 수의 합성 데이터입니다. y의 원래 값은 **고장=2, 정상=5**입니다. 요구하는 것은 고장일 확률 `P(y=2)`입니다. train/valid는 두 클래스를 모두 포함하며, 입력 결측과 test의 미지 범주 D가 있습니다.

반환은 `(97,1)` float 0~1 확률입니다. 지표는 고장=2를 양성으로 계산한 ROC-AUC, 클수록 좋습니다. 2/5 라벨, 0/1 결정, 정상일 확률, 두 열짜리 확률은 이 문제의 출력이 아닙니다.

### 힌트

정답을 `(y==2).astype(float)`로 바꾸세요. `LabelEncoder` 기본 정렬 결과만 믿으면 2가 0에 대응하므로 양성 의미가 뒤집힐 수 있습니다.

### 치트시트 수정 매뉴얼

1. [이진분류·확률 제출](/playbook/07/)와 [shape/loss 표](/cheatsheet.html#2-pytorch-shapeloss)를 엽니다. 입력 전처리는 B01의 train-only 함수를 재사용합니다.
2. `ytr=(data["y_train"]==2).astype(np.float32).reshape(-1,1)`로 만들고 valid도 동일하게 적용합니다.
3. `h.MLP(Xtr.shape[1],1,hidden=(32,16),dropout=0)`, task binary로 둡니다. 학습 loss는 BCEWithLogitsLoss이므로 모델 끝에 Sigmoid를 붙이지 않습니다.
4. AUC callback에서 raw logits를 sigmoid한 값과 0/1 truth를 비교합니다. `maximize=True`입니다. AUC 계산을 위해 threshold .5로 자르면 순위 정보를 잃습니다.

```python
import torch
from sklearn.metrics import roc_auc_score
def score_fn(truth, raw):
    probability = torch.sigmoid(torch.from_numpy(raw)).numpy()
    return roc_auc_score(truth.reshape(-1), probability.reshape(-1))
```

5. 공통 RUN의 `return_proba=True`가 inference에서 sigmoid를 한 번 적용합니다. 그 결과를 다시 sigmoid하거나 라벨로 복원하지 말고 그대로 반환합니다.
6. `--case B06`으로 검사합니다. 평균 확률이 0.5 근처인지만 보지 말고 valid의 양성/음성 score 분포와 AUC 방향을 확인하세요.

### 해설과 재도전

참고 풀이 `CASE B06`와 `predict_torch`의 binary 확률 경로를 함께 확인하세요. ‘BCE loss에는 logits, 제출에는 확률’이 핵심입니다. 이 문제는 확률 보정 성능이 아니라 AUC로 순위를 평가하지만 반환 계약은 0~1 확률입니다.

상수 확률 기준선의 AUC는 0.5입니다. 재도전: valid에서 p와 1-p의 AUC를 비교해 양성 방향을 확인합니다. p가 좋지 않다고 test 점수를 보고 뒤집는 방식은 금지하세요. 부호의 의미는 문제의 label 정의에서 결정해야 합니다.

자주 나는 오류: 5를 양성으로 학습, BCE target이 정수 long, 모델 안팎 sigmoid 두 번, return_proba=False로 threshold 적용, `(N,)`로 저장입니다.

## B07. 동시에 여러 상태가 켜지는 multilabel

Problem · 기본–응용 · 권장 45분. 세 클래스 중 하나를 고르는 문제와 세 개 독립 상태를 예측하는 문제를 구분합니다.

### 문제와 계약

B01과 같은 수치 6열·범주 1열·행 수입니다. y_train/y_valid는 float32 `(N,3)`의 0/1 indicator입니다. 열 순서는 과열, 진동, 과부하이며 한 행에서 둘 이상이 1이거나 전부 0일 수 있습니다. 각 분할의 각 label 열에 0과 1이 모두 있습니다.

반환은 `(97,3)` float 확률이며 열 순서는 그대로입니다. 각 원소는 0~1이어야 하지만 **행 합이 1일 필요는 없습니다**. 평가 지표는 label별 ROC-AUC를 평균한 Macro ROC-AUC입니다. argmax·one-hot·단일 라벨 제출은 요구하지 않습니다.

### 힌트

출력이 세 개라고 무조건 CrossEntropyLoss는 아닙니다. 이 문제는 각 노드가 독립적인 이진 질문을 답하므로 BCEWithLogitsLoss를 사용합니다.

### 치트시트 수정 매뉴얼

1. [다중분류와 multilabel](/playbook/08/)을 엽니다. B01과 비교하면서 차이를 세 군데 표시하세요: y shape, task/loss, inference.
2. 표형 전처리는 B01과 동일합니다. y는 이미 `(N,3)` float이므로 searchsorted·LabelEncoder·argmax를 적용하지 않습니다.
3. `h.MLP(Xtr.shape[1],3,hidden=(32,16),dropout=0)`, `task="multilabel"`로 설정합니다. 모델 출력은 raw `(N,3)`입니다.
4. callback은 logits에 sigmoid 후 `roc_auc_score(truth, probability, average="macro")`, maximize=True입니다. label 한 열이 valid에 한 값만 있으면 AUC가 정의되지 않으므로 학습 전에 분포를 검사합니다.
5. 공통 RUN에서 return_proba=True를 사용해 `(N,3)` sigmoid 확률을 반환합니다. 행 합을 1로 정규화하지 않습니다.
6. `--case B07`로 저장합니다. 세 열의 순서를 알파벳순으로 바꾸거나 ‘가장 큰 확률’ 한 열만 남기면 잘못된 제출입니다.

### 해설과 재도전

참고 풀이 `CASE B07`을 B01과 나란히 읽으세요. B01의 각 행에는 하나의 정수 정답이 있고 B07에는 세 개의 독립적인 정답이 있습니다. 같은 출력 노드 수 3이어도 loss와 inference가 달라집니다.

재도전: `[0.8,0.7,0.2]`가 유효한 예측인 이유를 설명하세요. threshold .5를 가정하면 두 상태가 켜지지만 B07의 제출은 threshold 적용 전 확률입니다. valid에서 label별 AUC도 출력해 평균에 가려진 약한 label을 찾으세요. threshold별 F1은 별도 진단이지 이 문제의 모델 선택 지표가 아닙니다.

자주 나는 오류: softmax로 상태 간 경쟁 강제, CE로 y를 argmax 변환, BCE에 long y 전달, 정확도만 모니터링, 클래스 확률 합=1 assert를 재사용하는 것입니다.
