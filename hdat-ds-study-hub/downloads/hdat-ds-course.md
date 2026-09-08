# HDAT-DS 완전정복 강의 — 필기부터 PyTorch 실기까지

> 기준일: 2026-08-20  
> 목표: 학부 수준의 머신러닝·딥러닝 이론을 **50분 필기 20문항**과 **170분 실기 Process 8문항 + Problem 1문항**에서 실제 점수로 바꾸는 것  
> 딥러닝 프레임워크: **PyTorch만 사용**. TensorFlow/Keras 코드는 사용하지 않는다.  
> 주의: 아래 문제는 학습을 위해 새로 만든 독자 문제다. 공식 연습문제를 복제하거나 재구성한 것이 아니다.

[TOC]

---

## 이 교재를 사용하는 법

이 교재는 사전식 요약본이 아니라 **1강부터 순서대로 읽는 강의**다. 각 강의는 다음 구조를 따른다.

1. `학습 목표`: 공부가 끝난 뒤 할 수 있어야 하는 행동
2. `핵심 이론`: 필기에서 선택지를 판별하는 수준의 원리
3. `실기 연결`: 같은 개념이 코드와 데이터에서 어떻게 나타나는지
4. `실습`: 직접 실행하거나 종이에 계산할 문제
5. `정답·해설`: 결과뿐 아니라 틀린 접근이 왜 틀렸는지 설명
6. `완료 기준`: 다음 강의로 넘어가도 되는지 확인

권장 루틴은 `읽기 → 책을 덮고 3줄 요약 → 코드 직접 입력 → 문제 풀이 → 오답 기록`이다. 코드를 눈으로만 읽으면 실기 속도가 생기지 않는다. 모든 텐서 옆에 shape를 주석으로 적는 습관을 들인다.

표기 규칙:

- `[필기]`: 폐쇄형 필기에서 정의·수식·비교를 묻기 좋은 내용
- `[Process]`: 함수 구현, shape 계산, 명세 준수와 연결되는 내용
- `[Problem]`: 데이터 전체를 분석하고 모델·검증·제출까지 완주하는 내용
- `[함정]`: 오답 선택지나 hidden test에서 자주 실패하는 지점
- 텐서 shape는 배치 `B`, 시점 `T`, 특성 `F`, 채널 `C`, 높이 `H`, 너비 `W`, 클래스 `K`로 쓴다.

---

## 1. 시험 지도를 먼저 그리기

### 1.1 학습 목표

이 강의를 마치면 다음을 할 수 있어야 한다.

- 시험의 세 영역이 각각 무엇을 평가하는지 한 문장으로 설명한다.
- 공식 범위를 10개 학습 묶음으로 분류한다.
- 실기 170분 동안 Process와 Problem을 모두 제출하는 시간표를 만든다.
- 오픈북이 허용하는 것과 허용하지 않는 것을 구분한다.

### 1.2 시험 구조

2026 공식 안내와 공개 연습자료를 기준으로 한 구조는 다음과 같다. 회차별 세부 시각과 규정은 시험 직전에 공식 매뉴얼에서 다시 확인한다.

| 영역 | 구성 | 시간 | 핵심 평가 |
|---|---:|---:|---|
| 필기 | 선다형 20문항 | 50분 | 개념 비교, 수식·shape 계산, 모델과 학습 원리 |
| 실기 Process | 8문항 | Problem과 합쳐 170분 | 요구된 작은 함수·모델을 정확한 명세로 구현 |
| 실기 Problem | 1문항 | Process와 합쳐 170분 | EDA→전처리→검증→모델→예측 파일까지 완주 |

공식 운영상 코드 실행과 모델 학습에 걸리는 시간도 실기 170분에 포함되고, 실행·학습 지연을 이유로 시간이 연장되지 않는다. 그래서 epoch 수, patience, 데이터 크기와 wall-clock time cap까지 풀이의 일부로 설계해야 한다.

필기는 평균 2분 30초에 한 문제다. 긴 유도보다 **정의, 방향, 출력 범위, shape, 장단점**을 빠르게 판별해야 한다. Process는 코드가 짧아도 함수명, 반환형, 원본 보존, 경계조건을 틀리면 공개 예시 밖 입력에서 실패한다. Problem은 가장 화려한 모델보다 **유효한 제출 파일을 먼저 확보하는 능력**이 중요하다.

2026 공식 FAQ가 안내한 주요 버전은 NumPy 1.26.4, pandas 2.2.3, scikit-learn 1.5.2, PyTorch 2.7.0(CUDA 11.8)이다. torchvision은 PyTorch와 호환 버전으로 안내된다. 시험 직전 공식 FAQ를 다시 확인하고, 추가 package 설치가 없어도 핵심 풀이가 돌아가게 준비한다. 이 교재의 딥러닝 코드는 PyTorch만 사용한다.

### 1.3 공식 범위를 학습 묶음으로 바꾸기

공식 공개 범위를 준비 관점에서 다시 묶으면 다음과 같다.

| 묶음 | 포함 내용 | 이 교재 |
|---|---|---|
| 수학 원리 | 선형대수, 미분, 확률·통계, 거리, 정보이론 | 4~6강 |
| 데이터 처리 | EDA, 결측·이상치, 인코딩, 스케일링, 증강 | 2~3, 7~8강 |
| 특성 | Feature Selection, Extraction, PCA/LDA | 9, 12강 |
| 고전 ML | 회귀, KNN, 트리, 앙상블, SVM, 군집 | 10~12강 |
| 검증·평가 | split, CV, leakage, metric, imbalance | 13~14강 |
| 규제·최적화 | L1/L2, dropout, BN, optimizer, scheduler | 15, 18강 |
| 기본 신경망 | 퍼셉트론, MLP/DNN, 활성·손실함수 | 16~18강 |
| CNN 계열 | CNN, pooling, ResNet, 전이학습 | 19~20강 |
| 순차 모델 | window, RNN/LSTM/GRU, Transformer | 21~23강 |
| 생성·응용 | Autoencoder, 이상탐지, VAE, GAN | 24~25강 |
| 시험 실행 | Process, Problem, 디버깅, 제출 | 26~34강 |

강화학습은 제공 교육자료에는 있으나 현재 공개된 핵심 출제범위에서 우선순위가 낮다. 시간이 부족하면 본 교재의 범위를 먼저 완성한다.

### 1.3A 학습 패러다임의 큰 그림

- 지도학습: `(x,y)`가 있고 target을 예측한다. 연속값이면 회귀, 서로 배타적인 label이면 분류, 여러 label이 동시에 참이면 multilabel이다.
- 비지도학습: 명시적 target 없이 구조·표현을 찾는다. 군집화, PCA, Autoencoder representation이 대표적이다.
- 자기지도학습: 데이터 자체에서 예측 target을 만든다. masked input 복원, 다음 시점 예측 등이 있다.
- 강화학습: agent가 state에서 action을 선택하고 reward 누적을 최대화한다. 현재 공개 핵심범위 대비 후순위다.

전통적 머신러닝은 사람이 만든 feature와 비교적 얕은 모델을 쓰는 경우가 많고, 딥러닝은 여러 층이 representation까지 end-to-end로 학습한다. 데이터 종류와 규모에 따라 고전 ML이 더 나을 수 있으므로 “딥러닝이 항상 상위”라고 생각하지 않는다.

### 1.4 오픈북의 정확한 의미

실기는 공식 안내가 허용한 범위의 단방향 검색만 이용하고, GitHub·Colab·Kaggle·Notion, 생성형 AI·AI 검색요약, 메신저·메일·협업 서비스 등 금지 수단은 사용하지 않는다. 개인 로컬 자료의 허용 여부도 회차별 규정과 감독관 안내를 확인하며, 데이터는 제공 IDE 밖으로 내려받아 처리하지 않는다.

오픈북의 효과는 “모르는 것을 시험 중 학습”하는 데 있지 않다. 다음을 빠르게 찾는 데 있다.

- loss와 출력층의 조합
- convolution 출력 크기 공식
- window index와 time/group split 코드
- 자주 쓰는 모델 뼈대
- prediction shape·NaN·파일명 검증 코드

### 1.5 170분 기본 운영안

| 시각 | 행동 | 종료 조건 |
|---:|---|---|
| 0~8분 | Process/Problem 전체 훑기, 변수·파일·metric 기록 | 문제 지도 완성 |
| 8~55분 | 확실한 Process부터 풀이 | 쉬운 문항 제출·저장 |
| 55~65분 | Problem 데이터 계약 확인, 가장 빠른 baseline | 첫 예측 파일 생성 |
| 65~125분 | Problem 검증·모델 개선 | 공식 metric 비교 |
| 125~150분 | 남은 Process와 자가 edge-case test | 8문항 모두 입력 |
| 150~163분 | Problem 최종 재학습·예측 | 정확한 shape 파일 |
| 163~170분 | reload 검사, Ctrl+S, 양쪽 제출 | 제출 상태 확인 |

상황에 따라 바꿀 수 있지만, **Problem 첫 제출 파일을 끝까지 미루지 않는다.**

### 1.6 확인문제

1. Process에서 성능이 아니라 명세 준수가 중요한 이유를 두 가지 쓰라.
2. 1.5의 170분 연습 계획에서 첫 baseline 구간이 끝나는 약 65분 시점까지 확보하려는 산출물은 무엇인가? 이 시각이 공식 채점 조건인가?
3. 실기에서 생성형 AI가 금지되어 있다면, 이 교재는 언제 어떻게 사용해야 하는가?
4. 필기 한 문항에 평균 몇 분을 쓸 수 있는가?

#### 정답·해설

1. 함수명·반환형·shape·경계조건이 명세와 다르면 공개 예시 밖 입력에서 실패할 수 있고, 요구한 동작과 다른 “더 좋은” 구현도 오답이 될 수 있기 때문이다.
2. 최소한 실행 가능한 baseline과 요구된 이름·shape·행 순서의 예측 파일이다. 약 65분은 본문 연습 계획의 누적 목표이며 공식 채점 조건이 아니다.
3. 시험 전에 충분히 학습하고, 시험 중에는 해당 회차 공식 안내가 허용한 검색·자료 범위에서만 짧은 공식·코드 조각을 찾는다.
4. `50 / 20 = 2.5분`, 즉 2분 30초다.

### 1.7 완료 기준

- [ ] 세 영역을 시간·문항 수와 함께 말할 수 있다.
- [ ] 공식 범위 10개 묶음을 보지 않고 적을 수 있다.
- [ ] 나만의 170분 계획을 종이에 썼다.

---

## 2. Python·Jupyter·shape 언어

### 2.1 학습 목표

- Python 함수의 입력 계약과 반환 계약을 명시한다.
- list, NumPy array, DataFrame, Tensor의 역할을 구분한다.
- indexing, slicing, copy/view, broadcasting의 함정을 피한다.
- 오류 메시지에서 최초의 shape 불일치를 찾는다.

### 2.2 시험 코드의 네 가지 객체

| 객체 | 강점 | 대표 shape 접근 | 주의점 |
|---|---|---|---|
| Python `list` | 가변 길이, 일반 객체 | `len(x)` | 수치 연산이 느리고 broadcasting 없음 |
| `np.ndarray` | 빠른 수치·저장 | `x.shape`, `x.dtype` | view가 생길 수 있음 |
| `pd.DataFrame` | 열 이름·혼합 dtype | `x.shape`, `x.dtypes` | 문자열·결측 혼합 시 object/string 함정 |
| `torch.Tensor` | autograd·GPU·신경망 | `x.shape`, `x.dtype`, `x.device` | dtype/device/축 순서를 맞춰야 함 |

실전에서 가장 먼저 출력할 것은 값 전체가 아니라 다음 다섯 가지다.

```python
def inspect(name, x):
    print(name)
    print("type:", type(x))
    print("shape:", getattr(x, "shape", None))
    print("dtype:", getattr(x, "dtype", None))
    print("head/sample:", x[:2] if hasattr(x, "__getitem__") else None)
```

### 2.3 축을 문장으로 읽기

`X.shape == (64, 20, 23)`이 시계열 mini-batch라는 계약이면 “한 배치에 샘플 64개, 각 샘플은 과거 20시점, 시점당 23특성”이라고 읽는다. shape만으로 축의 의미가 정해지는 것은 아니다. `Linear`는 마지막 축을 feature로 본다. `Conv1d`는 `[B,C,T]`, `RNN/LSTM/GRU`에 `batch_first=True`를 쓰면 `[B,T,F]`, `Conv2d`는 `[B,C,H,W]`다.

기호부터 낯설다면 [입문 6강: shape를 처음부터 읽기](https://markshincuhk.github.io/hdat-ds-study-hub/start/06/#start-06-1)를 먼저 읽는다. B=배치 안의 샘플 수, F=특성 수, T=시간 단계 수, C=채널 수, H/W=이미지 높이/너비, K=여기서는 클래스 수다. 전체 샘플 수 N과 B는 다르다. 다른 절에서 같은 글자를 은닉 크기·커널 크기 등에 쓸 때는 그 절의 정의를 따른다.

```text
표형 MLP       [B, F]
시계열 배치    [B, T, F]  ← window로 묶은 샘플 B개
Conv1d 입력    [B, F, T]  ← permute(0, 2, 1)
RNN 입력       [B, T, F]  ← batch_first=True일 때
이미지 Conv2d  [B, C, H, W]
다중분류 logit [B, K]     ← 샘플마다 클래스 점수 K개
```

원본 시계열 CSV는 `[전체 시점 수,F]`인 긴 표일 수 있다. window를 만들고 배치로 묶은 뒤 `[B,T,F]`가 된다. RNN의 기본값 `batch_first=False`에서는 `[T,B,F]`를 받으며, 은닉 상태는 별도의 축 순서를 따른다. logits는 확률이 아니며 `argmax(dim=1)`로 샘플마다 클래스 번호 하나를 고르면 `[B]`가 된다.

`reshape`는 원소 수를 바꾸지 않는다. `permute(0,2,1)`는 기존 축을 0번·2번·1번 순서로 놓는다. 시간·특성 축 교환을 reshape로 대체하지 않는다. broadcasting은 축 교환이 아니라 원소별 계산에서 값을 반복 적용하는 규칙이다. [같은 숫자로 비교하는 예제](https://markshincuhk.github.io/hdat-ds-study-hub/start/06/#start-06-5)에서 값의 대응까지 확인한다.

### 2.4 indexing과 경계

Python slice의 끝은 포함되지 않는다.

```python
import numpy as np

x = np.arange(10)
assert x[2:5].tolist() == [2, 3, 4]
assert x[-1] == 9
assert x[:, None].shape == (10, 1)
```

Boolean mask의 길이는 대상 행 수와 같아야 한다. DataFrame에서는 label index와 positional index를 구분한다.

```python
# label 기반: 인덱스 이름을 사용
row_a = df.loc["sample_a"]

# 위치 기반: 0번째 행을 사용
row_0 = df.iloc[0]
```

### 2.5 copy와 view

원본을 보존하라는 함수에서 가장 안전한 시작은 `out = df.copy()`다. NumPy slicing은 원본을 가리키는 view일 수 있다.

```python
a = np.arange(5)
b = a[1:4]       # view일 수 있음
b[0] = 999
assert a[1] == 999

c = a[1:4].copy()
c[0] = -1
assert a[1] == 999
```

[Process 함정] 요구가 “입력 DataFrame을 변경하지 말고 반환”인데 입력 자체를 수정하면 보이는 예시는 통과해도 hidden test에서 실패할 수 있다.

### 2.6 broadcasting

원소별 덧셈·뺄셈 등의 broadcasting은 뒤쪽 축부터 크기가 같거나 한쪽이 1이면 가능하다. 없는 왼쪽 축은 크기 1로 맞춰 읽는다. 숫자 하나를 배열의 모든 원소에 더하는 것도 broadcasting이다. 반복 적용은 개념적인 설명이며 반복된 입력 전체를 복사할 필요는 없다. [입문 6강의 손계산·PyTorch 예제와 회귀 함정](https://markshincuhk.github.io/hdat-ds-study-hub/start/06/#start-06-6)으로 확인한다.

- `[B,F] + [F] → [B,F]`
- `[B,T,F] + [F] → [B,T,F]`
- `[B,T,F] + [T]`에서 `[T]`는 마지막 축 `F`와 비교된다. T와 F가 다르고 둘 다 1이 아니면 불가능하다. T=F이면 실행되어도 시간별 값이 아니라 특성별 값처럼 적용된다.
- 시점별 값을 더하려면 `[1,T,1]`로 만든다.

의도하지 않은 broadcasting은 오류 없이 잘못된 loss를 만든다. 예를 들어 예측 `[B,1]`과 정답 `[B]`를 MSE에 넣으면 `[B,B]`로 확장될 수 있다. 회귀에서는 둘 다 `[B]` 또는 둘 다 `[B,1]`로 맞춘다.

### 2.7 함수 계약

함수를 쓰기 전 주석으로 계약을 작성한다.

```python
def selected_minmax(df, columns):
    """Contract
    input: pd.DataFrame, 존재하는 열 이름의 sequence
    output: 입력과 동일 index/열 순서/shape의 새 DataFrame
    behavior: 지정 열만 [0,1], 상수 열은 0.0, NaN은 유지
    mutation: 입력 변경 금지
    """
    out = df.copy()
    for col in columns:
        if col not in out.columns:
            raise KeyError(f"없는 열: {col}")
        lo, hi = out[col].min(), out[col].max()
        s = out[col]
        out[col] = s.where(s.isna(), 0.0) if hi == lo else (s - lo) / (hi - lo)
    return out
```

실제 문제의 상수 열 처리·NaN 처리 지시가 다르면 그 지시를 따른다. “상식적으로 좋은 처리”보다 명세가 우선이다.

### 2.8 실습문제

1. x의 shape가 [32,50,8]이고 축 의미가 [배치,시간,특성]이다. 특성 8개를 채널로 쓰는 Conv1d(in_channels=8, ...)에 넣기 직전 필요한 PyTorch 코드를 쓰라.
2. 예측 `[128,1]`, 정답 `[128]`의 MSE에서 생길 수 있는 문제와 수정법을 쓰라.
3. `a = np.arange(12).reshape(3,4)`에서 2번째 열만 `[3,1]`로 선택하는 코드를 두 가지 쓰라.
4. 위 `selected_minmax`가 빈 `columns=[]`를 받으면 무엇을 반환해야 하는가?
5. `reshape`와 `permute`의 차이를 한 문장으로 설명하라.

#### 정답·해설

1. `x = x.permute(0, 2, 1)`로 `[B,T,F] → [B,F,T]`를 만든다.
2. broadcasting으로 `[128,128]` 비교가 될 수 있다. `pred.squeeze(1)` 또는 `target.unsqueeze(1)`로 정확히 같은 shape를 만든다.
3. `a[:, 1:2]` 또는 `a[:, [1]]`다. `a[:,1]`은 `[3]`이 된다.
4. 원본과 값이 같은 **복사본**을 반환하는 것이 계약에 자연스럽다. 입력 객체 자체를 반환하면 이후 mutation에 취약하다.
5. reshape는 원소를 유지하며 차원 크기를 다시 묶고, permute는 축의 순서 자체를 바꾼다.

### 2.9 완료 기준

- [ ] 모든 모델 코드에 입력·출력 shape 주석을 붙인다.
- [ ] view와 copy 차이를 작은 배열로 재현했다.
- [ ] `[B]`와 `[B,1]` loss broadcasting을 직접 확인했다.

---

## 3. NumPy·pandas로 데이터 계약 지키기

### 3.1 학습 목표

- train·test·feature·target이 무엇인지 작은 표로 설명한다.
- 빈칸·무한대·중복·자료형을 확인하고 처리 이유를 말한다.
- merge와 groupby의 결과 행 수를 예상하고 정렬 후 예측 순서를 복원한다.
- 간단한 검사 함수를 실행하고 큰 배열의 메모리를 만들기 전에 계산한다.

### 3.2 2분 EDA의 순서

#### 먼저 상황부터: 자동차의 연비를 예측하고 싶다

이번 강의에서 **데이터 계약은 “모델에 줄 표가 지켜야 하는 약속”**이다. 어려운 법률 용어가 아니다. 예를 들어 “한 행은 자동차 한 대, 입력 열은 무게·속도 순서, 정답은 연비, 제출도 처음 받은 자동차 순서”라는 약속이다. 코드가 실행되어도 무게와 속도를 뒤집거나 다른 자동차의 예측을 제출하면 틀린 결과다.

처음부터 2분 안에 할 필요는 없다. 먼저 아래 예제를 천천히 읽고 직접 실행한 뒤, 익숙해졌을 때 점검 속도를 줄인다. 이번 강의는 모델을 학습하기 **전**에 표를 점검하는 단계다. pandas 자체가 낯설면 [입문 7강의 DataFrame 읽기](https://markshincuhk.github.io/hdat-ds-study-hub/start/07/)와 [입문 8강의 데이터 정리](https://markshincuhk.github.io/hdat-ds-study-hub/start/08/)를 함께 본다.

| 용어 | 쉬운 뜻 | 이번 예제 |
|---|---|---|
| DataFrame | 이름 붙은 행과 열이 있는 표 | `train`, `test` |
| Series | 표에서 꺼낸 한 열 | `train["speed"]` |
| train | 정답을 보고 규칙을 배우는 학습 데이터 | 연비 열이 있다 |
| test | 학습한 규칙으로 정답을 예측할 데이터 | 연비 열이 없다 |
| feature, X | 예측에 사용할 입력 정보 | 무게·속도 |
| target, y | 예측하려는 정답 | 연비 |
| ID | 어떤 자동차인지 식별하는 번호 | `vehicle_id`; 이 예제에서는 입력에서 제외 |
| dtype | 값의 자료형 | 숫자·문자·날짜 등 |
| schema | 열 이름·순서·자료형 같은 표의 구조 | 무게 다음 속도 |
| EDA | 학습 전에 크기·분포·이상한 값을 살펴보는 일 | 빈칸이 몇 개인지 확인 |

아래 코드 하나만 새 셀에서 실행해도 된다. 외부 CSV 파일은 필요 없다.

```python
import numpy as np
import pandas as pd

train = pd.DataFrame({
    "vehicle_id": ["A", "B", "C"],
    "weight": [1000., 1200., 1100.],
    "speed": [40., 60., 50.],
    "fuel_efficiency": [16., 12., 14.],
})
test = pd.DataFrame({
    "vehicle_id": ["D", "E"],
    "speed": [45., 55.],  # 일부러 train과 열 순서를 다르게 둠
    "weight": [1050., 1150.],
})
TARGET = "fuel_efficiency"  # 변수 안에는 열 이름 문자열을 저장
FEATURES = ["weight", "speed"]

print(train.to_string(index=False))
print("train/test:", train.shape, test.shape)
print("정답:", train[TARGET].tolist())
print("열별 빈칸:", train.isna().sum().to_dict())
print("완전히 같은 중복 행:", int(train.duplicated().sum()))

missing = [column for column in FEATURES if column not in test.columns]
if missing:
    raise ValueError(f"test에 필요한 열이 없습니다: {missing}")
X_train = train[FEATURES]  # FEATURES에 적은 순서대로 두 열 선택
X_test = test[FEATURES]    # 같은 열 순서로 맞춤
y_train = train[TARGET]   # 정답 열 하나 선택
print("입력 열 순서:", X_train.columns.tolist(), X_test.columns.tolist())
print("X_train/X_test/y:", X_train.shape, X_test.shape, y_train.shape)
assert X_train.columns.tolist() == X_test.columns.tolist()
```

주요 결과는 train/test가 `(3,4)`와 `(2,3)`, 정답 목록이 `[16.0,12.0,14.0]`, X_train/X_test/y가 `(3,2)`, `(2,2)`, `(3,)`이다. test에 정답 열이 없어 열 수가 하나 적은 것은 정상이다. 두 X의 열 순서는 모두 `['weight','speed']`다.

코드를 한 줄씩 읽으면 다음과 같다.

- `pd.DataFrame({...})`: 딕셔너리의 key를 열 이름, 리스트를 그 열의 값으로 삼아 표를 만든다. 각 리스트 길이는 같아야 한다.
- `TARGET = "fuel_efficiency"`: 대문자는 설정값임을 눈에 띄게 쓰는 관례다. 대문자라서 특별한 기능이 생기지는 않는다.
- `train[TARGET]`: TARGET 변수 안의 문자열을 열 이름으로 사용한다. `train["TARGET"]`은 문자 그대로 TARGET이라는 이름의 열을 찾으므로 다르다.
- `train[FEATURES]`: 리스트에 적은 여러 열을 그 순서로 골라 2차원 표를 만든다. `train[TARGET]`는 한 열짜리 Series다.
- `isna()`: 빈칸인 위치를 True로 표시한다. 그 뒤 `sum()`은 True를 1로 세어 열마다 빈칸 개수를 구한다.
- `duplicated()`: 앞에 완전히 같은 행이 있었는지 표시한다. 같은 자동차 ID가 반복된다는 사실만으로 전체 행이 중복인 것은 아니다.
- `assert 조건`: 조건이 거짓이면 멈춰 잘못된 가정을 알려 준다. 데이터를 자동으로 고치는 명령이 아니다.

예를 들어 FEATURES가 `['weight','speed']`, test 열이 `['vehicle_id','speeed','weight']`라면 missing에는 `'speed'`가 남는다. **열 이름의 오타를 NaN으로 덮어 진행하지 말고 원인을 확인한다.** ID를 입력에서 제외한 것도 이 예제의 선택이지 모든 문제에서 ID 사용이 금지된다는 뜻은 아니다.

익숙해진 뒤의 점검 순서는 **크기 → 열 이름·순서 → 자료형 → 빈칸·무한대 → 중복 → 정답의 의미**다. `head()`는 앞의 몇 행, `dtypes`는 열별 자료형, `info()`는 자료형과 비어 있지 않은 개수, `describe()`는 요약 통계를 보여 준다. 요약 통계를 보고도 단위·대상·시간은 문제 설명에서 확인해야 한다.

여기서 확인하는 것은 단순히 “데이터가 깨끗한가?”가 아니라 **문제의 약속이 무엇인가?**다.

- 행 하나가 무엇인가: 사람, 차량, 시점, 이미지?
- 같은 개체가 여러 행인가?
- 시간 열이 있는가?
- ID가 예측에 써도 되는 정보인가?
- target이 연속값인가, class인가, 여러 열인가?
- test 순서를 제출에서 그대로 유지해야 하는가?

### 3.3 수치형 유한성

“유한한 숫자”는 12.5나 -3처럼 일반적으로 계산할 수 있는 값을 뜻한다. **NaN은 값이 없거나 계산 결과가 정의되지 않은 상태, Inf는 양의 무한대, -Inf는 음의 무한대**다. 빈칸을 채우는 `fillna()`만으로 Inf가 없어지지는 않는다.

```python
import numpy as np
import pandas as pd

# 학습용 입력 열 한 개. 정답 y가 아니다.
speed = pd.Series([10., np.nan, np.inf, 30.], name="speed")
arr = speed.to_numpy()  # 이름 붙은 pandas 열 -> 숫자 배열
print("빈칸 위치:", speed.isna().tolist())
print("무한대 위치:", np.isinf(arr).tolist())
print("유한한 위치:", np.isfinite(arr).tolist())

# 이 연습에서는 Inf를 센서 오류로 보고 결측으로 취급한다는 계약을 둔다.
clean = speed.replace([np.inf, -np.inf], np.nan)
train_median = clean.median()  # NaN을 제외한 10과 30의 중앙값 = 20
filled = clean.fillna(train_median)
print("중앙값:", train_median)
print("처리 후:", filled.tolist())
assert np.isfinite(filled.to_numpy()).all()

# 새 데이터에는 새 중앙값을 구하지 않고 train에서 구한 20을 재사용한다.
new_speed = pd.Series([np.nan, np.inf, 100.])
new_clean = new_speed.replace([np.inf, -np.inf], np.nan)
print("새 데이터:", new_clean.fillna(train_median).tolist())
```

빈칸 위치는 `[False,True,False,False]`, 무한대 위치는 `[False,False,True,False]`, 유한한 위치는 `[True,False,False,True]`다. 처리 후 값은 `[10.0,20.0,20.0,30.0]`, 새 데이터는 `[20.0,20.0,100.0]`이 된다. 새 데이터에도 같은 Inf 처리 규칙을 적용하되, 채울 중앙값만 train에서 구한 값을 재사용한다.

`replace([np.inf,-np.inf], np.nan)`는 양·음의 무한대만 빈칸으로 바꾼다. `median()`은 중앙값을 구하고, `fillna(train_median)`은 **비어 있는 위치만** 채운다. 이 순서와 판단을 생략하고 “에러를 없애기 위해 전부 0으로 바꾸기”를 하면 신호가 바뀔 수 있다.

주의할 상황도 있다. 열 전체가 비어 있으면 중앙값도 NaN이므로 위 방법으로 해결되지 않는다. 그 열을 제외할지 상수로 채울지 문제와 데이터에 맞게 정한다. **정답 y의 빈칸을 입력 feature처럼 중앙값으로 채워 정답을 만들어 내면 안 된다.** 확인·제외·별도 처리 등 명세에 맞는 정책이 필요하다.

실제 학습에서는 train/valid를 먼저 나눈 뒤 **학습 부분에서만** 중앙값을 구해 valid/test에 적용한다. 검증 데이터까지 이용해 미리 채우면 검증 과정에 정보가 새어 들어갈 수 있다. 또한 float32로 변환할 때 매우 큰 값이 Inf가 될 수 있으므로 변환·추론 후에도 다시 검사한다.

### 3.4 범주형 혼합 타입

범주형은 크기를 계산하는 숫자라기보다 종류를 나타내는 값이다. 차종 A·B·C, 생산 공장 이름 등이 예다. 인코더는 이런 종류를 모델이 사용할 숫자 표현으로 바꾸는 도구다.

한 열에 숫자 `1`, 문자열 `'1'`, 빈칸 `None`, pandas의 결측 표시 `pd.NA`가 섞이면 인코더가 예상과 다르게 동작할 수 있다. **먼저 숫자 1과 문자열 '1'이 같은 종류를 뜻하는지 판단해야 한다.** 아래는 둘을 같은 종류로 합친다는 연습 계약이다.

```python
import numpy as np
import pandas as pd

def normalize_category_missing(s):
    strings = s.astype("string")  # 값은 문자열로, 결측은 결측으로 유지
    obj = strings.astype(object)  # np.nan을 담는 일반 객체 열로 변환
    return obj.where(pd.notna(obj), np.nan)  # 비결측은 유지, 결측만 np.nan

s = pd.Series([1, "1", None, pd.NA], dtype=object)
out = normalize_category_missing(s)
print("처리 후:", out.tolist())
print("빈칸 개수:", int(out.isna().sum()))
assert out.iloc[0] == out.iloc[1] == "1"
assert out.isna().tolist() == [False, False, True, True]
```

결과는 `['1','1',nan,nan]`, 빈칸은 2개다. `where(조건, 대체값)`은 조건이 True인 원래 값을 남기고 False인 곳만 바꾼다. `pd.notna(obj)`는 빈칸이 아닌 곳에 True를 준다.

`astype(str)`의 Python `str`와 `astype("string")`의 pandas nullable string은 다르다. 전자는 결측을 `'None'`, `'nan'`, `'<NA>'` 같은 **문자**로 만들 수 있다. 겉으로 빈칸처럼 보여도 `isna()`에서 빈칸으로 세지 않을 수 있다는 뜻이다. 반대로 숫자 1과 문자 '1'이 원래 다른 종류라면 위 정규화를 그대로 쓰면 안 된다. 자료형 정리는 의미를 확인한 후에 한다.

### 3.5 merge와 groupby의 함정

#### merge: 공통 이름표로 다른 표의 정보를 붙인다

주행 기록에는 속도만 있고, 별도 차량 정보표에는 무게가 있다고 하자. 자동차 ID를 기준으로 찾아 붙이는 것이 `merge`다. 두 표를 연결할 때 쓰는 공통 열을 **key(키)**라고 부른다.

```python
import numpy as np
import pandas as pd

trips = pd.DataFrame({"vehicle_id": ["A", "B", "A"], "speed": [10, 20, 30]})
meta = pd.DataFrame({"vehicle_id": ["A", "B"], "weight": [1000, 1500]})
work = trips.copy()  # 원래 표를 보존
work["__row_order__"] = np.arange(len(work))  # 0,1,2라는 원래 행 번호
merged = work.merge(meta, on="vehicle_id", how="left", validate="many_to_one")
merged = merged.sort_values("__row_order__", kind="stable")
print(merged[["vehicle_id", "speed", "weight"]].to_string(index=False))
assert len(merged) == len(trips)

# 잘못된 정보표: A가 두 번 있어서 어느 무게를 붙일지 하나로 정해지지 않는다.
bad_meta = pd.concat([meta, meta.iloc[[0]]], ignore_index=True)
try:
    work.merge(bad_meta, on="vehicle_id", how="left", validate="many_to_one")
except pd.errors.MergeError:
    print("정보표의 key 중복을 발견했습니다")
else:
    raise AssertionError("중복 검사가 작동하지 않았습니다")
```

결과는 A·10·1000, B·20·1500, A·30·1000의 세 행이다. `how="left"`는 왼쪽 주행 기록을 기준으로 유지한다는 뜻이다. 오른쪽에 같은 key가 여러 개면 왼쪽 행이 여러 오른쪽 행과 짝지어져 **행 수가 늘 수 있다**. 따라서 left merge가 언제나 원래 행 수를 보장하는 것은 아니다.

`validate="many_to_one"`은 “왼쪽 기록에는 A가 여러 번 나와도 되지만, 오른쪽 정보표의 A는 한 번이어야 한다”는 검사다. `one_to_one`이면 양쪽 모두 key가 고유해야 한다. 일치하는 오른쪽 key가 없으면 새로 붙인 열이 NaN이므로 병합 후 그 빈칸도 확인한다. 이 예제의 `__row_order__`는 새 작업용 이름이므로 실제 표에 이미 같은 이름이 있다면 다른 이름을 선택한다.

#### groupby: 같은 이름표끼리 묶어서 계산한다

`agg()`는 그룹을 요약한 **작은 표**를 만들고, `transform()`은 각 원래 행에 그룹 계산 결과를 붙일 수 있게 **원래 행 수로 돌려준다**.

```python
import pandas as pd

trips = pd.DataFrame({"vehicle_id": ["A", "B", "A"], "speed": [10., 100., 30.]})
summary = trips.groupby("vehicle_id", sort=False)["speed"].agg("mean")
per_row_mean = trips.groupby("vehicle_id", sort=False)["speed"].transform("mean")
with_mean = trips.assign(vehicle_mean=per_row_mean)
print("그룹 요약:", summary.to_dict())
print("행별 평균:", per_row_mean.tolist())
print("행 수:", len(summary), len(with_mean))
assert per_row_mean.index.equals(trips.index)
```

그룹 요약은 `{'A':20.0,'B':100.0}`이고 행 수는 2다. 각 원래 행에 대응하는 평균은 `[20.0,100.0,20.0]`이고 행 수는 3이다. A의 평균은 `(10+30)/2=20`, B는 기록이 하나이므로 평균 100이다. `assign(vehicle_mean=...)`은 계산한 열을 붙인 새 표를 만든다.

요약 결과를 무턱대고 원래 표의 열에 대입하면 index가 A·B와 0·1·2처럼 달라 NaN이 들어갈 수 있다. 같은 행에 붙일 값이면 `transform`의 의미부터 확인한다. 그룹 key가 결측일 때 포함할지는 `dropna` 정책을 정한다. **이 전체 그룹 평균 예제는 연산 연습용이며, 미래 예측에서 그대로 쓰면 미래 기록을 포함할 수 있다.** 시계열 feature는 예측 시점에 이미 알고 있는 과거만 이용한다.

### 3.6 시간 정렬

날짜가 문자로 들어 있으면 먼저 `pd.to_datetime()`으로 날짜 자료형으로 바꾼다. `errors="raise"`는 해석할 수 없는 날짜를 만났을 때 멈추게 한다. 조용히 빈 날짜로 바꿔 계속하는 것보다 이 예제의 오류를 찾기 쉽다.

다음 표는 1월 3일, 1일, 2일 순서로 주어졌다. 시간순 계산을 위해 정렬하더라도 제출은 처음 받은 행 순서여야 한다. **정렬한 표에 예측을 붙인 뒤, 표와 예측을 함께 되돌린다.**

```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    "vehicle_id": ["A", "A", "A"],
    "timestamp": ["2026-01-03", "2026-01-01", "2026-01-02"],
    "speed": [30., 10., 20.],
})
work = df.copy()
work["timestamp"] = pd.to_datetime(work["timestamp"], format="%Y-%m-%d", errors="raise")
work["__row_order__"] = np.arange(len(work))
work = work.sort_values(
    ["vehicle_id", "timestamp", "__row_order__"], kind="stable"
).reset_index(drop=True)

# 순서 연습을 위한 가짜 예측이다. 실제 예측 모델·공식 제출 예제가 아니다.
work["prediction"] = work["speed"] / 10
print("정렬 후 원래 행 번호:", work["__row_order__"].tolist())
print("정렬 상태 예측:", work["prediction"].tolist())
restored = work.sort_values("__row_order__", kind="stable")
pred_original = restored["prediction"].to_numpy()
print("제출할 원래 순서:", pred_original.tolist())
assert pred_original.tolist() == [3., 1., 2.]
```

정렬 후 원래 행 번호는 `[1,2,0]`, 정렬 상태의 예측은 `[1.0,2.0,3.0]`, 원래 순서로 복원한 예측은 `[3.0,1.0,2.0]`이다.

`np.arange(len(work))`는 0부터 행 수-1까지 원래 위치표를 만든다. `sort_values([...])`는 목록의 앞 열부터 우선순위를 적용한다. 같은 차량·시각이면 원래 행 번호가 마지막 순서를 정한다. `kind="stable"`은 동률인 값의 순서를 안정적으로 유지하도록 지정하는 옵션이다. `reset_index(drop=True)`는 화면의 index를 새로 붙일 뿐, 원래 순서로 복원하는 기능이 아니다. 원래 순서는 별도로 보관한 `__row_order__`가 기억한다.

예측이 `[N,K]`인 다중출력 배열이라면 원래 행 번호의 정렬 인덱스로 **예측 배열의 첫 축을 통째로** 재배열한다. 열마다 독립적으로 값을 정렬하면 다른 샘플의 값이 섞인다. 또한 시간 정렬은 시간순 train/valid 분할과 다른 작업이다. 정렬만 했다고 누수가 해결되는 것은 아니다.

### 3.7 메모리 감각

메모리는 실행 중 숫자를 올려 두는 작업 공간이다. 배열 메모리는 대략 `원소 수 × 숫자 하나의 byte 수`다. float64는 숫자 하나에 8 byte, float32는 4 byte다. 같은 원소 수라면 float32가 절반을 사용하지만 표현 범위·정밀도도 달라지므로 무조건 바꾸지는 않는다.

```python
# 실제로 큰 배열을 만들지 않고 필요한 크기만 계산한다.
rows, features = 500_000, 100
bytes_needed = rows * features * 8  # float64
print("byte:", bytes_needed)
print("MiB:", round(bytes_needed / 1024**2, 1))
```

결과는 `400000000` byte, 약 `381.5` MiB다. MiB는 1024×1024 byte이고, GB는 10억 byte라는 표기를 구분한다. DataFrame은 index·문자열 등 추가 메모리도 사용하므로 실제 사용량이 단순 숫자 계산보다 클 수 있다.

`1,000,000 × 20 × 23` float32 window를 모두 미리 만들면 약 `1.84GB`다. 복사본, gradient, 모델 activation까지 합치면 커널이 쉽게 다운된다. 큰 window는 lazy `Dataset`으로 인덱스만 저장한다.

### 3.8 미니 실습: 안전한 계약 검사기

지금까지 수동으로 확인한 일부 약속을 함수로 묶어 보자. **이 함수는 데이터를 고치거나 학습하지 않고, 기본 구조를 검사한 뒤 사용할 열 이름을 돌려준다.** 이 연습의 계약은 “정답은 train에만 있고, ID는 제외하며, test에는 필요한 입력 열이 전부 있다”이다. 실제 시험에서 target 자리표시자 열을 제공하는 등 계약이 다르면 검사도 바꿔야 한다.

```python
import pandas as pd

def assert_frame_contract(train, test, target, drop_cols=()):
    if not train.columns.is_unique or not test.columns.is_unique:
        raise ValueError("열 이름이 중복되어 있습니다")
    if target not in train.columns:
        raise KeyError(f"target 누락: {target}")
    if target in test.columns:
        raise ValueError("test에 target이 있습니다: 누수/파일 확인")
    if pd.isna(train[target].to_numpy()).any():
        raise ValueError("target에 결측치가 있습니다")

    unknown_drop = sorted(set(drop_cols) - set(train.columns))
    if unknown_drop:
        raise KeyError(f"DROP_COLS 오타 가능: {unknown_drop}")

    features = [c for c in train.columns if c != target and c not in drop_cols]
    if not features:
        raise ValueError("사용할 feature가 없습니다")
    missing = sorted(set(features) - set(test.columns))
    if missing:
        raise KeyError(f"test feature 누락: {missing}")
    return features

train = pd.DataFrame({"id": [1, 2], "speed": [40., 60.], "y": [16., 12.]})
test = pd.DataFrame({"id": [3], "speed": [50.]})
features = assert_frame_contract(train, test, target="y", drop_cols=("id",))
print(features)  # ['speed']
print(train[features].shape, test[features].shape)  # (2,1) (1,1)
```

`drop_cols=("id",)`는 제외할 열 이름 하나가 들어 있는 tuple이다. `("id")`는 문자열이므로 끝의 쉼표를 빠뜨리지 않는다. 함수 안의 리스트 컴프리헨션은 “train의 각 열 c를 보되, 정답도 제외 열도 아닌 것만 남긴다”는 뜻이다. `set(features)-set(test.columns)`는 test에서 찾을 수 없는 열 이름을 골라내고, `sorted()`는 오류 메시지를 일정한 순서로 보여 준다.

`raise KeyError(...)`는 필요한 이름을 못 찾았다고 멈추고, `raise ValueError(...)`는 값·구조의 약속이 어긋났다고 멈춘다. **검사 통과는 전처리 완료나 정확한 모델을 보장하지 않는다.** 단위·dtype·Inf·중복 행의 의미·범주 매핑·시간 누수·정답 범위는 추가로 검사해야 한다. test의 사용하지 않는 추가 열도 이 함수는 허용한다. 반환된 features로 실제 입력 열과 순서를 명시적으로 선택한다.

공식 참고: [pandas merge의 연결 방식과 validate](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html), [groupby transform의 원래 index를 유지하는 결과](https://pandas.pydata.org/docs/reference/api/pandas.core.groupby.SeriesGroupBy.transform.html).
### 3.9 실습문제

1. `test.reindex(columns=train_features)`가 test의 실제 feature 누락을 조용히 숨길 수 있는 이유는?
2. test를 시간순으로 정렬해 예측한 뒤 원래 순서를 복원하는 최소 코드를 작성하라.
3. float64 `(500_000, 100)` 배열 하나의 대략적인 메모리를 MiB로 계산하라.
4. `merge(..., validate="one_to_one")`가 실패했다. 가능한 데이터 문제 두 가지는?
5. 한 범주 열에서 숫자 1과 문자열 '1'을 같은 범주로 합쳐도 된다고 가정한다. pd.NA를 범주형 imputer가 받을 np.nan으로 바꾸고 결측을 문자열로 만들지 않는 코드를 쓰라.

#### 쉬운 확인문제부터 다시 풀기

6. train에는 `id, speed, y`, test에는 `speed, id`가 있다. ID를 제외하면 X_train, X_test의 열은 무엇이고 y는 어디서 꺼내는가? test의 전체 열 순서가 달라도 괜찮은 조건은?
7. 입력 `[10,NaN,Inf,30]`에 `fillna(0)`만 쓰면 무슨 문제가 남는가? Inf를 결측으로 취급하고 train 중앙값을 사용하면 결과는?
8. 차량 기록의 ID가 `[A,B,A]`, 오른쪽 정보표의 ID가 `[A,A,B]`다. 검사 없이 left merge하면 몇 행이 되는가? 어떻게 조기에 막는가?
9. A의 속도는 10·30, B의 속도는 100이다. `agg('mean')`와 `transform('mean')` 결과의 행 수와 값은 어떻게 다른가?
10. 시간 정렬 후 원래 행 번호가 `[2,0,1]`, 그 순서의 예측이 `[9,7,8]`이다. 원래 순서의 예측을 쓰고 코드로 확인하라.

#### 정답·해설

1. 없는 열을 새로 만들고 전부 NaN으로 채우기 때문에, 오타나 잘못된 test schema가 오류 없이 진행된다. 먼저 집합 차이를 검사해야 한다.
2. 정렬 전 `__row_order__`를 만들고 정렬·예측 후 그 번호로 stable sort한다. 1차원·다중출력 모두 되는 형태는 `order = sorted_df['__row_order__'].to_numpy(); pred_original = np.asarray(pred)[np.argsort(order, kind='stable')]`이다. `pd.Series(pred, ...)`는 `(N,D)` 예측에서 실패하므로 쓰지 않는다.
3. `500000×100×8 = 400,000,000 byte`, 약 `381.5 MiB`다.
4. 왼쪽 또는 오른쪽 key에 중복이 있거나, 둘 다 중복일 수 있다. key의 의미가 실제로 일대일이 아닌 설계 문제일 수도 있다.
5. `obj=s.astype('string').astype(object); out=obj.where(pd.notna(obj), np.nan)`. 단, 숫자 1과 문자열 '1'을 같은 범주로 취급해도 되는지 먼저 확인한다.
6. 입력 열은 둘 다 `['speed']`, 정답은 `train['y']`다. 모델에 넣기 전에 같은 FEATURES 목록으로 양쪽 열을 같은 순서로 골라야 한다. test에 필요한 열이 있는지 먼저 검사한다.
7. Inf가 그대로 남는다. Inf를 NaN으로 바꾼 뒤 유효한 10과 30에서 구한 중앙값 20으로 채우면 `[10,20,20,30]`이다. 정답 y를 이런 식으로 만들어 채워서는 안 된다.
8. A 기록 두 행이 각각 오른쪽 A 두 행과 연결되어 4행, B가 1행이라 총 5행이다. `validate='many_to_one'`으로 오른쪽 key 중복을 검사하면 멈출 수 있다. 중복을 임의로 삭제하기 전에 정보표의 의미를 확인한다.
9. agg는 A→20, B→100의 2행 요약이다. transform은 원래 `[A,B,A]` 각 행에 대응하는 `[20,100,20]`의 3개 값이다. 그룹 요약을 원래 표에 그냥 대입하면 index가 맞지 않을 수 있다.
10. 원래 순서는 `[7,8,9]`다. 원래 행 번호 0이 가운데, 1이 마지막, 2가 처음에 있었기 때문이다. 아래는 한 출력·다중출력 모두 첫 축을 되돌리는 예다.

```python
import numpy as np

order = np.array([2, 0, 1])
pred_sorted = np.array([9, 7, 8])
restore_index = np.argsort(order, kind="stable")
pred_original = pred_sorted[restore_index]
assert pred_original.tolist() == [7, 8, 9]

multi_sorted = np.array([[9, 90], [7, 70], [8, 80]])
multi_original = multi_sorted[restore_index]
assert multi_original.tolist() == [[7, 70], [8, 80], [9, 90]]
```

틀렸다면 단순히 답만 외우지 말고, **3.2에서 입력·정답을 분리 → 3.3에서 빈칸과 Inf 구분 → 3.5에서 병합 전후 행 수 비교 → 3.6에서 예측 순서 복원** 중 어느 단계의 약속을 놓쳤는지 기록한다.

### 3.10 완료 기준

- [ ] train·test·feature·target·ID를 내 말로 설명한다.
- [ ] 작은 DataFrame에서 열 이름·자료형·결측·Inf·중복을 이유와 함께 검사한다. 2분 점검은 익숙해진 뒤 목표로 한다.
- [ ] merge의 key 중복과 agg/transform의 행 수 차이를 작은 표로 설명한다.
- [ ] 정렬 후 원래 test 순서를 복원할 수 있다.
- [ ] window 메모리를 생성 전에 계산한다.
- [ ] 쉬운 확인문제 6–10번을 해설 없이 풀고 원래 실습 1–5번에도 다시 도전했다.

---

## 4. 선형대수: 모델의 shape를 계산하는 언어

### 4.1 학습 목표

- vector, matrix, tensor의 차원을 설명한다.
- 내적, 행렬곱, 전치, norm을 계산한다.
- 선형층과 convolution을 행렬 연산 관점에서 이해한다.
- 고유값·고유벡터와 PCA의 관계를 설명한다.

### 4.2 vector와 matrix

벡터 `x ∈ R^F`는 한 샘플의 F개 특성, 행렬 `X ∈ R^(B×F)`는 B개 샘플을 행으로 쌓은 것이다. 선형층은

`Z = X W^T + b`

로 쓸 수 있다. `X:[B,F]`, `W:[O,F]`, `b:[O]`라면 `Z:[B,O]`다. PyTorch `nn.Linear(F,O)`가 weight를 `[O,F]`로 저장하기 때문에 수식에서 전치를 쓴다.

### 4.3 내적과 유사도

`x·w = Σ_i x_i w_i`는 같은 위치 원소를 곱해 더한다. 기하학적으로 `x·w = ||x|| ||w|| cosθ`다. cosine similarity는 크기를 제거하고 방향만 비교한다.

- 내적이 양수: 같은 방향 성분
- 0: 직교
- 음수: 반대 방향 성분

KNN의 Euclidean distance, SVM의 hyperplane, attention의 query-key score 모두 내적·거리에서 출발한다.

### 4.4 norm과 정규화

- L1 norm: `||x||₁ = Σ|x_i|`
- L2 norm: `||x||₂ = sqrt(Σx_i²)`
- squared L2: 미분이 편해 손실·규제에 자주 사용

L1 규제는 0에서 뾰족한 형태라 일부 weight를 정확히 0으로 만들기 쉽고 feature selection 효과가 있다. L2 규제는 큰 weight를 부드럽게 줄인다.

### 4.5 rank와 선형 종속

rank는 독립적인 정보 방향의 수다. 서로 완전히 중복된 특성은 rank를 늘리지 않는다. 다중공선성이 크면 선형 회귀 계수가 불안정할 수 있다. PCA는 분산이 큰 직교 방향으로 좌표계를 바꾸어 저차원 표현을 만든다.

### 4.6 고유값·고유벡터와 PCA

정방행렬 `A`에 대해 `Av = λv`를 만족하는 `v`가 고유벡터, `λ`가 고유값이다. PCA에서는 중심화한 데이터의 covariance matrix 고유벡터가 주성분 방향이고, 고유값은 그 방향의 분산이다.

PCA 순서:

1. 수치 특성의 scale을 맞춘다.
2. train 데이터 평균으로 중심화한다.
3. covariance 또는 SVD로 주성분을 구한다.
4. 큰 고유값 방향부터 선택한다.
5. validation/test에는 train에서 학습한 scaler와 PCA만 적용한다.

[함정] PCA는 target을 쓰지 않는 비지도 feature extraction이다. LDA는 class 분리를 최대화하는 지도 차원축소다.

### 4.7 손계산 예제

`x=[1,2]`, `w=[3,-1]`, `b=0.5`라면 `x·w+b = 1×3 + 2×(-1) + 0.5 = 1.5`다.

배치 `X=[[1,2],[0,-1]]`, 같은 `w,b`의 출력은 `[1.5, 1.5]`다.

#### PCA 2차원 완전 손계산

세 점 `(-1,-1), (0,0), (1,1)`을 생각하자. 두 열 평균은 이미 0이므로 중심화 결과가 같다. 모집단 covariance를 쓰면

`Σ = XᵀX/3 = [[2/3,2/3],[2/3,2/3]]`.

고유값은 `4/3, 0`, 대응하는 단위 고유벡터는 각각

`v₁=(1/sqrt(2))[1,1]`, `v₂=(1/sqrt(2))[1,-1]`.

첫 주성분 투영값 `Xv₁`은 `[-sqrt(2), 0, sqrt(2)]`이고, 설명분산비는 `(4/3)/(4/3+0)=1`, 즉 100%다. 두 feature가 완전히 같은 방향의 정보라 1차원으로 손실 없이 줄일 수 있다. sample covariance로 `n-1`을 나누면 고유값의 절대 크기는 `2,0`으로 바뀌지만 고유벡터와 설명분산비는 같다.

### 4.8 PyTorch로 검산

```python
import torch

X = torch.tensor([[1., 2.], [0., -1.]])       # [B=2,F=2]
W = torch.tensor([[3., -1.]])                  # [O=1,F=2]
b = torch.tensor([0.5])                        # [O=1]
Z = X @ W.T + b                                # [B=2,O=1]
assert torch.allclose(Z[:, 0], torch.tensor([1.5, 1.5]))
```

### 4.9 실습문제

1. `X:[64,20]`, `Linear(20,7)`의 weight, bias, output shape를 쓰라.
2. `x=[3,4]`의 L1, L2 norm을 구하라.
3. 입력에 동일한 두 수치 feature만 있고 적어도 두 개의 서로 다른 관측값이 있다. 중심화한 covariance matrix의 rank와 양의 분산을 가진 PCA 방향 수는 얼마인가? 두 feature가 모두 상수이면 어떻게 달라지는가?
4. PCA를 split 전에 전체 데이터에 fit하면 왜 leakage인가?
5. 두 영벡터가 아닌 벡터의 cosine similarity가 1, 0, -1일 때 의미를 쓰라. 한쪽이 영벡터이면 같은 각도 해석을 적용할 수 있는가?

#### 정답·해설

1. weight `[7,20]`, bias `[7]`, output `[64,7]`.
2. L1은 7, L2는 5.
3. 두 열만 있고 동일한 비상수 값이면 중심화 후 독립 방향이 하나여서 covariance rank는 1, 양의 분산을 가진 PCA 방향도 1개다. 다른 한 고유값은 0이다. 두 열이 모두 상수이면 중심화 후 전부 0이므로 rank와 양의 분산 방향 수는 모두 0이다.
4. validation/test 분포의 평균·분산·주성분 방향이 train feature 생성에 들어가 검증 정보가 새어 들어간다.
5. 두 벡터 모두 영벡터가 아닐 때 각각 같은 방향, 직교, 반대 방향이다. 한쪽이 영벡터이면 길이 곱이 0이라 수학적 코사인 유사도와 각도가 정의되지 않는다.

### 4.10 완료 기준

- [ ] `nn.Linear`의 모든 parameter shape를 계산한다.
- [ ] L1/L2와 cosine을 손으로 계산한다.
- [ ] PCA와 LDA의 지도성 차이를 설명한다.

---

## 5. 미분·역전파·최적화의 원리

### 5.1 학습 목표

- 도함수와 gradient가 loss 감소 방향을 알려주는 이유를 설명한다.
- chain rule로 2층 계산 그래프의 gradient를 구한다.
- batch/SGD/mini-batch 차이를 설명한다.
- learning rate가 너무 크거나 작을 때 학습곡선을 진단한다.

### 5.2 미분은 민감도다

`dy/dx`는 x가 아주 조금 변할 때 y가 얼마나, 어느 방향으로 변하는지 나타낸다. 다변수 함수 `L(w₁,…,w_d)`의 gradient `∇L`은 각 변수의 편미분을 모은 벡터이며 가장 가파른 증가 방향이다. 따라서 gradient descent는

`w ← w - η ∇L(w)`

로 업데이트한다. `η`는 learning rate다.

### 5.3 자주 쓰는 도함수

| 함수 | 도함수 |
|---|---|
| `x²` | `2x` |
| `exp(x)` | `exp(x)` |
| `log(x)` | `1/x` |
| sigmoid `σ(x)` | `σ(x)(1-σ(x))` |
| tanh | `1-tanh²(x)` |
| ReLU | x>0에서 1, x<0에서 0 |

ReLU는 0에서 미분이 정의되지 않지만 구현은 한쪽 값을 정해 사용한다. sigmoid와 tanh는 절댓값이 큰 영역에서 gradient가 거의 0이 되어 vanishing gradient가 생길 수 있다.

### 5.4 chain rule과 역전파

`z = wx+b`, `ŷ = z`이고 `L=(ŷ-y)²`라면

- `∂L/∂ŷ = 2(ŷ-y)`
- `∂ŷ/∂w = x`
- `∂L/∂w = 2(ŷ-y)x`
- `∂L/∂b = 2(ŷ-y)`

역전파는 출력에서 시작해 chain rule로 각 parameter의 gradient를 재사용하며 계산하는 알고리즘이다. optimizer는 이 gradient로 parameter를 갱신한다.

#### 2층 계산 그래프 완전 손계산

scalar 입력 `x=2`, target `y=1`이고

`a=w₁x+b₁`, `h=ReLU(a)`, `ŷ=w₂h+b₂`, `L=0.5(ŷ-y)²`

라 하자. `w₁=1,b₁=0,w₂=3,b₂=0`이면 forward는 `a=2,h=2,ŷ=6,L=12.5`다.

Backward:

- `dL/dŷ = ŷ-y = 5`
- `dL/dw₂ = (dL/dŷ)h = 10`, `dL/db₂=5`
- `dL/dh = (dL/dŷ)w₂ = 15`
- `a>0`이므로 `dReLU/da=1`, 따라서 `dL/da=15`
- `dL/dw₁ = (dL/da)x = 30`, `dL/db₁=15`

이 예제를 값만 바꾸어 손으로 다시 풀 수 있어야 한다.

### 5.5 PyTorch autograd

```python
import torch

w = torch.tensor(2.0, requires_grad=True)
b = torch.tensor(0.5, requires_grad=True)
x = torch.tensor(3.0)
y = torch.tensor(8.0)

pred = w * x + b                 # 6.5
loss = (pred - y) ** 2           # 2.25
loss.backward()
print(w.grad.item(), b.grad.item())  # -9.0, -3.0
```

손계산: `2(6.5-8)×3=-9`, bias는 `2(6.5-8)=-3`이다.

학습 루프에서 매 step `optimizer.zero_grad()`가 필요한 이유는 PyTorch가 gradient를 기본적으로 **누적**하기 때문이다.

### 5.6 batch 방식 비교

| 방식 | 한 update에 쓰는 데이터 | 장점 | 단점 |
|---|---:|---|---|
| Batch GD | 전체 | 안정적 gradient | 느리고 메모리 큼 |
| SGD | 1개 | 빠른 update, noise | 매우 불안정·비효율 가능 |
| Mini-batch | 작은 묶음 | GPU 효율과 안정성 절충 | batch size도 hyperparameter |

시험에서는 “SGD”라는 말이 optimizer 종류와 한 샘플 학습 방식을 넓게 가리킬 수 있으므로 문맥을 본다.

### 5.7 optimizer 직관

- Momentum: 과거 gradient의 이동 방향을 누적해 진동을 줄이고 골짜기를 빠르게 이동한다.
- RMSprop: parameter별 최근 squared gradient 크기로 step을 조절한다.
- Adam: momentum의 1차 모멘트와 RMSprop식 2차 모멘트를 함께 사용한다.
- AdamW: weight decay를 gradient 기반 L2 항과 분리해 적용한다.

Adam이 언제나 최종 일반화가 가장 좋은 것은 아니다. 그러나 제한 시간 baseline에는 안정적인 기본값이다.

### 5.8 학습곡선 진단

| 관찰 | 가능 원인 | 먼저 할 조치 |
|---|---|---|
| loss가 NaN | 큰 LR, Inf 입력, log/0, exploding gradient | 입력 유한성, LR↓, gradient clip |
| loss가 심하게 출렁임 | LR 큼, batch 작음 | LR↓ 또는 batch↑ |
| train/valid 모두 높음 | underfitting | capacity/epoch↑, feature 확인 |
| train만 낮고 valid 높음 | overfitting/leakage split 문제 | 규제, 데이터, split 재검토 |
| 초반부터 변화 없음 | gradient 없음, LR 작음, 잘못된 detach | `.grad`, loss-output 연결 확인 |

### 5.9 실습문제

1. 예측 ŷ=wx+b, 손실 L=(ŷ−y)²인 한 샘플에서 x=2, y=5, w=1, b=0이다. dL/dw와 dL/db를 구하라.
2. `zero_grad()`를 생략하고 두 batch를 학습하면 어떤 일이 생기는가?
3. sigmoid가 깊은 신경망 hidden activation에 불리한 이유 두 가지는?
4. learning rate가 너무 클 때와 너무 작을 때의 loss 곡선을 설명하라.
5. AdamW와 Adam+L2 penalty의 핵심 차이는?

#### 정답·해설

1. pred=2, error=-3이므로 `dL/dw=2×(-3)×2=-12`, `dL/db=-6`.
2. 이전 batch gradient에 현재 gradient가 더해져 의도한 mini-batch update가 아니게 된다. gradient accumulation을 의도한 경우만 예외다.
3. 큰 |x|에서 미분이 0에 가까워 vanishing gradient가 생기며, 출력이 0 중심이 아니어서 최적화가 불리할 수 있다.
4. 너무 크면 진동·발산·NaN, 너무 작으면 완만하게 내려가거나 제한 epoch 안에 거의 학습하지 못한다.
5. AdamW는 weight decay를 adaptive gradient update와 분리해 parameter에 직접 적용한다.

### 5.10 완료 기준

- [ ] 1개 선형 뉴런의 gradient를 손으로 계산한다.
- [ ] `zero_grad → forward → loss → backward → step` 순서를 외웠다.
- [ ] 학습곡선 5가지 증상에 첫 조치를 말할 수 있다.

---

## 6. 확률·통계·정보이론

### 6.1 학습 목표

- 평균, 분산, 공분산, 상관을 구분한다.
- 조건부확률과 Bayes rule을 분류 문제에 연결한다.
- likelihood, cross-entropy, KL divergence의 관계를 설명한다.
- 표본·검증 점수의 불확실성을 인식한다.

### 6.2 기술통계

평균 `μ = (1/n)Σxᵢ`, 분산 `Var(X)=E[(X-μ)²]`, 표준편차는 분산의 제곱근이다. 분산은 원 단위의 제곱, 표준편차는 원 단위와 같다. 이상치가 큰 데이터에서는 평균보다 median, 표준편차보다 IQR이 robust하다.

공분산 `Cov(X,Y)=E[(X-μx)(Y-μy)]`는 함께 움직이는 방향과 크기를 담는다. 상관계수는 공분산을 각 표준편차로 나눠 `[-1,1]`로 표준화한다.

[함정] 상관 0은 일반적으로 독립을 뜻하지 않는다. 비선형 관계가 있을 수 있다. 상관이 높아도 인과관계를 증명하지 않는다.

### 6.3 조건부확률과 Bayes rule

`P(A|B)=P(A∩B)/P(B)`이고,

`P(A|B)=P(B|A)P(A)/P(B)`.

희귀 고장 탐지에서 classifier의 sensitivity가 높아도 base rate가 매우 낮으면 양성 예측 중 실제 고장의 비율(precision)이 낮을 수 있다. 그래서 불균형에서는 accuracy 하나만 보면 안 된다.

### 6.4 likelihood와 손실함수

모델 parameter θ가 관측 데이터 y를 만들어낼 확률 `p(y|x,θ)`를 likelihood라 한다. 최대우도추정(MLE)은 likelihood를 최대화하고, 계산 편의를 위해 negative log-likelihood를 최소화한다.

- Gaussian noise + 평균 예측 → MSE 최소화와 연결
- Bernoulli label → binary cross-entropy와 연결
- Categorical label → multiclass cross-entropy와 연결

독립 표본의 확률 곱은 log를 취하면 합이 된다. 작은 확률을 직접 곱할 때 생기는 underflow도 줄어든다.

### 6.5 entropy와 cross-entropy

Entropy `H(p)=-Σp log p`는 분포의 불확실성이다. 이진 class가 50:50이면 최대, 한 class가 100%면 0이다.

Cross-entropy `H(p,q)=-Σp log q`는 실제 분포 p를 예측 분포 q로 표현할 때의 비용이다. `H(p,q)=H(p)+KL(p||q)`이므로 p가 고정이면 cross-entropy 최소화는 KL divergence 최소화와 같다.

### 6.6 KL divergence

`KL(p||q)=Σ p log(p/q)`는 일반적으로 대칭이 아니며 거리 metric이 아니다. `KL(p||q) ≠ KL(q||p)`일 수 있다. VAE에서는 approximate posterior를 prior와 가깝게 하는 regularization 항으로 등장한다.

### 6.7 bias와 variance

- 높은 bias: 모델이 단순해 train도 못 맞추는 underfitting
- 높은 variance: train 변화에 민감하고 validation 일반화가 나쁜 overfitting

데이터 증가, bagging, 규제는 variance를 낮추는 데 도움을 준다. 지나친 규제는 bias를 높인다.

### 6.8 표본 분할의 불확실성

작은 validation 한 번의 점수는 우연에 민감하다. random seed를 고정하는 이유는 “진실한 점수”를 얻기 위해서가 아니라 실험 비교의 변동 요인을 줄이기 위해서다. 가능한 경우 교차검증 평균과 표준편차를 본다. 시간·group 문제에서는 독립성 가정을 지키는 split이 fold 수보다 중요하다.

### 6.9 실습문제

1. `[1,2,3]`의 모집단 평균과 분산을 구하라.
2. binary class 비율이 1:1일 때와 99:1일 때 어느 쪽 entropy가 큰가?
3. KL divergence가 거리 metric이 아닌 이유 두 가지는?
4. 고장률 1%, recall 90%, false positive rate 10%인 검사의 precision을 10,000대 기준으로 계산하라.
5. MSE가 Gaussian likelihood와 연결되는 직관을 설명하라.

#### 정답·해설

1. 평균 2, 모집단 분산 `(1+0+1)/3=2/3`.
2. 1:1이 더 크며 이진 entropy 최대다.
3. 대칭이 아니고 triangle inequality를 만족하는 일반 거리로 사용할 수 없다.
4. 실제 고장 100대 중 90 true positive. 정상 9,900대 중 990 false positive. precision=`90/(90+990)=8.33%`.
5. 실제값이 모델 평균 주변의 일정한 Gaussian noise로 생성된다고 가정하면 negative log-likelihood에서 상수항을 제외한 부분이 squared error가 된다.

### 6.10 완료 기준

- [ ] entropy가 최대·최소인 class 분포를 말한다.
- [ ] cross-entropy와 KL 관계를 설명한다.
- [ ] 희귀 사건에서 accuracy가 왜 위험한지 숫자로 설명한다.

---

## 7. EDA: 모델보다 먼저 데이터 생성 과정을 읽기

### 7.1 학습 목표

- 행·열·target의 의미를 질문으로 정리한다.
- 단변량·이변량·그룹·시간 EDA를 구분한다.
- EDA 중에도 validation 정보가 학습 feature로 새지 않게 한다.
- 그래프를 “예쁘게” 그리는 대신 모델 의사결정으로 연결한다.

### 7.2 EDA의 목적

EDA는 모든 그래프를 그리는 절차가 아니다. 다음 의사결정을 빠르게 내리는 과정이다.

1. 문제 유형: 회귀, 이진분류, 다중분류, multilabel, 다중출력 중 무엇인가?
2. split 단위: 행, 사람/차량 group, 시간 중 무엇인가?
3. 전처리: 수치·범주·결측·이상치·고유 범주를 어떻게 처리할 것인가?
4. baseline: 선형, 트리, MLP, CNN1D 중 무엇을 먼저 돌릴 것인가?
5. metric: 예측값이 label인가, 확률인가, 연속값인가?

### 7.3 데이터 사전 만들기

열마다 아래 표를 최소한으로 작성한다.

| 열 | 의미/단위 | dtype | 결측률 | 고유값 수 | 누수 가능성 | 처리 |
|---|---|---|---:|---:|---|---|
| `vehicle_id` | 차량 식별자 | category | 0% | 800 | group split 필요 | feature 사용은 검증 |
| `timestamp` | 측정 시각 | datetime | 0% | 큼 | 미래 정보 위험 | 정렬·파생 후 원본 제외 |
| `temp` | 온도 °C | float | 3% | 연속 | 낮음 | median+scale |
| `repair_result` | 수리 후 판정 | string | 0% | 3 | 매우 큼 | 제거 |

누수는 target 열을 실수로 포함하는 경우만이 아니다. 예측 시점 이후에 생성되는 열, target을 집계한 열, 전체 기간 통계, 동일 개체의 미래 행도 누수다.

### 7.4 단변량 EDA

수치형에서는 위치·산포·왜도·극단값을 본다.

```python
summary = train[num_cols].describe(
    percentiles=[0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
).T
summary["missing_rate"] = train[num_cols].isna().mean()
summary["nunique"] = train[num_cols].nunique(dropna=False)
print(summary.sort_values("missing_rate", ascending=False).head(20))
```

범주형에서는 빈도와 train/test category 차이를 본다.

```python
for c in cat_cols:
    print(c, train[c].value_counts(dropna=False).head(10))
    unseen = set(test[c].dropna().unique()) - set(train[c].dropna().unique())
    print("test-only categories:", list(unseen)[:10])
```

[함정] `nunique`가 행 수에 가깝다고 무조건 버릴 ID는 아니다. 제품 코드나 시계열 key가 정보일 수 있다. 반대로 target과 일대일 대응하는 식별자는 암기 누수가 될 수 있다. group validation으로 판단한다.

### 7.5 target EDA

회귀:

- 범위, 0/음수 가능 여부
- long-tail과 log 변환 가능성
- 단위와 metric
- 여러 target 사이 상관

분류:

- class 수와 label 값
- class imbalance
- group/time별 class 변화
- train fold에 모든 class가 존재하는지

```python
print(train[TARGET].value_counts(dropna=False, normalize=True))
print(pd.crosstab(train["vehicle_id"], train[TARGET]).head())
```

### 7.6 이변량 EDA와 상관

상관은 선형 관계의 단서이지 feature 유용성의 최종 판정이 아니다. 트리나 신경망은 비선형·상호작용을 학습할 수 있다. target과 상관이 지나치게 1에 가까운 열은 오히려 누수를 의심한다.

범주형 대 target:

- 회귀: category별 count, mean, median, 분산
- 분류: category별 class 비율, 최소 support

희귀 category의 target mean은 표본이 작아 불안정하고 target encoding은 fold 밖 정보가 새기 쉽다.

### 7.7 group·시간 EDA

```python
group_stats = train.groupby("vehicle_id", observed=True).agg(
    n=(TARGET, "size"),
    target_mean=(TARGET, "mean"),
    start=("timestamp", "min"),
    end=("timestamp", "max"),
)
print(group_stats.describe())
```

확인할 것:

- group별 행 수 차이가 큰가?
- train/test group이 겹치는가?
- 시간에 따라 target 분포가 이동하는가?
- sensor sampling interval이 일정한가?
- 행이 이미 시간순인가?
- 결측이 특정 group이나 시간대에 집중되는가?

### 7.8 EDA와 split의 순서

schema와 데이터 생성 구조는 전체 train에서 확인해도 되지만, label을 이용한 feature 선택·imputation 통계·scaler·PCA·target encoding은 train fold에만 fit한다. validation 점수를 반복해서 보며 사람이 feature를 선택하는 것도 넓은 의미의 validation overfitting이다. 최종 후보 수를 제한하고 기록한다.

### 7.9 실습문제

다음 상황마다 첫 번째로 확인할 항목을 쓰라.

1. random split AUC는 0.99인데 group split은 0.61이다.
2. 특정 열과 target 상관이 0.999다.
3. test-only category가 30%다.
4. 시계열의 timestamp 차이가 대부분 0.01초지만 일부 5초다.
5. 회귀 target 99%는 0~10인데 최대값이 100,000이다.

#### 정답·해설

1. ID성 feature나 동일 group 행이 양 fold에 섞인 암기 누수를 확인한다. 실제 test가 새 group이면 group split이 더 정직하다.
2. 예측 시점에 이용 가능한 열인지, target에서 파생된 사후 정보인지 확인한다.
3. one-hot unknown 정책, category의 의미, high-cardinality 처리와 train/test distribution shift를 확인한다.
4. 주행 단절인지 결측 구간인지 확인하고 gap을 넘는 window를 만들지 않도록 sequence를 분리한다.
5. 단위 오류·입력 오류인지 실제 long-tail인지 확인한다. 무조건 제거하지 말고 metric과 도메인에 따라 clip/log/robust loss를 검증한다.

### 7.10 완료 기준

- [ ] 임의 데이터에서 10분 안에 데이터 사전을 만든다.
- [ ] 누수 후보 열과 split 단위를 먼저 표시한다.
- [ ] 모든 그래프 옆에 “그래서 무엇을 바꿀지” 한 줄을 쓴다.

---

## 8. 전처리: 결측·이상치·스케일·인코딩·증강

### 8.1 학습 목표

- 결측 메커니즘과 처리 전략을 구분한다.
- standardization과 min-max normalization을 계산한다.
- 범주형 인코딩의 unknown·high-cardinality 문제를 처리한다.
- 모든 학습형 전처리를 train fold에만 fit한다.

### 8.2 결측의 세 유형

- MCAR: 결측 여부가 관측·비관측 값과 무관
- MAR: 관측된 다른 변수와 관련
- MNAR: 결측인 바로 그 값 또는 비관측 원인과 관련

실전에서는 유형을 완벽히 증명하기 어렵다. 중요한 것은 결측 자체가 정보인지, 삭제가 편향을 만드는지, test에서도 같은 패턴인지다.

처리 후보:

| 상황 | 먼저 검토할 처리 |
|---|---|
| 수치 결측 적음 | train median imputation |
| 범주 결측 | 별도 `__MISSING__` 또는 최빈값 |
| 결측 자체가 상태 | missing indicator 추가 |
| 시계열 짧은 공백 | causal forward fill/보간 여부 검토 |
| 열 대부분 결측 | 의미·test 가용성 확인 후 제거 검토 |

[함정] 전체 데이터 평균으로 채우거나 시간축에서 미래 값으로 backward fill하면 누수다.

### 8.3 이상치

IQR은 `Q3−Q1`이다. 하한 `Q1−1.5×IQR`보다 작거나 상한 `Q3+1.5×IQR`보다 큰 값을 flag한다. 예를 들어 Q1=10, Q3=14이면 IQR=4, 경계는 4와 20이다. 경계값 자체는 이상치로 표시하지 않는다. 그러나 “통계적 극단값”과 “오류”는 다르다. 고장 예측에서 극단 sensor 값은 가장 중요한 신호일 수 있다.

선택지:

- 명백한 오류를 결측 처리
- 도메인 범위로 clip/winsorize
- log1p 변환
- RobustScaler
- Huber loss
- 이상 여부 indicator
- 정상 데이터 기반 anomaly model

행 삭제는 target 분포와 test 대표성을 바꿀 수 있으므로 신중히 한다.

### 8.4 스케일링 공식

Standardization:

`z = (x - μ_train) / σ_train`

Min-Max:

`x' = (x - min_train) / (max_train - min_train)`

Robust scaling은 median과 IQR을 사용한다. 거리 기반 KNN/SVM, PCA, 선형모델, 신경망은 scale 영향을 크게 받는다. 결정트리는 threshold 순서 비교라 보통 scale에 둔감하다.

[함정] test 값이 train 범위를 벗어나면 min-max 결과가 0보다 작거나 1보다 클 수 있다. 이것은 오류가 아니다. 명세가 요구하지 않으면 임의 clip하지 않는다.

### 8.5 상수 열 Min-Max

`max=min`이면 0으로 나눌 수 없다. 문제 지시가 없다면 유효값을 0으로 두고 기존 결측은 유지하는 정책이 흔하다.

```python
def minmax_selected(df, columns):
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            raise KeyError(c)
        lo, hi = out[c].min(skipna=True), out[c].max(skipna=True)
        if pd.isna(lo) or pd.isna(hi):
            continue                           # 전부 결측: 그대로
        s = out[c]
        out[c] = s.where(s.isna(), 0.0) if hi == lo else (s - lo) / (hi - lo)
    return out
```

### 8.6 범주형 인코딩

| 방식 | 장점 | 위험 |
|---|---|---|
| One-hot | 해석 쉬움, 순서 가정 없음 | 고유값 많으면 차원 폭증 |
| Ordinal/label | 차원 작음 | 존재하지 않는 순서를 암시 |
| Frequency | target 미사용 | 빈도가 의미를 왜곡 가능 |
| Target encoding | 강력할 수 있음 | 누수 매우 쉬움, out-of-fold 필요 |
| Embedding | 고유 범주 학습 | 충분한 데이터와 unknown 정책 필요 |

다중분류 target label은 feature ordinal encoding과 다르다. target label을 `0..K-1`로 바꾸고 prediction 후 원래 문자열 label로 복원한다.

### 8.7 데이터 증강

증강은 label을 보존해야 한다.

- 이미지: crop, flip, color jitter, 작은 회전. 좌우 의미가 바뀌는 분야는 flip 금지.
- 시계열: 작은 noise, scaling, crop, time shift. event 시점이나 target horizon을 깨지 않아야 함.
- 표형: 무작정 noise를 넣기보다 regularization과 검증이 안전한 기본.

random augmentation은 train에만 적용하고 validation/test는 결정적 transform을 사용한다.

### 8.8 train-only fit 패턴

```python
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
    ("scaler", StandardScaler()),
])
category_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("onehot", OneHotEncoder(handle_unknown="ignore")),
])
preprocess = ColumnTransformer([
    ("num", numeric_pipe, num_cols),
    ("cat", category_pipe, cat_cols),
])

Xtr = preprocess.fit_transform(X_train)  # train만 fit
Xva = preprocess.transform(X_valid)
Xte = preprocess.transform(X_test)
```

PyTorch MLP로 넘길 때 희소 one-hot이 너무 크면 dense 변환으로 OOM이 날 수 있다. 고유 범주를 줄이거나 ordinal/frequency/embedding 전략을 검토한다.

### 8.9 실습문제

1. train `[0,5,10]`, test `[15]`에 train min-max를 적용한 test 값은?
2. tree와 KNN 중 scaling이 더 중요한 모델과 이유는?
3. target encoding을 안전하게 train feature로 만드는 원리는?
4. 시계열 forward fill은 어떤 조건에서 누수가 될 수 있는가? split 전에 실행했다는 사실만으로 항상 누수라고 할 수 있는지도 설명하라.
5. test-only category를 -1로 ordinal encoding할 때 unknown 코드의 의미와 충돌 여부에서 무엇을 확인해야 하는가? 이어서 PyTorch Embedding에 넣는다면 어떤 추가 변환·범위 검사가 필요한가?

#### 정답·해설

1. `(15-0)/(10-0)=1.5`.
2. KNN. 거리 계산이 feature 단위에 직접 영향받는다.
3. 각 train 행의 encoding은 그 행의 target을 포함하지 않는 out-of-fold 통계로 만들고, validation/test는 train fold 통계를 쓴다.
4. 시간 정렬이 잘못되어 미래 값을 전달하거나, 다른 개체의 값을 넘기거나, 예측 원점에서 아직 관측할 수 없는 값을 사용하면 누수 또는 잘못된 특성 처리가 된다. 같은 개체의 실제로 이용 가능한 과거 값만 인과적으로 전달했다면 split 전에 실행했다는 사실만으로 항상 누수인 것은 아니다. 일괄 다중시점 예측과 순차적으로 새 관측을 받는 예측의 정보 계약도 구분한다.
5. ordinal unknown -1이 기존 범주 코드와 충돌하지 않는지, 수치형 모델이 이를 어떤 순서·거리로 해석하는지 확인한다. Embedding을 쓰는 경우에는 -1을 직접 넣지 말고 unknown=0, 기존 범주=1부터 같은 유효 주소로 매핑하며 정수 dtype과 0 ≤ index < num_embeddings를 검사한다. 모든 모델이 embedding 범위 조건을 요구하는 것은 아니다.

### 8.10 완료 기준

- [ ] 상수·전부 결측·unknown category hidden test를 만든다.
- [ ] scaler를 train fold에만 fit하는 코드를 빈 화면에서 쓴다.
- [ ] 증강이 label을 보존하는지 설명한다.

---

## 9. Feature Engineering·Selection·Extraction

### 9.1 학습 목표

- selection과 extraction을 구분한다.
- filter, wrapper, embedded selection을 비교한다.
- PCA와 LDA의 목적·fit 범위를 설명한다.
- group/time 문제에서 안전한 파생변수를 만든다.

### 9.2 Feature engineering

좋은 feature는 모델이 배워야 할 불변성이나 관계를 미리 표현한다.

표형:

- 비율 `a/(b+eps)`, 차이 `a-b`, 곱 `a×b`
- log1p, 제곱, bin
- 날짜의 hour/dayofweek/month
- category frequency

시계열:

- 과거 구간 mean/std/min/max/slope
- lag와 difference
- rolling statistic
- FFT band energy 등 주파수 특징

이미지:

- 모델이 CNN이면 수동 edge feature보다 올바른 resize/normalize/augmentation이 먼저

[함정] rolling/aggregate는 반드시 해당 예측 시점 이전 데이터만 사용한다. `center=True` rolling이나 전체 group 평균은 미래 정보를 포함할 수 있다.

### 9.3 Feature selection 세 계열

| 계열 | 예 | 장점 | 단점 |
|---|---|---|---|
| Filter | variance, correlation, chi-square, mutual information | 빠르고 모델 독립 | 상호작용 놓침 |
| Wrapper | RFE, sequential selection | 모델 성능 직접 반영 | 계산 비쌈, overfit 위험 |
| Embedded | L1, tree importance | 학습과 함께 선택 | 모델 편향에 종속 |

selection은 원래 feature 일부를 남기므로 해석성이 좋다. 중요도는 인과가 아니며, correlated feature 사이에 분산될 수 있다.

### 9.4 PCA 상세

PCA는 분산이 큰 직교 방향으로 투영한다. 입력 scale에 민감하며 target을 사용하지 않는다.

Explained variance ratio 누적합으로 차원을 정할 수 있지만, 최종 기준은 validation 성능과 시간·메모리다. 95% 분산 보존이 항상 예측에 필요한 신호 95%를 뜻하지 않는다. 작은 분산 방향이 target에 중요할 수 있다.

```python
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

pca_pipe = Pipeline([
    ("scale", StandardScaler()),
    ("pca", PCA(n_components=0.95, random_state=42)),
])
Xtr_pca = pca_pipe.fit_transform(X_train_num)
Xva_pca = pca_pipe.transform(X_valid_num)
```

### 9.5 LDA 차원축소

Linear Discriminant Analysis는 class 사이 분산을 크게, class 내부 분산을 작게 하는 축을 찾는다. feature 수가 F이고 class 수가 C이면 일반적인 구현의 출력 차원 상한은 `min(F, C-1)`이며, 표본 수와 scatter matrix rank 제약 때문에 실제 유효 rank가 더 작을 수도 있다. label을 사용하는 지도 방법이므로 반드시 fold 내부에 fit한다.

### 9.6 leakage-safe 파생변수

개체별 평균 target encoding이나 집계 feature는 특히 위험하다.

- test 시점에 실제로 얻을 수 있는 값인가?
- 현재 행 target 또는 미래 행을 포함하는가?
- validation group 통계가 train feature에 들어가는가?
- rolling window가 causal인가?

정답이 하나라도 문제라면 feature를 다시 정의한다.

### 9.7 실습문제

1. variance가 거의 0인 feature를 제거하는 방법은 세 계열 중 어디에 속하는가?
2. L1 선형모델이 feature selection 효과를 갖는 이유는?
3. 입력 feature가 6개인 4-class LDA의 최대 출력 차원은?
4. `rolling(window=10, center=True)`가 미래 예측에서 위험한 이유는?
5. PCA 95% explained variance와 95% 예측 정보가 같은 뜻이 아닌 이유는?

#### 정답·해설

1. target을 보지 않는 filter 방식.
2. L1 penalty의 비매끄러운 0 지점 때문에 일부 coefficient가 정확히 0이 되기 쉽다.
3. `min(F,C-1)=min(6,3)=3`.
4. 현재 시점 양쪽의 값을 사용해 미래 관측이 feature에 포함된다.
5. PCA는 target과 무관하게 X 분산을 보존한다. target 예측에 중요한 저분산 방향을 버릴 수 있다.

### 9.8 완료 기준

- [ ] selection 세 계열을 예와 함께 구분한다.
- [ ] PCA를 Pipeline/fold 안에서 fit한다.
- [ ] 모든 시간 파생변수의 causal 여부를 검사한다.

---

## 10. 선형회귀·로지스틱회귀·KNN

### 10.1 학습 목표

- 회귀와 분류의 선형 decision function을 구분한다.
- MSE, BCE와 likelihood의 연결을 설명한다.
- KNN의 k·거리·scale 효과를 예측한다.
- 단순 baseline이 갖는 시험 전략상 가치를 이해한다.

### 10.2 선형회귀

`ŷ = wᵀx + b`, MSE `L=(1/n)Σ(ŷ-y)²`를 최소화한다. 선형은 feature 관계가 선형이라는 뜻이지 원본 feature만 써야 한다는 뜻은 아니다. `x²`, interaction, log feature를 만들면 parameter에는 여전히 선형이다.

장점:

- 빠르고 안정적이며 해석 가능
- 고차원에서도 규제와 함께 강한 baseline
- 시계열 window를 flatten한 첫 제출에 유용

한계:

- 비선형 상호작용을 직접 만들지 않으면 표현 못 함
- 이상치와 다중공선성에 민감할 수 있음

Ridge는 L2, Lasso는 L1, Elastic Net은 둘을 결합한다.

### 10.3 로지스틱회귀

이진분류 logit은 `z=wᵀx+b`, 확률은 `σ(z)`다. decision boundary는 `z=0`, 즉 확률 0.5 지점이다. BCE는

`-[y log p + (1-y)log(1-p)]`.

PyTorch에서는 수치 안정성을 위해 sigmoid와 BCE를 합친 `BCEWithLogitsLoss`를 쓴다. 모델 forward에는 sigmoid를 붙이지 않는다.

다중분류는 class별 logits와 softmax를 사용하고 `CrossEntropyLoss`가 log-softmax와 NLL을 합쳐 계산한다.

### 10.4 KNN

새 샘플과 가까운 k개 이웃을 찾아 분류는 투표, 회귀는 평균한다. 학습 단계가 거의 없지만 prediction 때 전체 train과 거리를 계산해 느릴 수 있다.

k 효과:

- 너무 작음: 복잡한 경계, noise에 민감, 높은 variance
- 너무 큼: 경계가 지나치게 부드러움, 높은 bias, 소수 class 무시

거리는 scale에 매우 민감하다. 고차원에서는 모든 점 사이 거리가 비슷해지는 curse of dimensionality가 생긴다.

### 10.5 distance

- Euclidean: `sqrt(Σ(x_i-y_i)²)`
- Manhattan: `Σ|x_i-y_i|`
- Minkowski: p에 따라 일반화
- Mahalanobis: covariance와 scale·상관을 고려
- cosine distance: 방향 중심

### 10.6 PyTorch 최소 구현

```python
import torch
import torch.nn as nn

class LinearRegressor(nn.Module):
    def __init__(self, n_features, n_outputs=1):
        super().__init__()
        self.head = nn.Linear(n_features, n_outputs)

    def forward(self, x):
        return self.head(x)

class LogisticRegressor(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.head = nn.Linear(n_features, 1)

    def forward(self, x):
        return self.head(x)  # logits, sigmoid 없음
```

### 10.7 실습문제

1. Ridge의 규제 강도를 너무 크게 하면 bias와 variance는 어떻게 변하는가?
2. KNN에서 feature 하나의 단위가 0~1, 다른 하나가 0~100000이면 무엇이 문제인가?
3. 이진 logit 0의 확률은?
4. `BCEWithLogitsLoss` 앞에 sigmoid를 붙이면 왜 좋지 않은가?
5. 선형모델이 비선형 feature를 사용할 수 있는 예를 하나 쓰라.

#### 정답·해설

1. 일반적으로 bias는 증가하고 variance는 감소한다. 지나치면 underfitting.
2. 큰 scale feature가 거리를 지배한다. scaling이 필요하다.
3. `σ(0)=0.5`.
4. loss 내부 sigmoid와 중복되고, 수치 안정적인 결합 계산의 이점을 잃는다.
5. `x`와 `x²`를 입력하면 `w1 x + w2 x² + b`라는 곡선을 표현하지만 parameter에는 선형이다.

### 10.8 완료 기준

- [ ] 회귀·이진·다중분류의 출력/loss를 구분한다.
- [ ] k 변화에 따른 bias-variance를 설명한다.
- [ ] 빠른 선형 baseline을 5분 안에 만든다.

---

## 11. 결정트리·앙상블·SVM

### 11.1 학습 목표

- entropy, Gini, information gain을 계산한다.
- bagging과 boosting을 구분한다.
- SVM의 margin, C, kernel, gamma 의미를 설명한다.
- 표형 Problem에서 빠른 강한 baseline을 선택한다.

### 11.2 결정트리

트리는 feature threshold로 공간을 반복 분할한다. 분류 node impurity:

- Entropy: `-Σ p_k log₂ p_k`
- Gini: `1-Σp_k²`

좋은 split은 자식 node의 가중 impurity를 줄인다.

`Information Gain = impurity(parent) - Σ (n_child/n_parent) impurity(child)`.

회귀는 MSE/variance 감소를 사용할 수 있다.

트리의 장점은 비선형·상호작용, scale 불필요, 혼합 관계다. 단일 깊은 트리는 작은 데이터 변화에 구조가 크게 바뀌는 높은 variance가 단점이다.

### 11.3 손계산

class가 4:4인 node의 Gini는 `1-(0.5²+0.5²)=0.5`. 8:0이면 0이다. 완전히 순수한 node는 더 분할해도 impurity 감소가 없다.

### 11.4 Bagging·Random Forest·Extra Trees

Bagging은 bootstrap sample로 여러 모델을 독립 학습해 평균/투표한다. variance를 낮춘다.

Random Forest는 데이터 bootstrap과 함께 각 split에서 일부 feature만 후보로 보아 tree 사이 상관을 줄인다. Extra Trees는 threshold도 더 무작위로 뽑아 빠르고 다양한 tree를 만든다.

feature importance 함정:

- impurity importance는 고유값이 많은 feature에 편향될 수 있음
- correlated feature 사이 중요도가 나뉨
- permutation importance도 validation 데이터에서 평가해야 함

### 11.5 Boosting

Boosting은 앞 모델의 오류를 다음 모델이 보완하도록 순차적으로 결합한다.

- Gradient Boosting: loss의 negative gradient/residual을 새 tree가 학습
- XGBoost/LightGBM 계열: 규제·효율화를 추가한 강력한 표형 모델

Bagging은 병렬·variance 감소, boosting은 순차·bias까지 줄이는 경향으로 기억한다. boosting은 noise/잘못된 validation에 민감하고 tuning 시간이 들 수 있다.

### 11.6 SVM

선형 SVM은 두 class 사이 margin을 최대화하는 hyperplane을 찾는다. support vector는 경계를 결정하는 가까운 샘플이다.

- 큰 C: 마진 위반 penalty 큼, train 위반을 더 엄격히 줄이려는 경향, overfit 가능. 마진 폭의 단조 변화가 항상 보장되지는 않는다.
- 작은 C: 위반을 더 허용하고 상대적으로 강한 규제를 주는 경향
- RBF gamma 큼: 각 sample 영향 범위 좁음, 복잡한 경계
- gamma 작음: 영향 범위 넓음, 부드러운 경계

SVM은 scaling이 중요하고 큰 n에서 kernel SVM이 느릴 수 있다. sparse high-dimensional에서는 linear SVM이 강한 baseline이다.

### 11.7 모델 선택 감각

| 상황 | 첫 후보 |
|---|---|
| 혼합 표형, 비선형, 빠른 baseline | Extra Trees/Random Forest |
| 희소 one-hot 고차원 | linear/logistic/linear SVM |
| 작은 데이터, smooth boundary | RBF SVM 검토 |
| 매우 큰 표형 | 효율적 tree/linear, 시간 예산 확인 |
| 모델 구조 구현 요구 | 문제 지시에 맞는 PyTorch |

### 11.8 실습문제

1. class 비율 3/4, 1/4 node의 Gini를 계산하라.
2. Random Forest가 단일 tree보다 variance를 줄이는 원리는?
3. bagging과 boosting의 학습 순서를 비교하라.
4. SVM에서 C와 gamma를 모두 크게 했을 때 일반적인 위험은?
5. tree에 scaling이 보통 필요 없는 이유는?

#### 정답·해설

1. `1-(0.75²+0.25²)=0.375`.
2. 서로 덜 상관된 여러 tree의 오류를 평균하면 개별 변동이 상쇄된다.
3. bagging 구성 모델은 독립·병렬 가능, boosting은 이전 오류를 이용하므로 순차적이다.
4. 큰 C는 마진 위반을 강하게 벌하고 큰 gamma는 각 샘플의 영향 범위를 좁게 만들어, 훈련 잡음에 맞춘 복잡한 경계와 overfitting 위험이 커질 수 있다. C 증가가 모든 데이터에서 마진 폭을 반드시 줄인다고 단정하지 말고 scaling과 검증 성능을 함께 확인한다.
5. 값의 절대 거리보다 `x_j <= threshold`의 순서 분할을 사용하며 단조 scaling이 순서를 보존한다.

### 11.9 완료 기준

- [ ] Gini와 entropy 방향을 계산한다.
- [ ] RF/Extra Trees/boosting 차이를 30초 안에 설명한다.
- [ ] SVM C·gamma 변화 방향을 외웠다.

---

## 12. 군집화·DBSCAN·PCA와 비지도학습

### 12.1 학습 목표

- K-means 목적함수와 반복 절차를 설명한다.
- DBSCAN의 core/border/noise를 구분한다.
- silhouette score를 해석한다.
- cluster 번호를 정답 class처럼 해석하는 오류를 피한다.

### 12.2 K-means

목적은 각 점과 할당된 centroid 사이 squared distance 합을 최소화하는 것이다.

`J = Σ_i ||x_i - μ_{c_i}||²`.

반복:

1. k개 centroid 초기화
2. 각 점을 가장 가까운 centroid에 할당
3. cluster별 평균으로 centroid 갱신
4. 수렴할 때까지 반복

local optimum과 초기화에 민감해 k-means++와 여러 초기화를 쓴다. 구형·비슷한 크기/밀도의 cluster에 적합하며 scale과 이상치에 민감하다.

### 12.3 k 선택

- elbow: inertia 감소 곡선이 꺾이는 지점
- silhouette: 같은 cluster 응집도 a와 가장 가까운 다른 cluster 거리 b를 비교, `(b-a)/max(a,b)`
- 도메인 요구와 downstream validation

silhouette가 1에 가까우면 잘 분리, 0 근처면 경계, 음수면 다른 cluster가 더 가까울 수 있다.

### 12.4 DBSCAN

parameter는 이웃 반경 `eps`와 최소 이웃 수 `min_samples`다.

- core: eps 안에 충분한 점
- border: core 이웃이지만 스스로 core는 아님
- noise: 어느 cluster에도 연결되지 않음

장점은 cluster 수를 미리 정하지 않고 임의 형태와 noise를 다루는 것. 단점은 밀도가 다른 cluster, 고차원, eps 선택에 민감한 것.

### 12.5 계층 군집

Agglomerative는 각 점을 cluster로 시작해 가까운 cluster를 합친다. linkage:

- single: 가장 가까운 점 사이 거리, chaining 가능
- complete: 가장 먼 점 사이 거리, compact
- average: 평균 거리
- Ward: 합친 뒤 within-cluster variance 증가 최소화

Dendrogram의 cut 높이로 cluster 수를 정한다.

### 12.6 비지도 결과 해석

cluster label 0,1,2에는 순서나 원래 class 의미가 없다. clustering accuracy를 계산하려면 label permutation을 정렬해야 하고, 더 중요한 것은 군집 목적이 class 복원인지 구조 발견인지 정의하는 것이다.

PCA 시각화에서 cluster가 겹친다고 원공간에서도 반드시 겹치는 것은 아니다. 2D projection이 정보를 버렸을 수 있다.

### 12.7 실습문제

1. K-means에서 feature scaling이 중요한 이유는?
2. DBSCAN에서 eps를 지나치게 크게 하면 어떤 결과가 예상되는가?
3. silhouette가 음수인 sample의 의미는?
4. cluster 0을 “정상”, 1을 “고장”으로 바로 제출하면 안 되는 이유는?
5. single linkage의 chaining 현상을 설명하라.

#### 정답·해설

1. Euclidean squared distance가 큰 단위 feature에 지배되기 때문이다.
2. 서로 다른 cluster가 하나로 연결되고 noise가 줄어들 수 있다.
3. 자기 cluster보다 다른 cluster에 평균적으로 더 가까워 잘못 할당됐을 가능성.
4. cluster 번호는 임의이며 semantic class와 자동 대응하지 않는다.
5. 가까운 점 하나씩 다리처럼 이어지며 길고 분리돼 보이는 군집이 하나로 합쳐지는 현상.

### 12.8 완료 기준

- [ ] K-means E/M과 유사한 두 단계를 설명한다.
- [ ] DBSCAN 세 점 유형을 구분한다.
- [ ] 비지도 label의 임의성을 설명한다.

---

## 13. Validation Strategy와 데이터 누수

### 13.1 학습 목표

- 실제 test 생성 과정을 모사하는 split을 선택한다.
- random, stratified, group, time split을 구현한다.
- overlapping window와 target encoding 누수를 차단한다.
- 교차검증의 평균과 분산을 해석한다.

### 13.2 validation의 질문

“어떤 split 함수가 좋은가?”보다 먼저 묻는다.

> 실제 배포/test에서는 무엇이 새로 등장하는가?

- 독립 행: random split
- 같은 분포의 class 비율 유지: stratified
- 처음 보는 차량/사람/설비: group split
- 같은 개체의 미래: chronological split
- 여러 개체의 미래: group 내부 시간 split 또는 시점 경계 설계

### 13.3 random·stratified

회귀는 random, 분류는 class 비율을 보존하는 stratified가 흔한 기본이다. multilabel stratification은 단순하지 않다. label 조합·빈도를 검토하고, 최소한 validation에 모든 평가 가능한 label이 있는지 확인한다.

### 13.4 group split

```python
from sklearn.model_selection import GroupShuffleSplit

splitter = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, va_idx = next(splitter.split(X, y, groups=groups))
assert set(groups[tr_idx]).isdisjoint(set(groups[va_idx]))
```

group 행 수가 크게 다르면 validation 비율과 class 분포가 흔들릴 수 있다. 여러 seed 또는 GroupKFold를 검토한다.

### 13.5 time split

시간순 정렬 뒤 앞부분 train, 뒷부분 validation을 쓴다. `TimeSeriesSplit`은 expanding window 방식이다. embargo/gap은 인접 sample 정보가 거의 같은 경우 경계 사이를 비워 누수를 줄인다.

```python
order = np.argsort(times, kind="stable")
cut = int(len(order) * 0.8)
tr_idx, va_idx = order[:cut], order[cut:]
assert times.iloc[tr_idx].max() <= times.iloc[va_idx].min()
```

### 13.6 overlapping window 누수

원본 시계열에서 먼저 window를 모두 만들고 random split하면 거의 같은 raw 시점을 공유하는 window가 train/valid에 들어간다. validation 성능이 과대평가된다.

안전한 방식:

- 원본 시간 경계를 정하고 target index로 window를 배정
- group 단위 분리 후 각 group에서 window 생성
- 필요하면 경계에 lookback/horizon에 맞는 gap 적용

validation window가 경계 이전 과거를 input으로 쓰는 것은 실제 미래 예측을 모사한다면 허용될 수 있다. target은 반드시 validation 구간이어야 한다.

### 13.7 전처리 누수 목록

- 전체 데이터 imputer/scaler/PCA
- split 전 oversampling/SMOTE
- 전체 target으로 target encoding
- validation을 보고 feature·threshold를 무한 반복 선택
- 시계열 미래 rolling
- 동일 이미지의 augmentation 사본이 양 fold에 존재
- 같은 환자/차량/제품이 양 fold에 존재

Pipeline은 fold별 fit을 강제하는 강력한 안전장치다.

### 13.8 class 존재 검사

ROC-AUC는 binary validation에 두 class가 모두 있어야 한다. validation-only class가 있으면 모델이 학습할 수 없다.

```python
train_classes = set(np.unique(y[tr_idx]))
valid_classes = set(np.unique(y[va_idx]))
if not valid_classes.issubset(train_classes):
    raise ValueError("validation에 train에 없는 class가 있습니다")
if metric == "roc_auc" and (len(train_classes) != 2 or len(valid_classes) != 2):
    raise ValueError("binary ROC-AUC는 양 fold에 두 class가 필요합니다")
```

### 13.9 교차검증

K-fold는 각 sample을 validation에 한 번씩 사용한다. 장점은 split 우연을 줄이고 모든 train을 평가에 쓰는 것. 단점은 시간 K배와 test prediction 결합 복잡성이다. 170분 실기에서는 빠른 holdout으로 방향을 잡고, 작은 표형 데이터에서만 제한된 CV를 고려한다.

### 13.10 실습문제

1. 한 차량의 1,000개 센서 행을 random split했을 때 모델이 암기할 수 있는 것은?
2. SMOTE를 split 전에 하면 왜 누수인가?
3. 같은 차량의 미래를 예측하는데 차량 전체를 group split하면 무엇을 측정하는가?
4. target index가 validation이지만 input 일부가 train 경계 이전이면 무조건 누수인가?
5. validation score 평균은 같지만 표준편차가 큰 모델과 작은 모델 중 어떤 것을 선택할지 논하라.

#### 정답·해설

1. 차량 고유 baseline, 거의 동일한 인접 센서 패턴, ID성 특징을 학습해 새 차량 일반화를 과대평가할 수 있다.
2. validation sample을 이용해 만든 synthetic point가 train에 들어가거나 이웃 구조 정보가 새어 들어간다.
3. 같은 차량의 미래가 아니라 처음 보는 차량 일반화를 측정한다. 목표와 다를 수 있다.
4. 아니다. 실제 예측 시점에 과거가 이용 가능하다면 올바르다. target·미래 정보가 input에 들어가는지가 기준이다.
5. 일반적으로 작은 표준편차 모델이 안정적이지만, fold 설계·비용·최종 목표를 함께 본다. 평균만으로 자동 결정하지 않는다.

### 13.11 완료 기준

- [ ] 데이터 설명을 보고 split을 30초 안에 선택한다.
- [ ] group 교집합과 time 경계를 assert한다.
- [ ] 누수 목록 7개를 말할 수 있다.

---

## 14. 평가지표와 불균형

### 14.1 학습 목표

- 회귀·분류 metric 공식을 계산한다.
- label과 probability가 필요한 metric을 구분한다.
- macro/micro/weighted 평균을 설명한다.
- validation threshold를 안전하게 선택한다.

### 14.2 회귀 metric

| metric | 공식/성질 | 주의 |
|---|---|---|
| MSE | squared error 평균 | 큰 오차에 강한 penalty, target 단위² |
| RMSE | sqrt(MSE) | target 원 단위 |
| MAE | absolute error 평균 | 이상치에 상대적으로 robust |
| R² | 1-SSE/SST | 음수 가능, scale 비교 주의 |
| RMSLE | log1p 공간 RMSE | 음수 target 불가, under/over 비율 관점 |

RMSLE에서는 일반적으로 예측을 0 이상으로 clip하지만 문제 정의를 우선한다. target을 log1p로 학습했다면 `expm1`로 원복한 후 공식 metric을 계산한다.

### 14.3 confusion matrix

이진분류:

- Precision `TP/(TP+FP)`: 양성 예측 중 실제 양성
- Recall `TP/(TP+FN)`: 실제 양성 중 탐지
- Specificity `TN/(TN+FP)`
- F1 `2PR/(P+R)`

고장 놓침 비용이 크면 recall, 불필요 정비 비용이 크면 precision을 더 중시할 수 있다.

### 14.4 accuracy 함정

양성이 1%일 때 전부 음성으로 예측해도 accuracy 99%다. Macro-F1은 각 class F1을 동일 가중해 소수 class 실패를 숨기지 않는다. Weighted-F1은 support로 가중해 다수 class 영향이 크다. Micro-F1은 전체 TP/FP/FN을 합쳐 계산하며 single-label multiclass에서는 accuracy와 같아질 수 있다.

### 14.5 ROC-AUC와 PR-AUC

ROC curve는 threshold별 TPR 대 FPR, AUC는 양성 sample이 음성보다 높은 score를 받을 확률로 해석할 수 있다. 확률 또는 ranking score가 필요하며 hard label을 넣으면 정보가 사라진다.

희귀 양성에서는 많은 TN 때문에 FPR이 작아 보여 ROC-AUC가 낙관적일 수 있다. PR curve는 precision-recall tradeoff를 직접 보므로 유용하다. 어떤 metric을 제출하는지는 문제 지시를 따른다.

### 14.6 multiclass 평균

- Macro: class별 metric 단순 평균
- Weighted: class support 가중 평균
- Micro: 전체 결정을 합산

Multiclass ROC-AUC는 one-vs-rest 또는 one-vs-one과 평균 방식 정의가 필요하다. probability column 순서는 label encoder class 순서와 일치해야 한다.

### 14.7 threshold 선택

binary probability 0.5는 기본일 뿐 최적 F1 threshold가 아니다. validation에서만 탐색한다.

```python
from sklearn.metrics import f1_score

best_t, best_score = 0.5, -1.0
for t in np.linspace(0.05, 0.95, 91):
    score = f1_score(y_valid, valid_prob >= t, average="macro", zero_division=0)
    if score > best_score:
        best_t, best_score = float(t), float(score)
```

test threshold를 보고 조정하면 누수다. 전체 train 재학습 후 probability calibration이 달라질 수 있으므로 threshold 안정성도 확인한다.

### 14.8 불균형 대응

우선순위:

1. 올바른 stratified/group validation과 metric
2. class weight 또는 `pos_weight`
3. threshold 조절
4. resampling은 train fold 안에서만
5. focal loss 등은 baseline 이후

Binary `BCEWithLogitsLoss(pos_weight=Nneg/Npos)`는 양성 loss를 키운다. Multiclass `CrossEntropyLoss(weight=class_weights)`는 class별 가중치를 받는다.

### 14.9 실습문제

confusion matrix가 TP=30, FP=10, FN=20, TN=940일 때:

1. accuracy, precision, recall, F1을 계산하라.
2. hard label보다 연속 score 또는 probability를 보존해야 하는 metric 두 개를 쓰고, ROC-AUC와 log loss의 입력 요구가 어떻게 다른지 설명하라.
3. Macro-F1과 Weighted-F1 중 소수 class에 더 동일한 중요도를 주는 것은?
4. validation에서 threshold를 골랐는데 전체 재학습 후 calibration이 달라지는 이유는?
5. `pos_weight`를 전체 train+validation에서 계산하면 왜 엄밀히 누수인가?

#### 정답·해설

1. accuracy=`970/1000=.97`, precision=`30/40=.75`, recall=`30/50=.60`, F1=`2×.75×.60/1.35≈.667`.
2. 예를 들어 ROC-AUC와 Average Precision이다. 이진 ROC-AUC와 AP는 적절한 연속 ranking score를 사용할 수 있어 보정된 확률이 필수는 아니다. log loss는 클래스 확률 계약이 필요하다. hard label만 저장하면 원래 점수의 순위와 확률 정보를 잃는다.
3. Macro-F1.
4. 학습 sample과 parameter가 바뀌어 score 분포가 달라질 수 있다.
5. validation label 비율을 loss hyperparameter에 사용한다. split 비교 단계에서는 train fold만 사용해야 한다.

### 14.10 완료 기준

- [ ] confusion matrix에서 네 metric을 1분 안에 계산한다.
- [ ] 각 metric에 label/score 중 무엇이 필요한지 구분한다.
- [ ] 불균형 대응 순서를 외웠다.

---

## 15. Regularization·Hyperparameter Tuning·실험 설계

### 15.1 학습 목표

- L1/L2, dropout, early stopping의 작동 위치를 구분한다.
- grid/random/Bayesian search의 장단점을 설명한다.
- 공정한 ablation과 실험 기록표를 만든다.
- 제한 시간에 맞는 탐색 예산을 설정한다.

### 15.2 규제의 목적

규제는 모델이 train의 우연한 패턴을 과도하게 맞추는 것을 줄인다.

- L1: loss에 `λΣ|w|`, sparsity 유도
- L2: loss에 `λΣw²`, 큰 weight 억제
- Weight decay: optimizer update에서 parameter 축소
- Dropout: train 중 activation 일부를 무작위 0
- Early stopping: validation 악화 전 best checkpoint 선택
- Data augmentation: label-preserving 변형으로 유효 데이터 다양화
- Batch normalization은 주 목적이 규제는 아니지만 batch noise가 약한 규제 효과를 낼 수 있음

### 15.3 underfit과 overfit 처방

| 상태 | train | valid | 처방 방향 |
|---|---|---|---|
| underfit | 나쁨 | 나쁨 | feature/capacity/epoch↑, 규제↓ |
| overfit | 좋음 | 나쁨 | 데이터/규제↑, capacity/epoch↓, split 확인 |
| 둘 다 좋음 | 좋음 | 좋음 | 제출 확보, 작은 개선만 |
| valid가 비정상적으로 훨씬 좋음 | 더 나쁨 | 더 좋음 | augmentation/dropout 차이 또는 누수·분포 확인 |

### 15.4 hyperparameter와 parameter

Parameter는 학습으로 얻는 weight/bias. Hyperparameter는 learning rate, depth, hidden size, batch size, regularization strength처럼 학습 전/외부에서 정한다.

### 15.5 탐색 방법

- Grid search: 후보 조합 전부. 차원이 늘면 폭발.
- Random search: 정해진 횟수만 무작위. 중요한 일부 차원 탐색에 효율적.
- Bayesian optimization: 과거 결과로 다음 후보 선택. 설정 overhead와 라이브러리 의존.
- Manual coarse-to-fine: 실기에서 가장 현실적. LR/모델 크기 등 핵심 한두 축만.

로그 scale parameter(LR, weight decay)는 `[1e-5,1e-4,1e-3]`처럼 log 간격으로 본다.

### 15.6 공정한 ablation

한 번에 한 요소만 바꾸고 같은 split, seed, epoch/time budget, metric을 사용한다.

| run | model | LR | dropout | weight decay | valid | sec | 비고 |
|---|---|---:|---:|---:|---:|---:|---|
| A | MLP 128-64 | 1e-3 | 0.0 | 0 | ... | ... | baseline |
| B | MLP 128-64 | 1e-3 | 0.2 | 0 | ... | ... | dropout만 |
| C | MLP 128-64 | 1e-3 | 0.2 | 1e-4 | ... | ... | wd 추가 |

여러 요소를 동시에 바꾸면 개선 원인을 모른다. validation을 반복해 선택할수록 그 validation에 overfit한다.

### 15.7 시험용 탐색 예산

Problem 170분에서 모델 search에 모든 시간을 쓰지 않는다.

1. baseline 1개
2. PyTorch 모델 1개
3. 가장 영향 큰 변경 2~3개
4. best로 전체 train 재학습

한 run의 최대 시간을 정하고, 개선이 없으면 early stop한다. 외부 package 설치가 필요한 tuner는 피한다.

### 15.8 실습문제

1. learning rate와 weight decay 후보를 선형 간격보다 log 간격으로 보는 이유는?
2. train score도 valid score도 나쁠 때 dropout을 크게 늘리는 것이 왜 부적절할 수 있는가?
3. random search가 grid보다 유리한 전형적 조건은?
4. 같은 validation에 100개 모델을 비교한 best score가 낙관적인 이유는?
5. 실험 A/B를 공정하게 비교하기 위한 고정 요소 네 가지를 쓰라.

#### 정답·해설

1. 효과적인 규모가 여러 자릿수에 걸쳐 있고 비율 변화가 중요하기 때문이다.
2. 이미 underfit인데 capacity를 더 제한해 악화시킬 수 있다.
3. 탐색 차원이 많고 실제 성능에 중요한 hyperparameter는 일부이며 평가 예산이 제한될 때.
4. 우연히 validation noise에 맞은 후보를 선택하는 multiple comparison/validation overfitting 때문.
5. split, seed, metric, 학습 epoch/시간 budget. 전처리와 데이터도 같아야 한다.

### 15.9 완료 기준

- [ ] 학습곡선으로 under/overfit을 판별한다.
- [ ] 6회 이하의 작은 탐색표를 설계한다.
- [ ] best checkpoint와 최종 전체 재학습을 구분한다.

---

## 16. PyTorch 기초: Tensor·autograd·Dataset·DataLoader

### 16.1 학습 목표

- Tensor의 shape, dtype, device를 일관되게 관리한다.
- autograd graph와 `detach/no_grad/inference_mode`를 구분한다.
- `Dataset`과 `DataLoader`가 batch를 만드는 과정을 설명한다.
- dummy forward로 모델·loss 계약을 학습 전에 검사한다.

### 16.2 시험 시작 환경 셀

```python
import os
import random
import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, TensorDataset

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("torch:", torch.__version__)
print("device:", DEVICE)
if DEVICE.type == "cuda":
    print("gpu:", torch.cuda.get_device_name(0))
```

재현성은 완벽한 동일 결과를 보장하는 마법이 아니다. 하드웨어·CUDA kernel·multi-worker에 따라 차이가 날 수 있다. 시험에서는 비교 가능한 실험을 만드는 정도로 사용하고, deterministic 강제 때문에 속도가 크게 느려지지 않게 한다.

### 16.3 Tensor 생성과 dtype

```python
x_np = np.array([[1, 2], [3, 4]], dtype=np.float32)
x = torch.from_numpy(x_np)             # 메모리 공유 가능
x_copy = torch.tensor(x_np)            # 새 복사본

assert x.dtype == torch.float32
assert x.shape == (2, 2)
```

대표 target dtype:

| Task | feature | target |
|---|---|---|
| 회귀 | `float32` | `float32` |
| 이진분류 | `float32` | `float32`, `[B,1]` 또는 `[B]` 일관 |
| 다중분류 | `float32` | `long`, `[B]` class index |
| multilabel | `float32` | `float32`, `[B,K]` |

이 강의의 기본 class-index 경로에서는 `CrossEntropyLoss` target을 `(N,)`, `torch.long`, 값 `0..C−1`로 준비한다. float 값을 long으로 바꾸기 전에 정수인지 확인한다. PyTorch는 `(N,C)` 확률 target도 지원하지만 이는 soft-target용 별도 경로이며 class index와 혼합하지 않는다(17.9 참고).

### 16.4 device

모델과 입력·target이 같은 device에 있어야 한다.

```python
model = model.to(DEVICE)
for xb, yb in loader:
    xb = xb.to(DEVICE, dtype=torch.float32)
    yb = yb.to(DEVICE)
```

validation prediction은 CPU로 옮기고 NumPy로 변환한다.

```python
pred_np = logits.detach().cpu().numpy()
```

GPU tensor에 바로 `.numpy()`를 호출하거나 gradient가 연결된 tensor에 호출하면 오류가 난다.

### 16.5 autograd graph

`requires_grad=True`인 leaf parameter에서 시작한 연산은 graph를 만든다. `loss.backward()`는 leaf `.grad`에 gradient를 누적한다.

- `tensor.detach()`: 같은 값을 graph에서 분리
- `with torch.no_grad()`: 블록에서 gradient 기록 안 함
- `with torch.inference_mode()`: 평가 전용으로 더 강한 최적화, tensor mutation 제약 가능

validation에서는 `model.eval()`도 함께 호출해야 dropout/BN 동작이 바뀐다. `no_grad()`만으로는 model mode가 바뀌지 않는다.

### 16.6 Dataset

Map-style Dataset은 `__len__`, `__getitem__`을 구현한다.

```python
class ArrayDataset(Dataset):
    def __init__(self, X, y=None):
        X = np.asarray(X, dtype=np.float32)
        if X.ndim < 2:
            raise ValueError(f"X는 batch 축을 포함해야 합니다: {X.shape}")
        self.X = torch.from_numpy(X)
        self.y = None if y is None else torch.as_tensor(y)
        if self.y is not None and len(self.X) != len(self.y):
            raise ValueError("X/y 행 수 불일치")

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx] if self.y is None else (self.X[idx], self.y[idx])
```

### 16.7 DataLoader

```python
train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True,
    num_workers=0,
    pin_memory=(DEVICE.type == "cuda"),
    drop_last=False,
)
```

- train은 일반적으로 shuffle, validation/test는 false.
- 시계열 window Dataset의 sample 순서를 shuffle해도 각 window 내부 시간은 유지된다. stateful RNN 등 특별한 구조는 예외.
- `drop_last=True`는 마지막 작은 batch를 버린다. 데이터가 작거나 예측에서 쓰면 안 된다.
- 시험 환경에서는 worker 문제를 피하려 `num_workers=0`이 안전한 시작이다.

### 16.8 dummy forward 계약 검사

학습 전에 한 batch로 검사한다.

```python
xb, yb = next(iter(train_loader))
with torch.no_grad():
    out = model(xb.to(DEVICE, dtype=torch.float32))
print("x/y/out:", xb.shape, yb.shape, out.shape)
assert torch.isfinite(out).all()
```

그 다음 loss 한 번과 backward를 한다.

```python
model.train()
optimizer.zero_grad(set_to_none=True)
out = model(xb.to(DEVICE, dtype=torch.float32))
loss = criterion(out, yb.to(DEVICE))
assert loss.ndim == 0 and torch.isfinite(loss)
loss.backward()
optimizer.step()
```

### 16.9 실습문제

1. `torch.from_numpy`와 `torch.tensor(np_array)`의 메모리 차이는?
2. validation에서 `no_grad()`만 쓰고 `model.eval()`을 안 쓰면 어떤 layer가 문제인가?
3. batch size 1에서 무인자 `squeeze()`가 위험한 이유는?
4. multiclass target이 문자열이면 DataLoader 전에 무엇을 해야 하는가?
5. test 표본 수가 batch_size로 나누어떨어지지 않을 때 DataLoader에 drop_last=True를 쓰면 제출에 어떤 일이 생기는가?

#### 정답·해설

1. `from_numpy`는 가능한 경우 원본 NumPy와 메모리를 공유하고, `torch.tensor`는 보통 복사한다.
2. Dropout이 계속 켜지고 BatchNorm이 batch 통계를 사용한다.
3. 출력 `[1,1]`에서 batch와 feature 축이 모두 사라져 scalar가 될 수 있다. `squeeze(-1)`처럼 축을 지정한다.
4. 고정 mapping으로 `0..K-1` long index로 바꾸고 역 mapping을 저장한다.
5. 마지막 불완전 batch의 행을 버려 예측 행 수가 test보다 작아진다.

### 16.10 완료 기준

- [ ] CPU/GPU에서 같은 코드가 실행된다.
- [ ] task별 target dtype 표를 외웠다.
- [ ] dummy forward와 one-step backward를 항상 먼저 실행한다.

---

## 17. 퍼셉트론·MLP·활성함수·손실함수

### 17.1 학습 목표

- 퍼셉트론의 선형 decision boundary와 XOR 한계를 설명한다.
- MLP가 비선형 activation으로 표현력을 얻는 이유를 설명한다.
- 활성함수의 범위·gradient·사용 위치를 비교한다.
- task별 출력층, loss, 후처리를 정확히 연결한다.

### 17.2 퍼셉트론

`z=wᵀx+b`를 계산하고 step function으로 class를 나눈다. 하나의 퍼셉트론 decision boundary는 hyperplane이므로 선형 분리 가능한 문제만 해결한다. XOR은 단일 직선으로 분리할 수 없지만 hidden layer와 비선형 activation을 가진 MLP는 해결할 수 있다.

### 17.3 MLP

2층 MLP:

`h = φ(W₁x+b₁)`  
`o = W₂h+b₂`

activation 없이 linear layer 여러 개를 쌓으면 하나의 linear transform과 같다. `W₂(W₁x+b₁)+b₂ = W'x+b'`이기 때문이다. 비선형 activation이 깊이의 표현력을 만든다.

### 17.4 활성함수 비교

| 함수 | 출력 | 장점 | 단점/용도 |
|---|---|---|---|
| sigmoid | `(0,1)` | binary probability | saturation, hidden에서 vanishing |
| tanh | `(-1,1)` | zero-centered | saturation |
| ReLU | `[0,∞)` | 단순·빠름, gradient 보존 | dead ReLU |
| LeakyReLU | 음수에 작은 기울기 | dead 완화 | 음수 slope 선택 |
| GELU | 부드러운 gating | Transformer에 흔함 | 계산 조금 큼 |
| Softmax | 합 1의 K분포 | multiclass 확률 | loss 전 중복 적용 금지 |

Hidden layer 기본은 ReLU/GELU 계열. 출력 activation은 task와 loss 계약으로 정한다.

### 17.5 출력·loss·후처리 결정표

| Task | raw output | target | loss | 평가/제출 변환 |
|---|---|---|---|---|
| regression | `[B,D]` | float `[B,D]` | MSE/L1/Huber | 그대로, 필요시 inverse scale |
| binary | `[B,1]` | float `[B,1]` | BCEWithLogits | sigmoid→확률, threshold→label |
| multiclass | `[B,K]` | long `[B]` | CrossEntropy | softmax→확률, argmax→index→원 label |
| multilabel | `[B,K]` | float `[B,K]` | BCEWithLogits | label별 sigmoid/threshold |

이 표는 필기와 실기에서 가장 중요한 표 중 하나다.

### 17.6 범용 MLP

```python
class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden=(128, 64), dropout=0.1):
        super().__init__()
        layers = []
        d = input_dim
        for h in hidden:
            layers.extend([
                nn.Linear(d, h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            d = h
        layers.append(nn.Linear(d, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        x = x.flatten(start_dim=1)  # [B,...]→[B,F_total], batch 유지
        return self.net(x)
```

`input_dim=int(np.prod(X_train.shape[1:]))`로 정하면 2D 표형뿐 아니라 고정 shape flatten feature도 처리한다.

### 17.7 parameter 수

`Linear(I,O)` parameter는 weight `I×O`와 bias `O`, 총 `I×O+O`. `Linear(10,5)`는 55개다.

MLP `(20→64→3)`은 `20×64+64 + 64×3+3 = 1539`개다.

### 17.8 손실함수 상세

- MSE: 큰 error에 민감, 회귀
- MAE/L1: 0에서 미분 처리 필요, 이상치 robust
- Huber: 작은 error는 squared, 큰 error는 linear
- BCEWithLogits: sigmoid+BCE 수치 안정 결합
- CrossEntropy: log-softmax+NLL 결합, class index target

Loss와 평가 metric은 같을 필요가 없다. Macro-F1은 미분 불가능하므로 BCE/CE로 학습하고 validation F1로 모델을 선택한다.

### 17.9 실습문제

1. activation 없는 3개 Linear 층이 여전히 선형인 이유는?
2. `Linear(32,16)` parameter 수는?
3. binary 모델 output에 sigmoid가 없는데 확률은 언제 만드는가?
4. PyTorch CrossEntropyLoss의 기본 class-index 경로에서 target의 shape와 dtype은 무엇인가? one-hot [B,K]도 사용할 수 있는 별도 경로의 조건과 구분 이유를 설명하라.
5. MLP 입력 `[B,10,3]`의 올바른 `input_dim`은?

#### 정답·해설

1. affine transform의 합성은 하나의 affine transform으로 정리된다.
2. `32×16+16=528`.
3. loss는 logits를 직접 받고, validation metric/예측 단계에서 `torch.sigmoid`를 쓴다.
4. 기본 class-index 경로는 [B]의 torch.long 정답이며 값은 0 이상 K 미만이다. 입력 logits와 같은 [B,K]의 float 확률 target도 별도 경로로 지원하므로 올바른 one-hot도 사용할 수 있다. 이 경우 각 행이 유효한 확률 분포여야 한다. 기본 과제에서는 지정된 class-index 계약을 따르고 두 경로의 shape·dtype을 혼합하지 않는다.
5. `10×3=30`.

### 17.10 완료 기준

- [ ] task 네 종류의 output-target-loss-postprocess를 외웠다.
- [ ] MLP parameter 수와 dummy output shape를 계산한다.
- [ ] XOR에 hidden 비선형층이 필요한 이유를 말한다.

---

## 18. PyTorch 학습 엔지니어링: optimizer·초기화·BN·Dropout

### 18.1 학습 목표

- 안전한 train/validation loop를 빈 화면에서 작성한다.
- SGD/Momentum/RMSprop/Adam의 차이를 설명한다.
- Xavier/He 초기화, BatchNorm, Dropout을 올바른 mode에서 사용한다.
- early stopping, gradient clipping, scheduler를 진단에 맞게 쓴다.

### 18.2 공통 학습 순서

```text
model.train()
for batch:
    optimizer.zero_grad(set_to_none=True)
    logits = model(x)
    loss = criterion(logits, y)
    finite 확인
    loss.backward()
    필요시 clip_grad_norm_
    optimizer.step()

model.eval()
with inference_mode():
    validation prediction/loss/metric
best checkpoint 저장
```

### 18.3 안전한 한 epoch

```python
def train_one_epoch(model, loader, criterion, optimizer, device, clip_norm=None):
    if len(loader) == 0:
        raise ValueError("빈 train loader")
    model.train()
    total_loss, total_n = 0.0, 0

    for batch in loader:
        xb, yb = batch
        xb = xb.to(device, dtype=torch.float32)
        yb = yb.to(device)

        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss: {loss.item()}")
        loss.backward()
        if clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip_norm)
        optimizer.step()

        n = xb.shape[0]
        total_loss += loss.detach().item() * n
        total_n += n
    return total_loss / total_n
```

criterion에 넣기 전에 task에 맞게 y shape/dtype을 바꾸는 함수가 필요하다. 거대한 범용 loop보다 계약을 명시한 task별 wrapper가 디버깅하기 쉽다.

### 18.4 optimizer

SGD: `w←w-ηg`. Momentum은 velocity `v←βv+g`, `w←w-ηv` 형태로 과거 방향을 누적한다. RMSprop은 recent squared gradient moving average로 각 parameter step을 나눈다. Adam은 gradient 평균과 제곱 평균을 모두 쓰고 초기 bias를 보정한다.

시험 실기 기본:

```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```

성능이 안 좋을 때 optimizer를 무작정 바꾸기보다 데이터·shape·LR·split을 먼저 확인한다.

#### 같은 gradient의 첫 update 비교

`w₀=1`, gradient `g=2`, learning rate `η=0.1`, 모든 optimizer state는 0이라고 하자.

- SGD: `w₁=1-0.1×2=0.8`.
- Momentum을 `v←0.9v+g`, `w←w-ηv`로 정의하면 첫 step은 `v=2`, `w₁=0.8`. 다음 step부터 과거 방향의 효과가 나타난다.
- RMSprop에서 decay 0.9, `s←0.9s+0.1g²`이면 `s=0.4`, ε를 생략한 첫 step은 `w₁≈1-0.1×2/sqrt(0.4)≈0.6838`.
- Adam에서 `β₁=0.9, β₂=0.999`이면 `m=0.2,v=0.004`. bias correction 후 `m_hat=2,v_hat=4`, 따라서 ε를 무시하면 `w₁=0.9`.

라이브러리마다 momentum 표기와 weight decay 결합 방식이 다를 수 있으므로 문제에 주어진 update 식을 우선한다. 이 계산의 목적은 Adam의 bias correction이 첫 moment를 원래 규모로 보정한다는 것을 이해하는 것이다.

### 18.5 초기화

- Xavier/Glorot: fan-in과 fan-out을 고려, tanh/sigmoid 계열
- He/Kaiming: ReLU가 일부 activation을 0으로 만드는 것을 고려
- bias는 흔히 0

PyTorch Linear/Conv는 합리적 기본 초기화를 제공한다. 명세가 요구하거나 깊은 custom network에서만 직접 적용한다.

```python
def init_relu(module):
    if isinstance(module, (nn.Linear, nn.Conv1d, nn.Conv2d)):
        nn.init.kaiming_normal_(module.weight, nonlinearity="relu")
        if module.bias is not None:
            nn.init.zeros_(module.bias)
```

### 18.6 Batch Normalization

mini-batch activation을 정규화한 뒤 learnable scale γ와 shift β를 적용한다. train에서는 batch mean/variance와 running statistics를 갱신하고 eval에서는 running statistics를 사용한다.

batch 평균 `μ_B`, 분산 `σ²_B`에 대해 `x_hat=(x-μ_B)/sqrt(σ²_B+ε)`, 출력은 `y=γx_hat+β`다. Conv2d의 BatchNorm2d는 channel별 통계를 batch와 공간축에서 계산한다. γ와 β는 gradient로 학습하고 running mean/variance는 optimizer가 학습하는 parameter가 아니라 buffer다.

효과:

- optimization 안정화, 더 큰 LR 가능
- 초기화 민감도 완화
- 약한 stochastic regularization

함정:

- batch가 매우 작으면 통계 불안정
- train/eval mode 오류
- RNN sequence에 기계적으로 적용하지 않음; LayerNorm이 자연스러운 경우 많음

### 18.7 Dropout

train에서 확률 p로 activation을 0으로 만들고 나머지를 scale한다. eval에서는 꺼진다. p가 높을수록 규제가 강하다. output layer에 무조건 붙이지 않고 hidden representation에 사용한다.

### 18.8 early stopping과 checkpoint

validation metric이 개선될 때 CPU state dict를 복사한다.

```python
import copy

best_value = float("inf")
best_state = None
wait = 0
for epoch in range(max_epochs):
    train_loss = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
    valid_value = evaluate(model, valid_loader)
    if valid_value < best_value - min_delta:
        best_value = valid_value
        best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        wait = 0
    else:
        wait += 1
        if wait >= patience:
            break
if best_state is None:
    raise RuntimeError("유효한 checkpoint가 없습니다")
model.load_state_dict(best_state)
model.to(DEVICE)
```

metric이 클수록 좋은 F1/AUC라면 비교 방향을 반대로 한다.

### 18.9 scheduler와 gradient clipping

- ReduceLROnPlateau: validation 정체 시 LR 감소
- Step/Cosine: 정해진 schedule
- Warmup: Transformer 등에서 초기 큰 update 완화
- Gradient clipping: RNN exploding gradient에 특히 유용. 근본적인 NaN 입력을 숨기는 용도가 아님

### 18.10 실습문제

1. PyTorch BatchNorm의 기본 track_running_stats=True에서 train/eval의 통계 사용과 running statistics 갱신 차이를 쓰라.
2. dropout p를 0.2에서 0.8로 높이면 일반적으로 train/valid에 어떤 영향이 가능한가?
3. ReLU network에 He 초기화가 적합한 직관은?
4. validation F1 early stopping에서 `min` 비교를 쓰면 어떤 오류인가?
5. gradient clipping이 vanishing gradient를 해결하는가?

#### 정답·해설

1. train은 batch 통계와 running 갱신, eval은 저장 running 통계 사용.
2. 규제가 매우 강해 train 성능이 나빠지고 underfit할 수 있다. valid가 좋아질 수도 있지만 보장되지 않는다.
3. ReLU로 일부 activation이 0이 되는 분산 변화를 fan-in 기준으로 보정해 층을 지나며 variance를 유지한다.
4. 낮은 F1을 best로 저장한다. metric 방향을 `max`로 해야 한다.
5. 아니다. 큰 gradient 폭발을 제한하며, 작은 gradient 소실을 키우지 않는다.

### 18.11 완료 기준

- [ ] train/eval loop를 15분 안에 작성한다.
- [ ] optimizer 4종의 핵심 상태를 비교한다.
- [ ] BN/Dropout mode와 checkpoint 방향을 검사한다.

---

## 19. CNN: convolution·padding·pooling·이미지 코드

### 19.1 학습 목표

- 1D/2D convolution 출력 크기와 parameter 수를 계산한다.
- channel, kernel, stride, padding, dilation, receptive field를 설명한다.
- PyTorch NCHW/NCT shape를 안전하게 다룬다.
- 작은 이미지 CNN을 구현한다.

### 19.2 convolution 직관

작은 kernel을 위치마다 공유하여 local pattern을 찾는다. weight sharing으로 fully connected보다 parameter가 적고 translation equivariance를 갖는다. Pooling이나 global aggregation은 위치 변화에 더 robust한 표현을 만든다.

### 19.3 출력 크기 공식

각 공간축:

`out = floor((in + 2P - D(K-1) - 1)/S + 1)`

- input `in`
- kernel `K`
- padding `P`
- stride `S`
- dilation `D`

`K=7,S=1,D=1`에서 input 크기를 유지하려면 `P=3`이다.

### 19.4 parameter 수

Conv2d parameter:

`out_channels × (in_channels/groups × K_h × K_w) + out_channels(bias)`.

`Conv2d(3,64,7,bias=True)`는 `64×(3×7×7)+64 = 9472`개다. H/W와 parameter 수는 무관하다.

### 19.5 channel 의미

첫 layer input channel은 RGB 3, grayscale 1, 센서 Conv1d에서는 feature 수 F다. output channel은 학습된 filter 수다. 이미지 NumPy HWC를 PyTorch CHW로 바꾼다.

```python
img = np.asarray(pil_img, dtype=np.float32) / 255.0  # [H,W,C]
if img.ndim == 2:
    img = img[..., None]
x = torch.from_numpy(img).permute(2, 0, 1)          # [C,H,W]
```

### 19.6 pooling

- MaxPool: local 최대 response, 강한 특징
- AveragePool: 평균
- AdaptiveAvgPool2d((1,1)): 입력 H/W와 무관하게 channel별 1값, classifier head dimension 고정

Pooling은 parameter가 없다. stride convolution으로 downsample할 수도 있다.

### 19.7 receptive field

깊은 layer 한 값이 원본에서 보는 영역이다. 여러 3×3 convolution을 쌓으면 parameter 효율적으로 receptive field가 커지고 중간 비선형성도 늘어난다. dilation은 kernel 사이 간격을 벌려 parameter 수 없이 receptive field를 키운다.

### 19.8 작은 Image CNN

```python
class ImageCNN(nn.Module):
    def __init__(self, in_channels, out_dim):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.head = nn.Linear(128, out_dim)

    def forward(self, x):
        x = self.features(x)       # [B,128,1,1]
        return self.head(x.flatten(1))
```

Dummy test:

```python
m = ImageCNN(3, 5)
assert m(torch.zeros(2, 3, 64, 64)).shape == (2, 5)
assert m(torch.zeros(1, 3, 80, 96)).shape == (1, 5)
```

Adaptive pooling 덕분에 서로 다른 H/W도 head에 들어간다. DataLoader batch 안에서는 기본 collate를 쓰려면 H/W가 같아야 하므로 resize/pad가 필요하다.

### 19.9 PIL crop

PIL crop box는 `(left, upper, right, lower)`이며 right/lower는 포함되지 않는다. 결과 width=`right-left`, height=`lower-upper`.

```python
from PIL import Image

def crop_to_array(image, box):
    if not isinstance(image, Image.Image):
        raise TypeError("PIL Image가 필요합니다")
    left, upper, right, lower = box
    if right <= left or lower <= upper:
        raise ValueError("유효하지 않은 crop box")
    return np.asarray(image.crop(box))
```

### 19.10 실습문제

1. input 32, K=3, P=1, S=2, D=1의 output 크기는?
2. `Conv2d(16,32,3,padding=1,bias=False)` parameter 수는?
3. RGB `[B,64,64,3]` tensor를 NCHW로 바꾸는 코드는?
4. pooling이 parameter 수를 늘리는가?
5. PIL box `(10,20,50,70)`의 결과 width/height는?

#### 정답·해설

1. `floor((32+2-2-1)/2+1)=16`.
2. `32×16×3×3=4608`.
3. `x = x.permute(0,3,1,2)`.
4. 일반 max/average/adaptive pooling은 학습 parameter가 없다.
5. width 40, height 50.

### 19.11 완료 기준

- [ ] convolution output/parameter를 1분 안에 계산한다.
- [ ] HWC↔CHW와 `[B,T,F]↔[B,F,T]`를 구분한다.
- [ ] 임의 H/W dummy forward를 통과한다.

---

## 20. ResNet과 전이학습

### 20.1 학습 목표

- 깊은 plain network의 degradation과 residual learning을 설명한다.
- skip addition 전 shape 조건을 검사한다.
- pretrained model의 head 교체, freeze, unfreeze 순서를 구현한다.
- train/validation transform과 pretrained normalization을 맞춘다.

### 20.2 residual learning

plain block이 원하는 mapping `H(x)`를 직접 학습하는 대신 residual `F(x)=H(x)-x`를 학습하고 `y=F(x)+x`로 만든다. identity 경로는 gradient와 정보를 직접 전달해 매우 깊은 network 최적화를 돕는다.

[필기 함정] ResNet의 핵심 문제는 단순한 overfitting만이 아니라 깊어질수록 train error조차 나빠질 수 있는 degradation/optimization 문제다.

### 20.3 shape 조건

`F(x)`와 shortcut의 shape가 같아야 더할 수 있다. channel 수나 stride로 공간 크기가 바뀌면 1×1 projection을 쓴다.

```python
class ResidualBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.main = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = (
            nn.Identity()
            if in_ch == out_ch and stride == 1
            else nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )
        )
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(self.main(x) + self.shortcut(x))
```

### 20.4 전이학습

큰 데이터에서 pretrained feature extractor를 작은 새 task에 재사용한다.

1. pretrained weight에 맞는 입력 channel·resize·normalization 확인
2. classifier head를 새 class/output 수로 교체
3. backbone freeze, head 먼저 학습
4. 필요하면 마지막 block부터 일부 unfreeze
5. backbone에는 작은 learning rate

```python
from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
for p in model.parameters():
    p.requires_grad = False
in_features = model.fc.in_features
model.fc = nn.Linear(in_features, NUM_CLASSES)
model = model.to(DEVICE)

optimizer = torch.optim.AdamW(model.fc.parameters(), lr=1e-3)
```

인터넷 다운로드가 시험 중 실패할 수 있으므로 weight가 로컬에 없으면 무리하게 의존하지 않는다. 문제에서 pretrained 사용을 요구하거나 환경에 확실히 준비된 경우 사용한다.

### 20.5 normalize

pretrained weight가 기대하는 mean/std를 사용한다. scratch model에서는 train 데이터에 맞는 정규화 또는 `[0,1]`로 시작할 수 있다. pretrained normalization을 임의로 생략하면 feature distribution이 달라진다.

### 20.6 freeze와 BatchNorm

`requires_grad=False`는 parameter gradient만 막는다. `model.train()`을 호출하면 frozen BatchNorm running statistics는 여전히 바뀔 수 있다. 작은 데이터에서는 backbone eval 유지 또는 BN 정책을 명시적으로 정해야 한다.

### 20.7 언제 scratch인가

- 입력이 자연 이미지와 매우 다름
- channel 수가 다름
- pretrained weight 없음/다운로드 위험
- 데이터가 충분하고 작은 custom CNN이 시간 예산에 적합
- Process가 정확한 architecture를 요구

### 20.8 실습문제

1. residual addition에서 input `[B,32,64,64]`, main `[B,64,32,32]`이면 shortcut에 무엇이 필요한가?
2. 학습 시작 전에 backbone을 requires_grad=False로 동결했고 각 frozen parameter의 .grad가 None이다. head를 교체한 뒤 일반적인 PyTorch optimizer에 model.parameters() 전체를 넣으면 frozen weight가 갱신되는가? 학습 중간에 동결하는 경우의 추가 주의점도 쓰라.
3. `requires_grad=False`와 `model.eval()`은 같은가?
4. pretrained ResNet에 grayscale image를 넣는 해결책 두 가지는?
5. 전이학습이 항상 scratch보다 좋은가?

#### 정답·해설

1. `Conv2d(32,64,kernel_size=1,stride=2)`와 보통 BN projection.
2. 주어진 grad=None 조건에서는 일반적인 PyTorch optimizer가 해당 parameter를 건너뛰므로 frozen weight는 갱신되지 않는다. 학습 중간에 동결하면 이전 .grad가 남아 있을 수 있으므로 gradient와 optimizer 구성을 정리해야 한다. trainable parameter만 optimizer에 넣으면 의도가 명료하다. BN running statistics의 변화는 별도로 확인한다.
3. 아니다. 전자는 parameter gradient, 후자는 Dropout/BN mode를 바꾼다.
4. grayscale을 3채널 반복하거나 첫 conv를 1채널로 교체하고 weight를 평균/재초기화한다.
5. domain 차이, 입력 규격, 데이터 크기, normalization, tuning에 따라 다르다. validation으로 판단한다.

### 20.9 완료 기준

- [ ] projection shortcut 필요 여부를 shape로 판단한다.
- [ ] head-only→partial unfreeze 순서를 구현한다.
- [ ] pretrained transform과 BN mode를 확인한다.

---

## 21. 시계열 데이터 계약과 window

### 21.1 학습 목표

- sampling rate, lookback, horizon을 행 index로 변환한다.
- group 경계를 넘지 않는 causal window를 만든다.
- target index 기준 time validation을 설계한다.
- 큰 데이터에서 materialized window와 lazy Dataset을 선택한다.

### 21.2 문제를 index로 번역하기

sampling rate가 100Hz면 0.2초는 20 sample, 1초는 100 sample이다. “과거 0.2초를 보고 마지막 관측으로부터 1초 뒤를 예측”이라면:

- lookback `L=20`
- horizon `H=100`
- 마지막 input index `e`
- input `X[e-L+1 : e+1]`
- target `y[e+H]`

길이 T sequence, stride 1에서 가능한 window 수는 `max(T-L-H+1,0)`다. horizon 정의가 “다음 시점”을 1로 세는지 0으로 세는지 문제 예시로 반드시 확인한다.

### 21.3 정렬과 sequence 경계

vehicle ID만 같아도 주행 사이 1시간 gap이 있으면 한 연속 sequence가 아닐 수 있다. group은 차량, trip, cycle, gap 기준 segment 중 실제 생성 과정에 맞게 정한다.

```python
work = df.copy()
work["timestamp"] = pd.to_datetime(work["timestamp"], errors="raise")
work["__row__"] = np.arange(len(work))
work = work.sort_values(["trip_id", "timestamp", "__row__"], kind="stable")

gap = work.groupby("trip_id", observed=True)["timestamp"].diff().dt.total_seconds()
work["segment"] = gap.gt(MAX_GAP_SECONDS).groupby(work["trip_id"]).cumsum()
work["sequence_id"] = list(zip(work["trip_id"], work["segment"]))
```

### 21.4 작은 데이터 window

```python
def make_windows(X, y, lookback, horizon, stride=1, assume_sorted=False):
    X = np.asarray(X)
    y = np.asarray(y)
    if not assume_sorted:
        raise ValueError("시간 정렬을 확인한 뒤 assume_sorted=True를 명시하세요")
    if X.ndim != 2 or y.ndim not in (1, 2):
        raise ValueError(f"X [T,F], y [T] 또는 [T,D] 필요: {X.shape}, {y.shape}")
    if len(X) != len(y) or lookback < 1 or horizon < 0 or stride < 1:
        raise ValueError("길이/lookback/horizon/stride 계약 오류")

    Xw, yw, target_idx = [], [], []
    last_e = len(X) - 1 - horizon
    for e in range(lookback - 1, last_e + 1, stride):
        Xw.append(X[e - lookback + 1:e + 1])
        yw.append(y[e + horizon])
        target_idx.append(e + horizon)

    y_tail = (() if y.ndim == 1 else (y.shape[1],))
    if not Xw:
        return (
            np.empty((0, lookback, X.shape[1]), dtype=X.dtype),
            np.empty((0, *y_tail), dtype=y.dtype),
            np.empty((0,), dtype=np.int64),
        )
    return np.stack(Xw), np.asarray(yw), np.asarray(target_idx)
```

### 21.5 group-safe window

각 group을 별도로 정렬·호출하고 target group/원본 행을 함께 반환한다. 반환된 window를 다시 random split하면 동일 group·인접 window가 양 fold에 섞일 수 있다.

- 새 차량 평가: target group으로 GroupShuffleSplit
- 같은 차량의 미래: target timestamp 경계
- 새 주행 평가: trip group split

### 21.6 target-index split

실시간 예측에서는 학습 label이 validation의 예측 원점(입력 끝)에 이미 알려져 있어야 한다. target index만 `cut`으로 나누면 horizon>1에서 미래 정답을 학습에 쓸 수 있다. 아래 `HORIZON`은 window를 생성할 때 사용한 같은 행 간격이다. label 확정 지연이 있으면 train 조건에 더한다. 별도 배치평가의 target-only split과 구분한다.

```python
train_mask = target_idx < cut
valid_mask = target_idx - HORIZON >= cut
Xtr, ytr = Xw[train_mask], yw[train_mask]
Xva, yva = Xw[valid_mask], yw[valid_mask]
assert len(Xtr) and len(Xva)
assert target_idx[train_mask].max() < (target_idx[valid_mask] - HORIZON).min()
```

### 21.7 sequence scaling

`Xtr:[N,T,F]` scaler는 sample·time을 합친 `[N×T,F]`에 fit하고 원래 shape로 복원한다.

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
F = Xtr.shape[-1]
Xtr_s = scaler.fit_transform(Xtr.reshape(-1, F)).reshape(Xtr.shape)
Xva_s = scaler.transform(Xva.reshape(-1, F)).reshape(Xva.shape)
Xte_s = scaler.transform(Xte.reshape(-1, F)).reshape(Xte.shape)
```

target scaling을 했다면 prediction을 inverse transform한 후 metric과 제출을 계산한다.

### 21.8 lazy Dataset

window 원소를 미리 복사하지 않고 마지막 input index 목록만 저장한다.

```python
class LazyWindowDataset(Dataset):
    def __init__(self, X, y, end_indices, lookback, horizon):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = None if y is None else torch.as_tensor(y, dtype=torch.float32)
        self.ends = np.asarray(end_indices, dtype=np.int64)
        self.lookback = int(lookback)
        self.horizon = int(horizon)

    def __len__(self):
        return len(self.ends)

    def __getitem__(self, i):
        e = int(self.ends[i])
        x = self.X[e - self.lookback + 1:e + 1]
        if self.y is None:
            return x
        return x, self.y[e + self.horizon]
```

`X` 전체 tensor 복사는 하나지만 `N×L×F` window 복사를 피한다. group별 연속 array 또는 index mapping은 데이터 구조에 맞게 확장한다.

### 21.9 실습문제

1. 50Hz, lookback 0.4초, horizon 2초는 각각 몇 sample인가?
2. T=1000, L=20, H=100, stride 1의 window 수는?
3. window를 먼저 random split할 때 발생하는 누수를 설명하라.
4. 시점 799의 관측과 정답이 확정된 직후, 그때까지 알려진 label만으로 학습을 마치고 target 800을 예측한다. 입력은 인덱스 780부터 799이고 H=1이다. validation target 구간이 800부터라는 조건에서, 입력에 train 기간의 과거가 들어갔다는 이유만으로 누수인가? 모델을 cut=800 직전에 확정하고 validation 예측 원점을 800 이상으로 제한하는 별도 설계와도 구분하라.
5. float32 `(900000,20,23)` window의 대략 메모리는?

#### 정답·해설

1. L=20, H=100.
2. `1000-20-100+1=881`.
3. 인접 window가 대부분의 raw 시점을 공유해 validation이 사실상 train 복사본과 비슷해진다.
4. 첫 설계에서는 원점 799에 입력과 모든 학습 label이 이미 알려졌으므로 과거 입력을 공유했다는 이유만으로 누수는 아니다. 별도 설계가 validation 원점을 800 이상으로 제한한다면 이 창의 원점은 799이므로 그 validation에는 포함하지 않는다. target index만으로 허용 여부를 정하지 말고 모델 확정 시점과 예측 원점을 함께 확인한다.
5. `900000×20×23×4≈1.656GB`(10진), 여기에 복사·activation이 추가된다.

### 21.10 완료 기준

- [ ] 초 단위를 sample로 즉시 바꾼다.
- [ ] off-by-one을 작은 `np.arange`로 검산한다.
- [ ] target-index split과 lazy Dataset을 구현한다.

---

## 22. RNN·LSTM·GRU와 1D CNN

### 22.1 학습 목표

- recurrent hidden state와 BPTT를 설명한다.
- RNN의 vanishing/exploding gradient 원인을 설명한다.
- LSTM gate와 GRU의 역할을 구분한다.
- 같은 `[B,T,F]` 입력을 CNN1D/RNN에 맞게 변환한다.

### 22.2 기본 RNN

`h_t = φ(W_x x_t + W_h h_{t-1} + b)`.

같은 parameter를 모든 시점에 공유한다. many-to-one은 마지막/집계 hidden으로 하나를 예측하고, many-to-many는 시점마다 출력을 낸다. sequence-to-sequence는 입력·출력 길이가 다를 수 있다.

BPTT는 시간축으로 계산 graph를 펼쳐 gradient를 전파한다. `W_h`가 반복 곱해져 고유값/activation derivative에 따라 gradient가 지수적으로 작아지거나 커질 수 있다.

### 22.3 LSTM

대표 gate:

- forget `f_t`: 이전 cell state를 얼마나 유지
- input `i_t`: 새 candidate를 얼마나 기록
- candidate `g_t`: 새 내용
- output `o_t`: cell에서 hidden으로 얼마나 노출

`c_t = f_t ⊙ c_{t-1} + i_t ⊙ g_t`  
`h_t = o_t ⊙ tanh(c_t)`

한 가지 표준 표기는 다음과 같다.

`f_t=σ(W_f[x_t,h_{t-1}]+b_f)`  
`i_t=σ(W_i[x_t,h_{t-1}]+b_i)`  
`g_t=tanh(W_g[x_t,h_{t-1}]+b_g)`  
`o_t=σ(W_o[x_t,h_{t-1}]+b_o)`

입력 크기 F, hidden H인 단방향 한 층 LSTM은 input-hidden weight `4H×F`, hidden-hidden weight `4H×H`와 구현별 bias를 가진다. PyTorch는 `bias_ih`, `bias_hh` 두 bias 묶음을 저장하므로 parameter 수를 계산할 때 둘 다 포함한다.

cell state의 additive path가 장기 gradient 전달을 돕는다. “완전히 vanishing을 제거”한다고 단정하지 않는다.

### 22.4 GRU

update gate와 reset gate를 사용하고 별도 cell state 없이 hidden을 갱신한다. LSTM보다 parameter가 적고 빠를 수 있다. 어느 것이 항상 우월하지 않으며 validation과 시간 예산으로 결정한다.

### 22.5 PyTorch sequence model

```python
class SequenceRNN(nn.Module):
    def __init__(self, n_features, out_dim, hidden=64, layers=1,
                 cell="gru", bidirectional=False, dropout=0.0):
        super().__init__()
        rnn_cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[cell]
        self.rnn = rnn_cls(
            input_size=n_features,
            hidden_size=hidden,
            num_layers=layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if layers > 1 else 0.0,
        )
        self.directions = 2 if bidirectional else 1
        self.head = nn.Linear(hidden * self.directions, out_dim)

    def forward(self, x):                # [B,T,F]
        _, state = self.rnn(x)
        # LSTM state=(h_n,c_n), RNN/GRU state=h_n.
        h_n = state[0] if isinstance(state, tuple) else state  # [layers*D,B,H]
        last_layer = h_n[-self.directions:]                    # [D,B,H]
        features = last_layer.transpose(0, 1).reshape(x.size(0), -1)
        return self.head(features)                             # [B,out_dim]
```

[함정] bidirectional 출력의 `seq[:,-1,:]`를 그대로 쓰면 backward 절반은 전체 sequence가 아니라 마지막 token 하나만 본 상태다. 위처럼 마지막 layer의 `h_n` 양방향을 합친다. padded variable-length sequence는 `h_n`도 padding을 처리하므로 lengths와 packed sequence를 사용한다.

### 22.6 bidirectional 함정

입력 window 전체가 예측 시점 이전이라면 window 내부 bidirectional RNN은 과거 구간 양방향 문맥을 사용할 수 있다. 하지만 online 시점별 출력에서 미래 시점을 사용하는 bidirectional은 causal 요구를 위반할 수 있다. 문제 정의를 확인한다.

### 22.7 CNN1D

센서 feature를 channel로, 시간을 길이로 둔다.

```python
class SensorCNN1D(nn.Module):
    def __init__(self, n_features, out_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(n_features, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv1d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.head = nn.Linear(128, out_dim)

    def forward(self, x):               # [B,T,F]
        x = x.permute(0, 2, 1)          # [B,F,T]
        x = self.net(x).squeeze(-1)      # [B,128]
        return self.head(x)
```

CNN1D는 병렬 계산이 빠르고 local pattern에 강해 첫 sequence baseline으로 좋다. RNN/GRU는 순서 상태를 명시적으로 누적한다. lookback이 짧고 데이터가 크면 CNN1D를 먼저 시도한다.

### 22.8 variable length와 mask

```python
from torch.nn.utils.rnn import pack_padded_sequence

packed = pack_padded_sequence(
    padded_x, lengths.cpu(), batch_first=True, enforce_sorted=False
)
packed_out, state = model.rnn(packed)
```

Transformer는 padding mask를 attention에 전달한다. padding 값을 0으로 만드는 것만으로 모델이 항상 무시하는 것은 아니다.

### 22.9 실습문제

1. `RNN(input_size=10, hidden_size=32, batch_first=True)` 입력 `[64,20,10]`의 sequence output shape는?
2. 2층 LSTM에 dropout을 0.3 지정했을 때 어디에 적용되는가?
3. GRU가 LSTM보다 항상 성능이 낮은가?
4. causal forecasting에서 bidirectional 사용 여부를 결정하는 기준은?
5. CNN1D 입력 `[B,T,F]`에 필요한 permute는?

#### 정답·해설

1. `[64,20,32]`(단방향).
2. layer 사이 출력에 적용되고 마지막 layer 이후에는 내장 dropout이 적용되지 않는다. 1층이면 PyTorch RNN dropout은 0으로 둬야 한다.
3. 아니다. 데이터와 sequence 길이·최적화·예산에 따라 GRU가 더 좋을 수도 있다.
4. 각 output이 이용 가능한 시점보다 미래 정보를 보게 되는지 확인한다.
5. `x.permute(0,2,1)`로 `[B,F,T]`.

### 22.10 완료 기준

- [ ] LSTM의 세 gate(input·forget·output), candidate, 두 state(h·c)를 구분해 설명한다.
- [ ] CNN1D와 GRU 모델을 같은 output 계약으로 교체한다.
- [ ] padded sequence의 마지막 시점 함정을 안다.

---

## 23. Attention과 Transformer

### 23.1 학습 목표

- query, key, value의 역할과 scaled dot-product attention을 계산한다.
- self-attention output shape와 mask를 설명한다.
- positional information이 필요한 이유를 설명한다.
- PyTorch Transformer encoder를 `[B,T,F]` 입력에 연결한다.

### 23.2 attention 직관

각 query가 모든 key와 관련도를 계산하고, 그 weight로 value를 가중합한다.

`Attention(Q,K,V) = softmax(QKᵀ / sqrt(d_k)) V`.

Q `[B,H,Tq,d]`, K `[B,H,Tk,d]`이면 score `[B,H,Tq,Tk]`, V `[B,H,Tk,dv]`, output `[B,H,Tq,dv]`다.

### 23.3 왜 sqrt(d_k)로 나누는가

dimension이 커질수록 독립 성분 내적의 분산이 커져 softmax가 지나치게 포화할 수 있다. `sqrt(d_k)` scaling이 score 규모를 안정화한다.

### 23.4 self-attention과 cross-attention

- self-attention: Q/K/V가 같은 sequence 표현에서 나옴
- cross-attention: query와 key/value source가 다름, encoder-decoder 연결

Multi-head는 서로 다른 projection subspace에서 여러 관계를 병렬 학습한 뒤 concat/projection한다. head 수가 늘어도 model dimension은 보통 고정이며 head dimension은 `d_model/nhead`다.

### 23.5 위치 정보

self-attention 자체는 입력 순서를 바꾸면 같은 방식으로 출력도 바뀌는 permutation equivariant 구조라 절대·상대 순서를 모른다. sinusoidal positional encoding 또는 learnable position embedding 등을 더한다.

### 23.6 mask

- padding mask: 실제 token/시점이 아닌 padding을 key로 보지 않음
- causal mask: 현재 query가 미래 key를 보지 못함

Encoder가 과거 window 전체를 보고 하나의 미래 target을 예측한다면 window 내부 causal mask가 항상 필요한 것은 아니다. 시점별 autoregressive output이면 필요하다.

### 23.7 Transformer encoder 최소 코드

```python
class TimeSeriesTransformer(nn.Module):
    def __init__(self, n_features, out_dim, d_model=64, nhead=4,
                 num_layers=2, max_len=512, dropout=0.1):
        super().__init__()
        if d_model % nhead != 0:
            raise ValueError("d_model은 nhead로 나누어져야 합니다")
        self.input_proj = nn.Linear(n_features, d_model)
        self.pos = nn.Parameter(torch.zeros(1, max_len, d_model))
        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Linear(d_model, out_dim)

    def forward(self, x, padding_mask=None):  # x [B,T,F]
        T = x.shape[1]
        if T > self.pos.shape[1]:
            raise ValueError("sequence가 max_len보다 깁니다")
        h = self.input_proj(x) + self.pos[:, :T]
        h = self.encoder(h, src_key_padding_mask=padding_mask)
        if padding_mask is None:
            pooled = h.mean(dim=1)
        else:
            valid = (~padding_mask).unsqueeze(-1)
            pooled = (h * valid).sum(1) / valid.sum(1).clamp_min(1)
        return self.head(pooled)
```

### 23.8 계산량과 시험 선택

self-attention score는 T×T라 memory/time이 `O(T²)`다. 긴 sequence에서는 downsampling, shorter window, local attention 등이 필요하다. 제한 시간에는 CNN1D/GRU baseline 이후 문제 명세가 요구하거나 validation 개선 근거가 있을 때만 Transformer를 쓴다.

### 23.9 손계산 예제

한 query의 scaled score가 `[0, ln 2]`라면 softmax는 `[1/3, 2/3]`. value가 `[3,0]`, `[0,6]`이면 weighted output은 `[1,4]`다.

### 23.10 실습문제

1. `d_model=60, nhead=8`이 기본 multi-head에 맞지 않는 이유는?
2. Q `[B,4,10,16]`, K `[B,4,20,16]` score shape는?
3. positional encoding, 위치 의존 mask, 기타 순서 단서가 없는 self-attention은 입력 순서를 바꾸었을 때 어떻게 동작하는가? 순서 없는 평균 pooling까지 적용하면 무엇을 구별하기 어려운가?
4. PyTorch nn.TransformerEncoder의 src_key_padding_mask에서 bool True는 무엇을 뜻하는가? functional scaled_dot_product_attention의 bool attn_mask와 같은 의미인가?
5. T를 2배로 하면 attention score 원소 수는 몇 배인가?

#### 정답·해설

1. 60이 8로 나누어지지 않아 head별 동일 dimension으로 분할할 수 없다.
2. `[B,4,10,20]`.
3. 주어진 위치 단서 없는 self-attention은 입력 행의 순열에 맞춰 출력 행도 재배열되는 permutation equivariance를 갖는다. 출력 배열이 그대로 동일하다는 뜻은 아니다. 이후 순서 없는 평균 pooling까지 적용하면 같은 token 집합의 서로 다른 배열 순서를 구별하기 어렵다.
4. nn.TransformerEncoder의 bool src_key_padding_mask에서 True는 attention의 key로 무시할 위치다. 반면 functional scaled_dot_product_attention의 bool attn_mask는 True가 참여 허용을 뜻한다. API별 계약이 반대이므로 변환 없이 같은 bool mask를 재사용하면 안 된다.
5. 약 4배.

### 23.11 완료 기준

- [ ] QKᵀ와 output shape를 계산한다.
- [ ] padding/causal mask를 구분한다.
- [ ] Transformer를 baseline보다 먼저 쓰지 않을 이유를 설명한다.

---

## 24. Autoencoder와 이상탐지

### 24.1 학습 목표

- encoder, bottleneck, decoder, reconstruction loss를 설명한다.
- undercomplete/denoising/sparse/sequence AE를 구분한다.
- 정상 train과 validation threshold를 분리한다.
- sample별 reconstruction error를 올바르게 계산한다.

### 24.2 기본 Autoencoder

`z = encoder(x)`, `x_hat = decoder(z)`. 입력을 target으로 reconstruction loss를 최소화한다. bottleneck이나 규제가 없고 모델 capacity가 너무 크면 identity mapping을 학습해 useful representation을 얻지 못할 수 있다.

종류:

- Undercomplete AE: latent dimension을 입력보다 작게
- Denoising AE: 손상된 입력에서 원본을 복원
- Sparse AE: latent activation sparsity 규제
- Convolutional AE: 이미지 local 구조
- Sequence AE: sequence encoder-decoder

### 24.3 이상탐지 원리

정상 데이터만 주로 학습한 AE는 정상 manifold를 잘 복원하고 낯선 anomaly는 reconstruction error가 클 것이라는 가정이다. 항상 성립하지 않는다. capacity가 크면 anomaly도 잘 복원할 수 있고, anomaly가 정상보다 단순할 수도 있다.

[필기 핵심] error가 **임계값보다 크면** anomaly로 보는 것이 일반적이다.

### 24.4 모델

```python
class Autoencoder(nn.Module):
    def __init__(self, input_dim, latent=16, hidden=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, latent),
        )
        self.decoder = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))
```

정상 train으로 scaler를 fit하고 AE를 MSE 학습한다.

### 24.5 sample error

```python
model.eval()
errors = []
with torch.inference_mode():
    for xb in loader:
        if isinstance(xb, (tuple, list)):
            xb = xb[0]
        xb = xb.to(DEVICE, dtype=torch.float32)
        recon = model(xb)
        err = (recon - xb).pow(2).flatten(1).mean(dim=1)
        errors.append(err.cpu().numpy())
errors = np.concatenate(errors)
```

전체 batch·feature를 한 scalar로 평균하면 sample별 anomaly score가 사라진다.

### 24.6 threshold

선택 방법:

- 정상 validation error의 95/99 percentile
- labeled validation anomaly가 있으면 F1, Youden J, recall 제약, 비용함수처럼 **threshold가 들어가는 기준**
- 원하는 false positive rate
- domain cost

ROC-AUC와 PR-AUC는 threshold-independent ranking metric이므로 score/model 비교에는 쓰지만 특정 binary threshold를 직접 골라 주지는 않는다. test score 분포를 보고 threshold를 선택하면 누수다. contamination 비율을 안다고 가정하지 않는다.

### 24.7 label 방향

라이브러리나 문제마다 정상/이상 label 방향이 다르다. `anomaly=1`인지 확인한다.

```python
pred_anomaly = (errors > threshold).astype(np.int64)
```

### 24.8 AE와 PCA

선형 activation, squared loss, 적절한 제약의 undercomplete AE는 PCA와 관련된 subspace를 학습할 수 있다. 비선형 AE는 nonlinear manifold를 표현한다. 그러나 latent axis가 PCA처럼 직교·분산순이라는 보장은 없다.

### 24.9 실습문제

1. 이 강의처럼 reconstruction error가 클수록 이상이며 score > threshold일 때 anomaly=1로 정의한다. 새 sample의 error가 threshold보다 낮으면 어떤 label로 판정하는가?
2. train에 anomaly를 대량 포함하면 어떤 문제가 가능한가?
3. AE latent dimension이 input보다 반드시 작아야 하는가?
4. sample별 `[B,F]` error를 `[B]`로 만드는 코드는?
5. 고장 라벨은 없지만 정상으로 확인된 validation sample만 별도로 있다. 이 조건에서 threshold 후보를 어떻게 정하고, 어떤 성능은 직접 계산할 수 없는가?

#### 정답·해설

1. 일반 AE 가정에서는 정상.
2. anomaly pattern까지 잘 복원해 분리력이 낮아질 수 있다.
3. 아니다. denoising/sparse 등 다른 규제로 useful representation을 만들 수 있다. 다만 무규제 overcomplete는 identity 위험.
4. `(recon-x).pow(2).flatten(1).mean(1)`.
5. 정상으로 확인된 validation error의 quantile 또는 허용 false positive rate에 맞춘 threshold 후보를 정하고 민감도를 분석한다. 고장 라벨이 없으므로 고장 recall·F1은 직접 계산할 수 없으며, 정체불명의 unlabeled validation을 정상이라고 가정하지 않는다.

### 24.10 완료 기준

- [ ] 정상 train/scaler와 threshold validation을 분리한다.
- [ ] sample reconstruction error를 계산한다.
- [ ] AE가 anomaly를 항상 못 복원한다는 보장이 없음을 안다.

---

## 25. VAE와 GAN

### 25.1 학습 목표

- deterministic AE와 VAE의 latent 표현 차이를 설명한다.
- reparameterization trick과 KL 항을 구현한다.
- GAN의 generator/discriminator 목적과 교대 학습을 설명한다.
- VAE/GAN을 시험 범위 개념·최소 구현 수준으로 대비한다.

### 25.2 VAE

Encoder가 하나의 latent vector가 아니라 approximate posterior `q(z|x)`의 `μ`와 `log σ²`를 출력한다. sample:

`z = μ + σ ⊙ ε`, `ε~N(0,I)`.

무작위 sample을 그대로 graph node로 두면 parameter로 gradient가 흐르기 어렵다. noise ε를 parameter와 독립적으로 떼고 μ,σ의 deterministic transform으로 표현하는 reparameterization trick을 쓴다.

### 25.3 VAE loss와 ELBO

`loss = reconstruction loss + β KL(q(z|x) || p(z))`.

standard normal prior에 대한 diagonal Gaussian KL:

`KL = -0.5 Σ(1 + logvar - μ² - exp(logvar))`.

Reconstruction은 데이터를 설명하고, KL은 latent distribution을 prior와 가깝게 해 sampling 가능한 연속 공간을 만든다. KL을 너무 강하게 하면 posterior collapse로 decoder가 z를 무시할 수 있다.

### 25.4 최소 VAE

```python
class VAE(nn.Module):
    def __init__(self, input_dim, latent=8, hidden=64):
        super().__init__()
        self.enc = nn.Sequential(nn.Linear(input_dim, hidden), nn.ReLU())
        self.mu = nn.Linear(hidden, latent)
        self.logvar = nn.Linear(hidden, latent)
        self.dec = nn.Sequential(
            nn.Linear(latent, hidden), nn.ReLU(), nn.Linear(hidden, input_dim)
        )

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        h = self.enc(x)
        mu, logvar = self.mu(h), self.logvar(h)
        z = self.reparameterize(mu, logvar)
        return self.dec(z), mu, logvar

def vae_loss(recon, x, mu, logvar, beta=1.0):
    recon_loss = torch.nn.functional.mse_loss(recon, x, reduction="sum") / x.shape[0]
    kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / x.shape[0]
    return recon_loss + beta * kl, recon_loss, kl
```

### 25.5 GAN

Generator G는 noise z에서 fake sample을 만들고, Discriminator D는 real/fake를 구분한다. 두 모델을 적대적으로 학습한다.

원 minimax:

`min_G max_D E_real[log D(x)] + E_z[log(1-D(G(z)))]`.

실전에서는 generator gradient가 약해지는 것을 피하려 non-saturating loss `-log D(G(z))`를 흔히 쓴다.

### 25.6 최소 구조

```python
class Generator(nn.Module):
    def __init__(self, noise_dim, data_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(noise_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, data_dim),
        )

    def forward(self, z):
        return self.net(z)

class Discriminator(nn.Module):
    def __init__(self, data_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, hidden), nn.LeakyReLU(0.2),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x)  # logits, sigmoid 없음
```

`BCEWithLogitsLoss`를 쓰므로 D에 sigmoid를 넣지 않는다.

### 25.7 교대 학습 계약

1. D step: real=1, `G(z).detach()` fake=0으로 D 갱신
2. G step: 새 fake가 D에서 real=1로 판정되도록 G 갱신

G step에서 D parameter까지 optimizer step하지 않으며, D step fake를 detach하지 않으면 불필요한 G gradient graph가 생긴다.

### 25.8 GAN 실패 형태

- mode collapse: G가 일부 패턴만 생성
- oscillation/non-convergence
- discriminator가 너무 강해 G gradient 약함
- 생성 품질과 loss 숫자가 직접 일치하지 않음

따라서 실기 Problem 첫 모델로 쓰지 않고 공식 필기의 구조·loss, Process 최소 구현 대비로 학습한다.

### 25.9 비교표

| 모델 | latent | 주 목적 | 핵심 loss |
|---|---|---|---|
| AE | deterministic | reconstruction/representation | reconstruction |
| VAE | distribution μ,σ | probabilistic generation | reconstruction+KL |
| GAN | sampled noise | adversarial generation | G/D adversarial |

### 25.10 실습문제

1. `logvar=0`이면 std는?
2. μ=0, logvar=0이면 standard normal prior KL은?
3. reparameterization에서 ε를 분리하는 이유는?
4. D step에서 fake를 detach하는 이유는?
5. GAN loss가 낮으면 sample 품질이 반드시 좋은가?

#### 정답·해설

1. `exp(0.5×0)=1`.
2. 0.
3. random sampling을 μ,σ에 미분 가능한 transform으로 바꿔 encoder gradient를 전달한다.
4. D만 갱신하는 단계에서 G graph/gradient를 만들지 않기 위해서다.
5. 아니다. G/D 균형과 mode collapse 때문에 loss만으로 품질을 보장하지 못한다.

### 25.11 완료 기준

- [ ] VAE loss 두 항과 reparameterization을 설명한다.
- [ ] GAN D/G step에서 label과 detach 위치를 말한다.
- [ ] AE/VAE/GAN을 비교한다.

---

## 26. Process형 완전 공략: 명세를 edge case까지 구현하기

### 26.1 학습 목표

- 문제 문장을 함수 계약과 test case로 변환한다.
- 데이터 처리·수학·PIL·PyTorch 구조 문제를 정확히 구현한다.
- “더 좋은 코드”를 임의로 추가하지 않고 요구한 구조만 만든다.
- 8문항을 시간·확신도 순으로 운영한다.

### 26.2 Process 풀이 7단계

1. 함수/클래스/변수의 **정확한 이름** 표시
2. 입력 type·shape·dtype·범위 작성
3. 출력 type·shape·순서 작성
4. mutation, index/column 보존 여부 작성
5. 일반 입력 구현
6. 최소·경계 입력 micro-test
7. cell 전체 순서 실행 후 수동 저장

문제 지시와 다른 예외정책을 독단적으로 넣지 않는다. 예를 들어 “최댓값과 최솟값으로 정규화”만 요구했는데 quantile clip을 추가하면 정답이 바뀐다.

### 26.3 자가 edge-case test 체크리스트

| 계열 | 경계조건 |
|---|---|
| 배열/DataFrame | 빈 선택, 상수 열, NaN/Inf, 음수, 비연속 index, 원본 보존 |
| 분류 | 문자열 label, class 하나인 작은 fold, unseen category |
| 이미지 | gray/RGB/RGBA, crop 경계, batch 1, H/W 다름 |
| 모델 | 정확한 layer 순서, activation 위치, bias, output shape |
| metric | 0 denominator, hard label/score 구분, averaging |
| split | class/group 교집합, 시간 정렬, random_state |

### 26.4 수치 함수: stable softmax

큰 logit에 `exp`를 직접 쓰면 overflow가 난다. 행별 최댓값을 뺀다.

```python
def stable_softmax(x, axis=-1):
    x = np.asarray(x, dtype=np.float64)
    if x.size == 0:
        raise ValueError("softmax 입력은 비어 있을 수 없습니다")
    if not np.isfinite(x).all():
        raise ValueError("softmax 입력은 finite여야 합니다")
    shifted = x - np.max(x, axis=axis, keepdims=True)
    exp_x = np.exp(shifted)
    denom = np.sum(exp_x, axis=axis, keepdims=True)
    return exp_x / denom
```

Micro-test:

```python
p = stable_softmax(np.array([[1000.0, 1001.0], [0.0, 0.0]]), axis=1)
assert np.isfinite(p).all()
assert np.allclose(p.sum(axis=1), 1.0)
assert np.allclose(p[1], [0.5, 0.5])

try:
    stable_softmax([np.inf, 0.0])
except ValueError:
    pass
else:
    raise AssertionError("non-finite 입력을 거절해야 합니다")
```

### 26.5 선택 열 standardization

문제가 sample standard deviation `ddof=1`인지 population `ddof=0`인지 지정할 수 있다. 그대로 따른다.

```python
def standardize_selected(df, columns, ddof=0):
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            raise KeyError(c)
        mean = out[c].mean()
        std = out[c].std(ddof=ddof)
        if pd.isna(std):
            continue
        s = out[c]
        out[c] = s.where(s.isna(), 0.0) if std == 0 else (s - mean) / std
    return out
```

### 26.6 IQR flag/clip

```python
def iqr_bounds(s, factor=1.5):
    q1, q3 = s.quantile([0.25, 0.75])
    iqr = q3 - q1
    return q1 - factor * iqr, q3 + factor * iqr

def clip_iqr_column(df, column, factor=1.5):
    out = df.copy()
    lo, hi = iqr_bounds(out[column], factor)
    out[column] = out[column].clip(lo, hi)
    return out
```

요구가 “이상 행 제거”라면 clip을 하면 안 된다. flag, replace, remove는 서로 다른 명세다.

### 26.7 PIL crop·layout

```python
def pil_crop_rgb_array(image, left, upper, right, lower):
    from PIL import Image
    if not isinstance(image, Image.Image):
        raise TypeError("image는 PIL.Image.Image여야 합니다")
    if not (0 <= left < right <= image.width and 0 <= upper < lower <= image.height):
        raise ValueError("crop 좌표가 이미지 범위를 벗어났습니다")
    return np.asarray(image.convert("RGB").crop((left, upper, right, lower)))
```

문제가 gray 보존을 요구하면 `.convert('RGB')`를 넣지 않는다. 정확한 반환 channel 명세를 확인한다.

### 26.8 정확한 CNN 명세 옮기기

예시 명세(독자 문제):

| 순서 | layer | 설정 |
|---:|---|---|
| 1 | Conv2d | 3→16, K=3, S=1, P=1 |
| 2 | ReLU |  |
| 3 | MaxPool2d | K=2, S=2 |
| 4 | Conv2d | 16→32, K=3, S=1, P=0 |
| 5 | ReLU |  |
| 6 | AdaptiveAvgPool2d | 1×1 |
| 7 | Linear | 32→4 |

정답 구현은 임의 BN/Dropout/추가 Conv 없이 표 그대로다.

```python
class SpecCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(32, 4)

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x.flatten(1))

spec_model = SpecCNN()
assert spec_model(torch.zeros(2, 3, 32, 32)).shape == (2, 4)
assert spec_model(torch.zeros(1, 3, 40, 48)).shape == (1, 4)
```

### 26.9 parameter·shape quick table

| layer | output | parameter |
|---|---|---|
| Linear(I,O) | `[...,O]` | `I×O+O` |
| Conv1d(Ci,Co,K) | 길이는 공식 | `Co×Ci/groups×K + Co` |
| Conv2d(Ci,Co,Kh,Kw) | H/W 공식 | `Co×Ci/groups×Kh×Kw + Co` |
| BatchNorm(C) | shape 동일 | learnable `2C` + running stats |
| ReLU/Pool/Dropout | 규칙에 따른 shape | 0 |
| Embedding(V,D) | 입력 shape+`D` | `V×D` |
| 단방향 LSTM (`batch_first=True`) | sequence 출력 `[B,T,H]`, 여기서 H는 은닉 크기 | layer당 gate 4개 weight/bias |

### 26.10 splitter 선택·코드

| 데이터 | splitter |
|---|---|
| IID 회귀 | KFold |
| IID 분류 | StratifiedKFold |
| 새 group | GroupKFold/GroupShuffleSplit |
| 시간 미래 | TimeSeriesSplit 또는 명시적 chronological |

```python
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, TimeSeriesSplit

kf = KFold(n_splits=5, shuffle=True, random_state=42)
skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
gkf = GroupKFold(n_splits=5)

def conservative_window_gap(lookback, horizon, label_span=1):
    """연속 end-index window 행에서 train label과 valid input까지 분리하는 보수적 gap."""
    if lookback < 1 or horizon < 0 or label_span < 1:
        raise ValueError("window 계약 오류")
    return lookback + horizon + label_span - 2

# 독립적인 시계열 행 예시
tscv_rows = TimeSeriesSplit(n_splits=5)
# 이미 만든 연속 window 행 예시. 숫자 20처럼 고정하지 말고 문제 계약에서 계산한다.
tscv_windows = TimeSeriesSplit(
    n_splits=5,
    gap=conservative_window_gap(lookback=20, horizon=40, label_span=1),
)

# 사용 예: for tr, va in skf.split(X, y): ...
# group: for tr, va in gkf.split(X, y, groups): ...
# time: 정렬된 X에 for tr, va in tscv_rows.split(X): ...
```

TimeSeriesSplit에 shuffle은 없다. 데이터가 먼저 시간순이어야 한다. window 문제는 명시적인 raw/target-index split이 가장 이해하기 쉽다. boundary에서 예측 시점에 이미 관측 가능한 과거 history를 valid input으로 다시 쓰는 것은 허용될 수 있으므로 위 gap은 보수적 예시다. 핵심은 label interval·horizon·배포 시점에 맞춰 future/target leakage를 assert하는 것이다.

### 26.11 feature selection/extraction 최소 코드

```python
from sklearn.feature_selection import VarianceThreshold, SelectKBest, mutual_info_classif, SelectFromModel
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.cluster import DBSCAN

variance_selector = VarianceThreshold(threshold=0.0)
univariate_selector = SelectKBest(mutual_info_classif, k=20)
model_selector = SelectFromModel(LogisticRegression(penalty="l1", solver="liblinear"))
pca = PCA(n_components=0.95, random_state=42)
lda = LinearDiscriminantAnalysis(n_components=None)
dbscan = DBSCAN(eps=0.5, min_samples=5)
```

모두 train fold에 fit한다. LDA는 y가 필요하고 최대 dimension 상한은 `min(n_features, n_classes-1)`이다. DBSCAN의 `-1`은 noise label이며 cluster 번호는 class 의미가 아니다.

### 26.12 Process 시간 운영

각 문항을 A/B/C로 표시한다.

- A: 즉시 풀 수 있음 → 먼저
- B: 확인하면 풀 수 있음 → 두 번째
- C: 긴 디버깅/모델 → Problem 첫 baseline 후

한 문항에 15분 이상 막히면 현재 코드와 오류를 저장하고 이동한다. 쉬운 7문항을 잃고 어려운 1문항에 매달리지 않는다.

### 26.13 실습문제

1. Conv2d(3,8,kernel_size=5,padding=2,bias=True) → BatchNorm2d(8,affine=True) → ReLU의 learnable parameter 총수를 계산하라. running mean/variance는 parameter에서 제외한다.
2. `stable_softmax([1000,1000])` 결과는?
3. feature가 5개인 3-class LDA 최대 차원은?
4. TimeSeriesSplit을 쓰기 전 반드시 보장할 것은?
5. 명세에 없는 Dropout을 CNN에 추가하면 왜 위험한가?

#### 정답·해설

1. Conv `8×3×5×5+8=608`, BN learnable `16`, 합 624. running mean/var는 parameter가 아니다.
2. `[0.5,0.5]`.
3. `min(5,3-1)=2`.
4. 행이 실제 시간순으로 안정 정렬돼 있어야 한다.
5. 정확한 architecture/출력을 검사하는 hidden test와 요구 동작이 달라질 수 있다.

### 26.14 완료 기준

- [ ] 문제를 읽고 계약 5줄을 먼저 쓴다.
- [ ] 데이터/PIL/CNN micro-test를 직접 만든다.
- [ ] 네 splitter와 feature method의 fit 경계를 안다.

---

## 27. Problem형 공통 파이프라인과 표형 문제

### 27.1 학습 목표

- 10분 안에 task·metric·split·제출 계약을 확정한다.
- 표형 baseline으로 첫 유효 제출 파일을 만든다.
- PyTorch MLP를 같은 validation에서 비교한다.
- 회귀·이진·다중분류·multilabel·다중출력을 분기한다.

### 27.2 첫 셀: 문제 계약

```python
TASK = "regression"       # regression/binary/multiclass/multilabel
TARGET_COLS = ["target"]
ID_COLS = ["id"]
DROP_COLS = []
METRIC = "rmse"
EXPECTED_TEST_ROWS = None  # 읽은 뒤 채움
EXPECTED_OUTPUTS = 1

print("TASK/METRIC:", TASK, METRIC)
print("target/id/drop:", TARGET_COLS, ID_COLS, DROP_COLS)
```

문제 문장에서 다음을 직접 적는다.

- 제출은 label인가 probability인가 raw regression인가?
- output shape는 `(N,)`, `(N,1)`, `(N,K)` 중 무엇인가?
- class label 원형과 column 순서는?
- 파일명·dtype·저장 셀은?

### 27.3 10분 EDA와 split 결정

1. shape/schema/target 결측
2. row 의미와 group/time 존재
3. class/target distribution
4. 누수 후보와 ID
5. official metric

그 뒤 split을 고정한다. 모델마다 split을 바꾸면 비교할 수 없다.

### 27.4 baseline의 역할

baseline은 세 가지를 검증한다.

- 전처리와 데이터 계약이 실행되는가?
- validation metric 계산이 맞는가?
- 제출 파일이 생성되는가?

표형에서는 선형/ExtraTrees 같은 빠른 baseline이 PyTorch MLP보다 강할 수 있다. 사용자 요구대로 딥러닝 학습은 PyTorch만 사용하지만, 고전 baseline은 scikit-learn으로 빠르게 확보할 수 있다.

### 27.5 PyTorch dense bridge

희소 matrix를 MLP 입력으로 바꿀 때 메모리를 계산한다.

```python
def to_float32_dense(x, max_gib=2.0):
    budget = int(max_gib * (1024 ** 3))
    output_bytes = int(np.prod(x.shape, dtype=np.int64)) * np.dtype(np.float32).itemsize
    if hasattr(x, "toarray"):
        if output_bytes > budget:
            raise MemoryError(f"float32 dense 예상 {output_bytes / 1024**3:.2f} GiB")
        # float64 dense를 먼저 만들지 않도록 sparse 상태에서 dtype을 바꾼다.
        x = x.astype(np.float32, copy=False).toarray()
    else:
        source = np.asarray(x)
        needs_copy = source.dtype != np.float32 or not source.flags.c_contiguous
        peak_bytes = source.nbytes + output_bytes if needs_copy else source.nbytes
        if peak_bytes > budget:
            raise MemoryError(f"dense 변환 peak 예상 {peak_bytes / 1024**3:.2f} GiB")
        x = np.asarray(source, dtype=np.float32, order="C")
    if not np.isfinite(x).all():
        raise ValueError("전처리 결과에 NaN/Inf")
    return x
```

one-hot이 너무 크면 high-cardinality category 처리 방식을 바꾼다.

### 27.6 task별 label

Binary original label이 `['normal','fault']`면 positive class를 명시한다.

```python
POSITIVE_LABEL = "fault"
y_bin = (train[TARGET_COLS[0]].to_numpy() == POSITIVE_LABEL).astype(np.float32)
```

Multiclass:

```python
classes = np.sort(train[TARGET_COLS[0]].unique())
class_to_idx = {c: i for i, c in enumerate(classes)}
idx_to_class = np.asarray(classes)
y_idx = train[TARGET_COLS[0]].map(class_to_idx).to_numpy(dtype=np.int64)
```

Prediction은 `idx_to_class[pred_idx]`로 복원한다. probability 열 순서는 `classes` 순서다.

### 27.7 MLP 선택

- 회귀 output D, MSE/Huber
- binary output 1, BCEWithLogits
- multiclass output K, CE
- multilabel output K, BCEWithLogits

hidden `(128,64)`부터 시작하고 dropout 0~0.2. feature 수·데이터 크기에 맞게 축소한다. training time cap을 둔다.

### 27.8 validation→전체 재학습

Validation으로 preprocessing·모델·epoch·threshold를 정한다. 최종에는:

1. train 전체에 preprocessing 새로 fit
2. validation에서 고른 epoch 수/설정으로 model 새로 학습
3. test transform·predict
4. inverse label/target transform
5. submission 검증

전체 재학습에서 validation early stopping이 없으므로 best epoch를 사용하거나 작은 holdout을 남길 수 있다. 시간과 점수 안정성을 고려한다.

### 27.9 오류 분석

회귀:

- 큰 absolute error 행
- target 구간/group별 RMSE
- residual mean·분산

분류:

- confusion matrix
- class별 recall/F1
- 확신 높은 오답
- group/time/category별 성능

오류 분석 후 한 번에 한 변경만 한다.

### 27.10 원본 연습: 차량 효율 회귀

다음 synthetic 문제를 직접 만든다.

```python
rng = np.random.default_rng(42)
n = 6000
speed = rng.normal(70, 18, n).clip(0)
weight = rng.normal(1500, 250, n).clip(700)
temp = rng.normal(20, 12, n)
mode = rng.choice(["eco", "normal", "sport"], n, p=[0.25, 0.6, 0.15])
mode_effect = pd.Series(mode).map({"eco": -1.5, "normal": 0.0, "sport": 2.0}).to_numpy()
fuel = 3 + 0.000003 * weight * speed + 0.002 * np.maximum(temp - 25, 0) ** 2 + mode_effect
fuel += rng.normal(0, 0.6, n)
toy = pd.DataFrame({"speed": speed, "weight": weight, "temp": temp, "mode": mode, "fuel": fuel})
```

과제:

1. 80/20 split과 RMSE baseline
2. median+one-hot+scale
3. PyTorch MLP
4. group 열을 새로 합성해 group split과 random split 비교
5. 5% 결측을 넣고 안전하게 처리
6. prediction `(N,1)`과 `(N,)` 제출 계약 비교

불균형 이진분류를 데이터 생성부터 group split, PyTorch MLP, threshold, 전체 재학습, NPY reload까지 실행하는 모범 실습은 같은 폴더의 `full-mock-tabular.py`에 있다. 먼저 `--generate-only`로 train/test만 만든 뒤 혼자 풀고, 그 다음 기본 실행 결과와 비교한다.

### 27.11 실습문제

1. MLP가 tree baseline보다 낮으면 반드시 코딩 오류인가?
2. multiclass probability column 순서를 어디서 얻는가?
3. validation에서 target scaler를 inverse하지 않고 RMSE를 계산하면 무엇을 측정하는가?
4. 전체 재학습 때 validation threshold를 test에서 다시 고쳐도 되는가?
5. 첫 baseline을 빨리 저장해야 하는 이유 세 가지는?

#### 정답·해설

1. 아니다. 표형 데이터에서 tree ensemble이 더 강할 수 있다. shape/loss를 확인한 뒤 validation 결과를 따른다.
2. 학습 때 고정한 label encoder/classes 순서.
3. 표준화된 target 공간의 RMSE로 공식 단위와 다르다.
4. 안 된다. test 누수다.
5. 파이프라인·metric·제출 계약 검증, 시간 부족 보험, 이후 개선의 비교 기준.

### 27.12 완료 기준

- [ ] 네 task의 출력을 10분 안에 구성한다.
- [ ] baseline→MLP→전체 재학습→제출을 한 번 완주한다.
- [ ] 원 label·target 단위로 metric과 제출을 복원한다.

---

## 28. 시계열 Problem 완주

### 28.1 학습 목표

- 대규모 다중출력 시계열 회귀를 시간·메모리 안에 완주한다.
- naive/flatten/MLP/CNN1D/GRU 순으로 모델을 확장한다.
- target-index validation과 test 순서를 지킨다.
- `(N_test,D)` NPY를 안전하게 생성한다.

### 28.2 60분 계획

1. 0~10분: sampling, L/H, X/y/test shape, 순서 확인
2. 10~20분: 작은 subset window, naive/flatten baseline
3. 20~30분: full lazy Dataset과 첫 valid/test prediction
4. 30~50분: CNN1D 한 모델
5. 50~60분: best 파일 저장·reload

나머지 시험 시간에 GRU 또는 feature 개선을 선택한다.

### 28.3 naive baseline

다중출력 target이 input feature 일부와 같은 물리량이면 last value, moving average, 선형 trend가 강한 baseline이다. target과 input channel mapping을 명세로 확인한다.

```python
# 예: target 3축이 input의 첫 3개 feature와 동일 의미일 때만
naive_pred = X_valid[:, -1, :3]
```

의미가 다르면 사용하지 않는다. train target 평균도 최소 baseline이다.

### 28.4 flatten baseline

`[N,L,F] → [N,L×F]`로 선형/Ridge 또는 MLP. L과 F가 작으면 빠르고 강하다. input_dim은 `L×F`다.

### 28.5 CNN1D first

장점:

- 모든 시점 병렬 처리
- local 변화 pattern
- RNN보다 빠른 경우 많음
- Adaptive pooling으로 head 단순

다만 먼 시점 관계가 중요한데 receptive field가 작으면 kernel/dilation/layer를 조절한다.

### 28.6 GRU second

CNN1D validation보다 개선 가능성이 있고 시간이 남을 때 GRU를 비교한다. hidden 64, 1~2 layer부터. 긴 sequence, 큰 hidden, bidirectional은 시간·메모리를 늘린다.

### 28.7 target shape와 metric

`pred:[N,3]`, `y:[N,3]`. 전체 MSE는 모든 원소 평균인지, output별 MSE 평균인지 문제 정의를 확인한다. 두 방식은 같은 weighting이면 같지만 missing/masked target이나 output weight가 있으면 달라진다.

```python
def rmse_by_output(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError((y_true.shape, y_pred.shape))
    return np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0))
```

### 28.8 test window 대응

문제가 test를 이미 window tensor로 제공하는지, 연속 raw sequence로 제공하는지 구분한다. test sample 각 행의 순서를 임의로 shuffle/정렬하지 않는다. group별 window를 만들었다면 target 원본 순서 mapping을 반환하고 복원한다.

### 28.9 OOM 축소 순서

1. materialized window → lazy Dataset
2. batch size 절반
3. hidden/channel 절반
4. worker/prefetch 줄이기
5. float64→float32
6. validation subset으로 방향 결정
7. model 하나만 유지, 불필요 tensor 참조 해제

Gradient accumulation은 activation memory를 batch별로 줄일 수 있지만 코드 복잡성이 늘어난다. mixed precision은 환경·안정성 확인 후 사용한다.

### 28.10 원본 모의 문제

100Hz, 12 sensor, 여러 trip. 과거 30 sample로 50 sample 뒤 3 output을 예측하라. synthetic 생성:

```python
def make_sensor_series(n=20000, features=12, seed=42):
    rng = np.random.default_rng(seed)
    x = rng.normal(0, 0.15, (n, features)).astype(np.float32)
    for t in range(1, n):
        x[t] += 0.92 * x[t - 1]
    y = np.stack([
        0.7 * x[:, 0] + 0.2 * x[:, 3],
        -0.4 * x[:, 1] + x[:, 4] ** 2,
        0.5 * x[:, 2] - 0.3 * x[:, 5],
    ], axis=1).astype(np.float32)
    return x, y
```

과제:

- L=30, H=50 정확한 window 수 손계산
- 마지막 20% target-index validation
- train-only sequence scaler
- flatten MLP와 CNN1D 비교
- output별 RMSE
- 마지막 1,000개 test window의 `(1000,3)` NPY 저장·reload

### 28.11 실습문제

1. test row 순서를 정렬해 학습했는데 복원하지 않으면 무엇이 틀리는가?
2. Conv1d kernel 3 두 층을 연속 적용하고 두 층 모두 stride=1, dilation=1이며 pooling은 없다. padding 경계를 제외한 두 번째 층의 한 출력 위치가 참조하는 이론적 receptive field는 몇 시점인가?
3. 유효한 CNN 제출은 이미 보존했고 GRU의 계산 시간이 cap을 넘는다. OOM은 아니다. 시간 예산을 줄일 조정 두 가지와, batch size 감소가 반드시 학습 시간을 줄이지는 않는 이유를 설명하라.
4. validation target scaler를 train+valid에 fit하면 왜 누수인가?
5. naive baseline이 CNN보다 좋으면 어떤 점을 점검할까?

#### 정답·해설

1. 각 prediction과 제출 대상 sample 대응이 어긋나 성능이 무너진다.
2. 5시점.
3. 이미 유효한 CNN 제출을 보존하고 GRU의 hidden 크기·층 수·epoch 수를 줄이는 조정을 비교한다. OOM이 아닌 계산 시간 문제에서는 batch 감소가 배치 수를 늘려 오히려 느려질 수 있어 실행 시간을 측정한다. lookback은 문제 계약을 바꾸지 않는 범위에서만 조정한다.
4. validation target 분포 통계가 학습 target 변환에 들어간다.
5. window/horizon off-by-one, target-channel 의미, scaling/inverse, overfit, sequence 모델 필요성을 점검한다.

### 28.12 완료 기준

- [ ] synthetic 문제를 90분 안에 완주한다.
- [ ] naive와 CNN1D를 반드시 비교한다.
- [ ] NPY shape·dtype·NaN·순서를 reload 검사한다.

---

## 29. 이미지·이상탐지 Problem 분기

### 29.1 학습 목표

- 문제 설명에서 CNN 분류/회귀와 AE anomaly를 구분한다.
- path-label 정렬과 image layout을 검증한다.
- 작은 CNN과 transfer learning 중 하나를 시간 안에 선택한다.
- anomaly threshold를 validation에서 정한다.

### 29.2 2분 분기표

| 주어진 것 | 첫 모델 |
|---|---|
| image+class label | 작은 CNN, 여유 시 ResNet |
| image+연속 좌표/값 | CNN regression head |
| 정상 image 위주, anomaly label 없음 | convolutional AE/간단 feature anomaly |
| tabular sensor 정상 data | MLP AE |
| labeled anomaly 충분 | supervised classifier baseline도 비교 |

### 29.3 path-label 계약

파일 경로를 독립적으로 sort하고 label 행을 그대로 쓰면 어긋날 수 있다. 하나의 DataFrame 행에서 path와 label을 함께 가져온다.

```python
class ImagePathDataset(Dataset):
    def __init__(self, frame, path_col, label_col=None, target_cols=None,
                 task="classification", transform=None, label_to_idx=None):
        self.frame = frame.reset_index(drop=True).copy()
        self.path_col = path_col
        self.label_col = label_col
        self.target_cols = None if target_cols is None else list(target_cols)
        self.task = task
        self.transform = transform
        if task not in {"classification", "regression", "inference"}:
            raise ValueError("task는 classification/regression/inference")
        if task == "classification" and label_col is None:
            raise ValueError("classification에는 label_col이 필요합니다")
        if task == "regression" and not self.target_cols:
            raise ValueError("regression에는 target_cols가 필요합니다")
        self.label_to_idx = None
        if task == "classification":
            labels = self.frame[label_col]
            if labels.isna().any():
                raise ValueError("classification label에 결측치가 있습니다")
            if label_to_idx is None:
                # 문자열/숫자 label 모두 0..K-1로 만든다. 순서는 train 첫 등장 기준이다.
                unique_labels = list(dict.fromkeys(labels.tolist()))
                label_to_idx = {label: i for i, label in enumerate(unique_labels)}
            self.label_to_idx = dict(label_to_idx)
            if set(self.label_to_idx.values()) != set(range(len(self.label_to_idx))):
                raise ValueError("label_to_idx 값은 연속된 0..K-1이어야 합니다")
            unknown = [label for label in labels.unique() if label not in self.label_to_idx]
            if unknown:
                raise ValueError(f"mapping에 없는 label: {unknown}")

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        from PIL import Image
        row = self.frame.iloc[idx]
        with Image.open(row[self.path_col]) as image:
            image = image.convert("RGB")
            if self.transform is not None:
                x = self.transform(image)
            else:
                array = np.array(image, dtype=np.float32, copy=True) / 255.0
                x = torch.from_numpy(array).permute(2, 0, 1)
        if self.task == "classification":
            return x, self.label_to_idx[row[self.label_col]]
        if self.task == "inference":
            return x
        target = row[self.target_cols].to_numpy(dtype=np.float32)
        return x, torch.from_numpy(target)
```

문자열 label도 위 mapping으로 처리한다. train Dataset에서 만든 `train_ds.label_to_idx`를 validation Dataset의 `label_to_idx`에도 전달해 class index를 동일하게 고정한다. test에 label이 없으면 `task="inference"`를 쓴다.

### 29.4 train/valid transform

Train에는 label-preserving random augmentation, valid/test에는 resize/normalize만. 모든 image를 같은 H/W로 batch한다.

전이학습은 pretrained weight transform을 우선한다. scratch는 `[0,1]` 후 데이터 기준 normalization을 검토한다.

### 29.5 이미지 회귀

head output D, MSE/Huber, target float. 좌표를 0~1로 normalize했다면 sigmoid head가 가능하지만, 원 target 범위 밖 extrapolation이 필요한 문제에는 제한이 된다. 제출 전에 pixel/원 단위로 inverse한다.

### 29.6 convolutional AE

Encoder downsample, decoder upsample로 입력 image를 복원한다. transpose convolution은 checkerboard artifact가 가능해 upsample+conv도 선택지다. anomaly score는 channel·H·W 전체 sample별 평균.

### 29.7 threshold와 validation

- train: 정상만
- valid normal: false positive 추정
- valid anomaly label이 있으면 AUC로 score ranking/model을 비교하고, F1·recall 제약·비용 기준으로 threshold 선택
- test: threshold 고정

이미지 augmentation을 anomaly validation에 random 적용하면 score 분산이 생기므로 deterministic transform을 쓴다.

### 29.8 원본 모의문제 설계

NumPy/PIL로 배경에 세로·가로·대각선 패턴과 noise를 넣은 synthetic image를 생성한다. 파일이 필요하면 학습 전에 생성하되 시험에서는 제공 객체를 사용한다.

Task A: 세 방향 pattern 3-class 분류  
Task B: 세로 pattern만 정상으로 AE 학습, 가로/대각선 anomaly 탐지

같은 폴더의 `full_mock_image_ae.py`는 외부 다운로드 없이 합성 이미지를 만들고, 작은 PyTorch CNN 분류와 convolutional AE 정상-only 학습, validation threshold, 두 NPY 제출·reload까지 실행한다. 빠른 검산은 `--samples-per-class 60 --epochs 2`, 제대로 된 학습은 기본값으로 실행한다.

채점:

- path/label alignment
- NCHW/float range
- train-only augmentation
- class/AE output shape
- validation metric/threshold
- row order와 제출

### 29.9 실습문제

1. validation에 random crop을 매 epoch 적용하면 metric에 어떤 영향이 있는가?
2. RGBA image를 `Conv2d(3,...)`에 넣을 때 필요한 처리 예는?
3. class 3개 CE 모델 output과 target shape/dtype은?
4. AE batch 전체 scalar loss를 anomaly score로 제출하면 왜 틀리는가?
5. labeled anomaly가 충분할 때 AE만 고집할 이유가 있는가?

#### 정답·해설

1. 평가 입력이 바뀌어 score가 불안정하고 공정한 모델 비교가 어렵다.
2. `convert('RGB')`로 alpha 제거 또는 명세에 맞는 4-channel 모델.
3. logits `[B,3]`, target `[B]` long.
4. sample별 순위·label을 만들 수 없고 모두 같은 score가 된다.
5. 없다. supervised classifier baseline과 validation 비교한다.

### 29.10 완료 기준

- [ ] Dataset에서 path-label 대응을 보장한다.
- [ ] CNN/AE task를 2분 안에 분기한다.
- [ ] threshold를 test 전에 고정한다.

---

## 30. 제출·디버깅·170분 운영

### 30.1 학습 목표

- 제공 skeleton과 저장 cell을 보존한다.
- prediction의 shape, dtype, finite, 순서를 검사한다.
- NPY/CSV를 다시 읽어 검증한다.
- OOM, NaN, shape 오류를 우선순위대로 해결한다.

### 30.2 절대 규칙

1. 제공 변수명·파일명·저장 cell을 임의로 바꾸지 않는다.
2. 코드 수정 후 `Ctrl+S`로 저장하고 `Autosaved`를 확인한다. 기존 저장 셀을 실행해 예측 파일을 검사한 뒤 다시 저장한다.
3. Process와 Problem을 각각 제출하고 정상 제출 팝업·로그를 확인한다. 파일 생성·노트북 저장·문항 제출은 별개다. Autosave Failed나 연결 오류는 감독관에게 알린다. 테스트 종료 후에는 수정할 수 없다.
4. 제출 파일을 먼저 만들고 성능을 개선한다.
5. test row order를 보존한다.
6. 전체 helper를 통째 import하거나 완성 pipeline을 무수정 복사하지 않고 필요한 블록만 사용한다.
7. 코드 실행과 모델 학습 시간도 170분에 포함되며, 실행 지연만으로 시험시간이 연장되지 않는다는 공식 운영 조건을 전제로 time cap을 둔다.

### 30.3 prediction 계약 검사

```python
def validate_prediction(pred, expected_shape, name="prediction"):
    arr = np.asarray(pred)
    if arr.shape != tuple(expected_shape):
        raise ValueError(f"{name} shape {arr.shape} != {tuple(expected_shape)}")
    if not np.issubdtype(arr.dtype, np.number):
        raise TypeError(f"{name} numeric dtype 필요: {arr.dtype}")
    if not np.isfinite(arr).all():
        bad = np.argwhere(~np.isfinite(arr))[:10]
        raise ValueError(f"{name} NaN/Inf 위치 예: {bad.tolist()}")
    return arr
```

문제에서 dtype을 지정했을 때만 변환한다. float32 변환 후 overflow가 생길 수 있어 다시 검사한다.

### 30.4 NPY

아래는 개인 연습에서 저장 원리를 확인하는 코드다. 실제 답안의 기존 저장 셀을 교체하지 말고, 그 셀이 참조하는 변수에 검증된 예측을 연결한다. 변수·파일명·대소문자·경로·Notebook 이름은 당일 스켈레톤 그대로 둔다.

```python
pred = validate_prediction(pred, (len(test), OUT_DIM))
path = "Submission_problem.npy"
np.save(path, pred, allow_pickle=False)

loaded = np.load(path, allow_pickle=False)
validate_prediction(loaded, pred.shape, name="reloaded")
if loaded.dtype != pred.dtype or not np.array_equal(loaded, pred):
    raise IOError("NPY reload 불일치")
print(path, loaded.shape, loaded.dtype)
```

`(N,)`이 요구되는데 `(N,1)`로 저장하거나 그 반대는 채점 실패가 될 수 있다. 공식 shape를 그대로 쓴다.

### 30.5 CSV

sample submission이 있으면 복사하고 target 열만 채운다.

```python
submission = sample_submission.copy()
if not submission.columns.is_unique:
    raise ValueError("submission 중복 열")
if not set(PRED_COLS).issubset(submission.columns):
    raise KeyError(f"PRED_COLS 오타: {PRED_COLS}")
if len(submission) != len(pred):
    raise ValueError("row 수 불일치")

arr = np.asarray(pred)
if len(PRED_COLS) == 1:
    if arr.shape not in {(len(submission),), (len(submission), 1)}:
        raise ValueError(f"단일 target shape 오류: {arr.shape}")
    submission[PRED_COLS[0]] = arr.reshape(-1)
else:
    if arr.shape != (len(submission), len(PRED_COLS)):
        raise ValueError((arr.shape, len(PRED_COLS)))
    submission.loc[:, PRED_COLS] = arr

numeric_before = submission[PRED_COLS].select_dtypes(include=["number"])
if numeric_before.shape[1] and not np.isfinite(numeric_before.to_numpy()).all():
    raise ValueError("CSV target NaN/Inf")

submission.to_csv("submission.csv", index=False)
loaded = pd.read_csv("submission.csv")
if list(loaded.columns) != list(submission.columns) or len(loaded) != len(submission):
    raise IOError("CSV schema/reload 불일치")
if loaded[PRED_COLS].isna().any().any():
    raise IOError("CSV target NaN")
numeric_after = loaded[PRED_COLS].select_dtypes(include=["number"])
if numeric_after.shape[1] and not np.isfinite(numeric_after.to_numpy()).all():
    raise IOError("CSV reload target Inf")
```

### 30.6 오류 진단 순서

`mat1 and mat2 shapes cannot be multiplied`:

1. model 직전 input shape
2. flatten/permute 축
3. Linear input_dim

`Expected target Long`:

- CE target을 `.long()`, shape `[B]`

`Target size must be same as input`:

- BCE/MSE prediction-target shape를 정확히 맞춤

`CUDA out of memory`:

- batch↓, lazy Dataset, model↓, 불필요 tensor 참조, CPU baseline

`loss NaN`:

- 입력/target finite, LR↓, log domain, scaling, gradient 확인

`AUC only one class`:

- split class 확인, stratify/group seed 조정. metric을 임의 변경하지 않음

### 30.7 시간 부족 축소 순서

1. 탐색 중지
2. best checkpoint/첫 제출 보존
3. epoch/patience 축소
4. 모델 하나만 선택
5. batch·hidden 축소
6. CV→holdout
7. augmentation/복잡 feature 제거
8. 제출 검증 시간 7분은 지킨다

### 30.8 최종 30초 체크

- [ ] Process 제출 버튼
- [ ] Problem 제출 버튼
- [ ] `Submission_problem.npy` 존재
- [ ] shape/dtype/NaN/Inf/reload
- [ ] test 순서와 class mapping
- [ ] 제공 저장 cell 마지막 실행
- [ ] Ctrl+S
- [ ] 오류·autosave/kernel 문제는 감독관에게 즉시 알림

### 30.9 실습문제

1. prediction `(200,2)`를 단일 target CSV 400행에 flatten해 넣으면 왜 위험한가?
2. `np.isfinite`를 float32 변환 전후 모두 보는 이유는?
3. CSV 문자열 빈 label이 reload 후 무엇이 될 수 있는가?
4. 성능 개선보다 마지막 7분 제출 검증이 중요한 이유는?
5. kernel 재시작 전에 해야 할 일은?

#### 정답·해설

1. 우연히 원소 수가 같아도 sample-target 구조를 파괴한 잘못된 제출이다. 허용 shape를 먼저 제한한다.
2. 매우 큰 float64가 float32에서 Inf로 overflow될 수 있다.
3. pandas가 NaN으로 읽을 수 있으므로 모든 target 열 결측을 검사한다.
4. 파일 계약 실패는 모델 성능과 무관하게 채점 불가/큰 감점으로 이어질 수 있다.
5. Ctrl+S로 코드 저장하고 공식 절차·감독관 안내를 따른다.

### 30.10 완료 기준

- [ ] NPY/CSV를 저장 후 reload한다.
- [ ] 여섯 오류 유형을 2분 안에 진단한다.
- [ ] 마지막 7분을 반드시 확보한다.

---

## 31. 필기 모의고사 1회 — 20문항 50분

### 31.1 응시 방법

- 타이머 50분을 켠다.
- 교재·검색 없이 푼다.
- 모르는 문제는 90초 뒤 표시하고 넘어간다.
- 1차 35분, 보류 10분, 제출 확인 5분으로 운영한다.
- 답을 바꾸기 전 “처음 답이 틀렸다는 구체적 근거”를 적는다.

아래 문항은 공식 문제와 무관한 독자 제작 문제다.

### 31.2 문제

#### 1번. Broadcasting

`x`의 shape가 `(16, 20, 8)`일 때 오류 없이 feature별 bias를 더하기 가장 적절한 bias shape는?

① `(16,)` ② `(20,)` ③ `(8,)` ④ `(16,20)`

#### 2번. 전처리 누수

다음 중 validation 누수를 일으키지 않는 것은?

① split 전 전체 데이터로 StandardScaler fit  
② split 전 전체 데이터로 PCA fit  
③ split 후 train fold로 median을 구해 valid에 적용  
④ split 전 전체 데이터에 SMOTE 후 나누기

#### 3번. 정보이론

이진 node의 class 비율이 0.5/0.5에서 1.0/0.0으로 바뀔 때 entropy는?

① 0에서 최대로 증가 ② 최대에서 0으로 감소 ③ 변하지 않음 ④ 음수가 됨

#### 4번. Validation

800대 차량에서 차량당 여러 sensor 행이 있고 test에는 처음 보는 차량만 등장한다. 가장 적절한 validation은?

① 행 random split ② stratified row split ③ vehicle group split ④ 전체 train으로 학습 후 train score

#### 5번. 불균형 metric

고장 class가 1%인 이진분류에서 소수 class 성능을 숨기지 않기 위한 첫 metric으로 가장 적절한 것은?

① accuracy ② Macro-F1 ③ MSE ④ R²

#### 6번. Loss 계약

4-class 단일정답 분류에서 `CrossEntropyLoss`의 올바른 입력은?

① sigmoid 확률 `(B,1)`, float target `(B,1)`  
② softmax 확률 `(B,4)`, one-hot target `(B,4)`만 가능  
③ raw logits `(B,4)`, long class index `(B,)`  
④ raw logits `(B,)`, long target `(B,4)`

#### 7번. 역전파

`pred=wx`, `L=(pred-y)²`, `x=2,w=1,y=5`일 때 `dL/dw`는?

① -12 ② -6 ③ 6 ④ 12

#### 8번. Optimizer

Momentum의 핵심 설명으로 가장 적절한 것은?

① gradient를 항상 0으로 만든다  
② 과거 update 방향을 누적해 일관된 방향은 가속하고 진동을 줄인다  
③ 모든 parameter를 같은 상수로 바꾼다  
④ validation loss를 직접 미분한다

#### 9번. Batch Normalization

track_running_stats=True인 BatchNorm을 model.eval()로 평가할 때 옳은 것은?

① 현재 sample 하나의 통계만 사용  
② 학습 중 저장한 running mean/variance 사용  
③ 모든 activation을 0으로 만듦  
④ dropout 확률을 자동 선택

#### 10번. CNN 출력 크기

한 공간축 input 64, kernel 5, padding 2, stride 2, dilation 1인 convolution의 output 크기는?

① 30 ② 31 ③ 32 ④ 64

#### 11번. CNN parameter

`Conv2d(3, 10, kernel_size=3, bias=True)`의 parameter 수는?

① 270 ② 280 ③ 300 ④ input H/W에 따라 다름

#### 12번. LSTM

LSTM forget gate의 주 역할은?

① class 확률 정규화 ② 이전 cell state 유지 정도 조절 ③ image downsampling ④ learning rate 감소

#### 13번. RNN shape

`GRU(input_size=6, hidden_size=32, batch_first=True)`에 `(B=8,T=15,F=6)`을 넣은 단방향 1층 sequence output shape는?

① `(8,32)` ② `(15,8,32)` ③ `(8,15,32)` ④ `(8,15,6)`

#### 14번. Attention

Scaled dot-product attention에서 `sqrt(d_k)`로 나누는 주된 이유는?

① sequence 길이를 줄이기 위해  
② score 규모가 dimension과 함께 커져 softmax가 포화하는 것을 완화  
③ value를 one-hot으로 만들기 위해  
④ positional encoding을 제거하기 위해

#### 15번. Transformer

기본 self-attention에 positional information이 필요한 이유는?

① attention은 channel을 처리할 수 없어서  
② attention만으로는 입력 순서를 명시적으로 알지 못해서  
③ softmax 출력 합이 1이 아니어서  
④ gradient를 계산할 수 없어서

#### 16번. ResNet

Residual addition 전에 main branch와 shortcut의 shape가 다를 때 일반적인 해결은?

① target을 one-hot 변환 ② 1×1 projection과 필요한 stride 사용 ③ sigmoid 추가 ④ loss를 MAE로 변경

#### 17번. 전이학습

작은 새 이미지 데이터에서 pretrained model을 쓰는 일반적인 첫 단계는?

① 모든 weight 무작위 초기화  
② classifier head를 교체하고 backbone을 freeze해 head부터 학습  
③ validation에도 random crop을 강제  
④ target label을 input에 추가

#### 18번. Autoencoder 이상탐지

정상 데이터로 학습한 AE의 일반적인 anomaly 판정은?

① reconstruction error가 작을수록 anomaly  
② reconstruction error가 validation threshold보다 클수록 anomaly  
③ latent dimension이 크면 모두 anomaly  
④ train과 test error 평균이 같으면 anomaly

#### 19번. VAE

VAE의 KL term 역할로 가장 적절한 것은?

① output을 hard label로 변환  
② approximate posterior를 prior와 가깝게 규제  
③ image 크기를 반으로 줄임  
④ discriminator를 학습

#### 20번. GAN

`BCEWithLogitsLoss`를 사용하는 Discriminator의 마지막 layer로 가장 적절한 것은?

① Linear로 logit 1개, sigmoid 없음  
② Linear 뒤 softmax class 10개  
③ ReLU만 사용, 출력 없음  
④ sigmoid를 두 번 적용

### 31.3 정답표

| 문항 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 정답 | ③ | ③ | ② | ③ | ② | ③ | ① | ② | ② | ③ |

| 문항 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 정답 | ② | ② | ③ | ② | ② | ② | ② | ② | ② | ① |

### 31.4 상세 해설

1. Broadcasting은 뒤축부터 맞춘다. feature 마지막 축 8에 bias `(8,)`가 맞는다.
2. train fold 통계만 valid에 적용하는 ③이 올바르다. 나머지는 validation 정보를 전처리 fit에 사용한다.
3. 50:50은 불확실성 최대, 순수 node는 entropy 0.
4. test가 새 차량이므로 vehicle-level group split이 배포 조건을 모사한다.
5. Macro-F1은 두 class F1을 동일 가중한다. accuracy는 전부 정상 예측도 99%.
6. CE는 raw logits와 class index long target을 안정적으로 처리한다.
7. pred=2, error=-3. `2×error×x=2×(-3)×2=-12`.
8. Momentum velocity가 과거 gradient 방향을 누적한다.
9. eval mode BN은 training에서 누적한 running statistics를 쓴다.
10. `floor((64+4-4-1)/2+1)=32`.
11. `10×3×3×3+10=280`.
12. forget gate가 `c_{t-1}`의 유지 비율을 조절한다.
13. batch_first sequence output은 `[B,T,H]=[8,15,32]`.
14. 내적 분산이 커져 softmax가 거의 one-hot으로 포화하는 것을 완화한다.
15. 순서 표현을 별도로 넣어야 위치를 구분한다.
16. 1×1 convolution은 channel과 공간 stride를 맞춘다.
17. 새 head부터 학습하면 작은 데이터에서 안정적이고 빠르다. 이후 일부 unfreeze.
18. 일반 가정은 정상 reconstruction이 좋고 anomaly error가 크다는 것.
19. KL이 latent posterior를 prior 방향으로 regularize해 sampling 가능한 공간을 만든다.
20. loss가 sigmoid까지 포함하므로 D는 raw logit 하나를 반환한다.

### 31.5 진단

- 18~20개: 필기 기본 완성. 시간 단축과 함정 반복
- 15~17개: 취약 영역 2개를 찾아 해당 강의 재학습
- 12~14개: 4~6, 13~25강 핵심표를 다시 손으로 정리
- 11개 이하: 모의 반복보다 이론 순서 학습을 먼저

오답은 `몰랐음 / 계산실수 / 문제오독 / 시간부족` 네 종류로 분류한다. 같은 실수를 방지할 행동까지 적는다.

### 31.6 필기 모의고사 2회 — 고전 ML·데이터·검증 중심

새 타이머 50분으로 응시한다. 1회 답을 외운 점수와 섞지 않는다.

#### 2회-1번

명시적 target 없이 데이터 구조를 찾는 학습은? ① 지도 ② 비지도 ③ 강화 ④ 회귀만

#### 2회-2번

validation fold의 중앙값을 imputation에 사용하지 않게 하는 가장 안전한 구조는? ① 전체 fit 후 split ② fold 내부 Pipeline ③ test 평균 사용 ④ 결측 행 전부 무조건 삭제

#### 2회-3번

일반적인 결정트리에 feature standardization이 필수가 아닌 이유는? ① gradient가 없어서가 아니라 threshold 순서 분할이 단조 scale에 거의 불변 ② 결측이 자동 제거 ③ 모든 tree가 선형 ④ target을 사용하지 않음

#### 2회-4번

Ridge의 L2 규제 강도 λ를 매우 크게 하면 일반적으로? 여기서 bias는 절편 parameter가 아니라 추정 편향을 뜻한다. ① weight 크기 감소·bias 증가 ② weight 폭발 ③ L1 sparsity만 발생 ④ test label 사용

#### 2회-5번

이진 logistic logit이 `ln(3)`이면 odds와 확률은? ① odds 3, p=0.75 ② odds 0.75, p=3 ③ odds 1, p=.5 ④ odds 3, p=.25

#### 2회-6번

KNN의 k를 1에서 매우 크게 늘릴 때 일반적 변화는? ① variance↑ bias↓ ② variance↓ bias↑ ③ 둘 다 반드시 0 ④ scale 영향 소멸

#### 2회-7번

RBF SVM에서 C와 gamma가 모두 지나치게 크면? ① 매우 단순한 경계만 가능 ② 복잡하고 국소적인 경계로 overfit 위험 ③ 선형회귀가 됨 ④ PCA가 자동 실행

#### 2회-8번

class 비율 0.8/0.2인 node의 Gini는? ① .16 ② .32 ③ .50 ④ .64

#### 2회-9번

Random Forest의 핵심은? ① boosting만 사용 ② bootstrap과 feature 무작위성으로 덜 상관된 tree를 평균 ③ 하나의 매우 깊은 tree ④ 거리 기반 voting

#### 2회-10번

Gradient boosting 설명으로 옳은 것은? ① 모든 tree 독립 병렬만 ② 앞 모델 residual/negative gradient를 순차 보완 ③ label 없이 centroid 갱신 ④ margin만 최대화

#### 2회-11번

L1 coefficient가 0이 된 feature를 제거하는 방식은? ① wrapper만 ② embedded selection ③ PCA extraction ④ clustering

#### 2회-12번

PCA에 대한 옳은 설명은? ① target class 분리만 최대화 ② X 분산이 큰 직교 방향을 찾는 비지도 extraction ③ 원 feature 일부만 선택 ④ scale과 무관

#### 2회-13번

입력 feature가 8개인 5-class LDA가 만들 수 있는 discriminant 축의 최대 수는? ① 2 ② 4 ③ 5 ④ feature 수와 무관하게 10

#### 2회-14번

K-means의 centroid 갱신은? ① cluster 내 점의 평균 ② 가장 먼 점 ③ target 평균 ④ support vector

#### 2회-15번

DBSCAN의 장점은? ① cluster 수 k가 반드시 필요 ② 임의 형태 cluster와 noise 식별 가능 ③ 고차원에서 항상 완벽 ④ scale이 전혀 무관

#### 2회-16번

불균형 single-label classification에서 fold별 class 비율을 유지하려면? ① KFold ② StratifiedKFold ③ GroupKFold만 ④ PCA

#### 2회-17번

Target encoding의 가장 큰 위험은? ① feature 수 감소 ② 자기 행/validation target 통계가 들어가는 누수 ③ 숫자가 됨 ④ category가 존재함

#### 2회-18번

큰 오차에 제곱 penalty를 주면서 target 원 단위로 해석되는 metric은? ① RMSE ② MAE ③ accuracy ④ entropy

#### 2회-19번

탐색 차원이 많고 평가 횟수가 제한될 때 Grid보다 자주 효율적인 것은? ① RandomizedSearch ② test 수동 선택 ③ 모든 조합 무한 반복 ④ validation 삭제

#### 2회-20번

SMOTE의 올바른 적용 위치는? ① split 전 전체 데이터 ② 각 train fold 내부 ③ validation+test ④ 제출 후

#### 2회 정답·핵심 해설

| 문항 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 정답 | ② | ② | ① | ① | ① | ② | ② | ② | ② | ② | ② | ② | ② | ① | ② | ② | ② | ① | ① | ② |

1. 비지도는 target 없이 구조를 찾는다. 2. Pipeline이 fold별 fit을 보장한다. 3. tree는 단조변환 후 순서를 유지한다. 4. L2가 weight를 수축시켜 underfit할 수 있다. 5. odds=`e^logit=3`, `p=3/(1+3)`. 6. k 증가는 smoothing이다. 7. 큰 C/gamma는 복잡도를 높인다. 8. `1-(.64+.04)=.32`. 9. RF는 bagging 계열이다. 10. boosting은 순차 보완이다. 11. 학습 모델 내부의 L1 선택이다. 12. PCA는 target 비사용. 13. `min(F,C-1)=min(8,4)=4`. 14. 할당된 점 평균. 15. density 연결과 noise가 장점. 16. stratification. 17. out-of-fold가 필요한 이유다. 18. RMSE. 19. Random search가 중요한 차원을 넓게 탐색한다. 20. resampling은 train fold에만 적용한다.

### 31.7 필기 모의고사 3회 — 계산·전 범위 혼합

이 회차는 계산 과정을 시험지 여백에 반드시 적는다.

#### 3회-1번

모집단 `[0,2,4]`의 평균과 분산은? ① 2, 8/3 ② 2, 4 ③ 3, 8/3 ④ 2, sqrt(8/3)

#### 3회-2번

고장률 10%, 고장일 때 경보 80%, 정상일 때 오경보 20%라면 경보가 울렸을 때 실제 고장 확률은? ① 8% ② 20% ③ 약 30.8% ④ 80%

#### 3회-3번

`Linear(12,5)`의 parameter와 batch 7 output은? ① 60개, `(7,5)` ② 65개, `(7,5)` ③ 65개, `(5,7)` ④ 17개, `(7,12)`

#### 3회-4번

활성함수 그래프가 원점을 지나고 홀함수이며, 출력 범위가 `(-1,1)`이고 양 끝에서 포화한다. 이 함수는? ① sigmoid ② tanh ③ ReLU ④ softmax

#### 3회-5번

실제 `[1,3]`, 예측 `[2,5]`의 MSE/RMSE는? ① 1.5/sqrt1.5 ② 2.5/sqrt2.5 ③ 5/2 ④ 3/1.5

#### 3회-6번

TP=40, FP=10, FN=40일 때 precision/recall/F1은? ① .8/.5 약 .615 ② .5/.8 약 .615 ③ .8/.8 .8 ④ .4/.5 .444

#### 3회-7번

이 모의의 RMSLE는 target과 prediction이 모두 0 이상이어야 한다고 정의한다. 이 입력 규약에 맞지 않는 값은? ① 0 ② 양수 ③ 음수 ④ 1

#### 3회-8번

이미지 분류의 data augmentation 적용으로 가장 올바른 것은? ① train/valid/test 모두 random crop·flip ② train에만 label-preserving random 변환, valid/test에는 deterministic resize·normalize ③ valid에만 강한 random 변환 ④ test label을 보고 변환 선택

#### 3회-9번

`MaxPool2d(2,2)`가 `(B,16,32,40)`에 적용된 output은? ① `(B,16,16,20)` ② `(B,8,16,20)` ③ `(B,16,32,40)` ④ `(B,32,16,20)`

#### 3회-10번

`BatchNorm1d(20)`의 learnable γ,β parameter 수는? ① 20 ② 40 ③ 60 ④ batch size에 따라 변함

#### 3회-11번

LSTM cell update로 옳은 것은? ① `c_t=f_t⊙c_{t-1}+i_t⊙g_t` ② `c_t=softmax(x)` ③ `c_t=QKᵀ` ④ `c_t=maxpool(x)`

#### 3회-12번

길이 T=500인 유효 시계열에서 input은 X[e-19:e+1], 단일 target은 y[e+30]이다. group 경계·결측이 없고 stride=1일 때 만들 수 있는 window 수는? ① 450 ② 451 ③ 470 ④ 500

#### 3회-13번

target index가 cut 이상이면 validation에 배정하는 규칙에서 cut=400이고 window target index가 400, input 마지막 index가 370일 때 time validation 배정은? ① train ② valid ③ 폐기만 가능 ④ random

#### 3회-14번

한 query score가 `[0, ln3]`이면 softmax weight는? ① `[.5,.5]` ② `[.25,.75]` ③ `[.75,.25]` ④ `[0,1]`

#### 3회-15번

VAE에서 `mu=0, logvar=0`인 1차원 posterior와 N(0,1) prior의 KL은? ① 0 ② .5 ③ 1 ④ 무한대

#### 3회-16번

AE validation 정상 error의 99 percentile을 threshold로 썼다. score가 threshold보다 큰 sample의 일반적 label은? ① 정상 ② anomaly ③ class 99 ④ 알 수 없어 항상 삭제

#### 3회-17번

GAN discriminator step에서 `G(z).detach()`의 목적은? ① G gradient graph 차단 ② D gradient 차단 ③ sigmoid 두 번 ④ noise 제거

#### 3회-18번

Residual main output `(B,64,16,16)`, input `(B,32,32,32)`일 때 shortcut은? ① identity ② 1×1, stride2, 32→64 ③ maxpool만 ④ softmax

#### 3회-19번

multilabel 5개 target의 올바른 계약은? ① logits `(B,5)`, float target `(B,5)`, BCEWithLogits ② logits `(B,5)`, long target `(B,)`, CE ③ logits `(B,1)`, target `(B,5)` ④ argmax만

#### 3회-20번

문제가 정확히 `(8333,3)` NPY를 요구할 때 `(8333,)`이 나온 모델의 올바른 대응은? ① 파일명만 변경 ② 3개 output을 내도록 모델/예측 계약 수정 ③ 같은 배열 세 번 저장하지 않고 무조건 제출 ④ flatten

#### 3회 정답·핵심 해설

| 문항 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 | 15 | 16 | 17 | 18 | 19 | 20 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 정답 | ① | ③ | ② | ② | ② | ① | ③ | ② | ① | ② | ① | ② | ② | ② | ① | ② | ① | ② | ① | ② |

1. 편차 제곱합 8을 3으로 나눈다. 2. `0.08/(0.08+0.18)=.3077`. 3. `12×5+5=65`. 4. tanh는 원점 대칭 S자이고 범위가 `(-1,1)`이다. 5. squared error 1,4 평균 2.5. 6. P=.8,R=.5,F1≈.615. 7. 이 문항의 RMSLE는 비음수 target·prediction을 요구한다. log1p 자체는 -1보다 큰 음수에서도 정의되므로 metric의 입력 규약과 구분한다. 8. augmentation은 train에만 stochastic하게 적용하고 평가 분포는 deterministic하게 고정한다. 9. 공간축 절반. 10. γ/β 각 20. 11. additive cell path. 12. `500-20-30+1=451`. 13. target index로 valid. 14. exp 비율 1:3. 15. posterior=prior. 16. 일반 판정 anomaly. 17. D만 학습. 18. channel·공간 projection. 19. 각 label 독립 binary. 20. shape를 억지 reshape하지 말고 output 의미부터 수정한다.

3회 독립 시험에서 모두 16개 이상이어야 “3회 연속 80%” 기준을 충족한 것으로 본다.

---

## 32. Process형 모의고사 — 독자 문제 8문항

### 32.1 응시 규칙

- 권장 70분. Problem 시간을 남기는 훈련이다.
- 아래 “문제”만 보고 먼저 푼다.
- 공개 test를 직접 2개씩 만든다.
- 답안과 hidden test는 풀이 후 확인한다.
- 실제 시험처럼 함수·클래스 이름을 정확히 지킨다.

### 32.2 문제 1 — 선택 열 Robust Scaling

`robust_scale_columns(df, columns)`를 작성하라.

요구사항:

- 각 지정 열을 `(x - median) / IQR`로 변환한다.
- 선택 열은 수치형이며 열 이름은 유일하다. Q1/Q3는 NaN을 제외한 pandas quantile(q, interpolation="linear")을 사용한다. 모든 값이 NaN인 열은 그대로 유지한다.
- IQR이 0이면 해당 열의 유효값을 0.0으로 만든다.
- 기존 NaN은 유지한다.
- 미지정 열, index, 열 순서를 보존한다.
- 입력 DataFrame은 변경하지 않는다.
- 없는 열 이름은 `KeyError`.

#### 모범답안 1

```python
def robust_scale_columns(df, columns):
    if not isinstance(df, pd.DataFrame):
        raise TypeError("df는 DataFrame이어야 합니다")
    out = df.copy()
    for c in columns:
        if c not in out.columns:
            raise KeyError(c)
        s = out[c]
        if s.isna().all():
            continue  # 전체 결측 열은 통계를 계산하지 않고 그대로 유지
        med = s.median(skipna=True)
        q1 = s.quantile(0.25)
        q3 = s.quantile(0.75)
        iqr = q3 - q1
        if pd.isna(iqr):
            continue
        if iqr == 0:
            out[c] = s.where(s.isna(), 0.0)
        else:
            out[c] = (s - med) / iqr
    return out
```

Hidden test: 비연속 index, 빈 columns, 상수 열+NaN, 전부 NaN, 원본 비교, 없는 열.

### 32.3 문제 2 — Causal Group Rolling Feature

`add_past_mean(df, group_col, time_col, value_col, window)`를 작성하라.

요구사항:

- group/time에는 결측이 없고 time은 비교 가능한 숫자·datetime 또는 변환 가능한 날짜 문자열이다. value는 유한 수치 또는 NaN이다. group별 time 오름차순에서 현재 행을 제외한 직전 최대 window행의 값으로 평균을 계산한다. NaN은 평균에서 제외하며 유효값이 없으면 NaN이다. 결과를 value_col + '_past_mean'이라는 새 열에 추가하고 이 열은 입력에 없다고 가정한다.
- 현재 행 값은 평균에서 제외한다.
- 계산 후 원래 행 순서와 index를 복원한다.
- 동일 time은 원래 행 순서로 안정 정렬한다. 이 모의에서는 같은 timestamp의 앞 행도 현재 행 이전에 관측된 것으로 정의한다.
- 입력 변경 금지, `window>=1`.

#### 모범답안 2

```python
def add_past_mean(df, group_col, time_col, value_col, window):
    if isinstance(window, (bool, np.bool_)) or not isinstance(window, (int, np.integer)) or window < 1:
        raise ValueError("window는 1 이상")
    for c in (group_col, time_col, value_col):
        if c not in df.columns:
            raise KeyError(c)

    out = df.copy()
    order_col = "__hdat_order__"
    while order_col in out.columns:
        order_col = "_" + order_col
    time_key = time_col
    drop_cols = [order_col]
    if not (
        pd.api.types.is_numeric_dtype(out[time_col])
        or pd.api.types.is_datetime64_any_dtype(out[time_col])
    ):
        time_key = "__hdat_time__"
        while time_key in out.columns:
            time_key = "_" + time_key
        out[time_key] = pd.to_datetime(out[time_col], errors="raise")
        drop_cols.append(time_key)
    out[order_col] = np.arange(len(out))
    original_index = out.index.copy()
    work = out.sort_values([group_col, time_key, order_col], kind="stable")
    new_col = f"{value_col}_past_mean"
    work[new_col] = (
        work.groupby(group_col, sort=False, observed=True)[value_col]
        .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
    )
    work = work.sort_values(order_col, kind="stable")
    work.index = original_index
    return work.drop(columns=drop_cols)
```

Hidden test: group 1행, 같은 timestamp, 섞인 입력, NaN, 문자열 index. 첫 행 past mean은 NaN이 자연스럽다.

### 32.4 문제 3 — Stable Softmax Cross-Entropy

`numpy_cross_entropy(logits, targets)`를 작성하라.

- logits는 유한 실수 shape (N,C), N>=1, C>=1이며 targets는 shape (N,)의 정수 class index이다. 각 target은 0 이상 C 미만이다.
- stable log-sum-exp를 사용한다.
- 각 sample NLL의 평균 scalar 반환.
- 빈 batch, 범위 밖 target, NaN/Inf, shape 오류는 `ValueError`.

#### 모범답안 3

```python
def numpy_cross_entropy(logits, targets):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(targets)
    if z.ndim != 2 or y.ndim != 1 or len(z) != len(y) or len(z) == 0:
        raise ValueError(f"shape 오류: {z.shape}, {y.shape}")
    if not np.issubdtype(y.dtype, np.integer):
        raise ValueError("targets는 정수 class index")
    if not np.isfinite(z).all():
        raise ValueError("logits NaN/Inf")
    if y.min() < 0 or y.max() >= z.shape[1]:
        raise ValueError("target 범위 오류")

    max_z = z.max(axis=1, keepdims=True)
    shifted = z - max_z
    logsumexp_shifted = np.log(np.exp(shifted).sum(axis=1))
    correct_shifted = shifted[np.arange(len(z)), y]
    return float(np.mean(logsumexp_shifted - correct_shifted))
```

검산: 모든 logits가 같고 C=4면 CE는 `log(4)`.

### 32.5 문제 4 — Split Validator

`validate_split(train_idx, valid_idx, n_rows, groups=None, times=None)`를 작성하라.

- n_rows는 양의 정수이다. train_idx/valid_idx는 각각 비어 있지 않은 1D integer이며 0 이상 n_rows 미만, 각 집합 내 중복이 없어야 한다. groups/times가 주어지면 각각 길이 n_rows의 1D이며 결측을 허용하지 않는다. 수치 times는 finite여야 한다. 어느 조건이든 위반하면 ValueError를 낸다.
- train/valid index 교집합 없음.
- groups가 주어지면 group 교집합 없음.
- times가 주어지면 train의 최대 시간이 valid 최소 시간 이하.
- 통과 시 `True`.

#### 모범답안 4

```python
def validate_split(train_idx, valid_idx, n_rows, groups=None, times=None):
    tr = np.asarray(train_idx)
    va = np.asarray(valid_idx)
    if isinstance(n_rows, (bool, np.bool_)) or not isinstance(n_rows, (int, np.integer)) or n_rows < 1:
        raise ValueError("n_rows는 양의 정수")
    for name, idx in (("train", tr), ("valid", va)):
        if idx.ndim != 1 or not np.issubdtype(idx.dtype, np.integer):
            raise ValueError(f"{name} index 형식")
        if len(idx) == 0 or len(np.unique(idx)) != len(idx):
            raise ValueError(f"{name} empty/duplicate")
        if idx.min() < 0 or idx.max() >= n_rows:
            raise ValueError(f"{name} 범위")
    if np.intersect1d(tr, va).size:
        raise ValueError("행 index 교집합")

    if groups is not None:
        g = np.asarray(groups)
        if g.ndim != 1 or len(g) != n_rows:
            raise ValueError("groups는 길이 n_rows의 1D")
        if pd.isna(g).any():
            raise ValueError("groups 결측")
        if set(g[tr]) & set(g[va]):
            raise ValueError("group 교집합")

    if times is not None:
        t = np.asarray(times)
        if t.ndim != 1 or len(t) != n_rows:
            raise ValueError("times는 길이 n_rows의 1D")
        if pd.isna(t).any():
            raise ValueError("times 결측")
        try:
            if np.issubdtype(t.dtype, np.number):
                if not np.isfinite(t).all():
                    raise ValueError("times NaN/Inf")
                comparable = t
            elif np.issubdtype(t.dtype, np.datetime64):
                comparable = t
            else:
                comparable = pd.to_datetime(t, errors="raise").to_numpy()
        except (TypeError, ValueError) as exc:
            raise ValueError("times를 숫자 또는 datetime으로 해석할 수 없습니다") from exc
        if comparable[tr].max() > comparable[va].min():
            raise ValueError("시간 역전")
    return True
```

[주의] 실제 목표가 같은 group의 미래라면 group 교집합 금지와 time 조건을 동시에 강제하지 않을 수 있다. 이 함수는 문제 명세가 둘 다 요구할 때 사용한다.

### 32.6 문제 5 — PIL 중앙 Crop

`center_crop_array(image, crop_width, crop_height)`를 작성하라.

- PIL image 입력.
- 중심 기준 crop. 홀수 차이는 왼쪽/위쪽에 floor offset을 둔다.
- crop_width/crop_height는 bool을 제외한 양의 정수여야 하며 원본보다 큰 crop 또는 잘못된 크기는 ValueError이다.
- 원 mode를 유지한 NumPy array 반환.

#### 모범답안 5

```python
def center_crop_array(image, crop_width, crop_height):
    from PIL import Image
    if not isinstance(image, Image.Image):
        raise TypeError("PIL Image 필요")
    for size in (crop_width, crop_height):
        if isinstance(size, (bool, np.bool_)) or not isinstance(size, (int, np.integer)) or size < 1:
            raise ValueError("crop 크기는 bool을 제외한 양의 정수")
    width, height = image.size
    if crop_width > width or crop_height > height:
        raise ValueError("crop이 원본보다 큼")
    left = (width - crop_width) // 2
    upper = (height - crop_height) // 2
    box = (left, upper, left + crop_width, upper + crop_height)
    return np.asarray(image.crop(box))
```

Hidden test: RGB/gray/RGBA, 원본과 같은 크기, 홀수 차이, 1×1.

### 32.7 문제 6 — Convolution 계산기

`conv2d_info(h, w, in_ch, out_ch, kernel, stride=1, padding=0, dilation=1, groups=1, bias=True)`를 작성하라.

- h,w,in_ch,out_ch,groups는 bool을 제외한 양의 정수이다. kernel,stride,padding,dilation은 bool을 제외한 int 또는 길이2 tuple/list (높이,너비)를 허용한다. padding은 0 이상, 나머지는 양수이며 bias는 bool이다. 잘못된 설정은 ValueError이다.
- output `(out_h,out_w)`와 parameter 수 반환.
- channel이 groups로 나누어지지 않거나 output이 1 미만이면 오류.

#### 모범답안 6

```python
def conv2d_info(h, w, in_ch, out_ch, kernel, stride=1,
                padding=0, dilation=1, groups=1, bias=True):
    from numbers import Integral

    def scalar(v, name, allow_zero=False):
        if isinstance(v, (bool, np.bool_)) or not isinstance(v, Integral):
            raise ValueError(f"{name}은 정수여야 합니다")
        value = int(v)
        invalid = value < 0 if allow_zero else value <= 0
        if invalid:
            raise ValueError(f"{name} 범위 오류")
        return value

    def pair(v, name, allow_zero=False):
        if isinstance(v, Integral) and not isinstance(v, (bool, np.bool_)):
            value = scalar(v, name, allow_zero)
            return value, value
        if not isinstance(v, (tuple, list)) or len(v) != 2:
            raise ValueError(f"{name}은 정수 또는 길이 2 tuple/list")
        return (
            scalar(v[0], f"{name}[0]", allow_zero),
            scalar(v[1], f"{name}[1]", allow_zero),
        )

    h, w = scalar(h, "h"), scalar(w, "w")
    in_ch, out_ch = scalar(in_ch, "in_ch"), scalar(out_ch, "out_ch")
    groups = scalar(groups, "groups")
    if not isinstance(bias, bool):
        raise ValueError("bias는 bool이어야 합니다")
    kh, kw = pair(kernel, "kernel")
    sh, sw = pair(stride, "stride")
    ph, pw = pair(padding, "padding", allow_zero=True)
    dh, dw = pair(dilation, "dilation")
    if ph < 0 or pw < 0:
        raise ValueError("padding은 음수 불가")
    if in_ch % groups or out_ch % groups:
        raise ValueError("channel/groups 불일치")

    oh = (h + 2 * ph - dh * (kh - 1) - 1) // sh + 1
    ow = (w + 2 * pw - dw * (kw - 1) - 1) // sw + 1
    if oh < 1 or ow < 1:
        raise ValueError("output 공간 크기 1 미만")
    params = out_ch * (in_ch // groups) * kh * kw
    if bias:
        params += out_ch
    return (int(oh), int(ow)), int(params)
```

### 32.8 문제 7 — 정확한 MLP 명세

다음 구조의 `ExamMLP`를 작성하라.

- input 20
- Linear 20→64, BatchNorm1d(64), ReLU, Dropout(0.25)
- Linear 64→32, ReLU
- Linear 32→3
- forward는 raw logits 반환
- 추가 layer 금지

#### 모범답안 7

```python
class ExamMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(20, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 3),
        )

    def forward(self, x):
        return self.network(x)

exam_mlp = ExamMLP()
assert exam_mlp(torch.zeros(2, 20)).shape == (2, 3)
```

BatchNorm train mode에서 batch 1 dummy는 오류가 날 수 있으므로 2개로 test하거나 `.eval()`에서 batch 1을 검사한다. CE를 쓸 예정이므로 softmax를 추가하지 않는다.

### 32.9 문제 8 — Many-to-One LSTM

`ExamLSTM(n_features, hidden_size, n_classes)`를 작성하라.

- 2층 단방향 LSTM, `batch_first=True`, layer 사이 dropout 0.2
- 입력은 padding 없는 고정 길이 [B,T,F]이며 B>=1,T>=1이다. projection과 bidirectional을 사용하지 않는다.
- 마지막 시점 sequence output으로 class logits `[B,n_classes]`
- hidden/cell을 외부에서 받지 않는다.

#### 모범답안 8

```python
class ExamLSTM(nn.Module):
    def __init__(self, n_features, hidden_size, n_classes):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_size,
            num_layers=2,
            batch_first=True,
            dropout=0.2,
            bidirectional=False,
        )
        self.head = nn.Linear(hidden_size, n_classes)

    def forward(self, x):
        sequence, _ = self.lstm(x)
        return self.head(sequence[:, -1, :])

exam_lstm = ExamLSTM(6, 32, 4)
assert exam_lstm(torch.zeros(3, 10, 6)).shape == (3, 4)
```

Variable-length padded input은 이 명세에 없다. 임의 pack 처리를 추가하지 않는다.

### 32.10 채점표

각 문항 12.5점:

- 일반 입력 정확성 5점
- 함수/클래스·반환 계약 2.5점
- 경계조건 3점
- 원본·shape·dtype 보존 2점

목표:

- 1차: 8문항 중 5개, 90분 이내
- 2차: 6개, 75분 이내
- 3차: 7개 이상, 70분 이내

### 32.11 오답 기록

각 실패마다 다음 네 줄을 작성한다.

1. 틀린 계약:
2. 재현하는 최소 입력:
3. 올바른 assert:
4. 시험장에서 검색할 키워드:

---

## 33. Problem형 완전 모의 — 차량 센서 다중출력 예측

### 33.1 문제

아래 생성기로 만든 100Hz 차량 센서 시계열이 있다.

- 입력 feature: 12개 sensor
- lookback: 과거 0.2초
- horizon: 마지막 관측 0.4초 뒤
- target: 연속값 3개
- validation: 시간순 마지막 구간
- metric: MSE
- test prediction shape: 코드에서 생성된 `X_test`의 window 수 × 3
- 제출: `Submission_problem.npy`, test 순서 유지, numeric finite array
- 딥러닝 모델은 PyTorch를 사용한다.

목표는 최고 성능이 아니라 170분에 Process와 함께 완주 가능한 해결책이다.

### 33.2 데이터 생성 셀

```python
import copy
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

problem_start = time.monotonic()

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def generate_mock_series(n=30000, n_features=12, horizon=40, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.2, (n, n_features)).astype(np.float32)
    seasonal = np.sin(np.arange(n, dtype=np.float32) / 80.0)
    for t in range(1, n):
        X[t] += 0.90 * X[t - 1]
        X[t, 6] += 0.15 * seasonal[t]

    y = np.full((n, 3), np.nan, dtype=np.float32)
    src = X[:-horizon]
    noise = rng.normal(0, 0.03, (n - horizon, 3)).astype(np.float32)
    y[horizon:, 0] = 0.8 * src[:, 0] + 0.2 * src[:, 3] + noise[:, 0]
    y[horizon:, 1] = -0.5 * src[:, 1] + 0.3 * src[:, 4] ** 2 + noise[:, 1]
    y[horizon:, 2] = 0.6 * src[:, 2] - 0.2 * src[:, 5] + 0.1 * src[:, 6] + noise[:, 2]
    return X, y

HORIZON = 40
LOOKBACK = 20
X_raw, y_raw = generate_mock_series(horizon=HORIZON)
print(X_raw.shape, y_raw.shape)
```

`y[e+HORIZON]`은 input 마지막 시점 `e`의 sensor 조합에 noise를 더해 만들었으므로 학습 가능하다. 실제 시험에서는 이런 생성식을 알 수 없으며 EDA와 validation으로 판단한다.

### 33.3 응시자 과제

1. `X_raw`, `y_raw`의 shape/dtype/finite와 첫 유효 target을 확인한다.
2. 원시 시간의 70%·85%에 경계를 두고, 이전 구간 label이 다음 구간의 예측 원점 전에 알려지도록 horizon만큼 경계 sample을 제외한다. 최종 sample 비율은 정확한 70/15/15가 아니다.
3. input window `[e-L+1:e+1]`, target `y[e+H]`를 정확히 만든다.
4. scaler는 train window에만 fit한다.
5. target 평균 또는 flatten linear/MLP baseline을 평가한다.
6. CNN1D를 학습하고 validation MSE를 비교한다.
7. best 모델로 test prediction `(N_test,3)`을 만든다.
8. `Submission_problem.npy` 저장 후 reload 검사한다.
9. 첫 유효 파일까지 걸린 시간을 기록한다.

### 33.4 정답 1 — target index와 window

```python
n = len(X_raw)
train_target_cut = int(n * 0.70)
valid_target_cut = int(n * 0.85)

all_end = np.arange(LOOKBACK - 1, n - HORIZON, dtype=np.int64)
all_target = all_end + HORIZON

train_end = all_end[all_target < train_target_cut]
valid_end = all_end[(all_end >= train_target_cut) & (all_target < valid_target_cut)]
test_end = all_end[all_end >= valid_target_cut]

def materialize(X, y, end_indices, lookback, horizon):
    Xw = np.stack([X[e - lookback + 1:e + 1] for e in end_indices]).astype(np.float32)
    yw = np.stack([y[e + horizon] for e in end_indices]).astype(np.float32)
    if Xw.shape != (len(end_indices), lookback, X.shape[1]):
        raise RuntimeError("window shape 오류")
    if yw.shape != (len(end_indices), 3):
        raise RuntimeError("target shape 오류")
    if not np.isfinite(Xw).all() or not np.isfinite(yw).all():
        raise ValueError("window NaN/Inf")
    return Xw, yw

X_train, y_train = materialize(X_raw, y_raw, train_end, LOOKBACK, HORIZON)
X_valid, y_valid = materialize(X_raw, y_raw, valid_end, LOOKBACK, HORIZON)
X_test, y_test_hidden = materialize(X_raw, y_raw, test_end, LOOKBACK, HORIZON)

assert train_end[-1] + HORIZON < train_target_cut
assert valid_end[0] >= train_target_cut
assert test_end[0] >= valid_target_cut
assert train_end[-1] + HORIZON < valid_end[0]
assert valid_end[-1] + HORIZON < test_end[0]
print(X_train.shape, X_valid.shape, X_test.shape)
```

실제 Problem에서는 test target이 없다. 여기서는 모의 종료 후 일반화 점수를 검산하기 위해 `y_test_hidden`을 보관한다. 모델 선택에는 사용하지 않는다.

### 33.5 정답 2 — train-only scaling

```python
n_features = X_train.shape[-1]
x_scaler = StandardScaler()
X_train_s = x_scaler.fit_transform(X_train.reshape(-1, n_features)).reshape(X_train.shape)
X_valid_s = x_scaler.transform(X_valid.reshape(-1, n_features)).reshape(X_valid.shape)
X_test_s = x_scaler.transform(X_test.reshape(-1, n_features)).reshape(X_test.shape)

y_scaler = StandardScaler()
y_train_s = y_scaler.fit_transform(y_train).astype(np.float32)
y_valid_s = y_scaler.transform(y_valid).astype(np.float32)

X_train_s = X_train_s.astype(np.float32)
X_valid_s = X_valid_s.astype(np.float32)
X_test_s = X_test_s.astype(np.float32)
```

target scaling은 필수가 아니지만 output 세 축의 scale이 다를 때 안정적이다. 다만 scaled MSE는 축마다 다른 가중치를 주는 셈이어서 공식 원단위 MSE와 동치가 아닐 수 있다. 학습 loss는 scaled 공간을 쓰더라도 checkpoint와 최종 모델 선택은 매 epoch inverse한 공식 원단위 MSE로 한다.

### 33.6 정답 3 — baseline

```python
def mse(y_true, y_pred):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if y_true.shape != y_pred.shape:
        raise ValueError((y_true.shape, y_pred.shape))
    return float(np.mean((y_true - y_pred) ** 2))

mean_pred = np.repeat(y_train.mean(axis=0, keepdims=True), len(y_valid), axis=0)
baseline_valid_mse = mse(y_valid, mean_pred)
print("mean baseline valid MSE:", baseline_valid_mse)

# 학습 전에 첫 유효 test 파일을 확보한다.
baseline_test_pred = np.repeat(y_train.mean(axis=0, keepdims=True), len(X_test), axis=0)
if baseline_test_pred.shape != (len(X_test), 3) or not np.isfinite(baseline_test_pred).all():
    raise ValueError("baseline prediction 계약 오류")
baseline_path = Path("Submission_problem.npy")
np.save(baseline_path, baseline_test_pred, allow_pickle=False)
baseline_reload = np.load(baseline_path, allow_pickle=False)
if not np.array_equal(baseline_reload, baseline_test_pred):
    raise IOError("baseline NPY reload 불일치")
first_valid_seconds = time.monotonic() - problem_start
print("first valid submission seconds:", first_valid_seconds)

# 이 synthetic 문제의 target 생성상 last sensor 조합을 모른다고 가정한다.
# 문제 의미가 일치하는 경우에만 last-value baseline을 추가한다.
```

첫 test 제출은 train target 평균으로도 만들 수 있다. 성능은 낮지만 파일 계약을 검증한다.

### 33.7 정답 4 — DataLoader와 CNN1D

```python
train_ds = TensorDataset(
    torch.from_numpy(X_train_s),
    torch.from_numpy(y_train_s),
)
valid_ds = TensorDataset(
    torch.from_numpy(X_valid_s),
    torch.from_numpy(y_valid_s),
)
test_ds = TensorDataset(torch.from_numpy(X_test_s))

BATCH_SIZE = 256
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

class MockCNN1D(nn.Module):
    def __init__(self, n_features, out_dim=3):
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

    def forward(self, x):
        x = x.permute(0, 2, 1)
        return self.head(self.features(x))

model = MockCNN1D(n_features).to(DEVICE)
assert model(torch.zeros(2, LOOKBACK, n_features, device=DEVICE)).shape == (2, 3)
```

### 33.8 정답 5 — 시간 제한 학습

```python
criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

best_metric = float("inf")
best_state = None
patience = 5
wait = 0
start = time.monotonic()
MAX_SECONDS = 120.0

for epoch in range(1, 41):
    model.train()
    train_sum, train_n = 0.0, 0
    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        pred = model(xb)
        loss = criterion(pred, yb)
        if not torch.isfinite(loss):
            raise FloatingPointError("non-finite train loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        train_sum += loss.detach().item() * len(xb)
        train_n += len(xb)

    model.eval()
    valid_sum, valid_n = 0.0, 0
    valid_pred_chunks = []
    with torch.inference_mode():
        for xb, yb in valid_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            out = model(xb)
            loss = criterion(out, yb)
            valid_sum += loss.item() * len(xb)
            valid_n += len(xb)
            valid_pred_chunks.append(out.cpu().numpy())
    valid_scaled_loss = valid_sum / valid_n
    valid_pred_epoch = y_scaler.inverse_transform(np.concatenate(valid_pred_chunks))
    valid_metric = mse(y_valid, valid_pred_epoch)  # 공식 원단위 MSE
    print(epoch, train_sum / train_n, valid_scaled_loss, valid_metric)

    if valid_metric < best_metric - 1e-6:
        best_metric = valid_metric
        best_state = copy.deepcopy({k: v.detach().cpu() for k, v in model.state_dict().items()})
        wait = 0
    else:
        wait += 1

    if wait >= patience or time.monotonic() - start >= MAX_SECONDS:
        break

if best_state is None:
    raise RuntimeError("best checkpoint 없음")
model.load_state_dict(best_state)
model.to(DEVICE)
```

### 33.9 정답 6 — validation 원 단위

```python
def predict_scaled(model, loader, device):
    model.eval()
    chunks = []
    with torch.inference_mode():
        for batch in loader:
            xb = batch[0].to(device, dtype=torch.float32)
            chunks.append(model(xb).cpu().numpy())
    if not chunks:
        raise ValueError("빈 prediction loader")
    return np.concatenate(chunks, axis=0)

valid_pred_s = predict_scaled(model, DataLoader(
    TensorDataset(torch.from_numpy(X_valid_s)),
    batch_size=BATCH_SIZE,
    shuffle=False,
), DEVICE)
valid_pred = y_scaler.inverse_transform(valid_pred_s)
cnn_valid_mse = mse(y_valid, valid_pred)
print("CNN valid MSE:", cnn_valid_mse)
print("output RMSE:", np.sqrt(np.mean((y_valid - valid_pred) ** 2, axis=0)))
```

### 33.10 정답 7 — test 저장

```python
cnn_test_pred_s = predict_scaled(model, test_loader, DEVICE)
cnn_test_pred = y_scaler.inverse_transform(cnn_test_pred_s)

if cnn_valid_mse < baseline_valid_mse:
    selected_name = "CNN1D"
    test_pred = cnn_test_pred
else:
    selected_name = "mean baseline"
    test_pred = baseline_test_pred
print("selected by original-unit valid MSE:", selected_name)

expected_shape = (len(X_test), 3)
if test_pred.shape != expected_shape:
    raise ValueError((test_pred.shape, expected_shape))
if not np.isfinite(test_pred).all():
    raise ValueError("test prediction NaN/Inf")

output_path = Path("Submission_problem.npy")
np.save(output_path, test_pred, allow_pickle=False)
reloaded = np.load(output_path, allow_pickle=False)
if reloaded.shape != expected_shape or not np.array_equal(reloaded, test_pred):
    raise IOError("저장/reload 불일치")
print(output_path, reloaded.shape, reloaded.dtype)
```

연습이 끝난 뒤에만 hidden test MSE를 본다.

```python
print("hidden test MSE (연습 채점용):", mse(y_test_hidden, test_pred))
```

### 33.11 채점 기준

| 항목 | 점수 | 0점 조건 |
|---|---:|---|
| 데이터 계약·window | 20 | horizon/window 잘못 대응 |
| validation·누수 | 20 | random overlapping split, 전체 scaler fit |
| 첫 baseline | 10 | 유효 prediction 없음 |
| PyTorch 모델·학습 | 20 | output/loss shape 오류 |
| metric·inverse | 10 | scaled metric만 보고 공식값 미계산 |
| 제출 파일 | 20 | 파일명/shape/NaN/순서/reload 실패 |

합격 조건은 점수와 별개로 `Submission_problem.npy`가 유효해야 한다.

### 33.12 확장 모의 A — group 일반화

여러 차량의 서로 다른 offset과 noise를 생성하고 test를 새 vehicle로 둔다.

- group 밖 window 금지
- vehicle GroupKFold/holdout
- vehicle ID를 feature로 쓰는 모델과 제거한 모델 비교
- unseen vehicle 성능 해석

### 33.13 확장 모의 B — 불균형 event 분류

target을 “향후 50 sample 안에 threshold event 발생”으로 바꾼다.

- causal label 생성
- BCEWithLogitsLoss/pos_weight
- validation Macro-F1/PR-AUC
- threshold는 validation에서 선택
- event 인접 window 양 fold 분리

### 33.14 회고 질문

1. 첫 유효 파일까지 몇 분 걸렸는가?
2. 가장 오래 걸린 shape 오류는 무엇인가?
3. validation 선택이 실제 test 생성 과정을 모사했는가?
4. CNN1D가 baseline을 이겼는가? 아니라면 왜인가?
5. 마지막 7분을 확보했는가?

### 33.15 통합 모의고사 — Process 8 + Problem 1, 총 170분

32강과 33강을 따로 푸는 것만으로는 두 영역이 시간을 경쟁하는 상황을 훈련할 수 없다. 다음 세트는 답안을 보지 않고 **한 번의 170분 타이머**로 연속 응시한다.

권장 시간:

| 구간 | 행동 |
|---:|---|
| 0~7분 | 9문항 전체 훑기, Process A/B/C 분류, Problem 계약 기록 |
| 7~57분 | Process A/B 6문항 목표 |
| 57~70분 | Problem 첫 baseline과 유효 NPY |
| 70~142분 | Problem 검증·PyTorch MLP·threshold 개선 |
| 142~160분 | 남은 Process |
| 160~170분 | 양쪽 reload·Ctrl+S·제출 확인 |

#### 통합 Process 1 — 결측 보고서

missing_report(df)를 구현하라. 입력은 열 이름이 유일한 DataFrame이다. 원래 열 이름을 index로 갖는 DataFrame을 원래 열 순서로 반환하고, 반환 열 순서는 dtype, missing_count, missing_rate, nunique_with_missing으로 한다. dtype에는 원 열 dtype 객체를 기록하고 missing_rate=missing_count/행 수로 계산하되 행이 0개이면 0.0으로 정의한다. nunique_with_missing은 NaN을 하나의 고유값으로 센다. 입력의 비연속 index를 처리하고 원본을 변경하지 않는다.

#### 통합 Process 2 — Group Z-score

group_zscore(df, group_col, value_col)을 구현하라. group_col은 결측이 없고 value_col은 유한 수치 또는 NaN이다. value_col을 group별 유효값의 population mean/std(ddof=0)로 z-score 변환한 복사 DataFrame을 반환한다. std=0인 유효값은 0, 기존 NaN과 전부 NaN인 group은 NaN을 유지하고 미지정 열은 보존한다. 원 index/순서를 지킨다.

#### 통합 Process 3 — Binary F1

NumPy만으로 `binary_f1(y_true, y_pred)`를 작성하라. 두 배열은 같은 1D shape와 `{0,1}` 값이어야 한다. precision/recall denominator가 0이면 해당 값과 F1을 0으로 정의한다.

#### 통합 Process 4 — Causal Lag

add_lag(df, group_col, time_col, value_col, lag=1)을 구현하라. group/time에는 결측이 없고 time은 비교 가능한 숫자 또는 datetime이다. group/time/원래 위치 순으로 안정 정렬한 뒤 group 내부에서 value_col을 lag행 이동해 value_col + '_lag'라는 새 열에 추가한다. 동일 time의 앞 행도 이전 관측으로 정의한다. 새 열은 입력에 없다고 가정한다. 원래 행 순서·index·나머지 열을 보존하며 입력을 변경하지 않는다. lag는 bool을 제외한 1 이상 정수다.

#### 통합 Process 5 — RGB Normalize

normalize_rgb(image, mean, std)를 구현하라. image는 H>=1,W>=1인 uint8 NumPy array (H,W,3)이고 mean/std는 길이3의 finite 실수이며 std>0이다. image를 float32로 바꾸어 255로 나누고 HWC에서 CHW로 축을 옮긴 뒤 채널별 (x-mean)/std를 적용한 float32 PyTorch tensor 하나를 반환한다. 입력 image를 변경하지 않는다.

#### 통합 Process 6 — 정확한 CNN Block

`ExactBlock`을 구현하라: Conv2d 3→8 K3 P1 bias=False → BatchNorm2d(8) → ReLU → MaxPool2d(2). 입력 `(B,3,32,32)`의 출력 assert와 parameter 수를 쓰라. 추가 layer 금지.

#### 통합 Process 7 — Class Weight

balanced_class_weights(y,n_classes)를 작성하라. y는 비어 있지 않은 1D integer class index, n_classes=C는 bool을 제외한 양의 정수이고 모든 y는 0 이상 C 미만이어야 한다. class별 weight N/(C×count_c)를 class0부터 C-1 순서의 shape(C,) float32 Tensor로 반환한다. 누락 class 또는 입력 계약 위반은 ValueError이다.

#### 통합 Process 8 — NPY 계약

`save_checked_npy(pred, path, n_rows, n_outputs)`를 구현하라. 정확히 `(n_rows,n_outputs)`, numeric, finite인지 검사하고 `allow_pickle=False`로 저장한 뒤 reload의 shape/dtype/value를 확인한다.

### 33.16 통합 Problem — 새 차량 고장 분류

다운로드한 full-mock-tabular.py를 저장한 폴더에서 python full-mock-tabular.py --generate-only로 train/test를 생성한다.

- 행: 차량 운행 segment
- 동일 `vehicle_id`가 여러 행
- test에는 train에 없던 차량만 존재
- 수치/범주형/결측 혼합
- target: `fault` 0/1, 불균형
- metric: Macro-F1
- 출력: test label `(N,)`
- 제출 파일: `Submission_problem.npy`

요구:

1. `vehicle_id`를 group으로 validation하고 feature에서는 제외한다.
2. imputer/encoder/scaler는 train fold만 fit한다.
3. 빠른 baseline 제출을 먼저 만든다.
4. PyTorch MLP와 `BCEWithLogitsLoss(pos_weight=...)`를 학습한다.
5. validation에서 threshold를 정한다.
6. 전체 train 재학습 후 test label을 예측한다.
7. 정확한 파일을 저장·reload한다.

### 33.17 통합 Process 정답 핵심

실제 응시 후 확인한다.

1. df.isna().sum(), missing_count/len(df)(행이 0개이면 missing_rate=0.0), nunique(dropna=False)를 원 열 이름 index로 조립한다. 반환 열 순서는 dtype, missing_count, missing_rate, nunique_with_missing이다.
2. `groupby.transform('mean')`, population std. std 0 mask에서 NaN을 덮지 않는다.
3. TP/FP/FN을 boolean sum하고 `2PR/(P+R)`의 0 denominator를 처리한다.
4. `__order__` 보존→stable sort→groupby shift(lag)→원 order 복원.
5. 입력 계약 검사→`torch.from_numpy(image).permute(2,0,1).float()/255`→mean/std `(3,1,1)`.
6. output `(B,8,16,16)`. Conv parameter `8×3×3×3=216`, BN learnable 16, 총 232.
7. `np.bincount(y,minlength=C)` 후 0 count 검사, `N/(C*count)`.
8. 변환 뒤에도 finite 검사, 저장 뒤 `np.load(...,allow_pickle=False)`와 `np.array_equal`.

### 33.18 통합 모의 채점표

| 영역 | 점수 | 합격 조건 |
|---|---:|---|
| Process 일반 정확성 | 24 | 8개 중 6개 이상 일반 test |
| Process 경계·계약 | 16 | 원본/NaN/batch/shape hidden test |
| Problem split·누수 | 15 | group 교집합 0, fold 내부 fit |
| 첫 유효 baseline | 10 | 70분 전에 NPY 존재 |
| PyTorch MLP·metric | 15 | logits/BCE/threshold 정확 |
| 최종 제출 | 20 | shape/dtype/finite/order/reload |

다음 중 하나면 총점과 무관하게 재응시한다.

- Problem 파일 미생성
- Process/Problem 한쪽 미제출
- test 정보로 threshold·전처리 fit
- 마지막 7분 미확보
- 전체 helper 무수정 복사로 문제 계약을 설명하지 못함

---

## 34. 21일 완주 계획·최종 암기표·합격 체크

### 34.1 하루 고정 루틴

하루 3~5시간 기준:

1. 20분: 전날 오답을 책 없이 다시 풀기
2. 60~90분: 오늘 강의 정독·3줄 요약
3. 60~120분: 코드를 직접 입력하고 변형
4. 30~60분: 확인문제·시간제한 실습
5. 15분: 오답 원인·검색 키워드 기록
6. 5분: 오픈북에는 필요한 최소 블록만 표시

읽는 시간보다 **빈 화면에서 재현하는 시간**이 길어야 한다.

### 34.2 21일 순서

| 일 | 강의 | 반드시 남길 산출물 |
|---:|---|---|
| 1 | 1~3 | 시험 시간표, shape 표, DataFrame 계약 검사기 |
| 2 | 4~6 | 선형층·gradient·Bayes 손계산지 |
| 3 | 7~8 | 2분 EDA 보고서, train-only 전처리 pipeline |
| 4 | 9~10 | selection/extraction 비교표, 선형·KNN baseline |
| 5 | 11~12 | entropy/Gini, RF/SVM/cluster 비교표 |
| 6 | 13 | random/group/time/window split 실습 |
| 7 | 14~15 | metric 손계산, 불균형 threshold, 6-run 탐색표 |
| 8 | 16~17 | Tensor/DataLoader, task별 MLP micro-test |
| 9 | 18 | 공통 train/eval/early stop loop |
| 10 | 19 | CNN 출력·parameter 20문제, ImageCNN |
| 11 | 20 | ResidualBlock, ResNet head-only 학습 순서 |
| 12 | 21 | group-safe window, target-index split, lazy Dataset |
| 13 | 22 | CNN1D·GRU/LSTM 동일 task 비교 |
| 14 | 23 | attention 손계산, Transformer dummy forward |
| 15 | 24~25 | AE threshold, VAE/GAN 최소 구현 + 필기 모의 1회(50분), 오답 재풀이 |
| 16 | 26 | Process 함수 8개를 90분 안에 재현 |
| 17 | 27 | 표형 회귀·분류 Problem 2개 제출 + 필기 모의 2회(50분), 오답 재풀이 |
| 18 | 28~29 | 시계열/이미지/AE mini Problem 제출 |
| 19 | 30~31 | 제출 검증 셀 + 필기 모의 3회(50분), 오답 재풀이·3회 점수표 완성 |
| 20 | 32 | Process 8문항 70분, 오답 재풀이 |
| 21 | 33~34 | 완전 모의 170분, 최종 오픈북·환경 점검 |

시간이 부족해도 13강 validation, 14강 metric, 21강 window, 26강 Process, 30강 제출은 생략하지 않는다. Transformer/VAE/GAN은 구현 반복 횟수를 줄이되 개념은 범위에서 빼지 않는다.

### 34.3 활성함수 암기표

| 함수 | 범위 | 0 중심 | 핵심 |
|---|---|---|---|
| sigmoid | `(0,1)` | 아니오 | binary probability, saturation |
| tanh | `(-1,1)` | 예 | RNN candidate 등에 사용, saturation |
| ReLU | `[0,∞)` | 아니오 | 빠름, dead ReLU 가능 |
| LeakyReLU | `(-∞,∞)` | 대체로 | 음수 작은 slope |
| GELU | 약 `[-0.17,∞)` | 대체로 | exact `xΦ(x)`, 부드러운 gating, Transformer |
| softmax | 각 `(0,1)`, 합 1 | 해당 없음 | mutually exclusive multiclass |

### 34.4 출력·loss 암기표

| Task | output | target | loss | 변환 |
|---|---|---|---|---|
| 회귀 D | `[B,D]` | float `[B,D]` | MSE/MAE/Huber | inverse scale |
| 이진 | `[B,1]` | float `[B,1]` | BCEWithLogits | sigmoid, threshold |
| 다중분류 K | `[B,K]` | long `[B]` | CrossEntropy | softmax/argmax |
| multilabel K | `[B,K]` | float `[B,K]` | BCEWithLogits | label별 sigmoid |

### 34.5 optimizer 암기표

| optimizer | 기억할 상태 | 강점/함정 |
|---|---|---|
| SGD | 현재 gradient | 단순, LR 민감 |
| Momentum | gradient 방향 velocity | 진동 완화·가속 |
| RMSprop | squared gradient EMA | parameter별 step 조절 |
| Adam | 1차+2차 moment, bias correction | 빠른 기본, 항상 최종 최고는 아님 |
| AdamW | Adam + decoupled weight decay | 실기 안정 기본 |

### 34.6 모델 비교 암기표

| 모델 | 강점 | 약점/함정 |
|---|---|---|
| Linear/Logistic | 빠름·해석·baseline | 비선형 상호작용 제한 |
| KNN | 단순·local | scale/고차원/prediction 비용 |
| Tree | 비선형·scale 불필요 | 단일 tree variance |
| RF/ExtraTrees | 강한 표형 baseline | 큰 memory, importance 편향 |
| SVM | margin·kernel | scaling, 큰 n kernel 비용 |
| MLP | 범용 differentiable | 전처리·tuning 필요 |
| CNN | local·weight sharing·병렬 | layout/receptive field |
| RNN/LSTM/GRU | 순차 state | 긴 계산, gradient 문제 |
| Transformer | global attention·병렬 | T² memory, 위치/mask 필요 |
| AE | reconstruction/representation | anomaly도 복원 가능 |
| VAE | probabilistic latent | reconstruction 흐림, KL balance |
| GAN | 선명한 생성 가능 | 불안정·mode collapse |

### 34.7 split 암기표

| test에서 새것 | split | assert |
|---|---|---|
| 독립 행 | random/stratified | class 비율 |
| 사람·차량·설비 | group | group 교집합 0 |
| 같은 대상 미래 | time | train max time ≤ valid min |
| overlapping window | target-index/time/group | random window 혼합 금지, target/future 누수 없음; 예측 때 이용 가능한 과거 history overlap은 허용 가능 |

### 34.8 CNN/RNN/Attention 공식

Convolution output:

`floor((in + 2P - D(K-1) - 1)/S + 1)`

Conv2d parameter:

`Co × (Ci/groups × Kh × Kw) + (bias면 Co)`

LSTM:

`c_t = f_t⊙c_{t-1} + i_t⊙g_t`, `h_t=o_t⊙tanh(c_t)`

Attention:

`softmax(QKᵀ/sqrt(d_k))V`

### 34.9 실기 시작 8줄

1. 한 행/샘플의 의미
2. `X_train/y_train/X_test` shape·dtype
3. TASK와 target mapping
4. official metric과 방향(min/max)
5. random/group/time split
6. output shape·row order
7. 파일명·dtype·저장 cell
8. 첫 baseline 완료 목표 시각

### 34.10 합격 준비 상태 체크

아래 수치는 자체 복습 목표이며 공식 합격선이 아니다. 공식 커트라인과 세부 채점 기준은 비공개다.

필기:

- [ ] 20문항 모의에서 3회 연속 16개 이상
- [ ] convolution/parameter/metric/gradient 계산을 2분 내 풀이
- [ ] optimizer, BN, dropout, transfer, AE/VAE/GAN/Transformer 비교 가능

Process:

- [ ] 8문항 중 6개 이상을 70분 안에 구현
- [ ] 각 함수에 hidden test 3개를 직접 만듦
- [ ] 정확한 CNN/MLP/LSTM 명세를 추가 layer 없이 구현

Problem:

- [ ] 표형 회귀·분류 각각 유효 제출
- [ ] time/group window 문제 유효 NPY 제출
- [ ] 이미지 CNN과 AE anomaly mini 제출
- [ ] baseline을 60~65분 안에 확보

제출·운영:

- [ ] NPY/CSV shape·dtype·finite·reload 자동 검사
- [ ] OOM 축소 순서 숙지
- [ ] 마지막 7분과 Ctrl+S 습관
- [ ] 당시 공식 오픈북·부정행위 규정 재확인

### 34.11 최종 자기설명 시험

아래 12개를 각각 30초 안에 말하지 못하면 해당 강의로 돌아간다.

1. 왜 scaler를 train fold에만 fit하는가?
2. `[B]`와 `[B,1]`이 loss에서 왜 위험한가?
3. group split과 time split을 어떻게 고르는가?
4. Macro-F1과 ROC-AUC는 무엇이 다른가?
5. BCEWithLogits/CE의 output-target 계약은?
6. convolution output과 parameter 수는?
7. BN/Dropout의 train/eval 차이는?
8. LSTM gate와 GRU 차이는?
9. attention scaling·position·mask가 왜 필요한가?
10. AE/VAE/GAN의 목적과 loss는?
11. overlapping window 누수는 어떻게 막는가?
12. 제출 직전 무엇을 검사하는가?

### 34.12 자료 연결

- 시험 전에 반복 검색할 PyTorch 치트시트: 학습 사이트의 `cheatsheet.html`
- 재사용 source: 학습 사이트 자료실의 `hdat-templates.py`
- 개별 강의와 공식 자료: 학습 사이트의 `learn/`, `resources/`
- 적응형 학습 일정표: 학습 사이트 자료실의 `23-day-plan.md`(21일·9일·3일 트랙 포함)

공개 사이트와 외부 링크는 **시험 전 학습용**이다. 실제 시험 중 허용되는 사이트·검색·개인 자료의 범위는 해당 회차 공식 규정과 감독관 안내를 다시 확인한다.

### 34.13 공식 범위 대조표

| 공식 공개 범주 | 본문 |
|---|---|
| 인공지능 기초·ANN·활성함수 | 4~6, 16~18 |
| 데이터 처리·Python 연산·증강 | 2~3, 7~9, 26 |
| DNN/MLP | 16~18, 27 |
| CNN | 19~20, 26, 29 |
| RNN/LSTM | 21~22, 28 |
| 손실·과적합·optimizer·BN | 14~18 |
| 전이학습·평가지표 | 14, 20 |
| AE·GAN·VAE | 24~25 |
| ResNet·Transformer | 20, 23 |
| 수학적 원리 | 4~6, 10~15, 17~25 |
| EDA·전처리 | 3, 7~8, 27~29 |
| Feature Selection·Extraction | 9, 12, 26 |
| Regularization·Optimization | 5, 15, 18 |
| Validation Strategy | 13, 21, 26~29 |
| Modeling·Hyperparameter Tuning | 10~15, 27~29 |

### 34.14 출처와 사용 주의

시험 구성·공개 범위·환경·응시 규정은 [2026 HDAT-DS 공식 매뉴얼](https://hdat.gitbook.io/2026-hdat-ds)과 [HDAT-DS 공식 연습문제 페이지](https://exam.hyundai-ngv.com/practice/13567)를 우선했다. 공식 연습문제는 저작권 보호 자료이므로 본 교재는 원문을 재수록하지 않고 같은 능력을 연습하는 독자 문제를 만들었다.

사용자가 제공한 2025 교육자료는 개념 범위와 학습 순서를 대조하는 참고로 사용했다. “작년 출제자 제작”이라는 전언은 공개 근거로 확인되지 않았으므로 사실로 단정하지 않는다. 교육자료의 오래된 API, 빈 코드, 잘못된 설명은 교정해 사용했다.

시험 당일에는 이 교재보다 해당 회차 문제 지시·제공 skeleton·공식 규정·감독관 안내가 항상 우선한다.

---

## 끝까지 공부한 뒤

이 교재를 한 번 읽은 것은 시작일 뿐이다. 다음 세 가지는 개인 학습 점검 기준이며 공식 합격 판정이 아니다.

1. 폐쇄형 필기 20문항을 시간 안에 안정적으로 푼다.
2. Process 함수와 정확한 PyTorch 구조를 빈 화면에서 구현한다.
3. 처음 보는 데이터에서도 170분 안에 **유효한 제출 파일**을 만든다.

마지막에는 새 모델을 더 배우기보다 오답, window index, loss-shape, 제출 검증을 반복한다.
