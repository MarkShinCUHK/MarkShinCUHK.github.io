# HDAT-DS 2026 적응형 학습계획 — PyTorch only

이 문서는 날짜가 아니라 **남은 기간**에 맞춰 쓰는 계획표다. Python이라는 이름만 아는 상태라면 먼저 10일 입문 트랙을 수행한 뒤 21일 본과정으로 이어지는 **총 31일 완전 기초 트랙**을 권장한다. 이미 Python·NumPy·pandas를 사용할 수 있다면 21일 본과정부터, 학부 수준 머신러닝·딥러닝까지 배웠다면 9일 압축 트랙부터 시작한다. 3일 트랙은 새로 배우는 일정이 아니라 마지막 점검용이다.

> 시험 구성·환경·금지행위는 2026 HDAT-DS 공식 매뉴얼과 해당 회차 안내를 매번 다시 확인한다. 이 계획의 시간 배분과 준비 기준은 개인 학습용 권장안이며 공식 합격선이나 채점 기준이 아니다.

## 매일 고정 루틴

1. 전날 내용을 보지 않고 10분 동안 설명한다.
2. 오늘 강의를 읽고 작은 예제의 출력·shape를 먼저 예측한다.
3. Python 또는 PyTorch 코드를 직접 실행하고 최소 한 곳을 바꾼다.
4. 해설을 닫은 채 확인문제와 실습을 푼다.
5. 실수를 개념 / shape·dtype / 누수·검증 / 코드 / 시간 / 제출 중 하나로 기록한다.
6. 강의의 완료 기준을 실제로 확인한 뒤 완료 표시한다.

권장 공부 시간은 입문 트랙 하루 1.5~2.5시간, 본과정 하루 2.5~4시간, 압축 트랙 하루 4~6시간이다. 코드가 처음이라면 진도보다 실행과 디버깅을 우선한다.

## 트랙 0 — Python 이름만 아는 사람의 입문 10일

이 트랙의 강의 번호는 웹사이트의 **Python 기초 1~10강**이다. 각 날의 완료 기준과 실습을 마친 뒤 다음 날로 넘어간다.

- [ ] **입문 1일차 · 기초 1강** — Jupyter 셀, 커널, 현재 폴더와 오류 마지막 줄을 읽는다.
- [ ] **입문 2일차 · 기초 2강** — 변수, 숫자·문자열·불리언, 연산 결과와 `type`을 출력한다.
- [ ] **입문 3일차 · 기초 3강** — list·tuple·dict·set을 만들고 index와 slice를 손으로 예측한다.
- [ ] **입문 4일차 · 기초 4강** — if·for·함수·예외 처리를 작은 값으로 직접 작성한다.
- [ ] **입문 5일차 · 기초 5강** — `np.array`, `shape`, `ndim`, `dtype`을 말로 읽는다.
- [ ] **입문 6일차 · 기초 6강** — indexing, axis, reshape, broadcasting, view·copy를 출력으로 구분한다.
- [ ] **입문 7일차 · 기초 7강** — `pd.DataFrame`, `head`, `info`, 열 선택, `loc`·`iloc`를 연습한다.
- [ ] **입문 8일차 · 기초 8강** — 결측·자료형·조건 필터·groupby·merge를 안전하게 처리한다.
- [ ] **입문 9일차 · 기초 9강** — histogram·막대·산점도를 그리고 축과 단위를 읽는다.
- [ ] **입문 10일차 · 기초 10강** — CSV 점검 → train-only 정제 → NumPy → float32 Tensor → DataLoader를 한 번 완주한다.

입문 10일차가 끝나면 `python-data-basics-starter.ipynb`를 빈칸부터 다시 풀고, 막힌 셀만 Solution과 대조한다. 그 다음 아래 21일 본과정을 시작한다.

## 트랙 A — Python 기초를 마친 뒤 진행하는 HDAT 본과정 21일

### 1단계: 시험과 기초 언어

- [ ] **1일차 · 1강** — 공식 시험 구조, 공개 범위, 금지행위, 저장·제출 지시를 읽고 나만의 시험 지도를 만든다.
- [ ] **2일차 · 2강** — Python 객체, axis, indexing, copy/view, broadcasting을 작은 배열로 재현한다.
- [ ] **3일차 · 3강** — DataFrame의 shape·dtype·결측·Inf·중복·정렬을 2분 안에 점검하는 함수를 만든다.
- [ ] **4일차 · 4강** — 행렬곱과 Linear layer의 입력·가중치·출력 shape를 손으로 계산한다.
- [ ] **5일차 · 5강** — chain rule, backward, gradient 누적, optimizer step을 PyTorch로 확인한다.
- [ ] **6일차 · 6강** — likelihood, entropy, cross-entropy, bias–variance를 손계산 예제와 연결한다.

### 2단계: 데이터·검증·머신러닝

- [ ] **7일차 · 7~8강** — target·group·시간 축 EDA 후 결측·이상치·스케일·인코딩을 train에만 fit한다.
- [ ] **8일차 · 9~10강** — Feature Engineering·Selection·PCA와 선형·로지스틱·KNN baseline을 비교한다.
- [ ] **9일차 · 11~12강** — tree·ensemble·SVM과 군집·DBSCAN·PCA의 가정 및 쓰임을 표로 정리한다.
- [ ] **10일차 · 13강** — random·stratified·group·time split을 선택하고 누수 사례를 세 가지 만든다.
- [ ] **11일차 · 14~15강** — 회귀·분류 지표, 불균형, threshold, regularization과 작은 tuning budget을 연습한다.

### 3단계: PyTorch 핵심 모델

- [ ] **12일차 · 16강** — Tensor → Dataset/DataLoader → nn.Module → train → inference → save/load를 한 번 완주한다.
- [ ] **13일차 · 17~18강** — MLP의 출력층·loss·target 계약과 optimizer·초기화·BN·Dropout을 확인한다.
- [ ] **14일차 · 19~20강** — convolution output/parameter 수를 계산하고 CNN·ResNet·전이학습 코드를 실행한다.
- [ ] **15일차 · 21~22강** — time-aware split과 window index를 만든 뒤 RNN·LSTM·GRU·CNN1D의 shape를 비교한다.
- [ ] **16일차 · 23~25강** — Attention·Transformer·AE·VAE·GAN의 목적, 핵심 loss와 입력 계약을 비교한다.

### 4~5단계: 실기 완주와 모의

- [ ] **17일차 · 26강** — Process 함수 계약을 읽고 원본 보존·경계값을 포함한 자가 edge-case test를 만든다.
- [ ] **18일차 · 27강** — 표형 Problem을 EDA → split → baseline → MLP → 검증 → 저장 순서로 끝낸다.
- [ ] **19일차 · 28~30강** — 시계열·이미지·AE 분기를 연습하고 제출 파일 reload와 Ctrl+S 루틴을 고정한다.
- [ ] **20일차 · 31~32강** — 필기 20문항 50분과 Process 8문항 모의를 답안 없이 수행하고 오답을 분류한다.
- [ ] **21일차 · 33~34강** — 170분 통합 모의를 수행한 뒤 최종 암기표와 취약 강의만 다시 본다.

## 트랙 B — 학부 ML/DL 경험자를 위한 9일 압축

- [ ] **1일차 · 1~3강** — 공식 규정, Python shape, NumPy·pandas 계약을 점검한다.
- [ ] **2일차 · 4~6강** — 선형대수·미분·확률에서 설명하지 못하는 항목만 보완한다.
- [ ] **3일차 · 7~9강 + 13~14강** — EDA·전처리·feature보다 split·누수·metric을 우선한다.
- [ ] **4일차 · 10~12강 + 15강** — 고전 ML baseline과 tuning budget을 짧게 복습한다.
- [ ] **5일차 · 16~18강** — PyTorch 데이터·모델·학습·loss 계약을 빈 화면에서 구현한다.
- [ ] **6일차 · 19~25강** — CNN·시계열·Transformer·생성모델의 대표 shape와 loss를 비교한다.
- [ ] **7일차 · 26~27강** — Process와 표형 Problem을 제한 시간 안에 제출 파일까지 완주한다.
- [ ] **8일차 · 28~32강** — 시계열/이미지 분기, 제출 검증, 필기와 Process 모의를 수행한다.
- [ ] **9일차 · 33~34강** — 170분 통합 모의 후 오답과 최종 체크만 복습한다.

## 트랙 C — 마지막 3일 점검

이 트랙은 처음 학습하는 사람에게 적합하지 않다. 앞 범위를 한 번 이상 학습하고 실행형 모의를 해본 경우에만 쓴다.

- [ ] **D-3** — 필기 20문항 50분 + Process 8문항. 틀린 개념을 13·14·17·18·21·26강에 연결한다.
- [ ] **D-2** — Problem 1문항을 제한 시간 안에 실행하고 저장 파일을 reload한다. 새 모델은 추가하지 않는다.
- [ ] **D-1** — 공식 공지·환경·금지행위·접속 준비를 재확인하고, 오답표와 제출 검사 함수만 복습한다.

## 통합 모의 운영안

아래 시간표는 **비공식 기본안**이다. Process 난도와 Problem 데이터에 따라 전환하되, Problem의 첫 유효 제출을 끝까지 미루지 않는다.

- 0~8분: 전체 문제, 변수·파일·metric·제출 위치를 기록한다.
- 8~55분: 확실한 Process부터 풀고 저장한다.
- 55~65분: Problem 데이터 계약과 가장 빠른 baseline을 만든다.
- 65~125분: Problem 검증과 모델을 한 번만 개선한다.
- 125~150분: 남은 Process와 자가 edge-case test를 처리한다.
- 150~163분: 최종 학습·예측·저장을 수행한다.
- 163~170분: Ctrl+S, 파일 존재, reload, shape, dtype, finite, row order, 정확한 파일명과 제출 상태를 확인한다.

전환 규칙: 55분까지 Process가 끝나지 않아도 Problem baseline으로 이동한다. 125분까지 개선 모델이 baseline을 이기지 못하면 baseline으로 돌아간다. 마지막 7분에는 새 코드를 쓰지 않는다.

## 개인 준비 상태 체크

아래 수치는 공식 합격선이 아니라 연습 안정성을 높이기 위한 개인 기준이다. 실제 커트라인과 채점 기준은 공개되지 않았다.

- [ ] 필기 20문항을 50분 안에 세 번 풀고, 같은 개념을 반복해서 틀리지 않는다.
- [ ] Process형 함수에 상수·NaN·비연속 index·빈 batch를 포함한 자가 edge-case test를 작성한다.
- [ ] 표형·시계열·이미지 중 두 유형 이상을 제출 파일까지 완주한다.
- [ ] PyTorch로 Dataset/DataLoader, nn.Module, 학습 루프, inference와 save/load를 빈 화면에서 작성한다.
- [ ] train-only fit, group/time split과 overlapping window 누수를 말로 설명한다.
- [ ] output·target·loss의 shape와 dtype 계약을 자동 assert한다.
- [ ] 저장한 예측 파일을 다시 읽어 shape·dtype·NaN/Inf·row order를 검사한다.
- [ ] 170분 통합 모의를 최소 두 번 완료하고 시간 기록을 남긴다.
- [ ] 시험 직전 공식 환경·금지행위·저장·제출 안내를 다시 확인한다.

## 최우선 공식 링크

- 2026 HDAT-DS 매뉴얼: https://hdat.gitbook.io/2026-hdat-ds
- HDAT-DS 공식 연습문제 안내: https://exam.hyundai-ngv.com/practice/13567
- PyTorch Learn the Basics: https://docs.pytorch.org/tutorials/beginner/basics/intro.html
- scikit-learn common pitfalls: https://scikit-learn.org/1.5/common_pitfalls.html
