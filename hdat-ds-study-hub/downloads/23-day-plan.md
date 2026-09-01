# HDAT-DS 2026: 23일 압축 학습계획

기준: 2026-08-18부터 공식 게시 회차인 2026-09-10까지  
권장 시간: 평일 2.5~3시간, 주말 5~6시간

## 매일 고정 루틴

- 필기 20문항 또는 개념 O/X 30개: 40분
- Process형 함수 2~3개: 50분
- Problem pipeline 또는 이론 보완: 60~90분
- 오답·코드 템플릿 정리: 20분

매일 결과물은 `오답 5개`, `재사용 함수 1개`, `shape/누수 체크 1개`를 남긴다.

## 1주차: 기본기와 명세 구현

### 8/18

- 공식 매뉴얼에서 시험 구성·금지사항·저장법 읽기
- 공식 필기 3문항을 7분 30초 안에 풀이
- NumPy/pandas의 shape, dtype, index, copy/view 복습

### 8/19

- 결측치, categorical encoding, standard/min-max/robust scaling
- Process: 지정 열 scaler 3종 구현
- 상수 열·NaN·비연속 index 테스트

### 8/20

- train/validation/test의 역할, leakage 유형
- stratified/group/time split 비교
- Process: split + metric 함수 구현

### 8/21

- 회귀 metric: MSE/RMSE/MAE/RMSLE
- 분류 metric: accuracy/F1-macro/ROC-AUC/PR-AUC
- 각 metric이 틀리게 평가할 수 있는 사례 만들기

### 8/22

- PIL crop/resize, NumPy image shape, normalize
- Process 공식 이미지 예시를 처음부터 다시 구현
- NCHW와 NHWC 변환 함수 작성

### 8/23

- 공식 Process 3문항을 60분 제한으로 다시 풀기
- 숨은 테스트 5개씩 직접 추가
- 공식 Problem 데이터를 읽고 shape·target·metric만 분석

### 8/24

- 첫 미니 모의: 필기 50분 + Process 90분
- 오답을 `개념/코딩/명세/시간` 네 종류로 분류

## 2주차: 표형 ML과 제출 완주

### 8/25

- 선형/로지스틱 회귀, KNN, Decision Tree, Random Forest
- 제공 `05.Lab/01` 첫 노트북을 누수 없는 pipeline으로 재작성

### 8/26

- SVM, scaling 필요성, PCA와 LDA
- `Pipeline`과 `ColumnTransformer` 빈 화면 구현

### 8/27

- 결측·이상치·불균형
- 제공 `table.ipynb`에서 baseline과 F1/PR 분석

### 8/28

- SMOTE, class weight, threshold의 차이
- SMOTE를 CV fold 안에 넣는 이유 설명·구현

### 8/29

- GridSearchCV/RandomizedSearchCV와 search budget
- 15분 안에 끝나는 작은 탐색 설정 만들기

### 8/30

- NGV 냉간단조 또는 자전거 문제를 170분 제한으로 풀기
- 시작 120분 안에 첫 제출 파일을 반드시 생성

### 8/31

- 1차 완전 모의: 필기 50분 + 실기 170분
- 성능보다 미제출·shape·NaN·파일명 실패를 먼저 교정

## 3주차: 딥러닝과 전체 범위

### 9/1

- perceptron, MLP, forward/backprop, activation/loss 조합
- PyTorch 또는 TensorFlow로 작은 MLP를 빈 화면 구현

### 9/2

- optimizer, initialization, L1/L2, dropout, BN, early stopping
- train/eval mode 차이를 코드로 확인

### 9/3

- CNN output size, parameter 수, padding/stride/pooling
- 공식 CNN Process 예시를 두 프레임워크 중 주력 하나로 재구현

### 9/4

- RNN/LSTM/GRU, window, many-to-one/many-to-many
- 작은 multi-output sequence regression baseline 작성

### 9/5

- AE/VAE/GAN/ResNet/Transformer 핵심 비교
- 각 모델의 목적, loss, 핵심 구조를 2문장씩 설명

### 9/6

- 공식 Problem과 유사한 시계열 회귀를 170분 제한으로 수행
- 선형/MLP baseline 후 1D CNN 또는 GRU 1개만 개선

### 9/7

- 2차 완전 모의: 필기 50분 + 실기 170분
- 10분 단위 시간 기록, 마지막 제출 검증 15분 고정

## 마무리

### 9/8

- 3차 실기 모의 170분
- 로컬 오픈북 노트북을 실제 시험처럼 검색해 사용
- 추가 설치 없이 모든 핵심 템플릿 실행 확인

### 9/9

- 공식 사전점검 완료 여부 확인
- 카메라 2대, 신분증, Chrome, VPN off, 듀얼 케이블 해제 확인
- 공식 안내 변경 여부 재확인
- 공식 필기 오답과 1쪽 공식만 복습; 새 모델 학습 금지

### 9/10 시험

- 07:30 접속 권장, 07:50 전 입실
- 필기는 35분 1차 풀이, 10분 보류 문항, 5분 검토
- 실기는 10분 전체 스캔, 60분 Process, 85분 Problem, 15분 저장·제출 검증을 기본값으로 시작
- 제출 전 `Ctrl+S`, 파일 존재, shape, dtype, NaN/Inf, row order, 파일명 확인

## 합격 준비 상태 체크

- [ ] 필기 20문항을 40분 내 80% 이상으로 3회 연속 해결
- [ ] 공식 Process 예시를 답안 없이 다시 작성
- [ ] 상수 열·NaN·index·dtype을 포함한 hidden test를 스스로 작성
- [ ] 표형 분류·회귀를 각각 90분 내 제출 파일까지 완주
- [ ] 시계열 다중출력 회귀를 170분 내 완주
- [ ] MLP/CNN/RNN 중 주력 프레임워크로 skeleton을 빈 화면에서 구현
- [ ] train-only fit과 time-aware validation을 설명하고 코드로 적용
- [ ] `Submission_problem.npy`의 shape·dtype·NaN을 자동 검사
- [ ] 170분 완전 모의 3회 완료
- [ ] 시험 환경 사전점검과 카메라 배치 완료
