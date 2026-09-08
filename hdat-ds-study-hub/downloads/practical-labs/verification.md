# 실전 연습팩 실행 검증 기록

검증일: 2026-09-08. 공식 시험 점수/합격선과 무관한 독립 합성 데이터 결과입니다.

- CPU, Python 3.9.6, NumPy 2.0.2, pandas 2.3.3, PyTorch 2.8.0, scikit-learn 1.5.2.
- Process 참고 구현 8개 공개 예제·경계 검사 통과. 빈 starter 8개는 의도대로 미구현 오류.
- Problem 7개 각각 2 epoch smoke test, 이후 실제 CLI 15 epoch(early stopping 가능) 학습·저장·reload·자가채점 통과.
- 잘못된 shape/ID/버전/복소수/NaN/라벨/확률 범위/음수 제출 거부, 정답과 예측 shape 불일치 거부.
- 완전 정답 배열에 대한 지표 oracle 검사, 회귀 float64 계산, 기존 파일 덮어쓰기 방지 검사 통과.
- Process 모델은 구조·파라미터·eval B=1·backward 검사. 공개 검사이며 완전한 정답 증명은 아닙니다.
- 모델과 입력 loader는 별도 test answer key를 읽지 않습니다. 학습 소스/데이터는 독립 제작했습니다.

아래는 참고 풀이가 valid로 선택한 모델을 고정한 뒤 test로 한 번 확인한 결과입니다.
환경에 따라 수치가 달라질 수 있습니다. 이 결과를 목표 점수·합격선·실제 난도라고 해석하지 마세요.
합성 이미지 등은 의도적으로 쉽게 학습되는 구조입니다.

| 문제 | 지표 | 관찰 점수 | 제출 shape |
|---|---|---|---|
| B01 | Macro-F1 ↑ | 0.861337 | [97] |
| B02 | RMSLE ↓ | 0.272612 | [97, 1] |
| B03 | 전체 원소 MSE ↓ | 31.042172 | [274, 3] |
| B04 | Accuracy ↑ | 1.000000 | [97] |
| B05 | ROC-AUC ↑ | 1.000000 | [127] |
| B06 | ROC-AUC ↑ | 0.987179 | [97, 1] |
| B07 | label별 Macro ROC-AUC ↑ | 0.987540 | [97, 3] |

재현: 저장소의 tests/labs-regression.py. 공개 ZIP의 각 solutions 구현도 --case/--output으로 개별 실행할 수 있습니다.
