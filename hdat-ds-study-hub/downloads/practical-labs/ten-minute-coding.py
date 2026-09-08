# -*- coding: utf-8 -*-
# HDAT 독자 입문 문제. Jupyter에서 문제별로 실행하세요.
# TODO를 구현하기 전 검사 실패는 정상입니다.
# 해설은 웹 연습실 /practice/code/에서 단계적으로 확인하세요.

# %% 문제 1: 조건에 맞는 값만 반환하기
# 점수 리스트에서 기준 이상인 점수만 원래 순서대로 반환하세요. 원본 리스트를 바꾸지 마세요. 빈 입력에는 빈 리스트를 반환합니다.
def select_scores(scores, minimum):
    # TODO: 새 리스트를 만들어 반환하세요.
    pass

# %% 검사 1
original = [50, 70, 90, 70]
assert select_scores(original, 70) == [70, 90, 70]
assert select_scores([], 70) == []
assert select_scores([69, 70], 70) == [70]
assert original == [50, 70, 90, 70]
print('통과: 경계값·빈 입력·원본 보존')

# %% 문제 2: 각 열의 평균을 빼기
# 2차원 수치 배열을 받아 각 열의 평균을 뺀 배열을 반환하세요. 입력과 shape가 같아야 하고 원본을 수정하면 안 됩니다. 이번 문제는 비어 있지 않은 2차원 배열만 입력됩니다.
import numpy as np

def center_columns(values):
    # TODO: 열별 평균을 구하고 빼세요.
    pass

# %% 검사 2
values = np.array([[1., 10.], [3., 30.], [5., 50.]], dtype=np.float32)
before = values.copy()
result = center_columns(values)
assert isinstance(result, np.ndarray)
assert result.shape == (3, 2)
np.testing.assert_allclose(result, [[-2., -20.], [0., 0.], [2., 20.]])
np.testing.assert_allclose(result.mean(axis=0), [0., 0.], atol=1e-6)
np.testing.assert_array_equal(values, before)
np.testing.assert_allclose(center_columns(np.array([[2., 8.]])), [[0., 0.]])
print('통과: 열별 계산·shape·단일 행·원본 보존')

# %% 문제 3: 표를 모델 입력으로 바꾸기
# speed가 50 이상인 행만 고르고 ['temperature', 'speed'] 순서로 float32 CPU Tensor를 반환하세요. 해당 열은 결측치 없는 숫자라고 가정합니다. 행 순서와 원본 표를 보존하며, 조건에 맞는 행이 없으면 shape (0, 2)여야 합니다.
import numpy as np
import pandas as pd
import torch

def make_features(frame):
    # TODO: 행 선택 → 열 순서 고정 → NumPy → Tensor
    pass

# %% 검사 3
frame = pd.DataFrame({'speed': [40, 50, 70], 'temperature': [20, 22, 30]}, index=[9, 3, 8])
before = frame.copy(deep=True)
result = make_features(frame)
assert isinstance(result, torch.Tensor)
assert result.shape == (2, 2)
assert result.dtype == torch.float32 and result.device.type == 'cpu'
torch.testing.assert_close(result, torch.tensor([[22., 50.], [30., 70.]]))
assert torch.isfinite(result).all()
assert make_features(frame.iloc[:1]).shape == (0, 2)
pd.testing.assert_frame_equal(frame, before)
print('통과: 조건·열 순서·shape·dtype·원본 보존')
