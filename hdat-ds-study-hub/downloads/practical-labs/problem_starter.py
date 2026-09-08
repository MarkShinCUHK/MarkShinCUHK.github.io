"""아래 solve만 작성하세요. 정답을 먼저 열지 말고 문제별 매뉴얼의 순서로 채웁니다."""
import numpy as np
import torch
import hdat_templates as h
from lab_runtime import run_cli


def solve(case, data):
    # 1. 확인: print({k: (v.shape, str(v.dtype)) for k,v in data.items()})
    # 2. 문제 페이지에서 입력·정답·split·출력 계약을 적습니다.
    # 3. 해당 playbook의 전처리/모델/학습/추론 블록을 이 함수 안에 연결합니다.
    #    source API: h.MLP / h.CNN1D / h.SmallImageCNN / h.Autoencoder
    #    loaders: h.make_tensor_loader
    #    train: model, history = h.train_torch_model(..., score_fn=..., maximize=...)
    # 4. data의 valid로만 설정·threshold·epoch를 고릅니다.
    # 5. data['test_ids'] 순서로 prediction 배열만 반환합니다.
    #    B01/B04: 정수 (N,), B02/B06: float (N,1), B03/B07: float (N,3), B05: float (N,)
    # solutions/answer_key.npz는 풀이에서 읽지 않습니다.
    raise NotImplementedError(f"{case}: 문제별 수정 매뉴얼을 보고 solve를 완성하세요")


if __name__ == "__main__":
    run_cli(solve)
