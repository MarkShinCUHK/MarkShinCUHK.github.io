"""템플릿 호출 매뉴얼: multilabel. 같은 폴더에 hdat_templates.py를 두세요.
비공식 합성 연습. 원문: /templates/ — 문서의 실행 셀에서 자동 생성됩니다.
"""

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
