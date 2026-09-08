"""템플릿 호출 매뉴얼: image. 같은 폴더에 hdat_templates.py를 두세요.
비공식 합성 연습. 원문: /templates/ — 문서의 실행 셀에서 자동 생성됩니다.
"""

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
