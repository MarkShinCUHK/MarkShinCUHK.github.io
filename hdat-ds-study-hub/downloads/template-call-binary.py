"""템플릿 호출 매뉴얼: binary. 같은 폴더에 hdat_templates.py를 두세요.
비공식 합성 연습. 원문: /templates/ — 문서의 실행 셀에서 자동 생성됩니다.
"""

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
