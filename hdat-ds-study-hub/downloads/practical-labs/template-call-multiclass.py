"""템플릿 호출 매뉴얼: multiclass. 같은 폴더에 hdat_templates.py를 두세요.
비공식 합성 연습. 원문: /templates/ — 문서의 실행 셀에서 자동 생성됩니다.
"""

import numpy as np
import torch
import hdat_templates as h

h.seed_everything(42)
torch.set_num_threads(1)
X = np.arange(72, dtype=np.float32).reshape(24, 3) / 72
classes = np.array(["정상", "마모", "파손"])  # 이 예제에서 정한 고정 순서
label_to_index = {name: i for i, name in enumerate(classes)}
original_y = np.tile(classes, 8)
y = np.array([label_to_index[name] for name in original_y], dtype=np.int64)
train_loader = h.make_tensor_loader(X=X[:15], y=y[:15], task="multiclass", batch_size=5, shuffle=True)
valid_loader = h.make_tensor_loader(X=X[15:21], y=y[15:21], task="multiclass", batch_size=6)
test_loader = h.make_tensor_loader(X=X[21:], task="multiclass")
model = h.MLP(n_features=X.shape[1], out_dim=len(classes), hidden=(8,), dropout=0)
model, history = h.train_torch_model(
    model=model, train_loader=train_loader, valid_loader=valid_loader,
    task="multiclass", epochs=2, device="cpu",
)
pred_index = h.predict_torch(model=model, loader=test_loader, task="multiclass", device="cpu")
pred_name = classes[pred_index]
prob = h.predict_torch(model=model, loader=test_loader, task="multiclass", return_proba=True, device="cpu")
print(pred_index.shape, prob.shape, pred_name)  # (3,) (3, 3), 원래 문자열로 복원한 정답
assert np.allclose(prob.sum(axis=1), 1)
