"""템플릿 호출 매뉴얼: autoencoder. 같은 폴더에 hdat_templates.py를 두세요.
비공식 합성 연습. 원문: /templates/ — 문서의 실행 셀에서 자동 생성됩니다.
"""

import numpy as np
import torch
import hdat_templates as h

h.seed_everything(42)
torch.set_num_threads(1)
rng = np.random.default_rng(42)
X_normal = rng.normal(size=(24, 4)).astype(np.float32)
X_test = rng.normal(size=(5, 4)).astype(np.float32)
X_test[-1] += 5  # 합성 이상 표본. 실제 성능을 보장하는 데이터가 아님
tr = h.make_tensor_loader(X=X_normal[:16], y=X_normal[:16], task="regression", batch_size=8, shuffle=True)
va = h.make_tensor_loader(X=X_normal[16:], y=X_normal[16:], task="regression")
te = h.make_tensor_loader(X=X_test, task="regression")
model = h.Autoencoder(input_dim=4, latent_dim=2, hidden_dim=8)
model, history = h.train_torch_model(
    model=model, train_loader=tr, valid_loader=va,
    task="regression", epochs=2, device="cpu",
)
valid_reconstruction = h.predict_torch(model=model, loader=va, task="regression", device="cpu")
valid_error = ((X_normal[16:] - valid_reconstruction) ** 2).mean(axis=1)
threshold = np.quantile(valid_error, 0.95)
reconstruction = h.predict_torch(model=model, loader=te, task="regression", device="cpu")
anomaly_score = ((X_test - reconstruction) ** 2).mean(axis=1)
anomaly_label = (anomaly_score > threshold).astype(np.int64)
print(reconstruction.shape, anomaly_score.shape)  # (5, 4) (5,)
assert np.isfinite(anomaly_score).all()
