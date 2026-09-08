"""템플릿 호출 매뉴얼: timeseries. 같은 폴더에 hdat_templates.py를 두세요.
비공식 합성 연습. 원문: /templates/ — 문서의 실행 셀에서 자동 생성됩니다.
"""

import numpy as np
import pandas as pd
import torch
import hdat_templates as h

h.seed_everything(42)
torch.set_num_threads(1)
df = pd.DataFrame({
    "time": pd.date_range("2026-01-01", periods=40, freq="s"),
    "temperature": np.linspace(10, 20, 40),
    "pressure": np.sin(np.arange(40) / 5),
    "next_value": np.linspace(20, 30, 40),
})
df = df.sort_values("time", kind="stable").reset_index(drop=True)
features = df[["temperature", "pressure"]].to_numpy(dtype=np.float32)
targets = df[["next_value"]].to_numpy(dtype=np.float32)
train_part, valid_part, cut = h.split_raw_time_then_window(
    features=features, targets=targets, train_ratio=0.7,
    lookback=4, horizon=1, stride=1, gap=0,
    assume_sorted=True, split_mode="forecast_origin", label_delay=0,
)
X_train_raw, y_train, train_target_rows = train_part
X_valid_raw, y_valid, valid_target_rows = valid_part
scaler = h.fit_scale_3d(X_train=X_train_raw)
X_train = h.transform_scale_3d(X=X_train_raw, scaler=scaler)
X_valid = h.transform_scale_3d(X=X_valid_raw, scaler=scaler)
print(cut, X_train.shape, X_valid.shape)  # 28 (24, 4, 2) (11, 4, 2)
assert train_target_rows.max() < cut
assert (valid_target_rows - 1).min() >= cut

tr = h.make_tensor_loader(X=X_train, y=y_train, task="regression", batch_size=8, shuffle=True)
va = h.make_tensor_loader(X=X_valid, y=y_valid, task="regression", batch_size=8)
model = h.CNN1D(n_features=X_train.shape[2], out_dim=y_train.shape[1], channels=(8, 16), dropout=0)
model, history = h.train_torch_model(
    model=model, train_loader=tr, valid_loader=va,
    task="regression", epochs=2, device="cpu",
)
valid_pred = h.predict_torch(model=model, loader=va, task="regression", device="cpu")
assert valid_pred.shape == y_valid.shape == (11, 1)

# 모델을 새로 만드는 예: 아래 모델은 아직 학습되지 않았다.
rnn = h.SequenceRNN(n_features=2, out_dim=1, hidden_size=8, kind="lstm")
with torch.no_grad():
    assert rnn(torch.from_numpy(X_valid[:2])).shape == (2, 1)
