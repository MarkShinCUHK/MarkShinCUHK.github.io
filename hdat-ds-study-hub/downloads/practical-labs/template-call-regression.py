"""템플릿 호출 매뉴얼: regression. 같은 폴더에 hdat_templates.py를 두세요.
비공식 합성 연습. 원문: /templates/ — 문서의 실행 셀에서 자동 생성됩니다.
"""

import tempfile
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import hdat_templates as h

h.seed_everything(seed=42)
torch.set_num_threads(1)
train = pd.DataFrame({
    "age": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
    "distance": [12, 25, 31, 48, 55, 63, 74, 82, 93, 108, 117, 130],
    "price": [28, 27, 25, 24, 22, 21, 19, 18, 16, 14, 13, 11],
})
test = pd.DataFrame({"age": [3, 8], "distance": [35, 86]})
feature_cols = ["age", "distance"]
target_cols = ["price"]
h.assert_frame_contract(train=train, test=test, target_cols=target_cols)
X = train[feature_cols]
y = train[target_cols].to_numpy(dtype=np.float32)
print(X.shape, y.shape, test.shape)  # (12, 2) (12, 1) (2, 2)

train_idx, valid_idx = h.safe_train_valid_indices(
    X=X, y=y, task="regression", valid_size=0.25, seed=42,
)
prep = h.make_preprocessor(
    X=X.iloc[train_idx], encoding="ordinal", scale_numeric=True,
)
X_train = prep.fit_transform(X.iloc[train_idx]).astype(np.float32)
X_valid = prep.transform(X.iloc[valid_idx]).astype(np.float32)
X_test = prep.transform(test[feature_cols]).astype(np.float32)
y_train = y[train_idx]
y_valid = y[valid_idx]
h.print_shapes(X_train=X_train, y_train=y_train, X_valid=X_valid, X_test=X_test)
assert X_train.shape == (9, 2)
assert y_train.shape == (9, 1)

train_loader = h.make_tensor_loader(
    X=X_train, y=y_train, task="regression", batch_size=4, shuffle=True,
)
valid_loader = h.make_tensor_loader(
    X=X_valid, y=y_valid, task="regression", batch_size=4, shuffle=False,
)
test_loader = h.make_tensor_loader(
    X=X_test, y=None, task="regression", batch_size=4, shuffle=False,
)
xb, yb = next(iter(train_loader))
print(xb.shape, yb.shape)  # torch.Size([4, 2]) torch.Size([4, 1])
print(xb.dtype, yb.dtype)  # torch.float32 torch.float32

model = h.MLP(
    n_features=X_train.shape[1],
    out_dim=y_train.shape[1],
    hidden=(16, 8),
    dropout=0.0,
)
model, history = h.train_torch_model(
    model=model,
    train_loader=train_loader,
    valid_loader=valid_loader,
    task="regression",
    epochs=3,
    lr=0.01,
    patience=2,
    device="cpu",
)
print(history[["epoch", "train_loss", "valid_loss", "monitor"]])

pred = h.predict_torch(
    model=model, loader=test_loader, task="regression", device="cpu",
)
print(pred.shape)  # (2, 1): 테스트 2행 × 가격 1개
h.validate_prediction_array(pred=pred, expected_shape=(len(test), 1))

# 연습 파일은 임시 폴더에 저장했다가 자동 정리한다.
with tempfile.TemporaryDirectory(prefix="hdat-call-") as folder:
    path = Path(folder) / "practice-pred.npy"
    saved = h.save_npy_submission(
        pred=pred, path=path, expected_shape=(len(test), 1), dtype=np.float32,
    )
    assert np.load(path, allow_pickle=False).shape == (2, 1)
