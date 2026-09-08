"""템플릿 호출 매뉴얼: metrics. 같은 폴더에 hdat_templates.py를 두세요.
비공식 합성 연습. 원문: /templates/ — 문서의 실행 셀에서 자동 생성됩니다.
"""

import numpy as np
import torch
import hdat_templates as h

def my_rmse(y_true, raw_output):
    return float(np.sqrt(np.mean((y_true - raw_output) ** 2)))

def my_binary_auc(y_true, raw_output):
    prob = torch.sigmoid(torch.as_tensor(raw_output)).numpy().reshape(-1)
    return h.evaluate_predictions(
        y_true=y_true.reshape(-1), y_pred=(prob >= 0.5).astype(int),
        metric="auc", y_score=prob, task="binary",
    )

truth = np.array([[1.0], [3.0]], dtype=np.float32)
raw = np.array([[2.0], [5.0]], dtype=np.float32)
print(my_rmse(truth, raw))  # sqrt((1 + 4) / 2) = 약 1.5811
assert np.isclose(my_rmse(truth, raw), np.sqrt(2.5))
assert my_binary_auc(np.array([[0], [1]]), np.array([[-2.0], [2.0]])) == 1.0

y_binary_train = np.array([0, 0, 0, 1], dtype=np.float32)
positive = (y_binary_train == 1).sum()
negative = (y_binary_train == 0).sum()
assert positive > 0 and negative > 0
positive_weight = float(negative / positive)  # 3.0
binary_loss = h.make_torch_loss(task="binary", pos_weight=positive_weight)

y_multi_train = np.array([0, 0, 1, 2, 2, 2], dtype=np.int64)
counts = np.bincount(y_multi_train, minlength=3)
assert (counts > 0).all()
class_weights = (len(y_multi_train) / (3 * counts)).astype(np.float32)
multi_loss = h.make_torch_loss(task="multiclass", class_weight=class_weights)
print(positive_weight, class_weights)  # 3.0 [1.0, 2.0, 약 0.6667]

import tempfile
from pathlib import Path
import pandas as pd

sample_submission = pd.DataFrame({"id": ["T02", "T01"], "price": [0.0, 0.0]})
pred = np.array([[12.0], [25.0]], dtype=np.float32)
with tempfile.TemporaryDirectory(prefix="hdat-csv-") as folder:
    out = h.save_csv_submission(
        sample_submission=sample_submission,
        pred=pred,
        target_cols=["price"],
        path=Path(folder) / "practice-submission.csv",
    )
    assert out["id"].tolist() == ["T02", "T01"]
