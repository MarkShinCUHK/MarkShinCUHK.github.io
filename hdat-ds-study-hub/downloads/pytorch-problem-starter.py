"""HDAT 비공식 합성 연습: 모델 선택 → 학습 → 공식 지표 checkpoint → 출력 검사.

같은 폴더에 hdat_templates.py를 놓는다. python pytorch-problem-starter.py --case binary
공식 기출/데이터가 아니며 시험장 파일 반입·import 허용을 뜻하지 않는다.
CHANGE DATA의 작은 배열을 실제 데이터와 올바른 split으로 바꾸는 학습용 출발점.
한 번에 하나의 --case만 선택한다. VAE/GAN/segmentation 전용 루프는 지원하지 않는다.
"""
import argparse
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import f1_score, roc_auc_score
import hdat_templates as h


def run(case="regression", epochs=3, output=None):
    h.seed_everything(42)
    torch.set_num_threads(1)
    rng = np.random.default_rng(42)
    # CHANGE DATA: 합성 독립 샘플이므로 이 split을 사용. 실제 time/group에는 복사 금지.
    X = rng.normal(size=(120, 6)).astype("float32")
    task = case
    if case == "regression":
        y = (2 * X[:, :1] - X[:, 1:2]).astype("float32")
        model = h.MLP(n_features=6, out_dim=1, hidden=(16, 8))
    elif case == "rmsle":
        task = "regression"
        y = np.log1p(np.exp(X[:, :1])).astype("float32")  # log1p(raw_y)로 학습
        model = h.MLP(n_features=6, out_dim=1, hidden=(16, 8))
    elif case == "binary":
        y = (X[:, :1] > 0).astype("float32")  # CHANGE: (original_y == POS_LABEL)
        model = h.MLP(n_features=6, out_dim=1, hidden=(16, 8))
    elif case == "multiclass":
        y = np.argmax(X[:, :3], axis=1).astype("int64")
        model = h.MLP(n_features=6, out_dim=3, hidden=(16, 8))
    elif case == "multilabel":
        y = (X[:, :3] > 0).astype("float32")
        model = h.MLP(n_features=6, out_dim=3, hidden=(16, 8))
    elif case == "window":
        task = "regression"
        X = rng.normal(size=(120, 12, 3)).astype("float32")  # 독립 window를 합성
        y = X[:, -3:, :2].mean(axis=1)
        model = h.CNN1D(n_features=3, out_dim=2, channels=(8, 16))
    elif case == "image":
        task = "multiclass"
        raw = rng.integers(0, 256, size=(120, 16, 20, 3), dtype="uint8")
        X = h.prepare_numpy_images(raw, layout="NHWC", divide_255=True)
        y = (raw[:, :, :, 0].mean(axis=(1, 2)) > 127).astype("int64")
        model = h.SmallImageCNN(in_channels=3, out_dim=2)
    elif case == "anomaly":
        task = "regression"
        y = X.copy()  # 합성 정상 train/valid. test만 일부 이상치를 넣는다.
        X[108:] += 4
        model = h.Autoencoder(input_dim=6, latent_dim=2, hidden_dim=16)
    else:
        raise ValueError(case)

    # CHANGE SPLIT: 0:80 train, 80:100 validation, 100:120 test.
    # 실제 표형 scaler/imputer는 train에만 fit. 합성 데이터는 이미 float·유한값.
    tr = h.make_tensor_loader(X[:80], y[:80], task, batch_size=16, shuffle=True)
    va = h.make_tensor_loader(X[80:100], y[80:100], task, batch_size=16)
    te = h.make_tensor_loader(X[100:], task=task, batch_size=16)

    # CHANGE METRIC: raw는 sigmoid/softmax 전 출력. 전체 valid에 한 번 평가한다.
    def official_score(truth, raw):
        if case == "rmsle":
            original_y = np.expm1(truth)
            original_pred = np.clip(np.expm1(raw), 0, None)
            return float(np.sqrt(np.mean((np.log1p(original_y) - np.log1p(original_pred)) ** 2)))
        if task == "regression":
            return float(np.sqrt(np.mean((truth - raw) ** 2)))
        if task == "binary":
            prob = torch.sigmoid(torch.from_numpy(raw)).numpy().reshape(-1)
            return roc_auc_score(truth.reshape(-1), prob)
        if task == "multiclass":
            return f1_score(truth.reshape(-1), raw.argmax(axis=1), average="macro", zero_division=0)
        return f1_score(truth, (raw >= 0).astype(int), average="macro", zero_division=0)

    # CHANGE LOSS: regression 기본은 MSE. MAE로 학습하려면 nn.L1Loss().
    model, history = h.train_torch_model(
        model, tr, va, task, epochs=epochs, patience=3, lr=1e-3, device="cpu",
        score_fn=official_score, maximize=(task != "regression"),
    )
    best_row = (history["monitor"].idxmax() if task != "regression"
                else history["monitor"].idxmin())
    print("selected epoch", int(history.loc[best_row, "epoch"]))

    # CHANGE OUTPUT: metric과 제출 형식은 별도. binary 예시는 양성 확률 제출.
    output_kind = "probability" if task == "binary" else "label" if task != "regression" else "value"
    pred = h.predict_torch(model, te, task, return_proba=(output_kind == "probability"), device="cpu")
    if case == "rmsle":
        pred = np.clip(np.expm1(pred), 0, None)
    if case == "anomaly":
        normal_recon = h.predict_torch(model, va, "regression", device="cpu")
        normal_error = np.mean((X[80:100] - normal_recon) ** 2, axis=1)
        threshold = np.quantile(normal_error, 0.99)  # 연습용. label이 있으면 valid F1 비교.
        score = np.mean((X[100:] - pred) ** 2, axis=1)
        output_kind = "anomaly_score"  # CHANGE: label이면 아래 분기
        pred = score if output_kind == "anomaly_score" else (score > threshold).astype(int)

    # CHANGE CONTRACT: 문제에 지정된 shape를 직접 적는다. pred.shape에서 추측하지 않는다.
    expected = (20,) if task == "multiclass" or case == "anomaly" else (20, 2) if case == "window" else (20, 3) if case == "multilabel" else (20, 1)
    assert pred.shape == expected, (pred.shape, expected)
    assert np.isfinite(pred).all()
    if output_kind == "probability":
        assert ((pred >= 0) & (pred <= 1)).all()
    if output is not None:
        path = Path(output)
        if path.exists():
            raise FileExistsError(f"기존 파일을 덮어쓰지 않습니다: {path}")
        if path.suffix != ".npy":
            raise ValueError("연습 출력 경로는 .npy여야 합니다")
        np.save(path, pred)
        np.testing.assert_array_equal(np.load(path, allow_pickle=False), pred)
    print(case, output_kind, pred.shape, pred.dtype, "finite OK")
    return pred, history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=["regression", "rmsle", "binary", "multiclass", "multilabel", "window", "image", "anomaly"], default="regression")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--output", help="선택: 아직 존재하지 않는 연습용 .npy 경로")
    args = parser.parse_args()
    if args.epochs < 1:
        parser.error("--epochs는 1 이상")
    run(args.case, args.epochs, args.output)
