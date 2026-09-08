"""정답 참고용. 먼저 problem_starter.py를 풀고 비교하세요. 각 CASE와 공통 RUN을 연결합니다."""
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
import hdat_templates as h
from lab_runtime import run_cli


def tabular(data):
    """수치 결측+범주형. dtype 문자열의 category는 numerical scaling 대상이 아니다."""
    frames = []
    for split in ("train", "valid", "test"):
        frame = pd.DataFrame(data[f"X_{split}"], columns=["pressure", "current", "temp", "vibration", "speed", "load"])
        frame["line"] = data[f"category_{split}"]
        frames.append(h.clean_tabular_values(frame))
    prep = h.make_preprocessor(frames[0], encoding="onehot", scale_numeric=True)
    out = [prep.fit_transform(frames[0]), prep.transform(frames[1]), prep.transform(frames[2])]
    return [np.asarray(v.toarray() if hasattr(v, "toarray") else v, dtype=np.float32) for v in out]


def mse(truth, raw):
    return float(np.mean((truth.astype(np.float64) - raw.astype(np.float64)) ** 2))


def solve(case, data, epochs=15):
    h.seed_everything(52)
    torch.set_num_threads(1)
    device = "cpu"  # EDIT: 실제 학습 환경에 맞추되 먼저 CPU 소규모 실행 확인
    ytr, yva = data.get("y_train"), data.get("y_valid")
    maximize, score_fn = False, mse
    target_scaler = None

    # CASE B01: playbook 08 + 06 전처리. 원 label 10/20/40 → CE index 0/1/2.
    if case == "B01":
        Xtr, Xva, Xte = tabular(data)
        labels = np.array([10, 20, 40])
        ytr = np.searchsorted(labels, ytr)
        yva = np.searchsorted(labels, yva)
        task = "multiclass"
        model = h.MLP(Xtr.shape[1], 3, hidden=(32, 16), dropout=0)
        maximize = True
        score_fn = lambda truth, raw: f1_score(truth, raw.argmax(1), labels=[0, 1, 2], average="macro", zero_division=0)

    # CASE B02: playbook 06. raw 양수 target → log1p 학습 → 원 단위 복원.
    elif case == "B02":
        Xtr, Xva, Xte = tabular(data)
        assert np.min(ytr) >= 0 and np.min(yva) >= 0
        ytr, yva = np.log1p(ytr), np.log1p(yva)
        task = "regression"
        model = h.MLP(Xtr.shape[1], 1, hidden=(32, 16), dropout=0)
        def score_fn(truth, raw):
            restored = np.clip(np.expm1(raw), 0, None)
            return float(np.sqrt(np.mean((truth - np.log1p(restored)) ** 2)))

    # CASE B03: playbook 10→09→06. ends는 입력 끝. target은 ends+6.
    elif case == "B03":
        raw = data["raw_X"]
        L, H = 12, 6
        def window(ends):
            return np.stack([raw[e-L+1:e+1] for e in ends]).astype(np.float32)
        Xtr, Xva, Xte = [window(data[f"ends_{s}"]) for s in ("train", "valid", "test")]
        assert (data["ends_train"] + H).max() < data["ends_valid"].min()
        assert (data["ends_valid"] + H).max() < data["ends_test"].min()
        scale = h.fit_scale_3d(Xtr)
        Xtr, Xva, Xte = [h.transform_scale_3d(v, scale) for v in (Xtr, Xva, Xte)]
        target_scaler = StandardScaler().fit(ytr)
        ytr, yva = [target_scaler.transform(v).astype(np.float32) for v in (ytr, yva)]
        task = "regression"
        model = h.CNN1D(raw.shape[1], 3, channels=(16, 32), dropout=0)
        def score_fn(truth, raw):
            return mse(target_scaler.inverse_transform(truth), target_scaler.inverse_transform(raw))

    # CASE B04: playbook 11→08. NHWC uint8 → NCHW float, CE는 raw logits.
    elif case == "B04":
        Xtr, Xva, Xte = [h.prepare_numpy_images(data[f"X_{s}"], "NHWC", True) for s in ("train", "valid", "test")]
        task = "multiclass"
        model = h.SmallImageCNN(3, 3)
        maximize = True
        score_fn = lambda truth, raw: float(np.mean(truth.reshape(-1) == raw.argmax(1)))

    # CASE B05: playbook 12. 학습·early stopping은 정상만. 제출은 error score.
    elif case == "B05":
        scale = StandardScaler().fit(data["X_train"])
        Xtr, Xva, Xte = [scale.transform(data[f"X_{s}"]).astype(np.float32) for s in ("train", "valid", "test")]
        ytr, yva = Xtr.copy(), Xva.copy()
        task = "regression"
        model = h.Autoencoder(Xtr.shape[1], latent_dim=2, hidden_dim=16)

    # CASE B06: playbook 07. fault=2, normal=5. P(label=2)를 출력.
    elif case == "B06":
        Xtr, Xva, Xte = tabular(data)
        ytr, yva = [(v == 2).astype(np.float32).reshape(-1, 1) for v in (ytr, yva)]
        task = "binary"
        model = h.MLP(Xtr.shape[1], 1, hidden=(32, 16), dropout=0)
        maximize = True
        score_fn = lambda truth, raw: roc_auc_score(truth.reshape(-1), torch.sigmoid(torch.from_numpy(raw)).numpy().reshape(-1))

    # CASE B07: playbook 08. 세 개 독립 0/1 indicator. softmax 사용 금지.
    elif case == "B07":
        Xtr, Xva, Xte = tabular(data)
        task = "multilabel"
        model = h.MLP(Xtr.shape[1], 3, hidden=(32, 16), dropout=0)
        maximize = True
        score_fn = lambda truth, raw: roc_auc_score(truth, torch.sigmoid(torch.from_numpy(raw)).numpy(), average="macro")
    else:
        raise ValueError(case)

    # COMMON RUN: case 분기 이후 공통. 어떤 전처리도 test에서 fit하지 않는다.
    tr = h.make_tensor_loader(Xtr, ytr, task, batch_size=32, shuffle=True)
    va = h.make_tensor_loader(Xva, yva, task, batch_size=64)
    te = h.make_tensor_loader(Xte, task=task, batch_size=64, shuffle=False)
    model, history = h.train_torch_model(model, tr, va, task, epochs=epochs, patience=5,
        lr=0.003, device=device, score_fn=score_fn, maximize=maximize)
    best = history.monitor.max() if maximize else history.monitor.min()
    print(f"validation monitor={best:.6f}; test 정답으로 선택하지 않았습니다")
    pred = h.predict_torch(model, te, task, return_proba=case in {"B06", "B07"}, device=device)

    # SUBMISSION ADAPTER: 같은 prediction 이름이어도 의미와 모양이 다르다.
    if case == "B01":
        pred = labels[pred].astype(np.int64)
    elif case == "B02":
        pred = np.clip(np.expm1(pred), 0, None)
    elif case == "B03":
        pred = target_scaler.inverse_transform(pred).astype(np.float32)
    elif case == "B05":
        pred = np.mean((Xte - pred) ** 2, axis=1)  # 이상=1일 확률이 아니라 큰 값=이상
    return pred


if __name__ == "__main__":
    run_cli(solve)
