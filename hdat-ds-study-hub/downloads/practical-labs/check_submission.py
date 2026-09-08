"""--score 없이는 형식만 검사. --score는 공개된 자가채점용 정답을 읽는다."""
import argparse
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from lab_runtime import ROOT, CASES, contract


def score(case, truth, pred):
    truth, pred = np.asarray(truth), np.asarray(pred)
    if truth.shape != pred.shape:
        raise ValueError("정답과 예측 shape가 다릅니다. broadcasting 채점은 금지합니다")
    if case in {"B02", "B03"}:
        truth, pred = truth.astype(np.float64), pred.astype(np.float64)
    if case == "B01":
        return "Macro-F1 ↑", float(f1_score(truth, pred, labels=[10, 20, 40], average="macro", zero_division=0))
    if case == "B02":
        return "RMSLE ↓", float(np.sqrt(np.mean((np.log1p(truth) - np.log1p(pred)) ** 2)))
    if case == "B03":
        return "전체 원소 MSE ↓", float(np.mean((truth - pred) ** 2))
    if case == "B04":
        return "Accuracy ↑", float(accuracy_score(truth, pred))
    if case in {"B05", "B06"}:
        target = (truth == 2).astype(int) if case == "B06" else truth
        return "ROC-AUC ↑", float(roc_auc_score(target.reshape(-1), pred.reshape(-1)))
    if case == "B07":
        return "label별 Macro ROC-AUC ↑", float(roc_auc_score(truth, pred, average="macro"))
    raise ValueError(case)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--file", required=True)
    parser.add_argument("--score", action="store_true")
    args = parser.parse_args()
    with np.load(args.file, allow_pickle=False) as saved:
        pred = contract(args.case, saved["ids"], saved["prediction"], saved["dataset_id"])
        dataset_id = str(saved["dataset_id"])
    print("제출 계약 통과 — 성능 합격을 의미하지 않습니다")
    if args.score:
        with np.load(ROOT / "solutions" / "answer_key.npz", allow_pickle=False) as answers:
            if str(answers[args.case + "_dataset_id"]) != dataset_id:
                raise ValueError("정답 키와 데이터 버전이 다릅니다")
            name, value = score(args.case, answers[args.case], pred)
        if not np.isfinite(value):
            raise ValueError("평가지표가 유한하지 않습니다. 지나치게 큰 값·dtype를 점검하세요")
        print(f"자가채점 {name}: {value:.6f}")
        print("공개 합성 정답으로 계산한 연습 점수입니다. HDAT 합격선과 무관합니다.")
