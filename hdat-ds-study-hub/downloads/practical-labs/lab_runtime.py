"""학습팩 공통 입출력. 정답 파일은 이 모듈에서 읽지 않는다."""
from pathlib import Path
import argparse
import json
import numpy as np

ROOT = Path(__file__).resolve().parent
CASES = [f"B{i:02}" for i in range(1, 8)]


def load_case(case):
    if case not in CASES:
        raise ValueError(f"문제는 {CASES} 중 하나입니다")
    with np.load(ROOT / "data" / f"{case}.npz", allow_pickle=False) as saved:
        return {key: saved[key] for key in saved.files}


def contract(case, ids, prediction, dataset_id=None):
    """정답을 열지 않고 제출 shape/순서/종류를 검사한다."""
    data = load_case(case)
    expected_ids = data["test_ids"]
    specs = json.loads((ROOT / "contracts.json").read_text(encoding="utf-8"))
    spec = specs[case]
    if str(data["dataset_id"]) != spec["dataset_id"] or (dataset_id is not None and str(dataset_id) != spec["dataset_id"]):
        raise ValueError("데이터 버전이 다릅니다. 같은 ZIP의 데이터·제출·정답을 사용하세요")
    pred = np.asarray(prediction)
    expected_shape = (len(expected_ids), *spec["tail_shape"])
    if not np.array_equal(ids, expected_ids):
        raise ValueError("test_ids와 제출 ids의 값·순서가 다릅니다")
    if pred.shape != expected_shape:
        raise ValueError(f"shape {pred.shape} → 요구 {expected_shape}; 임의 reshape로 숨기지 마세요")
    if not np.issubdtype(pred.dtype, np.number) or np.iscomplexobj(pred) or not np.isfinite(pred).all():
        raise ValueError("prediction은 유한한 실수 배열이어야 합니다")
    if spec["output"] == "label":
        if not np.issubdtype(pred.dtype, np.integer):
            raise ValueError("label은 정수 dtype이어야 합니다")
        if not set(np.unique(pred)).issubset(set(spec["labels"])):
            raise ValueError("허용되지 않은 label이 있습니다")
    elif spec["output"] == "probability":
        if not np.issubdtype(pred.dtype, np.floating):
            raise ValueError("확률은 float dtype이어야 합니다")
        if not ((pred >= 0) & (pred <= 1)).all():
            raise ValueError("확률 범위가 0~1이 아닙니다")
    elif not np.issubdtype(pred.dtype, np.floating):
        raise ValueError("회귀값·이상 score는 float dtype이어야 합니다")
    if spec.get("nonnegative") and np.any(pred < 0):
        raise ValueError("이 문제의 예측은 음수가 아니어야 합니다")
    return pred


def run_cli(solve):
    parser = argparse.ArgumentParser(description="합성 Problem 풀이 · 공식 시험 파일 형식 아님")
    parser.add_argument("--case", choices=CASES, required=True)
    parser.add_argument("--output", required=True, help="새 .npz 경로. 기존 파일은 보호합니다")
    args = parser.parse_args()
    output = Path(args.output)
    if output.exists():
        parser.error("이미 있는 파일입니다. 새 이름을 사용하세요")
    if output.suffix != ".npz":
        parser.error("출력은 .npz 파일이어야 합니다")
    data = load_case(args.case)
    prediction = contract(args.case, data["test_ids"], solve(args.case, data))
    output.parent.mkdir(parents=True, exist_ok=True)
    # exclusive create: 학습 중 다른 파일이 생겼어도 덮어쓰지 않는다.
    with output.open("xb") as handle:
        np.savez_compressed(handle, ids=data["test_ids"], prediction=prediction, dataset_id=data["dataset_id"])
    with np.load(output, allow_pickle=False) as saved:
        contract(args.case, saved["ids"], saved["prediction"], saved["dataset_id"])
    print(f"저장 및 reload 계약 검사 통과: {args.case}, {prediction.shape}")
    print("성능은 별도 check_submission.py --score로 한 번 확인하세요. 합격 판정이 아닙니다.")
