import numpy as np
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score

def summarize_clusters(X, labels):
    labels = np.asarray(labels)
    keep = labels != -1
    n = int(keep.sum())
    k = len(np.unique(labels[keep]))
    noise_ratio = float((~keep).mean())
    # silhouette는 평가 표본에 군집이 2개 이상, 표본수보다 적게 있어야 한다.
    score = silhouette_score(X[keep], labels[keep]) if 2 <= k < n else None
    return {"clusters": k, "noise_ratio": noise_ratio,
            "silhouette_without_noise": score}

X, _ = make_moons(n_samples=240, noise=0.06, random_state=31)
X = StandardScaler().fit_transform(X)
for name, model in (
    ("k-means", KMeans(n_clusters=2, n_init=10, random_state=31)),
    ("DBSCAN", DBSCAN(eps=0.30, min_samples=5)),
):
    report = summarize_clusters(X, model.fit_predict(X))
    assert 0 <= report["noise_ratio"] <= 1
    print(name, report)

# 전부 noise와 단일 군집에서는 점수를 만들지 않는 계약을 검사한다.
all_noise = DBSCAN(eps=0.01, min_samples=len(X)+1).fit_predict(X)
assert summarize_clusters(X, all_noise)["silhouette_without_noise"] is None
assert summarize_clusters(X, np.zeros(len(X), dtype=int))[
    "silhouette_without_noise"] is None
print("군집 퇴화 조건 검사 통과")
