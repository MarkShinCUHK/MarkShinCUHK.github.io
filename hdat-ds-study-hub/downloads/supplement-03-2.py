import time
import numpy as np
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.feature_selection import SelectFromModel
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, BaggingClassifier, VotingClassifier
from sklearn.metrics import f1_score

started = time.perf_counter()
X, y = make_classification(
    n_samples=540, n_features=12, n_informative=6, n_redundant=2,
    n_classes=3, n_clusters_per_class=1, weights=[0.55, 0.30, 0.15],
    class_sep=1.0, random_state=23,
)
# 인위적으로 단위를 다르게 하되 target은 전처리에 사용하지 않는다.
X *= np.logspace(0, 2, X.shape[1])
all_ids = np.arange(len(y))
dev_ids, test_ids = train_test_split(
    all_ids, test_size=0.2, stratify=y, random_state=23,
)
train_ids, valid_ids = train_test_split(
    dev_ids, test_size=0.25, stratify=y[dev_ids], random_state=24,
)
assert not (set(train_ids) & set(valid_ids))
assert not (set(dev_ids) & set(test_ids))
assert (len(train_ids), len(valid_ids), len(test_ids)) == (324, 108, 108)

knn = make_pipeline(StandardScaler(), KNeighborsClassifier(n_neighbors=7))
svm = make_pipeline(StandardScaler(), SVC(C=1.0, gamma="scale"))
models = {
    "scaled KNN": knn,
    "scaled SVM": svm,
    "PCA + KNN": make_pipeline(
        StandardScaler(), PCA(n_components=6), KNeighborsClassifier(7)),
    "LDA + KNN": make_pipeline(
        StandardScaler(), LinearDiscriminantAnalysis(n_components=2),
        KNeighborsClassifier(7)),
    "selected + KNN": make_pipeline(
        StandardScaler(),
        SelectFromModel(RandomForestClassifier(n_estimators=24, random_state=23),
                        threshold="median"),
        KNeighborsClassifier(7)),
    "bagged tree": BaggingClassifier(
        estimator=DecisionTreeClassifier(max_depth=5),
        n_estimators=8, random_state=23),
    "hard voting": VotingClassifier(
        estimators=[("knn", knn), ("svm", svm),
                    ("tree", DecisionTreeClassifier(max_depth=5, random_state=23))],
        voting="hard"),
}
folds = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=25).split(
    X[train_ids], y[train_ids]))
means = {}
for name, model in models.items():
    scores = cross_val_score(model, X[train_ids], y[train_ids], cv=folds,
                             scoring="f1_macro", error_score="raise")
    assert scores.shape == (3,) and np.isfinite(scores).all()
    means[name] = scores.mean()
    print(f"{name:16s} CV mean={scores.mean():.3f}, std={scores.std():.3f}")

chosen = max(means, key=means.get)  # 선택에는 train 내부 CV만 사용
selected = clone(models[chosen]).fit(X[train_ids], y[train_ids])
valid_pred = selected.predict(X[valid_ids])
assert valid_pred.shape == y[valid_ids].shape
print("CV로 고정한 모델:", chosen)
print("valid Macro-F1:", f1_score(y[valid_ids], valid_pred, average="macro"))

# 여기까지 선택을 고정한 뒤 마지막 평가를 한 번 수행한다.
final_model = clone(models[chosen]).fit(X[dev_ids], y[dev_ids])
test_pred = final_model.predict(X[test_ids])
print("final test Macro-F1:", f1_score(y[test_ids], test_pred, average="macro"))
print(f"총 실행 시간: {time.perf_counter() - started:.2f}초")
