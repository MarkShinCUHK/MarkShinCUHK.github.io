import numpy as np

# 이미 train/valid로 구분된 독립 합성 데이터
train = np.array([[-2., 0.], [-1., 0.], [1., 0.], [2., 0.]])
valid = np.array([[3., 1.], [-3., -1.]])
mean = train.mean(axis=0)
centered = train - mean
u, singular, vt = np.linalg.svd(centered, full_matrices=False)
components = vt[:1]                    # [k,F] = [1,2]
z_train = centered @ components.T      # [N,k]
z_valid = (valid - mean) @ components.T
recon_train = z_train @ components + mean
recon_valid = z_valid @ components + mean
variance = singular**2 / (len(train) - 1)
assert np.allclose(recon_train, train)
assert z_valid.shape == (2, 1)
assert np.isclose(np.square(recon_valid - valid).mean(), 0.5)
assert np.isclose(variance.sum(), np.var(train, axis=0, ddof=1).sum())
print("PCA 복원 검사 통과", "valid MSE:", 0.5)
