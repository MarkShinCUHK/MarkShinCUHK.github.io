"""템플릿 호출 매뉴얼: basics. 같은 폴더에 hdat_templates.py를 두세요.
비공식 합성 연습. 원문: /templates/ — 문서의 실행 셀에서 자동 생성됩니다.
"""

import numpy as np
import hdat_templates as h

print(h.__file__)  # 지금 불러온 파일의 실제 위치
print(hasattr(h, "MLP"))  # PyTorch가 설치되어 있다면 True

X_train = np.array([[10.0, 1.0], [20.0, 2.0]], dtype=np.float32)
feature_count = X_train.shape[1]
model = h.MLP(n_features=feature_count, out_dim=1, hidden=(8, 4))
print(X_train.shape, feature_count)  # (2, 2) 2
