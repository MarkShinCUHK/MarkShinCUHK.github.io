교육자료 확장 보강 원고

이 원고는 압축해제된 2025년 교육자료에서 확인한 역량 중 기존 34강에서 설명이나 실습이 부족한 부분을 독립적으로 다시 설명한다. 아래 항목을 모두 공식 HDAT-DS 필수 구현이라고 선언하지 않는다. 공식 필수 범위는 별도 공식 대조표를 따른다. 고전 ML 실습에는 scikit-learn을 사용하고, 신경망 구현은 전부 PyTorch를 사용한다.

# 01. 다중입력·모델 결합·전용 학습루프

## 여러 입력을 합치는 모델과 여러 정답을 내는 모델

교육자료 확장. 선수 학습은 16~19강이며, 순서 모델은 21~22강을 먼저 공부한다.

다중입력 모델은 서로 다른 관측을 각각 특징 벡터로 바꾼 뒤 합친다. 예를 들어 이미지와 예측 시점에 이미 측정한 센서를 함께 쓸 수 있다. 반면 다중 head 모델은 하나의 표현에서 여러 종류의 정답을 예측한다. 두 구조를 함께 사용할 수도 있다.

가장 먼저 각 입력의 이용 가능 시점을 적는다. 분류 정답을 one-hot으로 바꾸어 입력에 넣으면 정답 누수다. 과거 정답이 feature인 문제도 있지만, 해당 과거 정답이 실제 추론 시점에 알려진다는 계약이 있어야 한다. 입력이 많아졌다는 사실만으로 새로운 정보가 생기지는 않는다.

아래 코드는 이미지와 센서에서 공통 표현을 만들고, 3개 class의 logits와 두 개 수치 예측을 반환한다. 분류 target은 `[B]` long, 회귀 target은 `[B,2]` float다. 숫자 class ID를 그대로 회귀 정답으로 사용하면 class 번호 사이 거리가 의미 있다는 가정을 추가한다. 여기서는 분류와 별개의 연속 측정값을 회귀 정답으로 정의한다.

```python
import torch
from torch import nn
from torch.nn import functional as F

class SensorImageMultiTask(nn.Module):
    def __init__(self, sensor_features=5, classes=3, targets=2):
        super().__init__()
        self.image_encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
        )
        self.sensor_encoder = nn.Sequential(
            nn.Linear(sensor_features, 6), nn.ReLU(),
        )
        self.shared = nn.Sequential(nn.Linear(14, 12), nn.ReLU())
        self.class_head = nn.Linear(12, classes)
        self.value_head = nn.Linear(12, targets)

    def forward(self, image, sensor):
        if image.ndim != 4 or sensor.ndim != 2:
            raise ValueError("image=[B,3,H,W], sensor=[B,F]가 필요합니다")
        if image.shape[0] != sensor.shape[0]:
            raise ValueError("입력 batch가 다릅니다")
        # [B,8] + [B,6] -> [B,14] -> [B,12]
        h = self.shared(torch.cat([
            self.image_encoder(image), self.sensor_encoder(sensor)
        ], dim=1))
        return self.class_head(h), self.value_head(h)

def multitask_loss(logits, values, class_target, value_target, alpha=0.2):
    if values.shape != value_target.shape:
        raise ValueError("회귀 output-target shape 불일치")
    ce = F.cross_entropy(logits, class_target)
    mse = F.mse_loss(values, value_target)
    return ce + alpha * mse, ce.detach(), mse.detach()

torch.manual_seed(7)
net = SensorImageMultiTask()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-3)
for batch_size in (1, 4):
    image = torch.randn(batch_size, 3, 16, 20)
    sensor = torch.randn(batch_size, 5)
    labels = torch.arange(batch_size) % 3
    targets = torch.randn(batch_size, 2)
    optimizer.zero_grad(set_to_none=True)
    logits, values = net(image, sensor)
    assert logits.shape == (batch_size, 3)
    assert values.shape == targets.shape
    loss, ce, mse = multitask_loss(logits, values, labels, targets)
    assert loss.ndim == 0 and torch.isfinite(loss)
    loss.backward()
    assert net.image_encoder[0].weight.grad is not None
    assert net.sensor_encoder[0].weight.grad is not None
    optimizer.step()
print("다중입력·다중head shape/backward 검사 통과")
```

이 검사는 연결과 역전파가 가능한지 확인한다. 무작위 합성 target을 두 번 학습했다고 예측 성능이 검증된 것은 아니다. 실제 학습에서는 같은 표본 ID의 이미지·센서·target을 Dataset에서 함께 반환하고, 같은 분할 인덱스로 나눈다. 개별 배열을 따로 shuffle하면 행 연결이 깨진다. 기존 `(X,y)` 공통 루프는 다중입력·다중head를 자동 지원하지 않으므로 `image,sensor,class_y,value_y`를 받아 각 loss를 계산하는 전용 루프로 바꾼다.

회귀 target의 단위가 크면 MSE가 CE를 압도할 수 있다. train 통계로 target을 표준화하고 `alpha`를 validation에서 선택한다. 각 head의 metric을 따로 기록하며, 제출할 head와 원 단위 복원 규칙도 별도로 정한다.

실습: 센서 feature를 5개에서 7개로 바꿀 때 첫 Linear의 입력만 바꾸어라. 이미지 encoder 출력 폭을 8에서 10으로 바꾸면 결합 Linear의 입력은 얼마인가? 회귀 target 두 열의 단위를 각각 1배, 100배로 바꾸고 loss 기여를 비교하라.

<details><summary>해설 보기</summary>

센서 encoder는 계속 6개 특징을 반환하므로 첫 변경에서 결합 폭은 14다. 이미지 특징을 10개로 바꾸면 10+6=16이다. 같은 상대 오차라도 target을 100배 키우면 제곱오차는 10,000배 커질 수 있다. 단위·정규화·loss 가중치가 필요한 이유다.

</details>

## CNN과 RNN을 한 모델에서 연결하기

교육자료 확장. 같은 입력을 CNN과 RNN으로 각각 처리해 특징을 결합하는 병렬 구조와, 매 프레임 CNN 특징을 RNN에 순서대로 넣는 직렬 구조를 구분한다. 첫 구조는 두 종류의 시간 표현을 함께 사용한다. 두 번째 구조는 이미지의 공간 특징을 먼저 줄이고 프레임 간 변화를 학습한다. 두 구조 모두 정답을 입력으로 요구하지 않는다.

```python
import torch
from torch import nn

class ParallelSensorFusion(nn.Module):
    def __init__(self, features=4, outputs=2):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv1d(features, 8, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.gru = nn.GRU(features, 6, batch_first=True)
        self.head = nn.Linear(14, outputs)

    def forward(self, x):                    # [B,T,4]
        local = self.cnn(x.transpose(1, 2)).squeeze(-1)  # [B,8]
        _, state = self.gru(x)               # [1,B,6]
        return self.head(torch.cat([local, state[-1]], dim=1))

class FrameSensorGRU(nn.Module):
    def __init__(self, sensors=4, outputs=2):
        super().__init__()
        self.frame_encoder = nn.Sequential(
            nn.Conv2d(3, 8, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
        )
        self.gru = nn.GRU(8 + sensors, 10, batch_first=True)
        self.head = nn.Linear(10, outputs)

    def forward(self, frames, sensors):      # [B,T,3,H,W], [B,T,4]
        b, t, c, h, w = frames.shape
        if sensors.shape[:2] != (b, t):
            raise ValueError("영상과 센서의 batch/time이 다릅니다")
        visual = self.frame_encoder(frames.reshape(b*t, c, h, w))
        visual = visual.reshape(b, t, 8)     # 시간 순서를 유지한다
        merged = torch.cat([visual, sensors], dim=-1)  # [B,T,12]
        _, state = self.gru(merged)          # [1,B,10]
        return self.head(state[-1])          # [B,2]

torch.manual_seed(11)
for b, t in ((1, 3), (4, 6)):
    sensors = torch.randn(b, t, 4)
    frames = torch.randn(b, t, 3, 12, 16)
    for model, inputs in (
        (ParallelSensorFusion(), (sensors,)),
        (FrameSensorGRU(), (frames, sensors)),
    ):
        pred = model(*inputs)
        assert pred.shape == (b, 2) and torch.isfinite(pred).all()
        pred.square().mean().backward()
        assert all(p.grad is not None for p in model.parameters())
print("CNN/RNN 결합 shape/backward 검사 통과")
```

영상 파일의 디렉터리 나열 순서는 label 순서를 보장하지 않는다. 파일명·timestamp·camera ID로 구성한 manifest를 기준으로 센서와 연결해야 한다. 앞·좌·우 카메라를 합칠 때는 동일 시점의 세 영상인지 검사한다. 움직이는 차량의 인접 프레임은 매우 비슷하므로 프레임 random split보다 주행 구간이나 시간 분할을 검토한다. 원본 영상을 전부 거대한 NumPy 배열로 만들기보다 Dataset이 필요한 프레임만 읽도록 설계한다.

실습: 직렬 모델에서 `[B,T,3,H,W]`를 `[B,3,T,H,W]`로 잘못 바꾸면 어디에서 계약이 깨지는가? CNN 특징의 T축을 평균한 뒤 GRU에 넣으면 무엇을 잃는가?

<details><summary>해설 보기</summary>

구현이 세 번째 축을 channel로 해석하므로 채널 수가 T로 잘못 읽힌다. 우연히 T=3이어도 의미가 바뀌므로 shape 통과만으로 안전하지 않다. T축을 먼저 평균하면 프레임 순서를 잃어 GRU가 원래의 시간 변화를 받을 수 없다.

</details>

## 전용 학습 루프 실습 A · GAN의 두 optimizer와 gradient 경로

### 무엇을 배우는 예제인가요?

GAN은 정답 `y` 하나에 대해 모델 하나를 학습하는 구조가 아닙니다. 생성자 G는 noise를 받아 가짜 데이터를 만들고, 판별자 D는 진짜와 가짜를 구별합니다. 따라서 `(X,y)`와 optimizer 하나를 받는 공통 supervised 학습 함수를 그대로 사용할 수 없습니다. 아래는 **교대 학습의 연결을 확인하는 2차원 합성 예제**이지, 실제 문제에 GAN을 반드시 사용하라는 뜻은 아닙니다.

한 번의 반복에서 다음 두 번의 갱신을 구분합니다.

- D 단계: 진짜에는 1, 가짜에는 0을 붙입니다. `G(z).detach()`로 G까지 이어지는 gradient 경로를 끊고 D만 갱신합니다.
- G 단계: D가 가짜를 진짜로 판단하도록 목표를 1로 둡니다. 이번에는 **G 출력에 detach를 적용하면 안 됩니다.** D의 parameter는 고정하되 D 연산을 통과해 G까지 gradient가 흘러야 합니다.

D는 확률이 아니라 raw logit 한 개를 반환합니다. `BCEWithLogitsLoss`에 sigmoid가 포함되어 있기 때문입니다. 이 예제의 real 데이터는 `[-1,1]` 범위이고, G 마지막의 `Tanh`도 같은 범위입니다. 실제 데이터가 `[0,1]` 또는 표준화된 무제한 연속값이면 출력 activation과 전처리도 그 계약에 맞춰 다시 정합니다.

### 복사하여 2 step 실행하기

```python
import torch
from torch import nn

torch.manual_seed(17)
device = torch.device("cpu")
G = nn.Sequential(nn.Linear(3, 8), nn.ReLU(), nn.Linear(8, 2), nn.Tanh()).to(device)
D = nn.Sequential(nn.Linear(2, 8), nn.LeakyReLU(0.2), nn.Linear(8, 1)).to(device)
g_opt = torch.optim.Adam(G.parameters(), lr=1e-3)
d_opt = torch.optim.Adam(D.parameters(), lr=1e-3)
bce = nn.BCEWithLogitsLoss()

def check_gradients(module):
    assert all(p.grad is not None and torch.isfinite(p.grad).all()
               for p in module.parameters())

G.train()
D.train()
for step in range(2):
    # 합성 real: 두 점 주변의 작은 구름. 실제 데이터 계약의 대체물이 아닙니다.
    side = torch.randint(0, 2, (16, 1), device=device).float() * 2 - 1
    real = (0.5 * side + 0.1 * torch.randn(16, 2, device=device)).clamp(-1, 1)
    assert real.shape == (16, 2) and real.min() >= -1 and real.max() <= 1

    # 1. D 갱신: G의 graph를 끊습니다. 지난 G gradient도 지워 검사합니다.
    g_opt.zero_grad(set_to_none=True)
    d_opt.zero_grad(set_to_none=True)
    fake_for_d = G(torch.randn(16, 3, device=device)).detach()
    real_logit, fake_logit = D(real), D(fake_for_d)
    d_loss = (bce(real_logit, torch.ones_like(real_logit))
              + bce(fake_logit, torch.zeros_like(fake_logit))) / 2
    assert torch.isfinite(d_loss)
    d_loss.backward()
    check_gradients(D)
    assert all(p.grad is None for p in G.parameters())  # D 단계에서 G로 역전파 안 됨
    d_opt.step()

    # 2. G 갱신: D parameter는 고정하지만 D 연산의 autograd는 유지합니다.
    d_opt.zero_grad(set_to_none=True)
    for p in D.parameters():
        p.requires_grad_(False)
    g_opt.zero_grad(set_to_none=True)
    fake_for_g = G(torch.randn(16, 3, device=device))  # detach 금지
    fooled_logit = D(fake_for_g)                    # no_grad/inference_mode 금지
    g_loss = bce(fooled_logit, torch.ones_like(fooled_logit))
    assert torch.isfinite(g_loss)
    g_loss.backward()
    check_gradients(G)
    assert all(p.grad is None for p in D.parameters())
    g_opt.step()
    for p in D.parameters():
        p.requires_grad_(True)  # 다음 D 단계에서 다시 학습
    assert all(torch.isfinite(p).all() for p in list(G.parameters()) + list(D.parameters()))
    print(f"step={step + 1} D_loss={d_loss.item():.4f} G_loss={g_loss.item():.4f}")

G.eval()
with torch.inference_mode():
    sample = G(torch.randn(5, 3, device=device))
assert sample.shape == (5, 2) and torch.isfinite(sample).all()
assert (sample >= -1).all() and (sample <= 1).all()
print("GAN 연결 검사 통과: shape·유한값·D/G gradient 경로")
```

### 해설과 재도전

`requires_grad_(False)`는 D의 parameter gradient를 막는 것이지 입력에 대한 미분까지 막는 것은 아닙니다. 그래서 G 단계에서도 D를 통과한 미분으로 G를 갱신할 수 있습니다. 반면 D 호출을 `torch.no_grad()`로 감싸거나 `fake_for_g.detach()`를 쓰면 G 학습 경로가 사라집니다. D 단계에서 detach를 지우면 현재 코드의 “G gradient가 없어야 한다” 검사가 실패합니다. 두 가지를 각각 바꿔 보고 왜 다른 단계에서 실패하는지 설명한 뒤 원래대로 되돌리세요.

이 작은 D/G에는 BatchNorm이나 Dropout이 없습니다. 그런 층을 추가하면 parameter 고정만으로 running statistics와 dropout 동작까지 고정되지는 않습니다. 각 단계의 `train()/eval()` 정책도 따로 설계해야 합니다. 또한 `eval()`만으로 gradient 계산이 꺼지지는 않습니다.

통과 기준은 **두 optimizer가 연결되어 있고, D 단계에서 G gradient가 차단되며, G 단계에서는 G gradient가 유한하게 계산되는 것**입니다. loss가 유한하거나 한 번 감소했다는 사실만으로 생성 품질·다양성·mode collapse 해결을 입증할 수 없습니다. 이 예제는 두 번만 갱신하므로 real 분포를 학습했다고 해석하지 마세요. 실제 생성 품질 평가는 별도의 검증 방법이 필요합니다.

## 전용 학습 루프 실습 B · Denoising AE의 noisy 입력과 clean 정답

### 일반 Autoencoder와 무엇이 다른가요?

일반 AE의 기본 실습은 `clean → clean` 복원입니다. Denoising AE는 학습용 입력에 noise를 추가하되 정답은 원래의 clean 데이터로 유지하는 **`noisy → clean`** 학습입니다. `noisy → noisy`를 학습하면 원래 신호를 복원하라는 목표를 주지 않은 것입니다.

G/D 두 모델이 필요한 GAN과 달리, denoising AE는 모델·optimizer가 하나입니다. **이미 noisy 입력과 clean 정답을 만들어 놓았다면** 공통 supervised 루프에도 연결할 수 있습니다. 다만 매 반복마다 새 noise를 만드는 증강, clean 정답 보존, 고정된 validation corruption을 명확하게 제어하려고 아래에서는 짧은 전용 루프를 씁니다. 모든 AE 문제가 denoising을 요구하는 것은 아닙니다.

### 복사하여 2 step 실행하기

```python
import torch
from torch import nn

torch.manual_seed(23)
device = torch.device("cpu")

# 같은 합성 생성 규칙으로 독립 train/valid 샘플을 만듭니다.
# projection은 데이터 생성 규칙이며 valid에서 학습한 전처리 통계가 아닙니다.
projection = torch.randn(2, 6, device=device)
def make_clean(n):
    return torch.tanh(torch.randn(n, 2, device=device) @ projection)

clean_train = make_clean(32)
clean_valid = make_clean(12)
clean_before = clean_train.clone()
noise_std = 0.15
def corrupt(clean):
    # 원본에 in-place로 더하지 않습니다. clean 정답을 그대로 보존합니다.
    return (clean + noise_std * torch.randn_like(clean)).clamp(-1, 1)

# 모델 비교 기준이 noise 재추첨 때문에 흔들리지 않도록 한 번 만들고 고정합니다.
noisy_valid = corrupt(clean_valid)
ae = nn.Sequential(nn.Linear(6, 8), nn.ReLU(), nn.Linear(8, 2),
                   nn.Linear(2, 8), nn.ReLU(), nn.Linear(8, 6)).to(device)
optimizer = torch.optim.Adam(ae.parameters(), lr=1e-3)
mse = nn.MSELoss()
best_value, best_state = float("inf"), None

for step in range(2):
    ae.train()
    noisy_train = corrupt(clean_train)  # 매 학습 step에는 새 noise
    optimizer.zero_grad(set_to_none=True)
    reconstruction = ae(noisy_train)
    assert reconstruction.shape == clean_train.shape == (32, 6)
    loss = mse(reconstruction, clean_train)  # 정답은 noisy_train이 아닙니다.
    assert torch.isfinite(loss)
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in ae.parameters())
    optimizer.step()
    assert torch.equal(clean_train, clean_before)

    ae.eval()
    with torch.inference_mode():
        denoised_valid = ae(noisy_valid)
        assert torch.isfinite(denoised_valid).all()
        valid_denoise_mse = mse(denoised_valid, clean_valid).item()
    # 선택 기준: noisy 입력에서 clean을 복원한 validation MSE
    if valid_denoise_mse < best_value:
        best_value = valid_denoise_mse
        best_state = {k: v.detach().clone() for k, v in ae.state_dict().items()}
    print(f"step={step + 1} train_denoise_mse={loss.item():.6f} "
          f"valid_denoise_mse={valid_denoise_mse:.6f}")

assert best_state is not None
ae.load_state_dict(best_state)
ae.eval()
with torch.inference_mode():
    restored = ae(noisy_valid)
    assert restored.shape == (12, 6) and torch.isfinite(restored).all()
    chosen_denoise_mse = mse(restored, clean_valid).item()
    # 보조 진단 두 개. checkpoint 선택 기준과 혼동하지 않습니다.
    identity_mse = mse(noisy_valid, clean_valid).item()  # 입력을 그대로 돌려주는 baseline
    clean_recon_mse = mse(ae(clean_valid), clean_valid).item()  # clean → clean 복원
assert abs(chosen_denoise_mse - best_value) < 1e-7
print("선택 모델 noisy→clean MSE:", chosen_denoise_mse)
print("무처리 noisy 입력 baseline MSE:", identity_mse)
print("보조 진단 clean→clean MSE:", clean_recon_mse)
print("Denoising 연결 검사 통과: clean 보존·gradient·출력·선택 checkpoint")
```

### 해설과 재도전

모델이 실제로 받아야 할 입력이 noisy라면 주된 검증도 `noisy_valid → clean_valid`로 해야 합니다. `clean_valid → clean_valid` 점수는 clean 입력을 얼마나 잘 보존하는지에 관한 **다른 보조 진단**입니다. 둘 다 MSE라는 이름을 쓰더라도 같은 측정이 아닙니다. `noisy_valid → noisy_valid` MSE 역시 denoising 성능을 재는 지표가 아닙니다.

위 loss는 모든 샘플·6개 특성에 걸친 평균 제곱 오차입니다. validation noise는 고정하고 train noise만 다시 만듭니다. 별도의 scaler가 필요한 실제 데이터에서는 train에만 fit한 같은 변환을 적용한 뒤 어느 단위에서 noise와 오차를 정의할지 정하세요. noise 크기를 실제 센서 오차와 무관하게 키우거나 다른 신호를 훼손하는 변환을 무조건 추가하지 않습니다.

재도전은 두 가지입니다. 먼저 정답을 `noisy_train`으로 잘못 바꾸면 어떤 학습 목표가 되는지 설명하세요. 다음으로 충분한 학습 후 선택 모델의 `noisy→clean MSE`가 무처리 baseline보다 작은지 비교하세요. 두 번째 실험에서도 noise 크기와 모델은 validation으로 선택하고, 별도 test가 있다면 마지막에 한 번만 평가합니다.

2 step 검사의 통과는 **입력/정답 연결·clean 원본 보존·유한 gradient·출력 shape·최상 validation checkpoint 복원**이 맞다는 뜻입니다. 아직 학습이 짧아 무처리 baseline보다 나쁠 수 있으며 그것 자체가 구현 오류는 아닙니다. 이 예제에서 **계약 검사 통과와 실제 denoising 품질 확보는 별개**입니다. 또한 denoising AE의 복원 오차를 바로 이상탐지의 정답 label이나 보정된 이상 확률이라고 해석하면 안 됩니다.

참고: [PyTorch 공식 GAN 학습 튜토리얼](https://docs.pytorch.org/tutorials/beginner/dcgan_faces_tutorial.html). 위 코드는 원문의 이미지 예제를 복제한 것이 아니라 2차원 합성 입력으로 gradient 계약을 확인하는 독립 실습입니다.

# 02. Seq2Seq의 학습 입력과 독립 추론

교육자료 확장. 선수 학습은 16~18강의 학습 루프, 21~22강의 window와 RNN state다. 기존 23강의 attention 개념만으로는 encoder-decoder를 혼자 구현하기 어렵다. 핵심은 decoder가 시점 t의 정답을 직접 받지 않는다는 것이다.

정답이 `[a,b,eos]`이면 학습 decoder 입력은 `[bos,a,b]`로 한 칸 이동한다. 이전 정답을 주는 이 방법을 teacher forcing이라 한다. 추론에는 정답이 없으므로 `[bos]`에서 시작해서 직전 예측을 다음 입력으로 사용한다. `eos` 또는 최대 길이에서 멈춘다. 학습과 추론의 입력 분포가 달라진다는 점도 기억한다.

아래 예제는 토큰 시퀀스의 shape와 경사 흐름을 점검한다. 작은 무작위 모델의 예측 내용이 맞는지는 검증하지 않는다.

```python
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

PAD, BOS, EOS, VOCAB = 0, 1, 2, 12

class TinySeq2Seq(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(VOCAB, 8, padding_idx=PAD)
        self.encoder = nn.GRU(8, 10, batch_first=True)
        self.decoder = nn.GRU(8, 10, batch_first=True)
        self.head = nn.Linear(10, VOCAB)

    def encode(self, source, lengths):
        if (lengths <= 0).any() or (lengths > source.size(1)).any():
            raise ValueError("실제 길이는 1..T 사이여야 합니다")
        packed = pack_padded_sequence(self.embed(source), lengths.cpu(),
                                     batch_first=True, enforce_sorted=False)
        _, state = self.encoder(packed)
        return state

    def teacher_logits(self, source, lengths, target):
        state = self.encode(source, lengths)
        previous = torch.cat([
            torch.full_like(target[:, :1], BOS), target[:, :-1]
        ], dim=1)
        decoded, _ = self.decoder(self.embed(previous), state)
        return self.head(decoded)             # [B,U,VOCAB]

    @torch.no_grad()
    def generate(self, source, lengths, max_steps=6):
        if max_steps <= 0:
            raise ValueError("max_steps는 양수여야 합니다")
        self.eval()
        state = self.encode(source, lengths)
        token = torch.full((source.size(0), 1), BOS,
                           dtype=torch.long, device=source.device)
        finished = torch.zeros(source.size(0), dtype=torch.bool,
                               device=source.device)
        result = []
        for _ in range(max_steps):
            output, state = self.decoder(self.embed(token), state)
            scores = self.head(output[:, 0])
            scores[:, [PAD, BOS]] = float("-inf")
            next_token = scores.argmax(-1)
            next_token = torch.where(finished, PAD, next_token)
            result.append(next_token)
            finished = finished | next_token.eq(EOS)
            token = next_token[:, None]
            if finished.all():
                break
        return torch.stack(result, dim=1)

torch.manual_seed(13)
source = torch.tensor([[3, 4, 5], [6, 7, PAD]])
lengths = torch.tensor([3, 2])               # padding 후에도 실제 길이 보존
target = torch.tensor([[5, 4, 3, EOS], [7, 6, EOS, PAD]])
model = TinySeq2Seq()
model.train()
logits = model.teacher_logits(source, lengths, target)
assert logits.shape == (2, 4, VOCAB)
loss = F.cross_entropy(logits.transpose(1, 2), target, ignore_index=PAD)
assert torch.isfinite(loss)
loss.backward()
assert model.encoder.weight_ih_l0.grad is not None
prediction = model.generate(source, lengths) # target 인수가 전혀 없다
assert prediction.shape[0] == 2 and 1 <= prediction.shape[1] <= 6
assert prediction.dtype == torch.long
print("Seq2Seq 학습/독립 추론 계약 검사 통과")
```

`lengths`는 padding 이전 실제 길이다. padding하여 길이 5로 저장했더라도 원래 길이가 2이면 2를 전달한다. Encoder packing과 decoder loss masking은 서로 다른 처리다. Encoder가 padding을 무시해도 loss가 padding을 학습하지 않게 `ignore_index` 또는 명시적 mask가 필요하다. 모든 target이 padding이면 유효 학습 원소가 없으므로 입력 단계에서 차단한다.

실습: target 마지막 token만 바꾸었을 때 그 token을 예측하는 시점까지의 teacher-forced decoder 입력이 왜 같아야 하는가? `generate`에 target을 전달하면 무엇을 의심해야 하는가?

<details><summary>해설 보기</summary>

시점 t의 입력은 t−1 정답까지다. t 정답을 t 입력에 넣으면 자기 정답을 보는 문제가 된다. 독립 추론은 제공된 feature만 받아야 한다. 정답이 필요한 API라면 실제 평가에서 사용할 수 있는 정보인지 먼저 확인한다.

</details>

# 03. SVD·PCA·고전 ML·군집 실습

## PCA를 SVD와 역변환까지 연결하기

선형대수·차원축소 개념 보강. 선수 학습은 4강 행렬곱, 9강 PCA/LDA, 12강 군집화다. 이미지 생성과 대표표본 라벨전파 실험은 교육자료 선택심화다. PCA가 무엇을 보존하고 무엇을 버리는지 직접 확인하면 분산 보존률을 예측 정확도로 오해하지 않게 된다.

중심화한 데이터 행렬 `Xc:[N,F]`의 축소형 SVD는 `Xc=UΣVᵀ`다. `r=min(N,F)`로 두면 `U:[N,r]`, `Σ:[r,r]`, `Vᵀ:[r,F]`이며 특이값은 음수가 아니다. rank가 작으면 특이값 일부는 0이다. `U`는 `XcXcᵀ`, `V`는 `XcᵀXc`의 고유벡터와 연결된다. 임의 정방행렬의 SVD가 그 행렬의 고유분해와 같다는 주장은 틀리다.

상위 k개의 오른쪽 특이벡터를 `V_k:[F,k]`로 모으면 좌표는 `Z=XcV_k`, 복원은 `X_hat=ZV_kᵀ+mean_train`이다. 표본 공분산의 주성분 분산은 `singular_value²/(N−1)`이며 N>1을 가정한다. PCA 좌표계에서 k-means 중심을 고른 뒤 역변환하면 원공간의 대표 패턴을 볼 수 있다. 이것은 학습한 대표점을 복원하는 실험이며, 새로운 사람·다양한 현실 사례를 생성한다는 보장이 아니다.

손계산: 중심화된 행렬 `[[−1,0],[1,0]]`의 첫 방향은 첫 축, 특이값은 `sqrt(2)`다. 표본분산은 2, 둘째 방향 분산은 0이다. k=1로 줄여 복원해도 정보 손실이 없다.

```python
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
```

직접 풀 문제: train에서 scaler와 PCA를 fit하고 validation은 transform만 하는 이유는? k를 늘릴수록 train 복원오차가 줄어도 downstream 분류 성능이 반드시 높아지지 않는 이유는? 위 예제에서 validation의 두 번째 좌표에 이상탐지 신호가 들어 있다면 k=1의 비용은 무엇인가?

<details><summary>해설 보기</summary>

validation 분포로 축을 선택하면 모델 비교에 쓸 데이터가 전처리 학습에 섞인다. 분산 보존과 target 예측은 다른 목적이므로 작은 분산 방향도 정답에 중요할 수 있다. 예제의 k=1 투영은 두 번째 좌표를 전부 버려, 그 좌표의 구분 신호도 사라진다. 대신 원공간 복원오차를 anomaly score로 쓰는 별도 설계에서는 버려진 방향의 오차를 측정할 수 있다.

</details>

## 군집 대표표본과 제한된 라벨 예산

교육자료 선택심화. 공식 DS 필수 실기 목록으로 승격하지 않는다.

라벨이 일부만 있는 상황에서는 train feature를 군집화한 뒤 각 중심에 가장 가까운 관측을 골라 사람이 라벨링하는 실험을 할 수 있다. 비교군은 같은 개수의 무작위 라벨 표본이다. 전체 train 정답을 사용한 모델은 라벨 예산이 다른 참고 상한으로 따로 표시한다.

대표표본 라벨을 같은 군집 전체로 전파하면 학습 표본이 늘지만, 군집 안에 여러 class가 섞이면 오답도 함께 늘어난다. 군집 중심에 가깝다는 사실이 항상 정답 라벨의 대표성을 보장하지 않는다. 대표점까지의 거리가 작은 표본만 우선 전파하고 신뢰도별 성능을 비교할 수 있다. validation/test의 라벨은 표본 선택·전파에 사용하지 않는다.

실습 계약: train을 30개 군집으로 나누고 대표점 30개의 라벨만 공개한다. 무작위 30개 모델, 대표점 30개 모델, 거리 제한 전파 모델을 동일한 독립 validation에 비교한다. 선택한 대표점이 중복되면 실제 고유 라벨 예산도 기록한다. silhouette는 군집의 기하학적 분리 점수이며 라벨 예측의 정답률을 대신하지 않는다.

## 고전 ML을 같은 검증 조건에서 직접 비교하기

이 절은 교육자료 실습 확장이다. scikit-learn은 고전 ML 비교에 사용하며 신경망 구현은 계속 PyTorch로 한다. 특정 모델이나 아래 실습 전체가 공식 필수 구현이라는 뜻은 아니다. 선수 학습은 본강 8~13강의 scaling, feature selection, PCA/LDA, 앙상블, validation이다.

모델 이름을 아는 것과 새로운 데이터로 공정하게 비교하는 것은 다르다. 이 실습에서는 원자료의 split을 먼저 고정한 뒤, 전처리와 모델을 하나의 Pipeline으로 묶는다. CV가 각 fold에서 Pipeline을 새로 fit하므로 평균·표준편차·PCA 축·LDA 축·선택 feature를 validation fold에서 배우지 않는다.

합성 데이터는 독립 표형 행이라는 가정이다. 실제 데이터가 같은 차량의 반복 관측이거나 시간순 데이터라면 아래 random stratified split을 그대로 적용하지 말고 group/time split을 설계한다.

### 데이터와 모델의 계약

데이터는 540행, 수치 feature 12개, class 3개다. 먼저 최종 test 108행을 격리하고, 나머지에서 valid 108행을 따로 둔다. train 324행 내부의 동일한 3-fold만으로 후보를 비교한다. 이 실습에서 valid는 선택한 모델의 독립 점검이며 다시 후보를 고르는 데 사용하지 않는다. 최종 선택을 고정한 다음 train+valid로 재학습하고 test를 한 번만 평가한다. 실제 작업에서 valid를 보고 다시 수정했다면 그 valid는 더 이상 최종 일반화 성능의 독립 증거가 아니다.

PCA/LDA는 원래 feature를 새 좌표로 바꾸고, SelectFromModel은 원래 feature 일부만 남긴다. LDA는 class 3개에서 최대 2축이다. 아래 feature selection용 Random Forest는 중요도를 얻는 고전 ML 모델이며, 별도의 신경망을 사용하지 않는다.

### 독립 실행 예제

```python
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
```

실습을 새로 변형할 때는 위 코드의 test 출력 부분을 잠시 빼 두고 train CV/valid에서만 결정한다. 마지막 test 점수를 이미 본 같은 데이터를 계속 사용하면 ‘한 번만 평가’ 원칙이 깨진다. 추가 연습의 test는 새 seed로 새로 생성하여 최종 단계까지 보지 않는다.

### 결과를 읽는 법

CV 평균은 동일한 fold에서 측정한 세 점수의 평균이다. 표준편차는 fold 간 변동을 보여 주지만 세 fold만으로 엄밀한 신뢰구간을 제공하지 않는다. 후보를 많이 비교해 가장 높은 값을 고르면 그 CV 최고 점수도 낙관적일 수 있다. 낮은 CV 표준편차만으로 무조건 좋은 모델이라고 판정하지 않는다.

Bagging은 같은 종류의 tree를 bootstrap 표본에서 여러 번 학습한다. Voting은 서로 다른 종류의 모델의 class 투표를 합친다. 이 예제는 hard voting이므로 SVM의 확률 추정이 필요 없다. soft voting으로 바꾸려면 모든 구성 모델의 확률 출력과 class 열 순서를 확인해야 한다.

PCA가 분산을 많이 보존하거나 RF가 중요하다고 판단한 feature를 남겼다고 해서 KNN 성능이 반드시 개선되지는 않는다. PCA·LDA·선택을 거친 뒤 거리 공간과 남은 정보가 달라지므로, 어떤 변환이 실제 metric에서 도움을 주는지 비교한다. synthetic 데이터 한 번의 승자를 실제 자동차 데이터의 최선 모델로 일반화하지 않는다.

### 직접 바꾸어 풀기

1. PCA 차원을 6에서 3으로 바꿀 때 학습 시간이 줄고 성능이 달라지는 이유를 설명하라. LDA도 3으로 바꾸어도 되는가?
2. `SelectFromModel`을 split 전에 전체 X,y에 fit한 뒤 이 코드를 실행하면 왜 누수인가?
3. KNN의 StandardScaler를 제거한 후보를 추가하라. 단위가 100배 다른 feature가 거리 계산에 어떤 영향을 주는지 예측한 뒤 확인하라.
4. 선택된 모델을 train+valid로 다시 fit하면 valid 점수를 다시 계산해 ‘최종 일반화 성능’이라고 부를 수 있는가?
5. Voting이 가장 좋지 않아도 코드가 틀렸다는 뜻이 아닌 이유를 설명하라.

<details><summary>해설 보기</summary>

1. PCA 출력이 `[N,3]`이 되어 거리 계산은 작아지지만 target에 유용한 방향도 버릴 수 있다. 3-class LDA의 일반적 축 상한은 2이므로 3축 설정은 맞지 않는다.
2. feature 선택이 validation/test target을 보고 이루어진다. CV fold를 나누어도 이미 선택된 열에 정답 정보가 반영되어 낙관적 평가가 된다.
3. Euclidean 거리에서 각 feature 차이를 제곱한다. 값의 단위만 100배 커져도 해당 항은 10,000배 커져 다른 feature를 압도할 수 있다. 실제 영향은 값의 분포와 정보량에도 달려 있다.
4. 재학습에 valid가 들어갔으므로 그 점수는 훈련 성능의 일부다. 최종 평가는 아직 학습·선택에 쓰지 않은 test로 한다.
5. 구성 모델의 오류가 비슷하거나 상대적으로 약한 모델의 투표가 강한 모델의 정답을 바꾸면 개선되지 않을 수 있다. 앙상블은 자동적인 성능 보장이 아니다.

</details>

## 군집 실행과 silhouette를 계산할 수 없는 경우

교육자료 선택 실습이다. 군집은 정답 label 없이 feature의 구조를 찾는다. DBSCAN의 noise label −1을 실제 class로 해석하지 않으며, noise를 무조건 제거할 데이터 오류라고 단정하지 않는다. 아래 코드는 같은 feature를 k-means와 DBSCAN으로 탐색하고 **noise를 제외한 군집 silhouette**와 전체 noise 비율을 따로 보고한다.

이 예제는 주어진 데이터 전체의 군집 구조를 기술하는 탐색 분석이다. 새로운 test 예측 성능을 평가하는 실험이 아니다. 군집 feature를 분류기 입력으로 사용할 경우에는 앞 절처럼 train fold에서만 scaler/군집 모델을 fit해야 한다.

```python
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
```

silhouette의 `a`는 같은 군집 다른 표본까지의 평균거리다. `b`는 다른 군집 각각까지 평균거리를 구한 뒤 그중 가장 작은 값이다. 다른 군집 중심 한 점까지 거리와 다르다. 곡선형 군집은 눈으로는 의미 있게 나뉘어도 Euclidean silhouette가 기대보다 낮을 수 있다. DBSCAN 후보마다 noise를 제외하는 비율이 달라지므로 이 점수 하나만 비교하면 쉬운 표본만 남긴 후보를 선호할 수 있다. noise 비율과 실제 목적을 함께 읽는다.

직접 풀 문제: eps를 0.10과 0.60으로 바꾸어 군집 수·noise 비율을 관찰하라. silhouette가 가장 큰 eps를 무조건 고르면 안 되는 이유는? DBSCAN 결과가 모두 −1이면 −1을 하나의 일반 군집으로 간주해 점수를 계산해도 되는가?

<details><summary>해설 보기</summary>

eps가 작으면 이웃이 부족하여 noise가 늘기 쉽고, 크면 군집이 합쳐질 수 있다. 정확한 변화는 데이터에 따라 다르다. 서로 다른 후보가 서로 다른 평가 표본을 남기거나 실제 의미와 다른 모양을 만들 수 있으므로 점수만 최적화하지 않는다. 모두 −1이면 이 실습 정의상 군집화된 표본이 없다. silhouette도 정의 조건을 만족하지 않으므로 숫자를 억지로 출력하지 않는다.

</details>

참고: [Pipeline과 전처리 누수 예방](https://scikit-learn.org/1.5/common_pitfalls.html), [공식 앙상블 안내](https://scikit-learn.org/1.5/modules/ensemble.html), [PCA 문서](https://scikit-learn.org/1.5/modules/generated/sklearn.decomposition.PCA.html). 교육자료 확장 실습이며 모델 선택 권고나 실기 출제 보장이 아닙니다.

# 04. Bayesian 탐색·Expected Improvement·실험 보고

교육자료 개념 확장. 선수 학습은 6강 확률, 13강 validation, 15강 탐색·실험 설계다. 외부 tuner 설치나 특정 패키지를 공식 시험 요구사항으로 해석하지 않는다. 한 모델의 학습에 오래 걸리는 상황에서 다음 실험을 고르는 원리를 이해하기 위한 강의다.

각 hyperparameter 조합을 실제 학습하는 비용이 크면, 지금까지 얻은 조합과 validation 점수로 목적함수를 근사할 수 있다. 이 근사 모델이 surrogate다. Gaussian Process는 대표적인 선택이지만 모든 Bayesian 탐색 도구가 GP를 쓰는 것은 아니다. 근사 평균은 기대 성능, 불확실성은 아직 충분히 살펴보지 못한 정도를 나타낸다.

Acquisition function은 이 두 정보를 이용해 다음에 실제 평가할 후보를 고른다. 현재 좋은 곳 근처를 더 보는 exploitation과 불확실한 곳을 살펴보는 exploration을 함께 고려한다. acquisition의 높은 점수는 validation 성능이 확인되었다는 뜻이 아니다. 후보를 실제로 학습·검증한 뒤 surrogate를 갱신한다.

점수가 클수록 좋은 문제에서 현재 최고 실측 점수를 `f_best`라 하면 개선량은 `max(f(x)−f_best−ξ,0)`, EI는 그 기대값이다. GP 예측이 평균 μ, 표준편차 σ인 정규분포라면 σ>0에서 `z=(μ−f_best−ξ)/σ`, `EI=(μ−f_best−ξ)Φ(z)+σφ(z)`다. σ=0에서는 결정적 개선량을 직접 계산한다. 오차를 최소화하는 문제는 부호와 best 정의를 바꿔야 한다.

탐색 실습에서는 동일 split·동일 metric·동일 평가 예산으로 grid와 random을 먼저 비교한다. 3×4×2 조합에 5-fold이면 grid는 120번 학습한다. random 8회에 5-fold이면 40번이다. 최종 refit 비용은 별도다. PyTorch trial마다 모델과 optimizer를 새로 만들고 best checkpoint를 저장한다. 마지막 trial의 모델을 best parameter 모델로 착각하지 않는다. 시간 제한이 촉박하면 모델 1개와 핵심 hyperparameter 몇 개만 비교하는 편이 실행 가능하다.

```python
from math import erf, exp, isclose, pi, sqrt

def expected_improvement(mu, sigma, best, xi=0.0):
    """점수 최대화 문제의 단일 후보 EI. Gaussian surrogate 가정."""
    if sigma < 0:
        raise ValueError("표준편차는 음수일 수 없습니다")
    improvement = mu - best - xi
    if sigma == 0:
        return max(improvement, 0.0)
    z = improvement / sigma
    cdf = 0.5 * (1.0 + erf(z / sqrt(2.0)))
    density = exp(-0.5 * z*z) / sqrt(2.0*pi)
    return max(improvement * cdf + sigma * density, 0.0)

assert isclose(expected_improvement(0.8, 0.0, 0.7), 0.1)
assert expected_improvement(0.6, 0.0, 0.7) == 0.0
assert isclose(expected_improvement(0.7, 0.1, 0.7), 0.1/sqrt(2*pi))
# 같은 평균이라면 아직 불확실한 후보에도 개선 가능성이 있다.
assert expected_improvement(0.7, 0.1, 0.7) > expected_improvement(0.7, 0, 0.7)
print("EI 경계값·손계산 검사 통과")
```

이 코드는 acquisition의 수치만 계산하며 GP 학습이나 완전한 Bayesian optimizer를 구현한 것은 아니다. validation accuracy는 유한 구간의 값이므로 정규분포 surrogate도 근사라는 사실을 기억한다.

직접 풀 문제: μ=best, σ=0이면 EI는 얼마인가? RMSE를 그대로 ‘maximize’로 넣으면 어느 방향으로 탐색하는가? 각 trial이 다른 validation split을 쓰면 후보 비교에 어떤 잡음이 추가되는가?

<details><summary>해설 보기</summary>

첫 답은 0이다. 예측이 결정적이고 현재 최고와 같아 개선이 없다. RMSE 최대화는 오차가 큰 후보를 선호하므로 minimize로 설정하거나 음의 RMSE를 최대화해야 한다. split이 바뀌면 parameter 효과와 데이터 분할의 난도 차이가 섞인다. 비교 목적에 맞는 고정 split이나 동일 CV fold를 사용한다.

</details>

## 학습 결과 보고서의 최소 기록

교육자료 DOCX의 실험·결과·한계 서술 요구를 독립 학습지로 재구성한 항목이다. 기존 15강 ablation 표에 다음을 함께 적으면 결과를 다시 검토하기 쉽다.

- 데이터 계약: 행/그룹/시간 단위, target 의미와 단위, 실제 사용 가능한 feature
- 정제 판단: 결측·이상치를 오류 또는 유효 신호로 판단한 근거
- 검증 근거: split 방식, 누수 검사, class/그룹 분포
- 비교 조건: 바꾼 요소 하나, 고정 요소, seed, 시간·메모리 예산
- 결과: validation metric, class/target별 실패, 사용한 checkpoint
- 한계: 합성 데이터와 실제 데이터의 차이, 누락된 관측, 일반화할 수 없는 조건

빈칸에 점수만 적는 것으로 끝내지 않는다. 예를 들어 ‘RMSE가 줄었다’에 더해 어떤 target·기간에서 개선했는지와 왜 그 split이 배포 조건을 대변하는지를 설명한다.
