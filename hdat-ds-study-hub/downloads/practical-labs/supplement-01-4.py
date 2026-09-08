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
