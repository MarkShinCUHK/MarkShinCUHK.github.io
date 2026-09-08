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
