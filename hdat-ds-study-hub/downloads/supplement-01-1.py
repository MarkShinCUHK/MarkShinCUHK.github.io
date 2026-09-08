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
