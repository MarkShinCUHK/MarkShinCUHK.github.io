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
