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
