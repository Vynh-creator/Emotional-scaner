import torch.nn as nn
class AttnPool(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.proj = nn.Linear(dim, dim)
        self.scorer = nn.Linear(dim, 1)

    def forward(self, x):
        h = torch.tanh(self.proj(x))
        scores = self.scorer(h).squeeze(-1)
        w = torch.softmax(scores, dim=-1)
        return torch.sum(x * w.unsqueeze(-1), dim=1)

class Branch(nn.Module):
    def __init__(self, in_dim, hidden=256, out_dim=128, dropout=0.3):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=in_dim,
            hidden_size=hidden,
            batch_first=True,
            bidirectional=True,
            num_layers=1,
        )
        self.norm = nn.LayerNorm(hidden * 2)
        self.pool = AttnPool(hidden * 2)
        self.mlp = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden * 2, out_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.norm(out)
        pooled = self.pool(out)
        return self.mlp(pooled)

class Model_MFCC_Wave2Vec_v2(nn.Module):
    def __init__(self, n_classes=8, mfcc_dim=13, w2v_dim=768):
        super().__init__()
        self.mfcc_branch = Branch(in_dim=mfcc_dim, hidden=128, out_dim=128, dropout=0.3)
        self.w2v_branch  = Branch(in_dim=w2v_dim,  hidden=256, out_dim=128, dropout=0.3)

        self.gate = nn.Sequential(
            nn.Linear(256, 128),
            nn.Sigmoid()
        )

        self.head = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.4),
            nn.Linear(128, n_classes)
        )

    def forward(self, x_mfcc, x_w2v):
        mfcc_emb = self.mfcc_branch(x_mfcc)
        w2v_emb  = self.w2v_branch(x_w2v)

        fused = torch.cat([mfcc_emb, w2v_emb], dim=1)
        g = self.gate(fused)
        mixed = torch.cat([g * mfcc_emb, (1 - g) * w2v_emb], dim=1)

        return self.head(mixed)

import torch
from pathlib import Path

AUDIO_PICKLE = Path(r"src\models\best_model_audio.pth")
OUT_SD = Path(r"src\models\best_model_audio_state_dict.pth")

m = torch.load(AUDIO_PICKLE, map_location="cpu", weights_only=False)
torch.save(m.state_dict(), OUT_SD)

print("Saved:", OUT_SD)
