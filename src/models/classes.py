import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models
from torchvision.models import (
    resnet18,
    ResNet18_Weights,
    MobileNet_V2_Weights,
)

#Вовлеченность
class VideoEngagementModel(nn.Module):
    def __init__(self, num_classes=4, hidden_size=64, num_layers=1, dropout=0.5, unfreeze_last_block=True):
        super().__init__()
        self.backbone = models.mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        self.backbone.classifier = nn.Identity()

        for p in self.backbone.parameters():
            p.requires_grad = False

        if unfreeze_last_block:
            for p in self.backbone.features[-1].parameters():
                p.requires_grad = True

        self.lstm = nn.LSTM(1280, hidden_size, num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        B, T, C, H, W = x.size()
        x = x.view(B * T, C, H, W)
        feats = self.backbone(x)
        feats = feats.view(B, T, -1)
        lstm_out, _ = self.lstm(feats)
        out = self.fc(self.dropout(lstm_out[:, -1, :]))
        return out

#Эмоции по видео
class EmotionCNN(nn.Module):
    def __init__(self):
        super(EmotionCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.fc1 = nn.Linear(128 * 6 * 6, 128)
        self.fc2 = nn.Linear(128, 7)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.pool(nn.functional.relu(self.conv1(x)))
        x = self.pool(nn.functional.relu(self.conv2(x)))
        x = self.pool(nn.functional.relu(self.conv3(x)))
        x = x.view(-1, 128 * 6 * 6)
        x = nn.functional.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

#Расширенность зрачка
class DualEyeResNet(nn.Module):
    def __init__(self, img_size=64):
        super().__init__()

        backbone = models.resnet18(pretrained=True)
        backbone.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)

        self.feature_extractor = nn.Sequential(*list(backbone.children())[:-2])

        self.regressor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(512 * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 1)
        )

    def forward(self, left_eye, right_eye):

        feat_left = self.feature_extractor(left_eye)
        feat_right = self.feature_extractor(right_eye)


        feats = torch.cat([feat_left, feat_right], dim=1)


        diameter = self.regressor(feats)
        return diameter.squeeze(-1)

#Эмоции по звуку
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

#Пульс
import torch
import torch.nn as nn
import torch.nn.functional as F

class VitalSignsModel(nn.Module):
    def __init__(self, num_frames=64):
        super(VitalSignsModel, self).__init__()

        self.conv1 = nn.Conv3d(3, 32, kernel_size=(3, 5, 5), stride=(1, 2, 2), padding=(1, 2, 2))
        self.bn1 = nn.BatchNorm3d(32)

        self.conv2 = nn.Conv3d(32, 64, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn2 = nn.BatchNorm3d(64)

        self.conv3 = nn.Conv3d(64, 128, kernel_size=(3, 3, 3), stride=(2, 2, 2), padding=(1, 1, 1))
        self.bn3 = nn.BatchNorm3d(128)

        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))

        x = x.mean(dim=[2, 3, 4])
        x = self.fc(x)

        return x.squeeze()