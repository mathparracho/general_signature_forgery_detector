import torch
import torch.nn as nn
import torch.nn.functional as F


class SiameseNetwork1D(nn.Module):
    def __init__(self):
        super().__init__()

        # --- PS Layer
        self.conv_ps_shells   = nn.Conv1d(6, 64,  kernel_size=7, stride=1, padding=3)
        self.bn_ps_shells     = nn.BatchNorm1d(64)
        self.conv_ps_pressure = nn.Conv1d(66, 128, kernel_size=7, stride=1, padding=3)
        self.bn_ps_pressure   = nn.BatchNorm1d(128)

        self.conv_b1 = nn.Conv1d(192, 256, kernel_size=5, stride=2, padding=2)
        self.bn_b1   = nn.BatchNorm1d(256)

        self.conv_g1 = nn.Conv1d(256, 512, kernel_size=3, stride=2, padding=1)
        self.bn_g1   = nn.BatchNorm1d(512)

        self.conv_g2 = nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn_g2   = nn.BatchNorm1d(512)

        self.conv_g3 = nn.Conv1d(512, 512, kernel_size=3, stride=2, padding=1)
        self.bn_g3   = nn.BatchNorm1d(512)

        self.dropout = nn.Dropout(0.5)

        self.fc0 = nn.Linear(512 * 32, 16384)
        self.fc1 = nn.Linear(16384, 8192)
        self.fc2 = nn.Linear(8192, 2048)
        self.fc3 = nn.Linear(2048, 512)

        self.apply(self.init_weights)

    @staticmethod
    def init_weights(m):
        if isinstance(m, (nn.Linear, nn.Conv1d)):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0.01)

    def forward_once(self, x, x_pressure):
        # --- PS layer
        x_shells   = F.relu(self.bn_ps_shells(self.conv_ps_shells(x)))
        x_pressure = F.relu(self.bn_ps_pressure(self.conv_ps_pressure(x_pressure)))
        x = torch.cat([x_shells, x_pressure], dim=1)  # [B,192,512]

        # --- Conv stack
        x = F.relu(self.bn_b1(self.conv_b1(x)))   # [B,256,256]
        x = F.relu(self.bn_g1(self.conv_g1(x)))   # [B,512,128]
        x = F.relu(self.bn_g2(self.conv_g2(x)))   # [B,512, 64]
        x = F.relu(self.bn_g3(self.conv_g3(x)))   # [B,512, 32]

        x = self.dropout(x)

        # --- Flatten and FC head ---
        x = x.flatten(1)                          # [B, 512*32]=[B,16384]
        x = F.relu(self.fc0(x))                   # 16384
        x = F.relu(self.fc1(x))                   # 8192
        x = F.relu(self.fc2(x))                   # 2048
        x = self.fc3(x)                           # 512 (embedding)
        return x

    def forward(self, x, x_pressure):
        return self.forward_once(x, x_pressure)


class ContrastiveNetworkCascas(nn.Module):
    def __init__(self):
        super().__init__()
        self.embedding_network = SiameseNetwork1D()

    def forward(self, positive, pressure_pos, negative, pressure_neg):
        pos_out = self.embedding_network(positive, pressure_pos)
        neg_out = self.embedding_network(negative, pressure_neg)
        return pos_out, neg_out
