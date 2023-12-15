import torch
from torch import nn


class Spectral_attention(nn.Module):
    def __init__(self, in_features, hidden_features, out_features):
        super(Spectral_attention, self).__init__()
        self.AvgPool = nn.AdaptiveAvgPool2d((1, 1))
        self.MaxPool = nn.AdaptiveMaxPool2d((1, 1))
        self.SharedMLP = nn.Sequential(
            nn.Linear(in_features, hidden_features),
            nn.ReLU(),
            nn.Linear(hidden_features, out_features),
            nn.Sigmoid()
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, X):

        y1 = self.AvgPool(X)
        y2 = self.MaxPool(X)
        y1 = y1.view(y1.size(0), -1)
        y2 = y2.view(y2.size(0), -1)
        # print(y1.shape, y2.shape)
        y1 = self.SharedMLP(y1)
        y2 = self.SharedMLP(y2)
        y = y1 + y2
        y = torch.reshape(y, (y.shape[0], y.shape[1], 1, 1))
        return self.sigmoid(y)


class Spatial_attention(nn.Module):
    def __init__(self, in_chanels, kernel_size, out_chanel, stride, padding):
        super(Spatial_attention, self).__init__()
        self.conv1 = nn.Conv2d(in_chanels, out_chanel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.act = nn.Sigmoid()

    def forward(self, X):
        avg_out = torch.mean(X, dim=1, keepdim=True)
        max_out, _ = torch.max(X, dim=1, keepdim=True)
        y = torch.cat((avg_out, max_out), 1)
        y = self.conv1(y)
        return self.act(y)


class SSA_Module(nn.Module):
    def __init__(self, in_chanels, out_chanel, kernel_size, stride=1, padding=1):
        super(SSA_Module, self).__init__()
        self.conv1 = nn.Conv2d(in_chanels, out_chanel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_chanel, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU()

        self.conv2 = nn.Conv2d(out_chanel, out_chanel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn2 = nn.BatchNorm2d(out_chanel, eps=0.001, momentum=0.1, affine=True)

        self.spe_attention = Spectral_attention(out_chanel, out_chanel // 8, out_chanel)
        self.spa_attention = Spatial_attention(2, 3, 1, 1, 1)

        self.relu2 = nn.ReLU()

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.bn1(x1)
        x1 = self.relu1(x1)
        x1 = self.conv1(x1)
        x1 = self.bn2(x1)

        x2 = self.spe_attention(x1) * x1
        x3 = self.spa_attention(x2) * x2

        x4 = x3 * x1 + x
        return self.relu2(x4)


class RSSAN(nn.Module):
    def __init__(self, feature_class, in_chanels, kernel_size, out_chanel, stride=1, padding=0):
        # 16, 200, 3, 32, 1, 1
        super(RSSAN, self).__init__()
        self.attention1 = Spectral_attention(in_chanels, int(in_chanels//8), in_chanels)
        self.attention2 = Spatial_attention(2, 3, 1, 1, 1)
        self.conv1 = nn.Conv2d(in_chanels, out_chanel, kernel_size=kernel_size, stride=stride, padding=padding)
        self.bn1 = nn.BatchNorm2d(out_chanel, eps=0.001, momentum=0.1, affine=True)
        self.relu1 = nn.ReLU(inplace=True)

        self.ssa1 = SSA_Module(out_chanel, out_chanel, kernel_size)
        self.ssa2 = SSA_Module(out_chanel, out_chanel, kernel_size)
        self.ssa3 = SSA_Module(out_chanel, out_chanel, kernel_size)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.full_connection = nn.Sequential(
            nn.Linear(out_chanel, feature_class),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)

    def forward(self, X):
        x1 = self.attention1(X)
        x3 = x1 * X
        x4 = self.attention2(x3) * x3
        x5 = self.conv1(x4)
        x6 = self.bn1(x5)
        x7 = self.relu1(x6)

        x8 = self.ssa1(x7)
        x9 = self.ssa2(x8)
        x10 = self.ssa1(x9)

        x11 = self.avgpool(x10)
        x12 = x11.view(x11.size(0), -1)
        y = self.full_connection(x12)
        return y