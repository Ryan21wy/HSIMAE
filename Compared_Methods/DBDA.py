# References:
# DBDA: https://github.com/lironui/Double-Branch-Dual-Attention-Mechanism-Network/blob/master/global_module/network.py


import torch
from torch import nn
import math


class PAM_Module(nn.Module):
    """ Position attention module"""
    #Ref from SAGAN
    def __init__(self, in_dim):
        super(PAM_Module, self).__init__()
        self.chanel_in = in_dim

        # self.query_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.key_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim//8, kernel_size=1)
        # self.value_conv = nn.Conv3d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)


    def forward(self, x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW)
        """
        # m_batchsize, channle, height, width, C = x.size()
        x = x.squeeze(-1)
        # m_batchsize, C, height, width, channle = x.size()

        # proj_query = self.query_conv(x).view(m_batchsize, -1, width*height*channle).permute(0, 2, 1)
        # proj_key = self.key_conv(x).view(m_batchsize, -1, width*height*channle)
        # energy = torch.bmm(proj_query, proj_key)
        # attention = self.softmax(energy)
        # proj_value = self.value_conv(x).view(m_batchsize, -1, width*height*channle)
        #
        # out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        # out = out.view(m_batchsize, C, height, width, channle)
        # print('out', out.shape)
        # print('x', x.shape)

        m_batchsize, C, height, width = x.size()
        proj_query = self.query_conv(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        proj_key = self.key_conv(x).view(m_batchsize, -1, width * height)
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)
        proj_value = self.value_conv(x).view(m_batchsize, -1, width * height)

        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(m_batchsize, C, height, width)

        out = (self.gamma*out + x).unsqueeze(-1)
        return out


class CAM_Module(nn.Module):
    """ Channel attention module"""
    def __init__(self, in_dim):
        super(CAM_Module, self).__init__()
        self.chanel_in = in_dim

        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self,x):
        """
            inputs :
                x : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X C X C
        """
        m_batchsize, C, height, width, channle = x.size()
        #print(x.size())
        proj_query = x.view(m_batchsize, C, -1)
        proj_key = x.view(m_batchsize, C, -1).permute(0, 2, 1) #形状转换并交换维度
        energy = torch.bmm(proj_query, proj_key)
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy)-energy
        attention = self.softmax(energy_new)
        proj_value = x.view(m_batchsize, C, -1)

        out = torch.bmm(attention, proj_value)
        out = out.view(m_batchsize, C, height, width, channle)
        # print('out', out.shape)
        # print('x', x.shape)

        out = self.gamma*out + x  #C*H*W
        return out


class DBDA(nn.Module):
    def __init__(self, band, classes):
        super(DBDA, self).__init__()

        # spectral branch

        self.conv11 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, 7), stride=(1, 1, 2))
        # Dense block
        self.batch_norm11 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),  # 动量默认值为0.1
            nn.ReLU(inplace=True)
        )
        self.conv12 = nn.Conv3d(in_channels=24, out_channels=24, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm12 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv13 = nn.Conv3d(in_channels=48, out_channels=24, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm13 = nn.Sequential(
            nn.BatchNorm3d(72, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv14 = nn.Conv3d(in_channels=72, out_channels=24, padding=(0, 0, 3),
                                kernel_size=(1, 1, 7), stride=(1, 1, 1))
        self.batch_norm14 = nn.Sequential(
            nn.BatchNorm3d(96, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        kernel_3d = math.floor((band - 6) / 2)
        self.conv15 = nn.Conv3d(in_channels=96, out_channels=60,
                                kernel_size=(1, 1, kernel_3d), stride=(1, 1, 1))  # kernel size随数据变化

        # 注意力机制模块
        self.max_pooling1 = nn.AdaptiveAvgPool3d(1)
        self.avg_pooling1 = nn.AdaptiveAvgPool3d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv3d(in_channels=60, out_channels=30,
                      kernel_size=(1, 1, 1), stride=(1, 1, 1)),
            nn.Conv3d(in_channels=30, out_channels=60,
                      kernel_size=(1, 1, 1), stride=(1, 1, 1)),
        )

        self.activation1 = nn.Sigmoid()

        # Spatial Branch
        self.conv21 = nn.Conv3d(in_channels=1, out_channels=24,
                                kernel_size=(1, 1, band), stride=(1, 1, 1))
        # Dense block
        self.batch_norm21 = nn.Sequential(
            nn.BatchNorm3d(24, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv22 = nn.Conv3d(in_channels=24, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm22 = nn.Sequential(
            nn.BatchNorm3d(36, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv23 = nn.Conv3d(in_channels=36, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))
        self.batch_norm23 = nn.Sequential(
            nn.BatchNorm3d(48, eps=0.001, momentum=0.1, affine=True),
            nn.ReLU(inplace=True)
        )
        self.conv24 = nn.Conv3d(in_channels=48, out_channels=12, padding=(1, 1, 0),
                                kernel_size=(3, 3, 1), stride=(1, 1, 1))

        self.conv25 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=1, padding=(1, 1, 0),
                      kernel_size=(3, 3, 2), stride=(1, 1, 1)),
            nn.Sigmoid()
        )

        self.global_pooling = nn.AdaptiveAvgPool3d(1)
        self.full_connection = nn.Sequential(
            nn.Dropout(p=0.5),
            nn.Linear(120, classes)  # ,
            # nn.Softmax()
        )

        self.attention_spectral = CAM_Module(60)
        self.attention_spatial = PAM_Module(60)

    def forward(self, X):
        X = X.permute(0, 2, 3, 1).unsqueeze(1)
        # spectral
        x11 = self.conv11(X)
        # print('x11', x11.shape)
        x12 = self.batch_norm11(x11)
        x12 = self.conv12(x12)
        # print('x12', x12.shape)

        x13 = torch.cat((x11, x12), dim=1)
        # print('x13', x13.shape)
        x13 = self.batch_norm12(x13)
        x13 = self.conv13(x13)
        # print('x13', x13.shape)

        x14 = torch.cat((x11, x12, x13), dim=1)
        x14 = self.batch_norm13(x14)
        x14 = self.conv14(x14)

        x15 = torch.cat((x11, x12, x13, x14), dim=1)
        # print('x15', x15.shape)

        x16 = self.batch_norm14(x15)
        x16 = self.conv15(x16)
        # print('x16', x16.shape)  # 7*7*97, 60

        # print('x16', x16.shape)
        # 光谱注意力通道
        x1 = self.attention_spectral(x16)
        x1 = torch.mul(x1, x16)

        # spatial
        # print('x', X.shape)
        x21 = self.conv21(X)
        x22 = self.batch_norm21(x21)
        x22 = self.conv22(x22)

        x23 = torch.cat((x21, x22), dim=1)
        x23 = self.batch_norm22(x23)
        x23 = self.conv23(x23)

        x24 = torch.cat((x21, x22, x23), dim=1)
        x24 = self.batch_norm23(x24)
        x24 = self.conv24(x24)

        x25 = torch.cat((x21, x22, x23, x24), dim=1)
        # print('x25', x25.shape)
        # x25 = x25.permute(0, 4, 2, 3, 1)
        # print('x25', x25.shape)

        # 空间注意力机制
        x2 = self.attention_spatial(x25)
        x2 = torch.mul(x2, x25)

        # model1
        x1 = self.global_pooling(x1)
        x1 = x1.squeeze(-1).squeeze(-1).squeeze(-1)
        x2 = self.global_pooling(x2)
        x2 = x2.squeeze(-1).squeeze(-1).squeeze(-1)

        x_pre = torch.cat((x1, x2), dim=1)
        # print('x_pre', x_pre.shape)

        # model2
        # x1 = torch.mul(x2, x16)
        # x2 = torch.mul(x2, x25)
        # x_pre = x1 + x2
        #
        #
        # x_pre = x_pre.view(x_pre.shape[0], -1)
        output = self.full_connection(x_pre)
        # output = self.fc(x_pre)
        return output