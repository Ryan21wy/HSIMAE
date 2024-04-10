import torch
import math
import torch.nn as nn
from timm.models.layers import DropPath, trunc_normal_
import torch.nn.functional as F
from torch.nn import init


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.2):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.Conv1 = nn.Conv2d(in_features, hidden_features, kernel_size=(1,1), bias=False)
        self.act = act_layer()
        self.proj1 = nn.Conv2d(hidden_features, hidden_features, (3, 3), padding=(1, 1), groups=hidden_features, bias=False)
        self.Conv2 = nn.Conv2d(hidden_features, out_features, kernel_size=(1,1), bias=False)

    def forward(self, x):
        x = self.Conv1(x.permute(0,3,1,2)).permute(0,2,3,1)
        x = self.act(x)
        x= self.proj1(x.permute(0,3,1,2)).permute(0,2,3,1)
        x = self.act(x)
        x = self.Conv2(x.permute(0,3,1,2)).permute(0,2,3,1)
        return x


class EISA(nn.Module):
    def __init__(self, dim, segment_dim=8, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.segment_dim = segment_dim

        self.conv = nn.Conv2d(dim*3, dim, kernel_size=1, bias=qkv_bias)

        self.mlp_h = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.mlp_w = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)
        self.mlp_s = nn.Conv2d(dim, dim, kernel_size=1, bias=qkv_bias)

        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.pool_h = nn.AdaptiveAvgPool2d(1)
        self.pool_w = nn.AdaptiveAvgPool2d(1)
        self.pool_c = nn.AdaptiveAvgPool2d(1)

        if segment_dim == 8:
            self.linearH = nn.Linear(7,dim)
            self.linearW = nn.Linear(7,dim)
        else:
            self.linearH = nn.Linear(3, dim)
            self.linearW = nn.Linear(3, dim)

        self.convH = nn.Conv1d(1, 1, kernel_size=1, bias=qkv_bias)
        self.convH2 = nn.Conv1d(1, 1, kernel_size=3, padding=1)
        self.sigmoid=nn.Sigmoid()

        self.mix_hw = nn.Conv1d(2, 1, kernel_size=1, bias=qkv_bias)
        self.mix_hw2 = nn.Conv1d(1, 1, kernel_size=3, padding=1)

        self.mix_hws = nn.Conv1d(3, 1, kernel_size=1, bias=qkv_bias)
        self.mix_hws2 = nn.Conv1d(1, 1, kernel_size=3, padding=1)

        self.mix_all = nn.Conv2d(dim*3, dim, kernel_size=1, bias=qkv_bias)
        self.gelu = nn.GELU()
        self.weights = nn.Parameter(torch.ones(2))

    def forward(self, x):
        add = x
        x1 = x.permute(0, 3, 1, 2)
        ###Height###
        h = self.mlp_h(x1)
        h = self.gelu(h)
        h1 = h.permute(0, 2, 1, 3)
        h2 = self.pool_h(h1)
        h2 = h2.squeeze(-1).permute(0, 2, 1)
        h3 = self.convH(h2)
        h3 = self.sigmoid(h3)
        h3 = self.convH2(h3)
        h3 = self.sigmoid(h3)
        h3 = h3.permute(0, 2, 1).unsqueeze(-1)
        h3 = h1 * h3.expand_as(h1)
        h3 = h3.permute(0, 2, 1, 3)

        ###Weight###
        w = self.mlp_w(x1)
        w = self.gelu(w)
        w1 = w.permute(0, 3, 2, 1)
        w2 = self.pool_w(w1)
        w2 = w2.squeeze(-1).permute(0, 2, 1)
        mix_hw = torch.concat([h2, w2], 1)
        mix_hw = self.mix_hw(mix_hw)
        mix_hw = self.sigmoid(mix_hw)
        mix_hw = self.mix_hw2(mix_hw)
        mix_hw = self.sigmoid(mix_hw)
        mix_hw = mix_hw.permute(0, 2, 1).unsqueeze(-1)
        mix_hw = w1 * mix_hw.expand_as(w1)
        w3 = mix_hw.permute(0, 3, 2, 1)
        ###Spectral###
        s = self.mlp_s(x1)
        s = self.gelu(s)
        s1 = self.pool_c(s)
        s1 = s1.squeeze(-1).permute(0, 2, 1)
        h2 = self.linearH(h2)
        h2 = h2.view(s1.shape[0], s1.shape[1], s1.shape[2])
        w2 = self.linearW(w2)
        w2 = w2.view(s1.shape[0], s1.shape[1], s1.shape[2])

        mix_hws = torch.concat([h2, w2, s1], 1)
        mix_hws = self.mix_hws(mix_hws)
        mix_hws = self.sigmoid(mix_hws)
        mix_hws = self.mix_hws2(mix_hws)
        mix_hws = self.sigmoid(mix_hws)
        mix_hws = mix_hws.permute(0, 2, 1).unsqueeze(-1)
        s2 = s * mix_hws.expand_as(s)

        x = torch.concat([h3, w3, s2], 1)
        x = self.mix_all(x)
        x = x.permute(0, 2, 3, 1)
        x = self.proj(x)
        x = x*self.weights[0] + add*self.weights[1]
        x = self.proj_drop(x)
        return x


class MDCP(nn.Module):
    def __init__(self, img_size=16, patch_size=3, in_chans=3, embed_dim=256, dataset="PaviaU"):
        super().__init__()
        self.proj = nn.Conv3d(1, 4, (11, 7, 7), padding=(0, 3, 3))
        self.proj2 = nn.Conv3d(4, 8, (9, 5, 5), padding=(0, 2, 2))
        self.pool = nn.AdaptiveAvgPool3d((40, 15, 15))
        self.bn = nn.BatchNorm3d(8)
        if dataset == "PaviaU":
            groups = 5
            groups_width = 64
            channels = 103
            self.pool = nn.AdaptiveAvgPool3d((40, 15, 15))
        elif dataset == "Salinas":
            groups = 11
            groups_width = 40
            channels = 204
            self.pool = nn.AdaptiveAvgPool3d((55, 15, 15))
        elif dataset == "Houston2013":
            groups = 5
            groups_width = 64
            channels = 144
            self.pool = nn.AdaptiveAvgPool3d((40, 15, 15))
        else:
            groups = 5
            groups_width = 64
            channels = 270
            self.pool = nn.AdaptiveAvgPool3d((40, 15, 15))
        new_bands = math.ceil(channels/groups) * groups
        pad_size = new_bands - channels
        self.pad = nn.ReplicationPad3d((0, 0, 0, 0, 0, pad_size))
        self.conv_1 = nn.Conv2d(new_bands, groups * groups_width, (1, 1), groups=groups)
        self.bn_1 = nn.BatchNorm2d(groups * groups_width)
        self.add2D = Add2D(groups * groups_width, (3, 3), (1, 1), groups_s=groups)
        self.down_sample = nn.Conv2d(embed_dim, embed_dim, kernel_size=(3, 3), stride=(2,2))
        self.weights = nn.Parameter(torch.ones(2))

    def forward(self, x):
        x1 = self.pad(x).squeeze(1)
        x1 = F.relu(self.bn_1(self.conv_1(x1)))
        x1 = self.add2D(x1)
        x = self.proj(x)
        x = self.proj2(x)
        x = self.bn(self.pool(x))
        B, D, H, W, C = x.shape
        x = x.reshape(B, D*H, W, C)
        x = x * self.weights[0] + x1 * self.weights[1]
        x = self.down_sample(x)
        return x


class Add2D(nn.Module):
    def __init__(self, in_channels, kernel_size, padding, groups_s):
        super(Add2D, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, padding=padding, groups=groups_s)
        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv2 = nn.Conv2d(in_channels, in_channels, kernel_size=(5, 5), padding=(2, 2), groups=groups_s)
        self.bn2 = nn.BatchNorm2d(in_channels)
        self.conv3 = nn.Conv2d(in_channels, in_channels, kernel_size=1, padding=0, groups=1)
        self.bn3 = nn.BatchNorm2d(in_channels)

    def forward(self, X):
        Y = F.relu(self.bn1(self.conv1(X)))
        Y = self.bn2(self.conv2(Y))
        X= self.bn3(self.conv3(X))
        return F.relu(X + Y)


class PermutatorBlock(nn.Module):

    def __init__(self, dim, segment_dim, mlp_ratio=4., qkv_bias=True, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.GroupNorm, skip_lam=1.0, mlp_fn=EISA):
        super().__init__()
        if dim == 440:
            groups=8
        else:
            groups=16
        self.norm1 = norm_layer(groups, dim)
        self.attn = mlp_fn(dim, segment_dim=segment_dim, qkv_bias=qkv_bias, qk_scale=None, attn_drop=attn_drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(groups, dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = FFN(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)
        self.skip_lam = skip_lam

    def forward(self, x):
        x = x.permute(0,3,1,2)
        x = self.norm1(x)
        x = x.permute(0,2,3,1)
        x = x + self.drop_path(self.attn(x)) / self.skip_lam
        x = x.permute(0,3,1,2)
        x = self.norm2(x)
        x = x.permute(0,2,3,1)
        x = x + self.drop_path(self.mlp(x)) / self.skip_lam
        return x


class Downsample(nn.Module):

    def __init__(self, in_embed_dim, out_embed_dim, patch_size):
        super().__init__()
        self.proj = nn.Conv2d(in_embed_dim, out_embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = self.proj(x)
        x = x.permute(0, 2, 3, 1)
        return x


def basic_blocks(dim, index, layers, segment_dim, mlp_ratio=3., qkv_bias=True, qk_scale=None, \
                 attn_drop=0, drop_path_rate=0., skip_lam=1.0, mlp_fn=EISA, **kwargs):
    blocks = []
    for block_idx in range(layers[index]):
        block_dpr = drop_path_rate * (block_idx + sum(layers[:index])) / (sum(layers) - 1)
        blocks.append(PermutatorBlock(dim, segment_dim, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale, \
                                      attn_drop=attn_drop, drop_path=block_dpr, skip_lam=skip_lam, mlp_fn=mlp_fn))
    blocks = nn.Sequential(*blocks)
    return blocks


class DCTN(nn.Module):

    def __init__(self, layers, img_size=15, patch_size=3, in_chans=3, num_classes=1000,
                 embed_dims=None, transitions=None, segment_dim=None, mlp_ratios=None, skip_lam=1.0,
                 qkv_bias=False, qk_scale=None, drop_rate=0.1, attn_drop_rate=0.1, drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm, mlp_fn=EISA, dateset="PaviaU"):

        super().__init__()
        self.num_classes = num_classes
        self.patch_embed = MDCP(img_size=img_size, patch_size=patch_size, in_chans=in_chans,
                                      embed_dim=embed_dims[0], dataset=dateset)
        network = []
        for i in range(len(layers)):
            stage = basic_blocks(embed_dims[i], i, layers, segment_dim[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                                 qk_scale=qk_scale, attn_drop=attn_drop_rate, drop_path_rate=drop_path_rate,
                                 norm_layer=norm_layer, skip_lam=skip_lam,
                                 mlp_fn=mlp_fn)
            network.append(stage)
            if i >= len(layers) - 1:
                break
            if transitions[i] or embed_dims[i] != embed_dims[i + 1]:
                patch_size = 2 if transitions[i] else 1
                network.append(Downsample(embed_dims[i], embed_dims[i + 1], patch_size))
        self.network = nn.ModuleList(network)

        self.norm = norm_layer(embed_dims[-1])

        self.head = nn.Linear(embed_dims[-1], num_classes) if num_classes > 0 else nn.Identity()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.conv_cls_head = nn.Linear(368, num_classes)
        self.down_sample = Downsample(embed_dims[0], 512, 2)

        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(embed_dims[0], 128, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128)
        self.bn2_2 = nn.BatchNorm2d(128)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3,stride=1, padding=1)
        self.bn3 = nn.BatchNorm2d(256)
        self.pool = nn.MaxPool2d(2)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1,padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)

        self.conv4 = nn.Conv2d(256, embed_dims[3], kernel_size=3, stride=1,padding=1)
        self.bn4 = nn.BatchNorm2d(embed_dims[3])
        self.conv4_2 = nn.Conv2d(embed_dims[3], embed_dims[3], kernel_size=3, padding=1, groups=embed_dims[3])
        self.bn4_2 = nn.BatchNorm2d(embed_dims[3])

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma3 = nn.Parameter(torch.zeros(1))
        self.ConvEnd = nn.Conv2d(embed_dims[3]*3,embed_dims[3],kernel_size=1)
        self.weights = nn.Parameter(torch.ones(3))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

        elif isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight)

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_embeddings(self, x):
        x = self.patch_embed(x)
        x = x.permute(0, 2, 3, 1)
        return x

    def forward_tokens(self, x):
        for idx, block in enumerate(self.network):
            x = block(x)
        return x

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.forward_embeddings(x)
        x_c = x.permute(0, 3, 1, 2)
        x_embedding = x
        x = self.forward_tokens(x)
        x_embedding = self.down_sample(x_embedding)
        x_embedding = x_embedding.permute(0,3,1,2)
        x_c2 = self.act(self.bn2(self.conv2(x_c)))
        x_c2 = self.act(self.bn2_2(self.conv2_2(x_c2)))
        x_c3 = self.act(self.bn3(self.conv3(x_c2)))
        x_c3 = self.pool(x_c3)
        x_c3 = self.act(self.bn3_2(self.conv3_2(x_c3)))
        x_c4 = self.act(self.bn4(self.conv4(x_c3)))
        x_c4 = self.act(self.bn4_2(self.conv4_2(x_c4)))
        x = self.ConvEnd(torch.concat([x_c4, x.permute(0, 3, 1, 2), x_embedding],dim=1))
        x = x.permute(0, 2, 3, 1)
        B, H, W, C = x.shape
        x = x.reshape(B, -1, C)
        x = self.norm(x)
        return self.head(x.mean(1))