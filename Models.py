import torch
import torch.nn as nn


class PatchEmbed(nn.Module):
    """HSI to Patch Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=1,
        embed_dim=768,
        # spectral related:
        bands=32,
        b_patch_size=4,
    ):
        super().__init__()
        img_size = (img_size, img_size)
        patch_size = (patch_size, patch_size)
        assert img_size[1] % patch_size[1] == 0
        assert img_size[0] % patch_size[0] == 0
        assert bands % b_patch_size == 0
        num_patches = (
            (img_size[1] // patch_size[1])
            * (img_size[0] // patch_size[0])
            * (bands // b_patch_size)
        )
        self.input_size = (
            bands // b_patch_size,
            img_size[0] // patch_size[0],
            img_size[1] // patch_size[1],
        )
        print(
            f"img_size {img_size} patch_size {patch_size} bands {bands} b_patch_size {b_patch_size}"
        )
        self.img_size = img_size
        self.patch_size = patch_size

        self.bands = bands
        self.b_patch_size = b_patch_size

        self.num_patches = num_patches

        self.grid_size = img_size[0] // patch_size[0]
        self.b_grid_size = bands // b_patch_size

        kernel_size = [b_patch_size] + list(patch_size)
        self.proj = nn.Conv3d(
            in_chans, embed_dim, kernel_size=kernel_size, stride=kernel_size,
        )

    def forward(self, x):
        B, C, T, H, W = x.shape
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        assert T == self.bands
        x = self.proj(x).flatten(3)
        x = torch.einsum("ncts->ntsc", x)  # [N, T, H*W, C]
        self.output_size = x.shape
        return x


class Attention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        input_size=(4, 14, 14),
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.k = nn.Linear(dim, dim, bias=qkv_bias)
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        assert attn_drop == 0.0  # do not use
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.input_size = input_size
        assert input_size[1] == input_size[2]

    def adjust_drop_rate(self, drop_rate=0.):
        self.proj_drop = nn.Dropout(drop_rate)

    def forward(self, x, attn_bias=None):
        B, N, C = x.shape
        q = (
            self.q(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        k = (
            self.k(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )
        v = (
            self.v(x)
            .reshape(B, N, self.num_heads, C // self.num_heads)
            .permute(0, 2, 1, 3)
        )

        attn = (q @ k.transpose(-2, -1)) * self.scale
        if attn_bias is not None:
            attn += attn_bias
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        x = x.view(B, -1, C)
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks
    """
    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 bias=True,
                 drop=0.,
                 ):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        bias = (bias, bias)
        drop_probs = (drop, drop)

        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias[0])
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop_probs[0])
        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias[1])
        self.drop2 = nn.Dropout(drop_probs[1])

    def adjust_drop_rate(self, drop_rate=0.):
        self.drop1 = nn.Dropout(drop_rate)
        self.drop2 = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).

    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks).
    """
    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)

    def extra_repr(self):
        return f'drop_prob={round(self.drop_prob,3):0.3f}'


class Block(nn.Module):
    """
    Transformer Block with specified Attention function
    """

    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        qk_scale=None,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        attn_func=Attention,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = attn_func(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class HSIMAE(nn.Module):
    """Masked Autoencoder with VisionTransformer backbone"""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        bands=16,
        b_patch_size=4,
        no_qkv_bias=False,
        sep_pos_embed=False,
        trunc_init=False,
        cls_embed=False,
        pad_with_cls_token=False,
        use_learnable_pos_emb=False,
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.use_learnable_pos_emb = use_learnable_pos_emb
        self.cls_embed = cls_embed
        self.b_pred_patch_size = b_patch_size

        self.patch_embed = PatchEmbed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            bands,
            b_patch_size,
        )
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        self.pad_with_cls_token = pad_with_cls_token

        if self.pad_with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.trans_to_decoder = nn.Linear(embed_dim, decoder_embed_dim)

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed or self.pad_with_cls_token:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim),
            )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.decoder_pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], decoder_embed_dim)
            )
            self.decoder_pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], decoder_embed_dim)
            )
            if self.cls_embed:
                self.decoder_pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim)
                )
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, decoder_embed_dim),
            )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.b_pred_patch_size * patch_size ** 2 * in_chans,
            bias=True,
        )

        self.norm_pix_loss = norm_pix_loss

        self.initialize_weights()

        print("model initialized")

    def initialize_weights(self):
        if self.pad_with_cls_token:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            if self.use_learnable_pos_emb:
                torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        """
        imgs: (N, 3, H, W)
        x: (N, L, patch_size**2 *3)
        """
        N, _, T, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        u = self.b_pred_patch_size
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u

        x = imgs.reshape(shape=(N, 1, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * 1))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x

    def unpatchify(self, x):
        """
        x: (N, L, patch_size**2 *3)
        imgs: (N, 3, H, W)
        """
        N, T, H, W, p, u, t, h, w = self.patch_info

        x = x.reshape(shape=(N, t, h, w, u, p, p, 1))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, 1, T, H, W))
        return imgs

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def feature_random_masking(self, x, T, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, T, 1, device=x.device)  # noise in [0, 1]
        noise = noise.repeat(1, 1, L // T).reshape((N, -1))

        # sample random patch from unmask bands
        if (len_keep / L * T) % 1 > 0:
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            noise_sorted = torch.sort(noise, dim=1)[0]
            noise_sorted[:, :len_keep] -= torch.rand(N, len_keep, device=x.device)
            len_u_spec = int(len_keep // (L / T) * T)
            if len_u_spec > 0:
                noise_sorted[:, :len_u_spec] -= 1
            noise = torch.gather(noise_sorted, dim=1, index=ids_restore)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep

    def spatial_random_masking(self, x, l, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, 1, l, device=x.device)  # noise in [0, 1]
        noise = noise.repeat(1, L // l, 1).reshape((N, -1))

        # sample random patch from unmask bands
        if (len_keep / L * l) % 1 > 0:
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            noise_sorted = torch.sort(noise, dim=1)[0]
            noise_sorted[:, :len_keep] -= torch.rand(N, len_keep, device=x.device)
            len_u_spec = int(len_keep // (L / l) * l)
            if len_u_spec > 0:
                noise_sorted[:, :len_u_spec] -= 1
            noise = torch.gather(noise_sorted, dim=1, index=ids_restore)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep

    def forward_encoder(self, x, mask_ratio):
        # embed patches
        x = self.patch_embed(x)
        N, T, L, C = x.shape

        x = x.reshape(N, T * L, C)

        # masking: length -> length * mask_ratio
        x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        # x, mask, ids_restore, ids_keep = self.feature_random_masking(x, T, mask_ratio)
        # x, mask, ids_restore, ids_keep = self.spatial_random_masking(x, L,  mask_ratio)

        x = x.view(N, -1, C)
        # append cls token
        if self.pad_with_cls_token or self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.unsqueeze(1) + self.pos_embed_temporal.unsqueeze(2)
            pos_embed = pos_embed.reshape(1, -1, C)

            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed or self.pad_with_cls_token:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if self.cls_embed or self.pad_with_cls_token:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed or self.pad_with_cls_token:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        if self.use_learnable_pos_emb:
            x = x.view([N, -1, C]) + pos_embed
        else:
            x = x.view([N, -1, C])

        # apply Transformer blocks
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)

        if self.cls_embed or self.pad_with_cls_token:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x, mask, ids_restore, ids_keep

    def forward_decoder(self, x, ids_restore):
        if self.cls_embed or self.pad_with_cls_token:
            # remove cls token
            enc_cls_token = x[:, 0:1, :]
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]
        N = x.shape[0]
        T = self.patch_embed.b_grid_size
        H = W = self.patch_embed.grid_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        if self.pad_with_cls_token:
            mask_token = self.trans_to_decoder(enc_cls_token)
            mask_tokens = mask_token.repeat(1, T * H * W + 0 - x.shape[1], 1)
        else:
            mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([N, T * H * W, C])
        x_ = torch.gather(x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2]))
        x = x_.view([N, T * H * W, C])
        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.unsqueeze(1) + self.decoder_pos_embed_temporal.unsqueeze(2)
            decoder_pos_embed = decoder_pos_embed.reshape(1, -1, C)

            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class.expand(
                            decoder_pos_embed.shape[0], -1, -1
                        ),
                        decoder_pos_embed,
                    ],
                    1,
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        if self.use_learnable_pos_emb:
            x = x + decoder_pos_embed
        else:
            x = x

        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
        if requires_t_shape:
            x = x.view([N, T, H * W, C])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if requires_t_shape:
            x = x.view([N, T * H * W, -1])

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]
        return x

    def forward_loss(self, imgs, pred, mask):
        """
        imgs: [N, 1, B, H, W]
        pred: [N, t*h*w, u*p*p*3]
        mask: [N*t, h*w], 0 is keep, 1 is remove,
        """
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
            self.mean = mean
            self.var = (var + 1.0e-6) ** 0.5
        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def forward(self, imgs, target=None, mask_ratio=0.75):
        # N, c, b, h, w = imgs.shape
        latent, mask, ids_restore, ids_keep = self.forward_encoder(imgs, mask_ratio)
        pred = self.forward_decoder(latent, ids_restore)
        if target is not None:
            loss = self.forward_loss(target, pred, mask)
        else:
            loss = self.forward_loss(imgs, pred, mask)

        mask = mask.unsqueeze(2).repeat(1, 1, pred.shape[2])
        mask = self.unpatchify(mask)

        if self.norm_pix_loss:
            pred = pred * self.var + self.mean
        pred = self.unpatchify(pred)
        return loss, pred, mask


class DualHSIMAE(nn.Module):
    """Dual-Branch FineTuning HSIMAE"""
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        decoder_embed_dim=512,
        decoder_depth=8,
        decoder_num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        norm_pix_loss=False,
        bands=16,
        b_patch_size=4,
        no_qkv_bias=False,
        sep_pos_embed=False,
        trunc_init=False,
        cls_embed=False,
        pad_with_cls_token=False,
        use_learnable_pos_emb=False,
        num_class=100,
        ul_multi=8,
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.use_learnable_pos_emb = use_learnable_pos_emb
        self.cls_embed = cls_embed
        self.b_pred_patch_size = b_patch_size

        self.patch_embed = PatchEmbed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            bands,
            b_patch_size,
        )
        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        self.pad_with_cls_token = pad_with_cls_token

        if self.pad_with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.trans_to_decoder = nn.Linear(embed_dim, decoder_embed_dim)

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            self.decoder_cls_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed or self.pad_with_cls_token:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim),
            )

        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.decoder_embed = nn.Linear(embed_dim, decoder_embed_dim, bias=True)

        self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))

        if sep_pos_embed:
            self.decoder_pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], decoder_embed_dim)
            )
            self.decoder_pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], decoder_embed_dim)
            )
            if self.cls_embed:
                self.decoder_pos_embed_class = nn.Parameter(
                    torch.zeros(1, 1, decoder_embed_dim)
                )
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.decoder_pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, decoder_embed_dim),
            )

        self.decoder_blocks = nn.ModuleList(
            [
                Block(
                    decoder_embed_dim,
                    decoder_num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                )
                for _ in range(decoder_depth)
            ]
        )

        self.decoder_norm = norm_layer(decoder_embed_dim)
        self.decoder_pred = nn.Linear(
            decoder_embed_dim,
            self.b_pred_patch_size * patch_size ** 2 * in_chans,
            bias=True,
        )

        self.norm_pix_loss = norm_pix_loss

        self.fc_norm = norm_layer(embed_dim * self.patch_embed.b_grid_size)
        self.cls_head = nn.Linear(embed_dim * self.patch_embed.b_grid_size, num_class)

        self.ul_multi = ul_multi

        self.initialize_weights()
        print("model initialized")

    def initialize_weights(self):
        if self.pad_with_cls_token:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            torch.nn.init.trunc_normal_(self.decoder_pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.decoder_pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed_class, std=0.02)
        else:
            if self.use_learnable_pos_emb:
                torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)
                torch.nn.init.trunc_normal_(self.decoder_pos_embed, std=0.02)

        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
            torch.nn.init.trunc_normal_(self.mask_token, std=0.02)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
            torch.nn.init.normal_(self.mask_token, std=0.02)

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def patchify(self, imgs):
        N, _, T, H, W = imgs.shape
        p = self.patch_embed.patch_size[0]
        u = self.b_pred_patch_size
        assert H == W and H % p == 0 and T % u == 0
        h = w = H // p
        t = T // u

        x = imgs.reshape(shape=(N, 1, t, u, h, p, w, p))
        x = torch.einsum("nctuhpwq->nthwupqc", x)
        x = x.reshape(shape=(N, t * h * w, u * p**2 * 1))
        self.patch_info = (N, T, H, W, p, u, t, h, w)
        return x

    def unpatchify(self, x):
        N, T, H, W, p, u, t, h, w = self.patch_info

        x = x.reshape(shape=(N, t, h, w, u, p, p, 1))

        x = torch.einsum("nthwupqc->nctuhpwq", x)
        imgs = x.reshape(shape=(N, 1, T, H, W))
        return imgs

    def random_masking(self, x, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))

        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]

        # sort noise for each sample
        ids_shuffle = torch.argsort(
            noise, dim=1
        )  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore, ids_keep

    def feature_random_masking(self, x, T, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, T, 1, device=x.device)  # noise in [0, 1]
        noise = noise.repeat(1, 1, L // T).reshape((N, -1))

        # sample random patch from unmask bands
        if (len_keep / L * T) % 1 > 0:
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            noise_sorted = torch.sort(noise, dim=1)[0]
            noise_sorted[:, :len_keep] -= torch.rand(N, len_keep, device=x.device)
            len_u_spec = int(len_keep // (L / T) * T)
            if len_u_spec > 0:
                noise_sorted[:, :len_u_spec] -= 1
            noise = torch.gather(noise_sorted, dim=1, index=ids_restore)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep

    def spatial_random_masking(self, x, l, mask_ratio):
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        noise = torch.rand(N, 1, l, device=x.device)  # noise in [0, 1]
        noise = noise.repeat(1,  L // l, 1).reshape((N, -1))

        # sample random patch from unmask bands
        if (len_keep / L * l) % 1 > 0:
            ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
            ids_restore = torch.argsort(ids_shuffle, dim=1)
            noise_sorted = torch.sort(noise, dim=1)[0]
            noise_sorted[:, :len_keep] -= torch.rand(N, len_keep, device=x.device)
            len_u_spec = int(len_keep // (L / l) * l)
            if len_u_spec > 0:
                noise_sorted[:, :len_u_spec] -= 1
            noise = torch.gather(noise_sorted, dim=1, index=ids_restore)

        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)
        return x_masked, mask, ids_restore, ids_keep

    def forward_mask_encoder(self, x, mask_ratio):
        x = self.patch_embed(x)
        N, T, L, C = x.shape

        x = x.reshape(N, T * L, C)

        # masking: length -> length * mask_ratio
        # x, mask, ids_restore, ids_keep = self.random_masking(x, mask_ratio)
        # x, mask, ids_restore, ids_keep = self.feature_random_masking(x, T, mask_ratio)
        x, mask, ids_restore, ids_keep = self.spatial_random_masking(x, L,  mask_ratio)

        x = x.view(N, -1, C)
        # append cls token
        if self.pad_with_cls_token or self.cls_embed:
        # if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed = self.pos_embed_spatial.unsqueeze(1) + self.pos_embed_temporal.unsqueeze(2)
            pos_embed = pos_embed.reshape(1, -1, C)

            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed or self.pad_with_cls_token:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        else:
            if self.cls_embed or self.pad_with_cls_token:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind:, :].expand(x.shape[0], -1, -1)
            pos_embed = torch.gather(
                pos_embed,
                dim=1,
                index=ids_keep.unsqueeze(-1).repeat(1, 1, pos_embed.shape[2]),
            )
            if self.cls_embed or self.pad_with_cls_token:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )
        if self.use_learnable_pos_emb:
            x = x.view([N, -1, C]) + pos_embed
        else:
            x = x.view([N, -1, C])

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)

        if self.cls_embed or self.pad_with_cls_token:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]
        return x, mask, ids_restore, ids_keep

    def forward_encoder(self, x):
        # embed patches
        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        x = self.patch_embed(x)
        N, T, L, C = x.shape

        x = x.view(N, -1, C)

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed_temporal = self.pos_embed_temporal
            pos_embed = self.pos_embed_spatial.unsqueeze(1) + pos_embed_temporal.unsqueeze(2)
            pos_embed = pos_embed.reshape(1, -1, C)
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )

        elif self.con_pos_embed:
            pos_embed = 0

        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind: (self.num_patches + cls_ind), :].expand(x.shape[0], -1, -1)
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )

        if self.use_learnable_pos_emb:
            x = x.view([N, -1, C]) + pos_embed
        else:
            x = x.view([N, -1, C])

        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)

        cls = x[:, 0, :]
        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]
        return x, cls

    def forward_decoder(self, x, ids_restore):
        if self.cls_embed or self.pad_with_cls_token:
            # remove cls token
            enc_cls_token = x[:, 0:1, :]
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]
        N = x.shape[0]
        T = self.patch_embed.b_grid_size
        H = W = self.patch_embed.grid_size

        # embed tokens
        x = self.decoder_embed(x)
        C = x.shape[-1]

        # append mask tokens to sequence
        if self.pad_with_cls_token:
            mask_token = self.trans_to_decoder(enc_cls_token)
            mask_tokens = mask_token.repeat(1, T * H * W + 0 - x.shape[1], 1)
        else:
            mask_tokens = self.mask_token.repeat(N, T * H * W + 0 - x.shape[1], 1)
        x_ = torch.cat([x[:, :, :], mask_tokens], dim=1)  # no cls token
        x_ = x_.view([N, T * H * W, C])
        x_ = torch.gather(
            x_, dim=1, index=ids_restore.unsqueeze(-1).repeat(1, 1, x_.shape[2])
        )  # unshuffle
        x = x_.view([N, T * H * W, C])
        # append cls token
        if self.cls_embed:
            decoder_cls_token = self.decoder_cls_token
            decoder_cls_tokens = decoder_cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((decoder_cls_tokens, x), dim=1)

        if self.sep_pos_embed:
            decoder_pos_embed = self.decoder_pos_embed_spatial.unsqueeze(1) + self.decoder_pos_embed_temporal.unsqueeze(2)
            decoder_pos_embed = decoder_pos_embed.reshape(1, -1, C)
            if self.cls_embed:
                decoder_pos_embed = torch.cat(
                    [
                        self.decoder_pos_embed_class.expand(
                            decoder_pos_embed.shape[0], -1, -1
                        ),
                        decoder_pos_embed,
                    ],
                    1,
                )
        else:
            decoder_pos_embed = self.decoder_pos_embed[:, :, :]

        # add pos embed
        if self.use_learnable_pos_emb:
            x = x + decoder_pos_embed
        else:
            x = x

        attn = self.decoder_blocks[0].attn
        requires_t_shape = hasattr(attn, "requires_t_shape") and attn.requires_t_shape
        if requires_t_shape:
            x = x.view([N, T, H * W, C])

        # apply Transformer blocks
        for blk in self.decoder_blocks:
            x = blk(x)
        x = self.decoder_norm(x)

        # predictor projection
        x = self.decoder_pred(x)

        if requires_t_shape:
            x = x.view([N, T * H * W, -1])

        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]

        return x

    def forward_loss(self, imgs, pred, mask):
        target = self.patchify(imgs)
        if self.norm_pix_loss:
            mean = target.mean(dim=-1, keepdim=True)
            var = target.var(dim=-1, keepdim=True)
            target = (target - mean) / (var + 1.0e-6) ** 0.5
            self.mean = mean
            self.var = (var + 1.0e-6) ** 0.5

        loss = (pred - target) ** 2
        loss = loss.mean(dim=-1)  # [N, L], mean loss per patch
        mask = mask.view(loss.shape)

        loss = (loss * mask).sum() / mask.sum()  # mean loss on removed patches
        return loss

    def head(self, x, type='AGG'):
        N, T, L, C = self.patch_embed.output_size
        if type == 'GAP':
            x = x.reshape(N, -1, C)
        elif type == 'AGG':
            x = x.reshape(N, T, L, C)
            x = x.permute(0, 2, 1, 3).reshape(N, L, -1)
        x = x.mean(1)
        x = self.fc_norm(x)
        pred = self.cls_head(x)
        return pred, x

    def forward(self, imgs, imgs_u=None, mask_ratio=0.75):
        # N, c, b, h, w = imgs.shape
        if imgs_u is not None:
            imgs_all = torch.concat([imgs, imgs_u], dim=0)
        else:
            imgs_all = imgs
        latent_unmask, mask, ids_restore, ids_keep = self.forward_mask_encoder(imgs_all, mask_ratio)
        pred_rec = self.forward_decoder(latent_unmask, ids_restore)
        loss_rec = self.forward_loss(imgs_all, pred_rec, mask)

        mask = mask.unsqueeze(2).repeat(1, 1, pred_rec.shape[2])
        mask = self.unpatchify(mask)

        if self.norm_pix_loss:
            pred_rec = pred_rec * self.var + self.mean
        pred_rec = self.unpatchify(pred_rec)

        latent, cls = self.forward_encoder(imgs)
        class_pred, latent = self.head(latent)
        return loss_rec, pred_rec, mask, class_pred


class HSIViT(nn.Module):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        in_chans=3,
        embed_dim=1024,
        depth=24,
        num_heads=16,
        mlp_ratio=4.0,
        norm_layer=nn.LayerNorm,
        bands=16,
        b_patch_size=4,
        num_class=100,
        no_qkv_bias=False,
        sep_pos_embed=False,
        trunc_init=False,
        cls_embed=False,
        drop_rate=0.,
        drop_path=0.,
        use_learnable_pos_emb=False,
    ):
        super().__init__()
        self.trunc_init = trunc_init
        self.sep_pos_embed = sep_pos_embed
        self.use_learnable_pos_emb = use_learnable_pos_emb
        self.cls_embed = cls_embed
        self.b_pred_patch_size = b_patch_size

        self.patch_embed = PatchEmbed(
            img_size,
            patch_size,
            in_chans,
            embed_dim,
            bands,
            b_patch_size,
        )

        num_patches = self.patch_embed.num_patches
        input_size = self.patch_embed.input_size
        self.input_size = input_size

        if self.cls_embed:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        if sep_pos_embed:
            self.pos_embed_spatial = nn.Parameter(
                torch.zeros(1, input_size[1] * input_size[2], embed_dim)
            )
            self.pos_embed_temporal = nn.Parameter(
                torch.zeros(1, input_size[0], embed_dim)
            )
            if self.cls_embed:
                self.pos_embed_class = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            if self.cls_embed:
                _num_patches = num_patches + 1
            else:
                _num_patches = num_patches

            self.pos_embed = nn.Parameter(
                torch.zeros(1, _num_patches, embed_dim),
            )

            self.num_patches = _num_patches

        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    embed_dim,
                    num_heads,
                    mlp_ratio,
                    qkv_bias=not no_qkv_bias,
                    qk_scale=None,
                    norm_layer=norm_layer,
                    drop=drop_rate,
                    drop_path=dpr[i],
                )
                for i in range(depth)
            ]
        )

        self.fc_norm = norm_layer(embed_dim * self.patch_embed.b_grid_size)

        self.pooling = nn.AdaptiveAvgPool1d(1)

        self.cls_head = nn.Linear(embed_dim * self.patch_embed.b_grid_size, num_class)
        self.initialize_weights()

        print("model initialized")

    def initialize_weights(self):
        if self.cls_embed:
            torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        if self.sep_pos_embed:
            torch.nn.init.trunc_normal_(self.pos_embed_spatial, std=0.02)
            torch.nn.init.trunc_normal_(self.pos_embed_temporal, std=0.02)

            if self.cls_embed:
                torch.nn.init.trunc_normal_(self.pos_embed_class, std=0.02)
        else:
            if self.use_learnable_pos_emb:
                torch.nn.init.trunc_normal_(self.pos_embed, std=0.02)

        w = self.patch_embed.proj.weight.data
        if self.trunc_init:
            torch.nn.init.trunc_normal_(w)
        else:
            torch.nn.init.xavier_uniform_(w.view([w.shape[0], -1]))

        # initialize nn.Linear and nn.LayerNorm
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            if self.trunc_init:
                nn.init.trunc_normal_(m.weight, std=0.02)
            else:
                torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_encoder(self, x):
        # embed patches
        if len(x.shape) == 4:
            x = x.unsqueeze(1)

        x = self.patch_embed(x)
        N, T, L, C = x.shape

        x = x.view(N, -1, C)

        # append cls token
        if self.cls_embed:
            cls_token = self.cls_token
            cls_tokens = cls_token.expand(x.shape[0], -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        # add pos embed w/o cls token
        if self.sep_pos_embed:
            pos_embed_temporal = self.pos_embed_temporal
            pos_embed = self.pos_embed_spatial.unsqueeze(1) + pos_embed_temporal.unsqueeze(2)
            pos_embed = pos_embed.reshape(1, -1, C)
            pos_embed = pos_embed.expand(x.shape[0], -1, -1)
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed_class.expand(pos_embed.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )

        elif self.con_pos_embed:
            pos_embed = 0

        else:
            if self.cls_embed:
                cls_ind = 1
            else:
                cls_ind = 0
            pos_embed = self.pos_embed[:, cls_ind: (self.num_patches + cls_ind), :].expand(x.shape[0], -1, -1)
            if self.cls_embed:
                pos_embed = torch.cat(
                    [
                        self.pos_embed[:, :1, :].expand(x.shape[0], -1, -1),
                        pos_embed,
                    ],
                    1,
                )

        if self.use_learnable_pos_emb:
            x = x.view([N, -1, C]) + pos_embed
        else:
            x = x.view([N, -1, C])

        # apply Transformer blocks
        for i, blk in enumerate(self.blocks):
            x = blk(x)

        cls = x[:, 0, :]
        if self.cls_embed:
            # remove cls token
            x = x[:, 1:, :]
        else:
            x = x[:, :, :]
        return x, cls

    def head(self, x, type='AGG'):
        N, T, L, C = self.patch_embed.output_size
        if type == 'GAP':
            x = x.reshape(N, -1, C)
            x = x.permute(0, 2, 1)
        elif type == 'AGG':
            x = x.reshape(N, T, L, C)
            x = x.permute(0, 1, 3, 2).reshape(N, -1, L)
        x = self.pooling(x)
        x = torch.flatten(x, 1)
        x = self.fc_norm(x)
        pred = self.cls_head(x)
        return pred, x

    def forward(self, imgs):
        latent, cls = self.forward_encoder(imgs)
        pred, latent = self.head(latent)
        return pred