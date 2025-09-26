# Attention layers
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat, pack, unpack
from einops.layers.torch import Rearrange


class FeedForward(nn.Module):
    """
    Adapted from vit_pytorch.
    """

    def __init__(self, dim, hidden_dim, dropout=0.0):

        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):

        return self.net(x)


class Attention(nn.Module):
    """
    Adapted from vit_pytorch.
    """

    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0):

        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)
        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):

        x = self.norm(x)
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    """
    Adapted from vit_pytorch
    """

    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.0):

        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads=heads,
                          dim_head=dim_head, dropout=dropout),
                FeedForward(dim, mlp_dim, dropout=dropout)
            ]))

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ScaleFusionTransformer(nn.Module):
    """
    Adapted from vit_pytorch.
    """

    def __init__(
        self,
        num_scales,
        feature_dim,
        dim,
        depth,
        heads,
        mlp_dim,
        dim_head=64,
        dropout=0.0,
        emb_dropout=0.0,
    ):

        super().__init__()
        self.to_feat_embedding = nn.Sequential(
            Rearrange('b s v e -> (b v) s e', e=feature_dim, s=num_scales),
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, dim),
            nn.LayerNorm(dim),
        )

        self.scale_embedding = nn.Parameter(
            torch.randn(1, num_scales + 2, dim))
        self.cls_token = nn.Parameter(torch.randn(dim))
        self.rad_token = nn.Parameter(torch.randn(dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(
            dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, 1)
        )

    def forward(self, features: torch.Tensor, batch: torch.Tensor, **kwds):
        """
        Input features must be of shape (N, C)
        """

        V = kwds["nverts"]
        E = features.shape[-1]
        B = batch.max().item() + 1

        features = features.view(B, -1, V, E)  # (batch, scales, verts, embed)
        x = self.to_feat_embedding(features)
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, 'd -> b d', b=b)
        rad_tokens = repeat(self.rad_token, 'd -> b d', b=b)

        x, ps = pack([cls_tokens, rad_tokens, x], "b * d")

        x += self.scale_embedding[:, :(n + 2)]
        x = self.dropout(x)

        x = self.transformer(x)

        cls_tokens, rad_tokens, _ = unpack(x, ps, "b * d")
        cls_tokens = cls_tokens.view(B, V, -1)
        rad_tokens = rad_tokens.view(B, V, -1)

        return cls_tokens, rad_tokens


class MLP(nn.Module):
    """
    MLP Layer used as projection head.
    """

    def __init__(
        self,
        input_dim: int,
        num_hidden: int = 1,
        embed_dim: int = 64,
        out_dim: int = 64,
        actv: str = "relu",
    ):

        super(MLP, self).__init__()
        layers = [nn.Linear(input_dim, embed_dim)]
        for _ in range(num_hidden):
            if actv == "relu":
                layers += [nn.Linear(embed_dim, embed_dim), nn.ReLU()]
            elif actv == "elu":
                layers += [nn.Linear(embed_dim, embed_dim), nn.ELU()]
            elif actv == "gelu":
                layers += [nn.Linear(embed_dim, embed_dim), nn.GELU()]
        layers += [nn.Linear(embed_dim, out_dim)]

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor):

        return self.layers(x)


class ScaleAttnFusionLayer(nn.Module):
    """
    Fuse features on different scales into one using attention mechanism. For
    a node on the graph, its features across different scales first attend to
    each other, then get projected to logits. These logits are then used as
    weights to sum up features on all scales.
    Args:
        embed_dim(int): Embedded node feature length. Default: 64.
    """

    def __init__(
        self,
        embed_dim: int = 64,
        num_heads: int = 4,
    ):

        super(ScaleAttnFusionLayer, self).__init__()

        # Use MLP to project features to logits
        self.embed_dim = embed_dim
        self.out_proj = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 1)
        )

        self.MH_Attn = nn.MultiheadAttention(
            embed_dim=self.embed_dim,
            num_heads=num_heads,
            batch_first=False,
        )
        self.norm = nn.LayerNorm(normalized_shape=embed_dim)

    def __call__(self, features: torch.Tensor, batch: torch.Tensor, **kwds):
        """
        Input features must be of dimension (N, C).
        """

        V = kwds["nverts"]  # We must know the number of nodes on the graph
        E = features.shape[-1]
        B = batch.max().item() + 1

        fused_list = []
        attnw_list = []
        for b in range(B):
            index = torch.argwhere(batch == b).squeeze(-1)
            sub_feat = features.index_select(dim=0, index=index)
            sub_feat = sub_feat.view(-1, V, E)  # (num_scales, V, E)
            attn_out, attn_w = self.MH_Attn(sub_feat, sub_feat, sub_feat)
            attn_out = attn_out + sub_feat  # Implements Residual connection
            attn_out = self.norm(attn_out)

            # Feed forward to output logits: (num_scales, V)
            logits = self.out_proj(attn_out).squeeze()
            weights = F.softmax(logits, dim=0)
            fused = torch.einsum("sv,sve->ve", weights, sub_feat)  # (V, E)
            fused_list.append(fused)
            attnw_list.append(attn_w)
        out = torch.stack(fused_list, dim=0)  # (B, V, E)

        return out


class ScaleMLPFusionLayer(nn.Module):
    """
    Fuse features on different scales into one by gating mechanism. For a node
    on the graph, its features across different scales are first projected using
    MLP to logits. These logits are then normalized as weights to sum up features
    on all scales.
    Args:
        embed_dim(int): Embedded node feature length. Default: 64.
    """

    def __init__(
        self,
        embed_dim: int = 64,
    ):

        super(ScaleMLPFusionLayer, self).__init__()

        # Use MLP to project features to logits
        self.embed_dim = embed_dim
        self.out_proj = MLP(embed_dim, 1, embed_dim, 1)

    def __call__(self, features: torch.Tensor, batch: torch.Tensor, **kwds):
        """
        Input features must be of dimension (N, C).
        """

        V = kwds["nverts"]  # We must know the number of nodes on the graph
        E = features.shape[-1]
        B = batch.max().item() + 1

        fused_list = []
        weights_list = []
        for b in range(B):
            index = torch.argwhere(batch == b).squeeze(-1)
            sub_feat = features.index_select(dim=0, index=index)
            sub_feat = sub_feat.view(-1, V, E)  # (num_scales, V, E)

            # Feed forward to output logits: (num_scales, V)
            logits = self.out_proj(sub_feat).squeeze()  # (num_scales, V)
            weights = F.sigmoid(logits)  # Normalize to 0-1
            fused = torch.einsum("sv,sve->ve", weights, sub_feat)  # (V, E)
            fused_list.append(fused)
            weights_list.append(weights)
        out = torch.stack(fused_list, dim=0)  # (B, V, E)

        return out, weights_list


class ScaleSimpleMeanLayer(nn.Module):
    """
    Fuse features on different scales into one by simply adding them up. Serve
    as a baseline layer to other feature fusion layers.
    """

    def __init__(self):

        super(ScaleSimpleMeanLayer, self).__init__()

    def __call__(self, features: torch.Tensor, batch: torch.Tensor, **kwds):
        """
        Input features must be of dimension (N, C).
        """

        V = kwds["nverts"]  # We must know the number of nodes on the graph
        E = features.shape[-1]
        B = batch.max().item() + 1

        fused_list = []
        for b in range(B):
            index = torch.argwhere(batch == b).squeeze(-1)
            sub_feat = features.index_select(dim=0, index=index)
            sub_feat = sub_feat.view(-1, V, E)  # (num_scales, V, E)

            # Direct summation
            fused = torch.mean(sub_feat, dim=0)
            fused_list.append(fused)
        out = torch.stack(fused_list, dim=0)  # (B, V, E)

        return out


class PositionalEncoding(nn.Module):

    def __init__(self, d_model: int, dropout: float = 0.1, max_length: int = 5000):

        super(PositionalEncoding, self).__init__()

        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_length, d_model)
        k = torch.arange(0, max_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2)
                             * -(math.log(2 * 10000.0) / d_model))

        pe[:, 0::2] = torch.sin(k * div_term)
        pe[:, 1::2] = torch.cos(k * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor):

        x = x + self.pe[:, : x.size(1)].requires_grad_(False)

        return self.dropout(x)


if __name__ == "__main__":

    s = ScaleFusionTransformer(
        10, 64, 128, 2, 4, 256, 128, 0.2, 0.2
    ).to("cuda:0")

    t = torch.randn((4, 10, 642, 64)).to("cuda:0")
    t = t.view(-1, 64)  # a large graph

    b = torch.ones((5)) * 3

    c, r = s(t, b.to(torch.int32).to("cuda:0"), nverts=642)
