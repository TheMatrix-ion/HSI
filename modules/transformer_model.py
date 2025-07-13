import torch
import torch.nn as nn
import torch.nn.functional as F

class TransformerEncoderLayer(nn.Module):
    def __init__(self, embed_dim, n_heads=4, ff_dim=128, dropout=0.1):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, n_heads, dropout=dropout, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ff_dim),
            nn.ReLU(),
            nn.Linear(ff_dim, embed_dim)
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # Multi-head Self Attention
        attn_output, _ = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        # Feed-forward Network
        ff_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ff_output))
        return x

class TransformerClusterNet(nn.Module):
    def __init__(self, embed_dim=64, output_dim=64, n_layers=2):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, embed_dim)
        self.encoder = nn.Sequential(*[TransformerEncoderLayer(embed_dim) for _ in range(n_layers)])
        self.output_proj = nn.Linear(embed_dim, output_dim)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : Tensor
            Input features. Can be of shape ``(N, D)`` or ``(B, L, D)`` where
            ``B`` is batch size, ``L`` sequence length and ``D`` embedding
            dimension.

        Returns
        -------
        Tensor
            Output with the same batch and sequence dimensions as ``x`` and
            last dimension ``output_dim``.
        """

        orig_2d = False
        if x.dim() == 2:
            # (N, D) -> (N, 1, D) to reuse the same encoder
            x = x.unsqueeze(1)
            orig_2d = True

        x = self.input_proj(x)
        x = self.encoder(x)
        x = self.output_proj(x)

        if orig_2d:
            return x.squeeze(1)
        return x
