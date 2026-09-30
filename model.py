import torch
import torch.nn as nn
from torch.nn import functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super().__init__()
        self.c_attn = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.c_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.num_heads = num_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        B, T, C = x.size()
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.embed_dim, dim=2)

        head_dim = C // self.num_heads

        q = q.view(B, T, self.num_heads, head_dim).transpose(1, 2)
        k = k.view(B, T, self.num_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, self.num_heads, head_dim).transpose(1, 2)

        y = F.scaled_dot_product_attention(
            q, k, v, is_causal=True
        )

        y = (
            y.transpose(1, 2)
            .contiguous()
            .view(B, T, C)
        )

        return self.c_proj(y)


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()

        self.ln_1 = nn.LayerNorm(embed_dim)
        self.attn = CausalSelfAttention(embed_dim, num_heads)

        self.ln_2 = nn.LayerNorm(embed_dim)
        self.ffwd = nn.Sequential(
            nn.Linear(embed_dim, 4 * embed_dim, bias=False),
            nn.GELU(),
            nn.Linear(4 * embed_dim, embed_dim, bias=False),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.ln_1(x))
        x = x + self.ffwd(self.ln_2(x))
        return x


class MiniLLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int = 512,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_layers: int = 12,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.max_seq_len = max_seq_len

        self.token_embedding = nn.Embedding(
            vocab_size, embed_dim
        )
        self.position_embedding = nn.Embedding(
            max_seq_len, embed_dim
        )

        self.blocks = nn.Sequential(
            *[
                TransformerBlock(
                    embed_dim,
                    num_heads,
                    dropout,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_f = nn.LayerNorm(embed_dim)
        self.lm_head = nn.Linear(
            embed_dim,
            vocab_size,
            bias=False,
        )

        # Weight tying.
        self.token_embedding.weight = self.lm_head.weight

    def forward(self, idx, targets=None):
        B, T = idx.size()

        if T > self.max_seq_len:
            raise ValueError(
                f"Sequence length {T} exceeds "
                f"maximum context length {self.max_seq_len}."
            )

        pos = torch.arange(
            0,
            T,
            dtype=torch.long,
            device=idx.device,
        )

        x = (
            self.token_embedding(idx)
            + self.position_embedding(pos)
        )

        x = self.blocks(x)

        logits = self.lm_head(self.ln_f(x))

        loss = None

        if targets is not None:
            loss = F.cross_entropy(
                logits.view(B * T, -1),
                targets.view(B * T),
            )

        return logits, loss

    @torch.no_grad()
    def generate(
        self,
        idx,
        max_new_tokens,
        temperature=0.7,
        top_k=40,
    ):
        self.eval()

        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.max_seq_len:]

            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]

            logits = logits / temperature

            if top_k is not None:
                v, _ = torch.topk(
                    logits,
                    min(top_k, logits.size(-1)),
                )
                logits[
                    logits < v[:, [-1]]
                ] = -float("Inf")

            probs = F.softmax(logits, dim=-1)

            idx_next = torch.multinomial(
                probs,
                num_samples=1,
            )

            idx = torch.cat(
                (idx, idx_next),
                dim=1,
            )

        self.train()
        return idx
