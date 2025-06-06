"""
SASRecForStrategy
─────────────────
Transformer 기반 sequential recommender (SASRec) 구현.
입력 시퀀스(전략 id) → 다음 전략 id 분포(logits) 출력.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class SASRecForStrategy(nn.Module):
    """
    Parameters
    ----------
    n_items        : 아이템(전략) 개수 (+PAD)
    hidden_size    : 임베딩 & Transformer 차원
    max_seq_len    : 입력 시퀀스 길이
    n_heads        : Multi-Head Attention head 수
    n_layers       : Transformer layer 수
    dropout_rate   : 드롭아웃 확률
    """

    def __init__(self,
                 n_items: int,
                 hidden_size: int = 128,
                 max_seq_len: int = 50,
                 n_heads: int = 2,
                 n_layers: int = 2,
                 dropout_rate: float = 0.2,
                 pad_id: int = 8,
                 no_hist_id: int | None = None):
        super().__init__()
        self.item_emb = nn.Embedding(n_items, hidden_size, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(max_seq_len, hidden_size)
        self.dropout = nn.Dropout(dropout_rate)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=n_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout_rate,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.hidden_size = hidden_size
        self.max_seq_len = max_seq_len
        self.pad_id = pad_id
        self.output_layer = nn.Linear(hidden_size, n_items, bias=False)

        # 초기화
        nn.init.normal_(self.item_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight, std=0.02)
        nn.init.xavier_uniform_(self.output_layer.weight)
        
        # no_history_id 저장
        self.no_hist_id = no_hist_id

    # ────────────────────────────────────────────
    # Forward
    # ────────────────────────────────────────────
    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        seq: (B, L)  - padded 전략 id 시퀀스
        return: (B, n_items)  - 다음 전략 logits
        """
        device = seq.device
        B, L = seq.size()
        pos_ids = torch.arange(L, device=device).unsqueeze(0).expand(B, L)

        h = self.item_emb(seq) + self.pos_emb(pos_ids)         # (B, L, H)
        h = self.dropout(h)
        attn_mask = (seq == self.pad_id)                       # padding True

        h = self.encoder(h, src_key_padding_mask=attn_mask)
        h = self.layer_norm(h)

        # 마지막 실 아이템 위치(hidden) 추출
        seq_lengths = (seq != self.pad_id).sum(dim=1)          # (B,)
        
        # 안전성 검사: 모든 시퀀스가 적어도 하나의 실제 아이템을 가지도록 함
        seq_lengths = torch.max(seq_lengths, torch.ones_like(seq_lengths))  # 최소 1 보장
        
        last_idx = (seq_lengths - 1).clamp(min=0).unsqueeze(1).unsqueeze(2)
        last_idx = last_idx.expand(-1, 1, self.hidden_size)    # (B,1,H)
        last_hidden = h.gather(1, last_idx).squeeze(1)         # (B,H)

        logits = self.output_layer(last_hidden)                # (B, n_items)
        return logits 