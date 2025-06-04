#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
MultiESCStrategySequenceDataset
-------------------------------
MultiESC/data/{train|validation|test}.txt 파일을 읽어
(SASRec 입력 시퀀스, 타깃 전략) 쌍을 생성합니다.
"""

import json
from pathlib import Path
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

# ────────────────────────────────────────────────
# MultiESC 고유 전략 정의 (8개 + PAD)
# ────────────────────────────────────────────────
STRATEGIES = [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of Feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions or Information",
    "Greetings",
    "Others",
]

# id ↔ name 매핑
STR2ID = {s: i for i, s in enumerate(STRATEGIES)}
ID2STR = {i: s for s, i in enumerate(STRATEGIES)}

# padding / item 수
PAD_ID: int = len(STRATEGIES)           # 8
N_ITEMS: int = PAD_ID + 1               # 9

# ────────────────────────────────────────────────
# MultiESC 원본 문자열 → 위 정의한 canonical 이름 매핑
# (소문자로 통일하여 대소문자 차이 제거)
# ────────────────────────────────────────────────
_MAPPINGS = {
    "question": "Question",
    "restatement or paraphrasing": "Restatement or Paraphrasing",
    "reflection of feelings": "Reflection of Feelings",
    "self-disclosure": "Self-disclosure",
    "affirmation and reassurance": "Affirmation and Reassurance",
    "providing suggestions or information": "Providing Suggestions or Information",
    "greetings": "Greetings",       # 복수형
    "greeting":  "Greetings",       # 단수형
    "others": "Others",
}

def _map_strategy(raw: str) -> str:
    """원본 전략명을 canonical 이름으로 변환(미정의 시 Others)"""
    return _MAPPINGS.get(raw.lower(), "Others")


class MultiESCStrategySequenceDataset(Dataset):
    """
    MultiESC 데이터셋에서 system turn 의 strategy sequence를 추출해
    (이전 전략 시퀀스, 현재 전략) 예측 문제로 변환.
    """

    # ──────────────  training 스크립트에서 접근할 상수 ──────────────
    STRATEGIES = STRATEGIES
    PAD_ID     = PAD_ID
    N_ITEMS    = N_ITEMS

    def __init__(self,
                 split: str,
                 max_seq_len: int = 50,
                 tiny_frac: float | None = None) -> None:
        """
        Args
        ----
        split        : train / validation / test
        max_seq_len  : SASRec 입력 시퀀스 최대 길이
        tiny_frac    : 0~1 사이 값일 때, 데이터 일부만 사용(디버깅용)
        """
        assert split in {"train", "validation", "test"}
        self.max_seq_len = max_seq_len
        self.samples: List[Tuple[List[int], int]] = []  # (seq, target)

        self._load(split)

        if tiny_frac:
            n_keep = int(len(self.samples) * tiny_frac)
            self.samples = self.samples[: max(1, n_keep)]

    # ---------------------------------------- private
    def _load(self, split: str) -> None:
        """
        MultiESC 데이터셋은 validation 대신 valid 이름을 사용함
        """
        # "validation" -> "valid" 로 변환
        if split == "validation":
            file_split = "valid"
        else:
            file_split = split
        
        path = Path(f"MultiESC/data/{file_split}.txt")
        if not path.exists():
            raise FileNotFoundError(f"{path} not found")

        with path.open(encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                example = json.loads(line)
                self._extract_samples_from_dialog(example.get("dialog", []))

    def _extract_samples_from_dialog(self, dialog: list) -> None:
        sys_strats: List[int] = []
        
        for turn in dialog:
            if turn.get("speaker") != "sys":
                continue
            raw_strat = turn.get("strategy") or turn.get("all_strategy", [None])[0]
            if raw_strat is None:
                continue
            
            canon = _map_strategy(raw_strat)
            strat_id = STR2ID[canon]
            sys_strats.append(strat_id)

        # 전략 시퀀스 → (과거 seq, 현 전략) 쌍으로 분할
        for idx in range(1, len(sys_strats)):
            seq = sys_strats[:idx]
            # padding / truncation (왼쪽 pad)
            seq = seq[-self.max_seq_len:]
            pad_len = self.max_seq_len - len(seq)
            seq = [PAD_ID] * pad_len + seq

            target = sys_strats[idx]
            self.samples.append((seq, target))

    # ---------------------------------------- torch required
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        seq, tgt = self.samples[idx]
        return {
            "seq": torch.tensor(seq, dtype=torch.long),
            "target": torch.tensor(tgt, dtype=torch.long),
        } 