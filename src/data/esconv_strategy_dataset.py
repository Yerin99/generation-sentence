"""
ESConvStrategySequenceDataset
─────────────────────────────
ESConv 원본 대화(JSON) → 시스템(turn) 전략 id 시퀀스를 추출하여
SASRec 학습용 (seq, target) 샘플을 생성한다.

SRP:
1) 데이터 로딩 및 캐싱
2) 전략 id 시퀀스 → (고정 길이) 입력/타겟 텐서 변환
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import List, Dict, Tuple

import torch
from datasets import load_dataset
import warnings

from utils.strategy import STRATEGIES, STR2ID, to_refined  # 재활용

# ★ 추가: 전략 문자열 ↔ 정수 ID 매핑
STRATEGY2ID = {s: i for i, s in enumerate(STRATEGIES)}

# ────────────────────────────────────────────────
# Special IDs
# ────────────────────────────────────────────────
# 기존: 0~7 = 전략 id, 8 = PAD
# 추가: 9 = NO_HISTORY (빈 시퀀스용)

PAD_ID: int = len(STRATEGIES)          # 8  (padding 전용)
NO_HISTORY_ID: int = PAD_ID + 1        # 9  (cold-start용)
N_ITEMS: int = NO_HISTORY_ID + 1       # 10 (0~7 전략 + 8 pad + 9 no_history)


class ESConvStrategySequenceDataset(torch.utils.data.Dataset):
    """
    Args
    ----
    split           : "train" | "validation" | "test"
    max_seq_len     : Transformer 입력 길이
    cache_dir       : 전처리한 pt 파일 저장 경로
    """

    # 클래스 속성 추가 - train_sasrec_strategy.py에서 접근 가능하게 함
    STRATEGIES = STRATEGIES
    PAD_ID = PAD_ID
    NO_HISTORY_ID = NO_HISTORY_ID
    N_ITEMS = N_ITEMS

    def __init__(self,
                 split: str,
                 max_seq_len: int = 50,
                 cache_dir: str = "cache_sasrec",
                 tiny_frac: float | None = None):
        super().__init__()
        self.max_seq_len = max_seq_len

        cache_f = Path(cache_dir) / f"{split}_{max_seq_len}_{tiny_frac}.pt"
        cache_f.parent.mkdir(parents=True, exist_ok=True)
        if cache_f.exists():
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=FutureWarning, message="You are using `torch.load`")
                self.samples = torch.load(cache_f, map_location="cpu")
            return

        raw = load_dataset("thu-coai/esconv", split=split)
        if tiny_frac:
            raw = raw.shuffle(seed=42).select(range(int(len(raw) * tiny_frac)))

        self.samples: List[Tuple[List[int], int]] = []

        for ex in raw:
            dialog = json.loads(ex["text"])["dialog"]

            # 하나의 대화를 하나의 'user session' 으로 취급
            strat_seq: List[int] = []
            for t in dialog:
                if t["speaker"] != "sys":
                    continue
                cur_ref = to_refined(t.get("strategy", "Others"))
                strat_seq.append(STR2ID.get(cur_ref, STR2ID["Others"]))

            # 첫 turn(빈 history) 포함 모든 위치에서 샘플 생성
            for idx in range(len(strat_seq)):
                hist = strat_seq[:idx][-max_seq_len:]

                # 빈 시퀀스 → NO_HISTORY_ID 한 개만 넣기
                if len(hist) == 0:
                    hist = [NO_HISTORY_ID]

                # left-padding
                hist_pad = [PAD_ID] * (max_seq_len - len(hist)) + hist
                target = strat_seq[idx]
                self.samples.append((hist_pad, target))

        torch.save(self.samples, cache_f)

    # ───────── torch.Dataset 인터페이스 ─────────
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        seq, target = self.samples[idx]
        return {
            "seq": torch.tensor(seq, dtype=torch.long),
            "target": torch.tensor(target, dtype=torch.long),
        } 