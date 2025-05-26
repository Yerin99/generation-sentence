# -*- coding: utf-8 -*-
"""
utils.strategy
──────────────
• 디코더가 생성한 시퀀스에서 전략 id를 추출
• 전략 프리픽스를 제거해 순수 응답만 남기기
"""
from __future__ import annotations
from typing import List, Optional

STRATEGIES = [
    "Question",
    "Restatement or Paraphrasing",
    "Reflection of feelings",
    "Self-disclosure",
    "Affirmation and Reassurance",
    "Providing Suggestions",
    "Information",
    "Others",
]
STR2ID = {s: i for i, s in enumerate(STRATEGIES)}
ID2STR = {i: s for s, i in STR2ID.items()}
STRAT_TOKENS = [f"[STRAT_{s.replace(' ', '_')}]" for s in STRATEGIES]


# ────────────────────────────────────────────────────────────────────────
# 1) 전략 id 추출
# ────────────────────────────────────────────────────────────────────────
def get_token_id_map(tokenizer):
    """STRAT 토큰 id lookup (캐시)"""
    # 토큰 → ID 변환
    ids = tokenizer.convert_tokens_to_ids(STRAT_TOKENS)
    
    # 올바른 매핑: 토큰 ID → 전략 인덱스
    return {ids[i]: i for i in range(len(STRATEGIES))}

def parse_strategy_from_ids(pred_ids: List[int],
                            tokenizer,
                            mode: str) -> Optional[int]:
    """
    token mode : bos 이후 첫번째로 등장하는 전략 특수토큰을 탐색
    natural    : ':' 이전 문자열을 lower-case 비교
    """
    if mode == "token":
        tid2sid = get_token_id_map(tokenizer)
        for tid in pred_ids[1:]:          # bos(0번째) 건너뜀
            if tid in tid2sid:
                return tid2sid[tid]
        return None
    # ----- natural -----
    txt = safe_decode(pred_ids[1:30], tokenizer, skip_special_tokens=True)
    if ":" not in txt:
        return None
    prefix = txt.split(":", 1)[0].strip().lower()
    for s in STRATEGIES:                       # 완전·부분 일치 허용
        sl = s.lower()
        if sl == prefix or sl in prefix or prefix in sl:
            return STR2ID[s]
    return None


# ────────────────────────────────────────────────────────────────────────
# 2) 전략 프리픽스 제거 → 순수 응답 텍스트
# ────────────────────────────────────────────────────────────────────────
def strip_strategy_prefix(text: str, mode: str) -> str:
    if mode == "token":
        if text.startswith("[STRAT_") and "]" in text:
            text = text.split("]", 1)[1]
    else:  # natural
        if ":" in text:
            text = text.split(":", 1)[1]
    return text.lstrip() 

# ────────────────────────────────────────────────────────────────────────
# util: 안전 디코딩
# ────────────────────────────────────────────────────────────────────────
def safe_decode(ids, tokenizer, skip_special_tokens=True) -> str:
    """
    tokenizer.decode 대체:
    • OOV id 는 <unk> 로 치환
    • skip_special_tokens 옵션 지원
    """
    unk_tok = tokenizer.unk_token or "<unk>"
    tokens = []
    for tid in ids:
        if skip_special_tokens and tid in tokenizer.all_special_ids:
            continue
        tok = tokenizer._convert_id_to_token(int(tid))
        tokens.append(tok if tok is not None else unk_tok)
    return tokenizer.convert_tokens_to_string(tokens) 