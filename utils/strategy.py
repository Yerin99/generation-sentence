# -*- coding: utf-8 -*-
"""
utils.strategy
──────────────
• 디코더가 생성한 시퀀스에서 전략 id를 추출
• 전략 프리픽스를 제거해 순수 응답만 남기기
• Original → Refined 전략 이름 매핑 지원
• Refined 전략명을 사용하도록 수정
"""
from __future__ import annotations
from typing import List, Optional, Dict

# ──────────────────────────────────────────────────────────────
# 0) Original ↔ Refined 매핑 정의
# ──────────────────────────────────────────────────────────────
ORIG2REF: Dict[str, str] = {
    "Question": "Question",
    "Restatement or Paraphrasing": "Paraphrasing",
    "Reflection of feelings": "Reflection",
    "Self-disclosure": "Self-disclosure",
    "Affirmation and Reassurance": "Reassurance",
    "Providing Suggestions": "Suggestion",
    "Information": "Information",
    "Others": "Others",
}

# Refined 전략 리스트(모델이 생성할 이름)
STRATEGIES: List[str] = list({v for v in ORIG2REF.values()})  # 중복 제거 후 리스트화
STRATEGIES.sort(key=lambda x: list(ORIG2REF.values()).index(x))  # 원본 순서 유지

# id ↔ name 매핑
STR2ID = {s: i for i, s in enumerate(STRATEGIES)}
ID2STR = {i: s for s, i in STR2ID.items()}

# Original 이름도 같은 id 로 매핑(데이터셋 라벨 호환성)
for orig, ref in ORIG2REF.items():
    STR2ID[orig] = STR2ID[ref]

# token mode 특수 토큰(Refined 이름 사용)
STRAT_TOKENS = [f"[STRAT_{s.replace(' ', '_')}]" for s in STRATEGIES]

# ──────────────────────────────────────────────────────────────
# util: Original → Refined 이름 변환
# ──────────────────────────────────────────────────────────────
def to_refined(name: str) -> str:
    """주어진 전략명을 refined 형태로 변환(없는 경우 그대로 반환)."""
    return ORIG2REF.get(name, name)

# ──────────────────────────────────────────────────────────────
# 1) 전략 id 추출
# ──────────────────────────────────────────────────────────────
def get_token_id_map(tokenizer):
    """특수토큰 id → 전략 id 매핑"""
    ids = tokenizer.convert_tokens_to_ids(STRAT_TOKENS)
    return {ids[i]: i for i in range(len(STRATEGIES))}

def parse_strategy_from_ids(pred_ids: List[int],
                            tokenizer,
                            mode: str) -> Optional[int]:
    """
    token mode : BOS 이후 첫 번째 등장 전략 특수토큰을 탐색
    natural    : ':' 이전 문자열을 lower-case 비교
    """
    if mode == "token":
        tid2sid = get_token_id_map(tokenizer)
        for tid in pred_ids[1:]:  # BOS 이후 첫 번째 토큰부터 탐색
            if tid in tid2sid:
                return tid2sid[tid]
        return None

    # ----- natural mode -----
    from .strategy import safe_decode  # 순환 import 방지
    txt = safe_decode(pred_ids[1:30], tokenizer, skip_special_tokens=True)
    if ":" not in txt:
        return None
    prefix = txt.split(":", 1)[0].strip().lower()

    # refined 이름 우선 일치 확인
    for s in STRATEGIES:
        sl = s.lower()
        if sl == prefix or sl in prefix or prefix in sl:
            return STR2ID[s]

    # original 이름과도 매칭 시도
    for orig, ref in ORIG2REF.items():
        ol = orig.lower()
        if ol == prefix or ol in prefix or prefix in ol:
            return STR2ID[ref]

    return None

# ──────────────────────────────────────────────────────────────
# 2) 전략 프리픽스 제거 → 순수 응답 텍스트
# ──────────────────────────────────────────────────────────────
def strip_strategy_prefix(text: str, mode: str) -> str:
    if mode == "token":
        if text.startswith("[STRAT_") and "]" in text:
            text = text.split("]", 1)[1]
    else:  # natural
        if ":" in text:
            text = text.split(":", 1)[1]
    return text.lstrip()

# ──────────────────────────────────────────────────────────────
# util: 안전 디코딩
# ──────────────────────────────────────────────────────────────
def safe_decode(ids, tokenizer, skip_special_tokens=False) -> str:
    """
    안전한 토큰 디코딩 함수:
    1. 유효하지 않은 ID는 unk로 처리
    2. 특수 토큰은 온전히 보존 (skip_special_tokens=True일 때만 제외)
    """
    # 배열/텐서를 파이썬 리스트로 변환
    if hasattr(ids, 'tolist'):
        ids = ids.tolist()
    
    # 유효한 ID만 필터링
    valid_ids = []
    for i in ids:
        if i == -100:  # 손실 계산 무시 토큰
            i = tokenizer.pad_token_id
        i = int(i)
        if 0 <= i < len(tokenizer):
            if not (skip_special_tokens and i in tokenizer.all_special_ids):
                valid_ids.append(i)
    
    # 원본 decode 사용
    text = tokenizer.decode(valid_ids, skip_special_tokens=False)
    
    # [참고] 만약 특수 토큰이 여전히 안 보인다면 아래 디버깅 코드 추가
    # raw_tokens = tokenizer.convert_ids_to_tokens(valid_ids)
    # print(f"RAW TOKENS: {raw_tokens}")
    
    return text 