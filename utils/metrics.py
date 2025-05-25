"""
utils.metrics
─────────────
공통 generation / strategy 메트릭 모듈.
joint_bart_esconv.py 의 로직을 그대로 옮겨왔다.
"""
from __future__ import annotations
from typing import List, Dict
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

# scorers는 모듈 레벨에서 한 번만 초기화 → 반복 호출 시 속도↑
_bleu, _rouge, _meteor, _cider = Bleu(4), Rouge(), Meteor(), Cider()

def generation_metrics(preds: List[str], refs: List[str]) -> Dict[str, float]:
    """BLEU1-4 / ROUGE-L / METEOR / CIDEr 계산"""
    gts = {i: [r] for i, r in enumerate(refs)}
    res = {i: [p] for i, p in enumerate(preds)}
    bleu, _ = _bleu.compute_score(gts, res)
    rouge, _ = _rouge.compute_score(gts, res)
    meteor, _ = _meteor.compute_score(gts, res)
    cider, _ = _cider.compute_score(gts, res)
    return {
        "bleu1": bleu[0], "bleu2": bleu[1], "bleu3": bleu[2], "bleu4": bleu[3],
        "rouge_l": rouge, "meteor": meteor, "cider": cider,
    }

def add_strategy_metrics(gen_m: Dict[str, float],
                         sid_pred: List[int], sid_gt: List[int]) -> Dict[str, float]:
    """
    strategy accuracy / weighted-F1 합산.
    sid_pred, sid_gt 길이는 반드시 같다고 가정한다.
    """
    acc = accuracy_score(sid_gt, sid_pred)
    f1  = f1_score(sid_gt, sid_pred, average="weighted")
    gen_m.update({"strategy_accuracy": acc, "strategy_f1": f1})
    return gen_m
