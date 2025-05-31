# -*- coding: utf-8 -*-
"""
joint_bart_esconv_gen_strategy.py
============================
디코더 첫 토큰으로 '전략'을 **생성**(natural phrase 또는 특수 토큰) 하는 ESConv 파이프라인.

사용법 예시
----------
# 자연어 전략 프리픽스
CUDA_VISIBLE_DEVICES=2 python joint_bart_esconv_gen_strategy.py \
    --eval_init \
    --strategy_mode natural --ctx_strategy_rep natural --epochs 10 --output_dir outputs/natural

# 특수 토큰 전략 프리픽스
CUDA_VISIBLE_DEVICES=3 python joint_bart_esconv_gen_strategy.py \
    --eval_init \
    --strategy_mode token --ctx_strategy_rep token --epochs 10 --output_dir outputs/token

# 자연어 전략 + tiny 1% + patience 3
CUDA_VISIBLE_DEVICES=2 python joint_bart_esconv_gen_strategy.py \
    --strategy_mode natural --tiny_frac 0.01 --epochs 1\
    --eval_steps 10 --patience 3 --ctx_strategy_rep natural\
    --output_dir outputs/tiny_natural

# 특수 토큰 전략 + tiny 1% + patience 3
CUDA_VISIBLE_DEVICES=3 python joint_bart_esconv_gen_strategy.py \
    --strategy_mode token --tiny_frac 0.01 --epochs 1 \
    --eval_steps 10 --patience 3 --ctx_strategy_rep token\
    --output_dir outputs/tiny_token
"""
from __future__ import annotations

import argparse, json, logging, random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from datasets import load_dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    set_seed,
)
from utils.metrics import generation_metrics, add_strategy_metrics
from utils.strategy import (
    STRATEGIES, STR2ID, ID2STR, STRAT_TOKENS,
    parse_strategy_from_ids, strip_strategy_prefix,
    to_refined, safe_decode,
)
from tokenizers import AddedToken   # NEW (slow-tokenizer 사용 시 무시돼도 안전)

# ===================== 전역 정의 =====================
SPECIAL_TOKENS = {          # 대화 문맥용 역할 토큰
    "usr": "[USR]",
    "sys": "[SYS]",
    "strategy": "[STRATEGY]",  # 전략 앞에 올 토큰 (추가됨)
}

logger = logging.getLogger("joint_dec")


# ===================== 전략 프리픽스 빌더 =====================
def build_prefix(strategy_id: int, mode: str) -> str:
    """
    전략 id → 프리픽스 문자열 반환.
      - mode == "natural" : "Providing Suggestions: [SYS]"
      - mode == "token"   : "[STRAT_Providing_Suggestions] [SYS]"
    """
    sys_token = SPECIAL_TOKENS["sys"]
    if mode == "natural":
        return f"{ID2STR[strategy_id]}: {sys_token} "
    else:
        return f"{STRAT_TOKENS[strategy_id]} {sys_token} "


# ===================== 데이터셋 =====================
class ESConvGenDataset(torch.utils.data.Dataset):
    """
    전략 프리픽스를 타겟 시퀀스에 포함해 반환하는 생성용 데이터셋.

    SRP: 데이터 전처리 전담.
    """

    def __init__(
        self,
        split: str,
        tokenizer: BartTokenizer,
        strategy_mode: str,
        max_src: int = 512,
        max_tgt: int = 128,
        tiny_frac: float | None = None,
        cache_dir: str = "cache_gen",
        ctx_strategy_rep: str = "token",
    ):
        self.tok = tokenizer
        self.mode = strategy_mode
        self.max_src, self.max_tgt = max_src, max_tgt
        self.ctx_rep = ctx_strategy_rep

        raw = load_dataset("thu-coai/esconv", split=split)

        if tiny_frac:
            raw = raw.shuffle(seed=42).select(range(int(len(raw) * tiny_frac)))

        cache_f = (
            Path(cache_dir)
            / f"{split}_{self.mode}_{max_src}_{max_tgt}_{tiny_frac}_refined.pt"
        )
        cache_f.parent.mkdir(exist_ok=True, parents=True)
        if cache_f.exists():
            self.examples = torch.load(cache_f, map_location="cpu", weights_only=True)
            return

        self.examples: List[Dict] = []
        for ex in raw:
            dialog = json.loads(ex["text"])["dialog"]
            for turn in dialog:
                if turn["speaker"] != "sys":
                    continue
                cur_orig = turn.get("strategy", "Others")
                cur_ref  = to_refined(cur_orig)
                sid      = STR2ID.get(cur_ref, STR2ID["Others"])

                # ---------- encoder input ----------
                ctx_parts: List[str] = []
                for prev in dialog:
                    if prev is turn:
                        break
                    spk_tok = SPECIAL_TOKENS["usr"] if prev["speaker"] == "usr" else SPECIAL_TOKENS["sys"]

                    # --- 과거 system 턴 전략 표시 ---
                    if prev["speaker"] == "sys" and self.ctx_rep != "none":
                        prev_orig = prev.get("strategy", "Others")
                        prev_ref = to_refined(prev_orig)

                        if self.ctx_rep == "token":
                            st_tok = f"[STRAT_{prev_ref.replace(' ', '_')}]"
                            ctx_parts.append(f"{spk_tok} {SPECIAL_TOKENS['strategy']} {st_tok} {prev['text']}")
                        else:  # natural
                            st_nat = f"{prev_ref}:"
                            ctx_parts.append(f"{spk_tok} {st_nat} {prev['text']}")
                    else:
                        ctx_parts.append(f"{spk_tok} {prev['text']}")

                context = " ".join(ctx_parts)

                # ---------- decoder label ----------
                prefix = build_prefix(sid, self.mode)
                tgt_text = prefix + turn["text"]

                # ---------- tokenization ----------
                enc = self.tok(context,
                               max_length=self.max_src,
                               truncation=True,
                               padding="max_length")
                dec = self.tok(tgt_text,
                               max_length=self.max_tgt,
                               truncation=True,
                               padding="max_length")

                # label -100 masking
                labels = [(t if t != self.tok.pad_token_id else -100) for t in dec.input_ids]

                self.examples.append({
                    "input_ids": enc.input_ids,
                    "attention_mask": enc.attention_mask,
                    "labels": labels,
                    "strategy_id": sid,
                })
        torch.save(self.examples, cache_f)

    # -------------- torch Dataset interface --------------
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # 이미 토큰화된 예제를 그대로 반환
        return self.examples[idx]


# ===================== 유틸 함수 =====================
def safe_batch_decode(ids, tokenizer, **kwargs):
    """범위를 벗어난 ID를 UNK로 치환하여 안전하게 디코딩"""
    # numpy 배열로 변환
    ids_array = np.asarray(ids)
    
    # 범위를 벗어난 ID 마스킹 (음수 또는 어휘 크기 이상)
    invalid_mask = (ids_array < 0) | (ids_array >= len(tokenizer))
    
    # 마스킹된 위치가 있으면 복사 후 UNK로 대체
    if invalid_mask.any():
        ids_array = ids_array.copy()
        ids_array[invalid_mask] = tokenizer.unk_token_id
        
    return tokenizer.batch_decode(ids_array, **kwargs)


# ===================== 메인 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy_mode", choices=["natural", "token"], default="natural")
    parser.add_argument("--output_dir", type=str, default="outputs/decoder")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiny_frac", type=float, default=None, help="0~1: 디버그용 샘플 비율")
    parser.add_argument("--patience", type=int, default=5,
                        help="early stopping patience (eval_loss 기준)")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="evaluation interval (steps)")
    parser.add_argument("--show_samples", type=int, default=5,
                        help="샘플 n개(context/target) 출력 후 종료")
    parser.add_argument("--ctx_strategy_rep", choices=["token", "natural", "none"], 
                        default="token",
                        help="과거 system 턴에 전략을 어떻게 표기할지")
    parser.add_argument("--eval_init", action="store_true", 
                        help="학습 전 초기 모델(epoch 0)에서 평가 수행")
    parser.add_argument("--eval_only", action="store_true",
                        help="학습 없이 평가만 수행")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[logging.StreamHandler()],
    )
    set_seed(args.seed)

    # 토크나이저 및 모델 초기화
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    # *** 특수 토큰 추가 ***
    added = add_esconv_special_tokens(tokenizer)
    logger.info(f"{added} special tokens added (raw+space variants)")
    
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
    # 임베딩 크기 확장 (특수 토큰 수용)
    model.resize_token_embeddings(len(tokenizer))

    # -------- dataset --------
    train_ds = ESConvGenDataset(
        "train", tokenizer, args.strategy_mode, 
        tiny_frac=args.tiny_frac,
        ctx_strategy_rep=args.ctx_strategy_rep
    )
    val_ds = ESConvGenDataset(
        "validation", tokenizer, args.strategy_mode, 
        tiny_frac=args.tiny_frac,
        ctx_strategy_rep=args.ctx_strategy_rep
    )
    test_ds = ESConvGenDataset(
        "test", tokenizer, args.strategy_mode, 
        tiny_frac=args.tiny_frac,
        ctx_strategy_rep=args.ctx_strategy_rep
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    # -------- training args --------
    t_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        logging_strategy="steps",
        logging_steps=args.eval_steps,
        predict_with_generate=True,
        generation_max_length=128,
        generation_num_beams=5,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
    )

    # -------- metric fn --------
    def calculate_perplexity(loss):
        """
        Cross-entropy loss에서 perplexity 계산
        PPL = exp(loss)
        """
        return np.exp(loss)

    def build_compute_metrics(eval_dataset):
        def compute_metrics(eval_pred):
            preds, labels = eval_pred
            labels[labels == -100] = tokenizer.pad_token_id

            # 기본 메트릭 초기화
            metrics = {}
            
            # PPL 계산 추가
            if hasattr(eval_pred, 'metrics') and 'eval_loss' in eval_pred.metrics:
                metrics['perplexity'] = calculate_perplexity(eval_pred.metrics['eval_loss'])
            
            # 1) 텍스트 메트릭
            gen_txt = [strip_strategy_prefix(t, args.strategy_mode)
                       for t in safe_batch_decode(preds, tokenizer, skip_special_tokens=True)]
            ref_txt = [strip_strategy_prefix(t, args.strategy_mode)
                       for t in safe_batch_decode(labels, tokenizer, skip_special_tokens=True)]
            gen_m = generation_metrics(gen_txt, ref_txt)
            metrics.update(gen_m)  # 이렇게 변경

            # 2) 전략 id 파싱 (실패 → Others 로 치환)
            sid_pred, sid_gt = [], []
            for p_ids, ex in zip(preds, eval_dataset.examples):
                sid = parse_strategy_from_ids(p_ids, tokenizer, args.strategy_mode)
                if sid is None:
                    sid = STR2ID["Others"]
                sid_pred.append(sid)
                sid_gt.append(ex["strategy_id"])
            gen_m = add_strategy_metrics(metrics, sid_pred, sid_gt)  # 여기서 metrics로 변경

            # 3) classification_report 로그 (labels 명시, zero_division 방지)
            from sklearn.metrics import classification_report
            report = classification_report(
                sid_gt,
                sid_pred,
                labels=list(range(len(STRATEGIES))),   # ← 전체 클래스 인덱스
                target_names=STRATEGIES,
                digits=2,
                zero_division=0
            )
            logging.info("\n" + report)
            
            # PPL을 포함한 주요 메트릭 로깅
            if 'perplexity' in metrics:
                logging.info(f"📊 Eval Metrics: PPL={metrics['perplexity']:.4f}, BLEU-1={metrics['bleu1']:.4f}, "
                            f"Strategy Accuracy={metrics['strategy_accuracy']:.4f}")
            
            return metrics
        return compute_metrics

    # -------- trainer --------
    trainer = Seq2SeqTrainer(
        model=model,
        args=t_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        tokenizer=tokenizer,
        compute_metrics=build_compute_metrics(val_ds),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience)],
    )

    # --------------------- 초기 모델(epoch 0) 평가 ---------------------
    if args.eval_init or args.eval_only:
        logger.info("📊 Evaluating initial model (epoch 0) ...")
        
        # 검증 데이터셋 평가
        init_eval = trainer.evaluate()
        
        # 결과 저장
        init_path = Path(args.output_dir) / "init_eval_metrics.json"
        init_path.parent.mkdir(exist_ok=True, parents=True)
        with init_path.open("w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in init_eval.items()}, f, indent=2)
        
        # 테스트 데이터셋 평가
        logger.info("📊 Evaluating initial model on test split ...")
        
        # 생성 파라미터
        gen_kwargs = {
            "num_beams": 5,
            "early_stopping": True,
            "no_repeat_ngram_size": 3,
            "length_penalty": 1.0,
            "repetition_penalty": 1.2,
        }
        
        # 테스트셋 평가
        test_out = trainer.predict(test_ds, metric_key_prefix="init_test", **gen_kwargs)
        
        # PPL 계산
        test_loss = test_out.metrics.get('init_test_loss', 0)
        test_ppl = calculate_perplexity(test_loss)
        logger.info(f"Initial Test Perplexity: {test_ppl:.4f}")
        
        # 텍스트 생성 메트릭 계산
        lbl_ids = test_out.label_ids.copy()
        lbl_ids[lbl_ids == -100] = tokenizer.pad_token_id
        
        gen_txt = [strip_strategy_prefix(t, args.strategy_mode)
                for t in safe_batch_decode(test_out.predictions, tokenizer, skip_special_tokens=True)]
        ref_txt = [strip_strategy_prefix(t, args.strategy_mode)
                for t in safe_batch_decode(lbl_ids, tokenizer, skip_special_tokens=True)]
        
        gen_m = generation_metrics(gen_txt, ref_txt)
        
        # 전략 메트릭 계산
        sid_pred, sid_gt = [], []
        for g_ids, ex in zip(test_out.predictions, test_ds.examples):
            sid = parse_strategy_from_ids(g_ids, tokenizer, args.strategy_mode)
            if sid is None:
                sid = STR2ID["Others"]
            sid_pred.append(sid)
            sid_gt.append(ex["strategy_id"])
        
        gen_m = add_strategy_metrics(gen_m, sid_pred, sid_gt)
        
        # 메트릭 저장
        init_test_m = {f"init_test_{k}": v for k, v in gen_m.items()}
        init_test_m.update(test_out.metrics)
        init_test_m['init_test_perplexity'] = test_ppl
        
        # 분류 리포트
        from sklearn.metrics import classification_report
        init_report = classification_report(
            sid_gt, sid_pred,
            labels=list(range(len(STRATEGIES))),
            target_names=STRATEGIES,
            digits=2,
            zero_division=0
        )
        logger.info("\n=== Initial Test Strategy Classification Report ===\n" + init_report)
        
        # 저장
        init_test_path = Path(args.output_dir) / "init_test_metrics.json"
        with init_test_path.open("w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in init_test_m.items()}, f, indent=2)
        
        logger.info(f"📝 Saved initial model metrics to {args.output_dir}")
        
        # 샘플 저장
        sample_n = min(10, len(gen_txt))
        with open(Path(args.output_dir) / "init_samples.txt", "w", encoding="utf-8") as f:
            for ref, gen in zip(ref_txt[:sample_n], gen_txt[:sample_n]):
                f.write(f"REF: {ref}\nGEN: {gen}\n---\n")
    
    # 학습 수행 (eval_only가 True면 학습 건너뜀)
    if not args.eval_only:
        trainer.train()
        trainer.save_model(args.output_dir)
        
        # --------------------- test split 평가 ---------------------
        logger.info("evaluating on test split …")

        # 생성 파라미터 설정
        gen_kwargs = {
            "num_beams": 5,
            "early_stopping": True,  # EOS 토큰 생성 시 중단
            "no_repeat_ngram_size": 3,
            "length_penalty": 1.0,  
            "repetition_penalty": 1.2,
        }

        # 1) loss/runtime 만 위해 metrics 잠시 비활성화
        trainer.compute_metrics = None
        test_out = trainer.predict(test_ds, metric_key_prefix="test", **gen_kwargs)

        # PPL 계산 추가
        test_loss = test_out.metrics.get('test_loss', 0)
        test_ppl = calculate_perplexity(test_loss)
        logger.info(f"Test Perplexity: {test_ppl:.4f}")

        # 2) generation / strategy 메트릭 직접 계산
        lbl_ids = test_out.label_ids.copy()
        lbl_ids[lbl_ids == -100] = tokenizer.pad_token_id

        gen_txt = [strip_strategy_prefix(t, args.strategy_mode)
                   for t in safe_batch_decode(test_out.predictions, tokenizer, skip_special_tokens=True)]
        ref_txt = [strip_strategy_prefix(t, args.strategy_mode)
                   for t in safe_batch_decode(lbl_ids, tokenizer, skip_special_tokens=True)]

        gen_m = generation_metrics(gen_txt, ref_txt)

        sid_pred, sid_gt = [], []
        for g_ids, ex in zip(test_out.predictions, test_ds.examples):
            sid = parse_strategy_from_ids(g_ids, tokenizer, args.strategy_mode)
            if sid is None:                     # 미검출 → Others
                sid = STR2ID["Others"]
            sid_pred.append(sid)
            sid_gt.append(ex["strategy_id"])

        gen_m = add_strategy_metrics(gen_m, sid_pred, sid_gt)

        # 3) key 에 test_ 접두사 부여 → 중복 제거
        test_m = {f"test_{k}": v for k, v in gen_m.items()}
        test_m.update(test_out.metrics)         # test_loss, test_runtime 등만 추가
        test_m['test_perplexity'] = test_ppl    # PPL 추가

        # 4) 분류 리포트 로그
        from sklearn.metrics import classification_report
        rep = classification_report(
            sid_gt, sid_pred,
            labels=list(range(len(STRATEGIES))),
            target_names=STRATEGIES,
            digits=2,
            zero_division=0
        )
        logger.info("\n" + rep)

        # 5) 저장
        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        with open(Path(args.output_dir) / "test_metrics.json", "w") as f:
            json.dump({k: float(v) for k, v in test_m.items()}, f, indent=2)

        # 샘플 10개 저장
        sample_n = min(10, len(gen_txt))
        with open(Path(args.output_dir) / "samples.txt", "w", encoding="utf-8") as f:
            for ref, gen in zip(ref_txt[:sample_n], gen_txt[:sample_n]):
                f.write(f"REF: {ref}\nGEN: {gen}\n---\n")

        logger.info(f"saved model & test metrics to {args.output_dir}")

    if args.show_samples:
        import textwrap
        for i in random.sample(range(len(train_ds)), args.show_samples):
            ex = train_ds[i]
            ctx_plain = safe_decode(ex["input_ids"], tokenizer, skip_special_tokens=False)
            tgt_plain = safe_decode([t if t != -100 else tokenizer.pad_token_id for t in ex["labels"]], 
                                          tokenizer, skip_special_tokens=False)
            
            # 전략 토큰이 <unk>로 변환되었는지 확인하기 위한 디버깅 출력
            first_tokens = tokenizer.convert_ids_to_tokens(ex["labels"][:10])
            
            logging.info(
                "\n===== SAMPLE {:d} =====\n"
                "CONTEXT:\n{}\n\nTARGET:\n{}\n"
                "First tokens: {}\n"
                "strategy_id: {}\n{}".format(
                    i,
                    textwrap.fill(ctx_plain, 120),
                    tgt_plain,
                    first_tokens,
                    ex["strategy_id"],
                    "=" * 60
                )
            )
        return


def add_esconv_special_tokens(tokenizer):
    """
    ESConv 특수 토큰을 토크나이저에 추가하고, 모든 토큰이 단일 토큰으로 처리되게 함
    """
    # 1. 모든 특수 토큰 목록 생성 
    special_tokens = list(SPECIAL_TOKENS.values()) + STRAT_TOKENS
    
    # 2. 특수 토큰 추가 (tokenizer.add_special_tokens가 가장 신뢰할 수 있는 방법)
    special_tokens_dict = {"additional_special_tokens": special_tokens}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    
    # 3. 토큰이 제대로 추가됐는지 검증
    for token in STRAT_TOKENS[:2]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        is_special = token_id in tokenizer.all_special_ids
        logger.info(f"Token: {token} -> ID: {token_id} (special={is_special})")
    
    return num_added


if __name__ == "__main__":
    main() 