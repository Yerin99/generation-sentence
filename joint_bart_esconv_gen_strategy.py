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
    --strategy_mode natural --ctx_strategy_rep natural --epochs 10 --output_dir outputs/natural_nltk

# 특수 토큰 전략 프리픽스
CUDA_VISIBLE_DEVICES=3 python joint_bart_esconv_gen_strategy.py \
    --eval_init \
    --strategy_mode token --ctx_strategy_rep token --epochs 10 --output_dir outputs/token_nltk

# 자연어 전략 + tiny 1% + patience 3
CUDA_VISIBLE_DEVICES=2 python joint_bart_esconv_gen_strategy.py \
    --strategy_mode natural --tiny_frac 0.05 --epochs 1\
    --eval_steps 20 --patience 3 --ctx_strategy_rep natural\
    --output_dir outputs/tiny_natural_nltk

# 특수 토큰 전략 + tiny 1% + patience 3
CUDA_VISIBLE_DEVICES=3 python joint_bart_esconv_gen_strategy.py \
    --strategy_mode token --tiny_frac 0.05 --epochs 1 \
    --eval_steps 20 --patience 3 --ctx_strategy_rep token\
    --output_dir outputs/tiny_token_nltk
"""
from __future__ import annotations

import argparse, json, logging, random
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import nltk
from datasets import load_dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    EarlyStoppingCallback,
    TrainerCallback,
    GenerationConfig,
    BartConfig,
    set_seed,
)
from utils.metrics import generation_metrics, add_strategy_metrics
from utils.strategy import (
    STRATEGIES, STR2ID, ID2STR, STRAT_TOKENS,
    parse_strategy_from_ids, strip_strategy_prefix,
    to_refined, safe_decode,
)

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
        return f"{sys_token} {ID2STR[strategy_id]}: "
    else:
        return f"{sys_token} {STRAT_TOKENS[strategy_id]}"


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

                context = tokenizer.eos_token.join(ctx_parts)

                # ---------- decoder label ----------
                prefix = build_prefix(sid, self.mode)
                tgt_text = prefix + turn["text"]

                # ---------- tokenization ----------
                enc = self.tok(context,
                               max_length=self.max_src,
                               truncation=True,
                               add_special_tokens=True,
                               padding="max_length")
                dec = self.tok(tgt_text,
                               max_length=self.max_tgt,
                               truncation=True,
                               add_special_tokens=True,
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
def safe_batch_decode(ids, tokenizer, skip_special_tokens=False, **kwargs):
    """배열/텐서(batch) 디코딩 시 pad 토큰만 제거하고, 원할 경우 다른 special token은 유지.

    Args:
        ids (np.ndarray | list[List[int]] | torch.Tensor): 배치 토큰 ID
        tokenizer: Huggingface tokenizer
        skip_special_tokens (bool): True면 special token 모두 제거, False면 pad만 제거
    """
    # numpy 배열로 변환
    ids_array = np.asarray(ids)

    # 1) pad 토큰 제거용 마스킹
    pad_id = tokenizer.pad_token_id
    if skip_special_tokens:
        # Huggingface 옵션 사용 (pad 포함 모든 special 제거)
        return tokenizer.batch_decode(ids_array, skip_special_tokens=True, **kwargs)

    # skip_special_tokens=False 인 경우 → pad 토큰만 제거
    decoded = []
    for seq in ids_array:
        # -100 라벨이나 음수값은 pad로 대체
        seq = [pad_id if t == -100 else int(t) for t in seq]
        # pad 토큰 앞까지만 사용
        if pad_id in seq:
            seq = seq[: seq.index(pad_id)]

        decoded.append(tokenizer.decode(seq, skip_special_tokens=False, **kwargs))
    return decoded


# ===================== 메인 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--strategy_mode", choices=["natural", "token"], default="natural")
    parser.add_argument("--output_dir", type=str, default="outputs/decoder")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="그래디언트 누적 스텝(실효 배치 크기 = batch_size * gradient_accumulation_steps)")
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
    parser.add_argument("--max_src_len", type=int, default=1024,
                        help="인코더 입력 최대 길이 (BART 한계 1024)")
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
    tokenizer.truncation_side = "left"
    
    # *** 특수 토큰 추가 ***
    added = add_esconv_special_tokens(tokenizer)
    logger.info(f"{added} special tokens added (raw+space variants)")
    
    # 모델 로드 시 config 내 generation 파라미터 제거(경고 방지)
    model_cfg = BartConfig.from_pretrained("facebook/bart-base")
    for p in [
        "num_beams", "max_length", "early_stopping", "no_repeat_ngram_size",
        "length_penalty", "forced_bos_token_id", "forced_eos_token_id"
    ]:
        if hasattr(model_cfg, p):
            delattr(model_cfg, p)

    model = BartForConditionalGeneration.from_pretrained("facebook/bart-base", config=model_cfg, ignore_mismatched_sizes=True)
    # 임베딩 크기 확장 (특수 토큰 수용)
    model.resize_token_embeddings(len(tokenizer))

    # GenerationConfig 설정
    generation_config = GenerationConfig(
        max_length=128,
        num_beams=5,
        early_stopping=True,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        repetition_penalty=1.2,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.generation_config = generation_config

    # -------- dataset --------
    train_ds = ESConvGenDataset(
        "train", tokenizer, args.strategy_mode, 
        max_src=args.max_src_len,
        tiny_frac=args.tiny_frac,
        ctx_strategy_rep=args.ctx_strategy_rep
    )
    val_ds = ESConvGenDataset(
        "validation", tokenizer, args.strategy_mode, 
        max_src=args.max_src_len,
        tiny_frac=args.tiny_frac,
        ctx_strategy_rep=args.ctx_strategy_rep
    )
    test_ds = ESConvGenDataset(
        "test", tokenizer, args.strategy_mode, 
        max_src=args.max_src_len,
        tiny_frac=args.tiny_frac,
        ctx_strategy_rep=args.ctx_strategy_rep
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    # -------- training args --------
    # 실제 배치 크기 계산 (로깅용)
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    logger.info(f"배치 크기(per_device): {args.batch_size}, gradient_accumulation_steps: {args.gradient_accumulation_steps}")
    logger.info(f"실효 배치 크기: {effective_batch_size}")

    t_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=args.eval_steps,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",
        seed=args.seed,
        skip_memory_metrics=True,
        dataloader_drop_last=False,
    )

    # -------- training args --------
    trainer = Seq2SeqTrainer(
        model=model,
        args=t_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=build_compute_metrics(val_ds, tokenizer, args),
        callbacks=[
            EarlyStoppingCallback(
                early_stopping_patience=args.patience,
                early_stopping_threshold=0.0
            ),
            TokenEmbeddingCallback(tokenizer)
        ],
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
        
        # 테스트셋 평가
        test_out = trainer.predict(test_ds, metric_key_prefix="init_test")
        
        # PPL 계산
        test_ppl = float(np.exp(test_out.metrics.get('init_test_loss', 0)))
        logger.info(f"Initial Test Perplexity: {test_ppl:.4f}")
        
        # 텍스트 생성 메트릭 계산
        lbl_ids = test_out.label_ids.copy()
        lbl_ids[lbl_ids == -100] = tokenizer.pad_token_id
        
        gen_txt_raw = [strip_strategy_prefix(t, args.strategy_mode)
                      for t in safe_batch_decode(test_out.predictions, tokenizer, skip_special_tokens=True)]
        ref_txt_raw = [strip_strategy_prefix(t, args.strategy_mode)
                      for t in safe_batch_decode(lbl_ids, tokenizer, skip_special_tokens=True)]

        gen_txt = [' '.join(nltk.word_tokenize(t.lower())) for t in gen_txt_raw]
        ref_txt = [' '.join(nltk.word_tokenize(t.lower())) for t in ref_txt_raw]
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
        # 안전한 포맷으로 모델/토크나이저 저장
        model_path = Path(args.output_dir)
        model.save_pretrained(model_path, safe_serialization=True)
        tokenizer.save_pretrained(model_path)
        logger.info(f"Model saved to {args.output_dir}")
        
        # --------------------- test split 평가 ---------------------
        logger.info("evaluating on test split …")

        # 테스트셋 평가
        test_out = trainer.predict(test_ds, metric_key_prefix="test")

        # PPL 계산
        test_ppl = float(np.exp(test_out.metrics.get('test_loss', 0)))
        logger.info(f"Test Perplexity: {test_ppl:.4f}")

        # 2) generation / strategy 메트릭 직접 계산
        lbl_ids = test_out.label_ids.copy()
        lbl_ids[lbl_ids == -100] = tokenizer.pad_token_id

        gen_txt_raw = [strip_strategy_prefix(t, args.strategy_mode)
                      for t in safe_batch_decode(test_out.predictions, tokenizer, skip_special_tokens=True)]
        ref_txt_raw = [strip_strategy_prefix(t, args.strategy_mode)
                      for t in safe_batch_decode(lbl_ids, tokenizer, skip_special_tokens=True)]

        gen_txt = [' '.join(nltk.word_tokenize(t.lower())) for t in gen_txt_raw]
        ref_txt = [' '.join(nltk.word_tokenize(t.lower())) for t in ref_txt_raw]
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
                    ctx_plain,
                    tgt_plain,
                    first_tokens,
                    ex["strategy_id"],
                    "=" * 60
                )
            )

    # ----- 추가: 샘플 생성 테스트 -----
    if args.show_samples and not args.eval_only:
        logger.info("\n===== 생성 샘플 테스트 =====")
        model.eval()
        for i in random.sample(range(len(val_ds)), min(3, len(val_ds))):
            ex = val_ds[i]
            input_ids = torch.tensor([ex["input_ids"]]).to(model.device)
            attention_mask = torch.tensor([ex["attention_mask"]]).to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=128,
                )

            gen_text = safe_decode(outputs[0].tolist(), tokenizer, skip_special_tokens=False)
            tgt_text = safe_decode([t if t != -100 else tokenizer.pad_token_id for t in ex["labels"]], tokenizer, skip_special_tokens=False)
            ctx_text = safe_decode(ex["input_ids"], tokenizer, skip_special_tokens=False)

            logger.info(
                f"\n----- 생성 샘플 {i} -----\n"
                f"컨텍스트:\n{ctx_text}\n\n"
                f"타겟:\n{tgt_text}\n\n"
                f"생성:\n{gen_text}\n"
                + "-"*60
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
    for token in STRAT_TOKENS[:]:
        token_id = tokenizer.convert_tokens_to_ids(token)
        is_special = token_id in tokenizer.all_special_ids
        logger.info(f"Token: {token} -> ID: {token_id} (special={is_special})")
    
    return num_added


def build_compute_metrics(eval_dataset, tokenizer, args):
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels[labels == -100] = tokenizer.pad_token_id

        metrics = {}        
        # ── 텍스트 메트릭 ─────────────────────
        gen_txt_raw = [strip_strategy_prefix(t, args.strategy_mode)
                      for t in safe_batch_decode(preds, tokenizer, skip_special_tokens=True)]
        ref_txt_raw = [strip_strategy_prefix(t, args.strategy_mode)
                      for t in safe_batch_decode(labels, tokenizer, skip_special_tokens=True)]

        gen_txt = [' '.join(nltk.word_tokenize(t.lower())) for t in gen_txt_raw]
        ref_txt = [' '.join(nltk.word_tokenize(t.lower())) for t in ref_txt_raw]
        text_metrics = generation_metrics(gen_txt, ref_txt)
        for k, v in text_metrics.items():
            metrics[f"eval_{k}"] = v

        # ── 전략 메트릭 ───────────────────────
        sid_pred, sid_gt = [], []
        for p_ids, ex in zip(preds, eval_dataset.examples):
            sid = parse_strategy_from_ids(p_ids, tokenizer, args.strategy_mode)
            if sid is None:
                sid = STR2ID["Others"]
            sid_pred.append(sid)
            sid_gt.append(ex["strategy_id"])
        strat_metrics = add_strategy_metrics({}, sid_pred, sid_gt)
        for k, v in strat_metrics.items():
            metrics[f'eval_{k}'] = v

        from sklearn.metrics import classification_report
        report = classification_report(sid_gt, sid_pred, labels=list(range(len(STRATEGIES))), target_names=STRATEGIES, digits=2, zero_division=0)
        logging.info("\n" + report)
 
        return metrics
    return compute_metrics


# ---- 체크포인트 로드 시 임베딩 크기 보정 콜백 ----
class TokenEmbeddingCallback(TrainerCallback):
    """체크포인트 로드 등 이벤트에서 토큰 임베딩 크기 불일치 해결"""
    def __init__(self, tokenizer):
        self.tok = tokenizer

    def _ensure_size(self, model):
        if model.get_input_embeddings().weight.shape[0] != len(self.tok):
            logger.warning(f"임베딩 크기 불일치 -> resize {model.get_input_embeddings().weight.shape[0]} -> {len(self.tok)}")
            model.resize_token_embeddings(len(self.tok))

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._ensure_size(model)

    def on_load_checkpoint(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._ensure_size(model)

    def on_checkpoint_model_loading(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._ensure_size(model)


if __name__ == "__main__":
    main() 