# -*- coding: utf-8 -*-
"""
bart_dialog_generator.py
============================
BART 모델을 사용하여 대화 맥락에서 자연스러운 응답을 생성하는 파이프라인.

사용법 예시:
----------
# 기본 훈련
CUDA_VISIBLE_DEVICES=1 python bart_dialog_generator.py --output_dir outputs/dialog_generation_edit

# 작은 비율의 데이터로 빠른 테스트
CUDA_VISIBLE_DEVICES=0 python bart_dialog_generator.py --tiny_frac 0.01 --epochs 1 --output_dir outputs/dialog_tiny_test

# facebook/bart-base 원본 모델 평가
CUDA_VISIBLE_DEVICES=3 python bart_dialog_generator.py --eval_only --output_dir outputs/dialog_eval
"""

from __future__ import annotations

import argparse, json, logging, random, os
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
    TrainerCallback,
    GenerationConfig,
    set_seed,
    BartConfig,
)
from utils.metrics import generation_metrics
from tokenizers import AddedToken

# ===================== 전역 정의 =====================
SPECIAL_TOKENS = {          # 대화 문맥용 역할 토큰
    "usr": "[USR]",
    "sys": "[SYS]",
}

logger = logging.getLogger("dialog_gen")

# ===================== 데이터셋 =====================
class DialogGenDataset(torch.utils.data.Dataset):
    """
    대화 생성을 위한 데이터셋.
    
    대화 맥락(context)를 입력으로, 시스템 응답을 출력으로 하는 데이터셋
    """

    def __init__(
        self,
        split: str,
        tokenizer: BartTokenizer,
        max_src: int = 512,
        max_tgt: int = 128,
        tiny_frac: float | None = None,
        cache_dir: str = "cache_dialog_gen",
        dataset_name: str = "thu-coai/esconv",
        use_cache: bool = True,  # 캐시 사용 여부 선택 옵션
    ):
        self.tok = tokenizer
        self.max_src, self.max_tgt = max_src, max_tgt
        self.dataset_name = dataset_name

        # 데이터셋 로드
        raw = load_dataset(dataset_name, split=split)

        # 작은 비율만 사용 (디버깅용)
        if tiny_frac:
            raw = raw.shuffle(seed=42).select(range(int(len(raw) * tiny_frac)))

        cache_f = (
            Path(cache_dir)
            / f"{split}_{max_src}_{max_tgt}_{tiny_frac}_{dataset_name.replace('/', '_')}.pt"
        )
        # 캐시 사용이 설정된 경우에만 캐시 파일 확인 및 로드
        if use_cache and cache_f.exists():
            # 보안 경고 해결: torch.save/load 시 객체 직렬화 대신 pickle 사용
            import pickle
            with open(cache_f, 'rb') as f:
                self.examples = pickle.load(f)
            return

        self.examples: List[Dict] = []
        for ex in raw:
            dialog = json.loads(ex["text"])["dialog"]
            for turn_idx, turn in enumerate(dialog):
                if turn["speaker"] != "sys":
                    continue
                
                # 1) context를 </s>로 구분
                ctx_parts = []
                for prev in dialog[:turn_idx]:
                    spk_tok = SPECIAL_TOKENS["usr"] if prev["speaker"] == "usr" else SPECIAL_TOKENS["sys"]
                    ctx_parts.append(f"{spk_tok}{prev['text']}")

                # 특수 토큰을 문자열이 아니라 토크나이저의 bos/eos 토큰으로 사용
                context = tokenizer.bos_token + (tokenizer.eos_token.join(ctx_parts) if ctx_parts else "") + tokenizer.eos_token

                if not context.strip():
                    continue

                # ---------- decoder output (response) ----------
                tgt_text = SPECIAL_TOKENS["sys"] + turn["text"] + tokenizer.eos_token

                # 2) add_special_tokens=False로 토크나이즈
                enc = self.tok(
                    context,
                    max_length=self.max_src,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=False
                )
                dec = self.tok(
                    tgt_text,
                    max_length=self.max_tgt,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=False
                )

                # label -100 masking (패딩 토큰을 무시하기 위함)
                labels = [(t if t != self.tok.pad_token_id else -100) for t in dec.input_ids]

                self.examples.append({
                    "input_ids": enc.input_ids,
                    "attention_mask": enc.attention_mask,
                    "labels": labels,
                    "context": context,
                    "response": tgt_text,
                })
        # 캐시 사용이 설정된 경우에만 캐시 파일 저장
        if use_cache:
            # 캐시 디렉토리 생성
            cache_f.parent.mkdir(exist_ok=True, parents=True)
            # 보안 경고 해결: torch.save/load 시 객체 직렬화 대신 pickle 사용
            import pickle
            with open(cache_f, 'wb') as f:
                pickle.dump(self.examples, f)

    # -------------- torch Dataset interface --------------
    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        # 이미 토큰화된 예제를 그대로 반환
        return self.examples[idx]


# ===================== 유틸 함수 =====================
def safe_decode(ids, tokenizer, skip_special_tokens=False, **kwargs):
    """안전하게 디코딩하되, pad 토큰만 제외하고 다른 special token은 유지"""
    try:
        # 먼저 pad 토큰의 위치 찾기
        if isinstance(ids, list):
            # pad 토큰이 시작되는 첫 위치 찾기
            pad_positions = [i for i, id in enumerate(ids) if id == tokenizer.pad_token_id]
            # pad 토큰이 있으면 첫 pad 토큰 전까지만 사용
            ids_without_pad = ids[:pad_positions[0]] if pad_positions else ids
            # 디코딩 시 특수 토큰 유지
            return tokenizer.decode(ids_without_pad, skip_special_tokens=skip_special_tokens, **kwargs)
        else:
            return tokenizer.decode(ids, skip_special_tokens=skip_special_tokens, **kwargs)
    except Exception as e:
        logger.warning(f"디코딩 실패: {e}")
        # 음수 및 범위 초과 ID 제거
        valid_ids = [i for i in ids if i >= 0 and i < len(tokenizer)]
        return tokenizer.decode(valid_ids, skip_special_tokens=skip_special_tokens, **kwargs)


# ===================== 메인 =====================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/dialog_gen")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--tiny_frac", type=float, default=None, help="0~1: 디버그용 샘플 비율")
    parser.add_argument("--patience", type=int, default=5,
                        help="early stopping patience (eval_loss 기준)")
    parser.add_argument("--eval_steps", type=int, default=500,
                        help="evaluation interval (steps)")
    parser.add_argument("--show_samples", type=int, default=10,
                        help="샘플 n개(context/target) 출력 후 종료")
    parser.add_argument("--eval_init", action="store_true", 
                        help="학습 전 초기 모델(epoch 0)에서 평가 수행")
    parser.add_argument("--eval_only", action="store_true",
                        help="학습 없이 평가만 수행")
    parser.add_argument("--dataset", type=str, default="thu-coai/esconv",
                        help="사용할 데이터셋 (기본값: ESConv)")
    parser.add_argument("--learning_rate", type=float, default=3e-5,
                        help="학습률 (기본값: 3e-5, 큰 데이터셋은 5e-5도 가능)")
    parser.add_argument("--max_src_length", type=int, default=896,
                        help="최대 소스 길이 (기본값: 896)")
    parser.add_argument("--max_tgt_length", type=int, default=256,
                        help="최대 타겟 길이")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="warmup 비율 (전체 스텝의 %, 초기 불안정한 loss spike 방지)")
    parser.add_argument("--num_samples", type=int, default=5, 
                        help="학습/평가 중 출력할 샘플 수")
    parser.add_argument("--no_cache", action="store_true",
                        help="캐시를 사용하지 않고 항상 데이터를 새로 처리")
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
    
    # 특수 토큰 추가
    special_tokens_dict = {"additional_special_tokens": list(SPECIAL_TOKENS.values())}
    num_added = tokenizer.add_special_tokens(special_tokens_dict)
    logger.info(f"{num_added} 특수 토큰 추가됨")
    
    # 토크나이저 정보 출력
    logger.info(f"BOS token: {tokenizer.bos_token}, ID: {tokenizer.bos_token_id}")
    logger.info(f"EOS token: {tokenizer.eos_token}, ID: {tokenizer.eos_token_id}")
    logger.info(f"PAD token: {tokenizer.pad_token}, ID: {tokenizer.pad_token_id}")
    
    # 모델 로드 시 생성 관련 설정을 포함하지 않도록 구성
    model_config = BartConfig.from_pretrained("facebook/bart-base")
    # 생성 관련 설정 제거
    for param in ['num_beams', 'max_length', 'early_stopping', 'no_repeat_ngram_size', 
                  'length_penalty', 'forced_bos_token_id', 'forced_eos_token_id']:
        if hasattr(model_config, param):
            delattr(model_config, param)
    
    model = BartForConditionalGeneration.from_pretrained(
        "facebook/bart-base", 
        config=model_config,
        ignore_mismatched_sizes=True
    )
    
    # 임베딩 크기 확장 (특수 토큰 수용)
    model.resize_token_embeddings(len(tokenizer))
    
    # 생성 설정 구성 
    generation_config = GenerationConfig(
        max_length=args.max_tgt_length,
        num_beams=5,
        early_stopping=False,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        repetition_penalty=1.2,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    model.generation_config = generation_config

    # -------- dataset --------
    use_cache = not args.no_cache  # 캐시 사용 여부
    train_ds = DialogGenDataset(
        "train", tokenizer, args.max_src_length, args.max_tgt_length,
        tiny_frac=args.tiny_frac, dataset_name=args.dataset, use_cache=use_cache
    )
    val_ds = DialogGenDataset(
        "validation", tokenizer, args.max_src_length, args.max_tgt_length,
        tiny_frac=args.tiny_frac, dataset_name=args.dataset, use_cache=use_cache
    )
    test_ds = DialogGenDataset(
        "test", tokenizer, args.max_src_length, args.max_tgt_length,
        tiny_frac=args.tiny_frac, dataset_name=args.dataset, use_cache=use_cache
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, padding="longest")

    # -------- training args --------
    # PerplexityCallback 정의
    class PerplexityCallback(TrainerCallback):
        """Eval 루프 이후 metrics dict에 eval_ppl 키를 추가한다."""
        def on_evaluate(self, args, state, control, **kwargs):
            metrics = kwargs.get("metrics", {})
            if metrics is not None and "eval_loss" in metrics:
                metrics["eval_ppl"] = float(np.exp(metrics["eval_loss"]))
            return control

    t_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=0.01,
        warmup_ratio=args.warmup_ratio,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.eval_steps,
        save_total_limit=1,
        logging_strategy="steps",
        logging_steps=args.eval_steps,
        predict_with_generate=True,
        load_best_model_at_end=True,
        metric_for_best_model="eval_ppl",
        greater_is_better=False,  # PPL은 낮을수록 좋음
        report_to="none",
        seed=args.seed,
    )

    # 메트릭 계산 함수
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels[labels == -100] = tokenizer.pad_token_id

        # 텍스트로 디코딩
        gen_txt = tokenizer.batch_decode(preds, skip_special_tokens=True)
        ref_txt = tokenizer.batch_decode(labels, skip_special_tokens=True)

        # 생성 메트릭 계산 (BLEU, ROUGE 등)
        metrics = generation_metrics(gen_txt, ref_txt)
        
        # 평가 메트릭에 "eval_" 접두사 추가
        for k, v in list(metrics.items()):
            metrics[f'eval_{k}'] = v

        logger.info(
            f"📊 Eval Metrics: BLEU-1={metrics['eval_bleu1']:.4f}, ROUGE-L={metrics['eval_rouge_l']:.4f}")

        return metrics

    trainer = Seq2SeqTrainer(
        model=model,
        args=t_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=args.patience), PerplexityCallback()],
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
            json.dump({k: float(v) if isinstance(v, (int, float, np.number)) else v 
                       for k, v in init_eval.items()}, f, indent=2)
        
        # 테스트 데이터셋 평가
        logger.info("📊 Evaluating initial model on test split ...")
        
        
        # 테스트셋 평가
        test_out = trainer.predict(test_ds, metric_key_prefix="init_test")
        
        # PPL 계산
        test_ppl = float(np.exp(test_out.metrics.get('init_test_loss', 0)))
        logger.info(f"Initial Test Perplexity: {test_ppl:.4f}")
        
        # 결과 저장
        init_test_path = Path(args.output_dir) / "init_test_metrics.json"
        with init_test_path.open("w", encoding="utf-8") as f:
            json.dump({k: float(v) if isinstance(v, (int, float, np.number)) else v 
                       for k, v in test_out.metrics.items()}, f, indent=2)
        
        # 샘플 저장
        lbl_ids = test_out.label_ids.copy()
        lbl_ids[lbl_ids == -100] = tokenizer.pad_token_id
        gen_txt = tokenizer.batch_decode(test_out.predictions, skip_special_tokens=False)
        ref_txt = tokenizer.batch_decode(lbl_ids, skip_special_tokens=False)
        
        sample_n = min(10, len(gen_txt))
        with open(Path(args.output_dir) / "init_samples.txt", "w", encoding="utf-8") as f:
            for i, (ref, gen) in enumerate(zip(ref_txt[:sample_n], gen_txt[:sample_n])):
                context = test_ds.examples[i]["context"]
                f.write(f"CONTEXT: {context}\nREF: {ref}\nGEN: {gen}\n---\n")
        
        logger.info(f"📝 Saved initial model metrics to {args.output_dir}")
    
    # 학습 수행 (eval_only가 True면 학습 건너뜀)
    if not args.eval_only:
        trainer.train()
        trainer.save_model(args.output_dir)
        
        # --------------------- test split 평가 ---------------------
        logger.info("테스트 데이터셋 평가 중...")


        # 테스트 데이터 평가
        test_out = trainer.predict(test_ds, metric_key_prefix="test")

        # PPL 계산
        test_ppl = float(np.exp(test_out.metrics.get('test_loss', 0)))
        logger.info(f"Test Perplexity: {test_ppl:.4f}")

        # 메트릭 저장
        test_metrics = test_out.metrics
        test_metrics['test_perplexity'] = test_ppl
        
        with open(Path(args.output_dir) / "test_metrics.json", "w") as f:
            json.dump({k: float(v) if isinstance(v, (int, float, np.number)) else v 
                       for k, v in test_metrics.items()}, f, indent=2)

        # 샘플 저장
        lbl_ids = test_out.label_ids.copy()
        lbl_ids[lbl_ids == -100] = tokenizer.pad_token_id
        gen_txt = tokenizer.batch_decode(test_out.predictions, skip_special_tokens=False)
        ref_txt = tokenizer.batch_decode(lbl_ids, skip_special_tokens=False)

        sample_n = min(20, len(gen_txt))
        with open(Path(args.output_dir) / "test_samples.txt", "w", encoding="utf-8") as f:
            for i, (ref, gen) in enumerate(zip(ref_txt[:sample_n], gen_txt[:sample_n])):
                context = test_ds.examples[i]["context"]
                f.write(f"CONTEXT: {context}\nREF: {ref}\nGEN: {gen}\n---\n")

        logger.info(f"모델 및 테스트 메트릭 저장 완료: {args.output_dir}")

    if args.show_samples:
        import textwrap
        # 토크나이저로 확인하는 샘플 데이터
        for i in random.sample(range(len(train_ds)), args.show_samples):
            ex = train_ds[i]

            ctx_plain = safe_decode([t if t != -100 else tokenizer.pad_token_id for t in ex["input_ids"]], tokenizer, skip_special_tokens=False)
            tgt_plain = safe_decode([t if t != -100 else tokenizer.pad_token_id for t in ex["labels"]], tokenizer, skip_special_tokens=False)
            
            logging.info(
                "\n===== SAMPLE {:d} =====\n"
                "CONTEXT:\n{}\n\n"
                "TARGET:\n{}\n"
                "{}".format(
                    i,
                    ctx_plain, 
                    tgt_plain,
                    "=" * 60
                )
            )
        
        # 실제 생성 테스트
        logger.info("\n===== 생성 샘플 테스트 =====")
        # 모델을 평가 모드로 전환
        model.eval()
        
        # 몇 개의 랜덤 샘플로 생성 테스트
        for i in random.sample(range(len(val_ds)), min(3, len(val_ds))):
            ex = val_ds[i]
            input_ids = torch.tensor([ex["input_ids"]]).to(model.device)
            attention_mask = torch.tensor([ex["attention_mask"]]).to(model.device)
            
            # 텍스트 생성
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    max_length=args.max_tgt_length,
                )
            
            # 디코딩 및 출력 - pad 토큰만 제거하고 특수 토큰은 유지
            skip_special_tokens = False

            generated_text = safe_decode(outputs[0].tolist(), tokenizer, skip_special_tokens=skip_special_tokens)
            target_text = safe_decode([t if t != -100 else tokenizer.pad_token_id for t in ex["labels"]], tokenizer, skip_special_tokens=skip_special_tokens)
            context_text = safe_decode([t if t != -100 else tokenizer.pad_token_id for t in ex["input_ids"]], tokenizer, skip_special_tokens=skip_special_tokens)
            
            logger.info(
                "\n----- 생성 샘플 {:d} -----\n"
                "입력 문맥:\n{}\n\n"
                "정답 응답:\n{}\n\n"
                "생성 응답:\n{}\n"
                "{}".format(
                    i,
                    context_text,  
                    target_text,   
                    generated_text,
                    "-" * 60
                )
            )
        
        return


if __name__ == "__main__":
    main() 