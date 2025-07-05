# -*- coding: utf-8 -*-
"""
bart_dialog_generator_gt_prefix.py
============================
BART 모델을 사용하여 대화 맥락과 전략(strategy)을 기반으로 자연스러운 응답을 생성하는 파이프라인.

사용법 예시:
----------
# 기본 훈련
CUDA_VISIBLE_DEVICES=1 python bart_dialog_generator_gt_prefix.py --batch_size 16 --output_dir outputs/dialog_generation_gt_prefix --no_save_optimizer

# 작은 비율의 데이터로 빠른 테스트
CUDA_VISIBLE_DEVICES=0 python bart_dialog_generator_gt_prefix.py --tiny_frac 0.05 --epochs 1 --eval_steps 10 --output_dir outputs/dialog_tiny_gt_prefix --no_save_optimizer

# facebook/bart-base 원본 모델 평가
CUDA_VISIBLE_DEVICES=2 python bart_dialog_generator_gt_prefix.py --eval_only --output_dir outputs/dialog_eval_gt_prefix

# 그래디언트 누적을 사용한 대용량 배치 학습 (실효 배치 크기 32)
CUDA_VISIBLE_DEVICES=3 python bart_dialog_generator_gt_prefix.py --gradient_accumulation_steps 2 --output_dir outputs/dialog_batch_32_gt_prefix
"""

from __future__ import annotations

import argparse, json, logging, random
from pathlib import Path
from typing import Dict, List, Union, Any, Optional

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
    set_seed,
    BartConfig,
    PreTrainedTokenizerBase,
)
from utils.metrics import generation_metrics

# ===================== 전역 정의 =====================
SPECIAL_TOKENS = {         
    "usr": "[USR]",
    "sys": "[SYS]",
    "Question": "[STRAT_Question]",
    "Restatement or Paraphrasing": "[STRAT_Paraphrasing]",
    "Reflection of feelings": "[STRAT_Reflection]",
    "Self-disclosure": "[STRAT_SelfDisclosure]",
    "Affirmation and Reassurance": "[STRAT_Reassurance]",
    "Providing Suggestions": "[STRAT_Suggestion]",
    "Information": "[STRAT_Information]",
    "Others": "[STRAT_Others]",
}

logger = logging.getLogger("dialog_gen")

# ===================== 데이터셋 =====================
class DialogGenDataset(torch.utils.data.Dataset):
    """
    대화 생성을 위한 데이터셋.
    
    대화 맥락(context)을 입력으로, 시스템 응답을 출력으로 하는 데이터셋
    """

    def __init__(
        self,
        split: str,
        tokenizer: BartTokenizer,
        max_src: int = 1024,
        max_tgt: int = 256,
        tiny_frac: float | None = None,
        cache_dir: str = "cache_dialog_gen_gt_prefix",
        dataset_name: str = "thu-coai/esconv",
        use_cache: bool = True,
        bos_is_strategy: bool = False,
    ):
        self.tok = tokenizer
        self.max_src, self.max_tgt = max_src, max_tgt
        self.dataset_name = dataset_name
        self.bos_is_strategy = bos_is_strategy

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
                    if prev["speaker"] == "usr":
                        ctx_parts.append(f"{spk_tok}{prev['text']}")
                    else:
                        strategy = prev["strategy"]
                        ctx_parts.append(f"{spk_tok}{SPECIAL_TOKENS[strategy]}{prev['text']}")

                # 특수 토큰을 문자열이 아니라 토크나이저의 bos/eos 토큰으로 사용
                context = tokenizer.bos_token + (tokenizer.eos_token.join(ctx_parts) if ctx_parts else "") + tokenizer.eos_token

                if not context.strip():
                    continue

                # ---------- decoder output (response) ----------
                # 1) prefix 토큰 순서 결정 (변경: 맨 앞에 EOS 추가하여 길이 3)
                eos_id = tokenizer.eos_token_id
                sys_tok_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS["sys"])
                strat_tok_id = tokenizer.convert_tokens_to_ids(SPECIAL_TOKENS[turn["strategy"]])

                if self.bos_is_strategy:
                    prefix_ids = [eos_id, strat_tok_id, sys_tok_id]
                else:
                    prefix_ids = [eos_id, sys_tok_id, strat_tok_id]

                # 1) 응답 문장 토큰화 (EOS 포함)
                sent_ids = self.tok(
                    turn["text"],
                    add_special_tokens=False,
                    max_length=self.max_tgt - 4,  # prefix 3 + EOS 1 보장
                    truncation=True,
                ).input_ids

                tokens = sent_ids + [eos_id]  # w1..wn + EOS (길이 n+1)

                # 2) decoder_input / labels 구성 (길이 차 1)
                #   decoder_input :  </s> [SYS] [STRAT] w1 .. wN-1
                #   labels        :          [SYS] [STRAT] w1 .. wN   (첫 </s> 예측은 [SYS])
                decoder_input = prefix_ids + tokens[:-1]  # 마지막 EOS 제거
                labels        = prefix_ids[1:] + tokens   # prefix에도 loss 부여 (</s> 제외)

                # 3) 길이 맞춤 (max_tgt)
                # truncate
                decoder_input = decoder_input[: self.max_tgt]
                labels        = labels[: self.max_tgt]

                # pad (decoder_input은 labels보다 최대 1 짧음) -> 동일 길이 후 max_tgt까지 pad
                while len(decoder_input) < len(labels):
                    decoder_input.append(tokenizer.pad_token_id)

                pad_len = self.max_tgt - len(labels)
                decoder_input += [tokenizer.pad_token_id] * pad_len
                labels        += [-100] * pad_len

                decoder_attn = [1 if tid != tokenizer.pad_token_id else 0 for tid in decoder_input]

                # ----------- 인코더 토큰화 -----------
                enc = self.tok(
                    context,
                    max_length=self.max_src,
                    truncation=True,
                    padding="max_length",
                    add_special_tokens=False,
                )

                self.examples.append({
                    "input_ids": enc.input_ids,
                    "attention_mask": enc.attention_mask,
                    "decoder_input_ids": decoder_input,
                    "decoder_attention_mask": decoder_attn,
                    "labels": labels,
                    "context": context,
                    "response": turn["text"],
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
    # 리스트와 텐서를 모두 허용
    if isinstance(ids, torch.Tensor):
        ids = ids.tolist()

    if not isinstance(ids, list):
        ids = [ids]

    # 1) pad 이후 자르기
    if tokenizer.pad_token_id in ids:
        first_pad = ids.index(tokenizer.pad_token_id)
        ids = ids[:first_pad]

    # 2) 음수/None/범위 초과 ID 제거
    filtered = [int(t) for t in ids if isinstance(t, int) and 0 <= t < len(tokenizer)]

    if not filtered:
        return ""

    try:
        return tokenizer.decode(filtered, skip_special_tokens=skip_special_tokens, **kwargs)
    except Exception as e:
        logger.debug(f"safe_decode 재시도 실패: {e}")
        return ""


# ===================== 커스텀 데이터 콜레이터 =====================
class CustomDataCollatorWithDecoderPrefix(DataCollatorForSeq2Seq):
    """
    커스텀 데이터 콜레이터: decoder_input_ids와 labels 처리를 위한 특별 로직
    - decoder_input_ids: </s> [STRAT] [SYS] (또는 </s> [SYS] [STRAT]) 3-토큰 prefix 유지
    - labels: prefix 부분 -100 마스킹 보장
    """
    def __init__(
        self,
        tokenizer: PreTrainedTokenizerBase,
        model=None,
        label_pad_token_id: int = -100,
        decoder_prefix_len: int = 0,
        **kwargs
    ):
        super().__init__(tokenizer, model, label_pad_token_id=label_pad_token_id, **kwargs)
        self.decoder_prefix_len = decoder_prefix_len

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # 원본 패딩 처리
        batch = super().__call__(features)

        # 디버그 로깅 (첫 호출에서만)
        if not hasattr(self, '_logged_once'):
            logger.info("\n===== 커스텀 콜레이터 디버깅 =====")
            logger.info(f"배치 키: {list(batch.keys())}")
            if 'decoder_input_ids' in batch:
                logger.info(f"decoder_input_ids 샘플: {batch['decoder_input_ids'][0][:10].tolist()}")
            if 'labels' in batch:
                logger.info(f"labels 샘플: {batch['labels'][0][:10].tolist()}")
            self._logged_once = True

        # labels에서 prefix 부분에 -100 마스킹 보장
        if "labels" in batch and "decoder_input_ids" in batch:
            labels = batch["labels"]
            # prefix_len 길이만큼 -100으로 마스킹
            labels[:, :self.decoder_prefix_len] = self.label_pad_token_id
            batch["labels"] = labels

        return batch


# ===================== 메인 =====================
def main():
    # NLTK 데이터 다운로드 (필요시)
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        nltk.download('punkt')
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="outputs/dialog_gen")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="그래디언트 누적 스텝 수 (실제 배치 크기 = batch_size * gradient_accumulation_steps)")
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
    parser.add_argument("--max_src_length", type=int, default=1024,
                        help="최대 소스 길이")
    parser.add_argument("--max_tgt_length", type=int, default=256,
                        help="최대 타겟 길이")
    parser.add_argument("--warmup_ratio", type=float, default=0.05,
                        help="warmup 비율 (전체 스텝의 %, 초기 불안정한 loss spike 방지)")
    parser.add_argument("--num_samples", type=int, default=5, 
                        help="학습/평가 중 출력할 샘플 수")
    parser.add_argument("--no_cache", action="store_true",
                        help="캐시를 사용하지 않고 항상 데이터를 새로 처리")
    parser.add_argument("--no_save_optimizer", action="store_true",
                        help="체크포인트 저장 시 optimizer/scheduler state를 저장하지 않음 (디스크/I-O 절약)")
    parser.add_argument("--bos_is_strategy", action="store_true",
                        help="디코더 prefix를 [STRAT][SYS] 순서로 둘지 여부. 기본은 [SYS][STRAT].")
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
        min_length=5,
        num_beams=5,
        early_stopping=False,
        no_repeat_ngram_size=3,
        length_penalty=1.0,
        repetition_penalty=1.2,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model.generation_config = generation_config

    # -------- dataset --------
    use_cache = not args.no_cache  # 캐시 사용 여부
    train_ds = DialogGenDataset(
        "train", tokenizer, args.max_src_length, args.max_tgt_length,
        tiny_frac=args.tiny_frac, dataset_name=args.dataset, use_cache=use_cache,
        bos_is_strategy=args.bos_is_strategy
    )
    val_ds = DialogGenDataset(
        "validation", tokenizer, args.max_src_length, args.max_tgt_length,
        tiny_frac=args.tiny_frac, dataset_name=args.dataset, use_cache=use_cache,
        bos_is_strategy=args.bos_is_strategy
    )
    test_ds = DialogGenDataset(
        "test", tokenizer, args.max_src_length, args.max_tgt_length,
        tiny_frac=args.tiny_frac, dataset_name=args.dataset, use_cache=use_cache,
        bos_is_strategy=args.bos_is_strategy
    )

    # 커스텀 콜레이터 사용: decoder_input_ids 유지 (prefix 마스킹 해제)
    data_collator = CustomDataCollatorWithDecoderPrefix(
        tokenizer=tokenizer,
        model=None,  # shift 작업 비활성화
        padding="longest",
        decoder_prefix_len=0,  # prefix에도 loss를 주므로 마스킹하지 않음
    )

    # -------------------- Safe Trainer 정의 --------------------
    class SafeSeq2SeqTrainer(Seq2SeqTrainer):
        """RuntimeError 발생 시 optimizer state 저장을 건너뛰는 Trainer."""

        def _save_optimizer_and_scheduler(self, output_dir: str):  # type: ignore
            if args.no_save_optimizer:
                logger.info("⚠️  no_save_optimizer 플래그가 설정되어 optimizer/scheduler state 저장을 건너뜁니다.")
                return
            try:
                return super()._save_optimizer_and_scheduler(output_dir)
            except RuntimeError as e:
                logger.warning(f"optimizer/scheduler 저장 실패: {e}. 해당 스텝에서 저장을 건너뜁니다.")
                torch.cuda.empty_cache()
                return

    # -------------------- Prefix-aware Trainer --------------------
    class PrefixSeq2SeqTrainer(SafeSeq2SeqTrainer):
        """Seq2SeqTrainer 확장: generate 시 decoder prefix 세 토큰(</s> [STRAT] [SYS] 또는 </s> [SYS] [STRAT])을 사용"""

        def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
            if not self.args.predict_with_generate or prediction_loss_only:
                return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

            # prefix 세 토큰 추출 (패딩 제외)
            decoder_prefix = inputs["decoder_input_ids"][:, :3].to(model.device)  # 변경: prefix 길이 3

            # 생성 시 decoder_input_ids 로깅 (첫 예측에서만)
            if self.state.global_step < 10 or self.state.global_step % 100 == 0:
                logger.info("\n===== 생성 입력 디버깅 =====")
                logger.info(f"decoder_prefix 형태: {decoder_prefix.shape}")
                logger.info(f"첫 샘플 decoder_prefix: {decoder_prefix[0].tolist()}")
                
                # 디코딩해서 확인
                prefix_text = tokenizer.decode(decoder_prefix[0], skip_special_tokens=False)
                logger.info(f"첫 샘플 prefix 디코딩: '{prefix_text}'")

            gen_kwargs = {
                "max_length": self.args.generation_max_length or model.generation_config.max_length,
                "num_beams": self.args.generation_num_beams or model.generation_config.num_beams,
                "decoder_input_ids": decoder_prefix,
            }

            # 생성 직전 최종 확인
            logger.info(f"생성 kwargs: {gen_kwargs}")

            generated_tokens = model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )

            # padding output length
            if generated_tokens.shape[-1] < gen_kwargs["max_length"]:
                pad_length = gen_kwargs["max_length"] - generated_tokens.shape[-1]
                pad_tensor = torch.full((generated_tokens.shape[0], pad_length), model.generation_config.pad_token_id, dtype=generated_tokens.dtype, device=generated_tokens.device)
                generated_tokens = torch.cat([generated_tokens, pad_tensor], dim=-1)

            loss = None
            if "labels" in inputs:
                with torch.no_grad():
                    outputs = model(**inputs)
                    loss = outputs.loss.detach()

            return loss, generated_tokens, inputs.get("labels")

        def training_step(self, model, inputs, num_items_in_batch=None):
            """학습 스텝 오버라이드: decoder_input_ids와 labels 로깅"""
            # 첫 배치의 첫 샘플만 로깅 (과도한 출력 방지)
            if self.state.global_step == 0:
                logger.info("\n===== 학습 입력 디버깅 =====")
                logger.info(f"decoder_input_ids 샘플: {inputs['decoder_input_ids'][0][:10].tolist()}")
                logger.info(f"labels 샘플: {inputs['labels'][0][:10].tolist()}")
                
                # 첫 샘플로 직접 forward 호출 테스트
                logger.info("\n===== 직접 모델 호출 테스트 =====")
                test_inputs = {
                    "input_ids": inputs["input_ids"][0:1],
                    "attention_mask": inputs["attention_mask"][0:1],
                    "decoder_input_ids": inputs["decoder_input_ids"][0:1],
                    "labels": inputs["labels"][0:1]
                }
                with torch.no_grad():
                    outputs = model(**test_inputs)
                logger.info(f"직접 호출 loss: {outputs.loss.item()}")
                
                # 디코딩해서 확인
                dec_in = tokenizer.decode(inputs["decoder_input_ids"][0], skip_special_tokens=False)
                labels_masked = inputs["labels"][0].clone()
                labels_masked[labels_masked == -100] = tokenizer.pad_token_id
                labels_txt = tokenizer.decode(labels_masked, skip_special_tokens=False)
                logger.info(f"디코더 입력 디코딩: {dec_in[:100]}")
                logger.info(f"라벨 디코딩: {labels_txt[:100]}")
            
            return super().training_step(model, inputs, num_items_in_batch)

    # -------- training args --------
    t_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
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
        metric_for_best_model="eval_loss",
        greater_is_better=False,  # loss는 낮을수록 좋음
        report_to="none",
        seed=args.seed,
        # 메모리 체크 비활성화로 안정성 향상
        dataloader_drop_last=False,
        skip_memory_metrics=True,
    )
    
    # 실제 배치 크기 로그 출력
    effective_batch_size = args.batch_size * args.gradient_accumulation_steps
    logger.info(f"배치 크기: {args.batch_size}, 그래디언트 누적 단계: {args.gradient_accumulation_steps}")
    logger.info(f"실효 배치 크기: {effective_batch_size} (batch_size * gradient_accumulation_steps)")

    # 메트릭 계산 함수
    def compute_metrics(eval_pred):
        preds, labels = eval_pred
        labels[labels == -100] = tokenizer.pad_token_id

        # 메트릭 딕셔너리 초기화
        metrics = {}

        # 텍스트로 디코딩
        gen_raw = tokenizer.batch_decode(preds, skip_special_tokens=True)
        ref_raw = tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # MultiESC 스타일 전처리: 소문자화 및 nltk 토큰화
        gen_txt = []
        ref_txt = []
        for gen, ref in zip(gen_raw, ref_raw):
            # 소문자화 후 nltk 토큰화하여 공백으로 재결합
            gen_processed = ' '.join(nltk.word_tokenize(gen.lower()))
            ref_processed = ' '.join(nltk.word_tokenize(ref.lower()))
            gen_txt.append(gen_processed)
            ref_txt.append(ref_processed)

        # 생성 메트릭 계산 (BLEU, ROUGE 등)
        text_metrics = generation_metrics(gen_txt, ref_txt)
        
        # 평가 메트릭에 "eval_" 접두사 추가
        for k, v in text_metrics.items():
            metrics[f'eval_{k}'] = v

        return metrics

    # 체크포인트 로딩 관련 오류를 해결하기 위한 콜백
    class TokenEmbeddingCallback(TrainerCallback):
        """체크포인트 로드 시 토큰 임베딩 크기를 올바르게 유지"""
        def __init__(self, tokenizer, special_token_ids):
            self.tokenizer = tokenizer
            self.special_token_ids = special_token_ids
        
        def on_train_begin(self, args, state, control, model=None, **kwargs):
            """훈련 시작 시 모델 임베딩 크기 확인"""
            if model is not None:
                model.resize_token_embeddings(len(self.tokenizer))
                logger.info(f"모델 임베딩 크기 조정: {len(self.tokenizer)}")
        
        def on_step_end(self, args, state, control, model=None, **kwargs):
            """각 스텝 후 임베딩 크기 확인"""
            if state.global_step % 1000 == 0 and model is not None:
                if model.get_input_embeddings().weight.shape[0] != len(self.tokenizer):
                    logger.warning(f"임베딩 크기 불일치 감지: {model.get_input_embeddings().weight.shape[0]} vs {len(self.tokenizer)}")
                    model.resize_token_embeddings(len(self.tokenizer))
        
        def on_load_checkpoint(self, args, state, control, **kwargs):
            """체크포인트 로드 시 호출"""
            logger.info("체크포인트 로드 중...")
            
        def on_checkpoint_model_loading(self, args, state, control, model=None, **kwargs):
            """체크포인트에서 모델 로드 시 호출"""
            if model is not None:
                # 임베딩 크기 조정
                model.resize_token_embeddings(len(self.tokenizer))
                logger.info(f"체크포인트 로드 후 모델 임베딩 크기 조정: {len(self.tokenizer)}")
                
                # 생성 설정 다시 적용
                model.generation_config = generation_config
                
                # 미리 구성된 모델로 초기 가중치 복사 (선택 사항)
                if hasattr(model, 'lm_head') and not hasattr(model.lm_head, 'weight'):
                    logger.warning("lm_head에 weight가 없음, 가중치 초기화 필요")
                    # 필요한 경우 가중치 초기화 로직 추가

    # 특수 토큰 ID 목록
    special_token_ids = [
        tokenizer.bos_token_id, 
        tokenizer.eos_token_id, 
        tokenizer.pad_token_id,
        *[tokenizer.convert_tokens_to_ids(t) for t in SPECIAL_TOKENS.values()]
    ]
    
    # 최종 콜백 목록 생성
    callbacks = [
        EarlyStoppingCallback(
            early_stopping_patience=args.patience,
            early_stopping_threshold=0.0
        ),
        TokenEmbeddingCallback(tokenizer, special_token_ids)
    ]

    trainer = PrefixSeq2SeqTrainer(
        model=model,  # 초기 모델 (학습 전용)
        args=t_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        callbacks=callbacks,
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
        
        # 샘플 저장용 원본 텍스트 (MultiESC 전처리 전)
        lbl_ids = test_out.label_ids.copy()
        lbl_ids[lbl_ids == -100] = tokenizer.pad_token_id
        
        gen_raw = tokenizer.batch_decode(test_out.predictions, skip_special_tokens=True)
        ref_raw = tokenizer.batch_decode(lbl_ids, skip_special_tokens=True)
        
        # 메트릭 계산용 텍스트 (MultiESC 스타일 전처리)
        gen_txt = []
        ref_txt = []
        for gen, ref in zip(gen_raw, ref_raw):
            gen_processed = ' '.join(nltk.word_tokenize(gen.lower()))
            ref_processed = ' '.join(nltk.word_tokenize(ref.lower()))
            gen_txt.append(gen_processed)
            ref_txt.append(ref_processed)
            
        # 메트릭 계산
        text_metrics = generation_metrics(gen_txt, ref_txt)
        
        # 메트릭 저장
        init_test_metrics = {}
        for k, v in text_metrics.items():
            init_test_metrics[f'init_test_{k}'] = v
        init_test_metrics.update(test_out.metrics)
        init_test_metrics['init_test_perplexity'] = test_ppl
        
        # 결과 저장
        init_test_path = Path(args.output_dir) / "init_test_metrics.json"
        with init_test_path.open("w", encoding="utf-8") as f:
            json.dump({k: float(v) if isinstance(v, (int, float, np.number)) else v 
                       for k, v in init_test_metrics.items()}, f, indent=2)
        
        # 샘플 저장 (원본 텍스트로 저장)
        sample_n = min(10, len(gen_raw))
        with open(Path(args.output_dir) / "init_samples.txt", "w", encoding="utf-8") as f:
            for i, (ref, gen) in enumerate(zip(ref_raw[:sample_n], gen_raw[:sample_n])):
                context = test_ds.examples[i]["context"]
                f.write(f"CONTEXT: {context}\nREF: {ref}\nGEN: {gen}\n---\n")
        
        logger.info(f"📝 Saved initial model metrics to {args.output_dir}")
    
    # 학습 수행 (eval_only가 True면 학습 건너뜀)
    if not args.eval_only:
        trainer.train()
        
        # 학습 완료 후 저장 - 중요: 안전한 방식으로 저장
        # safe_serialization=True는 미래 호환성을 위한 옵션
        model_path = Path(args.output_dir)
        model.save_pretrained(model_path, safe_serialization=True)
        tokenizer.save_pretrained(model_path)
        
        logger.info(f"Model explicitly saved to {args.output_dir}")
        
        # --------------------- test split 평가 ---------------------
        logger.info("테스트 데이터셋 평가 중...")

        # 테스트 데이터 평가
        test_out = trainer.predict(test_ds, metric_key_prefix="test")

        # PPL 계산
        test_ppl = float(np.exp(test_out.metrics.get('test_loss', 0)))
        logger.info(f"Test Perplexity: {test_ppl:.4f}")

        # 샘플 저장용 원본 텍스트 (MultiESC 전처리 전)
        lbl_ids = test_out.label_ids.copy()
        lbl_ids[lbl_ids == -100] = tokenizer.pad_token_id
        
        gen_raw = tokenizer.batch_decode(test_out.predictions, skip_special_tokens=True)
        ref_raw = tokenizer.batch_decode(lbl_ids, skip_special_tokens=True)
        
        # 메트릭 계산용 텍스트 (MultiESC 스타일 전처리)
        gen_txt = []
        ref_txt = []
        for gen, ref in zip(gen_raw, ref_raw):
            gen_processed = ' '.join(nltk.word_tokenize(gen.lower()))
            ref_processed = ' '.join(nltk.word_tokenize(ref.lower()))
            gen_txt.append(gen_processed)
            ref_txt.append(ref_processed)

        # 메트릭 계산
        text_metrics = generation_metrics(gen_txt, ref_txt)
        
        # 메트릭 저장
        test_metrics = {}
        for k, v in text_metrics.items():
            test_metrics[f'test_{k}'] = v
        test_metrics.update(test_out.metrics)
        test_metrics['test_perplexity'] = test_ppl
        
        with open(Path(args.output_dir) / "test_metrics.json", "w") as f:
            json.dump({k: float(v) if isinstance(v, (int, float, np.number)) else v 
                       for k, v in test_metrics.items()}, f, indent=2)

        # 샘플 저장 (원본 텍스트로 저장)
        sample_n = min(20, len(gen_raw))
        with open(Path(args.output_dir) / "test_samples.txt", "w", encoding="utf-8") as f:
            for i, (ref, gen) in enumerate(zip(ref_raw[:sample_n], gen_raw[:sample_n])):
                context = test_ds.examples[i]["context"]
                f.write(f"CONTEXT: {context}\nREF: {ref}\nGEN: {gen}\n---\n")

        logger.info(f"모델 및 테스트 메트릭 저장 완료: {args.output_dir}")

    if args.show_samples:
        import textwrap
        # 토크나이저로 확인하는 샘플 데이터
        for i in random.sample(range(len(train_ds)), args.show_samples):
            ex = train_ds[i]

            ctx_plain = safe_decode(ex["input_ids"], tokenizer, skip_special_tokens=False)
            tgt_plain = safe_decode(ex["labels"], tokenizer, skip_special_tokens=False)
            
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
                # 디코더 prefix 로깅
                decoder_prefix = torch.tensor([ex["decoder_input_ids"][:3]]).to(model.device)
                logger.info(f"\n샘플 생성 테스트 - decoder_prefix: {decoder_prefix.tolist()}")
                logger.info(f"디코딩된 prefix: '{tokenizer.decode(decoder_prefix[0], skip_special_tokens=False)}'")
                
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_input_ids=decoder_prefix,
                    max_length=args.max_tgt_length,
                )
            
            # 디코딩 및 출력 - pad 토큰만 제거하고 특수 토큰은 유지
            skip_special_tokens = True

            generated_text = safe_decode(outputs[0].tolist(), tokenizer, skip_special_tokens=skip_special_tokens)
            target_text = safe_decode(ex["labels"], tokenizer, skip_special_tokens=skip_special_tokens)
            context_text = safe_decode(ex["input_ids"], tokenizer, skip_special_tokens=skip_special_tokens)
            
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