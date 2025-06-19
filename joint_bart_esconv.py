# -*- coding: utf-8 -*-
"""
joint_bart_esconv.py
====================
ESConv 전략 예측 + 응답 생성 Joint-Decoding 파이프라인

핵심 특징
----------
1. **데이터** : HuggingFace `thu-coai/esconv` → 시스템 턴마다 <BOS> <STRAT_xxx> 응답 시퀀스 구성.
2. **모델** : BART-base + classification head, 공동 loss `loss_gen + λ·loss_cls`.
3. **추론** : enc-CLS → strategy logits → 토큰 prefix 두고 `model.generate()`.
4. **멀티GPU** : `torchrun` 그대로 사용 (Trainer + DDP).
5. **메트릭** : 전략 acc/f1, 응답 BLEU1-4·ROUGE-L·METEOR·CIDEr·PPL.
6. **단일 파이썬 파일** 로 즉시 실행 가능.

실행 예시
----------
# 1 GPU
CUDA_VISIBLE_DEVICES=0 python joint_bart_esconv.py \
    --epochs 10 --batch_size 16 --lambda_cls 0.5 \
    --clean_checkpoints --eval_init \
    --output_dir outputs/joint

# 1GPU, tiny check
CUDA_VISIBLE_DEVICES=3 python joint_bart_esconv.py \
        --epochs 3 --batch_size 4 --tiny_frac 0.01 --lambda_cls 0.5 \
        --clean_checkpoints --eval_init --output_dir outputs/sanity1gpu
"""
from __future__ import annotations

import os, sys, argparse, json, logging, random, math
from datetime import datetime
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm.auto import tqdm

from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    DataCollatorForSeq2Seq,
    set_seed,
    GenerationConfig,
    EarlyStoppingCallback,
    TrainerCallback,
)

# ============================== global config & logging ==============================
SPECIAL_TOKENS = {
    "usr": "[USR]",
    "sys": "[SYS]",
    "sep": "[SEP]",
    "hist": "[STRAT_HIST]",
    "strategy": "[STRATEGY]",
    "strategy_placeholder": "[STRATEGY_EMBEDDING]",  # NEW: dynamic strategy embedding placeholder
}
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

logger = logging.getLogger("joint")

# ============================== dataset ==============================================
class JointESConvDataset(torch.utils.data.Dataset):
    """시스템 턴 단위 예제 생성 + joint decoding 포맷"""

    def __init__(self, split: str, tokenizer: BartTokenizer, max_src: int = 512, max_tgt: int = 64, cache_dir: str = "cache_joint"):
        self.tok = tokenizer
        self.max_src = max_src
        self.max_tgt = max_tgt
        raw = load_dataset("thu-coai/esconv")[split]

        cache_file = Path(cache_dir) / f"{split}.pt"
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        if cache_file.exists():
            self.examples = torch.load(cache_file, weights_only=True, map_location="cpu")
            return

        examples = []
        for ex in tqdm(raw, desc=f"proc {split}"):
            dialog = json.loads(ex["text"])["dialog"]
            sys_turns = [(i, t) for i, t in enumerate(dialog) if t["speaker"] == "sys"]
            for idx, (turn_idx, turn) in enumerate(sys_turns):
                strat = turn.get("strategy", "Others")
                strat_id = STR2ID.get(strat, STR2ID["Others"])

                # context : 모든 이전 turn + 역할 토큰
                ctx_parts = []
                for t in dialog[:turn_idx]:
                    spk_tok = SPECIAL_TOKENS["usr"] if t["speaker"] == "usr" else SPECIAL_TOKENS["sys"]
                    if t["speaker"] == "sys":
                        st = t.get("strategy", "Others")
                        st_tok = f"[STRAT_{st.replace(' ', '_')}]"
                        ctx_parts.append(f"{SPECIAL_TOKENS['strategy']} {st_tok} {spk_tok} {t['text']}")
                    else:
                        ctx_parts.append(f"{spk_tok} {t['text']}")
                context = " ".join(ctx_parts)

                # decoder input = BOS STRAT_TOKEN response
                d_in = f"{SPECIAL_TOKENS['strategy']} {SPECIAL_TOKENS['strategy_placeholder']} {SPECIAL_TOKENS['sys']} {turn['text']}"

                examples.append({
                    "context": context,
                    "decoder_text": d_in,
                    "response": turn["text"],
                    "strategy_id": strat_id,
                })
        self.examples = examples
        torch.save(self.examples, cache_file)

        if len(self.examples) > 0:
            print("\n===== 데이터셋 예제 샘플 =====")
            print(f"총 예제 수: {len(self.examples)}")
            sample_idx = min(len(self.examples)-1, 5)  # 5번째 예제 또는 마지막 예제
            ex = self.examples[sample_idx]
            print(f"Context: {ex['context']}...")  # 앞부분 100자만
            print(f"Decoder Text: {ex['decoder_text']}")
            print(f"Response: {ex['response']}")
            print(f"Strategy ID: {ex['strategy_id']} ({ID2STR[ex['strategy_id']]})")
            print("============================\n")

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, idx):
        ex = self.examples[idx]
        enc = self.tok(
            ex["context"], max_length=self.max_src, truncation=True, padding="max_length", return_tensors="pt"
        )
        dec = self.tok(
            ex["decoder_text"], max_length=self.max_tgt, truncation=True, padding="max_length", add_special_tokens=False, return_tensors="pt"
        )
        seq = dec["input_ids"].squeeze()             # length <= max_tgt
        seq = seq[: self.max_tgt - 1]                 # reserve 1 slot for EOS
        labels = torch.cat([seq, torch.tensor([self.tok.eos_token_id])])
        # pad to max_tgt
        if labels.size(0) < self.max_tgt:
            pad_len = self.max_tgt - labels.size(0)
            pad = torch.full((pad_len,), self.tok.pad_token_id, dtype=torch.long)
            labels = torch.cat([labels, pad])

        # mask loss on pad
        labels_masked = labels.clone()
        labels_masked[labels_masked == self.tok.pad_token_id] = -100

        # 첫 번째 토큰([STRATEGY])은 단순 마커이므로 loss 계산에서 제외
        labels_masked[0] = -100
        # 두 번째 토큰([STRATEGY_EMBEDDING])은 모델이 직접 임베딩을 주입하므로 loss 계산에서 제외
        labels_masked[1] = -100
        # 세 번째 토큰([SYS])도 단순 마커이므로 loss 계산에서 제외
        labels_masked[2] = -100

        decoder_input_ids = torch.cat([
            torch.tensor([self.tok.bos_token_id]),
            labels[:-1]  # everything except last token (EOS or PAD)
        ])

        return {
            "input_ids": enc["input_ids"].squeeze(),
            "attention_mask": enc["attention_mask"].squeeze(),
            "decoder_input_ids": decoder_input_ids,
            "labels": labels_masked,
            "strategy_id": torch.tensor(ex["strategy_id"], dtype=torch.long),
        }

# ============================== model ===============================================
class JointBart(nn.Module):
    def __init__(self, tokenizer: BartTokenizer, lambda_cls: float = 0.5):
        super().__init__()
        self.tok = tokenizer
        self.lambda_cls = lambda_cls
        self.model = BartForConditionalGeneration.from_pretrained("facebook/bart-base")
        self.model.resize_token_embeddings(len(tokenizer))
        self.classifier = nn.Linear(self.model.config.d_model, len(STRATEGIES))
        # h_cls → 디코더로 전달될 임베딩 투사용 projection
        self.strat_proj = nn.Linear(self.model.config.d_model, self.model.config.d_model, bias=False)
        nn.init.normal_(self.classifier.weight, std=0.02)
        self.model.generation_config.forced_bos_token_id = None
        self.model.generation_config.decoder_start_token_id = None   # prefix로 BOS 넣었으므로 안전장치

        # Seq2SeqTrainer 내부 로직은 self.model(generation_config)을 직접 참조하므로
        # wrapper 클래스에도 동일 속성을 노출해 준다.
        self.generation_config = self.model.generation_config

        # HF Trainer가 로드/세이브 과정에서 참조하는 안전용 attribute 추가
        self._keys_to_ignore_on_save: list[str] | None = []
        self._keys_to_ignore_on_load_missing: list[str] | None = []
        self._keys_to_ignore_on_load_unexpected: list[str] | None = []

    def forward(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids=None,
        labels=None,
        strategy_id=None,
    ):
        """Joint forward with dynamic strategy embedding injection."""
        # ---------------- 1) Encoder -----------------
        enc_outputs = self.model.model.encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
            return_dict=True,
        )
        h_cls = enc_outputs.last_hidden_state[:, 0]  # (B, d_model)

        # strategy classification
        strat_logits = self.classifier(h_cls)  # (B, |S|)

        # ---------------- 2') Classification only 경로 -----------------
        # evaluation 루틴 중 strategy accuracy 평가처럼, decoder_input_ids 없이
        # encoder CLS 기반 분류 점수만 필요할 때가 있다. 이 경우 디코더 및
        # 생성 loss 계산을 생략하고 strategy_logits 만 반환한다.

        if decoder_input_ids is None:
            from types import SimpleNamespace

            # 손쉬운 반환 객체 생성 (loss=None, encoder_last_hidden_state 포함)
            return SimpleNamespace(
                loss=None,
                strategy_logits=strat_logits,
                encoder_last_hidden_state=enc_outputs.last_hidden_state,
            )

        # ---------------- 3) Decoder embedding 구성 -----------------
        # token id → embedding
        # BART 모델은 인코더와 디코더가 동일한 토큰 임베딩(nn.Embedding)을 공유
        # 공유 임베딩 레이어가 바로 model.shared
        dec_emb = self.model.model.shared(decoder_input_ids)  # (B, L, d_model)

        # placeholder 위치는 항상 index 1 (BOS 뒤)
        dec_emb[:, 1, :] = self.strat_proj(h_cls)

        # ---------------- 4) 전체 모델 forward -----------------
        outputs = self.model(
            input_ids=None,  # encoder_outputs 로 대체
            attention_mask=attention_mask,
            encoder_outputs=enc_outputs,
            decoder_inputs_embeds=dec_emb,
            labels=labels,
            return_dict=True,
        )

        # ---------------- 5) Loss 합산 -----------------
        loss = outputs.loss
        if strategy_id is not None:
            cls_loss = F.cross_entropy(strat_logits, strategy_id)
            loss = loss + self.lambda_cls * cls_loss

        outputs.loss = loss
        outputs.strategy_logits = strat_logits
        return outputs

    # joint decoding
    def generate(self, input_ids, attention_mask, **gen_kwargs):
        with torch.no_grad():
            cls_hidden = self.model.model.encoder(
                input_ids=input_ids, attention_mask=attention_mask, return_dict=True
            ).last_hidden_state[:, 0]

            # 전략 로짓은 여전히 계산하되, 디코더 prefix는 고정 placeholder 사용
            strat_logits = self.classifier(cls_hidden)

            placeholder_id = self.tok.convert_tokens_to_ids(SPECIAL_TOKENS["strategy_placeholder"])
            batch_size = input_ids.size(0)
            prefix = torch.stack(
                [torch.tensor([self.tok.bos_token_id, placeholder_id]) for _ in range(batch_size)]
            ).to(input_ids.device)

            # -------- 디코딩 하이퍼파라미터 튜닝 --------
            # # (1) 문장 길이 제한: 10 ≤ length ≤ 50
            # gen_kwargs.setdefault("max_new_tokens", 50)
            # gen_kwargs.setdefault("min_length", 10)

            # (2) 샘플링 파라미터 – 자연스러움 & 다양성
            gen_kwargs.setdefault("do_sample", False)
            gen_kwargs.setdefault("no_repeat_ngram_size", 3)
            gen_kwargs.setdefault("repetition_penalty", 1.2)
            gen_kwargs.setdefault("length_penalty", 1.2)
            gen_kwargs.setdefault("num_beams", 5)
            gen_kwargs.setdefault("early_stopping", True)
            gen_kwargs.setdefault("eos_token_id", self.tok.eos_token_id)
             
            # 트레이너가 전달한 generation_max_length(=max_length) 값을 제거하여
            # "Both max_new_tokens and max_length" 경고 메시지를 방지한다.
            if "max_new_tokens" in gen_kwargs and "max_length" in gen_kwargs:
                gen_kwargs.pop("max_length")

            return self.model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                decoder_input_ids=prefix,
                **gen_kwargs,
            )

# ============================== metric utils ========================================
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.cider.cider import Cider

bleu_scorer = Bleu(4)
rouge_scorer = Rouge()
meteor_scorer = Meteor()
cider_scorer = Cider()


def generation_metrics(preds: list[str], refs: list[str]):
    gts = {i: [r] for i, r in enumerate(refs)}
    res = {i: [p] for i, p in enumerate(preds)}
    bleu, _ = bleu_scorer.compute_score(gts, res)
    rouge, _ = rouge_scorer.compute_score(gts, res)
    meteor, _ = meteor_scorer.compute_score(gts, res)
    cider, _ = cider_scorer.compute_score(gts, res)
    return {
        "bleu1": bleu[0],
        "bleu2": bleu[1],
        "bleu3": bleu[2],
        "bleu4": bleu[3],
        "rouge_l": rouge,
        "meteor": meteor,
        "cider": cider,
    }


def calculate_perplexity(loss):
    """
    손실값(cross-entropy loss)에서 perplexity 계산
    PPL = exp(평균 negative log likelihood) = exp(loss)
    """
    return float(np.exp(loss))

# ============================== trainer ==============================================
class JointTrainer(Seq2SeqTrainer):
    # ------------------------------------------------------------------
    # Seq2SeqTrainer 는 tokenizer 인자를 deprecated 처리하므로, 전달된 tokenizer
    #               → processing_class 로 옮겨서 경고를 잠재운다.
    # ------------------------------------------------------------------
    def __init__(self, *args, **kwargs):
        tok = kwargs.pop("tokenizer", None)
        if tok is not None and "processing_class" not in kwargs:
            kwargs["processing_class"] = tok  # 신버전 Trainer 가 권장하는 필드
        
        # 클래스 메서드를 compute_metrics로 설정 (중요: 기존에 전달된 것이 없을 때만)
        if 'compute_metrics' not in kwargs or kwargs['compute_metrics'] is None:
            kwargs['compute_metrics'] = self.compute_metrics
        
        # Best 모델 추적 변수
        self.best_metric = float("inf")  # eval_loss는 낮을수록 좋음
        self.best_model_dir = os.path.join(kwargs.get("args").output_dir, "best_model")
        os.makedirs(self.best_model_dir, exist_ok=True)
        self.metric_for_best_model = kwargs.get("args").metric_for_best_model
        self.greater_is_better = kwargs.get("args").greater_is_better
        
        import logging
        self.logger = logging.getLogger("joint")
        
        super().__init__(*args, **kwargs)
        
        # metric 계산용 캐시 초기화
        self.strategy_logits_cache: list[np.ndarray] = []
        self.sid_cache: list[np.ndarray] = []

    def compute_metrics(self, eval_pred):
        """텍스트 생성 및 전략 분류 메트릭 계산"""
        import logging
        logger = logging.getLogger("joint")
        
        preds, labels = eval_pred.predictions, eval_pred.label_ids
        
        # tokenizer 가져오기
        tok = getattr(self, "processing_class", None) or getattr(self, "tokenizer", None)
        
        # 기본 메트릭 초기화
        metrics = {}
        
        # PPL 계산
        if hasattr(eval_pred, 'metrics') and 'eval_loss' in eval_pred.metrics:
            metrics['eval_perplexity'] = calculate_perplexity(eval_pred.metrics['eval_loss'])
        
        # 생성 메트릭 계산 (predict_with_generate=True 필요)
        if isinstance(preds, np.ndarray) and preds.ndim > 1:
            try:
                # 안전한 디코딩을 위해 safe_batch_decode 사용
                preds_txt = safe_batch_decode(tok, preds)
                lbl = labels.copy()
                lbl[lbl == -100] = tok.pad_token_id
                refs_txt = safe_batch_decode(tok, lbl)
                
                # 생성 메트릭 계산 및 추가
                gen_m = generation_metrics(preds_txt, refs_txt)
                metrics.update(gen_m)
                
            except Exception as e:
                logger.warning(f"생성 메트릭 계산 중 오류 발생: {e}")
                # 기본값으로 0 설정
                metrics.update({
                    "bleu1": 0.0, "bleu2": 0.0, "bleu3": 0.0, "bleu4": 0.0,
                    "rouge_l": 0.0, "meteor": 0.0, "cider": 0.0
                })
        
        # strategy metrics 계산
        if len(self.strategy_logits_cache) > 0:
            logits = np.concatenate(self.strategy_logits_cache, axis=0)
            sid_gt = np.concatenate(self.sid_cache, axis=0)
            sid_pred = np.argmax(logits, axis=1)
            acc = accuracy_score(sid_gt, sid_pred)
            f1 = f1_score(sid_gt, sid_pred, average="weighted")
            metrics.update({"strategy_accuracy": acc, "strategy_f1": f1})
            
            # 다음 평가를 위해 캐시 정리
            self.strategy_logits_cache.clear()
            self.sid_cache.clear()
        
        # BLEU-1과 strategy_accuracy가 있다면 로그 출력
        if "bleu1" in metrics and "strategy_accuracy" in metrics:
            # 이 부분을 수정 - epoch가 None인 경우 처리
            if hasattr(self.state, "epoch") and self.state.epoch is not None:
                epoch_info = f"Epoch {self.state.epoch:.2f}"
            else:
                epoch_info = "Evaluation"
            logger.info(
                f"📊 {epoch_info} 메트릭: BLEU-1={metrics['bleu1']:.4f}, "
                f"Strategy Accuracy={metrics['strategy_accuracy']:.4f}"
            )
        
        # ----------------------- Best Model 관리 로직 ----------------------
        # 매 evaluation마다 metric을 확인하고 best model이 발견되면 즉시 저장
        current_metric = metrics.get(self.metric_for_best_model, None)
        
        # 항상 현재 best model 저장 (첫 eval에서는 무조건 best)
        if current_metric is None:
            logger.info(f"첫 번째 eval 결과, 모델 저장 (metric 없음)")
            self._save_best_model()
        else:
            # current_metric이 있으면 더 좋은지 비교
            is_better = self.greater_is_better and current_metric > self.best_metric
            is_better = is_better or (not self.greater_is_better and current_metric < self.best_metric)
            
            if is_better:
                old_metric = self.best_metric
                self.best_metric = current_metric
                logger.info(f"새로운 Best 모델 발견! {self.metric_for_best_model}: {old_metric:.4f} → {current_metric:.4f}")
                
                # best model 즉시 저장 (디렉터리는 덮어쓰기)
                self._save_best_model()
            else:
                logger.info(f"더 좋은 모델 아님: 현재 {self.metric_for_best_model}={current_metric:.4f}, 최고={self.best_metric:.4f}")
                
                # best_model 디렉토리가 비어있으면 현재 모델 저장 (첫 실행 시 대비)
                if not os.path.exists(os.path.join(self.best_model_dir, "pytorch_model.bin")):
                    logger.warning(f"Best 모델 파일이 없어 현재 모델 저장")
                    self._save_best_model()
        # ---------------------------------------------------------------------

        return metrics

    # custom caches
    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):  # type: ignore
        """Seq2SeqTrainer ➔ (loss, generated_tokens, labels) 반환 커스텀.

        1. super().prediction_step 로 loss / generated_tokens / labels 를 구한다.
        2. 같은 입력으로 model(**inputs) 을 한 번 더 호출해 strategy_logits 를 추출한다.
           (generate step에서는 strategy_logits 를 얻을 수 없기 때문)
        """
        if not hasattr(self, "strategy_logits_cache"):
            self.strategy_logits_cache, self.sid_cache = [], []

        # ---------------------------------------------
        # 1) strategy_id 분리 (metric 계산용) & 안전 복사
        # ---------------------------------------------
        inputs = inputs.copy()  # 손상 방지
        sid = inputs.pop("strategy_id")  # Tensor shape (B,)

        # ---------------------------------------------
        # 2) 기존 Seq2SeqTrainer prediction_step 실행
        #    loss, generated_tokens, labels 획득
        # ---------------------------------------------
        loss, generated_tokens, labels = super().prediction_step(
            model, inputs, prediction_loss_only=False, ignore_keys=ignore_keys
        )

        # ---------------------------------------------
        # 3) strategy_logits 추출 (no_grad 로 forward)
        # ---------------------------------------------
        with torch.no_grad():
            out = model(**inputs)
            if hasattr(out, "strategy_logits"):
                self.strategy_logits_cache.append(out.strategy_logits.cpu().numpy())
            else:
                # fallback: classifier 결과 직접 계산
                enc_cls = out.encoder_last_hidden_state[:, 0]
                logits = model.classifier(enc_cls).cpu().numpy()
                self.strategy_logits_cache.append(logits)

        self.sid_cache.append(sid.cpu().numpy())

        # ---------------------------------------------
        # 4) 반환 (generated_tokens 는 None 가능)
        # ---------------------------------------------
        return loss, generated_tokens, labels

    # ------------------------------------------------------------------
    # safetensors(shared-tensor) RuntimeError 회피용 커스텀 저장 함수
    # Trainer.save_model → self._save 를 호출하므로, 여기서 safe_serialization=False 로 저장
    # ------------------------------------------------------------------
    def _save(self, output_dir: str | None = None, state_dict=None):  # type: ignore
        """
        최적화된 저장 로직:
        1. 체크포인트(-NNNN)에는 모델 저장하지 않음 (metrics만 유지)
        2. best_model 디렉토리에 항상 현재까지의 best model 유지
        3. 마지막에는 학습 종료 시 best 모델 복원해서 최종 저장
        """
        import os, logging, torch

        logger = logging.getLogger("joint")
        if output_dir is None:
            output_dir = self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)

        # 체크포인트 판단: checkpoint-NNNN 형태의 경로인지 확인
        is_checkpoint = "checkpoint-" in os.path.basename(output_dir)
        # best model 디렉토리 판단
        is_best_model_dir = output_dir == self.best_model_dir
        
        if is_checkpoint:
            logger.info(f"[JointTrainer] 체크포인트는 metrics만 저장: {output_dir}")
            # 체크포인트에는 metrics만 유지하고 모델은 저장하지 않음
            # state 추적용 json 파일은 Trainer가 별도로 저장
            return

        model_to_save = self.model
        if hasattr(model_to_save, "module"):
            model_to_save = model_to_save.module  # unwrap DDP

        # -----------------------------------------
        # 1) 본체(BART) 저장
        # -----------------------------------------
        if hasattr(model_to_save, "save_pretrained"):
            # JointBart 는 자체 save_pretrained 가 없으므로 BartForConditionalGeneration에 대해 호출
            model_to_save.save_pretrained(output_dir, safe_serialization=False, state_dict=state_dict)
        elif hasattr(model_to_save, "model") and hasattr(model_to_save.model, "save_pretrained"):
            model_to_save.model.save_pretrained(output_dir, safe_serialization=False, state_dict=state_dict)
        else:
            # fallback: state_dict 전체를 일반 torch.save
            torch.save(model_to_save.state_dict(), os.path.join(output_dir, "pytorch_model.bin"))

        # -----------------------------------------
        # 2) 분류 헤드 별도 저장 (state_dict)
        # -----------------------------------------
        try:
            torch.save(model_to_save.classifier.state_dict(), os.path.join(output_dir, "classifier.bin"))
        except Exception:
            pass  # classifier 없을 수도 있음

        # -----------------------------------------
        # 3) tokenizer와 기타 정보 저장
        # -----------------------------------------
        if hasattr(self, "tokenizer") and self.tokenizer is not None:
            try:
                self.tokenizer.save_pretrained(output_dir)
                logger.info(f"[JointTrainer] tokenizer saved to {output_dir}")
            except Exception as e:
                logger.warning(f"tokenizer save failed: {e}")

            # training args 저장
            torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
            logger.info(f"[JointTrainer] training args saved to {output_dir}")

        logger.info(f"[JointTrainer] 모델 저장 완료: {output_dir} {'(Best Model)' if is_best_model_dir else ''}")

    # ------------------------------------------------------------------
    # Best 모델 저장 - compute_metrics에서 새로운 best 모델 발견 시 호출됨
    # ------------------------------------------------------------------
    def _save_best_model(self):
        """현재 모델을 best_model 디렉토리에 저장"""
        import logging
        logger = logging.getLogger("joint")
        
        logger.info(f"Best 모델 저장 시작: {self.best_model_dir}")
        # _save 메소드를 활용해 best_model 디렉토리에 저장
        # checkpoint 경로가 아니므로 모델 가중치가 저장됨
        self._save(output_dir=self.best_model_dir)
        logger.info(f"Best 모델 저장 완료: {self.best_model_dir}")

    # ------------------------------------------------------------------
    # 옵티마이저와 스케줄러를 체크포인트에 저장하지 않도록 오버라이드
    # 이로써 각 체크포인트에서 약 1GB의 공간을 절약할 수 있음
    # ------------------------------------------------------------------
    def _save_optimizer_and_scheduler(self, output_dir: str):
        """
        체크포인트에는 optimizer.pt와 scheduler.pt를 저장하지 않고
        최종 모델에도 기본적으로 저장하지 않음 (저장 실패 시 자동 무시)
        """
        import os, logging
        logger = logging.getLogger("joint")

        # 체크포인트 경로인지 확인
        is_checkpoint = "checkpoint-" in os.path.basename(output_dir)
        
        if is_checkpoint:
            # 체크포인트에는 저장하지 않음 - 공간 절약
            logger.info(f"[JointTrainer] 체크포인트에 optimizer/scheduler 저장 안함 ({output_dir})")
            return
        
        # 최종 모델에는 저장 시도 (원래 로직)
        try:
            super()._save_optimizer_and_scheduler(output_dir)
            logger.info(f"[JointTrainer] optimizer/scheduler saved to {output_dir}")
        except RuntimeError as e:
            logger.warning(f"optimizer/scheduler save failed → 건너뜀: {e}")

# ============================== main ================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch_size", type=int, default=8)
    ap.add_argument("--lambda_cls", type=float, default=0.5)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--output_dir", type=str, default="runs/joint")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tiny_frac", type=float, default=None, help="데이터셋의 일부만 사용 (e.g., 0.01 = 1%)")
    ap.add_argument("--patience", type=int, default=3, help="early stopping patience (number of epochs without improvement)")
    ap.add_argument("--clean_checkpoints", action="store_true", help="학습 완료 후 체크포인트 폴더 정리")
    ap.add_argument("--eval_init", action="store_true", help="학습 전 초기 모델(epoch 0)에서 평가 수행")
    args = ap.parse_args()

    # logging
    os.makedirs(args.output_dir, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.FileHandler(Path(args.output_dir)/"train.log"), logging.StreamHandler()])

    set_seed(args.seed)
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    tokenizer.add_special_tokens({"additional_special_tokens": list(SPECIAL_TOKENS.values()) + STRAT_TOKENS})

    train_ds = JointESConvDataset("train", tokenizer)
    val_ds   = JointESConvDataset("validation", tokenizer)
    test_ds  = JointESConvDataset("test", tokenizer)

    # -------------------------------------------------------------
    # tiny training option: 매우 작은 부분집합으로 epoch 속도 테스트
    # -------------------------------------------------------------
    if args.tiny_frac is not None and 0 < args.tiny_frac < 1:
        import random
        def _subset(ds, frac):
            n = max(1, int(len(ds) * frac))
            idx = random.sample(range(len(ds)), n)
            ds.examples = [ds.examples[i] for i in idx]
            return ds
        train_ds = _subset(train_ds, args.tiny_frac)
        val_ds   = _subset(val_ds, args.tiny_frac)
        test_ds  = _subset(test_ds, args.tiny_frac)  # 테스트셋도 동일하게 적용
        logging.info(f"[tiny] train={len(train_ds)}  val={len(val_ds)}  test={len(test_ds)} examples (fraction={args.tiny_frac})")

    model = JointBart(tokenizer, lambda_cls=args.lambda_cls)

    data_collator = DataCollatorForSeq2Seq(tokenizer, label_pad_token_id=-100, pad_to_multiple_of=8)

    train_args = Seq2SeqTrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,    # 최신 체크포인트 1개만 유지
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        predict_with_generate=True,
        generation_max_length=64,
        report_to="none",
    )

    # checkpoint에 모델이 저장되지 않으므로 각 평가마다 best_model 디렉토리에 저장하기 위한 커스텀 콜백
    class SaveBestModelCallback(TrainerCallback):
        def on_evaluate(self, args, state, control, metrics=None, **kwargs):
            if metrics is None:
                return
            
            metric_to_check = args.metric_for_best_model
            if not metric_to_check:
                return
                
            metric_value = metrics.get(metric_to_check)
            if metric_value is None:
                return
                
            # trainer 객체 가져오기
            trainer = kwargs.get("trainer", None)
            if trainer is None:
                return
                
            # best_model_dir이 정의되어 있지 않다면 생성
            if not hasattr(trainer, "best_model_dir"):
                trainer.best_model_dir = os.path.join(args.output_dir, "best_model")
                
            # best_metric 초기화
            if not hasattr(trainer, "best_metric"):
                trainer.best_metric = float("inf") if not args.greater_is_better else float("-inf")
                
            operator = np.greater if args.greater_is_better else np.less
            
            if operator(trainer.best_metric, metric_value):
                # 메트릭이 더 나빠졌으면 저장하지 않음
                return
                
            # 저장 로직
            logger.info(f"새로운 Best 모델 발견 ({metric_to_check}={metric_value}), 저장합니다.")
            trainer.best_metric = metric_value
            trainer.save_model(trainer.best_model_dir)
            
    # 콜백 목록 생성
    callbacks = [
        EarlyStoppingCallback(early_stopping_patience=args.patience),
        SaveBestModelCallback(),
    ]
    
    trainer = JointTrainer(
        model=model,
        args=train_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
        callbacks=callbacks,
    )

    # --------------------------- 초기 모델 (epoch 0) 평가 ---------------------------
    if args.eval_init:
        from sklearn.metrics import accuracy_score, f1_score, classification_report 
        
        logging.info("초기 모델(epoch 0) 평가 시작...")
        
        # -------------------------------------------------------------
        # (1) 기본 loss 등               : trainer.evaluate
        # (2) 생성 & 전략 메트릭 계산     : 직접 predict 후 계산
        # -------------------------------------------------------------

        # 1) loss 등 기본 메트릭
        init_val_metrics = trainer.evaluate(val_ds, metric_key_prefix="init_val")
        logging.info(f"초기 모델 validation 평가 결과: loss={init_val_metrics['init_val_loss']:.4f}")
        
        # validation PPL 계산
        if "init_val_loss" in init_val_metrics:
            init_val_metrics["init_val_perplexity"] = calculate_perplexity(init_val_metrics["init_val_loss"])
            logging.info(f"초기 모델 Validation Perplexity: {init_val_metrics['init_val_perplexity']:.4f}")
        
        # 테스트셋 기본 메트릭
        init_test_metrics = trainer.evaluate(test_ds, metric_key_prefix="init_test")
        
        # 테스트 PPL 계산
        if "init_test_loss" in init_test_metrics:
            init_test_metrics["init_test_perplexity"] = calculate_perplexity(init_test_metrics["init_test_loss"])
            logging.info(f"초기 모델 Test Perplexity: {init_test_metrics['init_test_perplexity']:.4f}")
        
        # 2) 추가 메트릭 계산
        logging.info("초기 모델 테스트셋 생성 & 전략 메트릭 계산 중...")
        preds_init = trainer.predict(test_ds, metric_key_prefix="init_test")
        
        # generation metrics
        gen_texts_init = safe_batch_decode(tokenizer, preds_init.predictions)
        # label_ids → refs
        lbl_ids = preds_init.label_ids.copy()
        lbl_ids[lbl_ids == -100] = tokenizer.pad_token_id
        refs_init = safe_batch_decode(tokenizer, lbl_ids)
        gen_metrics_init = generation_metrics(gen_texts_init, refs_init)
        
        # 생성 메트릭 결과 출력
        logging.info(f"초기 모델 생성 성능: BLEU={gen_metrics_init.get('bleu', 0):.4f}, ROUGE-L={gen_metrics_init.get('rouge_l', 0):.4f}")
        
        # strategy metrics (accuracy / weighted f1)
        logging.info("초기 모델 전략 분류 메트릭 계산 중...")
        strat_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=args.batch_size,
            collate_fn=data_collator,
            shuffle=False,
        )
        all_sid_pred_init, all_sid_gt_init = [], []
        device_eval = next(model.parameters()).device
        with torch.no_grad():
            for batch in strat_loader:
                sid_gt = batch.pop("strategy_id")
                input_ids = batch["input_ids"].to(device_eval)
                attention_mask = batch["attention_mask"].to(device_eval)
                outs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outs.strategy_logits
                sid_pred = torch.argmax(logits, dim=-1).cpu()
                all_sid_pred_init.append(sid_pred)
                all_sid_gt_init.append(sid_gt)
        
        all_sid_pred_init = torch.cat(all_sid_pred_init).numpy()
        all_sid_gt_init = torch.cat(all_sid_gt_init).numpy()
        strat_acc_init = accuracy_score(all_sid_gt_init, all_sid_pred_init)
        strat_f1_init = f1_score(all_sid_gt_init, all_sid_pred_init, average="weighted")
        
        # 전략 분류 결과 출력
        logging.info(f"초기 모델 전략 분류 성능: Accuracy={strat_acc_init:.4f}, F1={strat_f1_init:.4f}")
        
        # 전략별 상세 성능 보고서
        from utils.strategy import STRATEGIES
        init_strat_report = classification_report(
            all_sid_gt_init, all_sid_pred_init,
            labels=list(range(len(STRATEGIES))),
            target_names=STRATEGIES,
            digits=2,
            zero_division=0
        )
        logging.info("\n=== 초기 모델 전략 분류 상세 보고서 ===\n" + init_strat_report)
        
        # 메트릭 통합
        init_test_metrics.update({f"init_test_{k}": float(v) for k, v in gen_metrics_init.items()})
        init_test_metrics.update({
            "init_test_strategy_accuracy": float(strat_acc_init),
            "init_test_strategy_f1": float(strat_f1_init),
        })
        
        # 초기 테스트 메트릭 저장
        Path(args.output_dir).mkdir(exist_ok=True, parents=True)
        
        # validation 메트릭 저장
        init_val_path = Path(args.output_dir) / "init_val_metrics.json"
        with init_val_path.open("w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in init_val_metrics.items()}, f, indent=2)
        
        # test 메트릭 저장
        init_test_path = Path(args.output_dir) / "init_test_metrics.json"
        with init_test_path.open("w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in init_test_metrics.items()}, f, indent=2)
        
        # 샘플 저장
        sample_n = min(10, len(gen_texts_init))
        with open(Path(args.output_dir) / "init_samples.txt", "w", encoding="utf-8") as f:
            for ref, gen in zip(refs_init[:sample_n], gen_texts_init[:sample_n]):
                f.write(f"REF: {ref}\nGEN: {gen}\n---\n")
        
        logging.info(f"초기 모델 평가 결과 저장 완료: {args.output_dir}")

    # --------------------------- train + save best ---------------------------------
    trainer.train()

    if trainer.is_world_process_zero():
        # 학습 종료 후 현재 모델을 best_model 디렉토리에 강제 저장
        best_model_dir = os.path.join(args.output_dir, "best_model")
        logging.info(f"학습 완료: 현재 모델을 {best_model_dir}에 강제 저장합니다.")
        trainer.save_model(best_model_dir)
        
        # validation metrics dump
        metrics = trainer.evaluate()
        
        # 손실값에서 PPL 계산 추가
        if "eval_loss" in metrics:
            metrics["eval_perplexity"] = calculate_perplexity(metrics["eval_loss"])
            logging.info(f"Validation Perplexity: {metrics['eval_perplexity']:.4f}")
            
        metrics_path = Path(args.output_dir) / "val_metrics.json"
        with metrics_path.open("w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)

        # 전략 분류 리포트 출력 - validation 데이터셋
        logging.info("[main] computing strategy classification report for validation set...")
        model.eval()
        val_loader = torch.utils.data.DataLoader(
            val_ds,  # 이미 존재하는 validation 데이터셋 사용
            batch_size=args.batch_size,
            collate_fn=data_collator,
            shuffle=False,
        )
        val_sid_pred, val_sid_gt = [], []
        device_eval = next(model.parameters()).device
        with torch.no_grad():
            for batch in val_loader:
                sid_gt = batch.pop("strategy_id")
                input_ids = batch["input_ids"].to(device_eval)
                attention_mask = batch["attention_mask"].to(device_eval)
                outs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outs.strategy_logits
                sid_pred = torch.argmax(logits, dim=-1).cpu()
                val_sid_pred.append(sid_pred)
                val_sid_gt.append(sid_gt)

        val_sid_pred = torch.cat(val_sid_pred).numpy()
        val_sid_gt = torch.cat(val_sid_gt).numpy()

        # 분류 리포트 출력
        val_report = classification_report(
            val_sid_gt, val_sid_pred,
            labels=list(range(len(STRATEGIES))),
            target_names=STRATEGIES,
            digits=2,
            zero_division=0
        )
        logging.info("\n=== Validation Strategy Classification Report ===\n" + val_report)

        # -------------------- TEST DATASET EVALUATION --------------------
        logging.info("[main] evaluating on test split …")
        test_ds = JointESConvDataset("test", tokenizer)

        # tiny training 모드라면 test 세트도 동일 비율 서브샘플
        if args.tiny_frac is not None and 0 < args.tiny_frac < 1:
            import random
            n_test = max(1, int(len(test_ds) * args.tiny_frac))
            idx = random.sample(range(len(test_ds)), n_test)
            test_ds.examples = [test_ds.examples[i] for i in idx]
            logging.info(f"[tiny] test={len(test_ds)} examples (fraction={args.tiny_frac})")

        # -------------------------------------------------------------
        # (1) 기본 loss 등               : trainer.evaluate
        # (2) 생성 & 전략 메트릭 계산     : 직접 predict 후 계산
        # -------------------------------------------------------------

        # 1) loss 등 기본 메트릭
        test_metrics = trainer.evaluate(test_ds, metric_key_prefix="test")

        # 손실값에서 PPL 계산 추가
        if "test_loss" in test_metrics:
            test_metrics["test_perplexity"] = calculate_perplexity(test_metrics["test_loss"])
            logging.info(f"Test Perplexity: {test_metrics['test_perplexity']:.4f}")

        # 2) 추가 메트릭 계산
        logging.info("[main] computing generation & strategy metrics on test …")
        preds_full = trainer.predict(test_ds)

        # generation metrics
        gen_texts_full = safe_batch_decode(tokenizer, preds_full.predictions)
        # label_ids → refs
        lbl_ids = preds_full.label_ids.copy()
        lbl_ids[lbl_ids == -100] = tokenizer.pad_token_id
        refs_full = safe_batch_decode(tokenizer, lbl_ids)
        gen_metrics = generation_metrics(gen_texts_full, refs_full)

        # strategy metrics (accuracy / weighted f1)
        from sklearn.metrics import accuracy_score, f1_score
        logging.info("[main] computing strategy metrics …")
        model.eval()
        strat_loader = torch.utils.data.DataLoader(
            test_ds,
            batch_size=args.batch_size,
            collate_fn=data_collator,
            shuffle=False,
        )
        all_sid_pred, all_sid_gt = [], []
        device_eval = next(model.parameters()).device
        with torch.no_grad():
            for batch in strat_loader:
                sid_gt = batch.pop("strategy_id")
                input_ids = batch["input_ids"].to(device_eval)
                attention_mask = batch["attention_mask"].to(device_eval)
                outs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outs.strategy_logits
                sid_pred = torch.argmax(logits, dim=-1).cpu()
                all_sid_pred.append(sid_pred)
                all_sid_gt.append(sid_gt)

        all_sid_pred = torch.cat(all_sid_pred).numpy()
        all_sid_gt = torch.cat(all_sid_gt).numpy()
        strat_acc = accuracy_score(all_sid_gt, all_sid_pred)
        strat_f1 = f1_score(all_sid_gt, all_sid_pred, average="weighted")

        # 메트릭 통합
        test_metrics.update({f"test_{k}": float(v) for k, v in gen_metrics.items()})
        test_metrics.update({
            "test_strategy_accuracy": float(strat_acc),
            "test_strategy_f1": float(strat_f1),
        })

        # 저장
        test_path = Path(args.output_dir) / "test_metrics.json"
        with test_path.open("w", encoding="utf-8") as f:
            json.dump({k: float(v) for k, v in test_metrics.items()}, f, indent=2)

        # 분류 리포트 출력
        test_report = classification_report(
            all_sid_gt, all_sid_pred,
            labels=list(range(len(STRATEGIES))),
            target_names=STRATEGIES,
            digits=2,
            zero_division=0
        )
        logging.info("\n=== Test Strategy Classification Report ===\n" + test_report)

        # 샘플 10개 저장 (validation 샘플 파일은 생성하지 않음)
        sample_n_test = min(10, len(gen_texts_full))
        sample_path = Path(args.output_dir) / "samples.txt"
        with sample_path.open("w", encoding="utf-8") as f:
            for ref, gen in zip(refs_full[:sample_n_test], gen_texts_full[:sample_n_test]):
                f.write(f"REF: {ref}\nGEN: {gen}\n---\n")
        logging.info(f"saved best model + metrics and samples to {args.output_dir}")

    # ----------------------------- cleanup ----------------------------
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()
    # ------------------------------------------------------------------------------

# ------------------------ helper: safe decode ------------------------
def safe_batch_decode(tokenizer: BartTokenizer, predictions):
    """토크나이저 오류(NoneType) 방지를 위해 id 범위를 검사하며 디코드."""
    texts: list[str] = []
    vocab_size = len(tokenizer)
    
    for seq in predictions:
        try:
            # seq 가 numpy array / list / torch 텐서 모두 지원
            if isinstance(seq, torch.Tensor):
                seq = seq.tolist()
            elif isinstance(seq, np.ndarray):
                seq = seq.tolist()
                
            # id 범위 밖 값, None, 또는 비정수값을 unk 토큰으로 대체
            clean_ids = []
            for t in seq:
                if isinstance(t, (int, np.integer)) and 0 <= int(t) < vocab_size:
                    clean_ids.append(int(t))
                else:
                    clean_ids.append(tokenizer.unk_token_id)
                    
            text = tokenizer.decode(clean_ids, skip_special_tokens=True)
            texts.append(text)
            
        except Exception as e:
            # 디코딩 실패 시 빈 문자열 추가
            texts.append("")
            logging.warning(f"토큰 디코딩 실패: {e}")
            
    return texts

if __name__ == "__main__":
    main() 