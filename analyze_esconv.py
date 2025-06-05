#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ESConv 데이터셋의 context와 response 길이를 분석하는 스크립트
bart_dialog_generator.py의 데이터셋 처리 방식과 동일하게 구현
"""

from datasets import load_dataset
import numpy as np
import json
from transformers import BartTokenizer

def main():
    print("ESConv 데이터셋 로드 중...")
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    # 모든 스플릿 분석
    splits = ["train", "validation", "test"]
    
    for split in splits:
        print(f"\n--- {split} 데이터셋 분석 중... ---")
        esconv = load_dataset("thu-coai/esconv", split=split)
        
        contexts = []
        responses = []
        
        # 전체 데이터셋 분석
        for ex in esconv:
            dialog = json.loads(ex["text"])["dialog"]
            
            for turn_idx, turn in enumerate(dialog):
                if turn["speaker"] == "sys" and turn_idx > 0:
                    # ESConv 데이터셋의 dialog 구조에 맞게 context 추출
                    # 현재 턴 이전까지 모든 대화를 context로 사용
                    context = " ".join([prev["text"] for prev in dialog[:turn_idx]])
                    response = turn["text"]
                    
                    # 토큰화 (bart_dialog_generator.py와 같은 방식)
                    ctx_tokens = tokenizer.encode(context)
                    resp_tokens = tokenizer.encode(response)
                    
                    contexts.append(len(ctx_tokens))
                    responses.append(len(resp_tokens))
        
        print(f"데이터 수: {len(contexts)}")
        print(f"Context 토큰 길이 (최대/평균/90퍼센타일): {np.max(contexts)}/{np.mean(contexts):.1f}/{np.percentile(contexts, 90):.1f}")
        print(f"Response 토큰 길이 (최대/평균/90퍼센타일): {np.max(responses)}/{np.mean(responses):.1f}/{np.percentile(responses, 90):.1f}")
        
        # 길이 히스토그램 - 더 세분화
        bins = [0, 128, 256, 384, 512, 640, 768, 896, 1024, 2048, 4096]
        ctx_hist, _ = np.histogram(contexts, bins=bins)
        resp_hist, _ = np.histogram(responses, bins=bins)
        
        print("\n길이별 분포 (누적 %):")
        cum_ctx = 0
        cum_resp = 0
        for i in range(len(bins)-1):
            cum_ctx += ctx_hist[i]
            cum_resp += resp_hist[i]
            ctx_pct = cum_ctx / len(contexts) * 100
            resp_pct = cum_resp / len(responses) * 100
            
            print(f"{bins[i]}-{bins[i+1]} 토큰: context {ctx_hist[i]}개 ({ctx_pct:.1f}% 누적), "
                  f"response {resp_hist[i]}개 ({resp_pct:.1f}% 누적)")
        
        # 상세 분석 (512, 768, 896에서 얼마나 잘리는지)
        key_lengths = [512, 768, 896, 1024]
        for length in key_lengths:
            ctx_cut = sum(1 for x in contexts if x > length)
            resp_cut = sum(1 for x in responses if x > length)
            ctx_pct = ctx_cut / len(contexts) * 100
            resp_pct = resp_cut / len(responses) * 100
            
            print(f"\n길이 {length}에서 잘리는 비율: context {ctx_pct:.2f}%, response {resp_pct:.2f}%")

if __name__ == "__main__":
    main() 