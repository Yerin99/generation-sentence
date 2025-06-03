#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
OOV 분석 스크립트
=================
ESConv 및 MultiESC 데이터셋의 OOV 비율을 분석합니다.

사용법:
    python analyze_oov.py --dataset [esconv|multiesc] --split [train|validation|test]
"""
import argparse
import json
import os
from pathlib import Path
from tqdm import tqdm

from datasets import load_dataset
from transformers import BartTokenizer

def count_oov_tokens(text, tokenizer):
    """
    텍스트를 토큰화하고 OOV 토큰 수 계산
    """
    encoded = tokenizer.encode(text, add_special_tokens=False)
    
    # OOV 토큰은 unk_token_id로 인코딩됨
    oov_count = sum(1 for token_id in encoded if token_id == tokenizer.unk_token_id)
    total_count = len(encoded)
    
    return {
        "oov_count": oov_count,
        "total_tokens": total_count,
        "oov_ratio": oov_count / total_count if total_count > 0 else 0
    }

def analyze_esconv(split, tokenizer):
    """
    HuggingFace ESConv 데이터셋 분석
    """
    print(f"ESConv 데이터셋 {split} 분할 분석 중...")
    dataset = load_dataset("thu-coai/esconv", split=split)
    
    stats = {
        "total_oov": 0,
        "total_tokens": 0,
        "samples_with_oov": 0,
        "total_samples": len(dataset),
        "utterance_stats": {
            "usr": {"total_oov": 0, "total_tokens": 0},
            "sys": {"total_oov": 0, "total_tokens": 0}
        }
    }
    
    for example in tqdm(dataset):
        dialog = json.loads(example["text"])["dialog"]
        sample_has_oov = False
        
        for turn in dialog:
            speaker = turn["speaker"]
            text = turn["text"]
            turn_stats = count_oov_tokens(text, tokenizer)
            
            # 화자별 통계 업데이트
            stats["utterance_stats"][speaker]["total_oov"] += turn_stats["oov_count"]
            stats["utterance_stats"][speaker]["total_tokens"] += turn_stats["total_tokens"]
            
            # 전체 통계 업데이트
            stats["total_oov"] += turn_stats["oov_count"]
            stats["total_tokens"] += turn_stats["total_tokens"]
            
            if turn_stats["oov_count"] > 0:
                sample_has_oov = True
                
        if sample_has_oov:
            stats["samples_with_oov"] += 1
    
    # 비율 계산
    stats["overall_oov_ratio"] = stats["total_oov"] / stats["total_tokens"] if stats["total_tokens"] > 0 else 0
    stats["samples_with_oov_ratio"] = stats["samples_with_oov"] / stats["total_samples"]
    
    for speaker in ["usr", "sys"]:
        speaker_tokens = stats["utterance_stats"][speaker]["total_tokens"]
        stats["utterance_stats"][speaker]["oov_ratio"] = (
            stats["utterance_stats"][speaker]["total_oov"] / speaker_tokens if speaker_tokens > 0 else 0
        )
    
    return stats

def analyze_multiesc(split, tokenizer):
    """
    MultiESC 데이터셋 분석
    """
    print(f"MultiESC 데이터셋 {split} 분할 분석 중...")
    file_path = f"MultiESC/data/{split}.txt"
    
    if not os.path.exists(file_path):
        print(f"경고: {file_path} 파일이 존재하지 않습니다.")
        return None
    
    # JSONL 형식으로 파일 읽기
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:  # 빈 줄은 건너뜀
                continue
                
            try:
                # 각 줄을 개별 JSON 객체로 파싱
                example = json.loads(line)
                data.append(example)
            except json.JSONDecodeError as e:
                # 파싱 오류 발생 시 상세 정보 출력 후 중단
                print(f"심각한 오류: {file_path} 파일의 {line_num}번째 줄에서 JSON 파싱 실패")
                print(f"오류 내용: {e}")
                print(f"문제의 줄: {line[:100]}..." if len(line) > 100 else f"문제의 줄: {line}")
                print("데이터셋 파일을 확인하고 수정 후 다시 시도하세요.")
                return None
    
    stats = {
        "total_oov": 0,
        "total_tokens": 0,
        "samples_with_oov": 0,
        "total_samples": len(data),
        "utterance_stats": {
            "usr": {"total_oov": 0, "total_tokens": 0},
            "sys": {"total_oov": 0, "total_tokens": 0}
        }
    }
    
    for example in tqdm(data):
        if "dialog" not in example:
            continue
            
        dialog = example["dialog"]
        sample_has_oov = False
        
        for turn in dialog:
            if "text" not in turn or "speaker" not in turn:
                continue
                
            speaker = turn["speaker"]
            text = turn["text"]
            
            # speaker가 usr나 sys가 아니면 건너뜀
            if speaker not in ["usr", "sys"]:
                continue
                
            turn_stats = count_oov_tokens(text, tokenizer)
            
            # 화자별 통계 업데이트
            stats["utterance_stats"][speaker]["total_oov"] += turn_stats["oov_count"]
            stats["utterance_stats"][speaker]["total_tokens"] += turn_stats["total_tokens"]
            
            # 전체 통계 업데이트
            stats["total_oov"] += turn_stats["oov_count"]
            stats["total_tokens"] += turn_stats["total_tokens"]
            
            if turn_stats["oov_count"] > 0:
                sample_has_oov = True
                
        if sample_has_oov:
            stats["samples_with_oov"] += 1
    
    # 비율 계산
    stats["overall_oov_ratio"] = stats["total_oov"] / stats["total_tokens"] if stats["total_tokens"] > 0 else 0
    stats["samples_with_oov_ratio"] = stats["samples_with_oov"] / stats["total_samples"]
    
    for speaker in ["usr", "sys"]:
        speaker_tokens = stats["utterance_stats"][speaker]["total_tokens"]
        stats["utterance_stats"][speaker]["oov_ratio"] = (
            stats["utterance_stats"][speaker]["total_oov"] / speaker_tokens if speaker_tokens > 0 else 0
        )
    
    return stats

def print_stats(stats, dataset_name, split):
    """
    통계 출력
    """
    print(f"\n{'='*50}")
    print(f"{dataset_name} 데이터셋 ({split}) OOV 분석 결과:")
    print(f"{'='*50}")
    print(f"전체 샘플 수: {stats['total_samples']}")
    print(f"OOV가 포함된 샘플 수: {stats['samples_with_oov']} ({stats['samples_with_oov_ratio']*100:.2f}%)")
    print(f"전체 토큰 수: {stats['total_tokens']}")
    print(f"전체 OOV 토큰 수: {stats['total_oov']}")
    print(f"전체 OOV 비율: {stats['overall_oov_ratio']*100:.4f}%")
    print("\n화자별 OOV 비율:")
    for speaker in ["usr", "sys"]:
        speaker_stats = stats["utterance_stats"][speaker]
        print(f"  - {speaker}: {speaker_stats['oov_ratio']*100:.4f}% (OOV: {speaker_stats['total_oov']}, 총 토큰: {speaker_stats['total_tokens']})")
    print(f"{'='*50}\n")

def main():
    parser = argparse.ArgumentParser(description="BART 모델의 ESConv/MultiESC 데이터셋 OOV 분석")
    parser.add_argument("--dataset", choices=["esconv", "multiesc", "both"], default="both", help="분석할 데이터셋")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="all", help="분석할 데이터 분할")
    parser.add_argument("--output", type=str, default="oov_analysis.json", help="결과를 저장할 JSON 파일 경로")
    args = parser.parse_args()
    
    # BART 토크나이저 로드
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-base")
    
    datasets = ["esconv", "multiesc"] if args.dataset == "both" else [args.dataset]
    splits = ["train", "validation", "test"] if args.split == "all" else [args.split]
    
    all_results = {}
    
    for dataset in datasets:
        all_results[dataset] = {}
        for split in splits:
            if dataset == "esconv":
                stats = analyze_esconv(split, tokenizer)
            else:  # multiesc
                stats = analyze_multiesc(split, tokenizer)
                
            if stats:
                print_stats(stats, dataset.upper(), split)
                all_results[dataset][split] = stats
    
    # 결과 저장
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2)
    print(f"분석 결과가 {args.output}에 저장되었습니다.")

if __name__ == "__main__":
    main() 