"""
train_sasrec_strategy
─────────────────────
• ESConv 전략 시퀀스 → SASRec 학습/평가
• 주요 메트릭: accuracy, top-k accuracy
"""

from __future__ import annotations
import argparse, random, json, logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, top_k_accuracy_score, classification_report
)

from utils.strategy import STRATEGIES                                   # 8개 전략
from src.data.esconv_strategy_dataset import (
    ESConvStrategySequenceDataset, PAD_ID, N_ITEMS, NO_HISTORY_ID)
from src.models.sasrec_strategy import SASRecForStrategy

# 로거 설정
logger = logging.getLogger(__name__)

# reproducibility ------------------------------------------------------
def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# metric helpers -------------------------------------------------------
def batch_accuracy(pred_logits, targets):
    preds = pred_logits.argmax(dim=1).cpu().numpy()
    return accuracy_score(targets.cpu().numpy(), preds)


def batch_topk(pred_logits, targets, k=3):
    # 패딩 ID에 해당하는 열은 제외하고 계산 (실제 전략 분류만 고려)
    return top_k_accuracy_score(targets.cpu().numpy(),
                                pred_logits[:, :PAD_ID].cpu().numpy(),  # 패딩 ID 열 제외
                                k=k, labels=list(range(len(STRATEGIES))))


def batch_f1_weighted(pred_logits, targets):
    preds = pred_logits.argmax(dim=1).cpu().numpy()
    return f1_score(targets.cpu().numpy(), preds, average="weighted", zero_division=0)


# main -----------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_seq_len", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", type=str, default="outputs/sasrec")
    parser.add_argument("--tiny_frac", type=float, default=None)
    parser.add_argument("--patience", type=int, default=5,
                    help="early-stopping patience (#epochs without val improvement)")
    parser.add_argument("--n_heads", type=int, default=2)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--weight_exp", type=float, default=0.5,
                    help="클래스 가중치 계산 지수 (낮을수록 희소 클래스 강조)")
    parser.add_argument("--dataset", choices=["esconv", "multiesc"],
                        default="esconv",
                        help="사용할 데이터셋 종류")
    args = parser.parse_args()

    # 로깅 설정
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ───────── dataset & dataloader ─────────
    if args.dataset == "esconv":
        from src.data.esconv_strategy_dataset import ESConvStrategySequenceDataset as DS
    else:
        from src.data.multiesc_strategy_dataset import MultiESCStrategySequenceDataset as DS

    # 데이터셋별 상수 바인딩
    try:
        # 데이터셋 클래스에서 상수를 가져오려고 시도
        STRATEGIES = DS.STRATEGIES            # type: ignore
        PAD_ID     = DS.PAD_ID                # type: ignore
        N_ITEMS    = DS.N_ITEMS               # type: ignore
        NO_HISTORY_ID = DS.NO_HISTORY_ID      # type: ignore
    except AttributeError:
        # 속성이 없으면 utils.strategy에서 가져옴
        from utils.strategy import STRATEGIES
        from src.data.esconv_strategy_dataset import PAD_ID, N_ITEMS, NO_HISTORY_ID
        logger.warning(f"DS.STRATEGIES not found, using default STRATEGIES from utils.strategy")

    train_ds = DS("train", args.max_seq_len, tiny_frac=args.tiny_frac)
    val_ds   = DS("validation", args.max_seq_len, tiny_frac=args.tiny_frac)
    test_ds  = DS("test", args.max_seq_len, tiny_frac=args.tiny_frac)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=4)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False, num_workers=4)

    # ───────── model ─────────
    model = SASRecForStrategy(
        n_items=N_ITEMS,
        hidden_size=args.hidden_size,
        max_seq_len=args.max_seq_len,
        pad_id=PAD_ID,
        no_hist_id=NO_HISTORY_ID,
        n_heads=args.n_heads,
        n_layers=args.n_layers,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    # (기존 optimizer 정의 아래)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    # ---------- class weight 계산 ----------
    from collections import Counter
    def compute_class_weight(dataset, exp=0.5):
        counter = Counter([t for _, t in dataset.samples])
        total = sum(counter.values())
        weight = torch.tensor([
            0.0 if i in {PAD_ID, NO_HISTORY_ID} else 1.0/((counter[i]/total)**exp + 1e-9)
            for i in range(N_ITEMS)
        ])
        weight = weight / weight.mean()
        return weight.to(device)

    class_weight = compute_class_weight(train_ds, args.weight_exp)
    criterion = torch.nn.CrossEntropyLoss(weight=class_weight)

    # ───────── training loop ─────────
    best_val_acc = 0.0
    epochs_no_improve = 0                         # ← early-stopping counter
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss, total_acc = 0.0, 0.0
        for batch in train_loader:
            seq = batch["seq"].to(device)
            tgt = batch["target"].to(device)

            logits = model(seq)
            loss = criterion(logits, tgt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * seq.size(0)
            total_acc  += batch_accuracy(logits, tgt) * seq.size(0)

        avg_loss = total_loss / len(train_ds)
        avg_acc  = total_acc / len(train_ds)

        # ───── validation ─────
        model.eval()
        val_acc, val_top3, val_f1_w = 0.0, 0.0, 0.0
        y_true_val, y_pred_val = [], []

        with torch.no_grad():
            for batch in val_loader:
                seq, tgt = batch["seq"].to(device), batch["target"].to(device)
                logits = model(seq)

                val_acc  += batch_accuracy(logits, tgt) * seq.size(0)
                val_top3 += batch_topk(logits, tgt, k=3) * seq.size(0)

                y_true_val.extend(tgt.cpu().tolist())
                y_pred_val.extend(logits.argmax(dim=1).cpu().tolist())

        val_acc  /= len(val_ds)
        val_top3 /= len(val_ds)
        val_f1_w  = f1_score(y_true_val, y_pred_val, average="weighted", zero_division=0)

        print(f"[Epoch {epoch:02d}] loss={avg_loss:.4f} acc={avg_acc:.4f} "
              f"val_acc={val_acc:.4f} val_f1_w={val_f1_w:.4f} val_top3={val_top3:.4f}")

        # ───── 모델 저장 ─────
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), Path(args.output_dir) / "best_model.pt")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= args.patience:
                print(f"no improvement for {args.patience} epochs → early stop")
                break

        # val_acc 기반으로 스케줄러 호출 후
        sched.step(val_acc)

        # 현재 학습률 출력 (ReduceLROnPlateau는 get_last_lr()가 없음)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

    # ───────── test ─────────
    best_path = Path(args.output_dir) / "best_model.pt"
    state = torch.load(best_path, weights_only=True, map_location=device)
    model.load_state_dict(state)
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for batch in test_loader:
            seq = batch["seq"].to(device)
            tgt = batch["target"].to(device)
            logits = model(seq)

            y_true.extend(tgt.cpu().tolist())
            y_pred.extend(logits.argmax(dim=1).cpu().tolist())

    test_acc = accuracy_score(y_true, y_pred)
    test_f1_w = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\nTest accuracy={test_acc:.3f}  weighted_f1={test_f1_w:.3f}")
    print("\n=== Test Classification Report ===")
    report = classification_report(
        y_true, y_pred,
        labels=list(range(len(STRATEGIES))),
        target_names=STRATEGIES,
        digits=3, zero_division=0
    )
    print(report)

    # JSON 저장 시 추가 필드 포함
    with open(Path(args.output_dir) / "test_report.json", "w") as f:
        json.dump({
            "accuracy": test_acc,
            "weighted_f1": test_f1_w,
            "classification_report": classification_report(
                y_true, y_pred, output_dict=True,
                labels=list(range(len(STRATEGIES))),
                target_names=STRATEGIES,
                zero_division=0)
        }, f, indent=2)


if __name__ == "__main__":
    main()
