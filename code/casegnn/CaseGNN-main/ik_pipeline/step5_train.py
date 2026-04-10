"""
Step 5 – Train CaseGNN on ik data and evaluate on the ik test set.

Usage:
  cd ik_pipeline/
  python step5_train.py [--epochs 600] [--batch_size 16] [--lr 5e-5] ...
"""
import argparse
import json
import logging
import os
import sys
import time

import torch
from dgl.dataloading import GraphDataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# Add parent dir for model import
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import CaseGNN, early_stopping

from data_load import SyntheticDataset, PoolDataset, collate
from ik_train import forward
from config import (
    GRAPH_DIR, LABEL_DIR, EXPERIMENT_DIR,
    IN_DIM, H_DIM, OUT_DIM, DROPOUT, NUM_HEAD,
    EPOCHS, LR, WD, BATCH_SIZE, TEMP,
    RAN_NEG_NUM, HARD_NEG, HARD_NEG_NUM, EARLY_STOP_PATIENCE,
    ensure_dirs
)

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:
    SummaryWriter = None

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(message)s")


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dim", type=int, default=IN_DIM)
    p.add_argument("--h_dim", type=int, default=H_DIM)
    p.add_argument("--out_dim", type=int, default=OUT_DIM)
    p.add_argument("--dropout", type=float, default=DROPOUT)
    p.add_argument("--num_head", type=int, default=NUM_HEAD)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--lr", type=float, default=LR)
    p.add_argument("--wd", type=float, default=WD)
    p.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    p.add_argument("--temp", type=float, default=TEMP)
    p.add_argument("--hard_neg", action="store_true", default=HARD_NEG)
    p.add_argument("--hard_neg_num", type=int, default=HARD_NEG_NUM)
    return p.parse_args()


def main():
    args = parse_args()
    ensure_dirs()
    logging.info(args)

    # ── Log dir ──────────────────────────────────────────────────────────
    log_dir = os.path.join(
        EXPERIMENT_DIR,
        f"ik_bs{args.batch_size}_dp{args.dropout}_lr{args.lr}_wd{args.wd}"
        f"_t{args.temp}_hn{args.hard_neg_num}_{time.strftime('%m%d-%H%M%S')}"
    )
    os.makedirs(log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir) if SummaryWriter else None

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info(f"Device: {device}")

    # ── Model ────────────────────────────────────────────────────────────
    model = CaseGNN(args.in_dim, args.h_dim, args.out_dim,
                    dropout=args.dropout, num_head=args.num_head)
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    # ── Datasets ─────────────────────────────────────────────────────────
    gdir = GRAPH_DIR

    # Train
    train_dataset = SyntheticDataset(os.path.join(gdir, "bidirec_train_fact_Synthetic.bin"))
    train_sampler = SubsetRandomSampler(torch.arange(len(train_dataset)))
    train_loader = GraphDataLoader(train_dataset, sampler=train_sampler,
                                   batch_size=args.batch_size, drop_last=False,
                                   collate_fn=collate)

    train_fact_pool = PoolDataset(os.path.join(gdir, "bidirec_train_fact.bin"))
    train_issue_pool = PoolDataset(os.path.join(gdir, "bidirec_train_issue.bin"))

    # Test
    test_dataset = SyntheticDataset(os.path.join(gdir, "bidirec_test_fact_Synthetic.bin"))
    inference_bs = len(test_dataset)  # encode all test in one batch
    test_sampler = SubsetRandomSampler(torch.arange(len(test_dataset)))
    test_loader = GraphDataLoader(test_dataset, sampler=test_sampler,
                                  batch_size=inference_bs, drop_last=False,
                                  collate_fn=collate, shuffle=False)

    test_fact_pool = PoolDataset(os.path.join(gdir, "bidirec_test_fact.bin"))
    test_issue_pool = PoolDataset(os.path.join(gdir, "bidirec_test_issue.bin"))

    # ── Labels ───────────────────────────────────────────────────────────
    with open(os.path.join(LABEL_DIR, "train_labels.json")) as f:
        train_labels = json.load(f)
    with open(os.path.join(LABEL_DIR, "test_labels.json")) as f:
        test_labels = json.load(f)

    bm25_hard_neg_dict = {}
    hn_path = os.path.join(LABEL_DIR, "hard_neg_top50_train.json")
    if os.path.exists(hn_path):
        with open(hn_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    bm25_hard_neg_dict.update(json.loads(line))

    # ── Training loop ────────────────────────────────────────────────────
    highest_ndcg = 0
    con_epoch = 0

    for epoch in range(args.epochs):
        print(f"\n{'='*60}")
        print(f"Epoch {epoch + 1}/{args.epochs}")
        print(f"{'='*60}")

        forward(model, device, writer, train_loader,
                train_fact_pool, train_issue_pool, train_labels,
                epoch, args.temp, bm25_hard_neg_dict,
                args.hard_neg, args.hard_neg_num,
                train_flag=True, embedding_saving=False, optimizer=optimizer)

        with torch.no_grad():
            ndcg = forward(model, device, writer, test_loader,
                           test_fact_pool, test_issue_pool, test_labels,
                           epoch, args.temp, bm25_hard_neg_dict,
                           args.hard_neg, args.hard_neg_num,
                           train_flag=False, embedding_saving=False, optimizer=optimizer)

        # Early stopping
        stop = early_stopping(highest_ndcg, ndcg, epoch, con_epoch)
        highest_ndcg = stop[0]
        if stop[1]:
            print(f"\nEarly stopping at epoch {epoch + 1}. Best NDCG@5: {highest_ndcg:.4f}")
            break
        con_epoch = stop[2]

        # Save best model
        if ndcg >= highest_ndcg:
            ckpt_path = os.path.join(log_dir, "best_model.pt")
            torch.save(model.state_dict(), ckpt_path)

    # ── Save final embeddings ────────────────────────────────────────────
    print("\nSaving CaseGNN embeddings …")
    forward(model, device, writer, train_loader,
            train_fact_pool, train_issue_pool, train_labels,
            0, args.temp, bm25_hard_neg_dict,
            args.hard_neg, args.hard_neg_num,
            train_flag=True, embedding_saving=True, optimizer=optimizer)
    forward(model, device, writer, test_loader,
            test_fact_pool, test_issue_pool, test_labels,
            0, args.temp, bm25_hard_neg_dict,
            args.hard_neg, args.hard_neg_num,
            train_flag=False, embedding_saving=True, optimizer=optimizer)

    print("\nTraining complete!")
    if writer:
        writer.close()


if __name__ == "__main__":
    main()
