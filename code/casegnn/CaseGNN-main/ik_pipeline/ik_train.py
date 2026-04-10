"""
Training and evaluation loop for CaseGNN on the ik dataset.
Adapted from the original train.py with 10-digit case ID support
and without year-filter metrics.
"""
import torch
import torch.nn as nn
from tqdm import tqdm
import random
import dgl
import os
import json
import sys

# Add parent dir so we can import the original model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from model import CaseGNN, early_stopping
from torch_metrics import t_metrics, metric, rank

from config import ID_PAD_LEN


def forward(model, device, writer, dataloader,
            sumfact_pool_dataset, referissue_pool_dataset,
            label_dict, epoch, temp,
            bm25_hard_neg_dict, hard_neg, hard_neg_num,
            train_flag, embedding_saving, optimizer=None):
    """
    One epoch of training or evaluation.
    Returns NDCG@5 score during evaluation.
    """
    if train_flag:
        # ── Training ────────────────────────────────────────────────────
        loss_model = nn.CrossEntropyLoss()
        model.train()
        optimizer.zero_grad()

        for batched_graph, labels in tqdm(dataloader, desc="Train"):
            batched_case_list = []
            for i in range(len(labels)):
                batched_case_list.append(str(int(labels[i])).zfill(ID_PAD_LEN))

            query_sumfact_graph = []
            query_referissue_graph = []
            positive_sumfact_graph = []
            positive_referissue_graph = []
            ran_neg_sumfact_graph = []
            ran_neg_referissue_graph = []
            bm25_neg_sumfact_graph = []
            bm25_neg_referissue_graph = []

            for x in range(len(batched_case_list)):
                query_name = batched_case_list[x] + ".txt"

                # Query graphs
                query_sumfact_graph.append(sumfact_pool_dataset.graphs[batched_case_list[x]])
                query_referissue_graph.append(referissue_pool_dataset.graphs[batched_case_list[x]])

                # Positive sample
                pos_case = random.choice(label_dict[query_name]).split(".")[0]
                positive_sumfact_graph.append(sumfact_pool_dataset.graphs[pos_case])
                positive_referissue_graph.append(referissue_pool_dataset.graphs[pos_case])

                # Random negative (not a relevant case)
                ran_neg_case = random.choice(list(sumfact_pool_dataset.labels.keys()))
                for _ in range(5000):
                    ran_neg_case = random.choice(list(sumfact_pool_dataset.labels.keys()))
                    if ran_neg_case + ".txt" not in label_dict[query_name]:
                        break
                ran_neg_sumfact_graph.append(sumfact_pool_dataset.graphs[ran_neg_case])
                ran_neg_referissue_graph.append(referissue_pool_dataset.graphs[ran_neg_case])

                # BM25 hard negatives
                if hard_neg and query_name in bm25_hard_neg_dict:
                    for _ in range(hard_neg_num):
                        bm25_neg_case = random.choice(bm25_hard_neg_dict[query_name]).split(".")[0]
                        bm25_neg_sumfact_graph.append(sumfact_pool_dataset.graphs[bm25_neg_case])
                        bm25_neg_referissue_graph.append(referissue_pool_dataset.graphs[bm25_neg_case])
                elif hard_neg:
                    # Fallback if query not in BM25 dict
                    for _ in range(hard_neg_num):
                        ran = random.choice(list(sumfact_pool_dataset.labels.keys()))
                        bm25_neg_sumfact_graph.append(sumfact_pool_dataset.graphs[ran])
                        bm25_neg_referissue_graph.append(referissue_pool_dataset.graphs[ran])

            # ── Forward pass ────────────────────────────────────────────
            def encode_batch(graphs, pool):
                batch = dgl.batch(graphs)
                node_f = batch.ndata["w"]
                edge_f = batch.edata["w"]
                out = model(batch.to(device), node_f.to(device), edge_f.to(device))
                return out / out.norm(dim=1, keepdim=True)

            out_q_fact  = encode_batch(query_sumfact_graph, None)
            out_q_issue = encode_batch(query_referissue_graph, None)
            out_p_fact  = encode_batch(positive_sumfact_graph, None)
            out_p_issue = encode_batch(positive_referissue_graph, None)
            out_rn_fact = encode_batch(ran_neg_sumfact_graph, None)
            out_rn_issue= encode_batch(ran_neg_referissue_graph, None)

            # Positive logits
            l_pos = (torch.mm(out_q_fact, out_p_fact.T) +
                     torch.mm(out_q_issue, out_p_issue.T))

            # In-batch negatives (self-similarity with diagonal masked)
            l_neg = (torch.mm(out_q_fact, out_q_fact.T) +
                     torch.mm(out_q_issue, out_q_issue.T))
            l_neg.fill_diagonal_(float("-inf"))

            # Random negatives
            l_ran = (torch.mm(out_q_fact, out_rn_fact.T) +
                     torch.mm(out_q_issue, out_rn_issue.T))

            if hard_neg and bm25_neg_sumfact_graph:
                out_bn_fact = encode_batch(bm25_neg_sumfact_graph, None)
                out_bn_issue = encode_batch(bm25_neg_referissue_graph, None)
                l_bm25 = (torch.mm(out_q_fact, out_bn_fact.T) +
                          torch.mm(out_q_issue, out_bn_issue.T))
                logits = torch.cat([l_pos, l_neg, l_ran, l_bm25], dim=1).to(device)
            else:
                logits = torch.cat([l_pos, l_neg, l_ran], dim=1).to(device)

            target = torch.arange(0, len(labels)).long().to(device)
            loss = loss_model(logits / temp, target)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if writer is not None:
                writer.add_scalar("Loss/Train", loss.item(), epoch)
            print(f"  Loss: {loss.item():.4f}")

    else:
        # ── Evaluation ──────────────────────────────────────────────────
        model.eval()
        with torch.no_grad():
            test_label_list = []
            sumfact_reps = []
            referissue_reps = []

            for batched_graph, labels in tqdm(dataloader, desc="Eval"):
                # Fact graphs
                feat_n = batched_graph.ndata["w"]
                feat_e = batched_graph.edata["w"]
                sf_out = model(batched_graph.to(device), feat_n.to(device), feat_e.to(device))
                sumfact_reps.append(sf_out)

                # Issue graphs for the same cases
                ri_graphs = []
                for i in labels:
                    case_name = str(int(i)).zfill(ID_PAD_LEN)
                    test_label_list.append(case_name)
                    ri_graphs.append(referissue_pool_dataset.graphs[case_name])

                ri_batch = dgl.batch(ri_graphs)
                ri_out = model(ri_batch.to(device), ri_batch.ndata["w"].to(device),
                               ri_batch.edata["w"].to(device))
                referissue_reps.append(ri_out)

            sf_matrix = torch.cat(sumfact_reps, dim=0)
            ri_matrix = torch.cat(referissue_reps, dim=0)

            sf_norm = sf_matrix / sf_matrix.norm(dim=1, keepdim=True)
            ri_norm = ri_matrix / ri_matrix.norm(dim=1, keepdim=True)

            sim = (torch.mm(sf_norm, sf_norm.T) +
                   torch.mm(ri_norm, ri_norm.T))
            sim.fill_diagonal_(float("-inf"))

            # Rank w.r.t. each query
            sim_scores = []
            test_query_list = []
            for key in label_dict:
                test_query_list.append(key)
                q_idx = test_label_list.index(key.split(".")[0])
                sim_scores.append(sim[q_idx, :])
            sim_scores = torch.stack(sim_scores)

            final_pred = rank(sim_scores, len(test_label_list),
                              test_query_list, test_label_list)

            # Metrics
            (correct, retri, relevant,
             micro_p, micro_r, micro_f,
             macro_p, macro_r, macro_f) = metric(5, final_pred, label_dict)

            ndcg, mrr, map_s, p5 = t_metrics(label_dict, final_pred, 5)

            print(f"  Micro P/R/F: {micro_p:.4f} / {micro_r:.4f} / {micro_f:.4f}")
            print(f"  Macro P/R/F: {macro_p:.4f} / {macro_r:.4f} / {macro_f:.4f}")
            print(f"  NDCG@5: {ndcg:.4f}  MRR@5: {mrr:.4f}  MAP: {map_s:.4f}  P@5: {p5:.4f}")

    # ── Embedding saving ────────────────────────────────────────────────
    if embedding_saving:
        model.eval()
        with torch.no_grad():
            labels_list = []
            sf_reps = []
            ri_reps = []
            for i in range(len(sumfact_pool_dataset.graph_list)):
                g_sf = sumfact_pool_dataset.graph_list[i]
                sf_out = model(g_sf.to(device), g_sf.ndata["w"].to(device),
                               g_sf.edata["w"].to(device))
                sf_reps.append(sf_out)

                lbl = str(int(sumfact_pool_dataset.label_list[i])).zfill(ID_PAD_LEN)
                labels_list.append(lbl)

                if lbl in referissue_pool_dataset.graphs:
                    g_ri = referissue_pool_dataset.graphs[lbl]
                else:
                    g_ri = g_sf  # fallback
                ri_out = model(g_ri.to(device), g_ri.ndata["w"].to(device),
                               g_ri.edata["w"].to(device))
                ri_reps.append(ri_out)

            sf_all = torch.cat(sf_reps, dim=0)
            ri_all = torch.cat(ri_reps, dim=0)
            case_emb = torch.cat((sf_all, ri_all), dim=1)  # [N, 1536]

            split_tag = "train" if train_flag else "test"
            emb_dir = os.path.join(os.path.dirname(__file__), "output", "embeddings")
            torch.save(case_emb, os.path.join(emb_dir, f"ik_{split_tag}_casegnn_embedding.pt"))
            with open(os.path.join(emb_dir, f"ik_{split_tag}_casegnn_embedding_case_list.json"), "w") as f:
                json.dump(labels_list, f)
            print(f"  Saved CaseGNN embeddings: {case_emb.shape}")

    if not train_flag:
        return ndcg  # type: ignore[possibly-undefined]
