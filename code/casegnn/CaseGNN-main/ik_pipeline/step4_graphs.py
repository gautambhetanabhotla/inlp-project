"""
Step 4 – Build Text-Attributed Case Graphs (TACG) from structured CSVs
         and PromptCase embeddings.

Produces (for each split × feature):
  output/graphs/bidirec_{split}_{feat}.bin           – full pool graphs
  output/graphs/bidirec_{split}_{feat}_Synthetic.bin – query-only graphs (for dataloader)
"""
import json
import os
import sys
import torch
import dgl
from dgl.data.utils import save_graphs
from tqdm import tqdm

import _torch_compat  # noqa: F401 – patch transformers torch.load check
from transformers import AutoModel, AutoTokenizer

from config import (
    IE_DIR, EMBEDDING_DIR, GRAPH_DIR, LABEL_DIR,
    SAILER_MODEL_NAME, ID_PAD_LEN, EMBEDDING_DIM, ensure_dirs
)


def build_graphs_for_split(split, feature, model, tokenizer, device):
    """
    Build TACG graphs for one (split, feature) combination.
    """
    # Load PromptCase embeddings
    emb_path = os.path.join(EMBEDDING_DIR, f"{split}_fact_issue_cross_embedding.pt")
    emb_list_path = os.path.join(EMBEDDING_DIR, f"{split}_fact_issue_cross_embedding_case_list.json")
    candidate_matrix = torch.load(emb_path, weights_only=False)
    with open(emb_list_path, "r") as f:
        candidate_matrix_index = json.load(f)

    embedding_index = 0 if feature == "fact" else 1

    ie_path = os.path.join(IE_DIR, f"{split}_{feature}", "result")
    file_list = sorted([f for f in os.listdir(ie_path) if f.endswith(".csv")])

    graph_list = []
    graph_name_list = []
    zero_files = []

    model.eval()
    with torch.no_grad():
        for fname in tqdm(file_list, desc=f"Graphs [{split}/{feature}]"):
            case_id = fname.replace(".csv", "")
            case_name = case_id + ".txt"

            # Get PromptCase embedding for this case
            try:
                emb_idx = candidate_matrix_index.index(case_name)
            except ValueError:
                print(f"  Warning: {case_name} not in embedding list, skipping")
                continue
            promptcase_embedding = candidate_matrix[0][embedding_index][emb_idx]

            graph_name_list.append(int(case_id))

            list_node1 = []
            list_node2 = []
            relation_embedding_weights = []
            node_embedding_weights = []
            seen_triplets = []

            # Read structured CSV
            csv_path = os.path.join(ie_path, fname)
            with open(csv_path, "r", errors="replace") as f:
                lines = f.readlines()

            node_num = -1
            index_dict = {}

            for line in lines:
                stripped = line.strip()
                if stripped == "Type,Entity1,Relationship,Type,Entity2":
                    # Header → create virtual (promptcase) node
                    node_num += 1
                    index_dict["promptcase_node"] = node_num
                    node_embedding_weights.append(promptcase_embedding)
                    # Self-loop on virtual node
                    list_node1.append(node_num)
                    list_node2.append(node_num)
                    relation_embedding_weights.append(promptcase_embedding)
                    list_node2.append(node_num)
                    list_node1.append(node_num)
                    relation_embedding_weights.append(promptcase_embedding)
                    continue

                if not stripped:
                    continue

                # Parse CSV row: Type,Entity1,Relationship,Type,Entity2
                # Use csv module would be safer, but original code uses split
                parts = stripped.replace("×", "").split(",")
                if len(parts) < 5:
                    continue
                type1, ent1, rel, type2, ent2 = parts[0], parts[1], parts[2], parts[3], ",".join(parts[4:])

                triplet = [ent1, rel, ent2]
                if triplet in seen_triplets:
                    continue
                seen_triplets.append(triplet)

                # Encode entities and relation with SAILER
                texts = [ent1, rel, ent2]
                tokenized = tokenizer(texts, return_tensors="pt", padding=True,
                                      truncation=True, max_length=128).to(device)
                embedding = model(**tokenized)
                cls = embedding[0][:, 0].cpu()  # [3, 768]
                ent1_emb, rel_emb, ent2_emb = cls[0], cls[1], cls[2]

                # Add/reuse Entity1 node
                if ent1 in index_dict:
                    e1_idx = index_dict[ent1]
                else:
                    node_num += 1
                    e1_idx = node_num
                    index_dict[ent1] = e1_idx
                    node_embedding_weights.append(ent1_emb)

                # Add/reuse Entity2 node
                if ent2 in index_dict:
                    e2_idx = index_dict[ent2]
                else:
                    node_num += 1
                    e2_idx = node_num
                    index_dict[ent2] = e2_idx
                    node_embedding_weights.append(ent2_emb)

                pc_idx = index_dict["promptcase_node"]

                # Bidirectional edges: E1↔E2
                list_node1.append(e1_idx); list_node2.append(e2_idx)
                relation_embedding_weights.append(rel_emb)
                list_node1.append(e2_idx); list_node2.append(e1_idx)
                relation_embedding_weights.append(rel_emb)

                # Edges: promptcase ↔ E1
                list_node1.append(pc_idx); list_node2.append(e1_idx)
                relation_embedding_weights.append(ent1_emb)
                list_node1.append(e1_idx); list_node2.append(pc_idx)
                relation_embedding_weights.append(ent1_emb)

                # Edges: promptcase ↔ E2
                list_node1.append(pc_idx); list_node2.append(e2_idx)
                relation_embedding_weights.append(ent2_emb)
                list_node1.append(e2_idx); list_node2.append(pc_idx)
                relation_embedding_weights.append(ent2_emb)

            # Build DGL graph
            if len(relation_embedding_weights) == 0:
                # Fallback: create a minimal graph with just the virtual node
                zero_files.append(case_id)
                g = dgl.graph(([0, 0], [0, 0]))
                g.ndata["w"] = promptcase_embedding.unsqueeze(0)
                g.edata["w"] = torch.stack([promptcase_embedding, promptcase_embedding])
            else:
                g = dgl.graph((list_node1, list_node2))
                g.ndata["w"] = torch.stack(node_embedding_weights)
                g.edata["w"] = torch.stack(relation_embedding_weights)

            graph_list.append(g)

    if zero_files:
        print(f"  {len(zero_files)} cases had no relation triplets (minimal graphs created)")

    tensor_names = torch.LongTensor(graph_name_list)

    # Save a companion JSON for case name mapping (avoids float precision loss)
    name_map_path = os.path.join(GRAPH_DIR, f"bidirec_{split}_{feature}_names.json")
    with open(name_map_path, "w") as f:
        json.dump([str(n).zfill(ID_PAD_LEN) for n in graph_name_list], f)

    # Save pool graphs – use LongTensor to avoid float32 precision loss on 10-digit IDs
    pool_labels = {"name_list": tensor_names}
    pool_path = os.path.join(GRAPH_DIR, f"bidirec_{split}_{feature}.bin")
    save_graphs(pool_path, graph_list, pool_labels)
    print(f"  Pool graphs saved: {pool_path} ({len(graph_list)} graphs)")

    # Save synthetic graphs (query-only subset)
    if split == "test":
        # For test, synthetic = all test cases
        synth_labels = {"glabel": tensor_names}
        synth_path = os.path.join(GRAPH_DIR, f"bidirec_{split}_{feature}_Synthetic.bin")
        save_graphs(synth_path, graph_list, synth_labels)
    else:
        # For train, synthetic = only query cases
        labels_path = os.path.join(LABEL_DIR, "train_labels.json")
        with open(labels_path, "r") as f:
            train_labels = json.load(f)

        # Build lookup: case_id_int → graph
        case_graph_map = {}
        for i, name_int in enumerate(graph_name_list):
            case_graph_map[str(name_int).zfill(ID_PAD_LEN)] = (graph_list[i], name_int)

        query_graphs = []
        query_labels_int = []
        for qname in train_labels:
            qid = qname.replace(".txt", "")
            if qid in case_graph_map:
                g, name_int = case_graph_map[qid]
                query_graphs.append(g)
                query_labels_int.append(name_int)

        synth_labels = {"glabel": torch.LongTensor(query_labels_int)}
        synth_path = os.path.join(GRAPH_DIR, f"bidirec_{split}_{feature}_Synthetic.bin")
        save_graphs(synth_path, query_graphs, synth_labels)
        print(f"  Synthetic graphs saved: {synth_path} ({len(query_graphs)} graphs)")


def main():
    ensure_dirs()
    os.makedirs(GRAPH_DIR, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading SAILER model: {SAILER_MODEL_NAME}")
    model = AutoModel.from_pretrained(SAILER_MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(SAILER_MODEL_NAME)

    for split in ["train", "test"]:
        for feature in ["fact", "issue"]:
            build_graphs_for_split(split, feature, model, tokenizer, device)

    print("\nDone. Graphs saved to:", GRAPH_DIR)


if __name__ == "__main__":
    main()
