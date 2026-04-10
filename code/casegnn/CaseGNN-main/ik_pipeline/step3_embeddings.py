"""
Step 3 – Generate PromptCase-style SAILER embeddings from the structured CSVs.

Produces (for each split):
  output/embeddings/{split}_fact_issue_cross_embedding.pt
  output/embeddings/{split}_fact_issue_cross_embedding_case_list.json
"""
import json
import os
import sys
import torch
from tqdm import tqdm

import _torch_compat  # noqa: F401 – patch transformers torch.load check
from transformers import AutoModel, AutoTokenizer

from config import (
    IE_DIR, EMBEDDING_DIR, SAILER_MODEL_NAME, ensure_dirs
)


def generate_embeddings(split_name, model, tokenizer, device):
    fact_dir = os.path.join(IE_DIR, f"{split_name}_fact", "result")
    issue_dir = os.path.join(IE_DIR, f"{split_name}_issue", "result")

    # Use fact dir as the canonical file list
    files = sorted([f for f in os.listdir(fact_dir) if f.endswith(".csv")])
    if not files:
        print(f"  No CSVs found in {fact_dir}")
        return

    label_list = []
    fact_embeddings = []
    issue_embeddings = []
    cross_embeddings = []

    model.eval()
    with torch.no_grad():
        for pfile in tqdm(files, desc=f"Embeddings [{split_name}]"):
            case_id = pfile.replace(".csv", "")
            file_name = case_id + ".txt"
            label_list.append(file_name)

            # Read CSV content as text
            fact_csv_path = os.path.join(fact_dir, pfile)
            issue_csv_path = os.path.join(issue_dir, pfile)

            with open(fact_csv_path, "r", errors="replace") as f:
                fact_content = f.read()
            if os.path.exists(issue_csv_path):
                with open(issue_csv_path, "r", errors="replace") as f:
                    issue_content = f.read()
            else:
                issue_content = fact_content  # fallback

            fact_text = "Legal facts:" + fact_content
            issue_text = "Legal issues:" + issue_content
            cross_text = fact_text + " " + issue_text

            # Encode fact
            tok = tokenizer(fact_text, return_tensors="pt", padding=False,
                            truncation=True, max_length=512).to(device)
            emb = model(**tok)
            fact_embeddings.append(emb[0][:, 0].cpu())  # CLS token [1, 768]

            # Encode issue
            tok = tokenizer(issue_text, return_tensors="pt", padding=False,
                            truncation=True, max_length=512).to(device)
            emb = model(**tok)
            issue_embeddings.append(emb[0][:, 0].cpu())

            # Encode cross
            tok = tokenizer(cross_text, return_tensors="pt", padding=False,
                            truncation=True, max_length=512).to(device)
            emb = model(**tok)
            cross_embeddings.append(emb[0][:, 0].cpu())

    fact_matrix = torch.cat(fact_embeddings, dim=0)    # [N, 768]
    issue_matrix = torch.cat(issue_embeddings, dim=0)
    cross_matrix = torch.cat(cross_embeddings, dim=0)

    embedding_list = [[fact_matrix, issue_matrix, cross_matrix]]

    out_pt = os.path.join(EMBEDDING_DIR, f"{split_name}_fact_issue_cross_embedding.pt")
    out_json = os.path.join(EMBEDDING_DIR, f"{split_name}_fact_issue_cross_embedding_case_list.json")

    torch.save(embedding_list, out_pt)
    with open(out_json, "w") as f:
        json.dump(label_list, f)

    print(f"  Saved: {out_pt}  ({fact_matrix.shape})")


def main():
    ensure_dirs()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Loading SAILER model: {SAILER_MODEL_NAME}")
    model = AutoModel.from_pretrained(SAILER_MODEL_NAME).to(device)
    tokenizer = AutoTokenizer.from_pretrained(SAILER_MODEL_NAME)

    for split in ["train", "test"]:
        generate_embeddings(split, model, tokenizer, device)

    print("\nDone. Embeddings saved to:", EMBEDDING_DIR)


if __name__ == "__main__":
    main()
