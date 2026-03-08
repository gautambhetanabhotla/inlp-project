"""
infer_rr.py — Segment IK legal documents using OpenNyAI's pre-trained
Rhetorical Role model (LREC 2022, trained on Indian court judgments).

Reads .txt files from input_dir, assigns a rhetorical role to every
sentence, then writes only the sentences matching target roles to output_dir.

Install:
  pip install opennyai spacy==3.2.4

Usage:
  # Segment candidates and queries for ik_train and ik_test
  for SPLIT in ik_train ik_test; do
    for PART in candidate query; do
      python3 infer_rr.py \\
        --input_dir  ../../Models/BM25/data/corpus/$SPLIT/$PART \\
        --output_dir ../../Models/BM25/data/corpus/${SPLIT}_RR/$PART
    done
  done

Available roles (--roles):
  FAC              - Facts of the case
  ISSUE            - Legal questions framed by the court
  ARG_PETITIONER   - Petitioner lawyer arguments
  ARG_RESPONDENT   - Respondent lawyer arguments
  ANALYSIS         - Court analysis (parent of STA, PRE_RELIED, PRE_NOT_RELIED)
  STA              - Statutes discussed
  PRE_RELIED       - Precedent cases relied on
  PRE_NOT_RELIED   - Precedent cases not relied on
  Ratio            - Ratio of the decision
  RPC              - Ruling by present court
  RLC              - Ruling by lower court
  PREAMBLE         - Header / preamble
  NONE             - Unclassified
"""

import os
import gc
import argparse
import codecs
from tqdm import tqdm


def load_opennyai():
    try:
        from opennyai import RhetoricalRolePredictor
        from opennyai.utils import Data
        return RhetoricalRolePredictor, Data
    except ImportError:
        raise ImportError(
            "opennyai not installed.\n"
            "Run: pip install opennyai spacy==3.2.4"
        )


def parse_results(results):
    """Parse opennyai output into list of (sentence, label) lists."""
    docs_labeled = []
    for res in results:
        sentences = []
        for ann in res.get('annotations', []):
            sent_text = ann.get('text', '').strip()
            labels    = ann.get('labels', [])
            role      = labels[0] if labels else 'NONE'
            if sent_text:
                sentences.append((sent_text, role))
        docs_labeled.append(sentences)
    return docs_labeled


def main(args):
    # Load model once (downloads ~500MB on first run, cached to ~/.opennyai)
    print("Loading OpenNyAI Rhetorical Role model (downloads on first run) …")
    RhetoricalRolePredictor, Data = load_opennyai()
    predictor = RhetoricalRolePredictor(use_gpu=args.use_gpu, verbose=False)
    print("Model ready.")

    target_roles = set(args.roles)
    print(f"Target roles : {sorted(target_roles)}")
    print(f"Input  dir   : {args.input_dir}")
    print(f"Output dir   : {args.output_dir}")
    os.makedirs(args.output_dir, exist_ok=True)

    files = sorted(f for f in os.listdir(args.input_dir) if f.endswith(".txt"))
    if not files:
        print("No .txt files found in input_dir. Exiting.")
        return

    # Resume: skip files already written to output_dir
    already_done = set(os.listdir(args.output_dir)) if os.path.isdir(args.output_dir) else set()
    files = [f for f in files if f not in already_done]
    if already_done:
        print(f"Resuming: skipping {len(already_done)} already-processed files.")

    print(f"\nProcessing {len(files)} files in batches of {args.batch_size} …\n")

    skipped_empty = 0
    kept_full     = 0  # docs where no target sentences found → keep full text

    for batch_start in tqdm(range(0, len(files), args.batch_size), desc="Batches"):
        batch_files = files[batch_start : batch_start + args.batch_size]
        texts = []
        for fname in batch_files:
            fpath = os.path.join(args.input_dir, fname)
            with codecs.open(fpath, "r", "utf-8", errors="ignore") as fh:
                texts.append(fh.read().strip())

        data = None
        results = None
        labeled_docs = None
        try:
            data = Data(texts)
            results = predictor(data)
            # opennyai deletes its temp dir after each call — recreate for next batch
            os.makedirs(predictor.hsln_format_txt_dirpath, exist_ok=True)
            labeled_docs = parse_results(results)
        except Exception as e:
            print(f"\n[ERROR] batch starting at {batch_start}: {e}")
            # Fallback: write original text unmodified
            for fname, text in zip(batch_files, texts):
                out_path = os.path.join(args.output_dir, fname)
                with open(out_path, "w", encoding="utf-8") as fh:
                    fh.write(text)
            skipped_empty += len(batch_files)
            del data, results, labeled_docs, texts
            gc.collect()
            continue

        batch_results = list(zip(batch_files, texts, labeled_docs))
        del data, results, labeled_docs
        gc.collect()

        for fname, text, labeled in batch_results:
            out_path = os.path.join(args.output_dir, fname)

            if not labeled:
                # opennyai returned nothing — write original text
                with open(out_path, "w", encoding="utf-8") as fh:
                    fh.write(text)
                skipped_empty += 1
                continue

            kept_sents = [sent for sent, role in labeled if role in target_roles]

            if not kept_sents:
                # No sentences matched target roles — keep full text as fallback
                out_text = text
                kept_full += 1
            else:
                out_text = " ".join(kept_sents)

            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write(out_text)

        del batch_results, texts
        gc.collect()

    print(f"\nDone.")
    print(f"  Total files processed     : {len(files)}")
    print(f"  Empty/failed (kept orig)  : {skipped_empty}")
    print(f"  No role match (kept full) : {kept_full}")
    print(f"  Output dir                : {args.output_dir}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir",  required=True,
                   help="Folder containing .txt legal documents")
    p.add_argument("--output_dir", required=True,
                   help="Output folder for RR-filtered documents")
    p.add_argument("--roles", nargs="+",
                   default=["FAC", "ISSUE", "ARG_PETITIONER", "ARG_RESPONDENT",
                            "ANALYSIS", "STA", "PRE_RELIED", "PRE_NOT_RELIED",
                            "Ratio", "RPC"],
                   help="Rhetorical roles to keep. Default keeps all substantive "
                        "roles (drops PREAMBLE and NONE).")
    p.add_argument("--batch_size", type=int, default=2,
                   help="Documents per inference batch (default: 2)")
    p.add_argument("--use_gpu", action="store_true",
                   help="Use GPU for inference (default: CPU)")
    args = p.parse_args()
    main(args)
