"""
Configuration for the CaseGNN pipeline adapted for ik (Indian Kanoon) data.
"""
import os

# ── Base paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
PIPELINE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "..", "..", "..", "data"))

IK_TRAIN_DIR = os.path.join(DATA_ROOT, "ik_train")
IK_TEST_DIR = os.path.join(DATA_ROOT, "ik_test")

# ── Output directories ─────────────────────────────────────────────────────
OUTPUT_DIR = os.path.join(PIPELINE_DIR, "output")
LABEL_DIR = os.path.join(OUTPUT_DIR, "labels")
IE_DIR = os.path.join(OUTPUT_DIR, "ie")
EMBEDDING_DIR = os.path.join(OUTPUT_DIR, "embeddings")
GRAPH_DIR = os.path.join(OUTPUT_DIR, "graphs")
EXPERIMENT_DIR = os.path.join(OUTPUT_DIR, "experiments")

# ── Case ID format ──────────────────────────────────────────────────────────
ID_PAD_LEN = 10  # ik case IDs are 10-digit zero-padded

# ── Model ───────────────────────────────────────────────────────────────────
SAILER_MODEL_NAME = "CSHaitao/SAILER_en_finetune"
EMBEDDING_DIM = 768

# ── Training hyperparameters ────────────────────────────────────────────────
IN_DIM = 768
H_DIM = 768
OUT_DIM = 768
DROPOUT = 0.1
NUM_HEAD = 1
EPOCHS = 600
LR = 5e-5
WD = 5e-5
BATCH_SIZE = 128
TEMP = 0.1
RAN_NEG_NUM = 1
HARD_NEG = True
HARD_NEG_NUM = 5
EARLY_STOP_PATIENCE = 10

# ── spaCy model for NER / dependency parsing ───────────────────────────────
SPACY_MODEL = "en_core_web_trf"  # high-accuracy transformer model
# Alternative: "en_core_web_sm" for speed (less accurate)


def ensure_dirs():
    """Create all output directories."""
    for d in [LABEL_DIR, EMBEDDING_DIR, GRAPH_DIR, EXPERIMENT_DIR]:
        os.makedirs(d, exist_ok=True)
    for split in ["train", "test"]:
        for feat in ["fact", "issue"]:
            os.makedirs(os.path.join(IE_DIR, f"{split}_{feat}", "result"), exist_ok=True)
