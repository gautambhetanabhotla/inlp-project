#!/bin/bash
# Runs OpenNyAI RR inference on all IK corpus splits.
# Uses the dedicated opennyai venv to avoid dependency conflicts.

PYTHON=/Users/sanyamagrawal/Desktop/nlp/Project/IL-PCR/Opennyai/venv/bin/python
SCRIPT=/Users/sanyamagrawal/Desktop/nlp/Project/IL-PCR/Models/BM25/infer_rr.py
CORPUS=/Users/sanyamagrawal/Desktop/nlp/Project/IL-PCR/Models/BM25/data/corpus
LOG_DIR=/Users/sanyamagrawal/Desktop/nlp/Project/IL-PCR/Models/BM25/logs
mkdir -p "$LOG_DIR"

echo "==== RR Inference started: $(date) ===="

for SPLIT in ik_train ik_test; do
  for PART in candidate query; do
    INPUT="$CORPUS/$SPLIT/$PART"
    OUTPUT="$CORPUS/${SPLIT}_RR/$PART"
    LOG="$LOG_DIR/rr_${SPLIT}_${PART}.log"
    mkdir -p "$OUTPUT"
    echo "[$SPLIT/$PART] $(ls $INPUT | wc -l) files -> $OUTPUT"
    cd /tmp && $PYTHON "$SCRIPT" \
      --input_dir  "$INPUT" \
      --output_dir "$OUTPUT" \
      --batch_size 4 \
      > "$LOG" 2>&1
    echo "[$SPLIT/$PART] Done. See $LOG"
  done
done

echo "==== RR Inference complete: $(date) ===="
