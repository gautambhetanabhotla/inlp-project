#!/usr/bin/env bash
#
# End-to-end CaseGNN pipeline for ik (Indian Kanoon) data.
# Run from the ik_pipeline/ directory:
#   cd code/casegnn/CaseGNN-main/ik_pipeline
#   bash run_pipeline.sh
#
set -euo pipefail
cd "$(dirname "$0")"

echo "============================================================"
echo "Step 1: Prepare labels and BM25 hard negatives"
echo "============================================================"
python step1_prepare_labels.py

echo ""
echo "============================================================"
echo "Step 2: Feature extraction + NER + relation extraction"
echo "        (this step can take a while for large datasets)"
echo "============================================================"
python step2_preprocess.py

echo ""
echo "============================================================"
echo "Step 3: Generate SAILER embeddings"
echo "        (requires GPU for speed; CPU is slow)"
echo "============================================================"
python step3_embeddings.py

echo ""
echo "============================================================"
echo "Step 4: Build TACG graphs"
echo "        (requires GPU for SAILER encoding of entities)"
echo "============================================================"
python step4_graphs.py

echo ""
echo "============================================================"
echo "Step 5: Train CaseGNN"
echo "============================================================"
python step5_train.py \
    --epochs 600 \
    --batch_size 16 \
    --lr 5e-5 \
    --wd 5e-5 \
    --dropout 0.1 \
    --temp 0.1 \
    --num_head 1 \
    --hard_neg \
    --hard_neg_num 5

echo ""
echo "============================================================"
echo "Pipeline complete!"
echo "============================================================"
echo ""
echo "To retrieve relevant cases:"
echo "  python retrieve.py --query_id 0001104022 --top_k 10"
echo ""
echo "Outputs:"
echo "  output/labels/         – label files"
echo "  output/ie/             – structured CSVs"
echo "  output/embeddings/     – SAILER + CaseGNN embeddings"
echo "  output/graphs/         – DGL graph binaries"
echo "  output/experiments/    – training logs + best model"
