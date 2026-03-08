#!/bin/bash
VENV=/Users/sanyamagrawal/Desktop/nlp/Project/IL-PCR/Opennyai/venv/bin/python
BASE=/Users/sanyamagrawal/Desktop/nlp/Project/IL-PCR/Models/BM25
LOG=$BASE/logs

cd $BASE
echo "=== RR Overnight Run started: $(date) ===" | tee -a $LOG/rr_master.log

# Keep Mac awake for the duration of this script
caffeinate -i -w $$ &
CAFF_PID=$!
echo "caffeinate PID: $CAFF_PID" | tee -a $LOG/rr_master.log

$VENV infer_rr.py --input_dir data/corpus/ik_train/candidate --output_dir data/corpus/ik_train_RR/candidate --batch_size 2 >> $LOG/rr_ik_train_candidate.log 2>&1
echo "[$(date)] train/candidate done: $(ls data/corpus/ik_train_RR/candidate/ | wc -l) files" | tee -a $LOG/rr_master.log

$VENV infer_rr.py --input_dir data/corpus/ik_train/query --output_dir data/corpus/ik_train_RR/query --batch_size 2 >> $LOG/rr_ik_train_query.log 2>&1
echo "[$(date)] train/query done: $(ls data/corpus/ik_train_RR/query/ | wc -l) files" | tee -a $LOG/rr_master.log

$VENV infer_rr.py --input_dir data/corpus/ik_test/candidate --output_dir data/corpus/ik_test_RR/candidate --batch_size 2 >> $LOG/rr_ik_test_candidate.log 2>&1
echo "[$(date)] test/candidate done: $(ls data/corpus/ik_test_RR/candidate/ | wc -l) files" | tee -a $LOG/rr_master.log

$VENV infer_rr.py --input_dir data/corpus/ik_test/query --output_dir data/corpus/ik_test_RR/query --batch_size 2 >> $LOG/rr_ik_test_query.log 2>&1
echo "[$(date)] test/query done: $(ls data/corpus/ik_test_RR/query/ | wc -l) files" | tee -a $LOG/rr_master.log

echo "=== ALL DONE: $(date) ===" | tee -a $LOG/rr_master.log
