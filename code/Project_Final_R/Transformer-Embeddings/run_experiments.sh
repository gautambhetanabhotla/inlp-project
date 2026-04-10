#!/bin/bash
# Run from: Models/Transformer-Embeddings/
# Activate venv first: source ../../.venv/bin/activate

# ── Round 1 (done) ────────────────────────────────────────────────────────────
# python eval_tfidf_only.py
# python hybrid_retrieval_chunk_sbert.py config_files/configs_ILPCR_hybrid/chunk_tfidf_a06_n200_test.json        # best: F1=38.54%
# python hybrid_retrieval_chunk_sbert.py config_files/configs_ILPCR_hybrid/chunk_tfidf_a05_n300_test.json
# python hybrid_retrieval_chunk_sbert.py config_files/configs_ILPCR_hybrid/chunk_tfidf_bigram_a05_n200_test.json

# ── Round 2: trigrams/quadgrams + MiniLM (~15 min each) ──────────────────────
python hybrid_retrieval_chunk_sbert.py config_files/configs_ILPCR_hybrid/chunk_tfidf_trigram_a06_n200_minilm_test.json
python hybrid_retrieval_chunk_sbert.py config_files/configs_ILPCR_hybrid/chunk_tfidf_quadgram_a06_n200_minilm_test.json

# ── Round 2: unigram/trigram/quadgram + MPNet (~30 min each) ─────────────────
python hybrid_retrieval_chunk_sbert.py config_files/configs_ILPCR_hybrid/chunk_tfidf_unigram_a06_n200_mpnet_test.json
python hybrid_retrieval_chunk_sbert.py config_files/configs_ILPCR_hybrid/chunk_tfidf_trigram_a06_n200_mpnet_test.json
python hybrid_retrieval_chunk_sbert.py config_files/configs_ILPCR_hybrid/chunk_tfidf_quadgram_a06_n200_mpnet_test.json

# ── Round 3: InLegalBERT with best TF-IDF config (~30 min each) ──────────────
python hybrid_retrieval_chunk_sbert.py config_files/configs_ILPCR_hybrid/chunk_tfidf_unigram_a06_n200_inlegalbert_test.json
python hybrid_retrieval_chunk_sbert.py config_files/configs_ILPCR_hybrid/chunk_tfidf_quadgram_a06_n200_inlegalbert_test.json

# ── Round 4: BERT base / large (~30-60 min each) ──────────────────────────────
python hybrid_retrieval_chunk_sbert.py config_files/configs_ILPCR_hybrid/chunk_tfidf_quadgram_a06_n200_bertbase_test.json
python hybrid_retrieval_chunk_sbert.py config_files/configs_ILPCR_hybrid/chunk_tfidf_quadgram_a06_n200_bertlarge_test.json

# ── Compare all results ───────────────────────────────────────────────────────
echo ""
echo "======= RESULTS SUMMARY ======="
for d in exp_results/HYBRID_CHUNK_*/; do
  python3 -c "
import json
o=json.load(open('${d}output.json'))
c=json.load(open('${d}config_file.json'))
f1s=o['f1_vs_K']; bk=f1s.index(max(f1s))
ckpt=c.get('checkpoint','?').split('/')[-1]
print(f'F1={max(f1s)*100:.2f}%  K={bk+1}  P={o[\"precision_vs_K\"][bk]*100:.2f}%  R={o[\"recall_vs_K\"][bk]*100:.2f}%  alpha={c.get(\"alpha\")}  topn={c.get(\"tfidf_top_n\")}  ngram={c.get(\"tfidf_n_gram\",1)}  model={ckpt}')
" 2>/dev/null
done | sort -t= -k2 -rn
