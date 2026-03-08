import json
import numpy as np
from sklearn.metrics import f1_score

GOLD_FILENAME = "ilsi-test-gold.json"
PRED_FILENAME = "ilsi-test-pred.json"

with open(GOLD_FILENAME) as fr:
    gold = json.load(fr)

with open(PRED_FILENAME) as fr:
    pred = json.load(fr)

with open("label_vocab.json") as fr:
    lv = json.load(fr)

G = np.zeros((len(gold), len(lv)))
P = np.zeros((len(gold), len(lv)))

for i, (id, labs) in enumerate(gold.items()):
    if id in pred:
        labs2 = pred[id]
    else:
        labs2 = []
    for l in labs:
        if l in lv:
            G[i, lv[l]] = 1
    for l in labs2:
        if l in lv:
            P[i, lv[l]] = 1

f1 = f1_score(G, P, average='macro')
print("Macro-F1 on ILSI test set:", f1)
    