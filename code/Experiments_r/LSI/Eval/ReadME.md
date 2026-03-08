## Legal Statute Identification (LSI)

### Desired Output Format

You need to name your prediction file "ilsi-test-pred.json".
You need to keep your predictions in the same format as "ilsi-test-gold.json".

```
Dict{
    String: List[String]
}
```
Each key represents the case number/id as maintained in the gold standard file.
Each value represents the list of relevant statutes (must be kept in a list, even if there is just a single entry).
Each entry in this list is one of the labels mentioned in "label_vocab.json".

### Running Evaluation
python -m pip install sklearn
python lsi-eval.py
