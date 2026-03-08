## Prior Case Retrieval (PCR)

### Desired Output Format

You need to name your prediction file "ilpcr-test-pred.json".
You need to keep your predictions in the same format as "ilsi-test-gold.json".

```
Dict{
    String: List[String]
}
```
Each key represents the case number/id of the query as maintained in the gold standard file.
In "ilpcr-test-gold.json" each value represents the list of relevant candidates
But, in "ilpcr-test-pred.json", each value must represent a list of relevant candidates <i>ranked in descending order</i> of the similarity score assigned by your model.


### Running Evaluation
python pcr-eval.py
