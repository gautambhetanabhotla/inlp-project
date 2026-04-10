import os
import argparse

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm

def get_micro_scores_at_K(actual, predicted, k):
    act_set = set(actual)
    pred_set = set(predicted[:k])
    
    number_of_correctly_retrieved = len(act_set & pred_set) 
    number_of_relevant_cases = len(act_set)
    number_of_retrieved_cases = k

    return number_of_correctly_retrieved, number_of_relevant_cases, number_of_retrieved_cases

def get_f1_vs_K(gold_labels, similarity_df):
    # Build fast lookup indices once; repeated boolean indexing is very slow.
    gold_indexed = gold_labels.set_index("query_case_id")
    column_name = "query_case_id" if "query_case_id" in similarity_df.columns else "Unnamed: 0"
    sim_indexed = similarity_df.set_index(column_name)

    candidate_docs = np.asarray([int(i) for i in gold_labels.columns.values[1:]])

    # Precompute actual relevant set and ranked predictions once per query.
    query_data = []
    for query_case_id in sim_indexed.index.values:
        if query_case_id in [1864396, 1508893]:
            continue
        if query_case_id not in gold_indexed.index:
            continue

        gold_row = gold_indexed.loc[query_case_id]
        if isinstance(gold_row, pd.DataFrame):
            gold_row = gold_row.iloc[0]
        gold = gold_row.values
        actual = candidate_docs[np.logical_or(gold == 1, gold == -2)]
        actual = [str(i) for i in actual]

        sim_row = sim_indexed.loc[query_case_id]
        if isinstance(sim_row, pd.DataFrame):
            sim_row = sim_row.iloc[0]
        similarity_scores = sim_row.values
        assert len(similarity_scores) == len(candidate_docs)

        sorted_candidates = [
            x
            for _, x in sorted(
                zip(similarity_scores, candidate_docs),
                key=lambda pair: float(pair[0]),
                reverse=True,
            )
        ]
        if query_case_id in sorted_candidates:
            sorted_candidates.remove(query_case_id)
        sorted_candidates = [str(i) for i in sorted_candidates]

        query_data.append((actual, sorted_candidates))

    precision_vs_K = []
    recall_vs_K = []
    f1_vs_K = []
    for k in tqdm(range(1, 21)):
        number_of_correctly_retrieved_all = []
        number_of_relevant_cases_all = []
        number_of_retrieved_cases_all = []
        for actual, sorted_candidates in query_data:
            number_of_correctly_retrieved, number_of_relevant_cases, number_of_retrieved_cases = get_micro_scores_at_K(actual=actual, predicted=sorted_candidates, k=k)
            number_of_correctly_retrieved_all.append(number_of_correctly_retrieved)
            number_of_relevant_cases_all.append(number_of_relevant_cases)
            number_of_retrieved_cases_all.append(number_of_retrieved_cases)

        recall_scores = np.sum(number_of_correctly_retrieved_all)/np.sum(number_of_relevant_cases_all)
        precision_scores = np.sum(number_of_correctly_retrieved_all)/np.sum(number_of_retrieved_cases_all)
        if recall_scores == 0 or precision_scores == 0:
            f1_scores = 0
        else :    
            f1_scores = (2*precision_scores*recall_scores)/(precision_scores+recall_scores)

        recall_vs_K.append(recall_scores)
        precision_vs_K.append(precision_scores)
        f1_vs_K.append(f1_scores)
    
    return {
        "recall_vs_K" : recall_vs_K, 
        "precision_vs_K" : precision_vs_K, 
        "f1_vs_K" : f1_vs_K
    }
