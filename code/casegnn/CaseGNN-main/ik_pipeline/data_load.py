"""
Dataset utilities for loading DGL graphs – adapted for 10-digit ik case IDs.
"""
import json
import os
import sys

import dgl
from dgl.data.utils import load_graphs
from dgl.data import DGLDataset
import torch

from config import ID_PAD_LEN


class PoolDataset(DGLDataset):
    """Full case pool (all cases). Provides dict-based access by case ID."""

    def __init__(self, file_path):
        self.graph_and_label = load_graphs(file_path)
        # Try to load companion JSON for precise name mapping
        json_path = file_path.replace(".bin", "_names.json")
        self._name_json = None
        if os.path.exists(json_path):
            import json
            with open(json_path) as f:
                self._name_json = json.load(f)
        super().__init__(name="Pool")

    def process(self):
        case_pool, pool_label = self.graph_and_label
        labels = pool_label["name_list"].tolist()

        self.graphs = {}
        self.labels = {}
        for i in range(len(labels)):
            if self._name_json:
                key = self._name_json[i]
            else:
                key = str(int(labels[i])).zfill(ID_PAD_LEN)
            self.graphs[key] = case_pool[i]
            self.labels[key] = [key]

        self.graph_list = case_pool
        self.label_list = pool_label["name_list"].tolist()

    def __getitem__(self, i):
        return self.graph_list[i], self.label_list[i]

    def __len__(self):
        return len(self.graph_list)


class SyntheticDataset(DGLDataset):
    """Query-only dataset used by the DataLoader during training."""

    def __init__(self, file_path):
        self.graph_and_label = load_graphs(file_path)
        # Try to load companion JSON for precise name mapping
        base_path = file_path.replace("_Synthetic.bin", "")
        json_path = base_path + "_names.json"
        self._name_json = None
        if os.path.exists(json_path):
            import json
            with open(json_path) as f:
                all_names = json.load(f)
            # Build lookup from int → padded string
            self._int_to_name = {int(n): n for n in all_names}
        super().__init__(name="Synthetic")

    def process(self):
        graphs = self.graph_and_label[0]
        labels = self.graph_and_label[1]["glabel"].tolist()

        self.graphs = {}
        self.labels = {}
        for i in range(len(labels)):
            if hasattr(self, '_int_to_name') and self._int_to_name:
                key = self._int_to_name.get(int(labels[i]), str(int(labels[i])).zfill(ID_PAD_LEN))
            else:
                key = str(int(labels[i])).zfill(ID_PAD_LEN)
            self.graphs[key] = graphs[i]
            self.labels[key] = [key]

        self.graph_list = graphs
        self.label_list = labels

    def __getitem__(self, i):
        return self.graph_list[i], self.label_list[i]

    def __len__(self):
        return len(self.graph_list)


def collate(samples):
    graphs, labels = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    return batched_graph, torch.tensor(labels)
