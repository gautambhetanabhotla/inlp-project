"""
Configuration module for the Similar Case Matching pipeline.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


@dataclass
class PathConfig:
    """Paths for data and model artifacts."""
    project_root: Path = Path(__file__).resolve().parent.parent
    data_dir: Path = field(init=False)
    raw_data_dir: Path = field(init=False)
    processed_data_dir: Path = field(init=False)
    models_dir: Path = field(init=False)
    results_dir: Path = field(init=False)
    kg_dir: Path = field(init=False)

    def __post_init__(self):
        self.data_dir = self.project_root / "data"
        self.raw_data_dir = self.data_dir / "raw"
        self.processed_data_dir = self.data_dir / "processed"
        self.models_dir = self.project_root / "models"
        self.results_dir = self.project_root / "results"
        self.kg_dir = self.data_dir / "knowledge_graph"

    def create_dirs(self):
        for d in [
            self.raw_data_dir,
            self.processed_data_dir,
            self.models_dir,
            self.results_dir,
            self.kg_dir,
        ]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class RRConfig:
    """Rhetorical Role segmentation configuration."""
    model_name: str = "law-ai/InLegalBERT"
    num_labels: int = 13
    label_names: list = field(default_factory=lambda: [
        "Preamble",             # 0
        "Facts",                # 1
        "Ruling by Lower Court",# 2
        "Issues",               # 3
        "Argument by Petitioner",# 4
        "Argument by Respondent",# 5
        "Analysis",             # 6
        "Statute",              # 7
        "Precedent Relied",     # 8
        "Precedent Not Relied", # 9
        "Ratio of the decision",# 10
        "Ruling by Present Court",# 11
        "NONE",                 # 12
    ])
    max_seq_length: int = 512
    batch_size: int = 16
    learning_rate: float = 2e-5
    num_epochs: int = 5
    warmup_ratio: float = 0.1
    weight_decay: float = 0.01
    dropout: float = 0.1
    use_context: bool = True  # Use surrounding sentences as context
    context_window: int = 2   # Number of surrounding sentences to include


@dataclass
class KGConfig:
    """Knowledge Graph configuration."""
    embedding_dim: int = 128
    walk_length: int = 30
    num_walks: int = 200
    p: float = 1.0           # Node2Vec return parameter
    q: float = 1.0           # Node2Vec in-out parameter
    window_size: int = 10
    workers: int = 4
    epochs: int = 5


@dataclass
class EncoderConfig:
    """Segment encoder configuration."""
    model_name: str = "law-ai/InLegalBERT"
    max_seq_length: int = 512
    embedding_dim: int = 768
    pooling_strategy: str = "mean"  # mean, cls, max
    batch_size: int = 16


@dataclass
class RetrievalConfig:
    """Retrieval pipeline configuration."""
    top_k: int = 50
    # Weights for combining segment similarities (keys = RR label names)
    segment_weights: dict = field(default_factory=lambda: {
        "Facts": 0.30,
        "Ratio of the decision": 0.25,
        "Analysis": 0.15,
        "Statute": 0.10,
        "Argument by Petitioner": 0.05,
        "Argument by Respondent": 0.05,
        "Precedent Relied": 0.05,
        "Issues": 0.05,
    })
    graph_weight: float = 0.2       # Weight for graph-based similarity
    semantic_weight: float = 0.8    # Weight for semantic similarity
    rerank: bool = True
    rerank_top_k: int = 100         # Candidates to consider for re-ranking


@dataclass
class EvalConfig:
    """Evaluation configuration."""
    k_values: list = field(default_factory=lambda: [5, 10, 20, 50])


@dataclass
class Config:
    """Master configuration."""
    paths: PathConfig = field(default_factory=PathConfig)
    rr: RRConfig = field(default_factory=RRConfig)
    kg: KGConfig = field(default_factory=KGConfig)
    encoder: EncoderConfig = field(default_factory=EncoderConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    seed: int = 42
    device: Optional[str] = None  # auto-detect if None

    def __post_init__(self):
        if self.device is None:
            import torch
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.paths.create_dirs()
