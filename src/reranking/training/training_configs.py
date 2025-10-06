import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from reranking.models.base import BaseRerankingModel
from reranking.models.pairwise_mlp import PairwiseMLPReranker


@dataclass
class TrainingConfig:
    config_name: str
    model: BaseRerankingModel
    dataset: torch.utils.data.Dataset
    optimizer: torch.optim.Optimizer
    batch_size: int
    output_dir: str
    save_each: int
    validate_each: int
    validation_size: int = 100000

    def get_output_path(self, step: int) -> str:
        dir_path = Path(self.output_dir) / self.config_name
        dir_path.mkdir(parents=True, exist_ok=True)
        return f"{dir_path}/{step}.pth"


def pairwise_mlp() -> TrainingConfig:
    name = "pairwise_mlp"
    LR = 0.0001
    SAVE_EACH = 10000
    BATCH_SIZE = 1024
    VALIDATE_EACH = 5000
    VALIDATION_SIZE = 1000
    model = PairwiseMLPReranker(
        model_name_or_path="/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base",
        state_dict_path="/lnet/work/home-students-external/farhan/troja/outputs/workdirs/asi_se_to_rozbilo_init_all/models_5/ema.pth",
        mlp_hidden_dim=2048,
    )
    data = np.load(
        "/lnet/work/home-students-external/farhan/troja/outputs/reranking_test/reranker_dataset_with_qids.npz"
    )
    labels = torch.from_numpy(data["y"]).float()
    description_tokens = torch.from_numpy(data["description_tokens"]).long()
    link_tokens = torch.from_numpy(data["link_tokens"]).long()

    dataset = torch.utils.data.TensorDataset(link_tokens, description_tokens, labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    output_dir = "/lnet/work/home-students-external/farhan/troja/outputs/reranking_models"
    return TrainingConfig(
        config_name=name,
        model=model,
        dataset=dataset,
        optimizer=optimizer,
        output_dir=output_dir,
        save_each=SAVE_EACH,
        batch_size=BATCH_SIZE,
        validate_each=VALIDATE_EACH,
        validation_size=VALIDATION_SIZE,
    )
