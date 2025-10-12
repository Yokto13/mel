import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from reranking.models.base import BaseRerankingModel
from reranking.models.context_emb_with_attention import ContextEmbWithAttention
from reranking.models.full_lealla import FullLEALLAReranker
from reranking.models.pairwise_mlp import PairwiseMLPReranker
from reranking.models.pairwise_mlp_with_large_context_emb import (
    PairwiseMLPRerankerWithLargeContextEmb,
)
from reranking.training.reranking_iterable_dataset import RerankingIterableDataset
from utils.loaders import load_embs_and_qids


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
    epochs: int = 1

    def get_output_path(self, step: int) -> str:
        dir_path = Path(self.output_dir) / self.config_name
        dir_path.mkdir(parents=True, exist_ok=True)
        return f"{dir_path}/{step}.pth"


def pairwise_mlp(
    LR: float = 0.0001,
    SAVE_EACH: int = 5000,
    BATCH_SIZE: int = 1024,
    VALIDATE_EACH: int = 10000,
    VALIDATION_SIZE: int = 10000,
    DROPOUT: float = 0.5,
) -> TrainingConfig:
    name = "pairwise_mlp"
    model = PairwiseMLPReranker(
        model_name_or_path="/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base",
        state_dict_path="/lnet/work/home-students-external/farhan/troja/outputs/workdirs/asi_se_to_rozbilo_init_all/models_5/ema.pth",
        mlp_hidden_dim=2048,
        dropout=DROPOUT,
    )

    dataset = RerankingIterableDataset()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)
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


def full_lealla(
    LR: float = 0.0001,
    SAVE_EACH: int = 5000,
    BATCH_SIZE: int = 1536,
    VALIDATE_EACH: int = 10000,
    VALIDATION_SIZE: int = 10000,
    DROPOUT: float = 0.1,
) -> TrainingConfig:
    name = "full_lealla"
    model = FullLEALLAReranker(
        model_name_or_path="/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base",
        dropout=DROPOUT,
    )

    dataset = RerankingIterableDataset()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)
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


def full_lealla_r(
    LR: float = 0.0001,
    SAVE_EACH: int = 5000,
    BATCH_SIZE: int = 2048,
    VALIDATE_EACH: int = 10000,
    VALIDATION_SIZE: int = 10000,
    DROPOUT: float = 0.1,
) -> TrainingConfig:
    name = "full_lealla_r"
    model = FullLEALLAReranker(
        model_name_or_path="/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base",
        state_dict_path="/lnet/work/home-students-external/farhan/troja/outputs/workdirs/asi_se_to_rozbilo_init_all/models_5/ema.pth",
        dropout=DROPOUT,
    )

    dataset = RerankingIterableDataset()

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)
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


def pairwise_mlp_noise(
    LR: float = 0.0001,
    SAVE_EACH: int = 20000,
    BATCH_SIZE: int = 1024,
    VALIDATE_EACH: int = 10000,
    VALIDATION_SIZE: int = 10000,
) -> TrainingConfig:
    name = "pairwise_mlp_noise"
    model = PairwiseMLPReranker(
        model_name_or_path="/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base",
        state_dict_path="/lnet/work/home-students-external/farhan/troja/outputs/workdirs/asi_se_to_rozbilo_init_all/models_5/ema.pth",
        mlp_hidden_dim=2048,
        emb_noise=0.1,
    )
    data = np.load(
        "/lnet/work/home-students-external/farhan/troja/outputs/reranking_test/reranker_dataset_with_qids.npz"
    )
    # qids = torch.from_numpy(data["qids"]).long()
    labels = torch.from_numpy(data["y"]).float()
    description_tokens = torch.from_numpy(data["description_tokens"]).long()
    link_tokens = torch.from_numpy(data["link_tokens"]).long()

    dataset = torch.utils.data.TensorDataset(link_tokens, description_tokens, labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)
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


def pairwise_mlp_noise_dropout(
    LR: float = 0.0001,
    SAVE_EACH: int = 20000,
    BATCH_SIZE: int = 1024,
    VALIDATE_EACH: int = 10000,
    VALIDATION_SIZE: int = 10000,
) -> TrainingConfig:
    name = "pairwise_mlp_noise_dropout"
    model = PairwiseMLPReranker(
        model_name_or_path="/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base",
        state_dict_path="/lnet/work/home-students-external/farhan/troja/outputs/workdirs/asi_se_to_rozbilo_init_all/models_5/ema.pth",
        mlp_hidden_dim=2048,
        emb_noise=0.1,
        dropout=0.5,
    )
    data = np.load(
        "/lnet/work/home-students-external/farhan/troja/outputs/reranking_test/reranker_dataset_with_qids.npz"
    )
    # qids = torch.from_numpy(data["qids"]).long()
    labels = torch.from_numpy(data["y"]).float()
    description_tokens = torch.from_numpy(data["description_tokens"]).long()
    link_tokens = torch.from_numpy(data["link_tokens"]).long()

    dataset = torch.utils.data.TensorDataset(link_tokens, description_tokens, labels)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)
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


def pairwise_mlp_paraphrase(
    LR: float = 0.0001,
    SAVE_EACH: int = 5000,
    BATCH_SIZE: int = 1024,
    VALIDATE_EACH: int = 10000,
    VALIDATION_SIZE: int = 10000,
    DROPOUT: float = 0.5,
) -> TrainingConfig:
    name = "pairwise_mlp_paraphrase"

    paraphrase_embs, paraphrase_qids = load_embs_and_qids(
        "/lnet/work/home-students-external/farhan/troja/outputs/paraphrase_multilig_index"
    )
    paraphrase_embs = torch.from_numpy(paraphrase_embs)
    base_embs, base_qids = load_embs_and_qids(
        "/lnet/work/home-students-external/farhan/troja/outputs/workdirs/asi_se_to_rozbilo_init_all/damuel_for_index_6"
    )
    base_embs = torch.from_numpy(base_embs)

    dataset = RerankingIterableDataset()
    output_dir = "/lnet/work/home-students-external/farhan/troja/outputs/reranking_models"

    qid_to_paraphrase_emb = {qid: emb for qid, emb in zip(paraphrase_qids, paraphrase_embs)}
    qid_to_base_emb = {qid: emb for qid, emb in zip(base_qids, base_embs)}

    model = PairwiseMLPRerankerWithLargeContextEmb(
        model_name_or_path="/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base",
        state_dict_path="/lnet/work/home-students-external/farhan/troja/outputs/workdirs/asi_se_to_rozbilo_init_all/models_5/ema.pth",
        dropout=DROPOUT,
        qid_to_paraphrase_emb=qid_to_paraphrase_emb,
        qid_to_base_emb=qid_to_base_emb,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)

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


def context_emb_with_attention(
    LR: float = 0.0001,
    SAVE_EACH: int = 5000,
    BATCH_SIZE: int = 1024,
    VALIDATE_EACH: int = 1000,
    VALIDATION_SIZE: int = 10000,
    DROPOUT: float = 0.1,
) -> TrainingConfig:
    name = "context_emb_with_attention"

    paraphrase_embs, paraphrase_qids = load_embs_and_qids(
        "/lnet/work/home-students-external/farhan/troja/outputs/paraphrase_multilig_index"
    )
    paraphrase_embs = torch.from_numpy(paraphrase_embs)
    base_embs, base_qids = load_embs_and_qids(
        "/lnet/work/home-students-external/farhan/troja/outputs/workdirs/asi_se_to_rozbilo_init_all/damuel_for_index_6"
    )
    base_embs = torch.from_numpy(base_embs)

    dataset = RerankingIterableDataset()
    output_dir = "/lnet/work/home-students-external/farhan/troja/outputs/reranking_models"

    qid_to_paraphrase_emb = {qid: emb for qid, emb in zip(paraphrase_qids, paraphrase_embs)}
    qid_to_base_emb = {qid: emb for qid, emb in zip(base_qids, base_embs)}

    model = ContextEmbWithAttention(
        model_name_or_path="/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base",
        state_dict_path="/lnet/work/home-students-external/farhan/troja/outputs/workdirs/asi_se_to_rozbilo_init_all/models_5/ema.pth",
        dropout=DROPOUT,
        qid_to_paraphrase_emb=qid_to_paraphrase_emb,
        qid_to_base_emb=qid_to_base_emb,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, fused=True)

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


def get_config_from_name(config_name: str) -> TrainingConfig:
    if config_name == "pairwise_mlp":
        return pairwise_mlp()
    if config_name == "pairwise_mlp_debug":
        return pairwise_mlp(VALIDATE_EACH=1000, SAVE_EACH=1000000000000)
    if config_name == "pairwise_mlp_noise":
        return pairwise_mlp_noise()
    if config_name == "pairwise_mlp_noise_dropout":
        return pairwise_mlp_noise_dropout()
    if config_name == "pairwise_mlp_paraphrase":
        return pairwise_mlp_paraphrase()
    if config_name == "context_emb_with_attention":
        return context_emb_with_attention()
    if config_name == "full_lealla":
        return full_lealla()
    if config_name == "full_lealla_r":
        return full_lealla_r()
    if config_name == "full_lealla_debug":
        return full_lealla(VALIDATE_EACH=1000, SAVE_EACH=1000000000000, BATCH_SIZE=128)
    else:
        raise ValueError(f"Unknown training configuration: {config_name}")
