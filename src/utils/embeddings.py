"""Utils for embedding tokens."""

from collections.abc import Generator
import itertools
import logging
from pathlib import Path

import gin
import numpy as np
import torch
from tqdm import tqdm

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertModel

from utils.model_builder import ModelBuilder, ModelOutputType
from utils.model_factory import ModelFactory
from utils.multifile_dataset import MultiFileDataset

_logger = logging.getLogger("utils.embeddings")


def create_attention_mask(toks, padding_value=0):
    if isinstance(toks, torch.Tensor):
        return (toks != padding_value).long()
    elif isinstance(toks, np.ndarray):
        return (toks != padding_value).astype(np.int64)
    else:
        raise TypeError("Input must be a PyTorch tensor or NumPy array")


class _Processer:
    def __init__(self, each) -> None:
        self._currently = 0
        self._next_print_gen = iter(itertools.count(step=each))
        self._next_print = next(self._next_print_gen)

    def log(self, current_iter_items_cnt):
        self._currently += current_iter_items_cnt
        if self._should_log():
            self._log()
            self._next_print = next(self._next_print_gen)

    def _should_log(self):
        return self._currently >= self._next_print

    def _log(self):
        _logger.info(f"{self._currently} elements processed.")


def embed(
    dataset,
    model,
    batch_size=(16384 * 4),
    return_qids=True,
    return_tokens=False,
) -> list[np.ndarray]:
    """embeds dataset and returns embeddings, qids tuple.

    Note that all embeddings are held in memory at once which can consume a lot of RAM.
    TODO: We could implement some incremental saving.

    Args:
        dataset (Dataset/IterableDataset):
        model: Model for embedding the dataset, last pooling_layer is used
        batch_size (int, optional): Defaults to (16384 * 4).
        return_qids (bool, optional): Defaults to True.
        return_tokens (bool, optional): Defaults to False.

    Returns:
        list[np.arr...]: First element corresponds to embeddings,
            then qids then tokens follow (in this order) if requested.
    """
    model.eval()

    # We usually work with IterableDataset subclass so no multiprocessing
    data_loader = DataLoader(dataset, batch_size, num_workers=0)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    embeddings = []

    if return_tokens:
        tokens = []
    if return_qids:
        qids = []

    log_processer = _Processer(each=10**6)

    scaler = torch.cuda.amp.GradScaler()

    with torch.no_grad():
        for batch_toks, batch_qids in data_loader:
            batch_toks = batch_toks.to(torch.int64)
            attention_mask = create_attention_mask(batch_toks)
            if torch.cuda.is_available():
                batch_toks = batch_toks.cuda()
                attention_mask = attention_mask.cuda()

            with torch.amp.autocast():
                batch_embeddings = model(batch_toks, attention_mask)

            batch_embeddings = batch_embeddings.cpu().numpy().astype(np.float16)
            batch_embeddings = batch_embeddings / np.linalg.norm(
                batch_embeddings, ord=2, axis=1, keepdims=True
            )
            embeddings.extend(batch_embeddings)
            if return_tokens:
                tokens.extend(batch_toks.cpu().numpy())
            if return_qids:
                qids.extend(batch_qids)
            log_processer.log(len(batch_qids))
    res = [np.array(embeddings)]
    if return_qids:
        res.append(np.array(qids))
    if return_tokens:
        res.append(np.array(tokens))
    return res


def embed_generator(
    dataset,
    model,
    batch_size=262144,
) -> Generator[tuple[np.ndarray, np.ndarray, np.ndarray], None, None]:
    """embeds dataset and returns embeddings, qids tuple.

    Note that all embeddings are held in memory at once which can consume a lot of RAM.
    TODO: We could implement some incremental saving.

    Args:
        dataset (Dataset/IterableDataset):
        model: Model for embedding the dataset, last pooling_layer is used
        batch_size (int, optional): Defaults to (16384 * 4).
        return_qids (bool, optional): Defaults to True.
        return_tokens (bool, optional): Defaults to False.

    Returns:
        list[np.arr...]: First element corresponds to embeddings,
            then qids then tokens follow (in this order) if requested.
    """
    model.eval()

    # We usually work with IterableDataset subclass so no multiprocessing
    data_loader = DataLoader(dataset, batch_size, num_workers=0)

    if torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    log_processer = _Processer(each=10**6)

    with torch.no_grad():
        for batch_toks, batch_qids in data_loader:
            batch_toks = batch_toks.to(torch.int64)
            attention_mask = create_attention_mask(batch_toks)
            if torch.cuda.is_available():
                batch_toks = batch_toks.cuda()
                attention_mask = attention_mask.cuda()

            with torch.amp.autocast():
                batch_embeddings = model(batch_toks, attention_mask)

            batch_embeddings = batch_embeddings.cpu().numpy().astype(np.float16)
            batch_embeddings = batch_embeddings / np.linalg.norm(
                batch_embeddings, ord=2, axis=1, keepdims=True
            )
            yield batch_embeddings, batch_qids.cpu().numpy(), batch_toks.cpu().numpy()
            log_processer.log(len(batch_qids))


def get_embs_and_qids(source_dir: Path, model: nn.Module, batch_size=16384):
    """
    Given a directory with (multiple) token_qid npz files embedds them with model and returns
    tuple (embs, qids) of two very large arrays.
    """
    dataset = MultiFileDataset(source_dir)
    embs, qids = embed(dataset, model, batch_size)
    return embs, qids


@gin.configurable
def embs_from_tokens_and_model_name(
    source, model_name, batch_size, dest, output_type: str | None = None
):
    builder = ModelBuilder(model_name)
    output_type = (
        ModelOutputType(output_type)
        if output_type is not None
        else ModelOutputType.PoolerOutput
    )
    builder.set_output_type(output_type)
    model = builder.build()
    embs_from_tokens_and_model(source, model, batch_size, dest)


@gin.configurable
def embs_from_tokens_model_name_and_state_dict(
    source_path: str,
    model_name: str,
    batch_size: int,
    dest_path: str,
    state_dict_path: str | None,
    target_dim: int | None = None,
    output_type: str | None = None,
):
    model = ModelFactory.auto_load_from_file(model_name, state_dict_path, target_dim)
    embs_from_tokens_and_model(source_path, model, batch_size, dest_path)


def embs_from_tokens_and_model(source, model, batch_size, dest):
    embs, qids = get_embs_and_qids(Path(source), model, batch_size)
    np.savez_compressed(f"{dest}/embs_qids.npz", embs=embs, qids=qids)
