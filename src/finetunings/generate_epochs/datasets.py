from pathlib import Path
import sys

from models.batch_sampler import BatchSampler

sys.stdout.reconfigure(line_buffering=True, write_through=True)

import numpy as np
import numpy.typing as npt
from torch.utils.data import DataLoader, IterableDataset, Dataset

from utils.loaders import load_embs_qids_tokens


class TokensIterableDataset(IterableDataset):
    def __init__(self, dir_path: Path, known_qids: npt.NDArray[np.int_]):
        self.dir_path = dir_path
        self.embs, self.qids, self.tokens = load_embs_qids_tokens(dir_path)
        self.known_qids = known_qids

    def __iter__(self):
        for embs, qid, tok in zip(self.embs, self.qids, self.tokens):
            if qid not in self.known_qids:
                continue
            yield embs, qid, tok


def _numpy_collate(batch):
    if isinstance(batch[0], np.ndarray):
        return np.stack(batch)
    elif isinstance(batch[0], (tuple, list)):
        transposed = zip(*batch)
        return [_numpy_collate(samples) for samples in transposed]
    else:
        return np.array(batch)


class DamuelNeighborsIterator:
    def __init__(
        self,
        dataset: Dataset | IterableDataset,
        batch_size: int,
        neg_cnt: int,
        sampler: BatchSampler,
        sampler_tokens: npt.NDArray[np.int_],
        toks_size: int,
    ) -> None:
        self.dataloader = DataLoader(dataset, batch_size, collate_fn=_numpy_collate)
        self.batch_size = batch_size
        self.negative_cnt = neg_cnt
        self.batch_sampler = sampler
        self.sampler_tokens = sampler_tokens
        self.toks_size = toks_size

    def __iter__(self):
        per_mention = 1 + self.negative_cnt
        for embs, qids, toks in self.dataloader:
            batch = toks
            together_line = np.zeros(
                (self.batch_size * per_mention, self.toks_size), dtype=np.int32
            )
            batch_Y = np.zeros((self.batch_size, self.batch_size * per_mention))

            together_line_idx = 0

            positive, negative = self.batch_sampler.sample(
                embs, qids, self.negative_cnt
            )

            for i in range(len(embs)):
                pos_idx, neg_ids = positive[i], negative[i]

                batch_Y[i, i * per_mention] = 1

                together_line[together_line_idx] = self.sampler_tokens[pos_idx]
                together_line_idx += 1

                together_line[together_line_idx : together_line_idx + len(neg_ids)] = (
                    self.sampler_tokens[neg_ids]
                )
                together_line_idx += len(neg_ids)

            yield batch, together_line, batch_Y
