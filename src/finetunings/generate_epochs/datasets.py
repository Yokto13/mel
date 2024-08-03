from pathlib import Path
import pickle
import sys

from models.batch_sampler import BatchSampler

sys.stdout.reconfigure(line_buffering=True, write_through=True)

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from data_processors.index.token_index import TokenIndex
from utils.embeddings import create_attention_mask
from utils.multifile_dataset import MultiFileDataset
from utils.loaders import load_embs_qids_tokens


class TokensIterableDataset(IterableDataset):
    def __init__(self, dir_path: Path, known_qids):
        self.dir_path = dir_path
        data = load_embs_qids_tokens(dir_path)
        self.embs = data["embs"]
        self.qids = data["qids"]
        self.tokens = data["tokens"]
        self.known_qids = known_qids

    def __iter__(self):
        for embs, qid, tok in zip(self.embs, self.qids, self.tokens):
            if qid not in self.known_qids:
                continue
            yield embs, tok, qid


class DamuelNeighborsIterableDataset(IterableDataset):
    def __init__(
        self,
        batch_sampler: BatchSampler,
        tokenizer_embs_dataset: IterableDataset,
        batch_size: int,
        toks_size: int,
        positive_cnt: int,
        negative_cnt: int,
        model,
        device,
        tokens,
    ):
        self.batch_sampler = batch_sampler
        self.tokenizer_embs_dataset = tokenizer_embs_dataset
        self.batch_size = batch_size
        self.toks_size = toks_size
        self.positive_cnt = positive_cnt
        self.negative_cnt = negative_cnt
        self.model = model
        self.device = device
        self.sampler_qids = set(self.batch_sampler.qids)
        self.sampler_tokens = tokens

    def __iter__(self):
        per_mention = self.positive_cnt + self.negative_cnt
        self.model.to(self.device)
        for embs, qids, toks in self._batch_sampler():
            batch = np.zeros((self.batch_size, 2, self.toks_size), dtype=np.int64)
            together_line = np.zeros(
                (self.batch_size * per_mention, 2, self.toks_size), dtype=np.int64
            )
            batch_Y = np.zeros((self.batch_size, self.batch_size * per_mention))

            together_line_idx = 0

            positive, negative = self.batch_sampler.sample(
                embs, qids, self.negative_cnt
            )

            batch[:, 0] = toks
            batch[:, 1] = create_attention_mask(toks)

            for i in range(len(positive)):
                pos_idx, neg_ids = positive[i], negative[i]

                batch_Y[i, i * per_mention : i * per_mention + 1] = 1

                old_together_line_idx = together_line_idx

                together_line[together_line_idx][0] = self.sampler_tokens[pos_idx]
                together_line[together_line_idx][1] = create_attention_mask(
                    self.sampler_tokens[pos_idx]
                )
                together_line_idx += 1
                for neg_idx in neg_ids:
                    together_line[together_line_idx][0] = self.sampler_tokens[neg_idx]
                    together_line[together_line_idx][1] = create_attention_mask(
                        self.sampler_tokens[neg_idx]
                    )
                    together_line_idx += 1

                # Checks that the number of entities per mention is correct
                assert together_line_idx - old_together_line_idx == per_mention

            yield (
                torch.tensor(batch, dtype=torch.long),
                torch.tensor(together_line, dtype=torch.long),
                torch.tensor(batch_Y, dtype=torch.float32),
            )

    def _batch_sampler(self):
        toks, qids, embs = [], [], [], []
        for emb, qid, tok in self.tokenizer_embs_dataset:
            # This is not needed due to DaMuEL's structure
            # if qid not in self.index:
            # continue
            toks.append(tok)
            qids.append(qid)
            embs.append(emb)
            if len(toks) == self.batch_size:
                toks = np.stack(toks, axis=0)
                embs = np.array(embs)
                qids = np.array(qids)
                yield embs, qids, toks
                toks, qids, embs = [], [], []


class StatefulIterableDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset):
        self.dataset = dataset
        self._iterator = iter(dataset)

    def __iter__(self):
        for batch in self._iterator:
            yield batch
        self._iterator = iter(self.dataset)
