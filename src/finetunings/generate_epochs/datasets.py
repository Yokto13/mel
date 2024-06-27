from pathlib import Path
import pickle
import sys

sys.stdout.reconfigure(line_buffering=True, write_through=True)

import numpy as np
import torch
from torch.utils.data import DataLoader, IterableDataset

from data_processors.index.token_index import TokenIndex


class TokensIterableDataset(IterableDataset):
    def __init__(self, dir_path: Path, input_type="entity_names"):
        self.dir_path = dir_path
        self.worker_info = torch.utils.data.get_worker_info()
        self.type = input_type

    def __iter__(self):
        for fn in sorted(self.dir_path.iterdir()):
            if not fn.name.startswith(self.type):
                continue
            with open(self.dir_path / fn.name, "rb") as f:
                entity_names = pickle.load(f)
            for pair in entity_names:
                toks = pair.tokenization_output["input_ids"]
                att = pair.tokenization_output["attention_mask"]
                yield toks[0], att[0], pair.qid


class TokensEmbsIterableDataset(IterableDataset):
    def __init__(
        self,
        tokens_dataset: TokensIterableDataset,
        model,
        device,
        batch_size_embs: int = 4096,
    ):
        self.device = device
        self.tokens_dataset = tokens_dataset
        self.model = model
        self.model.to(device)
        self.data_loader = DataLoader(tokens_dataset, batch_size=batch_size_embs)

    def __iter__(self):
        for toks, atts, qids in self.data_loader:
            toks = toks.to(self.device)
            atts = atts.to(self.device)
            with torch.no_grad():
                embs = self.model(toks, atts).pooler_output.detach().cpu().numpy()
            toks = toks.detach().cpu().numpy()
            atts = atts.detach().cpu().numpy()
            for i in range(len(qids)):
                yield toks[i], atts[i], qids[i], embs[i]


class DamuelNeighborsIterableDataset(IterableDataset):
    def __init__(
        self,
        index: TokenIndex,
        tokenizer_embs_dataset: IterableDataset,
        batch_size: int,
        toks_size: int,
        positive_cnt: int,
        negative_cnt: int,
        model,
        device,
    ):
        self.index = index
        self.tokenizer_embs_dataset = tokenizer_embs_dataset
        self.batch_size = batch_size
        self.toks_size = toks_size
        self.positive_cnt = positive_cnt
        self.negative_cnt = negative_cnt
        self.model = model
        self.device = device

    def __iter__(self):
        per_mention = self.positive_cnt + self.negative_cnt
        self.model.to(self.device)
        for toks, atts, qids, embs in self._batch_sampler():
            batch = np.zeros((self.batch_size, 2, self.toks_size), dtype=np.int64)
            together_line = np.zeros(
                (self.batch_size * per_mention, 2, self.toks_size), dtype=np.int64
            )
            batch_Y = np.zeros((self.batch_size, self.batch_size * per_mention))

            together_line_idx = 0

            neighbors_batched = self.index.query_batched(
                embs, qids, positive_cnt=self.positive_cnt, neg_cnt=self.negative_cnt
            )

            batch[:, 0] = toks
            batch[:, 1] = atts

            for i in range(len(neighbors_batched)):
                pos_toks_atts, neg_toks_atts = neighbors_batched[i]

                if (
                    len(pos_toks_atts[0]) == 0
                ):  # prevents case when there is no positive, which messes up training
                    # will not happen in DaMuEL
                    pos_toks_atts = ([toks[i]], [atts[i]])

                    # drop last negative to preserve the per_mention_count
                    neg_toks_atts[0] = neg_toks_atts[0][:-1]
                    neg_toks_atts[1] = neg_toks_atts[1][:-1]

                batch_Y[
                    i, i * per_mention : i * per_mention + len(pos_toks_atts[0])
                ] = 1 / len(pos_toks_atts[0])

                old_together_line_idx = together_line_idx

                for pos in zip(*pos_toks_atts):
                    for k in range(2):
                        together_line[together_line_idx][k] = pos[k]
                    together_line_idx += 1
                for neg in zip(*neg_toks_atts):
                    for k in range(2):
                        together_line[together_line_idx][k] = neg[k]
                    together_line_idx += 1

                # Checks that the number of entities per mention is correct
                assert together_line_idx - old_together_line_idx == per_mention

            yield (
                torch.tensor(batch, dtype=torch.long),
                torch.tensor(together_line, dtype=torch.long),
                torch.tensor(batch_Y, dtype=torch.float32),
            )

    def _batch_sampler(self):
        toks, atts, qids, embs = [], [], [], []
        for tok, att, qid, emb in self.tokenizer_embs_dataset:
            # This is not needed due to DaMuEL's structure
            # if qid not in self.index:
            # continue
            toks.append(tok)
            atts.append(att)
            qids.append(qid)
            embs.append(emb)
            if len(toks) == self.batch_size:
                toks = np.stack(toks, axis=0)
                atts = np.stack(atts, axis=0)
                embs = np.array(embs)
                qids = np.array(qids)
                yield toks, atts, qids, embs
                toks, atts, qids, embs = [], [], [], []


class StatefulIterableDataset(IterableDataset):
    def __init__(self, dataset: IterableDataset):
        self.dataset = dataset
        self._iterator = iter(dataset)

    def __iter__(self):
        for batch in self._iterator:
            yield batch
        self._iterator = iter(self.dataset)
