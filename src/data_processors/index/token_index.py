from collections import defaultdict
from dataclasses import dataclass
import lzma
from math import inf
import os
from pathlib import Path
import pickle

import numpy as np
import scann
import torch
from torch.utils.data import DataLoader
import tqdm

from data_processors.index.index import Index
from models.data.tokens_dataset import TokensDataset


class TokenIndex(Index):
    def __init__(
        self,
        embs,
        qids,
        tokens,
        attentions,
        scann_index=None,
        default_index_build=True,
        k=100,
    ):
        if default_index_build and scann_index is not None:
            raise ValueError("Cannot build index when index is already in arguments.")
        super().__init__(embs, qids, default_index_build=default_index_build)

        if scann_index is not None:
            self.scann_index = scann_index

        self.tokens = tokens
        self.attentions = attentions

        self.neighbor_selector = NeighborsSelector(
            len(embs), k, self.qids, self.scann_index
        )

        print(embs.shape, qids.shape, tokens.shape, attentions.shape)

    @classmethod
    def from_dir(cls, path, max_per_qid=inf):
        raise NotImplementedError

    @classmethod
    def from_dirs(cls, embs_path, tokens_path, mentions=False):
        if isinstance(embs_path, str):
            embs_path = Path(embs_path)
        if isinstance(tokens_path, str):
            tokens_path = Path(tokens_path)

        embs, qids, tokens, attentions = cls.load_embs_and_toks(
            embs_path, tokens_path, mentions=mentions
        )

        return cls(embs, qids, tokens, attentions)

    @classmethod
    def from_token_dir(cls, path, model, max_per_qid=inf):
        if isinstance(path, str):
            path = Path(path)
        entity_names = cls.load_entity_names(path)

        entity_names = cls.filter_based_on_max_per_qid(entity_names, max_per_qid)

        dataset = TokensDataset(entity_names)
        dataloader = DataLoader(dataset, batch_size=128, shuffle=False)

        device = (
            torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        )
        model.to(device)

        all_embs = []
        all_tokens = []
        all_attentions = []
        all_qids = []

        for batch in tqdm(dataloader):
            toks, att, qids = batch

            toks = toks.to(device)
            att = att.to(device)

            toks = toks.squeeze(1)
            att = att.squeeze(1)

            embs = model(toks, att).pooler_output.detach().cpu().numpy()

            toks = toks.detach().cpu().numpy()
            att = att.detach().cpu().numpy()

            all_embs.append(embs)
            all_tokens.append(toks)
            # print(toks.shape, att.shape, embs.shape)
            all_attentions.append(att)
            all_qids.extend(qids)

        all_embs = np.concatenate(all_embs)
        all_qids = np.array(all_qids)
        all_tokens = np.concatenate(all_tokens)
        all_attentions = np.concatenate(all_attentions)

        return cls(all_embs, all_qids, all_tokens, all_attentions)

    @classmethod
    def filter_based_on_max_per_qid(cls, entity_names, max_per_qid):
        qid_counter = defaultdict(int)
        wanted_ids = set()
        for pair in entity_names:
            if qid_counter[pair.qid] < max_per_qid:
                qid_counter[pair.qid] += 1
                wanted_ids.add(pair.qid)

        return [pair for pair in entity_names if pair.qid in wanted_ids]

    @classmethod
    def load_entity_names(cls, path):
        all = []
        for fn in sorted(path.iterdir()):
            if not cls.is_entity_names(fn):
                continue
            with lzma.open(path / fn, "rb") as f:
                entity_names = pickle.load(f)
            all.extend(entity_names)
        return all

    @classmethod
    def load_embs_and_toks(cls, embs_path, tokens_path, mentions=False):
        if mentions:
            print("Loading only mentions")

        def get_size_dim_context():
            size = 0
            context = None
            dim = None
            print(embs_path, mentions)
            for fn in sorted(tokens_path.iterdir()):
                if not mentions and not cls.is_entity_names(fn):
                    continue
                if mentions and not cls.is_mentions(fn):
                    continue
                if context is None:
                    entity_names = np.load(fn)
                    context = entity_names["tokens"].shape[1]
            embs_fn = embs_path / f"embs_qids.npz"
            embs = np.load(embs_fn)["embs"]
            size += len(embs)
            dim = embs.shape[1]
            return size, dim, context

        data_size, dim, context_size = get_size_dim_context()

        print(data_size, dim, context_size)

        embs = np.zeros((data_size, dim), dtype=np.float16)
        qids = np.zeros(data_size, dtype=np.int32)
        tokens = np.zeros((data_size, 1, context_size), dtype=np.int32)
        attentions = np.zeros((data_size, 1, context_size), dtype=np.int32)

        current = 0

        for fn in sorted(tokens_path.iterdir()):
            if not mentions and not cls.is_entity_names(fn):
                continue
            if mentions and not cls.is_mentions(fn):
                continue
            with open(fn, "rb") as f:
                entity_names = pickle.load(f)
            hash_str = cls.extract_hash(fn)
            embs_fn = embs_path / f"embs_{hash_str}.npy"
            qids_fn = embs_path / f"qids_{hash_str}.npy"

            embs_from_file = np.load(embs_fn)
            embs_from_file = embs_from_file.astype(np.float16)
            qids_from_file = np.load(qids_fn)
            t, a = list(
                zip(
                    *(
                        (
                            en.tokenization_output["input_ids"].numpy(),
                            en.tokenization_output["attention_mask"].numpy(),
                        )
                        for en in entity_names
                    )
                )
            )

            embs[current : current + len(embs_from_file)] = embs_from_file
            qids[current : current + len(embs_from_file)] = qids_from_file
            tokens[current : current + len(embs_from_file)] = np.array(t)
            attentions[current : current + len(embs_from_file)] = np.array(a)

            current += len(embs_from_file)

        print("Loaded", current, "items")

        return embs, qids, tokens, attentions

    @classmethod
    def from_iterable_and_model(cls, dataloader, model, max_per_qid=inf):
        raise NotImplementedError

    @classmethod
    def is_entity_names(cls, fn):
        return fn.name.startswith("entity_names_")

    @classmethod
    def is_mentions(cls, fn):
        return fn.name.startswith("mentions_")

    @classmethod
    def from_saved(cls, path):
        if isinstance(path, str):
            path = Path(path)
        scann_index = scann.scann_ops_pybind.load_searcher(str(path / "index/"))
        data = np.load(path / "data.npz")
        embs = data["embs"]
        qids = data["qids"]
        tokens = data["tokens"]
        attentions = data["attentions"]
        qids = np.array(qids)
        tokens = np.array(tokens)
        attentions = np.array(attentions)
        embs = np.array(embs)
        return cls(
            embs, qids, tokens, attentions, scann_index, default_index_build=False
        )

    def query(self, query_emb, qid, positive_cnt=3, neg_cnt=5) -> tuple[list, list]:
        query_emb = self.get_query_in_searchable_format(query_emb)  # batch + norm

        pos_neighbors_inds, neg_neighbors_inds = (
            self.neighbor_selector.exponential_pos_neg_indices(
                query_emb, [qid], positive_cnt, neg_cnt
            )
        )

        pos_neighbors_inds = pos_neighbors_inds[0]
        neg_neighbors_inds = neg_neighbors_inds[0]

        assert (
            len(pos_neighbors_inds) + len(neg_neighbors_inds) == positive_cnt + neg_cnt
        )

        return (
            [self.tokens[pos_neighbors_inds], self.attentions[pos_neighbors_inds]],
            [self.tokens[neg_neighbors_inds], self.attentions[neg_neighbors_inds]],
        )

    def query_batched(
        self, query_embs, qids, positive_cnt=3, neg_cnt=5
    ) -> list[tuple[list, list]]:
        query_embs = self.get_query_in_searchable_format(query_embs)  # batch + norm

        pos_neighbors_inds, neg_neighbors_inds = (
            self.neighbor_selector.exponential_pos_neg_indices(
                query_embs, qids, positive_cnt, neg_cnt
            )
        )

        for i in range(len(pos_neighbors_inds)):
            assert (
                len(pos_neighbors_inds[i]) + len(neg_neighbors_inds[i])
                == positive_cnt + neg_cnt
            )
        return [
            (
                [
                    self.tokens[pos_neighbors_inds[i]],
                    self.attentions[pos_neighbors_inds[i]],
                ],
                [
                    self.tokens[neg_neighbors_inds[i]],
                    self.attentions[neg_neighbors_inds[i]],
                ],
            )
            for i in range(len(pos_neighbors_inds))
        ]

    def save(self, path):
        # make sure the index is active (based on Github issue)
        self.query(self.embs[0], self.qids[0], 1, 1)
        if isinstance(path, str):
            path = Path(path)
        os.makedirs(str(path / "index/"), exist_ok=True)
        self.scann_index.serialize(str(path / "index/"))

        np.savez_compressed(
            path / "data.npz",
            embs=self.embs,
            qids=self.qids,
            tokens=self.tokens,
            attentions=self.attentions,
        )

    def get_query_in_searchable_format(self, query_emb):
        if len(query_emb.shape) == 1:
            query_emb = np.expand_dims(query_emb, 0)
        query_emb = np.array(query_emb)
        query_emb = query_emb / np.linalg.norm(query_emb, axis=1, keepdims=True)
        return query_emb

    def __len__(self):
        return len(self.embs)


@dataclass
class NeighborsSelector:
    embs_cnt: int
    start_num_neighbors: int
    qids: np.ndarray
    scann_index: scann.scann_ops_pybind

    def __post_init__(self):
        self.qid_to_index = defaultdict(list)
        for i, qid in enumerate(self.qids):
            self.qid_to_index[qid].append(i)

    def exponential_pos_neg_indices(
        self, query_embs, query_qids, positive_cnt, negative_cnt
    ):
        expected_length = positive_cnt + negative_cnt

        pos_neighbors_inds = [[] for _ in range(len(query_embs))]
        neg_neighbors_inds = [[] for _ in range(len(query_embs))]

        num_neighbors = self.start_num_neighbors
        while self._exists_with_not_enough_neighbors(
            pos_neighbors_inds, neg_neighbors_inds, expected_length
        ):
            pos_neighbors_inds, neg_neighbors_inds = self._attempt_to_get_inds(
                num_neighbors, query_embs, query_qids, positive_cnt, negative_cnt
            )
            num_neighbors = self._increase_neighbors(num_neighbors)
        return pos_neighbors_inds, neg_neighbors_inds

    def _increase_neighbors(self, num_neighbors):
        if num_neighbors == self.embs_cnt:
            # cannot increase any more
            raise ValueError("Could not find enough neighbors")
        return min(num_neighbors * 2, self.embs_cnt)

    def _exists_with_not_enough_neighbors(self, pos, neg, expected_length):
        for p, n in zip(pos, neg):
            if len(p) + len(n) != expected_length:
                return True
        return False

    def _attempt_to_get_inds(
        self, num_neighbors, query_embs, query_qids, positive_cnt, negative_cnt
    ):
        neighbors, _ = self.scann_index.search_batched(
            query_embs, final_num_neighbors=num_neighbors
        )

        pos_neighbors_inds, neg_neighbors_inds = [], []
        for i in range(len(query_embs)):
            p, n = self._get_pos_neg_indices(
                neighbors[i], query_qids[i], positive_cnt, negative_cnt
            )
            pos_neighbors_inds.append(p)
            neg_neighbors_inds.append(n)
        return pos_neighbors_inds, neg_neighbors_inds

    def _get_pos_neg_indices(self, neighbors, qid, positive_cnt, neg_cnt):
        pos_neighbors_inds, neg_neighbors_inds = self._get_pos_neg_indices_from_scann(
            neighbors, qid, positive_cnt
        )
        # If we don't sample, the top neg_cnt are overused
        # Sampling uniquely is costly, so we migth try sampling with replacement
        # neg_neighbors_inds = [neg_neighbors_inds[i] for i in np.random.randint(0, len(neg_neighbors_inds), neg_cnt)]
        np.random.shuffle(neg_neighbors_inds)
        if len(pos_neighbors_inds) < positive_cnt:
            missing_pos = positive_cnt - len(pos_neighbors_inds)
            pos_neighbors_no_sim = self._try_sampling_pos_without_sim(qid, missing_pos)
            pos_neighbors_inds = np.concatenate(
                [pos_neighbors_inds, pos_neighbors_no_sim]
            )
        if len(pos_neighbors_inds) < positive_cnt:
            missing_pos = positive_cnt - len(pos_neighbors_inds)
            neg_neighbors_inds = neg_neighbors_inds[: missing_pos + neg_cnt]
        else:
            neg_neighbors_inds = neg_neighbors_inds[:neg_cnt]
        return pos_neighbors_inds.astype(int), neg_neighbors_inds.astype(int)

    def _get_pos_neg_indices_from_scann(self, neighbors, qid, positive_cnt):
        pos_mask = self.qids[neighbors] == qid
        neg_mask = ~pos_mask

        # Neg sampling as described in the thesis:
        neg_qids = self.qids[neighbors][neg_mask]
        indices_of_unique_neg_qids = np.unique(neg_qids, return_index=True)[1]

        return neighbors[pos_mask][:positive_cnt], neighbors[indices_of_unique_neg_qids]

    def _try_sampling_pos_without_sim(self, qid, wanted):
        options = self.qid_to_index[qid]
        np.random.shuffle(options)
        return options[:wanted]
