import numpy as np
from models.negative_sampler import NegativeSampler, NegativeSamplingType
from models.searchers.searcher import Searcher


class BatchSampler:
    def __init__(
        self,
        embs: np.ndarray,
        qids: np.ndarray,
        negative_searcher_constructor: type[Searcher],
        negative_sampling_type: NegativeSamplingType,
        **negative_sampler_kwargs,
    ) -> None:
        self.embs = embs
        self.qids = qids
        self.negative_sampler = NegativeSampler(
            embs,
            qids,
            negative_searcher_constructor,
            negative_sampling_type,
            **negative_sampler_kwargs,
        )
        self.qids_to_idx = {qid: i for i, qid in enumerate(qids)}

    def sample(
        self, batch_embs, batch_qids, negative_cnt
    ) -> tuple[np.ndarray, np.ndarray]:
        negative = self.negative_sampler.sample(batch_embs, batch_qids, negative_cnt)
        positive = np.array([self.qids_to_idx[qid] for qid in batch_qids])
        return positive, negative
