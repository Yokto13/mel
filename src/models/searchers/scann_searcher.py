import numpy as np

from models.searchers.searcher import Searcher


class ScaNNSearcher(Searcher):
    def __init__(
        self, embs: np.ndarray, results: np.ndarray, run_build_from_init: bool = True
    ):
        super().__init__(embs, results, run_build_from_init)

    def find(self, batch, num_neighbors) -> np.ndarray:
        neighbors, _ = self.searcher.search_batched(
            batch, final_num_neighbors=num_neighbors
        )
        return self.results[neighbors]

    def build(self):
        self.build_index()

    def build_index(
        self,
        num_leaves=5000,
        num_leaves_to_search=100,
        training_sample_size=10**6,
        reordering_size=2000,
        use_assymetric_hashing=True,
    ):
        import scann

        training_sample_size = int(min(0.5 * len(self.embs), training_sample_size))
        num_leaves = min(num_leaves, training_sample_size)
        n_of_clusters = min(
            training_sample_size, 100
        )  # so we can test with tiny datasets
        builder = scann.scann_ops_pybind.builder(
            self.embs, n_of_clusters, "dot_product"
        ).tree(
            num_leaves=num_leaves,
            num_leaves_to_search=num_leaves_to_search,
            training_sample_size=training_sample_size,
            # soar_lambda=1.5,
            # overretrieve_factor=2.0,
        )
        if use_assymetric_hashing:
            builder = builder.score_ah(2, anisotropic_quantization_threshold=0.2)
        else:
            builder = builder.score_brute_force(quantize=True)
        self.searcher = builder.reorder(reordering_size).build()
