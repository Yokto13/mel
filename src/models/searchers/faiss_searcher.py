"""FAISS-based searcher with GPU support."""

import math

import faiss
import numpy as np

from models.searchers.searcher import Searcher


class FaissSearcher(Searcher):
    def __init__(self, embs: np.ndarray, results: np.ndarray):
        super().__init__(embs, results, False)
        self.gpu_index = None
        self.is_trained = False

        # Check GPU availability
        self.ngpu = faiss.get_num_gpus()
        if self.ngpu == 0:
            raise RuntimeError("No GPU detected by Faiss")

        # IVFPQ parameters
        self.d = embs.shape[1]  # vector dimension
        self.nb = embs.shape[0]  # number of vectors to index
        self.nlist = int(4 * math.sqrt(self.nb))  # rule-of-thumb: 4 × √N
        self.m = 16 if self.d >= 16 else self.d  # PQ subvectors (must divide d)
        self.nbits = 8  # 8-bit codes
        self.nprobe = 10  # number of clusters to search

        self.build()

    def find(self, batch, num_neighbors) -> np.ndarray:
        if self.gpu_index is None:
            raise RuntimeError("Index not built. Call build() first.")

        if not self.is_trained:
            raise RuntimeError("Index not trained. Call build() first.")

        # Ensure batch is float32 and 2D
        query = np.asarray(batch, dtype=np.float32)
        if query.ndim == 1:
            query = query.reshape(1, -1)

        # Perform search
        distances, indices = self.gpu_index.search(query, num_neighbors)

        # Return results for the queries
        return self.results[indices]

    def build(self):
        self.build_index()

    def build_index(self):
        print(f"Building FAISS GPU index with {self.ngpu} GPU(s)...")

        # Ensure embeddings are float32
        embs_f32 = self.embs.astype(np.float32)

        # Create CPU index first
        cpu_quantizer = faiss.IndexFlatL2(self.d)
        cpu_index = faiss.IndexIVFPQ(
            cpu_quantizer, self.d, self.nlist, self.m, self.nbits
        )

        # Convert to GPU
        if self.ngpu == 1:
            print("Using single GPU mode")
            res = faiss.StandardGpuResources()
            self.gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        else:
            print(f"Using multi-GPU mode with {self.ngpu} GPUs")
            co = faiss.GpuMultipleClonerOptions()
            co.shard = True  # one shard per GPU
            self.gpu_index = faiss.index_cpu_to_all_gpus(cpu_index, co)

        # Train the index (required for IVFPQ)
        print("Training the IVFPQ index...")
        # Use subset of data for training if dataset is large
        training_size = min(50000, self.nb)
        training_data = embs_f32[:training_size]
        self.gpu_index.train(training_data)
        self.is_trained = True
        print("Training completed.")

        # Set search parameters
        self.gpu_index.nprobe = self.nprobe

        # Add all vectors to the index
        self.gpu_index.add(embs_f32)
        print(f"Added {self.gpu_index.ntotal} vectors to GPU IVFPQ index.")

        print("FAISS GPU index build completed.")
