import numpy as np
from pathlib import Path

import torch
import torch.utils.data

import sys

from tqdm import tqdm

sys.path.append("/lnet/work/home-students-external/farhan/mel-reborn/src")

from models.searchers.brute_force_searcher import BruteForceSearcher
from utils.model_factory import ModelFactory
from utils.loaders import load_embs_and_qids, load_mentions_from_dir
from utils.embeddings import create_attention_mask

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_binary_dataset(
    index_embs_dir: Path,
    index_tokens_path: Path,
    link_tokens_path: Path,
    model_name: str,
    embedding_model_path_dict: Path,
    output_path: Path,
    target_dim: int = None,
    batch_size: int = 1024,
) -> None:
    # Load index embeddings, qids, and tokens
    index_embs, index_qids = load_embs_and_qids(index_embs_dir)
    index_embs = index_embs.astype(np.float16)
    index_tokens, _ = load_mentions_from_dir(index_tokens_path)

    # Create BruteForceSearcher
    searcher = BruteForceSearcher(index_embs, index_qids)

    # Load link tokens and qids
    link_tokens, link_qids = load_mentions_from_dir(link_tokens_path)
    # Loaders order by qids which is not necessarily what we want
    p = np.random.permutation(len(link_tokens))
    link_tokens = link_tokens[p]
    link_qids = link_qids[p]

    # Load embedding model
    model = ModelFactory.auto_load_from_file(
        model_name,
        embedding_model_path_dict,
        target_dim=target_dim,
    )
    model.eval()
    model.to(device)
    # Create DataLoader
    dataset = list(zip(link_tokens, link_qids))
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False
    )

    # Initialize dataset arrays
    description_tokens = []
    link_tokens_list = []
    y = []

    # Iterate over batches
    for batch_tokens, batch_qids in tqdm(
        dataloader, desc="Creating dataset", total=len(dataloader)
    ):
        # Embed link tokens
        with torch.no_grad():
            batch_embs = model(
                batch_tokens.to(device).to(torch.int64),
                create_attention_mask(batch_tokens).to(device),
            ).cpu()

        # Find top matches
        top_qids = searcher.find(batch_embs.numpy().astype(np.float16), num_neighbors=2)

        for tokens, qid, top_qid in zip(batch_tokens, batch_qids, top_qids):
            # Add positive example
            if qid not in index_qids:
                continue
            description_tokens.append(index_tokens[index_qids == qid][0])
            link_tokens_list.append(tokens)
            y.append(1)

            # Add negative example
            neg_qid = top_qid[top_qid != qid][0]
            description_tokens.append(index_tokens[index_qids == neg_qid][0])
            link_tokens_list.append(tokens)
            y.append(0)

    # Convert to numpy arrays
    description_tokens = np.array(description_tokens)
    link_tokens_list = np.array(link_tokens_list)
    y = np.array(y)

    print(description_tokens.shape)
    print(link_tokens_list.shape)
    print(y.shape)

    # Save dataset
    np.savez(
        output_path,
        description_tokens=description_tokens,
        link_tokens=link_tokens_list,
        y=y,
    )


if __name__ == "__main__":
    index_embs_dir = Path(
        "/lnet/work/home-students-external/farhan/troja/outputs/workdirs/ml9/damuel_for_index_3"
    )
    index_tokens_path = Path(
        "/lnet/work/home-students-external/farhan/troja/outputs/workdirs/ml9/damuel_descs_together_tokens"
    )
    link_tokens_path = Path(
        "/lnet/work/home-students-external/farhan/troja/outputs/workdirs/ml9/damuel_links_together_tokens_0"
    )
    embedding_model_path = Path(
        "/lnet/work/home-students-external/farhan/troja/outputs/workdirs/ml9/models_2/final.pth"
    )
    output_path = Path(
        "/lnet/work/home-students-external/farhan/troja/outputs/reranking_test/reranker_dataset.npz"
    )
    model_name = (
        "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base"
    )

    create_binary_dataset(
        index_embs_dir,
        index_tokens_path,
        link_tokens_path,
        model_name,
        embedding_model_path,
        output_path,
    )
