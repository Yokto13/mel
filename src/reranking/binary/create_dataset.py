import sys
from pathlib import Path

import numba as nb
import numpy as np

import torch
import torch.utils.data

from tqdm import tqdm

sys.path.append("/lnet/work/home-students-external/farhan/mel-reborn/src")

from models.searchers.brute_force_searcher import BruteForceSearcher
from utils.embeddings import create_attention_mask
from utils.loaders import load_embs_and_qids, load_mentions_from_dir
from utils.model_factory import ModelFactory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@nb.njit
def get_neg_qids(top_qids, batch_qids):
    neg_qids = []
    for row in top_qids:
        if row[0] not in batch_qids:
            neg_qids.append(row[0])
        else:
            neg_qids.append(row[1])
    return neg_qids


def create_binary_dataset(
    index_embs_dir: Path,
    index_tokens_path: Path,
    link_tokens_path: Path,
    model_name: str,
    embedding_model_path_dict: Path,
    output_path: Path,
    target_dim: int = None,
    batch_size: int = 512,
) -> None:
    # Load index embeddings, qids, and tokens
    index_embs, index_qids = load_embs_and_qids(index_embs_dir)
    index_qids_set = set(index_qids)
    index_embs = index_embs.astype(np.float16)
    index_tokens, _ = load_mentions_from_dir(index_tokens_path)
    print(index_tokens.shape)
    print(len(index_qids_set))

    # Create BruteForceSearcher
    searcher = BruteForceSearcher(index_embs, index_qids)

    # Load link tokens and qids
    link_tokens, link_qids = load_mentions_from_dir(link_tokens_path)
    # Loaders order by qids which is not necessarily what we want
    print(link_tokens.shape)
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
    dataset = torch.utils.data.Subset(
        dataset, [i for i, (tokens, qid) in enumerate(dataset) if qid in index_qids_set]
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True
    )

    # Initialize dataset arrays
    description_tokens = []
    link_tokens_list = []
    y = []

    print("Dataset length:", len(dataset))
    description_tokens = np.zeros((len(dataset) * 2, index_tokens.shape[1]))
    link_tokens_list = np.zeros((len(dataset) * 2, link_tokens.shape[1]))
    y = np.zeros((len(dataset) * 2,))
    output_index = 0

    index_qid_to_index = {qid: i for i, qid in enumerate(index_qids)}

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

        positive_mask = [index_qid_to_index[qid] for qid in batch_qids.numpy()]
        data_size = len(batch_tokens)
        description_tokens[output_index : output_index + data_size] = index_tokens[
            positive_mask
        ]
        link_tokens_list[output_index : output_index + data_size] = batch_tokens.numpy()
        y[output_index : output_index + data_size] = 1

        output_index += data_size

        neg_qids = get_neg_qids(top_qids, set(batch_qids.numpy()))

        negative_mask = [index_qid_to_index[qid] for qid in neg_qids]
        description_tokens[output_index : output_index + data_size] = index_tokens[
            negative_mask
        ]
        link_tokens_list[output_index : output_index + data_size] = batch_tokens.numpy()
        y[output_index : output_index + data_size] = 0

        output_index += data_size

    # Convert to numpy arrays

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
