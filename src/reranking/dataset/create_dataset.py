from pathlib import Path

import numpy as np
import torch
import torch.utils.data
from tqdm import tqdm

from models.searchers.brute_force_searcher import BruteForceSearcher, DPBruteForceSearcherPT
from utils.embeddings import create_attention_mask
from utils.loaders import load_embs_and_qids, load_tokens_qids, load_tokens_qids_from_dir
from utils.model_factory import ModelFactory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_neg_qids(top_qids, batch_qids):
    neg_qids = []
    for row, batch_qid in zip(top_qids, batch_qids):
        if row[0] != batch_qid:
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
    index_embs = index_embs.astype(np.float16)
    index_tokens, index_qids_from_tokens = load_tokens_qids_from_dir(index_tokens_path)

    # Sort index_embs and index_qids based on index_qids
    sort_indices = np.argsort(index_qids)
    index_qids = index_qids[sort_indices]
    index_embs = index_embs[sort_indices]

    sort_indices_tokens = np.argsort(index_qids_from_tokens)
    index_qids_from_tokens = index_qids_from_tokens[sort_indices_tokens]
    index_tokens = index_tokens[sort_indices_tokens]

    np.testing.assert_array_equal(index_qids, index_qids_from_tokens)

    print(index_tokens.shape)

    # Create BruteForceSearcher
    searcher = BruteForceSearcher(index_embs, index_qids)

    # Load link tokens and qids
    link_tokens_path = Path(link_tokens_path)
    output_path = Path(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    link_files = sorted(
        [p for p in link_tokens_path.iterdir() if p.is_file() and p.suffix == ".npz"],
        key=lambda p: p.name,
    )

    if not link_files:
        raise FileNotFoundError(f"No .npz files found in {link_tokens_path}")

    # Load embedding model
    model = ModelFactory.auto_load_from_file(
        model_name,
        embedding_model_path_dict,
        target_dim=target_dim,
    )
    model.eval()
    model.to(device)
    model.to(torch.bfloat16)
    model = torch.compile(model)

    index_qid_to_index = {int(qid): i for i, qid in enumerate(index_qids)}
    index_qids_set = set(index_qid_to_index.keys())

    for link_file in tqdm(link_files, desc="Processing link files"):
        link_tokens, link_qids = load_tokens_qids(link_file)

        known_qids_mask = np.array([int(q) in index_qids_set for q in link_qids], dtype=bool)
        link_tokens = link_tokens[known_qids_mask]
        link_qids = link_qids[known_qids_mask]

        link_tokens_tensor = torch.from_numpy(link_tokens.astype(np.int32, copy=False))
        link_qids_tensor = torch.from_numpy(link_qids.astype(np.int64, copy=False))
        dataset = torch.utils.data.TensorDataset(link_tokens_tensor, link_qids_tensor)
        dataloader = torch.utils.data.DataLoader(
            dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=2
        )

        data_len = len(dataset)
        if data_len == 0:
            print(f"Skipping {link_file.name}: dataset length is zero after filtering")
            continue

        description_tokens = np.zeros((data_len * 2, index_tokens.shape[1]), dtype=np.int32)
        link_tokens_list = np.zeros((data_len * 2, link_tokens.shape[1]), dtype=np.int32)
        y = np.zeros((data_len * 2,), dtype=np.int8)
        qids = np.zeros((data_len * 2,), dtype=np.int32)
        output_index = 0

        for batch_tokens, batch_qids in tqdm(
            dataloader, desc=f"Creating dataset for {link_file.name}", total=len(dataloader)
        ):
            attention_mask = create_attention_mask(batch_tokens)

            with torch.inference_mode():
                batch_embs = (
                    model(batch_tokens.to(device), attention_mask.to(device))
                    .to(torch.float16)
                    .cpu()
                )

            top_qids = searcher.find(batch_embs.numpy(), num_neighbors=2)

            batch_qids_np = batch_qids.cpu().numpy()
            batch_tokens_np = batch_tokens.cpu().numpy().astype(np.int32, copy=False)
            positive_mask = [index_qid_to_index[int(qid)] for qid in batch_qids_np]
            data_size = len(batch_tokens)

            description_tokens[output_index : output_index + data_size] = index_tokens[
                positive_mask
            ]
            link_tokens_list[output_index : output_index + data_size] = batch_tokens_np
            y[output_index : output_index + data_size] = 1
            qids[output_index : output_index + data_size] = batch_qids_np.astype(
                np.int32, copy=False
            )

            output_index += data_size

            neg_qids = get_neg_qids(top_qids, batch_qids_np)

            negative_mask = [index_qid_to_index[int(qid)] for qid in neg_qids]
            description_tokens[output_index : output_index + data_size] = index_tokens[
                negative_mask
            ]
            link_tokens_list[output_index : output_index + data_size] = batch_tokens_np
            y[output_index : output_index + data_size] = 0
            qids[output_index : output_index + data_size] = np.array(neg_qids, dtype=np.int32)

            output_index += data_size

        if output_index != data_len * 2:
            description_tokens = description_tokens[:output_index]
            link_tokens_list = link_tokens_list[:output_index]
            y = y[:output_index]
            qids = qids[:output_index]

        output_file = output_path / f"{link_file.stem}_dataset.npz"

        print(
            f"Saving dataset for {link_file.name} -> {output_file.name} | "
            f"positives/negatives: {output_index // 2}"
        )

        np.savez(
            output_file,
            description_tokens=description_tokens,
            link_tokens=link_tokens_list,
            y=y,
            qids=qids,
        )


def create_default_binary_dataset():
    index_embs_dir = Path(
        "/lnet/work/home-students-external/farhan/troja/outputs/workdirs/asi_se_to_rozbilo_init_all/damuel_for_index_6"
    )
    index_tokens_path = Path(
        "/lnet/work/home-students-external/farhan/troja/outputs/v2_normal_filtered/descs_pages"
    )
    link_tokens_path = Path(
        "/lnet/work/home-students-external/farhan/troja/outputs/v2_normal_filtered/links"
    )
    embedding_model_path = Path(
        "/lnet/work/home-students-external/farhan/troja/outputs/workdirs/asi_se_to_rozbilo_init_all/models_5/ema.pth",
    )
    output_path = Path(
        "/lnet/work/home-students-external/farhan/troja/outputs/reranking_test/reranker_dataset_with_qids"
    )
    model_name = "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base"

    create_binary_dataset(
        index_embs_dir,
        index_tokens_path,
        link_tokens_path,
        model_name,
        embedding_model_path,
        output_path,
        batch_size=2560,
    )


if __name__ == "__main__":
    index_embs_dir = Path(
        "/lnet/work/home-students-external/farhan/troja/outputs/workdirs/asi_se_to_rozbilo_init_all/damuel_for_index_6"
    )
    index_tokens_path = Path(
        "/lnet/work/home-students-external/farhan/troja/outputs/v2_normal_filtered/descs_pages"
    )
    link_tokens_path = Path(
        "/lnet/work/home-students-external/farhan/troja/outputs/v2_normal_filtered/links"
    )
    embedding_model_path = Path(
        "/lnet/work/home-students-external/farhan/troja/outputs/workdirs/asi_se_to_rozbilo_init_all/models_5/ema.pth",
    )
    output_path = Path(
        "/lnet/work/home-students-external/farhan/troja/outputs/reranking_test/reranker_dataset_with_qids"
    )
    model_name = "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base"

    create_binary_dataset(
        index_embs_dir,
        index_tokens_path,
        link_tokens_path,
        model_name,
        embedding_model_path,
        output_path,
        batch_size=2048,
    )
