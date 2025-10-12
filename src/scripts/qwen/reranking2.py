import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from models.searchers.brute_force_searcher import BruteForceSearcher
from reranking.models.pairwise_mlp import PairwiseMLPReranker
from scripts.qwen.reranker import Reranker
from utils.embeddings import create_attention_mask
from utils.loaders import load_embs_and_qids, load_tokens_qids, load_tokens_qids_from_dir
from utils.model_factory import ModelFactory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    parser = argparse.ArgumentParser(description="Reranking for entity linking")
    parser.add_argument(
        "--damuel_token",
        type=str,
        default="/lnet/work/home-students-external/farhan/troja/outputs/v2_normal_filtered/descs_pages",
        help="Path to damuel token file or directory",
    )
    parser.add_argument(
        "--damuel_embs",
        type=str,
        default="/lnet/work/home-students-external/farhan/troja/outputs/workdirs/asi_se_to_rozbilo_init_all/damuel_for_index_6",
        help="Path to damuel embeddings directory or .npz file",
    )
    parser.add_argument(
        "--mewsli_tokens",
        type=str,
        default="/lnet/work/home-students-external/farhan/troja/outputs/tokens_mewsli_finetuning/en/mentions_1641252057782057661.npz",
        help="Path to mewsli token file or directory",
    )
    parser.add_argument(
        "--qwen_model_name",
        type=str,
        default="Qwen/Qwen3-Reranker-0.6B",
        help="Name of the QWEN model",
    )
    parser.add_argument(
        "--reranking_model_path",
        type=str,
        default="/lnet/work/home-students-external/farhan/troja/outputs/workdirs/asi_se_to_rozbilo_init_all/models_5/ema.pth",
    )
    parser.add_argument(
        "--num_neighbors",
        type=int,
        default=10,
        help="Number of neighbors to retrieve from the searcher",
    )
    args = parser.parse_args()

    reranking_model = ModelFactory.auto_load_from_file(
        "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base",
        args.reranking_model_path,
    )
    reranking_model.eval()
    reranking_model.to(device)

    reranking_tokenizer = AutoTokenizer.from_pretrained(
        "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base"
    )

    reranker = PairwiseMLPReranker(
        "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base",
        state_dict_path=args.reranking_model_path,
        mlp_hidden_dim=2048,
    )
    state_dict = torch.load(
        "/lnet/work/home-students-external/farhan/troja/outputs/reranking_models/pairwise_mlp_noise/20000.pth",
        map_location=device,
    )
    reranker.load_state_dict(state_dict)
    reranker.eval()

    mewsli_tokens, mewsli_qids = load_tokens_qids(args.mewsli_tokens)
    mewsli_tokens = torch.from_numpy(mewsli_tokens)
    mewsli_qids = torch.from_numpy(mewsli_qids)

    decode_mewsli_example = reranking_tokenizer.decode(mewsli_tokens[0], skip_special_tokens=True)
    print("Decoded mewsli example:", decode_mewsli_example)

    # Resolve tokens and embeddings (directories or files)
    damuel_tokens, damuel_token_qids = load_tokens_qids_from_dir(args.damuel_token, verbose=True)
    damuel_embs, damuel_qids = load_embs_and_qids(args.damuel_embs)

    qid_to_damuel_token = {qid: token for qid, token in zip(damuel_token_qids, damuel_tokens)}

    del damuel_token_qids

    # Take first four names (tokens) from each as a quick smoke-test
    damuel_tokens_preview = damuel_tokens[:4]
    mewsli_tokens_preview = mewsli_tokens[:4]

    print("First 4 damuel tokens:", damuel_tokens_preview)
    print("First 4 mewsli tokens:", mewsli_tokens_preview)

    # Create searcher using damuel embeddings and damuel qids
    searcher = BruteForceSearcher(damuel_embs, damuel_qids)
    print("Searcher created.")

    # Initialize reranker model (actual reranking logic to be implemented later)
    # reranker = Reranker(model_name=args.qwen_model_name)
    # print(f"Reranker initialized with model: {reranker.model_name}")

    # Stub for QWEN model loading
    print("QWEN model to be used:", args.qwen_model_name)

    # TODO: implement reranking logic
    print("Reranking logic not implemented. Exiting.")

    print("jupi")

    mewsli_dataset = torch.utils.data.TensorDataset(mewsli_tokens, mewsli_qids)
    mewsli_loader = DataLoader(mewsli_dataset, batch_size=1, shuffle=False)

    good = 0
    total = 0

    for batch in mewsli_loader:
        mewsli_token = batch[0]
        qid = batch[1].item()  # ensure scalar for a fair comparison with predicted_qid
        # print(f"Processing Mewsli token: {mewsli_token}, QID: {qid}")

        tokens = mewsli_token.to(device, dtype=torch.int64)

        with torch.inference_mode():
            attention_mask = create_attention_mask(tokens).to(device)
            mewsli_emb = (
                reranking_model(tokens, attention_mask)
                .to("cpu", dtype=torch.float16)
                .detach()
                .numpy()
            )

        neighbor_qids = searcher.find(mewsli_emb, num_neighbors=args.num_neighbors)

        damuel_candidates = [qid_to_damuel_token[nq] for nq in neighbor_qids[0]]
        damuel_candidates_str = [
            reranking_tokenizer.decode(dc, skip_special_tokens=True) for dc in damuel_candidates
        ]

        mewsli_str = reranking_tokenizer.decode(mewsli_token[0], skip_special_tokens=True)

        scores = []
        # print("Mewsli mention:", mewsli_str)
        for dc in damuel_candidates_str:
            with torch.inference_mode():
                # print(dc)
                score = reranker.score(mewsli_str, dc)
                scores.append(score)
        # print(scores)
        predicted_qid = int(neighbor_qids[0][scores.index(max(scores))])
        if predicted_qid == qid:
            good += 1
        total += 1

        print("Current accuracy:", round(good / total * 100, 4))


if __name__ == "__main__":
    main()
