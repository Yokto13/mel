import argparse
from pathlib import Path

import numpy as np
import torch
from einops import rearrange, repeat
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

from models.searchers.brute_force_searcher import BruteForceSearcher
from reranking.models.full_lealla import FullLEALLAReranker
from reranking.models.pairwise_mlp import PairwiseMLPReranker
from reranking.models.pairwise_mlp_with_large_context_emb import (
    PairwiseMLPRerankerWithLargeContextEmb,
)
from reranking.models.pairwise_mlp_with_retrieval_score import PairwiseMLPRerankerWithRetrievalScore
from scripts.qwen.reranker import Reranker
from utils.embeddings import create_attention_mask
from utils.loaders import load_embs_and_qids, load_tokens_qids, load_tokens_qids_from_dir
from utils.model_factory import ModelFactory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

LOGIT_MULTIPLIER = 20.0


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
        default="/lnet/work/home-students-external/farhan/troja/outputs/tokens_mewsli_finetuning",
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
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for DataLoader",
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

    # paraphrase_embs, paraphrase_qids = load_embs_and_qids(
    #     "/lnet/work/home-students-external/farhan/troja/outputs/paraphrase_multilig_index"
    # )
    # paraphrase_embs = torch.from_numpy(paraphrase_embs)
    # base_embs, base_qids = load_embs_and_qids(
    #     "/lnet/work/home-students-external/farhan/troja/outputs/workdirs/asi_se_to_rozbilo_init_all/damuel_for_index_6"
    # )
    # base_embs = torch.from_numpy(base_embs)

    reranker = PairwiseMLPReranker(
        "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base",
        state_dict_path=args.reranking_model_path,
        mlp_hidden_dim=2048,
    )
    # qid_to_paraphrase_emb = {qid: emb for qid, emb in zip(paraphrase_qids, paraphrase_embs)}
    # qid_to_base_emb = {qid: emb for qid, emb in zip(base_qids, base_embs)}
    # reranker = PairwiseMLPRerankerWithLargeContextEmb(
    #     model_name_or_path="/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base",
    #     state_dict_path="/lnet/work/home-students-external/farhan/troja/outputs/workdirs/asi_se_to_rozbilo_init_all/models_5/ema.pth",
    #     qid_to_paraphrase_emb=qid_to_paraphrase_emb,
    #     qid_to_base_emb=qid_to_base_emb,
    # )
    # reranker = FullLEALLAReranker(
    # model_name_or_path="/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base",
    # )
    reranker.load(
        "/lnet/work/home-students-external/farhan/troja/outputs/reranking_models/pairwise_mlp/70000.pth",
    )
    # reranker.load(
    # "/lnet/work/home-students-external/farhan/troja/outputs/reranking_models/full_lealla_r/380000.pth",
    # )
    reranker.eval()
    reranker.to(device)

    # Resolve tokens and embeddings (directories or files)
    damuel_tokens, damuel_token_qids = load_tokens_qids_from_dir(args.damuel_token, verbose=True)
    damuel_embs, damuel_qids = load_embs_and_qids(args.damuel_embs)

    qid_to_damuel_emb = {qid: emb for qid, emb in zip(damuel_qids, damuel_embs)}
    qid_to_damuel_token = {qid: token for qid, token in zip(damuel_token_qids, damuel_tokens)}

    del damuel_token_qids

    # Take first four names (tokens) from damuel as a quick smoke-test
    damuel_tokens_preview = damuel_tokens[:4]
    print("First 4 damuel tokens:", damuel_tokens_preview)

    # Create searcher using damuel embeddings and damuel qids
    searcher = BruteForceSearcher(damuel_embs, damuel_qids)
    print("Searcher created.")

    # Initialize reranker model (actual reranking logic to be implemented later)
    # reranker = Reranker(model_name=args.qwen_model_name)
    # print(f"Reranker initialized with model: {reranker.model_name}")

    # Stub for QWEN model loading
    print("QWEN model to be used:", args.qwen_model_name)

    print("jupi")

    mewsli_root = Path(args.mewsli_tokens)
    if not mewsli_root.exists():
        raise FileNotFoundError(f"MEWSLI tokens path not found: {mewsli_root}")

    language_paths: list[tuple[str, Path]] = []
    if mewsli_root.is_file():
        language_name = mewsli_root.parent.name or mewsli_root.stem
        language_paths.append((language_name, mewsli_root))
    else:
        for subdir in sorted(p for p in mewsli_root.iterdir() if p.is_dir()):
            language_paths.append((subdir.name, subdir))

    if not language_paths:
        raise ValueError(f"No MEWSLI language directories or files found under {mewsli_root}")

    for language, mewsli_path in language_paths:
        print(f"\nEvaluating language: {language}")

        if mewsli_path.is_dir():
            tokens_np, qids_np = load_tokens_qids_from_dir(mewsli_path, verbose=False)
        else:
            tokens_np, qids_np = load_tokens_qids(mewsli_path)

        if tokens_np.size == 0:
            print(f"No samples found for language {language}, skipping.")
            continue

        mewsli_tokens = torch.from_numpy(tokens_np)
        mewsli_qids = torch.from_numpy(qids_np)

        mewsli_dataset = torch.utils.data.TensorDataset(mewsli_tokens, mewsli_qids)
        mewsli_loader = DataLoader(mewsli_dataset, batch_size=args.batch_size, shuffle=False)

        good = 0
        upper_bound_hits = 0
        total = 0

        for mewsli_tokens_batch, qids_batch in mewsli_loader:
            tokens = mewsli_tokens_batch.to(device, dtype=torch.int64)
            qids_batch = qids_batch.view(-1).to(torch.long)

            with torch.inference_mode():
                attention_mask = create_attention_mask(tokens).to(device)
                mewsli_embs = reranking_model(tokens, attention_mask)

            neighbor_qids = searcher.find(
                mewsli_embs.to("cpu").numpy().astype(np.float16), num_neighbors=args.num_neighbors
            )
            neighbor_qids = torch.as_tensor(neighbor_qids, dtype=torch.long)

            retrieval_hits = (neighbor_qids == qids_batch.view(-1, 1)).any(dim=1)
            upper_bound_hits += retrieval_hits.sum().item()

            candidate_embs_lists = []
            candidate_tokens_lists = []
            for row in neighbor_qids.tolist():
                for nq in row:
                    emb = qid_to_damuel_emb[int(nq)]
                    token = qid_to_damuel_token[int(nq)]
                    candidate_embs_lists.append(emb)
                    candidate_tokens_lists.append(token)

            # assert len(candidate_embs_lists) == neighbor_qids.size(0) * neighbor_qids.size(1)

            candidate_embs = torch.as_tensor(
                candidate_embs_lists, dtype=torch.float16, device=device
            )
            together = torch.cat(
                (mewsli_embs.repeat_interleave(neighbor_qids.size(1), dim=0), candidate_embs),
                dim=-1,
            )
            together = together.to(device)
            candidate_tokens = torch.as_tensor(
                candidate_tokens_lists, dtype=torch.int64, device=device
            )
            with torch.inference_mode():
                if isinstance(reranker, PairwiseMLPRerankerWithLargeContextEmb):
                    scores = reranker.score_from_tokens(
                        repeat(tokens, "b d -> (b n) d", n=neighbor_qids.size(1)),
                        rearrange(neighbor_qids, "b n -> (b n)"),
                    )
                    scores = rearrange(scores, "(b n) -> b n", n=neighbor_qids.size(1))
                elif isinstance(reranker, FullLEALLAReranker):
                    score = reranker.score_from_tokens(
                        repeat(tokens, "b d -> (b n) d", n=neighbor_qids.size(1)), candidate_tokens
                    )
                    scores = rearrange(score, "(b n) -> b n", n=neighbor_qids.size(1))
                else:
                    logits = reranker.classifier_forward(together)
                    scores = torch.sigmoid(logits).reshape(
                        neighbor_qids.size(0), neighbor_qids.size(1)
                    )
                if (
                    isinstance(reranker, PairwiseMLPRerankerWithRetrievalScore)
                    # or isinstance(reranker, FullLEALLAReranker)
                    # or isinstance(reranker, PairwiseMLPReranker)
                    # or isinstance(reranker, PairwiseMLPRerankerWithLargeContextEmb)
                ):
                    candidate_embs = candidate_embs.reshape(
                        neighbor_qids.size(0), neighbor_qids.size(1), -1
                    )
                    out = (
                        torch.einsum(
                            "abc,ac->ab",
                            candidate_embs.to(torch.bfloat16),
                            mewsli_embs.to(torch.bfloat16),
                        )
                        * LOGIT_MULTIPLIER
                    )
                    out = out.reshape(neighbor_qids.size(0), neighbor_qids.size(1))
                    scores = (scores + torch.sigmoid(out)) / 2

            max_indices = scores.argmax(dim=1).cpu()
            predicted_qids = neighbor_qids[torch.arange(neighbor_qids.size(0)), max_indices]

            good += (predicted_qids.cpu() == qids_batch.cpu()).sum().item()
            total += qids_batch.numel()

        if total == 0:
            print(f"No valid predictions for language {language}.")
            continue

        final_accuracy = round(good / total * 100, 4)
        retrieval_upper_bound = round(upper_bound_hits / total * 100, 4)
        print(
            f"Final accuracy for {language}: {final_accuracy} (retrieval upper bound: {retrieval_upper_bound})"
        )


if __name__ == "__main__":
    main()
