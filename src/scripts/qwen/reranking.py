import argparse
from pathlib import Path

# Import BruteForceSearcher from models
from models.searchers.brute_force_searcher import BruteForceSearcher

# Import Reranker class
from scripts.qwen.reranker import Reranker

# Import necessary functions from loaders
from utils.loaders import load_embs_and_qids, load_tokens_qids, load_tokens_qids_from_dir


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
        default="/lnet/work/home-students-external/farhan/troja/outputs/tokens_mewsli_finetuning/es/mentions_1307770978027216442.npz",
        help="Path to mewsli token file or directory",
    )
    parser.add_argument(
        "--qwen_model_name",
        type=str,
        default="Qwen/Qwen3-Reranker-0.6B",
        help="Name of the QWEN model",
    )
    args = parser.parse_args()

    # Resolve tokens and embeddings (directories or files)
    damuel_tokens, damuel_token_qids = load_tokens_qids_from_dir(args.damuel_token, verbose=True)
    damuel_embs, damuel_qids = load_embs_and_qids(args.damuel_embs)

    qid_to_damuel_token = {qid: token for qid, token in zip(damuel_token_qids, damuel_tokens)}
    qid_to_damuel_emb = {qid: emb for qid, emb in zip(damuel_qids, damuel_embs)}

    del damuel_token_qids

    mewsli_tokens, mewsli_qids = load_tokens_qids(args.mewsli_tokens)

    # Take first four names (tokens) from each as a quick smoke-test
    damuel_tokens_preview = damuel_tokens[:4]
    mewsli_tokens_preview = mewsli_tokens[:4]

    print("First 4 damuel tokens:", damuel_tokens_preview)
    print("First 4 mewsli tokens:", mewsli_tokens_preview)

    # Create searcher using damuel embeddings and damuel qids
    searcher = BruteForceSearcher(damuel_embs, damuel_qids)
    print("Searcher created.")

    # Initialize reranker model (actual reranking logic to be implemented later)
    reranker = Reranker(model_name=args.qwen_model_name)
    print(f"Reranker initialized with model: {reranker.model_name}")

    # Stub for QWEN model loading
    print("QWEN model to be used:", args.qwen_model_name)

    # TODO: implement reranking logic
    print("Reranking logic not implemented. Exiting.")

    print("jupi")


if __name__ == "__main__":
    main()
