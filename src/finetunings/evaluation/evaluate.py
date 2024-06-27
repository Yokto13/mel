from finetunings.embs_generating.build_together_embs import generate_embs
from finetunings.evaluation.find_recall import find_recall

_RECALLS = [1, 10]


def run_recall_calculation(damuel_dir, mewsli_dir, recall=None):
    recalls = _RECALLS if recall is None else [recall]
    for recall in recalls:
        print(f"Running evaluation_scann with recall: {recall}")
        find_recall(damuel_dir, mewsli_dir, recall)


def evaluate(
    damuel_desc_tokens,
    mewsli_tokens,
    model_path,
    damuel_dir,
    mewsli_dir,
    state_dict=None,
):
    print("Running embs generating for mewsli")
    generate_embs(mewsli_tokens, model_path, mewsli_dir, state_dict)

    print("Running embs generating for damuel")
    generate_embs(damuel_desc_tokens, model_path, damuel_dir, state_dict)

    run_recall_calculation(damuel_dir, mewsli_dir)


if __name__ == "__main__":
    evaluate()
