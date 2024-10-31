from finetunings.evaluation.find_recall import find_recall

_RECALLS = [1, 10, 100]


def run_recall_calculation(damuel_dir, mewsli_dir, recall=None):
    recalls = _RECALLS if recall is None else [recall]
    find_recall(damuel_dir, mewsli_dir, recalls)


def evaluate(
    damuel_desc_tokens,
    mewsli_tokens,
    model_path,
    damuel_dir,
    mewsli_dir,
    state_dict=None,
):
    raise NotImplementedError()


if __name__ == "__main__":
    evaluate()
