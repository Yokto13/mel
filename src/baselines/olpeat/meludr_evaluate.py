from baselines.olpeat.meludr_find_recall import find_recall

_RECALLS = [1, 10]


def meludr_run_recall_calculation(damuel_descs, mewsli_dir, damuel_links, recall=None):
    recalls = _RECALLS if recall is None else [recall]
    for recall in recalls:
        print(f"Running evaluation_scann with recall: {recall}")
        find_recall(damuel_descs, mewsli_dir, recall, damuel_links=damuel_links)

