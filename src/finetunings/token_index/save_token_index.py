""" Used to build the index for hard negative mining."""

from data_processors.index.token_index import TokenIndex
import fire
from utils.argument_wrappers import paths_exist


@paths_exist([0, 1])  # dest is created by .save so no need to check whether it exists.
def build_and_save_token_index(
    embs_source, toks_source, dest, max_per_qid=1000, mentions=True
):
    ti = TokenIndex.from_dirs(embs_source, toks_source, mentions=mentions)
    ti.query(ti.embs[0], ti.qids[0], 1, 1)
    print(ti.embs[0].shape)
    ti.save(dest)


if __name__ == "__main__":
    fire.Fire(build_and_save_token_index)
