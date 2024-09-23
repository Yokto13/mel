import pandas as pd
from pathlib import Path
from collections.abc import Generator
from tqdm.auto import tqdm

from .base import LoaderStep
from .qid_parsing import parse_qid


class MewsliLoader(LoaderStep):
    def __init__(self, mewsli_tsv_path: str, use_context: bool = True):
        super().__init__(mewsli_tsv_path)
        self.use_context = use_context
        self.data_df = pd.read_csv(mewsli_tsv_path, sep="\t")

    def process(self) -> Generator[tuple, None, None]:
        for row in tqdm(
            self.data_df.itertuples(),
            total=len(self.data_df),
            desc=f"Processing {self.path}",
        ):
            qid = parse_qid(row.qid)
            if self.use_context:
                with open(Path(self.path).parent / "text" / row.docid, "r") as f:
                    text = f.read()
                    mention_slice = self.get_mention_slice_from_row(row)
                    yield mention_slice, text, qid
            else:
                yield row.mention, qid

    def get_mention_slice_from_row(self, row):
        mention_start = row.position
        mention_end = row.position + row.length
        return slice(mention_start, mention_end)
