from transformers import BertModel, BertTokenizerFast
import wandb

from baselines.olpeat.olpeat import OLPEAT
from pathlib import Path

wandb.init(project=f"test_olpeat")

t = BertTokenizerFast.from_pretrained("setu4993/LEALLA-base")
print(t.get_vocab()["[PAD]"])

# foundation_model = BertModel.from_pretrained("setu4993/LEALLA-base")
# embs, qids = embed(Path("/home/farhand/dump"), foundation_model)

ol = OLPEAT(
    Path("/home/farhand/tokens_damuel_at/damuel_1.0_ja/links"), "setu4993/LEALLA-base"
)
ol.train(gpus_available=4)
print(ol.recall_at(10, Path("")))
print(ol.recall_at(1, Path("/home/farhand/tokens_mewsli_at/ja")))
