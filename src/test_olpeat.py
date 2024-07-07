from transformers import BertModel, BertTokenizerFast
import wandb

from baselines.olpeat.olpeat import OLPEAT
from pathlib import Path

wandb.init(project=f"test_olpeat")

t = BertTokenizerFast.from_pretrained("setu4993/LEALLA-base")
print(t.get_vocab()["[PAD]"])

# foundation_model = BertModel.from_pretrained("setu4993/LEALLA-base")
# embs, qids = embed(Path("/home/farhand/dump"), foundation_model)

ol = OLPEAT(Path("/home/farhand/dump_small"), "setu4993/LEALLA-base")
ol.train(gpus_available=4)
ol.recall_at(10, Path("/home/farhand/dump_small"))
ol.recall_at(1, Path("/home/farhand/dump_small"))
