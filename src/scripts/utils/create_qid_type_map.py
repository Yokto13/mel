import json
import sys

sys.path.append("../../")

from tokenization.pipeline.loaders import DaMuELPageTypeLoader

OUT = "/lnet/work/home-students-external/farhan/troja/qid_type_agnostic2.json"

loader = DaMuELPageTypeLoader(
    "/lnet/work/home-students-external/farhan/damuel/dev/damuel_2.0-dev_kb_agnostic",
    extract_qid=True,
)

qid_type_map = {}
idx = 0
for page_type, qid in loader.process():
    if idx % 100000 == 0:
        print(f"Processed {idx} entries")
    if qid in qid_type_map:
        print("Shouldn't happen")
    qid_type_map[qid] = page_type
    idx += 1


with open(OUT, "w") as f:
    json.dump(qid_type_map, f)
print(f"Saved qid_type_map to {OUT}")
