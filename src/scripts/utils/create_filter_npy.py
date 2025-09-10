import json
import numpy as np

IN_PATH = "/lnet/work/home-students-external/farhan/troja/qid_type_agnostic2.json"
OUT_PATH = "/lnet/work/home-students-external/farhan/troja/filtered_qids2.npy"

with open(IN_PATH, "r") as f:
    qid_type_map = json.load(f)

filtered_qids = [int(qid) for qid, t in qid_type_map.items() if t != "none"]

np.save(OUT_PATH, np.array(filtered_qids))

print(f"Saved {len(filtered_qids)} QIDs to {OUT_PATH}")
