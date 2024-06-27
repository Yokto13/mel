import json
from pathlib import Path

import pyspark

conf = pyspark.SparkConf()
conf.set("spark.hadoop.io.compression.codecs", "io.sensesecure.hadoop.xz.XZCodec")
sc = pyspark.SparkContext(conf=conf)

damuel_workdir = Path("/home/farhand/damuel_spark_workdir")

damuel_path = damuel_workdir / "damuel_1.0_es"


def jsonify(l):
    return json.loads(l)


def get_qids(path):
    data = sc.textFile(str(path))
    data = data.map(jsonify)
    damuel_qids = data.map(lambda x: x["qid"]).map(lambda x: int(x[1:]))

    return damuel_qids


def qid_with_specifier(line):
    specifier = "UNK"
    if "wiki" in line:
        specifier = "WIKI"
    elif "description" in line:
        specifier = "WIKIDATA"
    return int(line["qid"][1:]), specifier


def qid_only(line):
    return int(line["qid"][1:])


def get_qids(path, qid_gather):
    data = sc.textFile(str(path))
    data = data.map(jsonify)
    damuel_qids = data.map(qid_gather)

    return damuel_qids


def get_all_damuel_qids(damuel_path, qid_gather=qid_only):
    damuel_qids = get_qids(damuel_path, qid_gather)
    return damuel_qids


if __name__ == "__main__":
    for lang in ["ar", "de", "es", "en", "fa", "ja", "sr", "ta", "tr"]:
        path = damuel_workdir / f"damuel_1.0_{lang}"
        damuel_qids = get_all_damuel_qids(path, qid_gather=qid_with_specifier)
        damuel_qids = damuel_qids.distinct()
        damuel_qids = damuel_qids.collect()
        with open(f"damuel_qids_{lang}.json", "w") as f:
            json.dump(damuel_qids, f)
