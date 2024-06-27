""" Calculates number of occurrencies of QIDs as mentions in the DaMuEL language given as argument. """

import argparse
from collections import defaultdict
from functools import partial
import json
from operator import add
from pathlib import Path

import pyspark
from pyspark.sql import SparkSession

conf = pyspark.SparkConf()
# conf.set("spark.hadoop.io.compression.codecs", "io.sensesecure.hadoop.xz.XZCodec")
conf.set("spark.shuffle.compress", "true")
conf.set("spark.io.compression.codec", "lz4")
sc = pyspark.SparkContext(conf=conf)

spark = SparkSession.builder.config(conf=sc.getConf()).getOrCreate()

damuel_workdir = Path("path/to/all/damuels")
spanish = Path("/path/to/damuel_1.0_es")

language_count = defaultdict(int)
language_qid_counts = defaultdict(dict)

parser = argparse.ArgumentParser()
parser.add_argument("--lang")
lang = parser.parse_args().lang


def jsonify(l):
    print("jsonify:", l)
    return json.loads(l)


def get_links_from_wiki(wiki, spanish_qids):
    for l in wiki["links"]:
        if "qid" not in l:
            continue
        if l["origin"] != "wiki":
            continue
        qid = int(l["qid"][1:])
        if qid in spanish_qids:
            continue
        yield qid


def get_mentions(path, spanish_qids):
    data = sc.textFile(str(path))
    data = data.map(jsonify)
    # filter wiki
    data = data.filter(lambda x: "wiki" in x)
    wikis = data.map(lambda x: x["wiki"])
    links_getter = partial(get_links_from_wiki, spanish_qids=spanish_qids)
    return wikis.flatMap(links_getter)


def is_damuel_specific_dir(path):
    return "damuel" in path.name and "wikidata" not in str(path)


def get_lang_from_fp(fp):
    return fp.name.split("_")[-1]


def get_lang_qids(path):
    data = sc.textFile(str(path))
    data = data.map(jsonify)
    data = data.filter(
        lambda x: "wiki" in x or "description" in x
    )  # either wikipedia or wikidata
    qids = data.map(lambda x: int(x["qid"][1:]))
    return qids


def get_all_qids():
    qids = set()
    for fp in damuel_workdir.iterdir():
        if not is_damuel_specific_dir(fp):
            continue
        qids.update(get_lang_qids(fp).collect())
    return qids


if __name__ == "__main__":
    spanish_qids = set(get_lang_qids(spanish).collect())
    qids = get_all_qids()
    for fp in damuel_workdir.iterdir():
        if not str(fp).endswith(lang):
            continue
        if not is_damuel_specific_dir(fp):
            continue
        print(fp)
        damuel_mentions = get_mentions(fp, spanish_qids)
        lang_name = get_lang_from_fp(fp)
        language_count[lang_name] = damuel_mentions.count()
        for_save = (
            damuel_mentions.map(lambda x: (x, 1))
            .reduceByKey(add)
            .toDF(["qid", "count"])
        )
        for_save.write.json(f"outputs/damuel_mentions_{lang_name}.json")

    import json

    with open(f"outputs/language_counts_{lang}.json", "w") as f:
        json.dump(language_count, f)
