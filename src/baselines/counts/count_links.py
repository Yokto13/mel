import json
from pathlib import Path

import pyspark

conf = pyspark.SparkConf()
conf.set("spark.hadoop.io.compression.codecs", "io.sensesecure.hadoop.xz.XZCodec")
sc = pyspark.SparkContext(conf=conf)

damuel_workdir = Path("/home/farhand/damuel_spark_workdir")


def jsonify(l):
    return json.loads(l)


def get_links_from_wiki(wiki):
    tokens = wiki["tokens"]
    text = wiki["text"]
    for l in wiki["links"]:
        if "qid" not in l:
            continue
        if l["origin"] != "wiki":
            continue
        start = l["start"]
        end = l["end"] - 1
        try:
            mention_slice = slice(tokens[start]["start"], tokens[end]["end"])
            yield text[mention_slice]
        except IndexError:
            print(start, end, len(tokens))


def get_mentions(path):
    data = sc.textFile(str(path))
    data = data.map(jsonify)
    # filter wiki
    data = data.filter(lambda x: "wiki" in x)
    wikis = data.map(lambda x: x["wiki"])
    return wikis.flatMap(get_links_from_wiki)


def get_all_damuel_mentions(path):
    mentions = get_mentions(path)
    return mentions.distinct().flatMap(lambda x: x).collect()


def is_damuel_specific_dir(path):
    return "damuel" in path.name and "wikidata" not in path.name


if __name__ == "__main__":
    # Print number of mentions
    all_mentions = set()
    for fp in damuel_workdir.iterdir():
        if not is_damuel_specific_dir(fp):
            continue
        print(fp)
        damuel_mentions = get_all_damuel_mentions(fp)
        print(len(damuel_mentions))
        all_mentions.update(damuel_mentions)
    print("Total mentions:", len(all_mentions))
