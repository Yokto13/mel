import torch
from transformers import AutoTokenizer

from models.searchers.brute_force_searcher import BruteForceSearcher
from utils.model_factory import ModelFactory
from utils.loaders import load_embs_and_qids

model_path = "/lnet/work/home-students-external/farhan/troja/outputs/models/LEALLA-base"
state_dict_path = "/lnet/work/home-students-external/farhan/troja/outputs/workdirs/ml9/models_2/final.pth"
index_path = "/lnet/work/home-students-external/farhan/troja/outputs/workdirs/ml9/damuel_for_index_3"

model = ModelFactory.auto_load_from_file(model_path, state_dict_path, None, None)
tokenizer = AutoTokenizer.from_pretrained(model_path)

embs, qids = load_embs_and_qids(index_path)

searcher = BruteForceSearcher(embs, qids)


def main():
    while True:
        query = input("Enter a query: ")
        tokenized = tokenizer(
            query,
            return_tensors="pt",
            padding="max_length",
            max_length=64,
            truncation=True,
        )
        toks, attn_mask = tokenized.input_ids, tokenized.attention_mask
        query_embs = model(toks, attn_mask)
        query_embs = query_embs.to(torch.float16)  # Convert to float16
        results = searcher.find(query_embs, 50)
        print(results)


if __name__ == "__main__":
    main()
