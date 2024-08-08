import torch
import torch.nn as nn

from utils.embeddings import create_attention_mask


class FinetuningWrapper(nn.Module):
    def __init__(self, model):
        print("FinetuningWrapper init called.")
        super(FinetuningWrapper, self).__init__()
        self.model = model

    def forward_only_embeddings(self, data):
        toks = data[:]
        att = create_attention_mask(toks)
        embeddings = self.model(toks, att).pooler_output
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def forward(self, mentions, frenemies):
        embeddings = self.forward_only_embeddings(mentions)
        print("embs done")

        frenemies = self.forward_only_embeddings(frenemies)
        print("frenemies done")

        dot_product = torch.mm(embeddings, frenemies.t())

        return dot_product
