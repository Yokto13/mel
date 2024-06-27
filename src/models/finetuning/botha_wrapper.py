import torch
import torch.nn as nn


class BothaWrapper(nn.Module):
    def __init__(self, mentions_model, entities_model):
        print("BothaWrapper init called.")
        super(BothaWrapper, self).__init__()
        self.logit_multiplier = nn.Parameter(torch.tensor(1.0))
        self.mentions_model = mentions_model
        self.entities_model = entities_model

    def _forward_only_embeddings(self, data, model):
        toks = data[:, 0, :]
        att = data[:, 1, :]
        embeddings = model(toks, att).pooler_output
        return torch.nn.functional.normalize(embeddings, p=2, dim=1)

    def forward_only_embeddings(
        self, data
    ):  # for compatibility with the finetuning wrapper
        return self._forward_only_embeddings(data, self.mentions_model)

    def forward_only_entities(self, data):
        return self._forward_only_embeddings(data, self.entities_model)

    def forward_only_mentions(self, data):
        return self._forward_only_embeddings(data, self.mentions_model)

    def forward(self, mentions, entities):
        mentions = self.forward_only_mentions(mentions)
        entities = self.forward_only_entities(entities)

        dot_product = torch.mm(mentions, entities.t())

        dot_product = dot_product * self.logit_multiplier

        return dot_product
