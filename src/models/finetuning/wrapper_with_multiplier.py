import torch
import torch.nn as nn

from models.finetuning.wrapper import FinetuningWrapper


class FinetuningWrapperWithMultiplier(FinetuningWrapper):
    def __init__(self, model, logit_multiplier):
        print("FinetuningWrapperWithMultiplier init called.")
        super(FinetuningWrapperWithMultiplier, self).__init__(model)
        self.logit_multiplier = nn.Parameter(torch.tensor(logit_multiplier))

    def forward(self, mentions, frenemies):
        embeddings = self.forward_only_embeddings(mentions)

        frenemies = self.forward_only_embeddings(frenemies)

        dot_product = torch.mm(embeddings, frenemies.t())

        dot_product = dot_product * self.logit_multiplier

        return dot_product
