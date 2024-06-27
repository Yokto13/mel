import torch
import torch.nn as nn

from models.finetuning.wrapper import FinetuningWrapper


class FinetuningWrapperGillick(FinetuningWrapper):
    def __init__(self, model, softmax_multiplier, sigmoid_multiplier, sigmoid_offset):
        print("FinetuningWrapperGillick init called.")
        super(FinetuningWrapperGillick, self).__init__(model)
        self.softmax_multiplier = nn.Parameter(torch.tensor(softmax_multiplier * 1.0))
        self.sigmoid_multiplier = nn.Parameter(torch.tensor(sigmoid_multiplier * 1.0))
        self.sigmoid_offset = nn.Parameter(torch.tensor(sigmoid_offset * 1.0))

    def forward(self, mentions, frenemies):
        embeddings = self.forward_only_embeddings(mentions)

        frenemies = self.forward_only_embeddings(frenemies)

        dot_product = torch.mm(embeddings, frenemies.t())

        dot_product = dot_product * self.softmax_multiplier

        return dot_product
