import torch
import torch.nn as nn


class BaselineModel(nn.Module):
    def __init__(self):
        super(BaselineModel, self).__init__()
        self.cosine_similarity = nn.CosineSimilarity(dim=2)

        self.dummy_parameter = nn.Parameter(torch.empty(0))

    def forward(self, embedding, frenemies):
        embedding_exp = embedding.unsqueeze(1).expand(-1, frenemies.shape[1], -1)

        output = self.cosine_similarity(embedding_exp, frenemies)

        return output
