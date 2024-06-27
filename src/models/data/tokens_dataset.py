from torch.utils.data import Dataset
from models.data.mention_qid_pair import MentionQidPair


class TokensDataset(Dataset):
    def __init__(self, tokens_list: list[MentionQidPair]):
        self.tokens_list = tokens_list

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        return (
            self.tokens_list[idx].tokenization_output["input_ids"],
            self.tokens_list[idx].tokenization_output["attention_mask"],
            self.tokens_list[idx].qid,
        )
