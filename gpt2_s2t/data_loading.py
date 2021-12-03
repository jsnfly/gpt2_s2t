import torch
from pathlib import Path
from torch.utils.data import Dataset

class S2TDataset(Dataset):
    def __init__(self, data_path):
        self.path = Path(data_path)
        self.files = list(self.path.iterdir())

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_path = self.files[idx]
        eg = torch.load(file_path)
        eg['file_path'] = file_path
        return eg

def make_collate_fn(tokenizer):
    def collate_fn(examples):
        encoder_hidden_states = torch.stack([eg['wave2vec_features'] for eg in examples], dim=0)
        input_ids = tokenizer([eg['transcription'] for eg in examples], return_tensors='pt', padding=True).input_ids
        return encoder_hidden_states, input_ids
    return collate_fn
