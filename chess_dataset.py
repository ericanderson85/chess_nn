import torch
from torch.utils.data import Dataset


class ChessDataset(Dataset):
    def __init__(self, file_path):
        data = torch.load(file_path, weights_only=True)
        self.inputs = data['inputs']
        self.targets = data['targets']

    def __len__(self):
        return self.inputs.size(0)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]
