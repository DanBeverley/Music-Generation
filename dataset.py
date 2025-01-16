from pathlib import Path

from torch.utils.data import Dataset, DataLoader
from typing import Callable, Any, List, Dict, Optional
import os

class AbstractMusicDataset(Dataset):
    """
    Abstract base class for music dataset

    Args:
        data_path (List[Path]) : Path to the dataset
        preprocess_fn (Callable): Function to preprocess a single sample
        max_seq_len (int) : Maximum sequence length of the data
        pad_token (int) : token used for padding sequences
    """
    def __init__(self, data_path:List[Path],
                 preprocess_fn:Callable,
                 max_seq_len:int, pad_token:int):
        self.data_path = data_path
        self.preprocess_fn = preprocess_fn
        self.max_seq_len = max_seq_len,
        self.pad_token = pad_token
        self.data = self.load_data()

    def load_data(self)->List[Any]:
        """Load any data from specified data path, Override this in subclasses"""
        raise NotImplementedError("Subclasses must implement this method")

    def preprocess(self, sample:Any) -> Dict[str, Any]:
        """Preprocess a single data sample"""
        return self.preprocess_fn(sample, max_seq_len=self.max_seq_len,
                                  pad_token=self.pad_token)
    def __len__(self)->int:
        return len(self.data)
    def __getitem__(self, idx:int)->Dict[str, Any]:
        return self.preprocess(self.data[idx])

