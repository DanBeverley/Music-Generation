from typing import List, Dict

from torch import LongTensor
from torch.nn.utils.rnn import pad_sequence


def collate_fn(batch: List[Dict[str, LongTensor]],
               pad_token: int) -> Dict[str, LongTensor]:
    """
    Collate function for dynamic padding.
    Args:
        batch: List of dictionaries, each containing 'input_ids' and 'labels'.
        pad_token: Token used for padding.
    Returns:
        A dictionary with padded 'input_ids' and 'labels' as LongTensor.
    """

    input_ids = [item["input_ids"] for item in batch]
    labels    = [item["labels"] for item in batch]

    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=pad_token).long()
    labels    = pad_sequence(labels, batch_first=True, padding_value=pad_token).long()

    return {"input_ids":input_ids, "labels":labels}