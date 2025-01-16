from typing import Any, Dict

import torch

def midi_preprocess(sample:Any, max_seq_len:int, pad_token:int)->Dict[str, torch.Tensor]:
    """
    Preprocess a MIDI sample: tokenize, truncate, and pad.

    Args:
        sample (MidiFile): MIDI file sample.
        max_seq_len (int): Maximum sequence length.
        pad_token (int): Padding token.
    Returns:
        Dict[str, torch.Tensor]: Preprocessed sample with input IDs and labels.
    """
    tokens = [msg.note for msg in sample.tracks[0] if msg.type == "note_on"]
    tokens = tokens[:max_seq_len]
    padded_tokens = tokens + [pad_token] * (max_seq_len - len(tokens))
    return {
        "input_ids": torch.tensor(padded_tokens, dtype=torch.long),
        "labels": torch.tensor(padded_tokens, dtype=torch.long)
    }