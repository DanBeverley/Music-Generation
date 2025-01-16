from typing import Any, Dict, List

import torch
import logging
from pretty_midi import PrettyMIDI

logger = logging.getLogger(__name__)

def midi_preprocess(sample:PrettyMIDI, max_seq_len:int, pad_token:int,
                    sos_token:int, eos_token:int,
                    default_sample = None)->Dict[str, torch.Tensor]:
    """
    Preprocess a MIDI sample: tokenize, truncate, and pad.

    Args:
        sample (MidiFile): MIDI file sample.
        max_seq_len (int): Maximum sequence length.
        pad_token (int): Padding token.
    Returns:
        Dict[str, torch.Tensor]: Preprocessed sample with input IDs and labels.
    """
    try:
        if not sample.instruments:
            logger.warning("MIDI file has no instruments.")
            return default_sample or {
                "input_ids": torch.full((max_seq_len,), pad_token, dtype=torch.long),
                "labels": torch.full((max_seq_len,), pad_token, dtype=torch.long)
            }
        tokens = []
        for instrument in sample.instruments:
            for note in instrument.notes:
                tokens.append(note.pitch)
        tokens = [sos_token] + tokens[:max_seq_len-2] + [eos_token]
        padded_tokens = torch.nn.functional.pad(torch.tensor(tokens, dtype=torch.long),
                                                (0, max_seq_len - len(tokens)),
                                                value=pad_token)
        logger.info(f"Processed MIDI file: {len(tokens)} tokens")
        return {"input_ids":padded_tokens,
                "labels":padded_tokens}
    except Exception as e:
        logger.error(f"Error preprocessing MIDI: {e}")
        return default_sample or {
            "input_ids": torch.full((max_seq_len,), pad_token, dtype=torch.long),
            "labels": torch.full((max_seq_len,), pad_token, dtype=torch.long)
        }


def csv_preprocess(
        sample: Dict[str, Any],
        max_seq_len: int,
        pad_token: int,
        sos_token: int,
        eos_token: int,
        default_sample: Dict[str, torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Preprocess CSV data for model input.

    Args:
        sample: A dictionary containing 'notes' as a key with a list of integer tokens.
        max_seq_len: The maximum sequence length.
        pad_token: The token used for padding.
        sos_token: The start-of-sequence token.
        eos_token: The end-of-sequence token.
        default_sample: Optional default sample for invalid data.

    Returns:
        A dictionary with 'input_ids' and 'labels' tensors.
    """
    try:
        # Validate input format
        if not isinstance(sample, dict) or 'notes' not in sample:
            raise ValueError("Invalid CSV sample format: missing 'notes' key.")

        tokens = sample['notes']

        # Check token validity
        if not all(isinstance(token, int) for token in tokens):
            raise TypeError("CSV tokens must be integers.")

        # Add special tokens and truncate
        tokens = [sos_token] + tokens[:max_seq_len - 2] + [eos_token]

        # Pad the sequence
        padded_tokens = torch.nn.functional.pad(
            torch.tensor(tokens, dtype=torch.long),
            (0, max_seq_len - len(tokens)),
            value=pad_token
        )

        logger.info(f"Processed CSV data: {len(tokens)} tokens (max {max_seq_len}).")
        return {
            "input_ids": padded_tokens,
            "labels": padded_tokens
        }
    except (ValueError, TypeError) as e:
        logger.error(f"Error preprocessing CSV: {e}")
        return default_sample or {
            "input_ids": torch.full((max_seq_len,), pad_token, dtype=torch.long),
            "labels": torch.full((max_seq_len,), pad_token, dtype=torch.long)
        }


def json_preprocess(
        sample: Dict[str, Any],
        max_seq_len: int,
        pad_token: int,
        sos_token: int,
        eos_token: int,
        default_sample: Dict[str, torch.Tensor] = None
) -> Dict[str, torch.Tensor]:
    """
    Preprocess JSON data for model input.

    Args:
        sample: A dictionary containing 'sequence' as a key with a list of integer tokens.
        max_seq_len: The maximum sequence length.
        pad_token: The token used for padding.
        sos_token: The start-of-sequence token.
        eos_token: The end-of-sequence token.
        default_sample: Optional default sample for invalid data.

    Returns:
        A dictionary with 'input_ids' and 'labels' tensors.
    """
    try:
        # Validate input format
        if not isinstance(sample, dict) or 'sequence' not in sample:
            raise ValueError("Invalid JSON sample format: missing 'sequence' key.")

        tokens = sample['sequence']

        # Check token validity
        if not all(isinstance(token, int) for token in tokens):
            raise TypeError("JSON tokens must be integers.")

        # Add special tokens and truncate
        tokens = [sos_token] + tokens[:max_seq_len - 2] + [eos_token]

        # Pad the sequence
        padded_tokens = torch.nn.functional.pad(
            torch.tensor(tokens, dtype=torch.long),
            (0, max_seq_len - len(tokens)),
            value=pad_token
        )

        logger.info(f"Processed JSON data: {len(tokens)} tokens (max {max_seq_len}).")
        return {
            "input_ids": padded_tokens,
            "labels": padded_tokens
        }
    except (ValueError, TypeError) as e:
        logger.error(f"Error preprocessing JSON: {e}")
        return default_sample or {
            "input_ids": torch.full((max_seq_len,), pad_token, dtype=torch.long),
            "labels": torch.full((max_seq_len,), pad_token, dtype=torch.long)
        }


