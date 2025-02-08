import json
import logging
from pathlib import Path
from typing import Dict, List, Union

from miditoolkit import MidiFile
from miditok import REMI, TokenizerConfig
from pretty_midi import PrettyMIDI

import torch as torch
from torchtoolkit.data import create_subsets
from torch import LongTensor
from torch.utils.data import Dataset
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class MaestroDataset(Dataset):
    """
    A dataset for processing Maestro MIDI files.

    Args:
        file_paths (list): List of paths to MIDI or JSON files.
        min_seq (int): Minimum sequence length.
        max_seq (int): Maximum sequence length.
        tokenizer_config (TokenizerConfig, optional): Configuration for the tokenizer (default: None)
        pad_token (int): The token used for padding sequences.
        preprocess (bool): Whether to preprocess and save tokenized files.
        output_dir (Path): Directory to save preprocessed token files.

    Attributes:
        samples (list): List of tokenized sequences.
    """
    def __init__(self, file_paths:List[Union[str,Path]],
                 min_seq:int, max_seq:int, pad_token:int,
                 tokenizer_config: TokenizerConfig = None,
                 preprocess:bool=True, output_dir:Path=None):

        self.file_paths = [Path(fp) for fp in file_paths]
        self.pad_token = pad_token
        self.samples = []

        if preprocess and output_dir:
            self._preprocessing_(output_dir, tokenizer_config)
            processed_file = list(output_dir.glob("*.json"))
            self.load_samples(processed_file, min_seq, max_seq)
        else:
            self.load_samples(self.file_paths, min_seq, max_seq)

    def _preprocessing_(self, output_dir:Path, tokenizer_config:TokenizerConfig):
        output_dir.mkdir(parents=True, exist_ok=True)
        tokenizer_config = tokenizer_config or TokenizerConfig(
            num_velocities=32,
            use_chords=True,
            use_rests=True,
            use_tempos=True,
            use_time_signatures=False,
            beat_res={(0,4):8}
        )
        tokenizer = REMI(tokenizer_config)

        # Discover MIDI files
        midi_files = self._find_midi_files()
        logger.info(f"Found {len(midi_files)} MIDI files for preprocessing")

        # For tracking
        for midi_file in tqdm(midi_files, desc="Preprocessing MIDI"):
            try:
                # Convert MIDI to tokens
                midi = PrettyMIDI(str(midi_file))
                tokens = tokenizer.midi_to_tokens(midi).ids

                if not tokens:
                    logger.warning(f"No tokens generated for {midi_file.name}")
                    continue

                # Save tokens with checksum in filename
                self._save_tokens(output_dir/f"{midi_file.stem}.json", tokens)
            except Exception as e:
                logger.error(f"Failed processing {midi_file.name}: {str(e)}",
                             exc_info = True)
                continue

    def _find_midi_files(self) -> List[Path]:
        """Efficiently locate MIDI files in input paths."""
        midi_files = []
        for path in self.file_paths:
            if path.is_dir():
                midi_files.extend(path.rglob("*.mid"))
                midi_files.extend(path.rglob("*.midi"))
            elif path.suffix.lower() in [".mid", ".midi"]:
                midi_files.append(path)
        return midi_files

    def _save_tokens(self, save_path: Path, tokens: List[int]):
        """Save tokens with atomic write operation."""
        try:
            with open(save_path, "w") as f:
                json.dump({"ids": tokens}, f)
            logger.debug(f"Saved {len(tokens)} tokens to {save_path.name}")
        except IOError as e:
            logger.error(f"Failed saving tokens to {save_path}: {str(e)}")

    def load_samples(self, json_files: List[Path],
                     min_seq: int,
                     max_seq: int):
        """Optimized token sequence loading with overlap handling"""
        if min_seq >= max_seq:
            raise ValueError(f"min_seq ({min_seq}) must be < max_seq ({max_seq})")
        logger.info(f"Loading from {len(json_files)} token files")

        for json_file in tqdm(json_files, desc="Loading_tokens"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                if not (tokens := data.get("ids")):
                    logger.warning(f"Empty tokens in {json_file.name}")
                    continue
                # Generate overlapping sequences
                self._create_sequences(tokens, min_seq, max_seq)
            except Exception as e:
                logger.warning(f"Error handling {json_file.name}: {str(e)}")
                continue
    def create_sequence(self, tokens:List[int],
                        min_seq:int, max_seq:int):
        """Generate training sequences with sliding window"""
        seq_length = len(tokens)
        if seq_length < min_seq:
            return
        # Overlapping sequences with 50% overlap
        for i in range(0, seq_length-max_seq+1, max_seq//2):
            seq = tokens[i:i+max_seq]
            self.samples.append(torch.LongTensor(seq))


    def __getitem__(self, idx)->Dict[str, LongTensor]:
        return {"input_ids":self.samples[idx],
                "labels":self.samples[idx]}
    def __len__(self) -> int:
        return len(self.samples)
    def __repr__(self)-> str:
        return f"MaestroDataset({len(self)} samples)"
    def __str__(self) -> str:
        return "No data loaded" if len(self)==0 else f"{len(self.samples)} samples"


# Loads tokens and create data loaders for training
tokens_paths = list(Path('/kaggle/working/preprocessedv2/').glob("**/*.json"))
tokenizer_config = TokenizerConfig(num_velocities=32, use_chords=True,
                                   use_rests=True, use_tempose=True,
                                   use_time_signatures=True,
                                   beat_res={(0, 4): 8, (4, 12): 4})
dataset = MaestroDataset(file_paths = tokens_paths
                         , max_seq=512, min_seq=384,
                         pad_token = 0, preprocess=True,
                         tokenizer_config=tokenizer_config,
                         output_dir=Path("/kaggle/working/"))
subset_train, subset_valid = create_subsets(dataset, [0.3])

