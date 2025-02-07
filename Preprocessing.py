import json
import logging
from pathlib import Path
from typing import Dict, List

from miditoolkit import MidiFile
from miditok import REMI, TokenizerConfig

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
    def __init__(self, file_paths:List[Path],
                 min_seq:int,
                 max_seq:int,
                 pad_token:int,
                 tokenizer_config: TokenizerConfig = None,
                 preprocess:bool=True,
                 output_dir:Path=None):

        if not isinstance(file_paths, list):
            file_paths = [file_paths]
        file_paths = [Path(fp) for fp in file_paths]

        self.file_paths = file_paths
        self.samples = []
        self.pad_token = pad_token
        # Preprocessing if needed
        if preprocess and output_dir is not None:
            self._preprocessing_(file_paths, tokenizer_config, output_dir)
            file_paths = list(output_dir.glob("*.json"))
        # Load preprocessed tokens
        self.load_samples(file_paths, min_seq, max_seq)
    def _preprocessing_(self, file_paths:List[Path],
                              tokenizer_config:TokenizerConfig,
                              output_dir:Path):
        try:
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            logger.error(f"Failed to create output directory: {e}")
            raise

        for i in tqdm(file_paths, desc = "Preprocessing MIDI files"):
            try:
                if i.suffix in ["MIDI","MID","midi","mid"]:
                    midi = MidiFile(i)
                    if tokenizer_config is None:
                        tokenizer_config = TokenizerConfig()
                    tokenizer = REMI(tokenizer_config) if tokenizer_config is not None else REMI(tokenizer_config)

                    all_tracks_tokens = [tokenizer.midi_to_tokens(midi)[0].ids for track in midi.tracks if len(track)>0]
                    tokens = [token for track in all_tracks_tokens for token in track]
                else:
                    continue # Skip non-MIDI files
                # Save tokens to JSON
                output_file = output_dir / f"{i.stem}_tokens.json"
                with open(output_file, "w") as f:
                    json.dump({"ids":tokens},f)
            except Exception as e:
                logger.warning(f"Error processing {i}: {e}")

    def load_samples(self, file_paths: List[Path],
                     min_seq: int,
                     max_seq: int):
        """Load tokenized samples and create sequences"""
        for file_path in tqdm(file_paths, desc="Loading tokenized files"):
            try:
                with open(file_path, "r") as f:
                    data = json.load(f)
                    tokens = data.get("ids", [])
                if len(tokens) < min_seq:
                    continue
                # Create sequence with overlap
                num_segments = len(tokens) - max_seq + 1
                for i in range(0, num_segments, max_seq // 2):
                    seq = tokens[i:i + max_seq]
                    if len(seq) >= min_seq:
                        self.samples.append(LongTensor(seq))
            except Exception as e:
                logger.warning(f"Error loading {file_path}: {e}")

    def __getitem__(self, idx)->Dict[str, LongTensor]:
        return {"input_ids":self.samples[idx],
                "labels":self.samples[idx]}
    def __len__(self) -> int:
        return len(self.samples)
    def __repr__(self):
        return self.__str__()
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

