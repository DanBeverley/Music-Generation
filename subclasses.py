import json
import logging
from pathlib import Path
from typing import List, Any, Dict
import pandas as pd
import os
from mido import MidiFile
from dataset import AbstractMusicDataset

logger = logging.getLogger(__name__)

class MIDIDataset(AbstractMusicDataset):
    def __init__(self, data_path:Path, max_seq_len:int, pad_token:int,
                 sos_token:int, eos_token:int):
        super().__init__(data_path, max_seq_len, pad_token,
                         sos_token, eos_token)
    def load_data(self) ->List[MidiFile]:
        """Load MIDI file from dataset directory"""
        try:
            files = list(self.data_path.glob("*.midi"))
            if not files:
                logger.warning(f"No MIDI files found in: {self.data_path}")
                return []
            return [MidiFile(str(file)) for file in files]
        except FileNotFoundError:
            logger.error(f"MIDI directory not found: {self.data_path}")
            return []
        except Exception as e:
            logger.error(f"Error loading MIDI files: {e}")
            return []

class CSVDataset(AbstractMusicDataset):
    def __init__(self, data_path: Path, csv_filename: str, max_seq_len: int, pad_token: int, sos_token: int, eos_token: int):
        self.csv_filename = csv_filename
        super().__init__(data_path, max_seq_len, pad_token, sos_token, eos_token)


    def load_data(self) -> List[Dict[str, Any]]:
        """Load a CSV file into a list of dictionaries."""
        csv_path = self.data_path / self.csv_filename # Use Path object for joining
        try:
            df = pd.read_csv(str(csv_path)) # Convert Path to string
            return df.to_dict(orient="records")
        except FileNotFoundError:
            logger.error(f"CSV file not found: {csv_path}")
            return []
        except pd.errors.ParserError as e:
            logger.error(f"Error parsing CSV file: {e}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading or parsing the CSV file: {e}")
            return []


class JSONDataset(AbstractMusicDataset):
    def __init__(self, data_path: Path, json_filename: str, max_seq_len: int, pad_token: int, sos_token: int, eos_token: int):
        self.json_filename = json_filename
        super().__init__(data_path, max_seq_len, pad_token, sos_token, eos_token)

    def load_data(self) -> List[Dict[str, Any]]:
        """Load a JSON file containing the dataset."""
        json_path = self.data_path / self.json_filename
        try:
            with open(str(json_path), "r") as f: # Convert Path to str
                return json.load(f)
        except FileNotFoundError:
            logger.error(f"JSON file not found: {json_path}")
            return []
        except json.JSONDecodeError as e:
            logger.error(f"Error decoding JSON: {e}")
            return []
        except Exception as e:
            logger.error(f"An unexpected error occurred while loading or parsing the JSON file: {e}")
            return []
