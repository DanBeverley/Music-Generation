import json
from typing import List, Any, Dict
import pandas as pd
import os
from mido import MidiFile
from dataset import AbstractMusicDataset


class MIDIDataset(AbstractMusicDataset):
    def load_data(self) ->List[MidiFile]:
        """Load MIDI file from dataset directory"""
        files = [os.path.join(self.data_path, f)
                 for f in os.listdir(self.data_path)
                 if f.endswith(".midi")]
        return [MidiFile(file) for file in files]

class CSVDataset(AbstractMusicDataset):
    def load_data(self)->List[Dict[str, Any]]:
        """Load a CSV file into a list of dictionaries"""
        CSV_path = os.path.join(self.data_path, "data.csv")
        df = pd.read_csv(CSV_path)
        return df.to_dict(orient = "records")

class JSONDataset(AbstractMusicDataset):
    def load_data(self) ->List[Dict[str,Any]]:
        """Load a JSON file containing the dataset"""
        json_path = os.path.join(self.data_path, "data.json")
        with open(json_path, "r") as f:
            return json.load(f)
