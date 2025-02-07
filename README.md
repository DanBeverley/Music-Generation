# AI Music Generation with Transformer

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)

A transformer-based AI system for generating musical sequences from MIDI data, implementing relative attention mechanisms and supporting multiple dataset formats.

## 🎹 Overview
This project enables AI-powered music generation using:
- **Transformer architecture** with relative positional encoding
- **MIDI preprocessing** with REMI tokenization
- Support for **Maestro dataset**, custom MIDI files, CSV/JSON sequences
- Training utilities with mixed precision & gradient clipping
- Music generation with autoregressive decoding

## ✨ Features
- 🎼 MIDI-to-token conversion preserving musical features (pitch, velocity, timing)
- 🧠 Custom transformer model with encoder-decoder architecture
- 📁 Multi-format support: MIDI (.mid/.midi), CSV, JSON
- ⚡ Mixed precision training
- 🎧 Audio sample generation
- 📊 Training monitoring metrics

## 📦 Requirements
```bash
Python 3.10+
torch==2.4.1
miditoolkit==1.0.1
pretty_midi==0.2.10
miditok==3.0.4
tqdm>=4.66.0
numpy>=1.26.0
```

## 🚀 Installation
- Clone repo:
```bash
git clone https://github.com/yourusername/ai-music-generation.git
cd ai-music-generation
```
- Install dependencies:
```bash
pip install -r requirements.txt
```
- Prepare datasets
```bash
mkdir -p data/maestro
```

## 🎛️ Usage
### 1.Data Preparation
Convert MIDI files to token sequences
```bash
from dataset import MaestroDataset

dataset = MaestroDataset(
    file_paths=["data/maestro"],
    min_seq=256,
    max_seq=512,
    pad_token=0,
    output_dir="processed_data")
```
### 2. Configuration
Edit config.json
```bash
{
  "model_params": {
    "num_classes": 512,
    "d_model": 256,
    "num_layers": 6,
    "num_heads": 8,
    "dff": 1024,
    "dropout_rate": 0.1,
    "max_seq_len": 512,
    "pad_token": 0
  },
  "training_params": {
    "batch_size": 32,
    "num_epochs": 100,
    "learning_rate": 0.0001,
    "clip_value": 1.0,
    "smoothing": 0.1,
    "generation_length": 512
  }
}
```
### 3. Training
```bash
python training.py --config config.json
```
### 4. Generation
```python
from utils import generate_sequence

sequence = generate_sequence(
    model=model,
    start_token=60,  # Middle C
    max_length=128,
    device="cuda"
)
```

## 🗃️ Ideal Datasets
1. MAESTRO Dataset (Recommended)
  - Classical piano performances
  - Download [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro)
2. MIDI Collections
  - [Lakh MIDI Dataset](https://colinraffel.com/projects/lmd/)
  - [Piano MIDI](https://www.piano-midi.de/)
3. Custom Formats
  - CSV with notes column
  - JSON with sequence arrays

## 🤝 Contributing
1. Fork the project
2. Create your feature branch (git checkout -b feature/AmazingFeature)
3. Commit changes (git commit -m 'Add some AmazingFeature')
4. Push to branch (git push origin feature/AmazingFeature)
5. Open a Pull Request

## 🙏 Acknowledgments
- Transformer architecture from [Attention is All You Need](https://arxiv.org/abs/1706.03762)
- MIDI processing with [miditok](https://github.com/Natooz/MidiTok)
- Dataset handling inspired by [Magenta](https://magenta.tensorflow.org/)

