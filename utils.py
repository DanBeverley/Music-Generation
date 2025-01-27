import torch
import torch.nn as nn
import torch.optim as optim
from miditoolkit import MidiFile, Instrument, Note
from torch.utils.data import DataLoader

from model import LabelSmoothingLoss, MusicTransformer

def build_music_transformer(num_classes:int,
                            config:dict):
    """
    Build a MusicTransformer model using the provided configuration.

    Args:
        num_classes (int): The number of output classes (vocabulary size).
        config (dict): A dictionary containing model hyperparameters.

    Returns:
        MusicTransformer: A configured MusicTransformer instance.
    """
    return MusicTransformer(num_classes = num_classes,
                            d_model = config['d_model'],
                            num_layers = config['num_layers'],
                            num_heads = config['num_heads'],
                            dff = config['dff'],
                            dropout_rate = config['dropout_rate'],
                            max_seq_len = config['max_seq_len'],
                            pad_token = config["pad_token"])


def get_loss_function(loss_type:str, num_classes:int,
                      smoothing:float, pad_token:int):
    """
    Get the appropriate loss function based on the provided type

    Args:
        loss_type (str): The type of loss function to use ("cross_entropy" or custom).
        num_classes (int): The number of output classes.
        smoothing (float): Label smoothing value for cross-entropy.
        pad_token (int): Padding token ID.

    Returns:
        nn.Module: The loss function.
    """
    if loss_type == "cross_entropy":
        return LabelSmoothingLoss(num_classes, smoothing, pad_token)
    elif loss_type == "mse":
        return nn.MSELoss()
    else:
        raise ValueError(f"Unsupported loss type: {loss_type}")

def get_optimizer(optimizer_type:str,
                  model:nn.Module,
                  learning_rate:float):
    """
    Get the optimizer based on the provided type.

    Args:
        optimizer_type (str): The type of optimizer ("adam", "adamw", etc.).
        model (nn.Module): The model to optimize.
        learning_rate (float): The learning rate.

    Returns:
        Optimizer: A configured optimizer instance."""
    if optimizer_type == "adam":
        return optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_type == "adamw":
        return optim.AdamW(model.parameters(), lr=learning_rate)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_type}")

def get_scheduler(scheduler_type:str,
                  optimizer, **kwargs):
    """
    Get the learning rate scheduler.

    Args:
        scheduler_type (str): Type of scheduler ("cosine", "step", etc.).
        optimizer (Optimizer): The optimizer instance.
        **kwargs: Additional parameters for specific schedulers.

    Returns:
        Scheduler: A configured scheduler instance.
    """
    if scheduler_type == "cosine":
        return optim.lr_scheduler.CosineAnnealingLR(optimizer,
                                                    T_max=kwargs.get("T_max",
                                                                     10))
    elif scheduler_type == "step":
        return optim.lr_scheduler.StepLR(optimizer,
                                         step_size=kwargs.get("step_size",10))
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_type}")

def evaluate_model(model:nn.Module,
                   dataloader:DataLoader,
                   criterion:nn.Module,
                   metrics:dict,
                   device:torch.device):
    """
    Evaluate the model on a dataset.

    Args:
        model (nn.Module): The model to evaluate.
        dataloader (DataLoader): The dataloader for evaluation.
        criterion (nn.Module): The loss function.
        metrics (dict): A dictionary of metric functions.

    Returns:
        tuple: Average loss and a dictionary of average metric values.
    """
    model.eval()
    total_loss = 0
    total_metrics = {name:0 for name in metrics.keys()}

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch["input_ids"].to(device)
            targets = batch["labels"].to(device)

            decoder_inputs = input_ids[:,:-1]
            decoder_targets = targets[:,1:]

            outputs = model(decoder_inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)),
                             decoder_targets.contiguous().view(-1)).item()
            total_loss += loss.item()

            for name, metric_fn in metrics.items():
                total_metrics[name] += metric_fn(outputs, targets)
    avg_loss = total_loss/len(dataloader)
    avg_metrics = {name:value/len(dataloader) for name,value in total_metrics.items()}
    return avg_loss, avg_metrics

def freeze_layers(model:nn.Module, freeze_embedding=True,
                  freeze_layers=[]):
    """
    Freeze specified layers of the model.

    Args:
        model (nn.Module): The model to modify.
        freeze_embedding (bool): Whether to freeze the embedding layer.
        freeze_layers (list[int]): List of layer indices to freeze.
    """
    if freeze_embedding:
        for param in model.embedding.parameters():
            param.requires_grad = False
    for idx in freeze_layers:
        for param in model.layers[idx].parameters():
            param.requires_grad = False

def save_generated_sequence(sequence, output_path):
    midi = MidiFile()
    track = Instrument(program=0)
    for pitch in sequence:
        track.notes.append(Note(velocity=64, pitch = pitch,
                               start = 0, end=480))
    midi.instruments.append(track)
    midi.dump(output_path)