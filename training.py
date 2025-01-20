import os
import random

import model
from collate_fn import collate_fn
from utils import (get_optimizer, get_loss_function,
                   get_scheduler, evaluate_model)
from Preprocessing import subset_train, subset_valid

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

def accuracy_fn(predictions, targets):
    """Compute accuracy of predictions"""
    _, pred_ids = torch.max(predictions, dim=-1)
    correct = (pred_ids == targets).float()
    mask = targets != pad_token # Ignore padding tokens
    return (correct*mask).sum().item()/mask.sum().item()

if __name__ == "__main__":
    num_classes = 429
    d_model = 512
    num_layers = 6
    num_heads = 8
    dff = 2048
    dropout_rate = 0.1
    max_seq_len = 512
    pad_token = 0
    learning_rate = 1e-4
    smoothing = 0.1
    num_epochs = 4
    batch_size = 32
    clip_value = 1.0
    generation_length = 100 # Length of generated sequence for evaluation
    checkpoint_dir = "./checkpoints"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_ = model.MusicTransformer(num_classes = num_classes,
                                    d_model = d_model,
                                    num_layers = num_layers,
                                    num_heads = num_heads,
                                    dff = dff,
                                    dropout_rate = dropout_rate,
                                    max_seq_len = max_seq_len,
                                    pad_token = pad_token).to(device)

    criterion = get_loss_function("cross_entropy", model_.embedding.num_embeddings, smoothing=0.1,
                                  pad_token = model_.pad_token)
    optimizer = get_optimizer("adamW", model_, learning_rate=1e-4)

    # Learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = get_scheduler("step", optimizer, step_size = 10 , gamma=.1)

    # Dataloader
    train_loader = DataLoader(subset_train, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda x:collate_fn(x, pad_token))
    val_loader   = DataLoader(subset_valid, batch_size=batch_size, shuffle=False,
                              collate_fn=lambda x:collate_fn(x, pad_token))

    # Metrics
    metrics = {"accuracy":accuracy_fn}

    # Training Loop
    os.makedirs(checkpoint_dir, exist_ok=True)

    for epoch in range(num_epochs):
        model_.train()
        epoch_loss = 0
        for inputs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs_ids = inputs["input_ids"].to(device)
            targets    = inputs["labels"].to(device)

            # Shift targets for autoregressive modeling
            decoder_inputs = inputs_ids[:,:-1]
            decoder_targets = targets[:,1:]

            # Forward Pass
            outputs = model_(decoder_inputs)
            loss = criterion(outputs.view(-1, outputs.size(-1)),
                             decoder_targets.contiguous().view(-1))

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_.parameters(), max_norm=clip_value)
            optimizer.step()

            epoch_loss += loss.item()

        train_loss = epoch_loss/len(train_loader)
        print(f"Epoch {epoch+1} | Training Loss : {train_loss:.4f}")

        # Validation Loss
        val_loss, val_metrics = evaluate_model(model = model_,
                                               dataloader = val_loader,
                                               criterion=criterion,
                                               metrics = metrics,
                                               device = device)
        print(f"Epoch {epoch + 1} | Validating Loss: {val_loss:.4f} | Validating Metrics: {val_metrics}")

        # Generate and save music samples
        if epoch%1==0: # Generate for every epoch
            model_.eval()
            with torch.no_grad():
                start_sequence = torch.tensor([random.randint(1, num_classes-1)],
                                              device=device).unsqueeze(0) # Random start token
                generated_sequence = start_sequence.clone()
                for _ in range(generation_length):
                    output = model_(generated_sequence)
                    next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
                    if next_token.item() == pad_token: # Stop on pad token
                        break
                print(f"Generated Sequence : {generated_sequence.tolist()}")

        # Save checkpoint
        torch.save(model_.state_dict(),os.path.join(checkpoint_dir,
                                                    f"model_epoch{epoch + 1}.pth"))
        scheduler.step()

