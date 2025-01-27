import os
import json
import random

import model
from collate_fn import collate_fn
from utils import (get_optimizer, get_loss_function,
                   get_scheduler, evaluate_model, save_generated_sequence)
from Preprocessing import subset_train, subset_valid

import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()
accumulation_steps = 4

def accuracy_fn(predictions, targets, pad_token):
    """Compute accuracy of predictions"""
    _, pred_ids = torch.max(predictions, dim=-1)
    correct = (pred_ids == targets).float()
    mask = targets != pad_token # Ignore padding tokens
    return (correct*mask).sum().item()/mask.sum().item()

if __name__ == "__main__":
    with open("config.json", "r") as f:
        config = json.load(f)

    model_params = config["model_params"]
    training_params = config["training_params"]
    scheduler_params = config["scheduler"]
    data_paths = config["data_paths"]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_ = model.MusicTransformer(num_classes = model_params["num_classes"],
                                    d_model = model_params["d_model"],
                                    num_layers = model_params["num_layers"],
                                    num_heads = model_params["num_heads"],
                                    dff = model_params["dff"],
                                    dropout_rate = model_params["dropout_rate"],
                                    max_seq_len = model_params["max_seq_len"],
                                    pad_token = model_params["pad_token"]).to(device)

    criterion = get_loss_function(config["loss_function"], model_.embedding.num_embeddings,
                                  smoothing=training_params["smoothing"],
                                  pad_token = model_params["pad_token"])
    optimizer = get_optimizer(config["optimizer"], model_,
                              learning_rate=training_params["learning_rate"])

    # Learning rate scheduler
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    scheduler = get_scheduler(scheduler_params["type"], optimizer,
                              step_size = scheduler_params["step_size"] ,
                              gamma = scheduler_params["gamma"])
    batch_size = min(training_params["batch_size"], len(subset_train))

    # Dataloader
    train_loader = DataLoader(subset_train, batch_size=batch_size,
                              shuffle=True, collate_fn=lambda x:collate_fn(x, model_params["pad_token"]))
    val_loader   = DataLoader(subset_valid, batch_size=batch_size,
                              shuffle=False, collate_fn=lambda x:collate_fn(x, model_params["pad_token"]))

    # Metrics
    metrics = {"accuracy":accuracy_fn}

    # Training Loop
    best_val_loss = float("inf")
    os.makedirs(data_paths["checkpoint_dir"], exist_ok=True)

    for epoch in range(training_params["num_epochs"]):
        model_.train()
        epoch_loss = 0
        optimizer.zero_grad()
        for i, inputs in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{training_params["num_epochs"]}")):
            inputs_ids = inputs["input_ids"].to(device)
            targets    = inputs["labels"].to(device)

            # Shift targets for autoregressive modeling
            decoder_inputs = inputs_ids[:,:-1]
            decoder_targets = targets[:,1:]

            # Forward Pass
            with autocast():
                outputs = model_(decoder_inputs)
                loss = criterion(outputs.view(-1, outputs.size(-1)),
                                 decoder_targets.contiguous().view(-1))/accumulation_steps

            # Backward pass and optimization
            scaler.scale(loss).backward()
            if (i+1)%accumulation_steps==0:
                scaler.step(optimizer)
                scaler.update()
            torch.nn.utils.clip_grad_norm_(model_.parameters(),
                                           max_norm=training_params["clip_value"])
            epoch_loss += loss.item()*accumulation_steps

        train_loss = epoch_loss/len(train_loader)
        print(f"Epoch {epoch+1} | Training Loss : {train_loss:.4f}")

        # Validation Loss
        val_loss, val_metrics = evaluate_model(model = model_,
                                               dataloader = val_loader,
                                               criterion=criterion,
                                               metrics = metrics,
                                               device = device)
        if val_loss<best_val_loss:
            best_val_loss=val_loss
            torch.save(model_.state_dict(),
                       os.path.join(data_paths["checkpoint_dir"],
                                    "best_model.pth"))

        print(f"Epoch {epoch + 1} | Validating Loss: {val_loss:.4f} | Validating Metrics: {val_metrics}")

        # Generate and save music samples
        if epoch%1==0: # Generate for every epoch
            model_.eval()
            with torch.no_grad():
                start_sequence = torch.tensor([random.randint(1, model_params["num_classes"]-1)],
                                              device=device).unsqueeze(0) # Random start token
                generated_sequence = start_sequence.clone()
                for _ in range(training_params["generation_length"]):
                    output = model_(generated_sequence)
                    next_token = output[:, -1, :].argmax(dim=-1, keepdim=True)
                    generated_sequence = torch.cat([generated_sequence, next_token], dim=1)
                    if next_token.item() == model_params["pad_token"]: # Stop on pad token
                        break
                print(f"Generated Sequence : {generated_sequence.tolist()}")
                save_generated_sequence(generated_sequence.squeeze().tolist(),
                                        f"generated_epoch_{epoch+1}.mid")

        # Save checkpoint
        torch.save(model_.state_dict(),os.path.join(data_paths["checkpoint_dir"],
                                                    f"model_epoch{epoch + 1}.pth"))
        scheduler.step()

