import model
import torch
from tqdm import tqdm
from Preprocessing import subset_train, subset_valid
from torch.utils.data import DataLoader
from collate_fn import collate_fn

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

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model_ = model.MusicTransformer(num_classes = num_classes,
                                    d_model = d_model,
                                    num_layers = num_layers,
                                    num_heads = num_heads,
                                    dff = dff,
                                    dropout_rate = dropout_rate,
                                    max_seq_len = max_seq_len,
                                    pad_token = pad_token).to(device)

    criterion, optimizer = model.loss_and_optimizer(model_, learning_rate = learning_rate,
                                                    smoothing = smoothing)

    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # Dataloader
    train_loader = DataLoader(subset_train, batch_size=batch_size, shuffle=True,
                              collate_fn=lambda x:collate_fn(x, pad_token))
    val_loader   = DataLoader(subset_valid, batch_size=batch_size, shuffle=False,
                              collate_fn=lambda x:collate_fn(x, pad_token))

    # Training Loop
    for epoch in range(num_epochs):
        model_.train()
        epoch_loss = 0
        for inputs in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            inputs_ids = inputs["input_ids"].to(device)
            targets    = inputs["labels"].to(device)

            # Forward Pass
            outputs = model_(inputs_ids)
            out = outputs.mean(dim=1, keepdim=True)
            loss = criterion(out.float(), targets.float())

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model_.parameters(), max_norm=clip_value)
            optimizer.step()

            epoch_loss += loss.item()

        # Validation Loss
        model_.eval()
        val_loss = 0
        with torch.no_grad():
            for inputs in val_loader:
                inputs_ids = inputs["input_ids"].to(device)
                targets = inputs["labels"].to(device)
                outputs = model_(inputs_ids)
                val_loss += criterion(outputs, targets).item()

        val_loss /= len(val_loader)
        print(f"Epoch {epoch + 1} | Train Loss: {epoch_loss / len(train_loader):.4f} | Val Loss: {val_loss:.4f}")

        # Save checkpoint
        torch.save(model_.state_dict(), f"model_epoch{epoch + 1}.pth")
        scheduler.step()

