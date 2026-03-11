import torch
import torch.nn.functional as F
import numpy as np
from EncoderMLPModel import EncoderMLPModel
from DOGMADataset import DOGMaDataset
from MapDataset import MapDataset
import matplotlib.pyplot as plt
from torch.utils.data import random_split
import os

def make_arith_weights(n, device, a=0.3, b=0.1):
    w = torch.linspace(a, b, steps=n, device=device)  # arithmetic sequence
    w = w / w.sum()  # guarantees sum=1 (your example already sums to 1)
    return w

def loss_fn(y_pred, y_true, device):
    
    losses = []
    weights = make_arith_weights(y_pred.shape[1], device)
    for i in range(y_pred.shape[1]):
      occ_loss = torch.norm(y_true[:, i] - y_pred[:, i])
      losses.append(occ_loss)

    loss = torch.sum(weights * torch.stack(losses))
    return loss

def train(n_epochs, use_prev_dataset=False, resume_training = False):

    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    file_path = "../../../../../../bigdata/capstone25W1/2026-03-01_data1"

    dataset = MapDataset(folder=file_path, T=5, F=20)

    print("dataset done")

    n = len(dataset)
    n_train = int(0.7 * n)
    n_val = int(0.15 * n)
    n_test = n - n_train - n_val

    train, val, test = random_split(
        dataset,
        [n_train, n_val, n_test],
        generator=torch.Generator().manual_seed(42)
    )

    # train_X = [item[0] for item in train]
    # train_Y = [item[1] for item in train]
    # torch.save(train_X, "../../../../../../bigdata/capstone25W1/datasets/train_X.pt")
    # torch.save(train_Y, "../../../../../../bigdata/capstone25W1/datasets/train_Y.pt")

    # X_train = train.dataset.X
    # X_test = test.dataset.X
    # X_val = val.dataset.X

    # Y_train = train.dataset.Y
    # Y_test = test.dataset.Y
    # Y_val = val.dataset.Y

    train_loader = torch.utils.data.DataLoader(
    train, batch_size=4, shuffle=True, drop_last=True,
    num_workers=2, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val, batch_size=4, shuffle=False, drop_last=False,
        num_workers=2, pin_memory=True
    )

    test_loader = torch.utils.data.DataLoader(
        test, batch_size=4, shuffle=False, drop_last=False,
        num_workers=2, pin_memory=True
    )


    train_losses = []
    val_losses = []

    best_val_loss = float("inf")
    best_model_path = "../../../../../../bigdata/capstone25W1/models/encoder_mlp_non_dogma_plab.pth"

    if resume_training and os.path.exists(best_model_path):
        model = EncoderMLPModel()
        state = torch.load(best_model_path, map_location='cpu')
        # If you saved only state_dict previously, load it into model:
        if isinstance(state, dict) and 'model_state_dict' not in state and 'epoch' not in state:
            model.load_state_dict(state)
        else:
            # If you saved full checkpoint dict with 'model_state_dict' key
            model.load_state_dict(state['model_state_dict'])
        print(f"Resumed model loaded from {best_model_path}")
    else:
        model = EncoderMLPModel()

    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    early_stopping_patience = 3
    min_delta = 1e-4   # require this much improvement to count as 'better'
    no_improve_epochs = 0
    best_val_loss = float('inf')

    for epoch in range(n_epochs):
        model.train()
        train_loss = 0.0

        for X, Y in train_loader:

            X = X.to(device)
            Y = Y.to(device)

            optimizer.zero_grad()
            Y_pred = model(X)

            loss = loss_fn(Y_pred, Y, device)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            # print(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for X, Y in val_loader:

                X = X.to(device)
                Y = Y.to(device)

                Y_pred = model(X)
                loss = loss_fn(Y_pred, Y, device)
                val_loss += loss.item()

        train_loss /= len(train_loader)
        val_loss /= len(val_loader)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}/{n_epochs} | Train Loss: {train_loss:.5f} | Val Loss: {val_loss:.5f}")

        # if abs(val_loss - prev_epoch_loss) < 1e-4:
        #     number_of_epochs_no_improvement = 0
        #     prev_epoch_loss = val_loss
        # else:
        #     number_of_epochs_no_improvement += 1
        
        # if number_of_epochs_no_improvement >= early_stopping_patience:
        #     print(f"Early stopping triggered after {epoch+1} epochs with no improvement.")
        #     break

        # # Save best model
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     torch.save(model.state_dict(), best_model_path)
        #     print(f" → Saved new best model with val_loss = {best_val_loss:.5f}")

        # check improvement (strict decrease by min_delta)
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            no_improve_epochs = 0
            # save state dict (simple)
            torch.save(model.state_dict(), best_model_path)
            print(f" → Saved new best model with val_loss = {best_val_loss:.5f}")
        else:
            no_improve_epochs += 1
            print(f"No improvement for {no_improve_epochs}/{early_stopping_patience} epochs.")

        if no_improve_epochs >= early_stopping_patience:
            print(f"Early stopping triggered after {epoch+1} epochs with no sufficient improvement.")
            break

    print(f"\nTraining complete. Best model saved at: {best_model_path}")

    # ===== Plot loss curves =====
    epochs = list(range(1, len(train_losses) + 1))
    plt.figure(figsize=(8,5))
    plt.plot(epochs, train_losses, label="Train Loss")
    plt.plot(epochs, val_losses, label="Validation Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Training vs Validation Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig('Loss Curves Non Autoregressive.png')

    return None

if __name__ == "__main__":

    train(n_epochs=20, use_prev_dataset=False, resume_training = False)


