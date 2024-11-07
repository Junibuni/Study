import torch
import os

def save_checkpoint(model, optimizer, filepath):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(model, optimizer, filepath, device='cpu'):
    if os.path.exists(filepath):
        checkpoint = torch.load(filepath, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        print(f"Checkpoint loaded from {filepath}")
    else:
        print(f"No checkpoint found at {filepath}")

def log_loss(epoch, batch_idx, loss):
    print(f"Epoch [{epoch + 1}], Batch [{batch_idx + 1}], Loss: {loss:.4f}")
