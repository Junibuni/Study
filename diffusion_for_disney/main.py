import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from src.data import CharacterDataset, get_dataloader
from src.model import get_model
from src.utils import save_checkpoint, load_checkpoint, log_loss
from src.config import Config, parse_args


def train(model, dataloader, optimizer, num_epochs, device, save_path="checkpoints/best_model.pth"):
    model.to(device)
    model.train()

    best_loss = float('inf')
    for epoch in range(num_epochs):
        epoch_loss = 0
        for batch_idx, x in enumerate(dataloader):
            x = x.to(device)
            batch_size = x.size(0)
            
            t = torch.randint(0, model.num_timesteps, (batch_size,), device=device).long()

            x_noisy, noise = model.add_noise(x, t)
            predicted_noise = model(x_noisy, t)

            loss = F.mse_loss(predicted_noise, noise)
            epoch_loss += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if batch_idx % 10 == 0:
                log_loss(epoch, batch_idx, loss.item())

        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {avg_epoch_loss:.4f}")

        if avg_epoch_loss < best_loss:
            print(f"Saving new best model with loss {avg_epoch_loss:.4f}")
            best_loss = avg_epoch_loss
            save_checkpoint(model, optimizer, save_path)

def main():
    args = parse_args()
    config = Config(args)

    dataloader = get_dataloader(
        data_dir=config.data_dir,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=True,
        num_workers=config.num_workers
    )

    model = get_model(image_channels=3)
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train(model, dataloader, optimizer, config.num_epochs, device)

if __name__ == "__main__":
    main()
