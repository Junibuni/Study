import os
import argparse
import torch

class Config:
    def __init__(self, args):
        self.data_dir = args.data_dir
        self.batch_size = args.batch_size
        self.image_size = args.image_size
        self.num_workers = args.num_workers
        self.num_epochs = args.num_epochs
        self.learning_rate = args.learning_rate
        self.device = args.device
        self.image_channels = args.image_channels
        self.num_timesteps = args.num_timesteps
        self.checkpoint_dir = args.checkpoint_dir
        self.results_dir = args.results_dir

        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description="Character Diffusion Model Training")

    parser.add_argument("--data_dir", type=str, default="diffusion_for_disney/data", help="Directory with character images")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--image_size", type=int, nargs=2, default=[128, 128], help="Image size (width, height)")
    parser.add_argument("--num_workers", type=int, default=4, help="Number of workers for data loading")
    parser.add_argument("--num_epochs", type=int, default=100, help="Number of epochs for training")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use for training (e.g., 'cuda' or 'cpu')")
    parser.add_argument("--image_channels", type=int, default=3, help="Number of channels in the images")
    parser.add_argument("--num_timesteps", type=int, default=1000, help="Number of timesteps for diffusion")
    parser.add_argument("--checkpoint_dir", type=str, default="diffusion_for_disney/checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--results_dir", type=str, default="diffusion_for_disney/results", help="Directory to save generated samples")

    return parser.parse_args()