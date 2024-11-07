import torch
import torch.nn as nn
import torch.nn.functional as F

class UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3, features=[64, 128, 256, 512]):
        super(UNet, self).__init__()
        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        # Encoder
        for feature in features:
            self.encoder.append(self._conv_block(in_channels, feature))
            in_channels = feature

        # Bottleneck
        self.bottleneck = self._conv_block(features[-1], features[-1] * 2)

        # Decoder
        for feature in reversed(features):
            self.decoder.append(
                nn.ConvTranspose2d(feature * 2, feature, kernel_size=2, stride=2)
            )
            self.decoder.append(self._conv_block(feature * 2, feature))

        # Final layer
        self.final_layer = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def _conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        skip_connections = []

        # Encoder
        for layer in self.encoder:
            x = layer(x)
            skip_connections.append(x)
            x = F.max_pool2d(x, kernel_size=2, stride=2)

        # Bottleneck
        x = self.bottleneck(x)

        # Decoder
        skip_connections = skip_connections[::-1]
        for idx in range(0, len(self.decoder), 2):
            x = self.decoder[idx](x)
            skip_connection = skip_connections[idx // 2]
            if x.shape != skip_connection.shape:
                x = F.interpolate(x, size=skip_connection.shape[2:])
            x = torch.cat((skip_connection, x), dim=1)
            x = self.decoder[idx + 1](x)

        return self.final_layer(x)

class DiffusionModel(nn.Module):
    def __init__(self, unet, num_timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super(DiffusionModel, self).__init__()
        self.unet = unet
        self.num_timesteps = num_timesteps

        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, axis=0)

        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1.0 - self.alphas_cumprod)

    def forward(self, x, t):
        return self.unet(x)

    def add_noise(self, x_start, t):
        self.sqrt_alphas_cumprod = self.sqrt_alphas_cumprod.to(x_start.device)
        self.sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod.to(x_start.device)

        noise = torch.randn_like(x_start)
        alpha_t = self.sqrt_alphas_cumprod[t].view(-1, 1, 1, 1)
        one_minus_alpha_t = self.sqrt_one_minus_alphas_cumprod[t].view(-1, 1, 1, 1)
        return alpha_t * x_start + one_minus_alpha_t * noise, noise

    def sample(self, shape, device):
        x = torch.randn(shape).to(device)
        for t in reversed(range(self.num_timesteps)):
            t_tensor = torch.full((shape[0],), t, device=device, dtype=torch.long)
            predicted_noise = self.forward(x, t_tensor)

            alpha_t = self.alphas[t]
            one_minus_alpha_t = 1 - self.alphas[t]
            alpha_t_sqrt = torch.sqrt(alpha_t)
            one_minus_alpha_t_sqrt = torch.sqrt(one_minus_alpha_t)
            x = (1 / alpha_t_sqrt) * (x - one_minus_alpha_t_sqrt * predicted_noise)

            if t > 0:
                noise = torch.randn_like(x)
                x += torch.sqrt(self.betas[t]) * noise

        return x


def get_model(image_channels=3):
    unet = UNet(in_channels=image_channels, out_channels=image_channels)
    model = DiffusionModel(unet)
    return model
