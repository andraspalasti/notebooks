import os
from typing import Tuple

import lightning as L
import torch
import torch.nn.functional as F
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms.functional import to_tensor


class DRAW(nn.Module):
    def __init__(self,
                 in_channels: int,
                 img_size: Tuple[int, int],
                 read_size: int,
                 write_size: int,
                 latent_dims: int = 100,
                 hidden_size: int = 256):
        super(DRAW, self).__init__()

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.img_w, self.img_h = img_size
        self.latent_dims = latent_dims
        self.read_size = read_size
        self.write_size = write_size

        # self.init_c = torch.zeros((self.in_channels, *img_size), dtype=torch.float32)
        # self.init_h_enc = torch.zeros(2*self.hidden_size, dtype=torch.float32)
        # self.init_h_dec = torch.zeros(2*self.hidden_size, dtype=torch.float32)

        self.encoder = nn.LSTMCell(
            input_size=2*self.read_size**2 + 2*self.hidden_size,
            hidden_size=self.hidden_size,
        )
        self.decoder = nn.LSTMCell(
            input_size=self.latent_dims,
            hidden_size=self.hidden_size,
        )

        self.Qsampler = nn.Linear(2*self.hidden_size, 2*self.latent_dims)

        self.reader_attn = nn.Linear(2*self.hidden_size, out_features=5)
        self.writer_attn = nn.Linear(2*self.hidden_size, out_features=5)

        self.writer = nn.Linear(2*self.hidden_size, self.write_size**2)

    def sample_img(self, glimpses: int):
        canvas = self.init_c.unsqueeze(dim=0)
        prev_h_dec = self.init_h_dec.unsqueeze(dim=0)
        for _ in range(glimpses):
            z = torch.randn((1, self.latent_dims))
            h_dec = torch.cat(self.decoder(
                input=z, hx=(prev_h_dec[:, :self.hidden_size],
                             prev_h_dec[:, self.hidden_size:])
            ), dim=1)
            canvas = canvas + self.write(h_dec)
            prev_h_dec = F.tanh(h_dec)
        return F.sigmoid(canvas)

    def forward(self, x: Tensor, glimpses: int):
        #  [B x C x H x W]
        batch_size = x.size(0)

        # canvas = torch.stack([self.init_c] * batch_size).to(x.device)
        # prev_h_dec = torch.stack([self.init_h_dec] * batch_size).to(x.device)
        # prev_h_enc = torch.stack([self.init_h_enc] * batch_size).to(x.device)

        canvas = torch.zeros((batch_size, self.in_channels, self.img_h, self.img_w)).to(x.device)
        prev_h_dec = torch.zeros((batch_size, 2*self.hidden_size)).to(x.device)
        prev_h_enc = torch.zeros((batch_size, 2*self.hidden_size)).to(x.device)

        mu, log_var = [0] * glimpses, [0] * glimpses
        for t in range(glimpses):
            x_error = x - F.sigmoid(canvas)
            r = self.read(x, x_error, prev_h_dec)
            h_enc = torch.cat(self.encoder(
                input=torch.cat((r, prev_h_dec), dim=1),
                hx=(prev_h_enc[:, :self.hidden_size],
                    prev_h_enc[:, self.hidden_size:])
            ), dim=1)
            z, mu[t], log_var[t] = self.sample_from_Q(h_enc)
            h_dec = torch.cat(self.decoder(
                input=z, hx=(prev_h_dec[:, :self.hidden_size],
                             prev_h_dec[:, self.hidden_size:])
            ), dim=1)
            canvas = canvas + self.write(h_dec)
            prev_h_dec, prev_h_enc = h_dec, h_enc

        return F.sigmoid(canvas), torch.stack(mu, dim=1), torch.stack(log_var, dim=1)

    def sample_from_Q(self, h_enc: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        s = self.Qsampler(h_enc)
        mu, log_var = s[:, :self.latent_dims], s[:, self.latent_dims:]
        eps = torch.randn_like(log_var)
        return mu + torch.exp(log_var*0.5) * eps, mu, log_var

    def write(self, h_dec: Tensor):
        w = torch.reshape(self.writer(h_dec),
                          (-1, self.in_channels, self.write_size, self.write_size))
        Fx, Fy, intensity = self.filter_bank(h_dec, 'write')
        Fx, FyT = Fx.unsqueeze(1), Fy.unsqueeze(1).transpose(-1, -2)
        return (FyT @ w @ Fx) / intensity.reshape((-1, 1, 1, 1))

    def read(self, x: Tensor, x_error: Tensor, prev_h_dec: Tensor):
        Fx, Fy, intensity = self.filter_bank(prev_h_dec, 'read')
        Fx, FyT = Fx.unsqueeze(1), Fy.unsqueeze(1).transpose(-1, -2)
        return intensity * torch.cat((Fx @ x @ FyT, Fx @ x_error @ FyT), dim=1).flatten(start_dim=1)

    def filter_bank(self, h_state: Tensor, op: str):
        img_size = max(self.img_h, self.img_w)
        if op == 'write':
            res = self.writer_attn(h_state)
            size = self.write_size
        elif op == 'read':
            res = self.reader_attn(h_state)
            size = self.read_size
        gx_, gy_, log_var, log_delta, log_intensity = tuple(
            res[:, [i]] for i in range(5))

        gx = (self.img_w+1)*(gx_+1) / 2
        gy = (self.img_h+1)*(gy_+1) / 2
        stride = (img_size-1)*torch.exp(log_delta) / (size-1)

        offsets = (torch.arange(end=size, device=h_state.device) +
                   0.5-size/2) * stride
        mu_x = torch.unsqueeze(gx + offsets, dim=-1)
        mu_y = torch.unsqueeze(gy + offsets, dim=-1)

        variance = torch.exp(log_var).unsqueeze(dim=1)
        Fx = torch.exp(-(torch.arange(end=self.img_w,
                       device=h_state.device) - mu_x)**2 / 2*variance)
        Fy = torch.exp(-(torch.arange(end=self.img_h,
                       device=h_state.device) - mu_y)**2 / 2*variance)
        return Fx, Fy, torch.exp(log_intensity)


def loss_function(x: Tensor, c: Tensor, mu: Tensor, log_var: Tensor):
    glimpses = mu.size(1)

    x = x.flatten(start_dim=1)
    c = c.flatten(start_dim=1)

    recon_loss = F.binary_cross_entropy(c, target=x, reduction='none').sum(dim=1).mean()
    latent_loss = (0.5 * ((mu**2 + log_var.exp() - log_var).sum(dim=1) - glimpses)).mean()
    return recon_loss, latent_loss


class DRAWExperiment(L.LightningModule):
    def __init__(self,
                 glimpses: int,
                 read_size: int,
                 write_size: int,
                 in_channels: int = 1,
                 img_size: Tuple[int, int] = (28, 28),
                 latent_dims: int = 100,
                 hidden_size: int = 256):
        super(DRAWExperiment, self).__init__()
        self.save_hyperparameters()

        self.glimpses = glimpses
        self.model = DRAW(in_channels, img_size, read_size,
                         write_size, latent_dims, hidden_size)

    def training_step(self, batch):
        x, _ = batch
        canvas, mu, log_var = self.model(x, self.glimpses)

        recon_loss, latent_loss = loss_function(x, canvas, mu, log_var)
        loss = recon_loss + latent_loss

        self.log('train/loss', loss, prog_bar=True)
        self.log('train/recon_loss', recon_loss)
        self.log('train/latent_loss', latent_loss)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=0.0003)
        return optimizer


def binarized(img):
    img = to_tensor(img)
    return (img > 0.5).type(torch.float32)


def main():
    cpu_count = os.cpu_count()
    if cpu_count is None: cpu_count = 4

    torch.manual_seed(1265)

    train_set = MNIST('datasets', train=True,
                      download=True, transform=binarized)
    validation_set = MNIST('datasets', train=False,
                           download=True, transform=binarized)

    # TODO: Change num_workers
    loader_args = dict(batch_size=100, pin_memory=True,
                       num_workers=1, persistent_workers=True)
    train_loader = DataLoader(train_set, **loader_args)
    val_loader = DataLoader(validation_set, **loader_args, shuffle=False)

    model = DRAWExperiment(glimpses=64, read_size=2, write_size=5)

    print(f'Training images: {len(train_set)}')
    print(f'Validation images: {len(validation_set)}')
    print(f"Batch size: {loader_args['batch_size']} Num workers: {loader_args['num_workers']}")

    # TODO: change overfit and max_epochs, log_every_n_step
    trainer = L.Trainer(
        deterministic=True,
        overfit_batches=1,
        log_every_n_steps=1,
        gradient_clip_val=10.0,
        max_epochs=500,
        accelerator='cpu',
    )
    trainer.fit(model, train_dataloaders=train_loader,
                val_dataloaders=val_loader)


if __name__ == '__main__':
    main()
