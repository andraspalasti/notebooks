import argparse
import os
from pathlib import Path
from typing import Tuple

import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.loggers import WandbLogger
from torch import Tensor, nn, optim
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms.functional import to_tensor
from torchvision.utils import save_image


class DRAW(nn.Module):
    """
    An implementation of the DRAW model with attention, 
    described in the paper: https://arxiv.org/abs/1502.04623
    """

    def __init__(self, in_channels: int, img_size: Tuple[int, int], read_size: int, write_size: int, latent_dims: int = 100, hidden_size: int = 256):
        """Initializies the draw architecture.

        Args:
            in_channels: The number of channels of input images.
            img_size: Tuple containing (width, height) of the input image.
            read_size: The size of the grid of Gaussan kernels when reading.
            write_size: The size of the grid of Gaussan kernels when writing.
            latent_dims: The size of the latent dimension.
            hidden_size: The number of lstm cells to use.
        """
        super(DRAW, self).__init__()

        self.in_channels = in_channels
        self.hidden_size = hidden_size
        self.img_w, self.img_h = img_size
        self.latent_dims = latent_dims
        self.read_size = read_size
        self.write_size = write_size

        self.canvas_init = nn.Parameter(torch.zeros((self.in_channels, self.img_h, self.img_w)))
        self.h_dec_init = nn.Parameter(torch.zeros((self.hidden_size)))
        self.h_enc_init = nn.Parameter(torch.zeros((self.hidden_size)))

        self.encoder = nn.LSTMCell(
            # input_size=2*self.in_channels*self.img_h*self.img_w + self.hidden_size, # no attention
            input_size=2*self.read_size**2 + self.hidden_size,
            hidden_size=self.hidden_size,
        )
        self.decoder = nn.LSTMCell(
            input_size=self.latent_dims,
            hidden_size=self.hidden_size,
        )

        self.sampler = nn.Linear(self.hidden_size, 2*self.latent_dims)

        self.reader_attn = nn.Linear(self.hidden_size, out_features=5)
        self.writer_attn = nn.Linear(self.hidden_size, out_features=5)

        self.writer = nn.Linear(
            in_features=self.hidden_size,
            out_features=self.write_size**2,
            # out_features=self.in_channels*self.img_h*self.img_w, # no attention
        )

    def sample(self, z: Tensor):
        glimpses = z.size(0)
        canvas = [self.canvas_init] * (glimpses+1)
        h_dec, c_dec = self.h_dec_init, torch.zeros((self.hidden_size,))
        for t in range(glimpses):
            h_dec, c_dec = self.decoder(z[t], (h_dec, c_dec))
            canvas[t+1] = canvas[t] + self.write(torch.unsqueeze(h_dec, dim=0)).squeeze(dim=0)
        return canvas

    def forward(self, x: Tensor, glimpses: int):
        """Performs a forward operation on a batch of inputs.

        Args:
            x: The inputs in the shape of [B x C x H x W].
            glimpses: The number of times the network is run.
        """
        batch_size = x.size(0)

        # Initalize starting states
        canvas = torch.stack([self.canvas_init]*batch_size)
        h_dec, c_dec = torch.stack([self.h_dec_init]*batch_size), \
            torch.zeros((batch_size, self.hidden_size), device=x.device)
        h_enc, c_enc = torch.stack([self.h_enc_init]*batch_size), \
            torch.zeros((batch_size, self.hidden_size), device=x.device)

        mu, log_var = [0] * glimpses, [0] * glimpses
        for t in range(glimpses):
            x_error = x - F.sigmoid(canvas)

            r = self.read(x, x_error, h_dec).flatten(start_dim=1)
            enc_input = torch.cat((r, h_dec), dim=1)
            h_enc, c_enc = self.encoder(enc_input, (h_enc, c_enc))

            z, mu[t], log_var[t] = self.sample_from_Q(h_enc)

            h_dec, c_dec = self.decoder(z, (h_dec, c_dec))
            canvas = canvas + self.write(h_dec)

        return F.sigmoid(canvas), torch.stack(mu, dim=1), torch.stack(log_var, dim=1)

    def sample_from_Q(self, h_enc: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        """Samples Zt ~ Q(h_enc) via reparameterization trick from normal distribution.

        Args:
            h_enc: The hidden state of the encoder.

        Returns:
            A tuple of tensors (z, mu, log_var).
        """
        s = self.sampler(h_enc)
        mu, log_var = s[:, :self.latent_dims], s[:, self.latent_dims:]
        eps = torch.randn_like(log_var)
        return mu + torch.exp(log_var*0.5) * eps, mu, log_var

    def write_no_attn(self, h_dec: Tensor):
        w = self.writer(h_dec)
        return torch.reshape(w, (-1, self.in_channels, self.img_h, self.img_w))

    def write(self, h_dec: Tensor):
        """Implementation of the write operation."""
        w = torch.reshape(self.writer(h_dec),
                          (-1, self.in_channels, self.write_size, self.write_size))
        Fx, Fy, intensity = self.attn_window(h_dec, 'write')
        Fx, FyT = Fx.unsqueeze(1), Fy.permute(0, 2, 1).unsqueeze(1)
        intensity = intensity.reshape((-1, 1, 1, 1))
        return (FyT @ w @ Fx) / intensity

    def read_no_attn(self, x: Tensor, x_error: Tensor, prev_h_dec: Tensor):
        return torch.cat((x, x_error), dim=1)

    def read(self, x: Tensor, x_error: Tensor, prev_h_dec: Tensor):
        """Implementation of the read operation."""
        Fx, Fy, intensity = self.attn_window(prev_h_dec, 'read')
        FxT, Fy = Fx.transpose(-1, -2).unsqueeze(dim=1), Fy.unsqueeze(dim=1)
        intensity = intensity.reshape((-1, 1, 1, 1))
        return intensity * torch.cat((Fy @ x @ FxT, Fy @ x_error @ FxT), dim=1)

    def attn_window(self, h_state: Tensor, op: str):
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

        return (*self.filter_bank(gx, gy, torch.exp(log_var), stride, size), torch.exp(log_intensity))

    def filter_bank(self, gx: Tensor, gy: Tensor, variance: Tensor, stride: Tensor, size: int):
        offsets = (torch.arange(end=size, device=gx.device) + 0.5 - size/2) * stride
        mu_x = torch.unsqueeze(gx + offsets, dim=-1)
        mu_y = torch.unsqueeze(gy + offsets, dim=-1)
        variance = variance.unsqueeze(dim=1)

        a = torch.arange(end=self.img_w, device=gx.device)
        Fx = torch.exp(-(a - mu_x)**2 / (2*variance))

        b = torch.arange(end=self.img_h, device=gx.device)
        Fy = torch.exp(-(b - mu_y)**2 / (2*variance))

        # Normalize the generated filterbanks
        Fx = Fx / Fx.sum(dim=2, keepdim=True).clamp_min(1e-8)
        Fy = Fy / Fy.sum(dim=2, keepdim=True).clamp_min(1e-8)
        return Fx, Fy


def loss_function(x: Tensor, c: Tensor, mu: Tensor, log_var: Tensor):
    eps = 1e-8  # epsilon for numerical stability

    x = x.flatten(start_dim=1)
    c = c.flatten(start_dim=1)
    recon_loss = (-(x*(c+eps).log() + (1.0-x)*(1.0-c+eps).log())).sum(dim=1).mean()

    glimpses = mu.size(1)
    latent_loss = (0.5 * ((mu**2+log_var.exp()-log_var).sum(dim=1) - glimpses)).mean()
    return recon_loss, latent_loss


class DRAWExperiment(L.LightningModule):
    def __init__(self,
                 glimpses: int,
                 read_size: int,
                 write_size: int,
                 learning_rate: float,
                 in_channels: int = 1,
                 img_size: Tuple[int, int] = (28, 28),
                 latent_dims: int = 100,
                 hidden_size: int = 256):
        super(DRAWExperiment, self).__init__()
        self.save_hyperparameters()

        self.glimpses = glimpses
        self.learning_rate = learning_rate
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

    def on_validation_start(self):
        self.generated_images = []
        self.val_start_step = self.global_step
        self.recon_loss, self.latent_loss = 0, 0

    def validation_step(self, batch):
        with torch.no_grad():
            x, _ = batch
            canvas, mu, log_var = self.model(x, self.glimpses)
            recon_loss, latent_loss = loss_function(x, canvas, mu, log_var)
        self.recon_loss += recon_loss
        self.latent_loss += latent_loss
        if len(self.generated_images) < 30:
            self.generated_images.append((x[0], canvas[0]))

    def on_validation_end(self):
        num_batches = self.global_step - self.val_start_step
        if isinstance(self.logger, WandbLogger):
            self.logger.log_image(
                key='expected',
                images=[x for x, _ in self.generated_images]
            )
            self.logger.log_image(
                key='generated',
                images=[generated for _, generated in self.generated_images]
            )
            self.logger.log_metrics({
                'val/loss': (self.recon_loss + self.latent_loss) / num_batches,
                'val/recon_loss': self.recon_loss / num_batches,
                'val/latent_loss': self.latent_loss / num_batches,
            })

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        return optimizer


def binarized(img):
    img = to_tensor(img)
    return (img > 0.5).type(torch.float32)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bs", "--batch-size", type=int, dest="batch_size",
                default=100, help="Size of each mini-batch")
    parser.add_argument("--lr", "--learning-rate", type=float, dest="learning_rate",
                default=1e-3, help="Learning rate")
    parser.add_argument("--load", type=str, dest="load",
                default=None, help="The weights to load")
    parser.add_argument("--sample", dest="sample", action='store_true',
                default=False, help="Sample image from model (use with load)")
    parser.add_argument("--attention", "-a", type=str, default="2,5",
                help="Use attention mechanism (read_window,write_window)")
    parser.add_argument("--glimpses", type=int, dest="glimpses",
                default=20, help="No. of iterations")
    parser.add_argument("--hidden-size", type=int, dest="hidden_size",
                default=256, help="RNN state dimension")
    parser.add_argument("--z-dim", type=int, dest="z_dim",
                default=100, help="Z-vector dimension")
    return parser.parse_args()


def main():
    args = parse_args()
    read_size, write_size = (int(n) for n in args.attention.split(','))

    if args.load is not None:
        experiment = DRAWExperiment.load_from_checkpoint(args.load)
        print(f'Checkpoint succesfully loaded from: {args.load}!')
    else:
        experiment = DRAWExperiment(
            glimpses=args.glimpses, read_size=read_size,
            hidden_size=args.hidden_size, learning_rate=args.learning_rate,
            write_size=write_size, latent_dims=args.z_dim)

    if args.sample:
        save_path = Path('sample.png')
        z = torch.randn((experiment.glimpses, experiment.model.latent_dims))
        canvases = experiment.model.sample(z)
        save_image(canvases, save_path)
        print(f'Successfully saved image to: {save_path}!')
        return

    cpu_count = os.cpu_count()
    if cpu_count is None:
        cpu_count = 2

    train_set = MNIST('datasets', train=True,
                      download=True, transform=binarized)
    validation_set = MNIST('datasets', train=False,
                           download=True, transform=binarized)

    loader_args = dict(batch_size=args.batch_size, pin_memory=True, num_workers=cpu_count)
    train_loader = DataLoader(train_set, **loader_args)
    val_loader = DataLoader(validation_set, **loader_args, shuffle=False, drop_last=True)

    print(f'Training images: {len(train_set)}')
    print(f'Validation images: {len(validation_set)}')
    print(f"Batch size: {loader_args['batch_size']} Num workers: {loader_args['num_workers']}")

    wandb_logger = WandbLogger(project="DRAW MNIST")
    wandb_logger.watch(experiment, log='all')

    trainer = L.Trainer(
        deterministic=True,
        max_epochs=10,
        log_every_n_steps=10,
        gradient_clip_algorithm='norm',
        gradient_clip_val=5.0,
        logger=wandb_logger,
    )
    trainer.fit(experiment, train_dataloaders=train_loader,
                val_dataloaders=val_loader)


if __name__ == '__main__':
    torch.manual_seed(1265)
    main()
