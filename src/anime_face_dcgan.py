import os

import cv2
import matplotlib.pyplot as plt
import opendatasets as open
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.utils import make_grid
from torchvision.utils import save_image
from tqdm import tqdm

# Download the dataset
open.download("https://www.kaggle.com/soumikrakshit/anime-faces")

DATA_DIR = "anime-faces"

# Get all the files downloaded
print(os.listdir(DATA_DIR))
print(os.listdir(f"{DATA_DIR}/data")[:5])

# Important variables.
image_size = 64
batch_size = 128
stats = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5)

# Loading the data.
train_ds = ImageFolder(
    DATA_DIR,
    transform=T.Compose(
        [
            T.Resize(image_size),
            T.CenterCrop(image_size),
            T.ToTensor(),
            T.Normalize(*stats),
        ]
    ),
)

train_dl = DataLoader(train_ds, batch_size, shuffle=True, num_workers=3, pin_memory=True)


# To show the images
def show_images(images, nmax=64):
    fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.imshow(make_grid(denorm(images.detach()[:nmax]), nrow=8).permute(1, 2, 0))


# Using show_images, Display a batch of the training images
def show_batch(dl, nmax=64):
    for images, _ in dl:
        show_images(images, nmax)
        break


# Denormalization script for the image grid, which is stored in the form of a tensor.
def denorm(img_tensors):
    return img_tensors * stats[1][0] + stats[0][0]


class DeviceDataLoader:
    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        return len(self.dl)


# Basically the loader for the device for training. Supported ones: CPU, GPU (Needs CUDA Enabled)
def to_device(data, device):
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]

    return data.to(device, non_blocking=True)


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    else:
        return torch.device("cpu")


device = get_default_device()
train_dl = DeviceDataLoader(train_dl, device)

print(f"Device used: {device}")

latent_size = 128

generator = nn.Sequential(
    nn.ConvTranspose2d(
        latent_size, 512, kernel_size=4, stride=1, padding=0, bias=False
    ),
    nn.BatchNorm2d(512),
    nn.ReLU(True),

    nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.ReLU(True),

    nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.ReLU(True),

    nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.ReLU(True),

    nn.ConvTranspose2d(64, 3, kernel_size=4, stride=2, padding=1, bias=False),
    nn.Tanh(),
)

discriminator = nn.Sequential(
    nn.Conv2d(3, 64, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(64),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(128),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(256),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(256, 512, kernel_size=4, stride=2, padding=1, bias=False),
    nn.BatchNorm2d(512),
    nn.LeakyReLU(0.2, inplace=True),

    nn.Conv2d(512, 1, kernel_size=4, stride=1, padding=0, bias=False),

    nn.Flatten(),
    nn.Sigmoid(),
)

# Convert to device
discriminator = to_device(discriminator, device)

# Generate the fake images
xb = torch.randn(batch_size, latent_size, 1, 1)
fake_images = generator(xb)

# Convert gen to device too.
generator = to_device(generator, device)


def train_generator(opt_g):
    opt_g.zero_grad()

    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    preds = discriminator(fake_images)
    targets = torch.ones(batch_size, 1, device=device)
    loss = F.binary_cross_entropy(preds, targets)

    loss.backward()
    opt_g.step()

    return loss.item()


def train_discriminator(real_images, opt_d):
    opt_d.zero_grad()

    real_preds = discriminator(real_images)
    real_targets = torch.ones(real_images.size(0), 1, device=device)
    real_loss = F.binary_cross_entropy(real_preds, real_targets)
    real_score = torch.mean(real_preds).item()

    latent = torch.randn(batch_size, latent_size, 1, 1, device=device)
    fake_images = generator(latent)

    fake_targets = torch.zeros(fake_images.size(0), 1, device=device)
    fake_preds = discriminator(fake_images)
    fake_loss = F.binary_cross_entropy(fake_preds, fake_targets)
    fake_score = torch.mean(fake_preds).item()

    loss = real_loss + fake_loss
    loss.backward()
    opt_d.step()

    return loss.item(), real_score, fake_score


# Variables
sample_dir = "generated"
os.makedirs(sample_dir, exist_ok=True)


def save_samples(index, latent_tensors, show=True):
    fake_images = generator(latent_tensors)
    fake_fname = "generated-images-{0:0=4d}.png".format(index)

    save_image(denorm(fake_images), os.path.join(sample_dir, fake_fname), nrow=8)
    print("Saving", fake_fname)

    if show:
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(make_grid(fake_images.cpu().detach(), nrow=8).permute(1, 2, 0))


# TRAINING TIME!
fixed_latent = torch.randn(64, latent_size, 1, 1, device=device)


def fit(epochs, lr, start_idx=1):
    torch.cuda.empty_cache()

    losses_g, losses_d = [], []
    real_scores, fake_scores = [], []

    # Create optimizers
    opt_d = torch.optim.Adam(discriminator.parameters(), lr=lr, betas=(0.5, 0.999))
    opt_g = torch.optim.Adam(generator.parameters(), lr=lr, betas=(0.5, 0.999))

    for epoch in range(epochs):
        for real_images, _ in tqdm(train_dl):
            # Train discriminator
            loss_d, real_score, fake_score = train_discriminator(real_images, opt_d)
            # Train generator
            loss_g = train_generator(opt_g)

        # Record losses & scores
        losses_g.append(loss_g)
        losses_d.append(loss_d)
        real_scores.append(real_score)
        fake_scores.append(fake_score)

        # Log losses & scores (last batch)
        print(
            "Epoch [{}/{}], loss_g: {:.4f}, loss_d: {:.4f}, real_score: {:.4f}, fake_score: {:.4f}".format(
                epoch + 1, epochs, loss_g, loss_d, real_score, fake_score
            )
        )

        # Save generated images
        save_samples(epoch + start_idx, fixed_latent, show=False)

    return losses_g, losses_d, real_scores, fake_scores


EPOCHS = 60
LR = 0.001

history = fit(EPOCHS, LR)

# Now save the models. It's hard to train over GPU took, Almost took me 3 hours, over NVIDIA TESLA GPU.
torch.save(generator.state_dict(), "generator_model.bin")
torch.save(discriminator.state_dict(), "discriminator_model.bin")


# Record the frames as video
def save_frames_as_video(filename, images_path):
    vid_fname = filename

    files = [
        os.path.join(sample_dir, f) for f in os.listdir(sample_dir) if images_path in f
    ]
    files.sort()

    out = cv2.VideoWriter(vid_fname, cv2.VideoWriter_fourcc(*"MP4V"), 1, (530, 530))
    [out.write(cv2.imread(fname)) for fname in files]

    out.release()


save_frames_as_video("anime_face_dcgan.avi", "generated")
