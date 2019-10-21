import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
from torch.optim import SGD
import numpy as np

np.random.seed(10)
W = np.random.randn(3, 3, 1, 1)
W = np.reshape(W, (1, 1, 3, 3))
W_t = torch.from_numpy(W)


def get_d_loss(d_logits_real, d_logits_fake):

    d_loss_r = torch.mean(torch.max(torch.tensor(0.0).float(), 1 - d_logits_real))
    d_loss_f = torch.mean(torch.max(torch.tensor(0.0).float(), 1 + d_logits_fake))

    return d_loss_r, d_loss_f

class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

def make_discriminator_model():
  model = nn.Sequential(
      spectral_norm(nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(1, 1), padding=1)))

  return model


if __name__ == '__main__':

    generated_image = torch.ones(1, 1, 3, 3)
    input_image = torch.ones(1, 1, 3, 3)
    discriminator = make_discriminator_model()
    _ = discriminator(input_image)
    discriminator[0].weight = W_t
    print(discriminator[0].weight[0, 0, :, :])
    print(discriminator[0].weight.size())
    optimizer = SGD(discriminator.parameters(), lr=0.1)
    discriminator.zero_grad()
    discriminator.train()
    d_logits_real = discriminator(input_image)
    d_logits_fake = discriminator(generated_image)
    d_loss_r, d_loss_f = get_d_loss(d_logits_real, d_logits_fake)
    d_loss = d_loss_r + d_loss_f
    print(d_loss)
    print(discriminator[0].weight[0, 0, :, :])
    print(discriminator[0].weight.size())
    d_loss.backward()
    optimizer.step()
    print(discriminator[0].weight[0, 0, :, :])
    print('Output:', d_logits_fake)

