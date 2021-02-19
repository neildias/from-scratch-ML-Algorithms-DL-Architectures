from torch import nn


class SqueezeExcitation(nn.Module):

  def __init__(self, in_channels, reduced_dim):
    super(SqueezeExcitation, self).__init__()
    self.squeezeExcitation = nn.Sequential(
      nn.AdaptiveAvgPool2d(1),
      nn.Conv2d(in_channels, reduced_dim, 1),
      nn.SiLU(),
      nn.Conv2d(reduced_dim, in_channels, 1),
      nn.Sigmoid(),
    )

  def forward(self, x):
    return x * self.squeezeExcitation(x)