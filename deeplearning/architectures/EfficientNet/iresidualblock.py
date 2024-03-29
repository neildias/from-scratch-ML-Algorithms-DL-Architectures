from torch import nn
import torch
from EfficientNet.convolutionBlock import ConvolutionBlock
from EfficientNet.sExcitation import SqueezeExcitation


class InvertedResidualBlock(nn.Module):

  def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        expand_ratio,
        reduction=4,  # squeeze excitation
        survival_prob=0.8,  # for stochastic depth
  ):
    super(InvertedResidualBlock, self).__init__()
    self.survival_prob = 0.8
    self.use_residual = in_channels == out_channels and stride == 1
    hidden_dim = in_channels * expand_ratio
    self.expand = in_channels != hidden_dim
    reduced_dim = int(in_channels / reduction)

    if self.expand:
      self.expand_conv = ConvolutionBlock(
        in_channels, hidden_dim, kernel_size=3, stride=1, padding=1,
      )

    self.conv = nn.Sequential(
      ConvolutionBlock(
        hidden_dim, hidden_dim, kernel_size, stride, padding, groups=hidden_dim,
      ),
      SqueezeExcitation(hidden_dim, reduced_dim),
      nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
      nn.BatchNorm2d(out_channels),
    )

  def stochastic_depth(self, x):
    if not self.training:
      return x

    binary_tensor = torch.rand(x.shape[0], 1, 1, 1, device=x.device) < self.survival_prob
    return torch.div(x, self.survival_prob) * binary_tensor

  def forward(self, inputs):
    x = self.expand_conv(inputs) if self.expand else inputs

    if self.use_residual:
      return self.stochastic_depth(self.conv(x)) + inputs
    else:
      return self.conv(x)