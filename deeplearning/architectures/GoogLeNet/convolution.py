import torch.nn as nn


class ConvolutionBlock(nn.Module):
  """Standard Convolution Block"""
  def __init__(self,
               in_channels,
               out_channels,
               **kwargs):

    super(ConvolutionBlock, self).__init__()
    self.relu = nn.ReLU()
    self.conv = nn.Conv2d(in_channels,
                          out_channels,
                          **kwargs)
    self.bn = nn.BatchNorm2d(out_channels)

  def forward(self, x):
    return self.relu(self.bn(self.conv(x)))