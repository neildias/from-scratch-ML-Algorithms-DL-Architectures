from torch import nn


class ConvolutionBlock(nn.Module):
  def __init__(
        self, in_channels, out_channels, kernel_size, stride, padding, groups=1
  ):
    super(ConvolutionBlock, self).__init__()
    self.cnn = nn.Conv2d(
      in_channels,
      out_channels,
      kernel_size,
      stride,
      padding,
      groups=groups,
      bias=False,
    )
    self.bn = nn.BatchNorm2d(out_channels)
    self.silu = nn.SiLU()

  def forward(self, x):
    return self.silu(self.bn(self.cnn(x)))