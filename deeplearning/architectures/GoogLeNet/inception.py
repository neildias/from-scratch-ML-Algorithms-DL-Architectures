import torch
import torch.nn as nn
from .convolution import ConvolutionBlock

class Inception(nn.Module):
  """Main Inception layer class"""

  def __init__(self,
               in_channels,
               out_1x1,
               reduction_3x3,
               out_3x3,
               reduction_5x5,
               out_5x5,
               out_1x1pool
               ):

    super(Inception, self).__init__()

    self.first_branch = ConvolutionBlock(in_channels,
                                         out_1x1,
                                         kernel_size=(1, 1))

    self.second_branch = nn.Sequential(

      ConvolutionBlock(in_channels,
                       reduction_3x3,
                       kernel_size=(1, 1)),
      ConvolutionBlock(reduction_3x3,
                       out_3x3,
                       kernel_size=(3, 3),
                       padding=(1, 1)),

    )

    self.third_branch = nn.Sequential(

      ConvolutionBlock(in_channels,
                       reduction_5x5,
                       kernel_size=(1, 1)),
      ConvolutionBlock(reduction_5x5,
                       out_5x5,
                       kernel_size=(5, 5),
                       padding=(2, 2)),

    )

    self.fourth_branch = nn.Sequential(

      nn.MaxPool2d(kernel_size=(3, 3),
                   stride=(1, 1),
                   padding=(1, 1)),
      ConvolutionBlock(in_channels,
                       out_1x1pool,
                       kernel_size=(1, 1)),

    )

  def forward(self, x):
    return torch.cat(
      [self.first_branch(x), self.second_branch(x), self.third_branch(x), self.fourth_branch(x)], 1
    )


class InceptionAuxillory(nn.Module):
  """Auxillory output block"""

  def __init__(self, in_channels, num_classes):

    super(InceptionAuxillory, self).__init__()

    self.relu = nn.ReLU()
    self.dropout = nn.Dropout(p=0.7)
    self.pool = nn.AvgPool2d(kernel_size=5, stride=3)
    self.conv = ConvolutionBlock(in_channels, 128, kernel_size=1)
    self.fc1 = nn.Linear(2048, 1024)
    self.fc2 = nn.Linear(1024, num_classes)

  def forward(self, x):
    x = self.pool(x)
    x = self.conv(x)
    x = x.reshape(x.shape[0], -1)
    x = self.relu(self.fc1(x))
    x = self.dropout(x)
    x = self.fc2(x)
    return x