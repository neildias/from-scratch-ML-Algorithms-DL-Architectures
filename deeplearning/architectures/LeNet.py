import torch
from torch import nn


class LeNet(nn.Module):

  def __init__(self):
    super(LeNet, self).__init__()
    # convolution layers
    self.conv1 = nn.Conv2d(
      in_channels=1,
      out_channels=6,
      kernel_size=(5,5),
      stride=(1,1),
      padding=(0,0)
    )

    self.conv2 = nn.Conv2d(
      in_channels=6,
      out_channels=16,
      kernel_size=(5, 5),
      stride=(1, 1),
      padding=(0, 0)
    )

    self.conv3 = nn.Conv2d(
      in_channels=16,
      out_channels=120,
      kernel_size=(5, 5),
      stride=(1, 1),
      padding=(0, 0)
    )

    self.pool = nn.AvgPool2d(
      kernel_size=(2,2),
      stride=(2,2)
    )

    self.fc1 = nn.Linear(120, 84)
    self.fc2 = nn.Linear(84,10)

  def forward(self, x):
    x = nn.ReLU()(self.conv1(x))
    x = self.pool(x)
    x = nn.ReLU()(self.conv2(x))
    x = self.pool(x)
    x = nn.ReLU()(self.conv3(x))
    x = x.reshape(x.shape[0],-1)
    x = nn.ReLU()(self.fc1(x))
    return self.fc2(x)


if __name__ == "__main__":
  test = LeNet()
  random_data = torch.randn((64, 1, 32, 32))
  print(random_data.shape)
  test(random_data)