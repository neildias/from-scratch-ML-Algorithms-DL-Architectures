import torch
from torch import nn
import torch.nn.functional as functional


# baseline model template
class SuperBasicCNN(nn.Module):
  
  def __init__(self, in_channel=3, classes=100):
    super(SuperBasicCNN, self).__init__()
    self.conv = nn.Conv2d(in_channels=in_channel,
                           out_channels=8,
                           kernel_size=(3,3),

                           #same convolution
                           stride=(1,1),
                           padding=(1,1))

    self.pool = nn.MaxPool2d(kernel_size=(2,2),
                              stride=(2,2))

    self.fc = nn.Linear(in_features=2048,
                        out_features=classes)

  def forward(self, x):
    x = functional.relu(self.conv(x))
    x = self.pool(x)
    #flatten
    x = x.reshape(x.shape[0], -1)
    x = self.fc(x)
    return x


if __name__ == "__main__":
  # testing the model with one random pass
  test = SuperBasicCNN()
  random_data = torch.randn((64, 3, 32, 32))
  print(random_data.shape)
  test(random_data)