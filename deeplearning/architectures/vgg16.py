import torch
from torch import nn

class VGG16(nn.Module):

  model_architecture = [
    64, 64, "pool",
    128, 128, "pool",
    256, 256, 256, "pool",
    512, 512, 512, "pool",
    512, 512, 512, "pool",
  ]

  def __init__(self, in_channels=3, classes=100):
    super(VGG16, self).__init__()
    self.in_channels = in_channels
    self.convolution_layers = self.vgg_architecture(self.model_architecture)
    self.fc_layers = nn.Sequential(
      nn.Linear(512, 4096),
      nn.ReLU(),
      nn.Dropout(p=0.5),
      nn.Linear(4096, 4096),
      nn.ReLU(),
      nn.Dropout(p=0.5),
      nn.Linear(4096,classes)
    )

  def vgg_architecture(self, design):
    layers = []
    in_channels = self.in_channels

    for layer in design:
      if isinstance(layer, int):
        out_channel = layer

        layers.extend([nn.Conv2d(in_channels=in_channels,
                                 out_channels=out_channel,
                                 kernel_size=(3,3),
                                 stride=(1,1),
                                 padding=(1,1)),
                       nn.BatchNorm2d(layer),
                       nn.ReLU(layer)])
        in_channels = layer

      else:
        layers.extend([nn.MaxPool2d(kernel_size=(2,2), stride=(2,2))])

    return nn.Sequential(*layers)

  def forward(self, x):
    x = self.convolution_layers(x)
    x = x.reshape(x.shape[0],-1)
    return self.fc_layers(x)


if __name__ == "__main__":
  test = VGG16()
  random_data = torch.randn((4, 3, 56, 56))
  print(random_data.shape)
  test(random_data)