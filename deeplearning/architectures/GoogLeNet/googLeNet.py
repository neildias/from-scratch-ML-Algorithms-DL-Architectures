import torch
import torch.nn as nn
from GoogLeNet.inception import Inception, InceptionAuxillory
from GoogLeNet.convolution import ConvolutionBlock


class GoogLeNet(nn.Module):
  """Main Architecture"""

  def __init__(self,
               auxillory_logits=True,
               classes=1000):

    super(GoogLeNet, self).__init__()

    # ensure a boolean is fed in aux_logits
    assert auxillory_logits == True or auxillory_logits == False
    self.auxillory_logits = auxillory_logits

    self.conv1 = ConvolutionBlock(
      in_channels=3,
      out_channels=64,
      kernel_size=(7, 7),
      stride=(2, 2),
      padding=(3, 3),
    )

    self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.conv2 = ConvolutionBlock(

      in_channels=64,
      out_channels=192,
      kernel_size=3,
      stride=1,
      padding=1

    )

    self.pool2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    # In this order: in_channels, out_1x1, red_3x3, out_3x3, red_5x5, out_5x5, out_1x1pool
    self.inception3a = Inception(

      in_channels=192,
      out_1x1=64,
      reduction_3x3=96,
      out_3x3=128,
      reduction_5x5=16,
      out_5x5=32,
      out_1x1pool=32,

    )

    self.inception3b = Inception(

      in_channels=256,
      out_1x1=128,
      reduction_3x3=128,
      out_3x3=192,
      reduction_5x5=32,
      out_5x5=96,
      out_1x1pool=64,

    )

    self.pool3 = nn.MaxPool2d(kernel_size=(3, 3), stride=2, padding=1)

    self.inception4a = Inception(

      in_channels=480,
      out_1x1=192,
      reduction_3x3=96,
      out_3x3=208,
      reduction_5x5=16,
      out_5x5=48,
      out_1x1pool=64,

    )

    self.inception4b = Inception(

      in_channels=512,
      out_1x1=160,
      reduction_3x3=112,
      out_3x3=224,
      reduction_5x5=24,
      out_5x5=64,
      out_1x1pool=64,

    )

    self.inception4c = Inception(

      in_channels=512,
      out_1x1=128,
      reduction_3x3=128,
      out_3x3=256,
      reduction_5x5=24,
      out_5x5=64,
      out_1x1pool=64,

    )

    self.inception4d = Inception(

      in_channels=512,
      out_1x1=112,
      reduction_3x3=144,
      out_3x3=288,
      reduction_5x5=32,
      out_5x5=64,
      out_1x1pool=64,

    )

    self.inception4e = Inception(

      in_channels=528,
      out_1x1=256,
      reduction_3x3=160,
      out_3x3=320,
      reduction_5x5=32,
      out_5x5=128,
      out_1x1pool=128,

    )

    self.maxpool4 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    self.inception5a = Inception(

      in_channels=832,
      out_1x1=256,
      reduction_3x3=160,
      out_3x3=320,
      reduction_5x5=32,
      out_5x5=128,
      out_1x1pool=128,

    )

    self.inception5b = Inception(

      in_channels=832,
      out_1x1=384,
      reduction_3x3=192,
      out_3x3=384,
      reduction_5x5=48,
      out_5x5=128,
      out_1x1pool=128,

    )

    self.avgpool = nn.AvgPool2d(kernel_size=7, stride=1)
    self.dropout = nn.Dropout(p=0.4)
    self.fc1 = nn.Linear(1024, 1000)

    if self.auxillory_logits:
      self.aux1 = InceptionAuxillory(512, classes)
      self.aux2 = InceptionAuxillory(528, classes)
    else:
      self.aux1 = self.aux2 = None

  def forward(self, x):
    x = self.conv1(x)
    x = self.pool1(x)
    x = self.conv2(x)
    # x = self.conv3(x)
    x = self.pool2(x)

    x = self.inception3a(x)
    x = self.inception3b(x)
    x = self.pool3(x)

    x = self.inception4a(x)

    # first Auxiliary Softmax classifier
    if self.auxillory_logits and self.training:
      first_auxillory_output = self.aux1(x)

    x = self.inception4b(x)
    x = self.inception4c(x)
    x = self.inception4d(x)

    # Second Auxiliary Softmax classifier
    if self.auxillory_logits and self.training:
      second_auxillory_output = self.aux2(x)

    x = self.inception4e(x)
    x = self.maxpool4(x)
    x = self.inception5a(x)
    x = self.inception5b(x)
    x = self.avgpool(x)
    x = x.reshape(x.shape[0], -1)
    x = self.dropout(x)
    x = self.fc1(x)

    if self.auxillory_logits and self.training:
      return first_auxillory_output, second_auxillory_output, x
    else:
      return x


if __name__ == "__main__":
  # testing
  random_data = torch.randn((1, 3, 224, 224))
  print(random_data.shape)
  test = GoogLeNet()
  auxillory_output1, auxillory_output2, model_output = test(random_data)

  # should give 1, 1000
  print("\n=============Auxillory logits turned on=============\n")
  print(auxillory_output1.shape)
  print(auxillory_output2.shape)
  print(model_output.shape)

  print("\n=============Auxillory logits turned off=============\n")

  test = GoogLeNet(auxillory_logits=False)
  model_output = test(random_data)
  print(model_output.shape)