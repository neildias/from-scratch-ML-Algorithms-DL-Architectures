import torch
from torch import nn
import torch.nn.functional as functional


input_size = 28
sequence_len = 28
num_layer = 2
hidden_size = 128

class BaselineBDRNN(nn.Module):

  def __init__(self,
               input_size,
               hidden_sz,
               num_layers,
               classes):
    super(BaselineBDRNN, self).__init__()
    self.hidden = hidden_sz
    self.numLayer = num_layer
    self.lstm = nn.LSTM(input_size,
                        hidden_sz,
                        num_layers,
                        batch_first=True,
                        bidirectional=True)
    self.fc = nn.Linear(56, classes)

  def forward(self, x):
    hidden_state = torch.zeros(self.numLayer*2, x.size(0), self.hidden)
    cell_state = torch.zeros(self.numLayer * 2, x.size(0), self.hidden)

    out, _ = self.lstm(x, (hidden_state, cell_state))
    return self.fc(out[:,-1,:])

if __name__ == "__main__":
  # testing the network
  test = BaselineBDRNN(input_size=28,
                       hidden_sz=28,
                       num_layers=2,
                       classes=10)
  random_data = torch.randn((64, 28, 28))
  print(random_data.shape)
  test.forward(random_data)