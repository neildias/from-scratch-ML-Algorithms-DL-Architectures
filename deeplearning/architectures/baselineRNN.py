import torch
from torch import nn


sequence_length = 28 # for mnist

class BasicRNN(nn.Module):

  def __init__(self,
               input_size,
               hidden_size,
               num_layers,
               classes):
    super(BasicRNN, self).__init__()

    self.hidden =  hidden_size
    self.num_layers = num_layers
    self.RNN = nn.RNN(input_size=input_size,
                      hidden_size=hidden_size,
                      num_layers=num_layers,
                      batch_first=True)
    self.fc = nn.Linear(hidden_size*sequence_length, classes) #28X28 where 28 is the sequence length


  def forward(self, x):
    h_0 = torch.zeros(self.num_layers, x.size(0), self.hidden)
    # fp
    out, hidden_state = self.RNN(x, h_0)
    out = out.reshape(out.shape[0],-1)
    return self.fc(out)

if __name__ == "__main__":
  test = BasicRNN(input_size=28,
                 hidden_size=28,
                 num_layers=2,
                 classes=10)
  random_data = torch.randn((64, 28, 28))
  print(random_data.shape)
  test(random_data)