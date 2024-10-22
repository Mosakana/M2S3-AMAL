import torch.nn as nn
import torch

class Seq2Seq(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(Seq2Seq, self).__init__()
        self.embedding = nn.Embedding(128)
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h0 = torch.randn(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)
        c0 = torch.randn(self.lstm.num_layers, x.size(0), self.lstm.hidden_size).to(x.device)

        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])

        return out


