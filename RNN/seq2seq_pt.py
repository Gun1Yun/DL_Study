# sequence to sequence with pytorch
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from ..Base.layers import TimeLSTM


class Encoder(nn.Module):
    def __init__(self, time_size, hidden_size, feature_size):
        super(Encoder, self).__init__()
        self.time_size = time_size
        self.hidden_size = hidden_size
        self.feature_size = feature_size

        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, batch_first=True)

    def forward(self, xs):
        """
        xd : batch, timestep, features
        """
        outputs, (hidden, cell) = self.lstm(xs)

        return hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_size, hidden_size, feature_size):
        super(Decoder, self).__init__()

        self.hidden_size = hidden_size
        self.feature_size = feature_size
        self.output_size = output_size

        self.lstm = nn.LSTM(input_size=feature_size, hidden_size=hidden_size, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, xs, hidden, cell):
        """
        xs : batch, time, feture
        """
        output, (hidden, cell) = self.lstm(xs, (hidden, cell))
        output = self.fc1(output.squeeze(0))
        output = self.fc2(output)

        return output, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super(Seq2Seq, self).__init__()

        self.encoder = encoder
        self.decoder = decoder
        self.device = device

    def forward(self, xs):
        hidden, cell = self.encoder(xs)
        output, hidden, cell = self.decoder(torch.zeros(xs.size(0), 1, 9), hidden, cell)

        return output


def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param.data, -0.08, 0.08)
