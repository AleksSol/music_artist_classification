import torch
import torch.nn as nn
import torch.nn.functional as F


class CRNN(nn.Module):
    """
    Convolutional Recurrent Neural Network for song-based artist recognition.
    """

    def __init__(self, freq_dim, channel_dim, num_classes,
                       filter_sizes=(64, 128, 128, 128), 
                       pool_sizes=((2, 2), (4, 2), (4, 2), (4, 2))):
        """
        :param freq_dim: size of frequency dimension
        :param channel_dim: number of channels (usually 1)
        :param num_classes: number of artists
        :param filter_sizes: sizes of convolutional filters
        :param pool_sizes: pooling strides
        """

        # TODO Parametrization
        super().__init__()

        self.bn_freq = nn.BatchNorm2d(freq_dim)
        self.conv = []

        prev_fs = channel_dim
        w = 1

        for fs, ps in zip(filter_sizes, pool_sizes):
            self.conv.extend([
                nn.Conv2d(prev_fs, fs, kernel_size=3, padding=1),
                nn.ELU(),
                nn.BatchNorm2d(fs),
                nn.MaxPool2d(ps),
                nn.Dropout(0.1)
            ])
            prev_fs = fs
            w *= ps[0]

        self.conv = nn.Sequential(*self.conv)
        self.gru = nn.ModuleList([
            nn.GRU(fs * (freq_dim // w), 32),
            nn.GRU(32, 32)
        ])
        self.output = nn.Sequential(
            nn.Dropout(0.3),
            nn.Linear(32, num_classes)
        )


    def forward(self, x):
        """
        :param x: tensor of shape (batch_size, channel_dim, freq_dim, time_dim)
        :return: logits of classes (to use with nn.CrossEntropyLoss)
        """

        z = x.transpose(2, 1)
        z = self.bn_freq(z)
        z = z.transpose(2, 1)

        z = self.conv(z)
        z = z.permute(3, 0, 1, 2)
        z = z.view(z.shape[0], z.shape[1], -1)

        for gru in self.gru:
            z, _ = gru(z)

        z = self.output(z[-1])
        return z
