import torch
import torch.nn as nn
import torch.nn.functional as F

class pNet(nn.Module):
    def __init__(self):
        super(pNet, self).__init__()
        self.conv1 = nn.Conv1d(1, 128, kernel_size=5)
        self.dropout1 = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(128)
        self.max_pool1 = nn.MaxPool1d(kernel_size=2)
        self.flatten = nn.Flatten()
        self.dense1 = nn.Linear(128 * 13, 100)
        self.dropout2 = nn.Dropout(0.3)
        self.dense2 = nn.Linear(100, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.dropout1(x)
        x = self.batch_norm1(x)
        x = self.max_pool1(x)
        x = self.flatten(x)
        x = F.relu(self.dense1(x))
        x = self.dropout2(x)
        x = F.softmax(self.dense2(x), dim=1)
        return x

