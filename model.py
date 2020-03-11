import numpy as np
from torch import nn
import torch.nn.functional as F

# TODO Different channels? can put derivative? see video

T = 10    # number of bottleneck features, TODO it depends on audio clip length? (paper2)
M = 128   # size of a bottleneck feature
H = 600   # size of hidden layers, TODO 500
DR = 0.4  # dropout rate, TODO 0.2
L = 6     # number of embedded mappings
K = 10    # number of classes


# Implements the block g (green big block in the main paper)
class EmbeddedMapping(nn.Module):

    def __init__(self):
        super(EmbeddedMapping, self).__init__()
        self.fc1, self.fc2, self.fc3 = [], [], []
        for t in range(T):
            self.fc1[t] = nn.Linear(M, H)
            self.fc2[t] = nn.Linear(H, H)
            self.fc3[t] = nn.Linear(H, H)

    # Input x has size (T, M)
    def forward(self, x):
        h1, h2, h = np.zeros((T, H)), np.zeros((T, H)), np.zeros((T, H))
        for t in range(T):
            h1[t] = F.dropout(F.relu(self.fc1[t](x[t])), DR)
            h2[t] = F.dropout(F.relu(self.fc2[t](h1[t])), DR)
            h[t] = F.dropout(F.relu(self.fc3[t](h2[t])), DR)   # TODO forse l'ultimo dropout non ci deve essere
        # Output h has size (T, H)
        return h


# Implements the blocks v, f, and p (orange big block in the main paper)
class AttentionModule(nn.Module):

    def __init__(self):
        super(AttentionModule, self).__init__()
        self.fcv, self.fcf = [], []
        for t in range(T):
            self.fcv[t] = nn.Linear(H, K)
            self.fcf[t] = nn.Linear(H, K)

    # Input h has size (T, H)
    def forward(self, h):
        v, f = np.zeros((T, K)), np.zeros((T, K))
        for t in range(T):
            v[t] = F.softmax(self.fcv[t](h[t]))
            f[t] = F.sigmoid(self.fcv[t](h[t]))
        p = v / np.sum(v, 0)
        y = np.sum(p, 0)
        # Output y has size (K)
        return y
