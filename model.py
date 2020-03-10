import torch
from torch import nn
import torch.nn.functional as F

# Different channels? can put derivative? see video

N_BOTTLENECKS = 10  # T
INPUT_SIZE = 128    # M (size of each bottleneck feature)
HIDDEN_SIZE = 600   # H, TODO 500
DROPOUT = 0.4       # TODO 0.2
N_CLASSES = 10      # K


class EmbeddedMapping(nn.Module):

    def __init__(self):
        super(EmbeddedMapping, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, HIDDEN_SIZE)
        self.relu1 = nn.ReLU()
        self.drop1 = nn.Dropout(DROPOUT)
        self.fc2 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.relu2 = nn.ReLU()
        self.drop2 = nn.Dropout(DROPOUT)
        self.fc3 = nn.Linear(HIDDEN_SIZE, HIDDEN_SIZE)
        self.relu3 = nn.ReLU()
        self.drop3 = nn.Dropout(DROPOUT)  # TODO forse questo non ci deve essere

    def forward(self, x):
        h1 = self.drop1(self.relu1(self.fc1(x)))
        h2 = self.drop2(self.relu2(self.fc2(h1)))
        h = self.drop3(self.relu3(self.fc3(h2)))
        return h


class AttentionModule(nn.Module):

    def __init__(self):
        super(AttentionModule, self).__init__()
        self.fc_att = nn.Linear(HIDDEN_SIZE, N_CLASSES)
        self.softmax = nn.Softmax()
        self.fc_class = nn.Linear(HIDDEN_SIZE, N_CLASSES)
        self.sigmoid = nn.Sigmoid()

    def forward(self, h):
        v = self.softmax(self.fc_att(h))
        f = self.sigmoid(self.fc_class(h))
        # TODO La rete deve conoscere la somma da t=1 a T di v(h_t) per fare la normalizzazione (???)
        return None
