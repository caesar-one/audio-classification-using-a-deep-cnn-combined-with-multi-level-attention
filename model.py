import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50
from tqdm import tqdm

# TODO Different channels? can put derivative? see video

T = 4     # number of bottleneck features, TODO it depends on audio clip length? (paper2)
M = 128   # size of a bottleneck feature
H = 600   # size of hidden layers, TODO 500
DR = 0.4  # dropout rate, TODO 0.2
L = 6     # number of levels
K = 10    # number of classes


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def initialize_cnn(num_classes, # number of output classes (makes sense only if combined with just_bottleneck=False)
                   use_pretrained=True, # Uses a pretrained version of resnet50
                   just_bottleneck=False, # if =True the FC part is removed, so that the model returns bottlenecks.
                   cnn_trainable=False, # if =True CNN part is trainable. Otherwise the gradient will NOT be calculated
                   first_cnn_layer_trainable=False, # Sets the first CNN layer trainable, to optimize for the dataset
                   in_channels=3): # the the number of input channels is reshaped

    m = resnet50(pretrained=use_pretrained)
    input_size = 224

    if not cnn_trainable:
        set_requires_grad(m, False)

    if first_cnn_layer_trainable:
        if in_channels == 3:
            set_requires_grad(m.conv1, True)
        else:
            m.conv1=nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3, bias=False)

    if just_bottleneck:
        modules = list(m.children())[:-1]  # delete the last fc layer.
        modules.append(CnnFlatten()) # TODO Rivedere questa istruzione (e solo questa!)
        m = nn.Sequential(*modules)

    else:
        num_ftrs = m.fc.in_features
        m.fc = nn.Linear(num_ftrs, num_classes)

    return m, input_size


class CnnFlatten(nn.Module):
    def forward(self, input):
        return torch.flatten(input, 1) #TODO check second parameter (it should be correct considering batches)


# Implements the block g (green big block in the main paper)
class EmbeddedMappingCat(nn.Module):

    def __init__(self):
        super(EmbeddedMappingCat, self).__init__()
        self.fc1 = [nn.Linear(M, H) for _ in range(T)]
        self.fc2 = [nn.Linear(H, H) for _ in range(T)]
        self.fc3 = [nn.Linear(H, H) for _ in range(T)]

    # Input x has size (T, M)
    def forward(self, x):
        h1 = torch.tensor(()).new_zeros((T, H))
        h2 = torch.tensor(()).new_zeros((T, H))
        h = torch.tensor(()).new_zeros((T, H))
        for t in range(T):
            h1[t] = F.dropout(F.relu(self.fc1[t](x[t])), DR)
            h2[t] = F.dropout(F.relu(self.fc2[t](h1[t])), DR)
            h[t] = F.dropout(F.relu(self.fc3[t](h2[t])), DR)   # TODO forse l'ultimo dropout non ci deve essere
        # Output h has size (T, H)
        return h


class EmbeddedMapping(nn.Module):

    def __init__(self):
        super(EmbeddedMapping, self).__init__()
        self.fc1 = nn.Linear(M, H)
        self.fc2 = nn.Linear(H, H)
        self.fc3 = nn.Linear(H, H)

    # Input x has size M
    def forward(self, x):
        h1 = F.dropout(F.relu(self.fc1(x)), DR)
        h2 = F.dropout(F.relu(self.fc2(h1)), DR)
        h = F.dropout(F.relu(self.fc3(h2)), DR)   # TODO forse l'ultimo dropout non ci deve essere
        # Output h has size H
        return h


# Implements the blocks v, f, and p (orange big block in the main paper)
class AttentionModule(nn.Module):

    def __init__(self):
        super(AttentionModule, self).__init__()
        self.fcv = nn.Linear(H, K)
        self.fcf = nn.Linear(H, K)

    # Input h has size (T, H)
    def forward(self, h):
        v = F.softmax(self.fcv(h))  # attention
        f = F.sigmoid(self.fcv(h))  # classification
        p = v / np.sum(v, 0)  # normalized attention
        y = np.sum(f * p, 0)
        # Output y has size (K)
        return y


# Implements the multi-level attention model
class MultiLevelAttention(nn.Module):

    def __init__(self, model):
        super(MultiLevelAttention, self).__init__()

        self.model = model

    def forward(self, h):
        pass


if __name__ == '__main__':
    np.random.seed(0)
    Xtrain = torch.from_numpy(np.random.random((3, T, H))).float()

    use_cuda = torch.cuda.is_available()
    print("CUDA available: " + str(use_cuda))
    device = torch.device("cuda" if use_cuda else "cpu")

    net = AttentionModule()
    net.to(device)
    net.train()
    with torch.no_grad():
        for epoch in tqdm(range(9)):
            for Xb in Xtrain:
                v, f, p, y = net(Xb)
    print(y)