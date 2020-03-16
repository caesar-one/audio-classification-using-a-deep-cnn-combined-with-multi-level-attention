import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50
from tqdm import tqdm
from torchvision import transforms

# TODO Different channels? can put derivative? see video

T = 4  # number of bottleneck features, TODO it depends on audio clip length? (paper2)
M = 2048  # size of a bottleneck feature
H = 600  # size of hidden layers, TODO 500
DR = 0.4  # dropout rate, TODO 0.2
K = 10  # number of classes


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def initialize_cnn(num_classes,  # number of output classes (makes sense only if combined with just_bottleneck=False)
                   use_pretrained=True,  # Uses a pretrained version of resnet50
                   just_bottleneck=False,  # if =True the FC part is removed, so that the model returns bottlenecks.
                   cnn_trainable=False,  # if =True CNN part is trainable. Otherwise the gradient will NOT be calculated
                   first_cnn_layer_trainable=False,  # Sets the first CNN layer trainable, to optimize for the dataset
                   in_channels=3):  # the the number of input channels is reshaped

    m = resnet50(pretrained=use_pretrained)
    # input_size = 224

    if not cnn_trainable:
        set_requires_grad(m, False)

    if first_cnn_layer_trainable:
        if in_channels == 3:
            set_requires_grad(m.conv1, True)
        else:
            m.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3,
                                bias=False)

    if just_bottleneck:
        modules = list(m.children())[:-1]  # delete the last fc layer.
        modules.append(CnnFlatten())  # TODO Rivedere questa istruzione (e solo questa!)
        m = nn.Sequential(*modules)

    else:
        num_ftrs = m.fc.in_features
        m.fc = nn.Linear(num_ftrs, num_classes)

    return m  # , input_size


class CnnFlatten(nn.Module):
    def forward(self, input):
        return torch.flatten(input, 1)  # TODO check second parameter (it should be correct considering batches)


# Implements the block g (green big block in the main paper)
class EmbeddedMapping(nn.Module):

    def __init__(self, n_fc, is_first):
        super(EmbeddedMapping, self).__init__()
        self.n_fc = n_fc
        if is_first:
            self.fc = [nn.Linear(M, H)] + [nn.Linear(H, H) for _ in range(n_fc - 1)]
        else:
            self.fc = [nn.Linear(H, H) for _ in range(n_fc)]

    # Input x has shape (batch_size, T, M) if is_first=True
    # otherwise x has shape (batch_size, T, H)
    def forward(self, x):
        emb = x
        for i in range(self.n_fc):
            emb = F.dropout(F.relu(self.fc[i](emb)), p=DR)  # TODO forse l'ultimo dropout non ci deve essere
        # Output emb has shape (batch_size, T, H)
        return emb


# Implements the blocks v, f, and p (orange big block in the main paper)
class AttentionModule(nn.Module):

    def __init__(self):
        super(AttentionModule, self).__init__()
        self.fcv = nn.Linear(H, K)
        self.fcf = nn.Linear(H, K)

    # Input h has shape (batch_size, T, H)
    def forward(self, h):
        att = F.softmax(self.fcv(h), dim=2)  # attention, shape (batch_size, T, K)
        cla = torch.sigmoid(self.fcv(h))  # classification, shape (batch_size, T, K)
        norm_att = att / torch.sum(att, dim=1)[:, None, :]  # normalized attention, shape (batch_size, T, K)
        y = torch.sum(cla * norm_att, dim=1)
        # Output y has size (batch_size, K)
        return y


# Implements the multi-level attention model
class MultiLevelAttention(nn.Module):

    def __init__(self, model_conf):
        super(MultiLevelAttention, self).__init__()
        self.model = model_conf
        self.embedded_mappings = [EmbeddedMapping(model_conf[0], is_first=True)] + \
                                 [EmbeddedMapping(n_layers, is_first=False) for n_layers in model_conf[1:]]
        self.attention_modules = [AttentionModule() for _ in model_conf]
        self.fc = nn.Linear(len(model_conf) * K, K)

    def forward(self, x):
        # embs contains the outputs of all the embedding layers
        embs = [self.embedded_mappings[0](x)]
        for i in range(1, len(self.model)):
            embs.append(self.embedded_mappings[i](embs[i - 1]))
        # ys contains the outputs of all the attention modules
        ys = []
        for i in range(len(self.model)):
            ys.append(self.attention_modules[i](embs[i]))
        conc_ys = torch.cat(ys, dim=1)
        out = torch.sigmoid(self.fc(conc_ys))
        return out


class Input(nn.Module):

    def __init__(self, input_conf):
        super(Input, self).__init__()
        self.conf = input_conf

    def forward(self, x):
        if self.conf == "repeat":
            x_out = torch.cat([x, x, x], dim=2)
        elif self.conf == "single":
            x_out = torch.cat([x, torch.zeros(x.shape), torch.zeros(x.shape)], dim=2)
        else:
            raise Exception("Invalid input type")

        # normalize = transforms.Compose([
        #    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])

        # TODO handle channels differently?
        _min, _max = torch.min(x_out), torch.max(x_out)
        x_out = (x_out - _min) / (_max - _min)

        x_out[:, :, 0, :, :] = (x_out[:, :, 0, :, :] - 0.485) / 0.229
        x_out[:, :, 1, :, :] = (x_out[:, :, 1, :, :] - 0.456) / 0.224
        x_out[:, :, 2, :, :] = (x_out[:, :, 2, :, :] - 0.406) / 0.225

        return x_out.reshape((-1, 3, 224, 224))


class Ensemble(nn.Module):

    def __init__(self, input_conf, cnn_conf, model_conf):
        super(Ensemble, self).__init__()
        self.input = Input(input_conf)
        self.cnn = initialize_cnn(**cnn_conf)
        self.mla = MultiLevelAttention(model_conf)

    def forward(self, x):
        x_proc = self.input(x)
        features = self.cnn(x_proc)
        print(features.shape)
        out = self.mla(features.reshape(-1, 4, M))
        return out
