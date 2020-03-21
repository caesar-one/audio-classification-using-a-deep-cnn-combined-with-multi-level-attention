from typing import List, Dict, Union

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50

# TODO Different channels? can put derivative? see video

s_resnet_shape = (224, 224)
s_vggish_shape = (96, 64)

T = 10  # number of bottleneck features
M = 2048  # size of a bottleneck feature
H = 600  # size of hidden layers
DR = 0.4  # dropout rate
K = 10  # number of classes


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


def initialize_cnn(#num_classes,  # number of output classes (makes sense only if combined with just_bottleneck=False)
                   use_pretrained=True,  # Uses a pretrained version of resnet50
                   #just_bottleneck=False,  # if =True the FC part is removed, so that the model returns bottlenecks.
                   cnn_trainable=False,  # if =True CNN part is trainable. Otherwise the gradient will NOT be calculated
                   first_cnn_layer_trainable=False,  # Sets the first CNN layer trainable, to optimize for the dataset
                   in_channels=3):  # the the number of input channels is reshaped

    m = resnet50(pretrained=use_pretrained)

    if not cnn_trainable:
        set_requires_grad(m, False)

    if first_cnn_layer_trainable:
        if in_channels == 3:
            set_requires_grad(m.conv1, True)
        else:
            m.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3,
                                bias=False)
    '''
    if just_bottleneck:
        modules = list(m.children())[:-1]  # delete the last fc layer.
        modules.append(CnnFlatten())  # TODO Rivedere questa istruzione (e solo questa!)
        m = nn.Sequential(*modules)

    else:
        num_ftrs = m.fc.in_features
        m.fc = nn.Linear(num_ftrs, num_classes)
    '''
    modules = list(m.children())[:-1]  # delete the last fc layer.
    modules.append(CnnFlatten())  # TODO Rivedere questa istruzione (e solo questa!)
    m = nn.Sequential(*modules)

    return m  # , input_size


class ResNet50_ft(nn.Module):
    def __init__(self, num_classes=10,  # number of output classes (makes sense if combined with just_bottleneck=False)
                 use_pretrained=True,  # Uses a pretrained version of resnet50
                 just_bottleneck=False,  # if =True the FC part is removed, so that the model returns bottlenecks.
                 cnn_trainable=False,  # if =True CNN part is trainable. Otherwise the gradient will NOT be calculated
                 first_cnn_layer_trainable=False,  # Sets the first CNN layer trainable, to optimize for the dataset
                 in_channels=3):  # the the number of input channels is reshaped
        super(ResNet50_ft, self).__init__()
        self.resnet_model = resnet50(pretrained=use_pretrained)

        if not cnn_trainable:
            set_requires_grad(self.resnet_model, False)

        if first_cnn_layer_trainable:
            if in_channels == 3:
                set_requires_grad(self.resnet_model.conv1, True)
            else:
                self.resnet_model.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2,
                                                    padding=3,
                                                    bias=False)
        '''
        if just_bottleneck:
            modules = list(self.resnet_model.children())[:-1]  # delete the last fc layer.
            # modules.append(CnnFlatten())  # TODO Rivedere questa istruzione (e solo questa!)
            self.resnet_model = nn.Sequential(*modules)

        else:
            num_ftrs = self.resnet_model.fc.in_features
            self.resnet_model.fc = nn.Linear(num_ftrs, num_classes)
        '''
        modules = list(self.resnet_model.children())[:-1]  # delete the last fc layer.
        # modules.append(CnnFlatten())  # TODO Rivedere questa istruzione (e solo questa!)
        self.resnet_model = nn.Sequential(*modules)
    def forward(self, x):
        x = self.resnet_model(x)
        return torch.flatten(x, 1)


class CnnFlatten(nn.Module):
    def forward(self, x):
        return torch.flatten(x, 1)  # TODO check second parameter (it should be correct considering batches)


# Implements the block g (green big block in the main paper)
class EmbeddedMapping(nn.Module):

    def __init__(self, n_fc, is_first):
        super(EmbeddedMapping, self).__init__()
        self.n_fc = n_fc
        self.norm0 = nn.BatchNorm1d(T)

        if is_first:
            self.fc = nn.ModuleList([nn.Linear(M, H)] + [nn.Linear(H, H) for _ in range(n_fc - 1)])
        else:
            self.fc = nn.ModuleList([nn.Linear(H, H) for _ in range(n_fc)])

        self.norms = nn.ModuleList([nn.BatchNorm1d(T) for _ in range(n_fc)])
        '''
        if is_first:
            fc_list = [nn.Linear(M, H)] + [nn.Linear(H, H) for _ in range(n_fc - 1)]
        else:
            fc_list = [nn.Linear(H, H) for _ in range(n_fc)]

        modules = []
        for i in range(self.n_fc):
            modules.append(fc_list[i])
            modules.append(nn.ReLU())
            modules.append(nn.Dropout(p=DR))

        self.emb = nn.Sequential(*modules)
        '''

    # Input x has shape (batch_size, T, M) if is_first=True
    # otherwise x has shape (batch_size, T, H)
    def forward(self, x):
        x = self.norm0(x)
        for i in range(self.n_fc):
            x = F.dropout(F.relu(self.norms[i](self.fc[i](x))), p=DR)
        # Output emb has shape (batch_size, T, H)
        return x
        '''
        return self.emb(x)
        '''


# Implements the blocks v, f, and p (orange big block in the main paper)
class AttentionModule(nn.Module):

    def __init__(self):
        super(AttentionModule, self).__init__()
        self.fcv = nn.Linear(H, K)
        self.fcf = nn.Linear(H, K)
        self.normv = nn.BatchNorm1d(T)
        self.normf = nn.BatchNorm1d(T)

    # Input h has shape (batch_size, T, H)
    def forward(self, h):
        att = F.softmax(self.normv(self.fcv(h)), dim=2)  # attention, shape (batch_size, T, K)
        cla = torch.sigmoid(self.normf(self.fcv(h)))  # classification, shape (batch_size, T, K)
        norm_att = att / torch.sum(att, dim=1)[:, None, :]  # normalized attention, shape (batch_size, T, K)
        y = torch.sum(cla * norm_att, dim=1)
        # Output y has size (batch_size, K)
        return y


# Implements the multi-level attention model
class MultiLevelAttention(nn.Module):

    def __init__(self, model_conf):
        super(MultiLevelAttention, self).__init__()
        self.model = model_conf
        self.embedded_mappings = nn.ModuleList(
            [EmbeddedMapping(model_conf[0], is_first=True)] +
            [EmbeddedMapping(n_layers, is_first=False) for n_layers in model_conf[1:]])
        self.attention_modules = nn.ModuleList([AttentionModule() for _ in model_conf])
        self.fc = nn.Linear(len(model_conf) * K, K)
        self.norm = nn.BatchNorm1d(K)

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
        out = torch.sigmoid(self.norm(self.fc(conc_ys)))
        return out


class Input(nn.Module):

    def __init__(self, input_conf, cnn_type, device):
        super(Input, self).__init__()
        self.conf = input_conf
        self.device = device
        self.cnn_type = cnn_type

    def forward(self, x):
        if self.cnn_type == "resnet":
            if self.conf == "repeat":
                x = torch.cat([x, x, x], dim=2)
            elif self.conf == "single":
                x = torch.cat([x, torch.zeros(x.shape, device=self.device), torch.zeros(x.shape, device=self.device)],
                                  dim=2)
            else:
                raise Exception("Invalid input type")

            x[:, :, 0, :, :] = (x[:, :, 0, :, :] - 0.485) / 0.229
            x[:, :, 1, :, :] = (x[:, :, 1, :, :] - 0.456) / 0.224
            x[:, :, 2, :, :] = (x[:, :, 2, :, :] - 0.406) / 0.225

        if self.cnn_type == "vggish":
            return x.reshape((-1, 1, s_vggish_shape[0], s_vggish_shape[1]))
        elif self.cnn_type == "resnet":
            return x.reshape((-1, 3, s_resnet_shape[0], s_resnet_shape[1]))
        else:
            raise Exception("CNN type is not valid.")


class Ensemble(nn.Module):

    def __init__(self, input_conf: str, cnn_conf: Dict[str, Union[str, int]], model_conf: List[int],
                 device, cnn_type: str = "vggish", ):
        super(Ensemble, self).__init__()
        self.input = Input(input_conf, cnn_type, device)
        if cnn_type == "vggish":
            self.cnn = VGGish()
        elif cnn_type == "resnet":
            self.cnn = ResNet50_ft(**cnn_conf)
        else:
            raise Exception("CNN type is not valid.")
        self.cnn_type = cnn_type
        self.mla = MultiLevelAttention(model_conf)

    def forward(self, x):
        x_proc = self.input(x)
        features = self.cnn(x_proc)
        if self.cnn_type == "vggish":
            out = self.mla(features.reshape(-1, T, 128))
        elif self.cnn_type == "resnet":
            out = self.mla(features.reshape(-1, T, M))
        else:
            raise Exception("CNN type is not valid.")
        return out


class VGGish(nn.Module):
    """
    PyTorch implementation of the VGGish model.
    Adapted from: https://github.com/harritaylor/torch-vggish
    The following modifications were made: (i) correction for the missing ReLU layers, (ii) correction for the
    improperly formatted data when transitioning from NHWC --> NCHW in the fully-connected layers, and (iii)
    correction for flattening in the fully-connected layers.
    """

    def __init__(self):
        super(VGGish, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 64, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(64, 128, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(128, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2),

            nn.Conv2d(256, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, stride=2)
        )
        self.fc = nn.Sequential(
            nn.Linear(512 * 24, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 128),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        x = self.features(x).permute(0, 2, 3, 1).contiguous()
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
