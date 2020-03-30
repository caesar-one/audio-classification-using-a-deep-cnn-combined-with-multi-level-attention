from typing import List, Dict, Union

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvggish.vggish import VGGish

from params import *


# TODO Different channels? can put derivative? see video


def set_requires_grad(model, value):
    for param in model.parameters():
        param.requires_grad = value


class CNN(nn.Module):
    def __init__(self,
                 cnn_type="vggish", # the pretrained model to use. It can either "resnet" or "vggish" (default)
                 num_classes=10,  # number of output classes (makes sense if combined with just_bottlenecks=False)
                 use_pretrained=True,  # Uses a pretrained version of resnet50
                 just_bottlenecks=False,  # if =True the FC part is removed, so that the model returns bottlenecks.
                 cnn_trainable=False,  # if =True CNN part is trainable. Otherwise the gradient will NOT be calculated
                 first_cnn_layer_trainable=False,  # Sets the first CNN layer trainable, to optimize for the dataset
                 in_channels=3):  # the the number of input channels is reshaped
        super(CNN, self).__init__()
        if cnn_type == "resnet":
            self.cnn_model = resnet50(pretrained=use_pretrained)

            if not cnn_trainable:
                set_requires_grad(self.cnn_model, False)

            if first_cnn_layer_trainable:
                if in_channels == 3:
                    set_requires_grad(self.cnn_model.conv1, True)
                else:
                    self.cnn_model.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2,
                                                     padding=3,
                                                     bias=False)
            if just_bottlenecks:
                modules = list(self.cnn_model.children())[:-1]  # delete last fc layer.

                modules.append(CnnFlatten(cnn_type))
                self.cnn_model = nn.Sequential(*modules)

            else:
                num_ftrs = self.cnn_model.fc.in_features
                self.cnn_model.fc = nn.Linear(num_ftrs, num_classes)

        elif cnn_type == "vggish":

            model_urls = {
                "vggish": "https://github.com/harritaylor/torchvggish/releases/download/v0.1/vggish-10086976.pth"}

            self.cnn_model = VGGish(urls=model_urls, pretrained=use_pretrained, preprocess=False, postprocess=False, progress=True)

            if not cnn_trainable:
                set_requires_grad(self.cnn_model, False)

            if just_bottlenecks:
                modules = []
                modules.append(list(self.cnn_model.children())[0])  # just use the first group of layers.
                print()
                modules.append(CnnFlatten(cnn_type))
                self.cnn_model = nn.Sequential(*modules)

        else:
            raise Exception("Invalid CNN model name specified.")

    def forward(self, x):
        x = self.cnn_model(x)
        #return torch.flatten(x, 1)
        return x


class CnnFlatten(nn.Module):
    def __init__(self, cnn_type):
        super(CnnFlatten,self).__init__()
        self.cnn_type = cnn_type

    def forward(self, x):
        if self.cnn_type == "resnet":
            x = torch.flatten(x, 1)
        elif self.cnn_type == "vggish":
            #x = self.features(x)
            # Transpose the output from features to
            # remain compatible with vggish embeddings
            x = torch.transpose(x, 1, 3)
            x = torch.transpose(x, 1, 2)
            x = x.contiguous()
            x = x.view(x.size(0), -1)
        else:
            raise Exception("Invalid CNN model name specified.")
        return x


# Implements the block g (green big block in the main paper)
class EmbeddedMapping(nn.Module):

    def __init__(self, n_fc, is_first, emb_input_size):
        super(EmbeddedMapping, self).__init__()
        self.n_fc = n_fc
        self.norm0 = nn.BatchNorm1d(T)

        if is_first:
            self.fc = nn.ModuleList([nn.Linear(emb_input_size, H)] + [nn.Linear(H, H) for _ in range(n_fc - 1)])
        else:
            self.fc = nn.ModuleList([nn.Linear(H, H) for _ in range(n_fc)])

        self.dropouts = nn.ModuleList([nn.Dropout(p=DR) for _ in range(n_fc)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(T) for _ in range(n_fc)])


    # Input x has shape (batch_size, T, M) if is_first=True
    # otherwise x has shape (batch_size, T, H)
    def forward(self, x):
        x = self.norm0(x)
        for i in range(self.n_fc):
            x = self.dropouts[i](F.relu(self.norms[i](self.fc[i](x))), p=DR)
        # Output emb has shape (batch_size, T, H)
        return x


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

    def __init__(self, model_conf, emb_input_size):
        super(MultiLevelAttention, self).__init__()
        self.model = model_conf
        self.embedded_mappings = nn.ModuleList(
            [EmbeddedMapping(model_conf[0], is_first=True, emb_input_size=emb_input_size)] +
            [EmbeddedMapping(n_layers, is_first=False, emb_input_size=emb_input_size) for n_layers in model_conf[1:]])
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
            return x.reshape((-1, 1, S_VGGISH_SHAPE[0], S_VGGISH_SHAPE[1]))
        elif self.cnn_type == "resnet":
            return x.reshape((-1, 3, S_RESNET_SHAPE[0], S_RESNET_SHAPE[1]))
        else:
            raise Exception("CNN type is not valid.")


class Ensemble(nn.Module):

    def __init__(self, input_conf: str, cnn_conf: Dict[str, Union[str, int]], model_conf: List[int],
                 device):
        super(Ensemble, self).__init__()
        self.cnn_type = cnn_conf["cnn_type"]
        self.just_bottlenecks = cnn_conf["just_bottlenecks"]
        self.num_classes = cnn_conf["num_classes"]

        if self.cnn_type == "vggish" and self.just_bottlenecks:
            self.emb_input_size = M_VGGISH_JB
        elif self.cnn_type == "vggish" and not self.just_bottlenecks:
            self.emb_input_size = M_VGGISH
        elif self.cnn_type == "resnet" and self.just_bottlenecks:
            self.emb_input_size = M_RESNET
        elif self.cnn_type == "resnet" and not self.just_bottlenecks:
            self.emb_input_size = self.num_classes
        else:
            raise Exception("CNN type is not valid.")

        self.input = Input(input_conf=input_conf, cnn_type=self.cnn_type, device=device)
        self.mla = MultiLevelAttention(model_conf, self.emb_input_size)
        self.cnn = CNN(**cnn_conf)

    def forward(self, x):
        x_proc = self.input(x)
        features = self.cnn(x_proc)
        out = self.mla(features.reshape(-1, T, self.emb_input_size))
        return out
