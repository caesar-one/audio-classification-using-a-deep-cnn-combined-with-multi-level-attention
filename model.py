from typing import List, Dict, Union

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.models import resnet50
from torchvggish.vggish import VGGish

from params import *


class Ensemble(nn.Module):

    def __init__(self, input_conf: str, cnn_conf: Dict[str, Union[str, int]], model_conf: List[int],
                 device):
        """
        This class models the whole network, which is made up of a CNN module followed by a multi-level attention (MLA)
            module. The CNN performs the feature extraction, then the MLA makes the classification.
            The MLA is composed of two parts: an embedding module and an attention one. The embedding module is
            responsible for extracting the embeddings from the features, while the attention module performs the
            classification.

        :param input_conf: the configuration of the channels of the CNN input.
        :param cnn_conf: the configuration parameter of the CNN. It's a dict composed by the following key-value pairs:
            'cnn_type' (str):            type of CNN, can be 'resnet' or 'vggish'
            'num_classes' (int):         number of output classes (not used)
            'use_pretrained' (bool):     if True, use a pretrained model
            'just_bottleneck' (bool):    if True, the CNN extracts just the bottleneck feature, else ???
            'cnn_trainable' (bool):      if True, the CNN is trainable, otherwise is not (overrides the following param)
            'first_cnn_layer_trainable': if True, the first layer of the CNN is trainable, otherwise is not.
            'in_channels' (int):         number of channels of the input
        :param model_conf: the configuration of the MLA
            e.g. [2] means a MLA (actually a single level attention!) of type 2A
                 [2, 1] means a MLA of type 2A-1A, etc.
        :param device: self-explanatory
        """

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



class Input(nn.Module):

    def __init__(self, input_conf, cnn_type, device):
        """
        This class is just for pre-processing the CNN input and actually doesn't represent any module of the network.
            - Resnet50 inputs must be images of size 224x224 with 3 channels.
            - VGGish inputs must be images of size 96x64 with 1 channel.
            For an explanation of the parameters, see the class Ensemble.
        """
        super(Input, self).__init__()
        self.conf = input_conf
        self.device = device
        self.cnn_type = cnn_type

    def forward(self, x):
        if self.cnn_type == "resnet":
            if self.conf == "repeat":
                # Put the frame in all the channels
                x = torch.cat([x, x, x], dim=2)
            elif self.conf == "single":
                # Put the frame only on the first channel, the other channels contains only zeros.
                x = torch.cat([x, torch.zeros(x.shape, device=self.device), torch.zeros(x.shape, device=self.device)],
                              dim=2)
            else:
                raise Exception("Invalid input type")

            # Normalize the input for the Resnet
            x[:, :, 0, :, :] = (x[:, :, 0, :, :] - 0.485) / 0.229
            x[:, :, 1, :, :] = (x[:, :, 1, :, :] - 0.456) / 0.224
            x[:, :, 2, :, :] = (x[:, :, 2, :, :] - 0.406) / 0.225

        # Reshape the input according to the network specifics.
        if self.cnn_type == "vggish":
            return x.reshape((-1, 1, S_VGGISH_SHAPE[0], S_VGGISH_SHAPE[1]))
        elif self.cnn_type == "resnet":
            return x.reshape((-1, 3, S_RESNET_SHAPE[0], S_RESNET_SHAPE[1]))
        else:
            raise Exception("CNN type is not valid.")


class CNN(nn.Module):
    def __init__(self,
                 cnn_type="vggish",
                 num_classes=10,
                 use_pretrained=True,
                 just_bottlenecks=False,
                 cnn_trainable=False,
                 first_cnn_layer_trainable=False,
                 in_channels=3):
        """
        Creates an instance of the CNN part used as features extractor. It can either load a pretrained version of
        ResNet50 or VGGish

        :param cnn_type: str, can be either "resnet" or "vggish"
        :param num_classes: the number of classes, i.e. the dimension of the output of the last layer
        :param use_pretrained: if True, downloads a pre-trained version of the corresponding architecture
        :param just_bottlenecks: if True the last fully connected part is removed, so that the model returns bottlenecks
        :param cnn_trainable: if True, the whole CNN part is trainable. Otherwise the gradient will NOT be calculated
        :param first_cnn_layer_trainable: if True, sets the first CNN layer trainable, to optimize for the dataset
        :param in_channels: the number of desired input channels for the network (to reshape the original one).
        """
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

            self.cnn_model = VGGish(urls=model_urls, pretrained=use_pretrained, preprocess=False, postprocess=False,
                                    progress=True)

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
        # return torch.flatten(x, 1)
        return x


class CnnFlatten(nn.Module):
    def __init__(self, cnn_type):
        """
        This class is only used to reshape the output of the CNN part, depending on the architecture used.
        """
        super(CnnFlatten, self).__init__()
        self.cnn_type = cnn_type

    def forward(self, x):
        if self.cnn_type == "resnet":
            x = torch.flatten(x, 1)
        elif self.cnn_type == "vggish":
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
            x = self.dropouts[i](F.relu(self.norms[i](self.fc[i](x))))
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


def set_requires_grad(model, value):
    """
    Sets the *requires_grad* attribute to *value* for all the parameters in the *model*.

    :param model: A PyTorch model
    :param value: Truth value to set the model's parameters to.
    """
    for param in model.parameters():
        param.requires_grad = value
