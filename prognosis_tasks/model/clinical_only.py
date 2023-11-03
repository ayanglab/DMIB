import torch.nn as nn
import torch.nn.functional as F
import torch
import re
import torch.utils.model_zoo as model_zoo

import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import softplus
from collections import OrderedDict


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    # print(classname)
    if classname.find('Conv') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        init.kaiming_normal_(m.weight.data, a=0, mode='fan_out')
        init.constant_(m.bias.data, 0.0)
    elif classname.find('BatchNorm1d') != -1:
        init.normal_(m.weight.data, 1.0, 0.02)
        init.constant_(m.bias.data, 0.0)

def weights_init_classifier(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        init.normal_(m.weight.data, std=0.001)
        init.constant_(m.bias.data, 0.0)

class Clinical(nn.Module):

    def __init__(self, num_classes=1000, len_clinical=2):
        super(Clinical, self).__init__()
        add_block = []
        add_block += [nn.Linear(len_clinical, 128)]
        add_block += [nn.Linear(128, 256)]
        add_block += [nn.Linear(256, 512)]
        add_block += [nn.Linear(512, 1024)]
        # add_block += [nn.BatchNorm1d(1024)]
        # add_block += [nn.Dropout(0.2)]
        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.upsample = add_block

        # add_block += [nn.BatchNorm1d(num_bottleneck)]
        # layers = [
        #         ("aux_base", nn.Linear(1024, 512)),
        #         ("aux_relu", nn.ReLU()),
        #         # # ("aux_dropout", nn.Dropout(p=0.2, inplace=True)),
        #         ("aux_1", nn.Linear(512, 256)),
        #         ("aux_relu", nn.ReLU()),
        #         ("aux_2", nn.Linear(256, num_classes)),
        #     ]
        # self.bottleneck = VIB(in_ch=2048, z_dim=self.args.z_dim, num_class= num_classes)
        self.classifier_observation = nn.Linear(1024, 2)
        self.classifier_observation.apply(weights_init_classifier)

        layers = [
            ("aux_base", nn.Linear(1024, 512)),
            ("aux_relu", nn.ReLU()),
            # # ("aux_dropout", nn.Dropout(p=0.2, inplace=True)),
            ("aux_1", nn.Linear(512, 256)),
            ("aux_relu", nn.ReLU()),
        ]
        self.bottleneck = nn.Sequential(OrderedDict(layers))
        self.classifier_representation = nn.Linear(256, num_classes)


    def forward(self, clinical):

        observation = self.upsample(clinical)
        output_observation = self.classifier_observation(observation)
        representation = self.bottleneck(observation)
        output_representation = self.classifier_representation(representation)
        # return out
        # return clinical_observation, clinical_pred
        return output_observation, output_representation, observation, representation

