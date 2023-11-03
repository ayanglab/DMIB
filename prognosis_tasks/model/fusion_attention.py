from .backbones.densenet import DenseNet_2fc_relu, model_urls
import torch.nn as nn
import torch.nn.functional as F
import torch
import re
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict


def forward_attention_densenet(self, x, clinical):
    '''
    shared by all densenet backbones
    '''
    # clinical side:
    co1 = self.dense_1(clinical)
    co1 = F.relu(co1)
    if self.clinical_dropout:
        co1 = F.droput(co1)

    co2 = self.dense_2(co1)
    co2 = F.relu(co2)
    if self.clinical_dropout:
        co2 = F.droput(co2)

    co3 = self.dense_3(co2)
    co3 = F.relu(co3)
    if self.clinical_dropout:
        co3 = F.droput(co3)
    #
    # co4 = self.dense_4(co3)
    # co4 = F.relu(co4)
    # if self.clinical_dropout:
    #     co4 = F.droput(co4)

    # img side: no
    x = self.features.conv0(x)
    x = self.features.norm0(x)
    x = self.features.relu0(x)
    feature_initial = self.features.pool0(x)

    feature_denseblock1 = self.features.denseblock1(feature_initial)
    feature_transition1 = self.features.transition1(feature_denseblock1)
    # add
    feature_transition1 = feature_transition1 * co1.unsqueeze(2).unsqueeze(2)

    feature_denseblock2 = self.features.denseblock2(feature_transition1)
    feature_transition2 = self.features.transition2(feature_denseblock2)
    # add
    feature_transition2 = feature_transition2 * co2.unsqueeze(2).unsqueeze(2)

    feature_denseblock3 = self.features.denseblock3(feature_transition2)
    feature_transition3 = self.features.transition3(feature_denseblock3)
    #  add
    feature_transition3 = feature_transition3 * co3.unsqueeze(2).unsqueeze(2)


    feature_denseblock4 = self.features.denseblock4(feature_transition3)
    features = self.features.norm5(feature_denseblock4)

    out = F.relu(features, inplace=True)
    out = F.adaptive_avg_pool2d(out, (1, 1))
    out = torch.flatten(out, 1)
    out = self.classifier(out)

    # tensor.device
    return out


def attention_densenet121_m1(pretrained=False, **kwargs):
    r"""Densenet-161 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = densenet121(pretrained=False, **kwargs)

    model.clinical_dropout = False
    model.dense_1 = nn.Linear(kwargs['len_clinical'], 128)
    model.dense_2 = nn.Linear(128, 256)
    model.dense_3 = nn.Linear(256, 512)
    model.dense_4 = nn.Linear(512, 1024)

    from functools import partial
    model.forward = partial(forward_attention_densenet, model)

    return model


class Attention_DenseNet(DenseNet_2fc_relu):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, len_clinical=0):
        super().__init__(growth_rate, block_config,
                         num_init_features, bn_size, drop_rate, num_classes)


        # fc_feature_in = self.classifier.in_features
        # self.classifier = nn.Linear(fc_feature_in + len_clinical, num_classes)
        self.dense_1 = nn.Linear(len_clinical, 128)
        self.dense_2 = nn.Linear(128, 256)
        self.dense_3 = nn.Linear(256, 512)
        # model.dense_4 = nn.Linear(512, 1024)

    def forward(self, x, clinical):
        '''
        shared by all densenet backbones
        '''
        # clinical side:
        co1 = self.dense_1(clinical)
        co1 = F.relu(co1)
        # if self.clinical_dropout:
        #     co1 = F.droput(co1)

        co2 = self.dense_2(co1)
        co2 = F.relu(co2)
        # if self.clinical_dropout:
        #     co2 = F.droput(co2)

        co3 = self.dense_3(co2)
        co3 = F.relu(co3)
        # if self.clinical_dropout:
        #     co3 = F.droput(co3)

        # img side: no
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        feature_initial = self.features.pool0(x)

        feature_denseblock1 = self.features.denseblock1(feature_initial)
        feature_transition1 = self.features.transition1(feature_denseblock1)
        # add
        feature_transition1 = feature_transition1 * co1.unsqueeze(2).unsqueeze(2)

        feature_denseblock2 = self.features.denseblock2(feature_transition1)
        feature_transition2 = self.features.transition2(feature_denseblock2)
        # add
        feature_transition2 = feature_transition2 * co2.unsqueeze(2).unsqueeze(2)

        feature_denseblock3 = self.features.denseblock3(feature_transition2)
        feature_transition3 = self.features.transition3(feature_denseblock3)
        #  add
        feature_transition3 = feature_transition3 * co3.unsqueeze(2).unsqueeze(2)

        feature_denseblock4 = self.features.denseblock4(feature_transition3)
        features = self.features.norm5(feature_denseblock4)

        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)


        out = self.classifier2(F.relu(self.bottleneck(out)))
        return out


def attention_densenet121_addfc_addrelu(num_fold=None, opt=None, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Attention_DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)

    if opt.pretrained:
        print("pretrained")
        # '.'s are no longer allowed in module names, but pervious _DenseLayer
        # has keys 'norm.1', 'relu.1', 'conv.1', 'norm.2', 'relu.2', 'conv.2'.
        # They are also in the checkpoints in model_urls. This pattern is used
        # to find such keys.
        pattern = re.compile(
            r'^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$')
        state_dict = model_zoo.load_url(model_urls['densenet121'])

        for key in list(state_dict.keys()):
            res = pattern.match(key)
            if res:
                new_key = res.group(1) + res.group(2)
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        for key in list(state_dict.keys()):
            if "classifier" in key:
                print("delete: ", key)
                del state_dict[key]
        model.load_state_dict(state_dict, strict=False)

    if opt.use_selfpretrain:
        path_pretrain = '../train_covid_mortality/result_exp/'

        state_dict = \
            torch.load(path_pretrain + opt.pretrain_ct_axial + '/model/' + str(num_fold) + '/best_auc.pt')[
                'model_state_dict']

        new_state_dict = OrderedDict()

        for key, value in state_dict.items():
            if 'image_axial' in key:
                new_key = key[12:]
                new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict, strict=False)

    return model