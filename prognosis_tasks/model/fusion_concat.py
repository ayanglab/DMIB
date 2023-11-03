from .backbones.densenet import DenseNet, model_urls, DenseNet_2fc_relu
from model.clinical_only import Clinical
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.utils.model_zoo as model_zoo
from torch.nn import init
from torch.nn.functional import softplus
import re
from collections import OrderedDict


class Concat_DenseNet(DenseNet):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, len_clinical=0):
        super().__init__(growth_rate, block_config,
                         num_init_features, bn_size, drop_rate, num_classes)
        fc_feature_in = self.classifier.in_features
        self.classifier = nn.Linear(fc_feature_in + len_clinical, num_classes)

    def forward(self, x, clinical):

        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        feature_initial = self.features.pool0(x)

        feature_denseblock1 = self.features.denseblock1(feature_initial)
        feature_transition1 = self.features.transition1(feature_denseblock1)

        feature_denseblock2 = self.features.denseblock2(feature_transition1)
        feature_transition2 = self.features.transition2(feature_denseblock2)

        feature_denseblock3 = self.features.denseblock3(feature_transition2)
        feature_transition3 = self.features.transition3(feature_denseblock3)

        feature_denseblock4 = self.features.denseblock4(feature_transition3)
        features = self.features.norm5(feature_denseblock4)

        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        # add
        out = torch.concat([out, clinical], 1)
        out = self.classifier(out)
        return out


class Concat_DenseNet_addfc(DenseNet):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, len_clinical=0):

        super().__init__(growth_rate, block_config,
                         num_init_features, bn_size, drop_rate, num_classes)
        # fc_feature_in = self.classifier.in_features
        self.classifier = nn.Linear(1024 + len_clinical, 256)
        self.classifier2 = nn.Linear(256, num_classes)

    def forward(self, x, clinical):

        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        feature_initial = self.features.pool0(x)

        feature_denseblock1 = self.features.denseblock1(feature_initial)
        feature_transition1 = self.features.transition1(feature_denseblock1)

        feature_denseblock2 = self.features.denseblock2(feature_transition1)
        feature_transition2 = self.features.transition2(feature_denseblock2)

        feature_denseblock3 = self.features.denseblock3(feature_transition2)
        feature_transition3 = self.features.transition3(feature_denseblock3)

        feature_denseblock4 = self.features.denseblock4(feature_transition3)
        features = self.features.norm5(feature_denseblock4)

        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        # add
        out = torch.concat([out, clinical], 1)
        out = self.classifier(out)
        out = self.classifier2(out)

        return out


class Concat_DenseNet_addfc_addrelu(DenseNet_2fc_relu):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, len_clinical=0):

        super().__init__(growth_rate, block_config,
                 num_init_features, bn_size, drop_rate, num_classes)

        fc_feature_in = self.bottleneck.in_features
        fc_feature_out = self.bottleneck.out_features
        self.bottleneck = nn.Linear(fc_feature_in + len_clinical, fc_feature_out)
        # self.classifier2 = nn.Linear(256, num_classes)

        # for m in self.modules():
        #     # if isinstance(m, nn.Conv2d):
        #     #     nn.init.kaiming_normal(m.weight.data)
        #     # elif isinstance(m, nn.BatchNorm2d):
        #     #     m.weight.data.fill_(1)
        #     #     m.bias.data.zero_()
        #     if isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

    def forward(self, x, clinical):

        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        feature_initial = self.features.pool0(x)

        feature_denseblock1 = self.features.denseblock1(feature_initial)
        feature_transition1 = self.features.transition1(feature_denseblock1)

        feature_denseblock2 = self.features.denseblock2(feature_transition1)
        feature_transition2 = self.features.transition2(feature_denseblock2)

        feature_denseblock3 = self.features.denseblock3(feature_transition2)
        feature_transition3 = self.features.transition3(feature_denseblock3)

        feature_denseblock4 = self.features.denseblock4(feature_transition3)
        features = self.features.norm5(feature_denseblock4)

        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        # add
        out = torch.concat([out, clinical], 1)
        out = F.relu(self.bottleneck(out))
        out = self.classifier2(out)

        return out


class Concat_DenseNet_addfc_addrelu_addconv(DenseNet):

    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000, len_clinical=0):

        super().__init__(growth_rate, block_config,
                 num_init_features, bn_size, drop_rate, num_classes)

        fc_feature_in = self.classifier.in_features
        fc_feature_out = self.classifier.out_features
        # self.classifier = nn.Linear(1024 + len_clinical, 256)
        # self.classifier2 = nn.Linear(256, num_classes)

        # for m in self.modules():
        #     # if isinstance(m, nn.Conv2d):
        #     #     nn.init.kaiming_normal(m.weight.data)
        #     # elif isinstance(m, nn.BatchNorm2d):
        #     #     m.weight.data.fill_(1)
        #     #     m.bias.data.zero_()
        #     if isinstance(m, nn.Linear):
        #         m.bias.data.zero_()

        self.dense_1 = nn.Linear(len_clinical, 128)
        self.dense_2 = nn.Linear(128, 256)
        self.dense_3 = nn.Linear(256, 512)
        self.dense_4 = nn.Linear(512, fc_feature_in)
        self.classifier = nn.Linear(fc_feature_in*2, fc_feature_out)

    def forward(self, x, clinical):


        f1 = self.dense_1(clinical)
        f1 = F.relu(f1)
        f2 = self.dense_2(f1)
        f2 = F.relu(f2)
        f3 = self.dense_3(f2)
        f3 = F.relu(f3)
        f4 = self.dense_4(f3)
        f4 = F.relu(f4)


        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        feature_initial = self.features.pool0(x)

        feature_denseblock1 = self.features.denseblock1(feature_initial)
        feature_transition1 = self.features.transition1(feature_denseblock1)

        feature_denseblock2 = self.features.denseblock2(feature_transition1)
        feature_transition2 = self.features.transition2(feature_denseblock2)

        feature_denseblock3 = self.features.denseblock3(feature_transition2)
        feature_transition3 = self.features.transition3(feature_denseblock3)

        feature_denseblock4 = self.features.denseblock4(feature_transition3)
        features = self.features.norm5(feature_denseblock4)

        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)

        # add
        out = torch.concat([out, f4], 1)
        out = F.relu(self.classifier(out))
        out = self.classifier2(out)

        return out


def concat_densenet121(num_fold=None, opt=None, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Concat_DenseNet(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
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

    if opt.use_selfpretrain:
        path_pretrain = '/media/NAS03/yyfang/yanglab4_backup/reproduce_weakly/cleanup/train_covid_mortality/result_exp/'

        state_dict = \
            torch.load(path_pretrain + opt.pretrain_ct_axial + '/model/' + str(num_fold) + '/best_auc.pt')[
                'model_state_dict']

        new_state_dict = OrderedDict()

        for key, value in state_dict.items():
            if 'image_axial' in key:
                new_key = key[12:]
                new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict, strict=False)


        model.load_state_dict(state_dict, strict=False)
    return model

def concat_densenet121_addfc(num_fold=None, opt=None,  **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Concat_DenseNet_addfc(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
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

        # del state_dict['classifier.weight']
        # del state_dict['classifier.bias']
        model.load_state_dict(state_dict, strict=False)

    if opt.use_selfpretrain:
        path_pretrain = '/media/NAS03/yyfang/yanglab4_backup/reproduce_weakly/cleanup/train_covid_mortality/result_exp/'

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

def concat_densenet121_addfc_addrelu(num_fold=None, opt=None, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Concat_DenseNet_addfc_addrelu(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
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

        # del state_dict['classifier.weight']
        # del state_dict['classifier.bias']
        model.load_state_dict(state_dict, strict=False)

    if opt.use_selfpretrain:
        path_pretrain = '../train_covid_mortality/result_exp/'

        state_dict = \
            torch.load(path_pretrain + opt.pretrain_ct_axial + '/model/' + str(num_fold) + '/best_auc.pt')[
                'model_state_dict']

        new_state_dict = OrderedDict()

        for key, value in state_dict.items():
            if 'image_axial' in key and not ("bottleneck" in key) and not ("classifer" in key):
                new_key = key[12:]
                new_state_dict[new_key] = value

        model.load_state_dict(new_state_dict, strict=False)

    return model

# concat + conv
def concat_densenet121_addfc_addrelu_addconv(num_fold=None, opt=None, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Concat_DenseNet_addfc_addrelu_addconv(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
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

        # del state_dict['classifier.weight']
        # del state_dict['classifier.bias']
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
