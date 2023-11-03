import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import re
import torch.utils.model_zoo as model_zoo

import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import softplus
from model.backbones.densenet import DenseNet, DenseNet_2fc_relu
from model.clinical_only import Clinical


class IB_basic_densenet_2fc_relu(DenseNet_2fc_relu):
    def __init__(self, growth_rate=32, block_config=(6, 12, 24, 16),
                 num_init_features=64, bn_size=4, drop_rate=0, num_classes=1000):

        # super(DenseNet_BI, self).__init__()
        super().__init__(growth_rate, block_config,
                         num_init_features, bn_size, drop_rate, num_classes)

    def forward(self, x):
        features = self.features(x)

        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        observation = torch.flatten(out, 1) # observation[1]

        # densenet121: 1fc
        observation_output = self.classifier1(observation) # observation[0]

        # # 2fc
        representation = self.bottleneck(observation)
        representation_output = self.classifier2(F.relu(representation))

        return observation_output, representation_output, observation, representation


def IB_basic_densenet121_2fc_relu(pretrained=False, **kwargs):
    r"""Densenet-121 model from
    `"Densely Connected Convolutional Networks" <https://arxiv.org/pdf/1608.06993.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = IB_basic_densenet_2fc_relu(num_init_features=64, growth_rate=32, block_config=(6, 12, 24, 16),
                     **kwargs)
    if pretrained:
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

        del state_dict['classifier.weight']
        del state_dict['classifier.bias']

        model.load_state_dict(state_dict, strict=False)
    return model


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

class ChannelCompress(nn.Module):
    def __init__(self, in_ch=2048, out_ch=256):
        """
        reduce the amount of channels to prevent final embeddings overwhelming shallow feature maps
        out_ch could be 512, 256, 128
        """
        super(ChannelCompress, self).__init__()
        num_bottleneck = 1000
        add_block = []
        add_block += [nn.Linear(in_ch, num_bottleneck)]
        # add_block += [nn.BatchNorm1d(num_bottleneck)]
        add_block += [nn.ReLU()]

        add_block += [nn.Linear(num_bottleneck, 500)]
        # add_block += [nn.BatchNorm1d(500)]
        add_block += [nn.ReLU()]
        add_block += [nn.Linear(500, out_ch)]

        # Extra BN layer, need to be removed
        # add_block += [nn.BatchNorm1d(out_ch)]

        add_block = nn.Sequential(*add_block)
        add_block.apply(weights_init_kaiming)
        self.model = add_block

    def forward(self, x):
        x = self.model(x)
        return x

class VIB(nn.Module):
    def __init__(self, in_ch=2048, z_dim=256, num_class=395):
        super(VIB, self).__init__()
        self.in_ch = in_ch
        self.out_ch = z_dim * 2
        self.num_class = num_class
        self.bottleneck = ChannelCompress(in_ch=self.in_ch, out_ch=self.out_ch)
        
        classifier = []
        classifier += [nn.Linear(self.out_ch, self.out_ch // 2)]
        # classifier += [nn.BatchNorm1d(self.out_ch // 2)]
        classifier += [nn.LeakyReLU(0.1)]
        # classifier += [nn.Dropout(0.5)]
        classifier += [nn.Linear(self.out_ch // 2, self.num_class)]
        classifier = nn.Sequential(*classifier)
        self.classifier = classifier
        self.classifier.apply(weights_init_classifier)

    def forward(self, v):
        z_given_v = self.bottleneck(v)
        p_y_given_z = self.classifier(z_given_v)
        # return p_y_given_z, z_given_v
        return p_y_given_z

class fuse_bottleneck_withobservation(nn.Module):
    def __init__(self, in_ch=2048, num_class=395, dim_bottleneck=256):
        super(fuse_bottleneck_withobservation, self).__init__()

        layers = []

        layers += [nn.Linear(in_ch, dim_bottleneck)]
        # layers += [nn.Dropout(0.66)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(dim_bottleneck, in_ch)]

        self.bottleneck = nn.Sequential(*layers)
        self.classifier = nn.Linear(in_ch, num_class)

        self.classifier.apply(weights_init_classifier)

        self.classifier_observation = nn.Linear(in_ch, num_class)
        self.classifier_observation.apply(weights_init_classifier)

    def forward(self, v):
        output_observation = self.classifier_observation(v)
        representation = self.bottleneck(v)
        output_representation = self.classifier(representation)
        return output_observation,  output_representation


class Concat_DenseNet_Image_Clinical_Generalised_withConfidence(nn.Module):
    def __init__(self, image_backbone1=None, image_backbone2=None, clinical_backbone=None, fuse_backbone1=None, fuse_backbone2=None, num_classes=1000, len_clinical=0, opt=None):
        super(Concat_DenseNet_Image_Clinical_Generalised_withConfidence, self).__init__()
        
        self.use_only_clinical = opt.use_only_clinical
        self.use_only_ct_axial = opt.use_only_ct_axial
        self.use_only_ct_coronal = opt.use_only_ct_coronal
        self.use_fuse_clinical_axial = opt.use_fuse_clinical_axial
        self.use_fuse_axial_coronal = opt.use_fuse_axial_coronal
        self.use_confidence = opt.use_CONF
        self.use_attention = opt.use_ATTEN
        self.image_coronal = image_backbone1
        self.image_axial = image_backbone2
        self.clinical_backbone = clinical_backbone
        self.fuse_image_clinical = fuse_backbone1
        self.fuse_image_image = fuse_backbone2
        if self.use_confidence:
            self.ConfidenceLayer_clinical = nn.Linear(1024, 1)
            self.ConfidenceLayer_axial = nn.Linear(1024, 1)
            self.ConfidenceLayer_coronal = nn.Linear(1024, 1)

        if self.use_attention:
            self.Attention_clinical = nn.Linear(1024, 1024) # input, input
            self.Attention_axial = nn.Linear(1024, 1024)
            self.Attention_coronal = nn.Linear(1024, 1024)

    def forward(self, x_axial, x_coronal, clinical):

        if self.use_only_clinical:
            output_observation_clinical, output_representation_clinical, observation_clinical, representation_clinical = self.clinical_backbone(clinical)
            return (output_observation_clinical, output_observation_clinical), (
            output_observation_clinical, output_observation_clinical)

        if self.use_only_ct_axial:
            output_observation_axial, output_representation_axial, observation_axial, representation_axial = self.image_axial(x_axial)
            return (output_observation_axial, output_observation_axial), (output_observation_axial, output_observation_axial)

        if self.use_only_ct_coronal:
            output_observation_coronal, output_representation_coronal, observation_coronal, representation_coronal = self.image_coronal(x_coronal)
            return (output_observation_coronal, output_observation_coronal), (output_observation_coronal, output_observation_coronal)

        if self.use_fuse_clinical_axial:
            output_observation_clinical, output_representation_clinical, observation_clinical, representation_clinical = self.clinical_backbone(clinical)
            output_observation_axial, output_representation_axial, observation_axial, representation_axial = self.image_axial(x_axial)

            if self.use_confidence:
                conf_observation_axial = self.ConfidenceLayer_axial(observation_axial)
                conf_observation_clinical = self.ConfidenceLayer_clinical(observation_clinical)
            else:
                conf_observation_axial = 1
                conf_observation_clinical = 1

            if self.use_attention:
                atten_observation_axial = torch.sigmoid(self.Attention_axial(observation_axial))
                atten_observation_clinical = torch.sigmoid(self.Attention_clinical(observation_clinical))
            else:
                atten_observation_axial = 1
                atten_observation_clinical = 1
            
            observation_axial = observation_axial * atten_observation_axial
            observation_clinical = observation_clinical * atten_observation_clinical

            # Masking
            # if 0<=rand<0.25, use axial only, if 0.25<=rand<0.5, use clinical only, else use both
            rand = np.random.uniform()
            # rand = 1
            if rand < 0.25:
                fuse_observation = torch.concat([observation_clinical * 0, observation_axial * conf_observation_axial], dim=1)
            elif rand < 0.5:
                fuse_observation = torch.concat([observation_clinical * conf_observation_clinical, observation_axial * 0], dim=1)
            else:
                fuse_observation = torch.concat([observation_clinical * conf_observation_clinical, observation_axial * conf_observation_axial], dim=1)
            output_observation_fuse, output_representation_fuse = self.fuse_image_clinical(fuse_observation)

            return (output_observation_fuse, output_representation_fuse), (output_observation_axial, output_observation_clinical), (conf_observation_axial, conf_observation_clinical), (atten_observation_axial, atten_observation_clinical)

        if self.use_fuse_axial_coronal:
            output_observation_axial, output_representation_axial, observation_axial, representation_axial = self.image_axial(x_axial)
            output_observation_coronal, output_representation_coronal, observation_coronal, representation_coronal = self.image_coronal(x_coronal)

            fuse_observation = torch.concat([observation_coronal, observation_axial], dim=1)
            output_observation_fuse, output_representation_fuse = self.fuse_image_image(fuse_observation)
            return (output_observation_fuse, output_representation_fuse), (output_observation_axial, output_observation_coronal)


def concat_densenet_image_clinical_densenet121(pretrained, opt, **kwargs):

    clinical_backbone = Clinical(num_classes=2, len_clinical=opt.len_clinical)
    image_backbone1 = IB_basic_densenet121_2fc_relu(pretrained=pretrained, num_classes=2, drop_rate=opt.drop_rate)
    image_backbone2 = IB_basic_densenet121_2fc_relu(pretrained=pretrained, num_classes=2, drop_rate=opt.drop_rate)

    fc_img = image_backbone1.classifier1.in_features
    fc_clinical = clinical_backbone.classifier_observation.in_features
    fuse_backbone1 = fuse_bottleneck_withobservation(in_ch=fc_img + fc_clinical, num_class=2, dim_bottleneck=opt.dim_bottleneck)
    fuse_backbone2 = fuse_bottleneck_withobservation(in_ch=fc_img*2, num_class=2, dim_bottleneck=opt.dim_bottleneck)

    model = Concat_DenseNet_Image_Clinical_Generalised_withConfidence(clinical_backbone=clinical_backbone,image_backbone1=image_backbone1, image_backbone2=image_backbone2, fuse_backbone1=fuse_backbone1, fuse_backbone2=fuse_backbone2, opt=opt)


    return model


