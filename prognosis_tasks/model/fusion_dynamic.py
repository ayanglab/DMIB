import torch.nn as nn
import torch.nn.functional as F
import torch
import re
import torch.utils.model_zoo as model_zoo

import torch
from torch import nn
from torch.nn import init
from torch.nn.functional import softplus
from model.backbones.densenet import DenseNet
from model.clinical_only import Clinical
from model.BI import BI_basic_densenet121_2fc_relu

def xavier_init(m):
    if type(m) == nn.Linear:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0.0)

class LinearLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.clf = nn.Sequential(nn.Linear(in_dim, out_dim))
        self.clf.apply(xavier_init)

    def forward(self, x):
        x = self.clf(x)
        return x


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
        # classifier of VIB, maybe modified later.
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

# class fuse_bottleneck(nn.Module):
#     def __init__(self, in_ch=2048, num_class=395):
#         super(fuse_bottleneck, self).__init__()
#
#         layers = []
#         layers += [nn.Linear(in_ch, in_ch//2)]
#         layers += [nn.ReLU()]
#         layers += [nn.Linear(in_ch//2, 256)]
#         layers += [nn.ReLU()]
#         # layers += [nn.Dropout(0.5)]
#         # layers += [nn.Linear(self.out_ch // 2, self.num_class)]
#         self.bottleneck = nn.Sequential(*layers)
#         self.classifier = nn.Linear(256, num_class)
#         self.classifier.apply(weights_init_classifier)
#
#     def forward(self, v):
#         representation = self.bottleneck(v)
#         output_representation = self.classifier(representation)
#         # return output_representation, representation
#         return output_representation

class fuse_bottleneck_withobservation(nn.Module):

    def __init__(self, in_ch=2048, num_class=395, dim_bottleneck=256):
        super(fuse_bottleneck_withobservation, self).__init__()

        layers = []

        # 2 layer
        # layers += [nn.Linear(in_ch, in_ch//2)]
        # layers += [nn.ReLU()]
        # layers += [nn.Linear(in_ch//2, dim_bottleneck)]
        # layers += [nn.ReLU()]

        # 1 layer
        # layers += [nn.Linear(in_ch,dim_bottleneck)]
        # layers += [nn.ReLU()]

        # current model for IB
        layers += [nn.Linear(in_ch, dim_bottleneck)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(dim_bottleneck, in_ch)]


        # layers += [nn.Dropout(0.5)]
        # layers += [nn.Linear(self.out_ch // 2, self.num_class)]
        self.bottleneck = nn.Sequential(*layers)

        # self.classifier = nn.Linear(dim_bottleneck, num_class)
        self.classifier = nn.Linear(in_ch, num_class)

        self.classifier.apply(weights_init_classifier)

        self.classifier_observation = nn.Linear(in_ch, num_class)
        self.classifier_observation.apply(weights_init_classifier)

    def forward(self, v):

        output_observation = self.classifier_observation(v)
        representation = self.bottleneck(v)
        output_representation = self.classifier(representation)
        # return output_representation, representation
        return output_observation,  output_representation
        # return output_observation

class Concat_DenseNet_Image_Clinical_Generalised(nn.Module):

    def __init__(self, image_backbone1=None, image_backbone2=None, clinical_backbone=None, fuse_backbone1=None, fuse_backbone2=None, num_classes=1000, len_clinical=0, opt=None):

        super(Concat_DenseNet_Image_Clinical_Generalised, self).__init__()

        self.use_only_clinical = opt.use_only_clinical
        self.use_only_ct_axial = opt.use_only_ct_axial
        self.use_only_ct_coronal = opt.use_only_ct_coronal
        self.use_fuse_clinical_axial = opt.use_fuse_clinical_axial
        self.use_fuse_axial_coronal = opt.use_fuse_axial_coronal

        # if opt.use_only_ct_coronal:
        self.image_coronal = image_backbone1
        # if opt.use_only_ct_axial:
        self.image_axial = image_backbone2
        # if opt.use_only_clinical:
        self.clinical_backbone = clinical_backbone
        # fc_clinical = self.clinical_backbone.classifier.in_features

        # add: try different ways
        # self.fuse_Bottleneck = VIB(in_ch=fc_img + 1024, z_dim=256, num_class=num_classes)
        # fc_img = self.image_axial.classifier1.in_features
        # if opt.use_fuse_clinical_axial:
        self.fuse_image_clinical = fuse_backbone1
            # VIB(in_ch=fc_img + 1024, z_dim=256, num_class=num_classes)
        # if opt.use_fuse_axial_coronal:
        self.fuse_image_image = fuse_backbone2
            # self.fuse_image_clinical = fuse_backbone
            # VIB(in_ch=fc_img + 1024, z_dim=256, num_class=num_classes)

        self.use_CONF = opt.use_CONF
        self.fuseadd = opt.fuseadd
        self.fuseend = opt.fuseend
        fc_axial = self.image_axial.classifier1.in_features
        fc_cro = fc_axial
        fc_cli = self.clinical_backbone.classifier_observation.in_features
        self.TCPConfidenceLayer_axial = LinearLayer(fc_axial, 1)
        self.TCPConfidenceLayer_cro = LinearLayer(fc_cro, 1)
        self.TCPConfidenceLayer_cli = LinearLayer(fc_cli, 1)

        dim_cli = self.clinical_backbone.upsample[0].in_features
        self.attention_layer = LinearLayer(dim_cli, dim_cli)

    def forward(self, x_axial, x_coronal, clinical):

        if self.use_only_clinical:
            output_observation_clinical, output_representation_clinical, observation_clinical, representation_clinical = self.clinical_backbone(clinical)
            # return (output_observation_clinical, output_representation_clinical), (0,0)
            return (output_observation_clinical, output_observation_clinical), (output_observation_clinical, output_observation_clinical)

        if self.use_only_ct_axial:
            output_observation_axial, output_representation_axial, observation_axial, representation_axial = self.image_axial(x_axial)
            # return (output_observation_axial,output_representation_axial), (0, 0)
            return (output_observation_axial, output_observation_axial), (output_observation_axial, output_observation_axial)

        if self.use_only_ct_coronal:
            output_observation_coronal, output_representation_coronal, observation_coronal, representation_coronal = self.image_coronal(x_coronal)
            # return (output_observation_coronal, output_representation_coronal), (0, 0)
            return (output_observation_coronal, output_observation_coronal),(output_observation_coronal, output_observation_coronal)

        if self.use_fuse_clinical_axial:
            layer_attention = torch.sigmoid(self.attention_layer(clinical))
            clinical = layer_attention * clinical

            output_observation_clinical, output_representation_clinical, observation_clinical, representation_clinical = self.clinical_backbone(clinical)
            output_observation_axial, output_representation_axial, observation_axial, representation_axial = self.image_axial(x_axial)


            if self.fuseend:
                output_observation_fuse = output_observation_axial + output_observation_clinical
                output_representation_fuse = output_representation_axial + output_representation_clinical
                conf_axial = 0
                conf_cli = 0
            else:
                conf_axial = self.TCPConfidenceLayer_axial(observation_axial)
                conf_cli = self.TCPConfidenceLayer_cli(observation_clinical)
                if self.fuseadd:
                    if self.use_CONF:
                        fuse_observation = observation_clinical * conf_cli + observation_axial * conf_axial
                    else:
                        fuse_observation = observation_clinical + observation_axial
                else:
                    if self.use_CONF:
                        fuse_observation = torch.concat([observation_clinical * conf_cli, observation_axial * conf_axial],dim=1)
                    else:
                        fuse_observation = torch.concat([observation_clinical, observation_axial], dim=1)

                output_observation_fuse, output_representation_fuse = self.fuse_image_clinical(fuse_observation)

            return (output_observation_fuse, output_representation_fuse), (output_observation_axial, output_observation_clinical), (conf_axial, conf_cli), layer_attention

        if self.use_fuse_axial_coronal:
            # to refer to farewell
            output_observation_axial, output_representation_axial, observation_axial, representation_axial = self.image_axial(x_axial)
            output_observation_coronal, output_representation_coronal, observation_coronal, representation_coronal = self.image_coronal(x_coronal)

            conf_axial = self.TCPConfidenceLayer_axial(observation_axial)
            conf_cro = self.TCPConfidenceLayer_cro(observation_coronal)

            if self.fuseend:
                output_observation_fuse = output_observation_axial + output_observation_coronal
                output_representation_fuse = output_representation_axial + output_representation_coronal
            else:
                if self.fuseadd:
                    if self.use_CONF:
                        fuse_observation = observation_coronal * conf_cro + observation_axial * conf_axial
                    else:
                        fuse_observation = observation_coronal + observation_axial
                else:
                    if self.use_CONF:
                        fuse_observation = torch.concat([observation_coronal * conf_cro, observation_axial * conf_axial],
                                                        dim=1)
                    else:
                        fuse_observation = torch.concat([observation_coronal, observation_axial], dim=1)
                output_observation_fuse, output_representation_fuse = self.fuse_image_image(fuse_observation)

            # return output_fuse, output_observation_axial, output_observation_coronal
            return (output_observation_fuse, output_representation_fuse), (output_observation_axial, output_observation_coronal), (conf_axial, conf_cro)

        if self.use_fuse_axial_coronal_farewell:
            # to refer to farewell
            output_observation_axial, output_representation_axial, observation_axial, representation_axial = self.image_axial(x_axial)
            output_observation_coronal, output_representation_coronal, observation_coronal, representation_coronal = self.image_coronal(x_coronal)

            fuse_observation = torch.concat([observation_coronal, observation_axial], dim=1)
            # output_fuse = self.fuse_image_image(fuse_observation)
            output_observation_fuse, output_representation_fuse = self.fuse_image_image(fuse_observation)
            # return output_fuse, output_observation_axial, output_observation_coronal
            return (output_observation_fuse, output_representation_fuse), (output_observation_axial, output_observation_coronal)


        if self.use_fuse_axial_coronal_clinical:

            output_observation_clinical, output_representation_clinical, observation_clinical, representation_clinical = self.clinical_backbone(clinical)
            output_observation_axial, output_representation_axial, observation_axial, representation_axial = self.image_axial(x_axial)
            output_observation_coronal, output_representation_coronal, observation_coronal, representation_coronal = self.image_coronal(x_coronal)


            fuse_observation = torch.concat([observation_coronal, observation_axial], dim=1)
            # output_fuse = self.fuse_image_image(fuse_observation)
            output_observation_fuse, output_representation_fuse = self.fuse_image_image(fuse_observation)


# add confidence
class Concat_DenseNet_Image_Clinical_Generalised_withConfidence(nn.Module):

    def __init__(self, image_backbone1=None, image_backbone2=None, clinical_backbone=None, fuse_backbone1=None,
                 fuse_backbone2=None, num_classes=1000, len_clinical=0, opt=None):

        super(Concat_DenseNet_Image_Clinical_Generalised_withConfidence, self).__init__()

        self.use_only_clinical = opt.use_only_clinical
        self.use_only_ct_axial = opt.use_only_ct_axial
        self.use_only_ct_coronal = opt.use_only_ct_coronal
        self.use_fuse_clinical_axial = opt.use_fuse_clinical_axial
        self.use_fuse_axial_coronal = opt.use_fuse_axial_coronal
        self.use_confidence = opt.use_CONF
        self.use_attention = opt.use_ATTEN
        # if opt.use_only_ct_coronal:
        self.image_coronal = image_backbone1
        # if opt.use_only_ct_axial:
        self.image_axial = image_backbone2
        # if opt.use_only_clinical:
        self.clinical_backbone = clinical_backbone
        # fc_clinical = self.clinical_backbone.classifier.in_features

        # add: try different ways
        # self.fuse_Bottleneck = VIB(in_ch=fc_img + 1024, z_dim=256, num_class=num_classes)
        # fc_img = self.image_axial.classifier1.in_features
        # if opt.use_fuse_clinical_axial:
        self.fuse_image_clinical = fuse_backbone1
        # VIB(in_ch=fc_img + 1024, z_dim=256, num_class=num_classes)
        # if opt.use_fuse_axial_coronal:
        self.fuse_image_image = fuse_backbone2
        # self.fuse_image_clinical = fuse_backbone
        # VIB(in_ch=fc_img + 1024, z_dim=256, num_class=num_classes)
        if self.use_confidence:
            self.ConfidenceLayer_clinical = nn.Linear(1024, 1)
            self.ConfidenceLayer_axial = nn.Linear(1024, 1)
            self.ConfidenceLayer_coronal = nn.Linear(1024, 1)

        if self.use_attention:
            self.Attention_clinical = nn.Linear(1024, 1024) # input, input
            self.Attentioo_axial = nn.Linear(1024, 1024)
            self.Attention_coronal = nn.Linear(1024, 1024)



    def forward(self, x_axial, x_coronal, clinical):

        if self.use_only_clinical:
            output_observation_clinical, output_representation_clinical, observation_clinical, representation_clinical = self.clinical_backbone(
                clinical)
            # return (output_observation_clinical, output_representation_clinical), (0,0)
            return (output_observation_clinical, output_observation_clinical), (
            output_observation_clinical, output_observation_clinical)

        if self.use_only_ct_axial:
            output_observation_axial, output_representation_axial, observation_axial, representation_axial = self.image_axial(
                x_axial)
            # return (output_observation_axial,output_representation_axial), (0, 0)
            return (output_observation_axial, output_observation_axial), (
            output_observation_axial, output_observation_axial)

        if self.use_only_ct_coronal:
            output_observation_coronal, output_representation_coronal, observation_coronal, representation_coronal = self.image_coronal(
                x_coronal)
            # return (output_observation_coronal, output_representation_coronal), (0, 0)
            return (output_observation_coronal, output_observation_coronal), (output_observation_coronal, output_observation_coronal)

        if self.use_fuse_clinical_axial:
            output_observation_clinical, output_representation_clinical, observation_clinical, representation_clinical = self.clinical_backbone(
                clinical)
            output_observation_axial, output_representation_axial, observation_axial, representation_axial = self.image_axial(
                x_axial)

            if self.use_confidence:
                conf_observation_axial = self.ConfidenceLayer_axial(observation_axial)
                conf_observation_clinical = self.ConfidenceLayer_clinical(observation_clinical)
            else:
                conf_observation_axial = 1
                conf_observation_clinical = 1

            if self.use_attention:
                atten_observation_axial = torch.sigmoid(self.Attentioo_axial(observation_axial))
                atten_observation_clinical = torch.sigmoid(self.Attention_clinical(observation_clinical))
            else:
                atten_observation_axial = 1
                atten_observation_clinical = 1

            observation_axial = observation_axial * atten_observation_axial
            observation_clinical = observation_clinical * atten_observation_clinical

            # fuse_observation = torch.concat([observation_clinical, observation_axial], dim=1)
            fuse_observation = torch.concat([observation_clinical * conf_observation_clinical, observation_axial * conf_observation_axial], dim=1)
            output_observation_fuse, output_representation_fuse = self.fuse_image_clinical(fuse_observation)

            return (output_observation_fuse, output_representation_fuse), (output_observation_axial, output_observation_clinical), (conf_observation_axial, conf_observation_clinical), (atten_observation_axial, atten_observation_clinical)

        if self.use_fuse_axial_coronal:
            # to refer to farewell
            output_observation_axial, output_representation_axial, observation_axial, representation_axial = self.image_axial(
                x_axial)
            output_observation_coronal, output_representation_coronal, observation_coronal, representation_coronal = self.image_coronal(
                x_coronal)

            fuse_observation = torch.concat([observation_coronal, observation_axial], dim=1)
            # output_fuse = self.fuse_image_image(fuse_observation)
            output_observation_fuse, output_representation_fuse = self.fuse_image_image(fuse_observation)
            # return output_fuse, output_observation_axial, output_observation_coronal
            return (output_observation_fuse, output_representation_fuse), (
            output_observation_axial, output_observation_coronal)


def concat_densenet_image_clinical_densenet121(pretrained, opt, **kwargs):

    clinical_backbone = Clinical(num_classes=2, len_clinical=opt.len_clinical)
    image_backbone1 = BI_basic_densenet121_2fc_relu(pretrained=pretrained, num_classes=2, drop_rate=opt.drop_rate)
    image_backbone2 = BI_basic_densenet121_2fc_relu(pretrained=pretrained, num_classes=2, drop_rate=opt.drop_rate)

    fc_img = image_backbone1.classifier1.in_features
    fc_clinical = clinical_backbone.classifier_observation.in_features
    fuse_backbone1 = fuse_bottleneck_withobservation(in_ch=fc_img + fc_clinical, num_class=2, dim_bottleneck=opt.dim_bottleneck)
    # fuse_backbone2 = fuse_bottleneck(in_ch=fc_img*2, num_class=2)
    if opt.fuseadd:
        fuse_backbone2 = fuse_bottleneck_withobservation(in_ch=fc_img, num_class=2,dim_bottleneck=opt.dim_bottleneck)
    else:
        fuse_backbone2 = fuse_bottleneck_withobservation(in_ch=fc_img*2, num_class=2, dim_bottleneck=opt.dim_bottleneck)

    # if opt.use_CONF:
    # model = Concat_DenseNet_Image_Clinical_Generalised_withConfidence(clinical_backbone=clinical_backbone,
    #                                                        image_backbone1=image_backbone1,
    #                                                        image_backbone2=image_backbone2,
    #                                                        fuse_backbone1=fuse_backbone1, fuse_backbone2=fuse_backbone2,
    #                                                        opt=opt)
    # else:
    model = Concat_DenseNet_Image_Clinical_Generalised(clinical_backbone=clinical_backbone,image_backbone1=image_backbone1, image_backbone2=image_backbone2, fuse_backbone1=fuse_backbone1, fuse_backbone2=fuse_backbone2, opt=opt)

    # if opt.use_only_clinical:
    #     clinical_backbone = Clinical(num_classes=2, len_clinical=opt.len_clinical)
    #     model = Concat_DenseNet_Image_Clinical_Generalised(clinical_backbone=clinical_backbone, opt=opt)
    #
    # if opt.use_only_ct_axial:
    #     image_backbone = BI_basic_densenet121_2fc_relu(pretrained=opt.pretrained, num_classes=2, drop_rate=opt.drop_rate)
    #     model = Concat_DenseNet_Image_Clinical_Generalised(image_backbone=image_backbone, opt=opt)
    #
    # if opt.use_only_ct_coronal:
    #     image_backbone = BI_basic_densenet121_2fc_relu(pretrained=opt.pretrained, num_classes=2, drop_rate=opt.drop_rate)
    #     model = Concat_DenseNet_Image_Clinical_Generalised(image_backbone=image_backbone, opt=opt)
    #
    # if opt.use_fuse_clinical_axial:
    #     clinical_backbone = Clinical(num_classes=2, len_clinical=opt.len_clinical)
    #     image_backbone = BI_basic_densenet121_2fc_relu(pretrained=opt.pretrained, num_classes=2,drop_rate=opt.drop_rate)
    #
    #     fc_img =  image_backbone.classifier.in_features
    #     fc_clinical = clinical_backbone.bottleneck.in_features
    #     fuse_backbone = fuse_bottleneck(in_ch = fc_img + fc_clinical, num_class = 2)
    #     model = Concat_DenseNet_Image_Clinical_Generalised(clinical_backbone=clinical_backbone, image_backbone=image_backbone, fuse_backbone=fuse_backbone, opt=opt)
    #
    # if opt.use_fuse_axial_coronal:
    #     image_backbone = BI_basic_densenet121_2fc_relu(pretrained=opt.pretrained, num_classes=2,drop_rate=opt.drop_rate)
    #
    #     fc_img = image_backbone.classifier.in_features
    #     fuse_backbone = fuse_bottleneck(in_ch=fc_img*2, num_class=2)
    #     model = Concat_DenseNet_Image_Clinical_Generalised(image_backbone=image_backbone, fuse_backbone=fuse_backbone, opt=opt)

    return model


