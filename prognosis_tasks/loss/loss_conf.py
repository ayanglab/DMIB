import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import kl_div
from loss.loss_focal import FocalLoss

def loss_CONF(pred_axial, pred_cli_cro, conf_axial, conf_cli_cro, label):
    # criterion = torch.nn.CrossEntropyLoss(reduction='none')
    p_target_axial = torch.gather(input=pred_axial, dim=1, index=label.unsqueeze(dim=1)).view(-1)
    conf_loss_axial = torch.mean(
        F.mse_loss(conf_axial.view(-1), p_target_axial))
    p_target_cli_cro = torch.gather(input=pred_cli_cro, dim=1, index=label.unsqueeze(dim=1)).view(-1)
    conf_loss_cli_cro = torch.mean(F.mse_loss(conf_cli_cro.view(-1), p_target_cli_cro))

    return conf_loss_axial + conf_loss_cli_cro



