import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.functional import kl_div
from loss.loss_focal import FocalLoss

############################################################################################
# Classification Loss
############################################################################################
class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """
    def __init__(self, num_classes, epsilon=0.0, use_gpu=True):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = num_classes
        self.epsilon = epsilon
        self.use_gpu = use_gpu
        self.logsoftmax = nn.LogSoftmax(dim=1)

    def forward(self, inputs, targets):
        """
        Args:
            inputs: prediction matrix (before softmax) with shape (batch_size, num_classes)
            targets: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(inputs)
        targets = torch.zeros(log_probs.size()).scatter_(1, targets.unsqueeze(1).data.cpu(), 1)
        if self.use_gpu: targets = targets.cuda()
        targets = (1 - self.epsilon) * targets + self.epsilon / self.num_classes
        loss = (- targets * log_probs).mean(0).sum()
        return loss


class Loss_IB(nn.Module):
    def __init__(self, args):
        super(Loss_IB, self).__init__()
        
        self.args = args
        self.CE_Loss = nn.CrossEntropyLoss().cuda()
        self.focal = FocalLoss(class_num=2, alpha=args.alpha, gamma=args.gamma, size_average=True)
        self.softmax = torch.nn.Softmax(dim=1)
        
    def forward(self, inputs, label):

        # i_representation, i_observation = inputs[0], inputs[1]
        i_observation, i_representation = inputs[0][0], inputs[0][1]

        # Classification loss
        focal_representation = self.focal(self.softmax(i_representation), label)
        focal_observation = self.focal(self.softmax(i_observation), label)
        vsd_loss = kl_div(input=self.softmax(i_observation.detach()/self.args.temperature), target=self.softmax(i_representation/self.args.temperature))

        L = self.args.focal_loss_representation * focal_representation + self.args.focal_loss_observation * focal_observation + self.args.VSD_loss * vsd_loss

        return L, focal_observation, focal_representation, vsd_loss


class Loss_IB_only(nn.Module):
    def __init__(self, args):
        super(Loss_IB_only, self).__init__()
        self.args = args
        self.softmax = torch.nn.Softmax(dim=1)

    def forward(self, inputs, label):
        i_observation, i_representation = inputs[0][0], inputs[0][1]

        vsd_loss = kl_div(input=self.softmax(i_observation.detach()/self.args.temperature), target=self.softmax(i_representation/self.args.temperature))

        return vsd_loss



