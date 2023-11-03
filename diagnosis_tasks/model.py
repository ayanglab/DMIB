import torch.nn as nn
import torch
import torch.nn.functional as F
from torch.nn.functional import kl_div

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


class MMDynamic(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)

    def forward(self, data_list, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            FeatureInfo[view] = torch.sigmoid(self.FeatureInforEncoder[view](data_list[view]))
            feature[view] = data_list[view] * FeatureInfo[view]
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])
            TCPConfidence[view] = self.TCPConfidenceLayer[view](feature[view])
            feature[view] = feature[view] * TCPConfidence[view]

        MMfeature = torch.cat([i for i in feature.values()], dim=1)
        MMlogit = self.MMClasifier(MMfeature)
        if infer:
            return MMlogit
        MMLoss = torch.mean(criterion(MMlogit, label))
        for view in range(self.views):
            MMLoss = MMLoss + torch.mean(FeatureInfo[view])
            pred = F.softmax(TCPLogit[view], dim=1)
            p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            confidence_loss = torch.mean(
                F.mse_loss(TCPConfidence[view].view(-1), p_target) + criterion(TCPLogit[view], label))
            MMLoss = MMLoss + confidence_loss
        return MMLoss, MMlogit

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit


class MMDynamic_blank_VSD(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))

        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier_ob = nn.Sequential(*self.MMClasifier)

        layers = []
        layers += [nn.Linear(1500, 1000)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(1000, 1500)]
        
        self.bottleneck1 = nn.Sequential(*layers)
        self.MMClasifier_re1 = nn.Linear(1500, num_class)


    def forward(self, data_list, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()

        for view in range(self.views):
            feature[view] = data_list[view]
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)

            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])

        MMfeature = torch.cat([i for i in feature.values()], dim=1)
        MMlogit_ob = self.MMClasifier_ob(MMfeature)
        feature_hidden = self.bottleneck1(MMfeature)
        MMlogit_re = self.MMClasifier_re1(feature_hidden)

        MMlogit = MMlogit_re

        if infer:
            return MMlogit

        MMLoss_ob = torch.mean(criterion(MMlogit_ob, label))
        MMLoss_re = torch.mean(criterion(MMlogit_re, label))
        MMLoss_IB = kl_div(input=F.softmax(MMlogit_ob.detach()/1),target=F.softmax(MMlogit_re/1))

        MMLoss = MMLoss_re + MMLoss_ob + MMLoss_IB * 500

        for view in range(self.views):
            pred = F.softmax(TCPLogit[view], dim=1)
            confidence_loss = criterion(TCPLogit[view], label) * 1000
            MMLoss = MMLoss + confidence_loss

        return MMLoss, MMlogit

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit


class MMDynamic_blank_directfuse_re(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))

        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier_ob = nn.Sequential(*self.MMClasifier)

        layers = []

        layers += [nn.Linear(1500, 1000)]
        layers += [nn.Linear(1000, 1500)]

        self.bottleneck1 = nn.Sequential(*layers)
        self.MMClasifier_re1 = nn.Linear(1500, num_class)

    def forward(self, data_list, label=None, infer=False):

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()

        for view in range(self.views):
            feature[view] = data_list[view]
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)

            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])
            
        MMfeature = torch.cat([i for i in feature.values()], dim=1)
        MMlogit_ob = self.MMClasifier_ob(MMfeature)
        feature_hidden = self.bottleneck1(MMfeature)
        MMlogit_re = self.MMClasifier_re1(feature_hidden)

        MMlogit = MMlogit_re

        if infer:
            return MMlogit

        MMLoss_ob = torch.mean(criterion(MMlogit_ob, label))
        MMLoss_re = torch.mean(criterion(MMlogit_re, label))
        MMLoss_IB = kl_div(input=F.softmax(MMlogit_ob.detach()/1),
                             target=F.softmax(MMlogit_re/1))

        MMLoss = MMLoss_re
        
        for view in range(self.views):
            pred = F.softmax(TCPLogit[view], dim=1)
            confidence_loss = criterion(TCPLogit[view], label) * 1000
        return MMLoss, MMlogit

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit

    
class MMDynamic_blank_directfuse_re_sided(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))

        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier_ob = nn.Sequential(*self.MMClasifier)

        layers = []
        layers += [nn.Linear(1500, 1000)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(1000, 1500)]
        
        self.bottleneck1 = nn.Sequential(*layers)
        self.MMClasifier_re1 = nn.Linear(1500, num_class)

    def forward(self, data_list, label=None, infer=False):

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()

        for view in range(self.views):
            feature[view] = data_list[view]
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)

            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])

        MMfeature = torch.cat([i for i in feature.values()], dim=1)
        MMlogit_ob = self.MMClasifier_ob(MMfeature)
        feature_hidden = self.bottleneck1(MMfeature)
        MMlogit_re = self.MMClasifier_re1(feature_hidden)

        MMlogit = MMlogit_re
        if infer:
            return MMlogit
        MMLoss_ob = torch.mean(criterion(MMlogit_ob, label))
        MMLoss_re = torch.mean(criterion(MMlogit_re, label))
        MMLoss_IB = kl_div(input=F.softmax(MMlogit_ob.detach()/1),
                             target=F.softmax(MMlogit_re/1))

        MMLoss = MMLoss_re
        
        for view in range(self.views):
            pred = F.softmax(TCPLogit[view], dim=1)
            confidence_loss = criterion(TCPLogit[view], label) * 1000
            MMLoss = MMLoss + confidence_loss

        return MMLoss, MMlogit

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit


class MMDynamic_blank_directfuse_re_IB(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))

        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier_ob = nn.Sequential(*self.MMClasifier)

        layers = []
        layers += [nn.Linear(1500, 1000)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(1000, 1500)]
        
        self.bottleneck1 = nn.Sequential(*layers)
        self.MMClasifier_re1 = nn.Linear(1500, num_class)
    
    def forward(self, data_list, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()

        for view in range(self.views):
            feature[view] = data_list[view]
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)

            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])

        MMfeature = torch.cat([i for i in feature.values()], dim=1)
        MMlogit_ob = self.MMClasifier_ob(MMfeature)
        feature_hidden = self.bottleneck1(MMfeature)
        MMlogit_re = self.MMClasifier_re1(feature_hidden)

        MMlogit = MMlogit_re

        if infer:
            return MMlogit

        MMLoss_ob = torch.mean(criterion(MMlogit_ob, label))
        MMLoss_re = torch.mean(criterion(MMlogit_re, label))
        MMLoss_IB = kl_div(input=F.softmax(MMlogit_ob.detach()/1),
                             target=F.softmax(MMlogit_re/1))

        MMLoss = MMLoss_re + MMLoss_ob + MMLoss_IB * 500
        
        for view in range(self.views):
            pred = F.softmax(TCPLogit[view], dim=1)
            confidence_loss = criterion(TCPLogit[view], label) * 1000
            MMLoss = MMLoss + confidence_loss

        return MMLoss, MMlogit

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit


class MMDynamic_blank_DMIB_ablation_IBsided(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))

        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier_ob = nn.Sequential(*self.MMClasifier)

        layers = []
        layers += [nn.Linear(1500, 1000)] # 1000>1200
        layers += [nn.ReLU()]
        layers += [nn.Linear(1000, 1500)]
        
        self.bottleneck1 = nn.Sequential(*layers)
        self.MMClasifier_re1 = nn.Linear(1500, num_class)
    
    def forward(self, data_list, label=None, infer=False):

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()

        for view in range(self.views):
            feature[view] = data_list[view]
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)

            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])

        MMfeature = torch.cat([i for i in feature.values()], dim=1)
        MMlogit_ob = self.MMClasifier_ob(MMfeature)
        feature_hidden = self.bottleneck1(MMfeature)
        MMlogit_re = self.MMClasifier_re1(feature_hidden)

        MMlogit = MMlogit_re
        
        if infer:
            return MMlogit
        MMLoss_ob = torch.mean(criterion(MMlogit_ob, label))
        MMLoss_re = torch.mean(criterion(MMlogit_re, label))
        MMLoss_IB = kl_div(input=F.softmax(MMlogit_ob.detach()/1),
                             target=F.softmax(MMlogit_re/1))

        MMLoss = MMLoss_re + MMLoss_ob

        for view in range(self.views):
            pred = F.softmax(TCPLogit[view], dim=1)
            confidence_loss = criterion(TCPLogit[view], label) * 1000
            MMLoss = MMLoss + confidence_loss

        return MMLoss, MMlogit

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit

    
class MMDynamic_blank_DMIB_ablation_IB(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.FeatureEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))

        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier_ob = nn.Sequential(*self.MMClasifier)

        layers = []
        layers += [nn.Linear(1500, 1000)]  # 1000>1200
        layers += [nn.ReLU()]
        layers += [nn.Linear(1000, 1500)]

        self.bottleneck1 = nn.Sequential(*layers)
        self.MMClasifier_re1 = nn.Linear(1500, num_class)

    def forward(self, data_list, label=None, infer=False):

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()

        for view in range(self.views):
            feature[view] = data_list[view]
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)

            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])
            
        MMfeature = torch.cat([i for i in feature.values()], dim=1)
        MMlogit_ob = self.MMClasifier_ob(MMfeature)
        feature_hidden = self.bottleneck1(MMfeature)
        MMlogit_re = self.MMClasifier_re1(feature_hidden)

        MMlogit = MMlogit_re

        if infer:
            return MMlogit
        
        MMLoss_ob = torch.mean(criterion(MMlogit_ob, label))
        MMLoss_re = torch.mean(criterion(MMlogit_re, label))
        MMLoss_IB = kl_div(input=F.softmax(MMlogit_ob.detach() / 1),
                           target=F.softmax(MMlogit_re / 1))

        MMLoss = MMLoss_re + MMLoss_ob
        
        return MMLoss, MMlogit

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit


class MMDynamic_blank_DMIB_ablation_directfuse(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.FeatureEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))

        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier_ob = nn.Sequential(*self.MMClasifier)

        layers = []
        layers += [nn.Linear(1500, 1000)]
        layers += [nn.ReLU()]
        layers += [nn.Linear(1000, 1500)]
        
        self.bottleneck1 = nn.Sequential(*layers)
        self.MMClasifier_re1 = nn.Linear(1500, num_class)

    def forward(self, data_list, label=None, infer=False):

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()

        for view in range(self.views):
            feature[view] = data_list[view]
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)

            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])

        MMfeature = torch.cat([i for i in feature.values()], dim=1)
        MMlogit_ob = self.MMClasifier_ob(MMfeature)
        feature_hidden = self.bottleneck1(MMfeature)
        MMlogit_re = self.MMClasifier_re1(feature_hidden)

        MMlogit = MMlogit_ob

        if infer:
            return MMlogit
        
        MMLoss_ob = torch.mean(criterion(MMlogit_ob, label))
        MMLoss_re = torch.mean(criterion(MMlogit_re, label))
        MMLoss_IB = kl_div(input=F.softmax(MMlogit_ob.detach() / 1),
                           target=F.softmax(MMlogit_re / 1))

        MMLoss = MMLoss_ob

        return MMLoss, MMlogit

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit


class MMDynamic_blank_DMIB_ablation_sided(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.FeatureEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))

        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier_ob = nn.Sequential(*self.MMClasifier)

        layers = []
        layers += [nn.Linear(1500, 1000)]  # 1000>1200
        layers += [nn.ReLU()]
        layers += [nn.Linear(1000, 1500)]
        
        self.bottleneck1 = nn.Sequential(*layers)
        self.MMClasifier_re1 = nn.Linear(1500, num_class)
        
    def forward(self, data_list, label=None, infer=False):

        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()

        for view in range(self.views):
            feature[view] = data_list[view]
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)

            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])

        MMfeature = torch.cat([i for i in feature.values()], dim=1)
        MMlogit_ob = self.MMClasifier_ob(MMfeature)
        feature_hidden = self.bottleneck1(MMfeature)
        MMlogit_re = self.MMClasifier_re1(feature_hidden)

        MMlogit = MMlogit_ob

        if infer:
            return MMlogit

        MMLoss_ob = torch.mean(criterion(MMlogit_ob, label))
        MMLoss_re = torch.mean(criterion(MMlogit_re, label))
        MMLoss_IB = kl_div(input=F.softmax(MMlogit_ob.detach() / 1),
                           target=F.softmax(MMlogit_re / 1))

        MMLoss = MMLoss_ob
        for view in range(self.views):
            pred = F.softmax(TCPLogit[view], dim=1)
            confidence_loss = criterion(TCPLogit[view], label) * 1000
            MMLoss = MMLoss + confidence_loss

        return MMLoss, MMlogit

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit


class MMDynamic_blank(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)

    def forward(self, data_list, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            feature[view] = data_list[view]
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])

        MMfeature = torch.cat([i for i in feature.values()], dim=1)
        MMlogit = self.MMClasifier(MMfeature)
        if infer:
            return MMlogit

        MMLoss = torch.mean(criterion(MMlogit, label))

        for view in range(self.views):
            # MMLoss = MMLoss + torch.mean(FeatureInfo[view])
            pred = F.softmax(TCPLogit[view], dim=1)
            p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            # confidence_loss = torch.mean(F.mse_loss(TCPConfidence[view].view(-1), p_target) + criterion(TCPLogit[view], label))
            confidence_loss = criterion(TCPLogit[view], label)
            # MMLoss = MMLoss + confidence_loss

        return MMLoss, MMlogit

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit


class MMDynamic_blank_CONF(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)

    def forward(self, data_list, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            # FeatureInfo[view] = torch.sigmoid(self.FeatureInforEncoder[view](data_list[view]))
            # feature[view] = data_list[view] * FeatureInfo[view]
            feature[view] = data_list[view]
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])
            TCPConfidence[view] = self.TCPConfidenceLayer[view](feature[view])
            feature[view] = feature[view] * TCPConfidence[view]

        MMfeature = torch.cat([i for i in feature.values()], dim=1)
        MMlogit = self.MMClasifier(MMfeature)
        if infer:
            return MMlogit

        MMLoss = torch.mean(criterion(MMlogit, label))

        for view in range(self.views):
        #     MMLoss = MMLoss + torch.mean(FeatureInfo[view])
            pred = F.softmax(TCPLogit[view], dim=1)
            p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            confidence_loss = torch.mean(
                F.mse_loss(TCPConfidence[view].view(-1), p_target) + criterion(TCPLogit[view], label))
            # confidence_loss = torch.mean( 0 + criterion(TCPLogit[view], label))
            MMLoss = MMLoss + confidence_loss

        return MMLoss, MMlogit

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit


class MMDynamic_blank_ATTEN(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)

    def forward(self, data_list, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            FeatureInfo[view] = torch.sigmoid(self.FeatureInforEncoder[view](data_list[view]))
            feature[view] = data_list[view] * FeatureInfo[view]
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
            
        MMfeature = torch.cat([i for i in feature.values()], dim=1)
        MMlogit = self.MMClasifier(MMfeature)
        if infer:
            return MMlogit

        MMLoss = torch.mean(criterion(MMlogit, label))
        return MMLoss, MMlogit

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit


class MMDynamic_endfuse(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(self.views * hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)

    def forward(self, data_list, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence, preds = dict(), dict(), dict(), dict(), dict()
        for view in range(self.views):
            feature[view] = data_list[view]

            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])
            TCPConfidence[view] = self.TCPConfidenceLayer[view](feature[view])
            preds[view] = F.softmax(TCPLogit[view], dim=1)

        MMlogit = 0
        for view in range(self.views):
            MMlogit += preds[view]

        if infer:
            return MMlogit
        MMLoss = torch.mean(criterion(MMlogit, label))

        for view in range(self.views):
            pred = F.softmax(TCPLogit[view], dim=1)
            p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            confidence_loss = torch.mean(
                F.mse_loss(TCPConfidence[view].view(-1), p_target) + criterion(TCPLogit[view], label))
        
        return MMLoss, MMlogit

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit


class MMDynamic_directfuse(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_class, dropout):
        super().__init__()
        self.views = len(in_dim)
        self.classes = num_class
        self.dropout = dropout

        self.FeatureInforEncoder = nn.ModuleList(
            [LinearLayer(in_dim[view], in_dim[view]) for view in range(self.views)])
        self.TCPConfidenceLayer = nn.ModuleList([LinearLayer(hidden_dim[0], 1) for _ in range(self.views)])
        self.TCPClassifierLayer = nn.ModuleList([LinearLayer(hidden_dim[0], num_class) for _ in range(self.views)])
        self.FeatureEncoder = nn.ModuleList([LinearLayer(in_dim[view], hidden_dim[0]) for view in range(self.views)])

        self.MMClasifier = []
        for layer in range(1, len(hidden_dim) - 1):
            self.MMClasifier.append(LinearLayer(hidden_dim[0], hidden_dim[layer]))
            self.MMClasifier.append(nn.ReLU())
            self.MMClasifier.append(nn.Dropout(p=dropout))
        if len(self.MMClasifier):
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        else:
            self.MMClasifier.append(LinearLayer(hidden_dim[-1], num_class))
        self.MMClasifier = nn.Sequential(*self.MMClasifier)

    def forward(self, data_list, label=None, infer=False):
        criterion = torch.nn.CrossEntropyLoss(reduction='none')
        FeatureInfo, feature, TCPLogit, TCPConfidence = dict(), dict(), dict(), dict()
        for view in range(self.views):
            feature[view] = data_list[view]
            feature[view] = self.FeatureEncoder[view](feature[view])
            feature[view] = F.relu(feature[view])
            feature[view] = F.dropout(feature[view], self.dropout, training=self.training)
            TCPLogit[view] = self.TCPClassifierLayer[view](feature[view])
            TCPConfidence[view] = self.TCPConfidenceLayer[view](feature[view])
            feature[view] = feature[view] * TCPConfidence[view]

        MMfeature = torch.zeros_like(feature[0])
        for view in range(self.views):
            MMfeature += feature[view]
        MMlogit = self.MMClasifier(MMfeature)
        if infer:
            return MMlogit

        MMLoss = torch.mean(criterion(MMlogit, label))

        for view in range(self.views):
            pred = F.softmax(TCPLogit[view], dim=1)
            p_target = torch.gather(input=pred, dim=1, index=label.unsqueeze(dim=1)).view(-1)
            confidence_loss = torch.mean(
                F.mse_loss(TCPConfidence[view].view(-1), p_target) + criterion(TCPLogit[view], label))
            MMLoss = MMLoss + confidence_loss

        return MMLoss, MMlogit

    def infer(self, data_list):
        MMlogit = self.forward(data_list, infer=True)
        return MMlogit
