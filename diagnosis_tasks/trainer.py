""" Training and testing of the model
"""
import os
from re import I
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
import torch
import torch.nn.functional as F
from model import *
import sys
cuda = True if torch.cuda.is_available() else False
from torch.nn.functional import kl_div


def one_hot_tensor(y, num_dim):
    y_onehot = torch.zeros(y.shape[0], num_dim)
    y_onehot.scatter_(1, y.view(-1, 1), 1)
    return y_onehot


def prepare_trte_data(data_folder):
    num_view = 3
    labels_tr = np.loadtxt(os.path.join(data_folder, "labels_tr.csv"), delimiter=',')
    labels_te = np.loadtxt(os.path.join(data_folder, "labels_te.csv"), delimiter=',')
    labels_tr = labels_tr.astype(int)
    labels_te = labels_te.astype(int)
    data_tr_list = []
    data_te_list = []
    for i in range(1, num_view + 1):

        data_tr_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_tr.csv"), delimiter=','))
        data_te_list.append(np.loadtxt(os.path.join(data_folder, str(i) + "_te.csv"), delimiter=','))
        print(str(i)+": success")

    eps = 1e-10
    X_train_min = [np.min(data_tr_list[i], axis=0, keepdims=True) for i in range(len(data_tr_list))]
    data_tr_list = [data_tr_list[i] - np.tile(X_train_min[i], [data_tr_list[i].shape[0], 1]) for i in
                    range(len(data_tr_list))]
    data_te_list = [data_te_list[i] - np.tile(X_train_min[i], [data_te_list[i].shape[0], 1]) for i in
                    range(len(data_tr_list))]
    X_train_max = [np.max(data_tr_list[i], axis=0, keepdims=True) + eps for i in range(len(data_tr_list))]
    data_tr_list = [data_tr_list[i] / np.tile(X_train_max[i], [data_tr_list[i].shape[0], 1]) for i in
                    range(len(data_tr_list))]
    data_te_list = [data_te_list[i] / np.tile(X_train_max[i], [data_te_list[i].shape[0], 1]) for i in
                    range(len(data_tr_list))]

    num_tr = data_tr_list[0].shape[0]
    num_te = data_te_list[0].shape[0]
    data_mat_list = []
    for i in range(num_view):
        data_mat_list.append(np.concatenate((data_tr_list[i], data_te_list[i]), axis=0))
    data_tensor_list = []
    for i in range(len(data_mat_list)):
        data_tensor_list.append(torch.FloatTensor(data_mat_list[i]))
        if cuda:
            data_tensor_list[i] = data_tensor_list[i].cuda()
    idx_dict = {}
    idx_dict["tr"] = list(range(num_tr))
    idx_dict["te"] = list(range(num_tr, (num_tr + num_te)))
    data_train_list = []
    data_all_list = []
    data_test_list = []
    for i in range(len(data_tensor_list)):
        data_train_list.append(data_tensor_list[i][idx_dict["tr"]].clone())
        data_all_list.append(torch.cat((data_tensor_list[i][idx_dict["tr"]].clone(),
                                        data_tensor_list[i][idx_dict["te"]].clone()), 0))
        data_test_list.append(data_tensor_list[i][idx_dict["te"]].clone())
    labels = np.concatenate((labels_tr, labels_te))
    return data_train_list, data_test_list, idx_dict, labels


def train_epoch(data_list, label, model, optimizer):
    model.train()
    optimizer.zero_grad()
    loss, _ = model(data_list, label)
    loss = torch.mean(loss)
    loss.backward()
    optimizer.step()


def test_epoch(data_list, model):
    model.eval()
    with torch.no_grad():
        logit = model.infer(data_list)
        prob = F.softmax(logit, dim=1).data.cpu().numpy()
    return prob


def save_checkpoint(model, checkpoint_path, filename="checkpoint.pt"):
    os.makedirs(checkpoint_path, exist_ok=True)
    filename = os.path.join(checkpoint_path, filename)
    torch.save(model, filename)


def load_checkpoint(model, path):
    best_checkpoint = torch.load(path)
    model.load_state_dict(best_checkpoint)


def train(data_folder, modelpath, testonly, testmodel):

    test_inverval = 50
    if 'BRCA' in data_folder:
        hidden_dim = [500]
        num_epoch = 2500
        lr = 1e-4
        step_size = 500
        num_class = 5
    elif 'ROSMAP' in data_folder:
        hidden_dim = [300]
        num_epoch = 1000
        lr = 1e-4
        step_size = 500
        num_class = 2

    # dataset
    data_tr_list, data_test_list, trte_idx, labels_trte = prepare_trte_data(data_folder)
    labels_tr_tensor = torch.LongTensor(labels_trte[trte_idx["tr"]])
    onehot_labels_tr_tensor = one_hot_tensor(labels_tr_tensor, num_class)
    labels_tr_tensor = labels_tr_tensor.cuda()
    onehot_labels_tr_tensor = onehot_labels_tr_tensor.cuda()


    dim_list = [x.shape[1] for x in data_tr_list]
    if testmodel == 'dynamic':
        model = MMDynamic(dim_list, hidden_dim, num_class, dropout=0.5)

    if testmodel == 'DMIB':
        model = MMDynamic_blank_directfuse_re_IB(dim_list, hidden_dim, num_class, dropout=0.5)

    if testmodel == 'DMIB_ablation_IBsided':
        model = MMDynamic_blank_DMIB_ablation_IBsided(dim_list, hidden_dim, num_class, dropout=0.5)

    if testmodel == 'DMIB_ablation_IB':
        model = MMDynamic_blank_DMIB_ablation_IB(dim_list, hidden_dim, num_class, dropout=0.5)

    if testmodel == 'DMIB_ablation_directfuse':
        model = MMDynamic_blank_DMIB_ablation_directfuse(dim_list, hidden_dim, num_class, dropout=0.5)
    
    if testmodel == 'DMIB_ablation_sided':
        model = MMDynamic_blank_DMIB_ablation_sided(dim_list, hidden_dim, num_class, dropout=0.5)


    # model = MMDynamic_directfuse(dim_list, hidden_dim, num_class, dropout=0.5)
    # model = MMDynamic_endfuse(dim_list, hidden_dim, num_class, dropout=0.5)
    # model = MMDynamic_blank(dim_list, hidden_dim, num_class, dropout=0.5)
    # model = MMDynamic_blank_ATTEN(dim_list, hidden_dim, num_class, dropout=0.5)
    # model = MMDynamic_blank_CONF(dim_list, hidden_dim, num_class, dropout=0.5)
    # model = MMDynamic_blank_VSD(dim_list, hidden_dim, num_class, dropout=0.5)
    # model = MMDynamic_blank_directfuse_re(dim_list, hidden_dim, num_class, dropout=0.5)
    # model = MMDynamic_blank_directfuse_re_sided(dim_list, hidden_dim, num_class, dropout=0.5)
    # model = MMDynamic_blank_directfuse_re_IB(dim_list, hidden_dim, num_class, dropout=0.5)

    model.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=0.2)


    if testonly:
        load_checkpoint(model, os.path.join(modelpath, data_folder, 'checkpoint_author.pt'))
        te_prob = test_epoch(data_test_list, model)
        if num_class == 2:
            print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            print("Test F1: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            print("Test AUC: {:.5f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1])))
        else:
            print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
            print("Test F1 weighted: {:.5f}".format(
                f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
            print("Test F1 macro: {:.5f}".format(
                f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
    else:
        print("\nTraining...")
        for epoch in range(num_epoch + 1):
            train_epoch(data_tr_list, labels_tr_tensor, model, optimizer)
            scheduler.step()
            if epoch % test_inverval == 0:
                te_prob = test_epoch(data_test_list, model)
                print("\nTest: Epoch {:d}".format(epoch))
                if num_class == 2:
                    result_acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                    result_f1  = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                    result_auc = roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1])
                    print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                    print("Test F1: {:.5f}".format(f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                    print("Test AUC: {:.5f}".format(roc_auc_score(labels_trte[trte_idx["te"]], te_prob[:, 1])))
                else:
                    result_acc = accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))
                    result_weighted  = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')
                    result_macro = f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')
                    print("Test ACC: {:.5f}".format(accuracy_score(labels_trte[trte_idx["te"]], te_prob.argmax(1))))
                    print("Test F1 weighted: {:.5f}".format(
                        f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='weighted')))
                    print("Test F1 macro: {:.5f}".format(
                        f1_score(labels_trte[trte_idx["te"]], te_prob.argmax(1), average='macro')))
        save_checkpoint(model.state_dict(), os.path.join(modelpath, data_folder), 'checkpoint_blank.pt')

    if num_class == 2:
        return result_acc, result_f1, result_auc
    else:
        return result_acc, result_weighted, result_macro