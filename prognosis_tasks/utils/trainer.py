import sys
sys.path.append("..")
# from utility import *
from argparse import ArgumentParser
import os
import re
import seaborn as sns
sns.set()
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import SimpleITK as sitk
import matplotlib.pyplot as plt
import math
import datetime
import os, glob, json
import cv2
import copy
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.metrics import roc_auc_score
import wandb
# loss
import torch.nn as nn
from loss.loss_sinkhorn import SinkhornDistance
from loss.loss_focal import FocalLoss
from loss.loss_IB import Loss_IB_only
from loss.loss_conf import loss_CONF


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

def get_roc(pos_prob, y_true):
    pos = y_true[y_true == 1]
    neg = y_true[y_true == 0]
    threshold = np.sort(pos_prob)[::-1]
    y = y_true[pos_prob.argsort()[::-1]]
    tpr_all = [0]
    fpr_all = [0]
    th_all = [0]
    tpr = 0
    fpr = 0
    x_step = 1 / (float(len(neg)))
    y_step = 1 / (float(len(pos)))
    y_sum = 0
    for i in range(len(threshold)):
        th_all.append(threshold[i])
        if y[i] == 1:
            tpr += y_step
            tpr_all.append(tpr)
            fpr_all.append(fpr)
        else:
            fpr += x_step
            fpr_all.append(fpr)
            tpr_all.append(tpr)
            y_sum += tpr
    return tpr_all, fpr_all, y_sum * x_step, th_all


def adjust_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def plot_statistic_continuous(group1, group2, group_info, features, plot_hist = False):
    for feature in features:
        gtoup1_attributes = []
        gtoup2_attributes = []
        for patient_id in group1:
            gtoup1_attributes.append(group_info.loc[(group_info['Pseudonym'] == patient_id), feature].values[0])
        for patient_id in group2:
            gtoup2_attributes.append(group_info.loc[(group_info['Pseudonym'] == patient_id), feature].values[0])
        if plot_hist:
            sns.distplot(gtoup1_attributes, IBns=2, kde=False, hist_kws={"color": "steelblue"}, label="Train")
            plot.show()
            sns.distplot(gtoup1_attributes, hist=False, kde_kws={"color": "red", "linestyle": "-"}, norm_hist=True,
                         label="Train-line")
            sns.distplot(gtoup2_attributes, hist=False, kde_kws={"color": "blue", "linestyle": "-"}, norm_hist=True,
                         label="Teat-line")
            plt.show()

            
def plot_statistic_category(group1, group2, group_info, features):

    for feature in features:
        gtoup1_attributes = []
        gtoup2_attributes = []
        for patient_id in group1:
            gtoup1_attributes.append(group_info.loc[(group_info['Pseudonym'] == patient_id), feature].values[0])
        for patient_id in group2:
            gtoup2_attributes.append(group_info.loc[(group_info['Pseudonym'] == patient_id), feature].values[0])

        attributes_all = set(gtoup1_attributes) | set(gtoup2_attributes)
        group1_fre = []
        group2_fre = []
        v1 = pd.value_counts(gtoup1_attributes)
        v2 = pd.value_counts(gtoup2_attributes)

        for name in attributes_all:
            group1_fre.append(v1[name])
            group2_fre.append(v2[name])

        _, p = chisquare(group1_fre/np.sum(group1_fre), f_exp=group2_fre/np.sum(group2_fre))
        print("Feature: {},  train: {}, test: {}, P value: {}".format(feature, group1_fre, group2_fre, p))


def load_temp(savefile):
    tables = pd.ExcelFile(savefile)
    table_names = tables.sheet_names
    start_num_fold = len(table_names)

    result_bestloss_train_folds = []
    result_bestloss_val_folds = []
    result_bestloss_test_folds = []
    result_bestauc_train_folds = []
    result_bestauc_val_folds = []
    result_bestauc_test_folds = []
    result_bestjw5_train_folds = []
    result_bestjw5_val_folds = []
    result_bestjw5_test_folds = []
    result_bestjw6_train_folds = []
    result_bestjw6_val_folds = []
    result_bestjw6_test_folds = []

    for fold_name in table_names:

        # [loss_train, auc_train, acc_train, sens_train, spec_train]
        tbs = pd.read_excel(savefile, sheet_name=fold_name)
        read_column = list(range(2, 8))
        # bestloss_train = tbs.iloc[tbs['Stage']=='Bestloss_Train', read_column]
        bestloss_train = tbs.iloc[0, 2:8].to_list()#tbs.iloc[1,2:]
        bestloss_val = tbs.iloc[1, 2:8].to_list()#tbs['Bestloss_Val', read_column]
        bestloss_test = tbs.iloc[2, 2:8].to_list()#tbs['Bestloss_Test', read_column]

        bestauc_train = tbs.iloc[3, 2:8].to_list()#tbs['Bestauc_Train', read_column]
        bestauc_val = tbs.iloc[4, 2:8].to_list()#tbs['Bestauc_Val', read_column]
        bestauc_test = tbs.iloc[5, 2:8].to_list()

        #tbs['Bestauc_Test', read_column]

        bestjw5_train = tbs.iloc[6, 2:8].to_list()#tbs['Bestjw5_Train', read_column]
        bestjw5_val = tbs.iloc[7, 2:8].to_list()#tbs['Bestjw5_Val', read_column]
        bestjw5_test = tbs.iloc[8, 2:8].to_list()#tbs['Bestjw5_Test', read_column]

        bestjw6_train = tbs.iloc[9, 2:8].to_list()#tbs['Bestjw6_Train', read_column]
        bestjw6_val = tbs.iloc[10, 2:8].to_list()#tbs['Bestjw6_Val', read_column]
        bestjw6_test = tbs.iloc[11, 2:8].to_list()#tbs['Bestjw6_Test', read_column]


        result_bestloss_train_folds.append([0]+bestloss_train)
        result_bestloss_val_folds.append([0]+bestloss_val)
        result_bestloss_test_folds.append([0]+bestloss_test)
        result_bestauc_train_folds.append([0]+bestauc_train)
        result_bestauc_val_folds.append([0]+bestauc_val)
        result_bestauc_test_folds.append([0]+bestauc_test)
        result_bestjw5_train_folds.append([0]+bestjw5_train)
        result_bestjw5_val_folds.append([0]+bestjw5_val)
        result_bestjw5_test_folds.append([0]+bestjw5_test)
        result_bestjw6_train_folds.append([0]+bestjw6_train)
        result_bestjw6_val_folds.append([0]+bestjw6_val)
        result_bestjw6_test_folds.append([0]+bestjw6_test)

    return start_num_fold, result_bestloss_train_folds, result_bestloss_val_folds, result_bestloss_test_folds,\
           result_bestauc_train_folds, result_bestauc_val_folds, result_bestauc_test_folds,\
           result_bestjw5_train_folds, result_bestjw5_val_folds, result_bestjw5_test_folds,\
           result_bestjw6_train_folds, result_bestjw6_val_folds, result_bestjw6_test_folds

def write_result(writer, setname, epoch, results):
    '''
    setname: train, val, test
    '''
    if 'wandb' in str(type(writer)):

        results_dict = {}
        for key, values in results.items():
            results_dict['{}/{}'.format(setname, key)] = values

        wandb.log(results_dict, step=epoch)
    else:
        reults = {}
        for key, values in results.items():
            writer.add_scalar('{}/{}'.format(setname, key), values, epoch)

        
def save_temp(result_bestloss_train_folds, result_bestloss_val_folds, result_bestloss_test_folds,
              result_bestauc_train_folds, result_bestauc_val_folds, result_bestauc_test_folds,
              result_bestjw5_train_folds, result_bestjw5_val_folds, result_bestjw5_test_folds,
              result_bestjw6_train_folds, result_bestjw6_val_folds, result_bestjw6_test_folds,
              savefile):

    with pd.ExcelWriter(savefile) as writer:

        for fold in range(len(result_bestloss_train_folds)):
            # [loss_train, auc_train, acc_train, sens_train, spec_train]
            result_temp = pd.DataFrame(columns=('Stage', 'AUC', 'ACC', 'Sens', 'Spec', 'jw0.5', 'jw0.6'))

            # bestloss
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestloss_Train', 'AUC': [result_bestloss_train_folds[fold][1]],
                 'ACC': [result_bestloss_train_folds[fold][2]],
                 'Sens': [result_bestloss_train_folds[fold][3]], 'Spec': [result_bestloss_train_folds[fold][4]],
                 'jw0.5': [0.5*result_bestloss_train_folds[fold][3]+0.5*result_bestloss_train_folds[fold][4]],
                 'jw0.6': [0.6*result_bestloss_train_folds[fold][3]+0.4*result_bestloss_train_folds[fold][4]]}), ignore_index=True)
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestloss_Val', 'AUC': [result_bestloss_val_folds[fold][1]],
                 'ACC': [result_bestloss_val_folds[fold][2]],
                 'Sens': [result_bestloss_val_folds[fold][3]], 'Spec': [result_bestloss_val_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestloss_val_folds[fold][3] + 0.5 * result_bestloss_val_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestloss_val_folds[fold][3] + 0.4 * result_bestloss_val_folds[fold][4]]}),ignore_index=True)
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestloss_Test', 'AUC': [result_bestloss_test_folds[fold][1]],
                 'ACC': [result_bestloss_test_folds[fold][2]],
                 'Sens': [result_bestloss_test_folds[fold][3]], 'Spec': [result_bestloss_test_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestloss_test_folds[fold][3] + 0.5 * result_bestloss_test_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestloss_test_folds[fold][3] + 0.4 * result_bestloss_test_folds[fold][4]]}),ignore_index=True)
            # bestauc
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestauc_Train', 'AUC': [result_bestauc_train_folds[fold][1]],
                 'ACC': [result_bestauc_train_folds[fold][2]],
                 'Sens': [result_bestauc_train_folds[fold][3]], 'Spec': [result_bestauc_train_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestauc_train_folds[fold][3] + 0.5 * result_bestauc_train_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestauc_train_folds[fold][3] + 0.4 * result_bestauc_train_folds[fold][4]]}),
                ignore_index=True)
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestauc_Val', 'AUC': [result_bestauc_val_folds[fold][1]],
                 'ACC': [result_bestauc_val_folds[fold][2]],
                 'Sens': [result_bestauc_val_folds[fold][3]], 'Spec': [result_bestauc_val_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestauc_val_folds[fold][3] + 0.5 * result_bestauc_val_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestauc_val_folds[fold][3] + 0.4 * result_bestauc_val_folds[fold][4]]}),
                ignore_index=True)
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestauc_Test', 'AUC': [result_bestauc_test_folds[fold][1]],
                 'ACC': [result_bestauc_test_folds[fold][2]],
                 'Sens': [result_bestauc_test_folds[fold][3]], 'Spec': [result_bestauc_test_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestauc_test_folds[fold][3] + 0.5 * result_bestauc_test_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestauc_test_folds[fold][3] + 0.4 * result_bestauc_test_folds[fold][4]]}),
                ignore_index=True)
            # best_jw0.5
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestjw5_Train', 'AUC': [result_bestjw5_train_folds[fold][1]],
                 'ACC': [result_bestjw5_train_folds[fold][2]],
                 'Sens': [result_bestjw5_train_folds[fold][3]], 'Spec': [result_bestjw5_train_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestjw5_train_folds[fold][3] + 0.5 * result_bestjw5_train_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestjw5_train_folds[fold][3] + 0.4 * result_bestjw5_train_folds[fold][4]]}),
                ignore_index=True)
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestjw5_Val', 'AUC': [result_bestjw5_val_folds[fold][1]],
                 'ACC': [result_bestjw5_val_folds[fold][2]],
                 'Sens': [result_bestjw5_val_folds[fold][3]], 'Spec': [result_bestjw5_val_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestjw5_val_folds[fold][3] + 0.5 * result_bestjw5_val_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestjw5_val_folds[fold][3] + 0.4 * result_bestjw5_val_folds[fold][4]]}),
                ignore_index=True)
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestjw5_Test', 'AUC': [result_bestjw5_test_folds[fold][1]],
                 'ACC': [result_bestjw5_test_folds[fold][2]],
                 'Sens': [result_bestjw5_test_folds[fold][3]], 'Spec': [result_bestjw5_test_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestjw5_test_folds[fold][3] + 0.5 * result_bestjw5_test_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestjw5_test_folds[fold][3] + 0.4 * result_bestjw5_test_folds[fold][4]]}),
                ignore_index=True)
            # best_jw0.6
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestjw6_Train', 'AUC': [result_bestjw6_train_folds[fold][1]],
                 'ACC': [result_bestjw6_train_folds[fold][2]],
                 'Sens': [result_bestjw6_train_folds[fold][3]], 'Spec': [result_bestjw6_train_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestjw6_train_folds[fold][3] + 0.5 * result_bestjw6_train_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestjw6_train_folds[fold][3] + 0.4 * result_bestjw6_train_folds[fold][4]]}),
                ignore_index=True)
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': ['Bestjw6_Val'], 'AUC': [result_bestjw6_val_folds[fold][1]],
                 'ACC': [result_bestjw6_val_folds[fold][2]],
                 'Sens': [result_bestjw6_val_folds[fold][3]], 'Spec': [result_bestjw6_val_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestjw6_val_folds[fold][3] + 0.5 * result_bestjw6_val_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestjw6_val_folds[fold][3] + 0.4 * result_bestjw6_val_folds[fold][4]]}),
                ignore_index=True)
            result_temp = result_temp.append(pd.DataFrame(
                {'Stage': 'Bestjw6_Test', 'AUC': [result_bestjw6_test_folds[fold][1]],
                 'ACC': [result_bestjw6_test_folds[fold][2]],
                 'Sens': [result_bestjw6_test_folds[fold][3]], 'Spec': [result_bestjw6_test_folds[fold][4]],
                 'jw0.5': [0.5 * result_bestjw6_test_folds[fold][3] + 0.5 * result_bestjw6_test_folds[fold][4]],
                 'jw0.6': [0.6 * result_bestjw6_test_folds[fold][3] + 0.4 * result_bestjw6_test_folds[fold][4]]}),
                ignore_index=True)

            result_temp.to_excel(writer, 'Fold'+str(fold))

            
def save_final(result_bestloss_train_folds, result_bestloss_val_folds,result_bestloss_test_folds,
               result_bestauc_train_folds, result_bestauc_val_folds, result_bestauc_test_folds,
               result_bestjw5_train_folds,  result_bestjw5_val_folds, result_bestjw5_test_folds,
               result_bestjw6_train_folds,  result_bestjw6_val_folds, result_bestjw6_test_folds, save_result):

    result_bestloss_train_folds = np.array(result_bestloss_train_folds)
    result_bestloss_val_folds = np.array(result_bestloss_val_folds)
    result_bestloss_test_folds = np.array(result_bestloss_test_folds)
    result_bestauc_train_folds = np.array(result_bestauc_train_folds)
    result_bestauc_val_folds = np.array(result_bestauc_val_folds)
    result_bestauc_test_folds = np.array(result_bestauc_test_folds)
    result_bestjw5_train_folds = np.array(result_bestjw5_train_folds)
    result_bestjw5_val_folds = np.array(result_bestjw5_val_folds)
    result_bestjw5_test_folds = np.array(result_bestjw5_test_folds)
    result_bestjw6_train_folds = np.array(result_bestjw6_train_folds)
    result_bestjw6_val_folds = np.array(result_bestjw6_val_folds)
    result_bestjw6_test_folds = np.array(result_bestjw6_test_folds)

    ##### selecting model: best loss #######
    Bestloss_Train_Auc = "{:.2f} + {:.2f}".format(result_bestloss_train_folds[:, 1].mean(), result_bestloss_train_folds[:, 1].std())
    Bestloss_Train_Acc = "{:.2f} + {:.2f}".format(result_bestloss_train_folds[:, 2].mean(), result_bestloss_train_folds[:, 2].std())
    Bestloss_Train_Sens = "{:.2f} + {:.2f}".format(result_bestloss_train_folds[:, 3].mean(), result_bestloss_train_folds[:, 3].std())
    Bestloss_Train_Spec = "{:.2f} + {:.2f}".format(result_bestloss_train_folds[:, 4].mean(), result_bestloss_train_folds[:, 4].std())
    Bestloss_Train_jw5 = "{:.2f} + {:.2f}".format(result_bestloss_train_folds[:, 5].mean(), result_bestloss_train_folds[:, 5].std())
    Bestloss_Train_jw6 = "{:.2f} + {:.2f}".format(result_bestloss_train_folds[:, 6].mean(), result_bestloss_train_folds[:, 6].std())

    Bestloss_Val_Auc = "{:.2f} + {:.2f}".format(result_bestloss_val_folds[:, 1].mean(), result_bestloss_val_folds[:, 1].std())
    Bestloss_Val_Acc = "{:.2f} + {:.2f}".format(result_bestloss_val_folds[:, 2].mean(), result_bestloss_val_folds[:, 2].std())
    Bestloss_Val_Sens = "{:.2f} + {:.2f}".format(result_bestloss_val_folds[:, 3].mean(), result_bestloss_val_folds[:, 3].std())
    Bestloss_Val_Spec = "{:.2f} + {:.2f}".format(result_bestloss_val_folds[:, 4].mean(), result_bestloss_val_folds[:, 4].std())
    Bestloss_Val_jw5 = "{:.2f} + {:.2f}".format(result_bestloss_val_folds[:, 5].mean(), result_bestloss_val_folds[:, 5].std())
    Bestloss_Val_jw6 = "{:.2f} + {:.2f}".format(result_bestloss_val_folds[:, 6].mean(), result_bestloss_val_folds[:, 6].std())

    Bestloss_Test_Auc = "{:.2f} + {:.2f}".format(result_bestloss_test_folds[:, 1].mean(), result_bestloss_test_folds[:, 1].std())
    Bestloss_Test_Acc = "{:.2f} + {:.2f}".format(result_bestloss_test_folds[:, 2].mean(), result_bestloss_test_folds[:, 2].std())
    Bestloss_Test_Sens = "{:.2f} + {:.2f}".format(result_bestloss_test_folds[:, 3].mean(), result_bestloss_test_folds[:, 3].std())
    Bestloss_Test_Spec = "{:.2f} + {:.2f}".format(result_bestloss_test_folds[:, 4].mean(), result_bestloss_test_folds[:, 4].std())
    Bestloss_Test_jw5 = "{:.2f} + {:.2f}".format(result_bestloss_test_folds[:, 5].mean(), result_bestloss_test_folds[:, 5].std())
    Bestloss_Test_jw6 = "{:.2f} + {:.2f}".format(result_bestloss_test_folds[:, 6].mean(), result_bestloss_test_folds[:, 6].std())

    Bestloss_Test_index = np.where(result_bestloss_val_folds[:, 1] == result_bestloss_val_folds[:, 1].max())[0][0]
    Bestloss_Test_Auc_bestepoch = "{:.2f}".format(result_bestloss_test_folds[Bestloss_Test_index, 1])
    Bestloss_Test_Acc_bestepoch = "{:.2f}".format(result_bestloss_test_folds[Bestloss_Test_index, 2])
    Bestloss_Test_Sens_bestepoch = "{:.2f}".format(result_bestloss_test_folds[Bestloss_Test_index, 3])
    Bestloss_Test_Spec_bestepoch = "{:.2f}".format(result_bestloss_test_folds[Bestloss_Test_index, 4])
    Bestloss_Test_jw5_bestepoch = "{:.2f}".format(result_bestloss_test_folds[Bestloss_Test_index, 5])
    Bestloss_Test_jw6_bestepoch = "{:.2f}".format(result_bestloss_test_folds[Bestloss_Test_index, 6])

    ##### selecting model: best auc #######
    Bestauc_Train_Auc = "{:.2f} + {:.2f}".format(result_bestauc_train_folds[:, 1].mean(), result_bestauc_train_folds[:, 1].std())
    Bestauc_Train_Acc = "{:.2f} + {:.2f}".format(result_bestauc_train_folds[:, 2].mean(), result_bestauc_train_folds[:, 2].std())
    Bestauc_Train_Sens = "{:.2f} + {:.2f}".format(result_bestauc_train_folds[:, 3].mean(), result_bestauc_train_folds[:, 3].std())
    Bestauc_Train_Spec = "{:.2f} + {:.2f}".format(result_bestauc_train_folds[:, 4].mean(), result_bestauc_train_folds[:, 4].std())
    Bestauc_Train_jw5 = "{:.2f} + {:.2f}".format(result_bestauc_train_folds[:, 5].mean(), result_bestauc_train_folds[:, 5].std())
    Bestauc_Train_jw6 = "{:.2f} + {:.2f}".format(result_bestauc_train_folds[:, 6].mean(), result_bestauc_train_folds[:, 6].std())

    Bestauc_Val_Auc = "{:.2f} + {:.2f}".format(result_bestauc_val_folds[:, 1].mean(), result_bestauc_val_folds[:, 1].std())
    Bestauc_Val_Acc = "{:.2f} + {:.2f}".format(result_bestauc_val_folds[:, 2].mean(), result_bestauc_val_folds[:, 2].std())
    Bestauc_Val_Sens = "{:.2f} + {:.2f}".format(result_bestauc_val_folds[:, 3].mean(), result_bestauc_val_folds[:, 3].std())
    Bestauc_Val_Spec = "{:.2f} + {:.2f}".format(result_bestauc_val_folds[:, 4].mean(), result_bestauc_val_folds[:, 4].std())
    Bestauc_Val_jw5 = "{:.2f} + {:.2f}".format(result_bestauc_val_folds[:, 5].mean(), result_bestauc_val_folds[:,5].std())
    Bestauc_Val_jw6 = "{:.2f} + {:.2f}".format(result_bestauc_val_folds[:, 6].mean(), result_bestauc_val_folds[:, 6].std())

    Bestauc_Test_Auc = "{:.2f} + {:.2f}".format(result_bestauc_test_folds[:, 1].mean(), result_bestauc_test_folds[:, 1].std())
    Bestauc_Test_Acc = "{:.2f} + {:.2f}".format(result_bestauc_test_folds[:, 2].mean(), result_bestauc_test_folds[:, 2].std())
    Bestauc_Test_Sens = "{:.2f} + {:.2f}".format(result_bestauc_test_folds[:, 3].mean(), result_bestauc_test_folds[:, 3].std())
    Bestauc_Test_Spec = "{:.2f} + {:.2f}".format(result_bestauc_test_folds[:, 4].mean(), result_bestauc_test_folds[:, 4].std())
    Bestauc_Test_jw5 = "{:.2f} + {:.2f}".format(result_bestauc_test_folds[:, 5].mean(), result_bestauc_test_folds[:, 5].std())
    Bestauc_Test_jw6 = "{:.2f} + {:.2f}".format(result_bestauc_test_folds[:, 6].mean(), result_bestauc_test_folds[:, 6].std())

    Bestauc_Test_index = np.where(result_bestauc_val_folds[:, 1] == result_bestauc_val_folds[:, 1].max())[0][0]
    Bestauc_Test_Auc_bestepoch = "{:.2f}".format(result_bestauc_test_folds[Bestauc_Test_index, 1])
    Bestauc_Test_Acc_bestepoch = "{:.2f}".format(result_bestauc_test_folds[Bestauc_Test_index, 2])
    Bestauc_Test_Sens_bestepoch = "{:.2f}".format(result_bestauc_test_folds[Bestauc_Test_index, 3])
    Bestauc_Test_Spec_bestepoch = "{:.2f}".format(result_bestauc_test_folds[Bestauc_Test_index, 4])
    Bestauc_Test_jw5_bestepoch = "{:.2f}".format(result_bestauc_test_folds[Bestauc_Test_index, 5])
    Bestauc_Test_jw6_bestepoch = "{:.2f}".format(result_bestauc_test_folds[Bestauc_Test_index, 6])

    ##### selecting model: best jw #######
    Bestjw5_Train_Auc = "{:.2f} + {:.2f}".format(result_bestjw5_train_folds[:, 1].mean(), result_bestjw5_train_folds[:, 1].std())
    Bestjw5_Train_Acc = "{:.2f} + {:.2f}".format(result_bestjw5_train_folds[:, 2].mean(), result_bestjw5_train_folds[:, 2].std())
    Bestjw5_Train_Sens = "{:.2f} + {:.2f}".format(result_bestjw5_train_folds[:, 3].mean(), result_bestjw5_train_folds[:, 3].std())
    Bestjw5_Train_Spec = "{:.2f} + {:.2f}".format(result_bestjw5_train_folds[:, 4].mean(), result_bestjw5_train_folds[:, 4].std())
    Bestjw5_Train_jw5 = "{:.2f} + {:.2f}".format(result_bestjw5_train_folds[:, 5].mean(), result_bestjw5_train_folds[:, 5].std())
    Bestjw5_Train_jw6 = "{:.2f} + {:.2f}".format(result_bestjw5_train_folds[:, 6].mean(), result_bestjw5_train_folds[:, 6].std())

    Bestjw5_Val_Auc = "{:.2f} + {:.2f}".format(result_bestjw5_val_folds[:, 1].mean(), result_bestjw5_val_folds[:, 1].std())
    Bestjw5_Val_Acc = "{:.2f} + {:.2f}".format(result_bestjw5_val_folds[:, 2].mean(), result_bestjw5_val_folds[:, 2].std())
    Bestjw5_Val_Sens = "{:.2f} + {:.2f}".format(result_bestjw5_val_folds[:, 3].mean(), result_bestjw5_val_folds[:, 3].std())
    Bestjw5_Val_Spec = "{:.2f} + {:.2f}".format(result_bestjw5_val_folds[:, 4].mean(), result_bestjw5_val_folds[:, 4].std())
    Bestjw5_Val_jw5 = "{:.2f} + {:.2f}".format(result_bestjw5_val_folds[:, 5].mean(), result_bestjw5_val_folds[:, 5].std())
    Bestjw5_Val_jw6 = "{:.2f} + {:.2f}".format(result_bestjw5_val_folds[:, 6].mean(), result_bestjw5_val_folds[:, 6].std())

    Bestjw5_Test_Auc = "{:.2f} + {:.2f}".format(result_bestjw5_test_folds[:, 1].mean(), result_bestjw5_test_folds[:, 1].std())
    Bestjw5_Test_Acc = "{:.2f} + {:.2f}".format(result_bestjw5_test_folds[:, 2].mean(), result_bestjw5_test_folds[:, 2].std())
    Bestjw5_Test_Sens = "{:.2f} + {:.2f}".format(result_bestjw5_test_folds[:, 3].mean(), result_bestjw5_test_folds[:, 3].std())
    Bestjw5_Test_Spec = "{:.2f} + {:.2f}".format(result_bestjw5_test_folds[:, 4].mean(), result_bestjw5_test_folds[:, 4].std())
    Bestjw5_Test_jw5 = "{:.2f} + {:.2f}".format(result_bestjw5_test_folds[:, 5].mean(), result_bestjw5_test_folds[:, 5].std())
    Bestjw5_Test_jw6 = "{:.2f} + {:.2f}".format(result_bestjw5_test_folds[:, 6].mean(), result_bestjw5_test_folds[:, 6].std())

    Bestjw5_Test_index = np.where(result_bestjw5_val_folds[:, 5] == result_bestjw5_val_folds[:, 5].max())[0][0]
    Bestjw5_Test_Auc_bestepoch = "{:.2f}".format(result_bestjw5_test_folds[Bestjw5_Test_index, 1])
    Bestjw5_Test_Acc_bestepoch = "{:.2f}".format(result_bestjw5_test_folds[Bestjw5_Test_index, 2])
    Bestjw5_Test_Sens_bestepoch = "{:.2f}".format(result_bestjw5_test_folds[Bestjw5_Test_index, 3])
    Bestjw5_Test_Spec_bestepoch = "{:.2f}".format(result_bestjw5_test_folds[Bestjw5_Test_index, 4])
    Bestjw5_Test_jw5_bestepoch = "{:.2f}".format(result_bestjw5_test_folds[Bestjw5_Test_index, 5])
    Bestjw5_Test_jw6_bestepoch = "{:.2f}".format(result_bestjw5_test_folds[Bestjw5_Test_index, 6])

    Bestjw6_Train_Auc = "{:.2f} + {:.2f}".format(result_bestjw6_train_folds[:, 1].mean(), result_bestjw6_train_folds[:, 1].std())
    Bestjw6_Train_Acc = "{:.2f} + {:.2f}".format(result_bestjw6_train_folds[:, 2].mean(), result_bestjw6_train_folds[:, 2].std())
    Bestjw6_Train_Sens = "{:.2f} + {:.2f}".format(result_bestjw6_train_folds[:, 3].mean(), result_bestjw6_train_folds[:, 3].std())
    Bestjw6_Train_Spec = "{:.2f} + {:.2f}".format(result_bestjw6_train_folds[:, 4].mean(), result_bestjw6_train_folds[:, 4].std())
    Bestjw6_Train_jw5 = "{:.2f} + {:.2f}".format(result_bestjw6_train_folds[:, 5].mean(), result_bestjw6_train_folds[:, 5].std())
    Bestjw6_Train_jw6 = "{:.2f} + {:.2f}".format(result_bestjw6_train_folds[:, 6].mean(), result_bestjw6_train_folds[:, 6].std())

    Bestjw6_Val_Auc = "{:.2f} + {:.2f}".format(result_bestjw6_val_folds[:, 1].mean(), result_bestjw6_val_folds[:, 1].std())
    Bestjw6_Val_Acc = "{:.2f} + {:.2f}".format(result_bestjw6_val_folds[:, 2].mean(), result_bestjw6_val_folds[:, 2].std())
    Bestjw6_Val_Sens = "{:.2f} + {:.2f}".format(result_bestjw6_val_folds[:, 3].mean(), result_bestjw6_val_folds[:, 3].std())
    Bestjw6_Val_Spec = "{:.2f} + {:.2f}".format(result_bestjw6_val_folds[:, 4].mean(), result_bestjw6_val_folds[:, 4].std())
    Bestjw6_Val_jw5 = "{:.2f} + {:.2f}".format(result_bestjw6_val_folds[:, 3].mean(), result_bestjw6_val_folds[:, 5].std())
    Bestjw6_Val_jw6 = "{:.2f} + {:.2f}".format(result_bestjw6_val_folds[:, 4].mean(), result_bestjw6_val_folds[:, 6].std())

    Bestjw6_Test_Auc = "{:.2f} + {:.2f}".format(result_bestjw6_test_folds[:, 1].mean(), result_bestjw6_test_folds[:, 1].std())
    Bestjw6_Test_Acc = "{:.2f} + {:.2f}".format(result_bestjw6_test_folds[:, 2].mean(), result_bestjw6_test_folds[:, 2].std())
    Bestjw6_Test_Sens = "{:.2f} + {:.2f}".format(result_bestjw6_test_folds[:, 3].mean(), result_bestjw6_test_folds[:, 3].std())
    Bestjw6_Test_Spec = "{:.2f} + {:.2f}".format(result_bestjw6_test_folds[:, 4].mean(), result_bestjw6_test_folds[:, 4].std())
    Bestjw6_Test_jw5 = "{:.2f} + {:.2f}".format(result_bestjw6_test_folds[:, 5].mean(), result_bestjw6_test_folds[:, 5].std())
    Bestjw6_Test_jw6 = "{:.2f} + {:.2f}".format(result_bestjw6_test_folds[:, 6].mean(), result_bestjw6_test_folds[:, 6].std())

    Bestjw6_Test_index = np.where(result_bestjw6_val_folds[:, 6] == result_bestjw6_val_folds[:, 6].max())[0][0]
    Bestjw6_Test_Auc_bestepoch = "{:.2f}".format(result_bestjw6_test_folds[Bestjw6_Test_index, 1])
    Bestjw6_Test_Acc_bestepoch = "{:.2f}".format(result_bestjw6_test_folds[Bestjw6_Test_index, 2])
    Bestjw6_Test_Sens_bestepoch = "{:.2f}".format(result_bestjw6_test_folds[Bestjw6_Test_index, 3])
    Bestjw6_Test_Spec_bestepoch = "{:.2f}".format(result_bestjw6_test_folds[Bestjw6_Test_index, 4])
    Bestjw6_Test_jw5_bestepoch = "{:.2f}".format(result_bestjw6_test_folds[Bestjw6_Test_index, 5])
    Bestjw6_Test_jw6_bestepoch = "{:.2f}".format(result_bestjw6_test_folds[Bestjw6_Test_index, 6])

    ## print
    print("Model selected by the best auc")
    print("Fold 1-5-Bestauc (Train)\t  AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestauc_Train_Auc,
                                                                                          Bestauc_Train_Acc,
                                                                                          Bestauc_Train_Sens,
                                                                                          Bestauc_Train_Spec,Bestauc_Train_jw5,Bestauc_Train_jw6))
    print("Fold 1-5-Bestauc (Val)\t\t  AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestauc_Val_Auc, Bestauc_Val_Acc,
                                                                                          Bestauc_Val_Sens,
                                                                                          Bestauc_Val_Spec, Bestauc_Val_jw5, Bestauc_Val_jw6))
    print("Fold 1-5-Bestauc (Test)\t\t AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestauc_Test_Auc, Bestauc_Test_Acc,
                                                                                        Bestauc_Test_Sens,
                                                                                        Bestauc_Test_Spec, Bestauc_Test_jw5, Bestauc_Test_jw6))
    print("Fold 1-5-Bestauc (Test*)\t\t AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestauc_Test_Auc_bestepoch, Bestauc_Test_Acc_bestepoch,
                                                                                        Bestauc_Test_Sens_bestepoch,
                                                                                        Bestauc_Test_Spec_bestepoch, Bestauc_Test_jw5_bestepoch, Bestauc_Test_jw6_bestepoch))

    print("Model selected by the best loss")
    print("Fold 1-5-Bestloss (Train)\t  AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestloss_Train_Auc,
                                                                                           Bestloss_Train_Acc,
                                                                                           Bestloss_Train_Sens,
                                                                                           Bestloss_Train_Spec, Bestloss_Train_jw5, Bestauc_Test_jw6))
    print("Fold 1-5-Bestloss (Val)\t\t  AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestloss_Val_Auc, Bestloss_Val_Acc,
                                                                                         Bestloss_Val_Sens,
                                                                                         Bestloss_Val_Spec, Bestloss_Val_jw5, Bestloss_Val_jw6))
    print("Fold 1-5-Bestloss (Test)\t\t AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestloss_Test_Auc,
                                                                                           Bestloss_Test_Acc,
                                                                                           Bestloss_Test_Sens,
                                                                                           Bestloss_Test_Spec, Bestloss_Test_jw5, Bestloss_Test_jw6))
    print("Fold 1-5-Bestloss (Test*)\t\t AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestloss_Test_Auc_bestepoch,
                                                                                           Bestloss_Test_Acc_bestepoch,
                                                                                           Bestloss_Test_Sens_bestepoch,
                                                                                           Bestloss_Test_Spec_bestepoch, Bestloss_Test_jw5_bestepoch, Bestloss_Test_jw6_bestepoch))

    print("Model selected by the best jw0.5")
    print("Fold 1-5-Bestjw5 (Train)\t  AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestjw5_Train_Auc,
                                                                                           Bestjw5_Train_Acc,
                                                                                           Bestjw5_Train_Sens,
                                                                                           Bestjw5_Train_Spec,Bestjw5_Train_jw5,
                                                                                           Bestjw5_Train_jw6))
    print("Fold 1-5-Bestjw5 (Val)\t\t  AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestjw5_Val_Auc, Bestjw5_Val_Acc,
                                                                                         Bestjw5_Val_Sens,
                                                                                         Bestjw5_Val_Spec,Bestjw5_Val_jw5,
                                                                                         Bestjw5_Val_jw6))
    print("Fold 1-5-Bestjw5 (Test*)\t\t AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestjw5_Test_Auc_bestepoch,
                                                                                           Bestjw5_Test_Acc_bestepoch,
                                                                                           Bestjw5_Test_Sens_bestepoch,
                                                                                           Bestjw5_Test_Spec_bestepoch,Bestjw5_Test_jw5_bestepoch,
                                                                                           Bestjw5_Test_jw6_bestepoch))

    print("Model selected by the best jw0.6")
    print("Fold 1-5-Bestjw6 (Train)\t  AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestjw6_Train_Auc,
                                                                                           Bestjw6_Train_Acc,
                                                                                           Bestjw6_Train_Sens,
                                                                                           Bestjw6_Train_Spec,Bestjw6_Train_jw5,
                                                                                           Bestjw6_Train_jw6))
    print("Fold 1-5-Bestjw6 (Val)\t\t  AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestjw6_Val_Auc, Bestjw6_Val_Acc,
                                                                                         Bestjw6_Val_Sens,
                                                                                         Bestjw6_Val_Spec,Bestjw6_Val_jw5,
                                                                                         Bestjw6_Val_jw6))
    print("Fold 1-5-Bestjw6 (Test)\t\t AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestjw6_Test_Auc,
                                                                                           Bestjw6_Test_Acc,
                                                                                           Bestjw6_Test_Sens,
                                                                                           Bestjw6_Test_Spec,Bestjw6_Test_jw5,
                                                                                           Bestjw6_Test_jw6))
    print("Fold 1-5-Bestjw6 (Test*)\t\t AUC: {},\t ACC: {},\t Sens: {},\t Spec: {},\t jw5: {},\t jw6:{}".format(Bestjw6_Test_Auc_bestepoch,
                                                                                           Bestjw6_Test_Acc_bestepoch,
                                                                                           Bestjw6_Test_Sens_bestepoch,
                                                                                           Bestjw6_Test_Spec_bestepoch,Bestjw6_Test_jw5_bestepoch,
                                                                                           Bestjw6_Test_jw6_bestepoch))

    ## save as csv
    result = pd.DataFrame(columns=('Stage', 'AUC', 'ACC', 'Sens', 'Spec'))

    result = result.append(pd.DataFrame(
        {'Stage': 'Bestloss_Train', 'AUC': [Bestloss_Train_Auc], 'ACC': [Bestloss_Train_Acc], 'Sens': [Bestloss_Train_Sens], 'Spec': [Bestloss_Train_Spec], 'JW5': [Bestloss_Train_jw5], 'JW6': [Bestloss_Train_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestloss_Val', 'AUC': [Bestloss_Val_Auc], 'ACC': [Bestloss_Val_Acc], 'Sens': [Bestloss_Val_Sens], 'Spec': [Bestloss_Val_Spec], 'JW5': [Bestloss_Val_jw5], 'JW6': [Bestloss_Val_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestloss_Test', 'AUC': [Bestloss_Test_Auc], 'ACC': [Bestloss_Test_Acc], 'Sens': [Bestloss_Test_Sens], 'Spec': [Bestloss_Test_Spec], 'JW5': [Bestloss_Test_jw5], 'JW6': [Bestloss_Test_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestloss_Test_unique', 'AUC': [Bestloss_Test_Auc_bestepoch], 'ACC': [Bestloss_Test_Acc_bestepoch], 'Sens': [Bestloss_Test_Sens_bestepoch], 'Spec': [Bestloss_Test_Spec_bestepoch], 'JW5': [Bestloss_Test_jw5_bestepoch], 'JW6': [Bestloss_Test_jw6_bestepoch]}), ignore_index=True)


    result = result.append(pd.DataFrame(
        {'Stage': 'Bestauc_Train', 'AUC': [Bestauc_Train_Auc], 'ACC': [Bestauc_Train_Acc], 'Sens': [Bestauc_Train_Sens], 'Spec': [Bestauc_Train_Spec], 'JW5': [Bestauc_Train_jw5], 'JW6': [Bestauc_Train_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestauc_Val', 'AUC': [Bestauc_Val_Auc], 'ACC': [Bestauc_Val_Acc], 'Sens': [Bestauc_Val_Sens], 'Spec': [Bestauc_Val_Spec], 'JW5': [Bestauc_Val_jw5], 'JW6': [Bestauc_Val_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestauc_Test', 'AUC': [Bestauc_Test_Auc], 'ACC': [Bestauc_Test_Acc], 'Sens': [Bestauc_Test_Sens], 'Spec': [Bestauc_Test_Spec], 'JW5': [Bestauc_Test_jw5], 'JW6': [Bestauc_Test_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestauc_Test_unique', 'AUC': [Bestauc_Test_Auc_bestepoch], 'ACC': [Bestauc_Test_Acc_bestepoch], 'Sens': [Bestauc_Test_Sens_bestepoch], 'Spec': [Bestauc_Test_Spec_bestepoch], 'JW5': [Bestauc_Test_jw5_bestepoch], 'JW6': [Bestauc_Test_jw6_bestepoch]}), ignore_index=True)

    result = result.append(pd.DataFrame(
        {'Stage': 'Bestjw5_Train', 'AUC': [Bestjw5_Train_Auc], 'ACC': [Bestjw5_Train_Acc], 'Sens': [Bestjw5_Train_Sens], 'Spec': [Bestjw5_Train_Spec], 'JW5': [Bestjw5_Train_jw5], 'JW6': [Bestjw5_Train_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestjw5_Val', 'AUC': [Bestjw5_Val_Auc], 'ACC': [Bestjw5_Val_Acc], 'Sens': [Bestjw5_Val_Sens], 'Spec': [Bestjw5_Val_Spec], 'JW5': [Bestjw5_Val_jw5], 'JW6': [Bestjw5_Val_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestjw5_Test', 'AUC': [Bestjw5_Test_Auc], 'ACC': [Bestjw5_Test_Acc], 'Sens': [Bestjw5_Test_Sens], 'Spec': [Bestjw5_Test_Spec], 'JW5': [Bestjw5_Test_jw5], 'JW6': [Bestjw5_Test_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestjw5_Test_unique', 'AUC': [Bestjw5_Test_Auc_bestepoch], 'ACC': [Bestjw5_Test_Acc_bestepoch], 'Sens': [Bestjw5_Test_Sens_bestepoch], 'Spec': [Bestjw5_Test_Spec_bestepoch],'JW5': [Bestjw5_Test_jw5_bestepoch], 'JW6': [Bestjw5_Test_jw6_bestepoch]}), ignore_index=True)

    result = result.append(pd.DataFrame(
        {'Stage': 'Bestjw6_Train', 'AUC': [Bestjw6_Train_Auc], 'ACC': [Bestjw6_Train_Acc], 'Sens': [Bestjw6_Train_Sens], 'Spec': [Bestjw6_Train_Spec], 'JW5': [Bestjw6_Train_jw5], 'JW6': [Bestjw6_Train_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestjw6_Val', 'AUC': [Bestjw6_Val_Auc], 'ACC': [Bestjw6_Val_Acc], 'Sens': [Bestjw6_Val_Sens], 'Spec': [Bestjw6_Val_Spec], 'JW5': [Bestjw6_Val_jw5], 'JW6': [Bestjw6_Val_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestjw6_Test', 'AUC': [Bestjw6_Test_Auc], 'ACC': [Bestjw6_Test_Acc], 'Sens': [Bestjw6_Test_Sens], 'Spec': [Bestjw6_Test_Spec], 'JW5': [Bestjw6_Test_jw5], 'JW6': [Bestjw6_Test_jw6]}), ignore_index=True)
    result = result.append(pd.DataFrame(
        {'Stage': 'Bestjw6_Test_unique', 'AUC': [Bestjw6_Test_Auc_bestepoch], 'ACC': [Bestjw6_Test_Acc_bestepoch], 'Sens': [Bestjw6_Test_Sens_bestepoch], 'Spec': [Bestjw6_Test_Spec_bestepoch], 'JW5': [Bestjw6_Test_jw5_bestepoch], 'JW6': [Bestjw6_Test_jw6_bestepoch]}), ignore_index=True)
    result.to_csv(save_result)


def result_to_excel(result_path, setname, epoch, results):
    '''
    setname: train, val, test
    '''
    for key, values in results.items():
        writer.add_scalar('{}/{}'.format(setname, key), values, epoch)


def trainer_fuse(opt, dataset_loader, model, optimizer, writer, iteration, epoch, train=True):
    iters = 0

    label_list = []
    pred_list_fuse_ob = []
    pred_list_fuse_re = []
    pred_list_axial = []
    pred_list_cli_cro = []
    score_list_fuse_ob = []
    score_list_fuse_re = []
    score_list_axial = []
    score_list_cli_cro = []

    total_loss = 0
    total_loss_IB = 0
    total_loss_MI = 0
    total_loss_fuse_ob = 0
    total_loss_fuse_re = 0
    total_loss_axial = 0
    total_loss_cli_cro = 0
    total_loss_sparse = 0
    total_loss_conf = 0
    total_loss_similarity = 0

    lf_focal = FocalLoss(class_num=2, alpha=opt.alpha, gamma=opt.gamma, size_average=True).cuda()
    lf_ce = nn.CrossEntropyLoss().cuda()
    lf_MI = SinkhornDistance().cuda()
    lf_IB = Loss_IB_only(opt).cuda()
    cossim = nn.CosineSimilarity().cuda()

    loss_fun = lf_focal

    for image_axial, image_coronal, clinical, labels, img_name, img_path in dataset_loader:
        image_axial, image_coronal, labels, clinical = image_axial.to(opt.device), image_coronal.to(opt.device), labels.to(opt.device), clinical.to(opt.device)
        optimizer.zero_grad()

        outputs = model(image_axial, image_coronal, clinical)

        output_fuse_ob = outputs[0][0]
        output_fuse_re = outputs[0][1]
        output_axial   = outputs[1][0]
        output_cli_cro = outputs[1][1]  # or coronal
        pred_fuse_ob = F.softmax(output_fuse_ob, dim=1).reshape(output_fuse_ob.size()[0], -1)
        pred_fuse_re = F.softmax(output_fuse_re, dim=1).reshape(output_fuse_re.size()[0], -1)
        pred_axial = F.softmax(output_axial, dim=1).reshape(output_axial.size()[0], -1)
        pred_cli_cro = F.softmax(output_cli_cro, dim=1).reshape(output_cli_cro.size()[0], -1)

        score_fuse_ob = pred_fuse_ob.data[:, 1]
        score_fuse_re  = pred_fuse_re.data[:, 1]
        score_axial    = pred_axial.data[:, 1]
        score_cli_cro  = pred_cli_cro.data[:, 1]
        _, predictions_fuse_ob = torch.max(output_fuse_ob, dim=1)
        _, predictions_fuse_re = torch.max(output_fuse_re, dim=1)
        _, predictions_axial = torch.max(output_axial, dim=1)
        _, predictions_cli_cro = torch.max(output_cli_cro, dim=1)

        label_list.extend(labels.tolist())
        pred_list_fuse_ob.extend(predictions_fuse_ob.tolist())
        pred_list_fuse_re.extend(predictions_fuse_re.tolist())
        pred_list_axial.extend(predictions_axial.tolist())
        pred_list_cli_cro.extend(predictions_cli_cro.tolist())
        score_list_fuse_ob.extend(score_fuse_ob.tolist())
        score_list_fuse_re.extend(score_fuse_re.tolist())
        score_list_axial.extend(score_axial.tolist())
        score_list_cli_cro.extend(score_cli_cro.tolist())
        
        if not train and opt.test_xai:
            torch.cuda.empty_cache()
            from xai.method_gradcam import grad_cam_2d_comIBned

            label = str(labels[0].cpu().numpy())
            score_fuse_1 = str(np.round(score_fuse_ob[0].cpu().numpy(),2))
            score_fuse_0 = str(np.round(1-score_fuse_ob[0].cpu().numpy(), 2))
            score_axial_1 = str(np.round(score_axial[0].cpu().numpy(),2))
            score_axial_0 = str(np.round(1-score_axial[0].cpu().numpy(),2))
            score_clinical_1 = str(np.round(score_cli_cro[0].cpu().numpy(), 2))

            fold1 = str(labels[0].detach().cpu().numpy()) + str(pred_list_fuse_ob[0])
            fold2 = str(predictions_axial[0].cpu().numpy())
            fold3 = str(predictions_cli_cro[0].cpu().numpy())

            savepath = os.path.join(opt.path_pretrain + opt.pretrain_fused_model, 'xai', 'fuse'+fold1, 'axial'+fold2, 'cli_cro'+fold3)
            os.makedirs(savepath, exist_ok=True)
            savename = label + '_fuse' + score_fuse_1 + '_axial' + score_axial_1 + '_clinical' + score_clinical_1

            opt.perform_vis_feature = True
            opt.perform_vis_clinical = False
            opt.perform_vis_gradcam = False
            vis_feature, vis_clinical = grad_cam_2d_comIBned(model, model.image_axial.features.denseblock4.denselayer16.conv2, img_path[0], image_axial, image_coronal, clinical, 0, save_path=savepath+'/'+savename + '_' + img_name[0], save_title=savename, score0=score_fuse_1, score1=score_fuse_0, score2=score_axial_1, score3=score_axial_0, perform_vis_feature=opt.perform_vis_feature, perform_vis_clinical=opt.perform_vis_clinical, perform_vis_gradcam=opt.perform_vis_gradcam)

            vis_feature_clinical.append(vis_feature[0][0][0].detach().cpu().numpy())
            vis_feature_axial.append(vis_feature[1][0][0].detach().cpu().numpy())
            vis_feature_fuse_ob.append(vis_feature[2][0][0].detach().cpu().numpy())
            vis_feature_fuse_re.append(vis_feature[3][0][0].detach().cpu().numpy())

        loss_fuse_ob = loss_fun(output_fuse_ob, labels) * opt.weight_ob
        loss_fuse_re = loss_fun(output_fuse_re, labels) * opt.weight_re
        loss_axial   = loss_fun(output_axial, labels)   * opt.weight_axial
        loss_cli_cro = loss_fun(output_cli_cro, labels) * opt.weight_cli_cro
        
        loss_MI = 0
        loss_IB = 0
        loss_sparse = 0
        loss_conf = 0
        loss_similarity = 0
        
        if opt.use_only_clinical or opt.use_only_ct_coronal:
            loss = loss_cli_cro
        elif opt.use_only_ct_axial:
            loss = loss_axial
        else:
            if opt.use_MI:
                loss_MI_value = lf_MI(output_axial, output_cli_cro)
                loss_MI = opt.weight_MI * loss_MI_value
            if opt.use_CONF:
                conf_axial = outputs[2][0]
                conf_cli_cro = outputs[2][1]
                loss_conf = opt.weight_CONF * loss_CONF(pred_axial, pred_cli_cro, conf_axial, conf_cli_cro, labels)
           
            if opt.use_SPARSE:
                sparsity_layer = outputs[3]
                loss_sparse = opt.weight_SPARSE * torch.mean(sparsity_layer)
            
            if opt.use_similarity:
                loss_similarity = torch.mean(1-cossim(output_axial, output_cli_cro))
            
            if opt.use_IB:
                loss_IB = opt.weight_IB * lf_IB(outputs, labels)
            else:
                loss_fuse_ob = 0
            
            loss = loss_IB + loss_MI + loss_fuse_ob + loss_fuse_re + loss_axial + loss_cli_cro + loss_conf + loss_sparse + loss_similarity

        if train:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        total_loss_IB += loss_IB
        total_loss_MI += loss_MI
        total_loss_fuse_ob += loss_fuse_ob.item()
        total_loss_fuse_re += loss_fuse_re.item()
        total_loss_axial   += loss_axial.item()
        total_loss_cli_cro += loss_cli_cro.item()
        total_loss_sparse += loss_sparse
        total_loss_conf  += loss_conf
        total_loss_similarity += loss_similarity

        if iters % 50 == 0:
            print('Epoch: {} step: {}/{} loss: {}'.format(epoch, iters, len(dataset_loader), loss.item()))
        iters += 1

    try:
        auc_fuse_ob = roc_auc_score(label_list, score_list_fuse_ob)
    except:
        auc_fuse_ob = 0
    acc_fuse_ob = accuracy_score(label_list, pred_list_fuse_ob)
    cfm_fuse_ob = confusion_matrix(label_list, pred_list_fuse_ob, labels=[0,1])
    spec_fuse_ob = cfm_fuse_ob[0][0] / np.sum(cfm_fuse_ob[0])
    sens_fuse_ob = cfm_fuse_ob[1][1] / np.sum(cfm_fuse_ob[1])
    if np.isnan(spec_fuse_ob):
        spec_fuse_ob = 0
    if np.isnan(sens_fuse_ob):
        sens_fuse_ob = 0

    try:
        auc_fuse_re = roc_auc_score(label_list, score_list_fuse_re)
    except:
        auc_fuse_re = 0
    acc_fuse_re = accuracy_score(label_list, pred_list_fuse_re)
    cfm_fuse_re = confusion_matrix(label_list, pred_list_fuse_re, labels=[0, 1])
    spec_fuse_re = cfm_fuse_re[0][0] / np.sum(cfm_fuse_re[0])
    sens_fuse_re = cfm_fuse_re[1][1] / np.sum(cfm_fuse_re[1])
    if np.isnan(spec_fuse_re):
        spec_fuse_re = 0
    if np.isnan(sens_fuse_re):
        sens_fuse_re = 0

    try:
        auc_axial = roc_auc_score(label_list, score_list_axial)
    except:
        auc_axial = 0
    acc_axial = accuracy_score(label_list, pred_list_axial)
    cfm_axial = confusion_matrix(label_list, pred_list_axial, labels=[0, 1])
    spec_axial = cfm_axial[0][0] / np.sum(cfm_axial[0])
    sens_axial = cfm_axial[1][1] / np.sum(cfm_axial[1])
    if np.isnan(spec_axial):
        spec_axial = 0
    if np.isnan(sens_axial):
        sens_axial = 0

    try:
        auc_cli_cro = roc_auc_score(label_list, score_list_cli_cro)
    except:
        auc_cli_cro = 0
    acc_cli_cro = accuracy_score(label_list, pred_list_cli_cro)
    cfm_cli_cro = confusion_matrix(label_list, pred_list_cli_cro, labels=[0, 1])
    spec_cli_cro = cfm_cli_cro[0][0] / np.sum(cfm_cli_cro[0])
    sens_cli_cro = cfm_cli_cro[1][1] / np.sum(cfm_cli_cro[1])
    if np.isnan(spec_cli_cro):
        spec_cli_cro = 0
    if np.isnan(sens_cli_cro):
        sens_cli_cro = 0
    
    if not train and opt.test_xai:
        if opt.perform_vis_clinical:
            vis_clinical_fromclinical = np.array(vis_clinical_fromclinical)
            vis_clinical_fromfuse = np.array(vis_clinical_fromfuse)
            import seaborn as sns
            sns.heatmap(vis_clinical_fromfuse)
            plt.show()
            print("here")

        if opt.perform_vis_feature:
            vis_feature_clinical = np.array(vis_feature_clinical)
            vis_feature_axial = np.array(vis_feature_axial)
            vis_feature_fuse_ob  = np.array(vis_feature_fuse_ob)
            vis_feature_fuse_re = np.array(vis_feature_fuse_re)
            label_list_array = np.array(label_list)
            from sklearn.manifold import TSNE
            import seaborn as sns
            X_embedded = TSNE(n_components=2, learning_rate='auto', init = 'random', perplexity = 3).fit_transform(vis_feature_clinical,label_list_array)
            # X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit(vis_feature_clinical, label_list_array)
            X_tsne_data = np.vstack((X_embedded.T, label_list_array)).T
            df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class'])
            df_tsne.head()
            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2')
            plt.show()
            print(X_embedded.shape)
            print("here")

            X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(vis_feature_axial,label_list_array)
            X_tsne_data = np.vstack((X_embedded.T, label_list_array)).T
            df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class'])
            df_tsne.head()
            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2')
            plt.show()
            print(X_embedded.shape)
            print("here")

            X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(vis_feature_fuse_ob,label_list_array)
            X_tsne_data = np.vstack((X_embedded.T, label_list_array)).T
            df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class'])
            df_tsne.head()
            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2')
            plt.show()
            print(X_embedded.shape)
            print("here")

            X_embedded = TSNE(n_components=2, learning_rate='auto', init='random', perplexity=3).fit_transform(vis_feature_fuse_re, label_list_array)
            X_tsne_data = np.vstack((X_embedded.T, label_list_array)).T
            df_tsne = pd.DataFrame(X_tsne_data, columns=['Dim1', 'Dim2', 'class'])
            df_tsne.head()
            plt.figure(figsize=(8, 8))
            sns.scatterplot(data=df_tsne, hue='class', x='Dim1', y='Dim2')
            plt.show()
            print(X_embedded.shape)
            print("here")
    
    results_dict = {'loss_total': total_loss/len(dataset_loader), 
                    'loss_MI': total_loss_MI / len(dataset_loader), 
                    'loss_IB': total_loss_IB / len(dataset_loader), 
                    'loss_CONF': total_loss_conf/len(dataset_loader), 
                    'loss_SPARSE': total_loss_sparse/len(dataset_loader), 
                    'loss_similarity': total_loss_similarity/len(dataset_loader),
                    'loss_ob': total_loss_fuse_ob/len(dataset_loader), 
                    'auc_ob': auc_fuse_ob, 
                    'acc_ob': acc_fuse_ob,
                    'sens_ob': sens_fuse_ob,
                    'spec_ob': spec_fuse_ob,
                    'jw5_ob': sens_fuse_ob * 0.5 + spec_fuse_ob * (1 - 0.5),
                    'jw6_ob': sens_fuse_ob * 0.6 + spec_fuse_ob * (1 - 0.6),
                    'loss_re': total_loss_fuse_re/len(dataset_loader), 
                    'auc_re': auc_fuse_re,
                    'acc_re': acc_fuse_re, 
                    'sens_re': sens_fuse_re,
                    'spec_re': spec_fuse_re,
                    'jw5_re': sens_fuse_re * 0.5 + spec_fuse_re * (1 - 0.5),
                    'jw6_re': sens_fuse_re * 0.6 + spec_fuse_re * (1 - 0.6),
                    'loss_axial': total_loss_axial / len(dataset_loader),
                    'auc_axial': auc_axial,
                    'acc_axial': acc_axial,
                    'sens_axial': sens_axial,
                    'spec_axial': spec_axial,
                    'jw5_axial': sens_axial * 0.5 + spec_axial * (1 - 0.5),
                    'jw6_axial': sens_axial * 0.6 + spec_axial * (1 - 0.6),
                    'loss_cli_cro': total_loss_cli_cro/len(dataset_loader),
                    'auc_cli_cro': auc_cli_cro,
                    'acc_cli_cro': acc_cli_cro,
                    'sens_cli_cro': sens_cli_cro,
                    'spec_cli_cro': spec_cli_cro,
                    'jw5_cli_cro': sens_cli_cro * 0.5 + spec_cli_cro * (1 - 0.5), 
                    'jw6_cli_cro': sens_cli_cro * 0.6 + spec_cli_cro * (1 - 0.6)
                    }
    
    return results_dict
