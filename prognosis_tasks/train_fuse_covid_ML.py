import os
import numpy as np
import pandas as pd
import sys

sys.path.append("..")
# from utils import quality_check
from select_dataloader import *
from select_model import *
from select_optimizer import *
from utils import *
from sklearn.model_selection import train_test_split, StratifiedKFold

from select_parameters import covid_mortality_parameter, write_parameter
from torch.utils.data import DataLoader
from utils.trainer import *
import wandb
from tensorboardX import SummaryWriter


def get_roc(pos_prob, y_true):
    pos = y_true[y_true == 1]
    neg = y_true[y_true == 0]
    threshold = np.sort(pos_prob)[::-1]
    y = y_true[pos_prob.argsort()[::-1]]
    tpr_all = [0];
    fpr_all = [0]
    tpr = 0;
    fpr = 0
    x_step = 1 / float(len(neg))
    y_step = 1 / float(len(pos))
    y_sum = 0
    for i in range(len(threshold)):
        if y[i] == 1:
            tpr += y_step
            tpr_all.append(tpr)
            fpr_all.append(fpr)
        else:
            fpr += x_step
            fpr_all.append(fpr)
            tpr_all.append(tpr)
            y_sum += tpr
    return tpr_all, fpr_all, y_sum * x_step
# setup_seed(20)

opt = covid_mortality_parameter()

if not opt.clinical_category == '':
    opt.clinical_category = [str(item) for item in opt.clinical_category.split(',')]
else:
    opt.clinical_category = []

if not opt.clinical_continuous == '':
    opt.clinical_continuous = [str(item) for item in opt.clinical_continuous.split(',')]
else:
    opt.clinical_continuous = []

opt.len_clinical = len(opt.clinical_category) + len(opt.clinical_continuous)
print("clinical data: {}".format(opt.len_clinical))
os.environ['CUDA_VISIBLE_DEVICES'] = opt.gpu_num

save_location = './covid_mortality_experiments/'
save_logdir = save_location + opt.expname + '/logdir/'  # fold1/2/3/4/5

save_model = save_location + opt.expname + '/model/'  # fold1/2/3/4/5, best_auc, best_loss
save_param = save_location + opt.expname + '/para.txt'
save_result_ob = save_location + opt.expname + '/result_ob.csv'
save_result_temp_ob = save_location + opt.expname + '/result_temp_ob.xlsx'
save_result_re = save_location + opt.expname + '/result_re.csv'
save_result_temp_re = save_location + opt.expname + '/result_temp_re.xlsx'

os.makedirs(save_logdir, exist_ok=True)
os.makedirs(save_model, exist_ok=True)
# writer = wandb.init(project='proposed_fusion', name=opt.expname, config=opt, dir=save_logdir, entity="crossmodal-fusion", reinit=True)
write_parameter(opt, save_param)

patient_died_ct = pd.read_csv(opt.patient_died_ct_csv)
patient_survived_ct = pd.read_csv(opt.patient_survived_ct_csv)
patient_info = pd.read_csv(opt.patients_info_csv)
patient_died_ct['ID'] = patient_died_ct['ID'].astype(str)
patient_survived_ct['ID'] = patient_survived_ct['ID'].astype(str)
patient_info['ID'] = patient_info['ID'].astype(str)

# patient_died = list(set(patient_died_ct['Pseudonym']))
# patient_survived = list(set(patient_survived_ct['Pseudonym']))
patient_died = list(patient_died_ct['ID'])
patient_survived = list(patient_survived_ct['ID'])
print(len(patient_died), len(patient_survived))

'''
Split the train and test by 3:1
'''
x_train_died, x_test_died, y_train_died, y_test_died = train_test_split(patient_died,
                                                                        np.ones(len(patient_died)).tolist(),
                                                                        test_size=0.25, random_state=20)
x_train_survived, x_test_survived, y_train_survived, y_test_survived = train_test_split(patient_survived, np.ones(
    len(patient_survived)).tolist(), test_size=0.25, random_state=20)
x_train = x_train_died + x_train_survived
y_train = np.ones(len(x_train_died)).tolist() + np.zeros(len(x_train_survived)).tolist()
x_test = x_test_died + x_test_survived
y_test = np.ones(len(x_test_died)).tolist() + np.zeros(len(x_test_survived)).tolist()

'''
Perform statistics on train and test: P value should be big
'''
if opt.perform_statistics:
    x_train_unique = [patient_id.split('-')[0] for patient_id in x_train]
    x_test_unique = [patient_id.split('-')[0] for patient_id in x_test]
    patient_info_unique = patient_info[patient_info['Pseudonym'].isin(x_train_unique + x_test_unique)]

    plot_statistic_continuous(x_train, x_test, pd.concat([patient_survived_ct, patient_died_ct]), ['DiffFromLast'],
                              plot_hist=True)
    plot_statistic_continuous(x_train_unique, x_test_unique, patient_info_unique, ['Age'])
    plot_statistic_category(x_train_unique, x_test_unique, patient_info_unique, ['Sex'])

'''
Split the train dataset by 10-fold: select the model with largest AUC on the valiadation for the test set .
'''

skf = StratifiedKFold(n_splits=5, random_state=20, shuffle=True)
# skf = StratifiedKFold(n_splits=5, shuffle=True)
Y_train = np.array(y_train)
X_train = np.array(x_train)

result_bestloss_train_folds_ob = []
result_bestloss_val_folds_ob = []
result_bestloss_test_folds_ob = []
result_bestauc_train_folds_ob = []
result_bestauc_val_folds_ob = []
result_bestauc_test_folds_ob = []
result_bestjw5_train_folds_ob = []
result_bestjw5_val_folds_ob = []
result_bestjw5_test_folds_ob = []
result_bestjw6_train_folds_ob = []
result_bestjw6_val_folds_ob = []
result_bestjw6_test_folds_ob = []

result_bestloss_train_folds_re = []
result_bestloss_val_folds_re = []
result_bestloss_test_folds_re = []
result_bestauc_train_folds_re = []
result_bestauc_val_folds_re = []
result_bestauc_test_folds_re = []
result_bestjw5_train_folds_re = []
result_bestjw5_val_folds_re = []
result_bestjw5_test_folds_re = []
result_bestjw6_train_folds_re = []
result_bestjw6_val_folds_re = []
result_bestjw6_test_folds_re = []

num_fold = 0
start_num_fold = 0
# opt.continue_train = True
if opt.continue_train:
    start_num_fold, \
        result_bestloss_train_folds_ob, result_bestloss_val_folds_ob, result_bestloss_test_folds_ob, \
        result_bestauc_train_folds_ob, result_bestauc_val_folds_ob, result_bestauc_test_folds_ob, \
        result_bestjw5_train_folds_ob, result_bestjw5_val_folds_ob, result_bestjw5_test_folds_ob, \
        result_bestjw6_train_folds_ob, result_bestjw6_val_folds_ob, result_bestjw6_test_folds_ob \
        = load_temp(save_result_temp_ob)

    start_num_fold, \
        result_bestloss_train_folds_re, result_bestloss_val_folds_re, result_bestloss_test_folds_re, \
        result_bestauc_train_folds_re, result_bestauc_val_folds_re, result_bestauc_test_folds_re, \
        result_bestjw5_train_folds_re, result_bestjw5_val_folds_re, result_bestjw5_test_folds_re, \
        result_bestjw6_train_folds_re, result_bestjw6_val_folds_re, result_bestjw6_test_folds_re \
        = load_temp(save_result_temp_re)

for train_index, val_index in skf.split(X_train, Y_train):

    if num_fold < start_num_fold:
        num_fold = num_fold + 1
        continue

    save_logdir_fold = save_logdir + str(num_fold) + '/'
    save_model_fold = save_model + str(num_fold) + '/'
    writer = SummaryWriter(log_dir=save_logdir_fold)

    os.makedirs(save_logdir_fold, exist_ok=True)
    os.makedirs(save_model_fold, exist_ok=True)

    x_train, x_val = X_train[train_index], X_train[val_index]
    y_train, y_val = Y_train[train_index], Y_train[val_index]
    print(y_val.sum())

    trainset = select_dataloader(x_train, y_train, 'ID', patient_died_ct, patient_survived_ct,
                                 opt.datapath_train, opt.datapath_mask_train, opt.aug_train, opt.use_clinical,
                                 opt.data_clinical, opt.clinical_category,
                                 opt.clinical_continuous, '2D_montage_ITAC_fusion', opt, '')
    valset = select_dataloader(x_val, y_val, 'ID', patient_died_ct, patient_survived_ct,
                               opt.datapath_train, opt.datapath_mask_train, opt.aug_val, opt.use_clinical,
                               opt.data_clinical, opt.clinical_category,
                               opt.clinical_continuous, '2D_montage_ITAC_fusion', opt, '')
    testset = select_dataloader(x_test, y_test, 'ID', patient_died_ct, patient_survived_ct, opt.datapath_test,
                                opt.datapath_test, opt.aug_test, opt.use_clinical, opt.data_clinical,
                                opt.clinical_category,
                                opt.clinical_continuous, '2D_montage_ITAC_fusion', opt, '')


    clinical_train_x = trainset.clinical_list
    clinical_train_y = trainset.label_list

    clinical_val_x = valset.clinical_list
    clinical_val_y = valset.label_list

    clinical_test_x = testset.clinical_list
    clinical_test_y = testset.label_list

    from sklearn.linear_model import LogisticRegression
    from sklearn.svm import SVC
    from sklearn.ensemble import RandomForestClassifier
    from xgboost import XGBClassifier
    from sklearn.metrics import confusion_matrix
    from sklearn.metrics import precision_score, recall_score

    # model = SVC(kernel='linear', gamma=0.5, C=1, probability=True, class_weight={1:2})
    # model = SVC(kernel='rbf', gamma=0.5, C=0.8, probability=True, class_weight={1: 1})
    # model = RandomForestClassifier(random_state=42)

    model = XGBClassifier(learning_rate=0.001,
                            n_estimators=10,
                            max_depth=2
                            min_child_weight=1,
                            gamma=0.,
                            subsample=1,
                            colsample_btree=1, 
                            scale_pos_weight=1,
                            slient=0
                            )
    model.fit(clinical_train_x, clinical_train_y)


    y_train_pred = model.predict(clinical_train_x)
    y_val_pred = model.predict(clinical_val_x)
    y_test_pred = model.predict(clinical_test_x)

    y_train_score = model.predict_proba(clinical_train_x)[:, 1]
    y_val_score = model.predict_proba(clinical_val_x)[:, 1]
    y_test_score = model.predict_proba(clinical_test_x)[:, 1]

    # show result
    acc_train = model.score(clinical_train_x, clinical_train_y)
    precision_train = precision_score(clinical_train_y, y_train_pred)
    recall_train = recall_score(clinical_train_y, y_train_pred)

    acc_val = model.score(clinical_val_x, clinical_val_y)
    precision_val = precision_score(clinical_val_y, y_val_pred)
    recall_val = recall_score(clinical_val_y, y_val_pred)

    acc_test = model.score(clinical_test_x, clinical_test_y)
    precision_test = precision_score(clinical_test_y, y_test_pred)
    recall_test = recall_score(clinical_test_y, y_test_pred)

    matrix_train = confusion_matrix(clinical_train_y, y_train_pred)
    matrix_val = confusion_matrix(clinical_val_y, y_val_pred)
    matrix_test = confusion_matrix(clinical_test_y, y_test_pred)

    tpr_train, fpr_train, auc_train = get_roc(y_train_score, np.array(clinical_train_y))
    tpr_val, fpr_val, auc_val = get_roc(y_val_score, np.array(clinical_val_y))
    tpr_test, fpr_test, auc_test = get_roc(y_test_score, np.array(clinical_test_y))


    result_bestloss_train_ob = [0, auc_train, acc_train, precision_train, recall_train, precision_train*0.5 + recall_train*0.5, precision_train*0.6 + recall_train*0.4]
    result_bestloss_val_ob = [0, auc_val, acc_val, precision_val, recall_val,  precision_val*0.5 + recall_val*0.5, precision_val*0.6 + recall_val*0.4]
    result_bestloss_test_ob = [0, auc_test, acc_test, precision_test, recall_test,  precision_test*0.5 + recall_test*0.5, precision_test*0.6 + recall_test*0.4]
    result_bestloss_train_re = [0, auc_train, acc_train, precision_train, recall_train, precision_train*0.5 + recall_train*0.5, precision_train*0.6 + recall_train*0.4]
    result_bestloss_val_re = [0, auc_val, acc_val, precision_val, recall_val,  precision_val*0.5 + recall_val*0.5, precision_val*0.6 + recall_val*0.4]
    result_bestloss_test_re = [0, auc_test, acc_test, precision_test, recall_test,  precision_test*0.5 + recall_test*0.5, precision_test*0.6 + recall_test*0.4]
    result_bestauc_train_ob = [0, auc_train, acc_train, precision_train, recall_train,
                               precision_train * 0.5 + recall_train * 0.5, precision_train * 0.6 + recall_train * 0.4]
    result_bestauc_val_ob = [0, auc_val, acc_val, precision_val, recall_val, precision_val * 0.5 + recall_val * 0.5,
                             precision_val * 0.6 + recall_val * 0.4]
    result_bestauc_test_ob = [0, auc_test, acc_test, precision_test, recall_test,
                              precision_test * 0.5 + recall_test * 0.5, precision_test * 0.6 + recall_test * 0.4]
    result_bestauc_train_re = [0, auc_train, acc_train, precision_train, recall_train,
                               precision_train * 0.5 + recall_train * 0.5, precision_train * 0.6 + recall_train * 0.4]
    result_bestauc_val_re = [0, auc_val, acc_val, precision_val, recall_val, precision_val * 0.5 + recall_val * 0.5,
                             precision_val * 0.6 + recall_val * 0.4]
    result_bestauc_test_re = [0, auc_test, acc_test, precision_test, recall_test,
                              precision_test * 0.5 + recall_test * 0.5, precision_test * 0.6 + recall_test * 0.4]
    result_bestjw5_train_ob = [0, auc_train, acc_train, precision_train, recall_train,
                               precision_train * 0.5 + recall_train * 0.5, precision_train * 0.6 + recall_train * 0.4]
    result_bestjw5_val_ob = [0, auc_val, acc_val, precision_val, recall_val, precision_val * 0.5 + recall_val * 0.5,
                             precision_val * 0.6 + recall_val * 0.4]
    result_bestjw5_test_ob = [0, auc_test, acc_test, precision_test, recall_test,
                              precision_test * 0.5 + recall_test * 0.5, precision_test * 0.6 + recall_test * 0.4]
    result_bestjw5_train_re = [0, auc_train, acc_train, precision_train, recall_train,
                               precision_train * 0.5 + recall_train * 0.5, precision_train * 0.6 + recall_train * 0.4]
    result_bestjw5_val_re = [0, auc_val, acc_val, precision_val, recall_val, precision_val * 0.5 + recall_val * 0.5,
                             precision_val * 0.6 + recall_val * 0.4]
    result_bestjw5_test_re = [0, auc_test, acc_test, precision_test, recall_test,
                              precision_test * 0.5 + recall_test * 0.5, precision_test * 0.6 + recall_test * 0.4]
    result_bestjw6_train_ob = [0, auc_train, acc_train, precision_train, recall_train,
                               precision_train * 0.5 + recall_train * 0.5, precision_train * 0.6 + recall_train * 0.4]
    result_bestjw6_val_ob = [0, auc_val, acc_val, precision_val, recall_val, precision_val * 0.5 + recall_val * 0.5,
                             precision_val * 0.6 + recall_val * 0.4]
    result_bestjw6_test_ob = [0, auc_test, acc_test, precision_test, recall_test,
                              precision_test * 0.5 + recall_test * 0.5, precision_test * 0.6 + recall_test * 0.4]
    result_bestjw6_train_re = [0, auc_train, acc_train, precision_train, recall_train,
                               precision_train * 0.5 + recall_train * 0.5, precision_train * 0.6 + recall_train * 0.4]
    result_bestjw6_val_re = [0, auc_val, acc_val, precision_val, recall_val, precision_val * 0.5 + recall_val * 0.5,
                             precision_val * 0.6 + recall_val * 0.4]
    result_bestjw6_test_re = [0, auc_test, acc_test, precision_test, recall_test,
                              precision_test * 0.5 + recall_test * 0.5, precision_test * 0.6 + recall_test * 0.4]

    result_bestloss_train_folds_ob.append(result_bestloss_train_ob)
    result_bestloss_val_folds_ob.append(result_bestloss_val_ob)
    result_bestloss_test_folds_ob.append(result_bestloss_test_ob)
    result_bestauc_train_folds_ob.append(result_bestauc_train_ob)
    result_bestauc_val_folds_ob.append(result_bestauc_val_ob)
    result_bestauc_test_folds_ob.append(result_bestauc_test_ob)
    result_bestjw5_train_folds_ob.append(result_bestjw5_train_ob)
    result_bestjw5_val_folds_ob.append(result_bestjw5_val_ob)
    result_bestjw5_test_folds_ob.append(result_bestjw5_test_ob)
    result_bestjw6_train_folds_ob.append(result_bestjw6_train_ob)
    result_bestjw6_val_folds_ob.append(result_bestjw6_val_ob)
    result_bestjw6_test_folds_ob.append(result_bestjw6_test_ob)

    save_temp(result_bestloss_train_folds_ob, result_bestloss_val_folds_ob, result_bestloss_test_folds_ob, \
              result_bestauc_train_folds_ob, result_bestauc_val_folds_ob, result_bestauc_test_folds_ob, \
              result_bestjw5_train_folds_ob, result_bestjw5_val_folds_ob, result_bestjw5_test_folds_ob, \
              result_bestjw6_train_folds_ob, result_bestjw6_val_folds_ob, result_bestjw6_test_folds_ob,
              save_result_temp_ob)

    result_bestloss_train_folds_re.append(result_bestloss_train_re)
    result_bestloss_val_folds_re.append(result_bestloss_val_re)
    result_bestloss_test_folds_re.append(result_bestloss_test_re)
    result_bestauc_train_folds_re.append(result_bestauc_train_re)
    result_bestauc_val_folds_re.append(result_bestauc_val_re)
    result_bestauc_test_folds_re.append(result_bestauc_test_re)
    result_bestjw5_train_folds_re.append(result_bestjw5_train_re)
    result_bestjw5_val_folds_re.append(result_bestjw5_val_re)
    result_bestjw5_test_folds_re.append(result_bestjw5_test_re)
    result_bestjw6_train_folds_re.append(result_bestjw6_train_re)
    result_bestjw6_val_folds_re.append(result_bestjw6_val_re)
    result_bestjw6_test_folds_re.append(result_bestjw6_test_re)

    save_temp(result_bestloss_train_folds_re, result_bestloss_val_folds_re, result_bestloss_test_folds_re, \
              result_bestauc_train_folds_re, result_bestauc_val_folds_re, result_bestauc_test_folds_re, \
              result_bestjw5_train_folds_re, result_bestjw5_val_folds_re, result_bestjw5_test_folds_re, \
              result_bestjw6_train_folds_re, result_bestjw6_val_folds_re, result_bestjw6_test_folds_re,
              save_result_temp_re)

    num_fold = num_fold + 1

save_final(result_bestloss_train_folds_ob, result_bestloss_val_folds_ob, result_bestloss_test_folds_ob,
           result_bestauc_train_folds_ob, result_bestauc_val_folds_ob, result_bestauc_test_folds_ob,
           result_bestjw5_train_folds_ob, result_bestjw5_val_folds_ob, result_bestjw5_test_folds_ob,
           result_bestjw6_train_folds_ob, result_bestjw6_val_folds_ob, result_bestjw6_test_folds_ob, save_result_ob)

save_final(result_bestloss_train_folds_re, result_bestloss_val_folds_re, result_bestloss_test_folds_re,
           result_bestauc_train_folds_re, result_bestauc_val_folds_re, result_bestauc_test_folds_re,
           result_bestjw5_train_folds_re, result_bestjw5_val_folds_re, result_bestjw5_test_folds_re,
           result_bestjw6_train_folds_re, result_bestjw6_val_folds_re, result_bestjw6_test_folds_re, save_result_re)
