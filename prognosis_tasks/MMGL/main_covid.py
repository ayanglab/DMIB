import os
import sys
sys.path.append("..")
sys.path.append("../..")
# from utils import quality_check
import sys
sys.path.append("..")

from select_parameters import covid_mortality_parameter, write_parameter
from select_dataloader import *
from sklearn.model_selection import train_test_split, StratifiedKFold
from torch.utils.data import DataLoader
import torch
import numpy as np
import pandas as pd
def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

import pickle
import random
import sys
import tempfile
import time

import gc
import matplotlib.cm
import networkx as nx
import numpy as np
import scipy.sparse as spsprs
from sklearn.model_selection import KFold,StratifiedKFold
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as fn
import torch.optim as optim
import pandas as pd
from network import *
from utils import *
from model import *
import dgl

from torch.utils.data import DataLoader


setup_seed(20)

class RedirectStdStreams:
    def __init__(self, stdout=None, stderr=None):
        self._stdout = stdout or sys.stdout
        self._stderr = stderr or sys.stderr

    def __enter__(self):
        self.old_stdout, self.old_stderr = sys.stdout, sys.stderr
        self.old_stdout.flush()
        self.old_stderr.flush()
        sys.stdout, sys.stderr = self._stdout, self._stderr

    def __exit__(self, exc_type, exc_value, traceback):
        self._stdout.flush()
        self._stderr.flush()
        sys.stdout = self.old_stdout
        sys.stderr = self.old_stderr

def set_rng_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    dgl.seed(seed)
    dgl.random.seed(seed)

def sen(con_mat, n):

    sen = []
    for i in range(n):
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        sen1 = tp / (tp + fn)
        sen.append(sen1)

    return sen

def spe(con_mat, n):
    spe = []
    for i in range(n):
        number = np.sum(con_mat[:, :])
        tp = con_mat[i][i]
        fn = np.sum(con_mat[i, :]) - tp
        fp = np.sum(con_mat[:, i]) - tp
        tn = number - tp - fn - fp
        spe1 = tn / (tn + fp)
        spe.append(spe1)

    return spe


def train_and_eval(datadir, datname, hyperpm):

    set_rng_seed(hyperpm.seed)
    path = './data/ITAC/'
    modal_feat_dict = np.load(path + 'modal_feat_dict.npy', allow_pickle=True).item()
    data = pd.read_csv(path + 'processed_standard_data.csv').values

    print('data shape: ', data.shape)
    if datname == 'TADPOLE':
        hyperpm.nclass = 3
        hyperpm.nmodal = 6
    elif datname == 'ABIDE':
        hyperpm.nclass = 2
        hyperpm.nmodal = 2
    elif datname == 'ITAC':
        hyperpm.nclass = 2
        hyperpm.nmodal = 2
    # np.random.shuffle(data)

    use_cuda = torch.cuda.is_available()
    dev = torch.device('cuda' if use_cuda else 'cpu')
    input_data_dims = []
    for i in modal_feat_dict.keys():
        input_data_dims.append(len(modal_feat_dict[i]))
    print('Modal dims ', input_data_dims)

    input_data = data[:, :-1]
    label = data[:, -1] - 1
    # skf = StratifiedKFold(n_splits=5, random_state=0, shuffle=True)

    set_rng_seed(hyperpm.seed)
    val_acc, tst_acc, tst_auc = [], [], []
    shared_acc_list, shared_auc_list = [], []
    sp_acc_list, sp_auc_list = [], []
    sens = []
    clk = 0
    
    
    if not hyperpm.clinical_category == '':
        hyperpm.clinical_category = [str(item) for item in hyperpm.clinical_category.split(',')]
    else:
        hyperpm.clinical_category = []

    if not hyperpm.clinical_continuous == '':
        hyperpm.clinical_continuous = [str(item) for item in hyperpm.clinical_continuous.split(',')]
    else:
        hyperpm.clinical_continuous = []

    hyperpm.len_clinical = len(hyperpm.clinical_category) + len(hyperpm.clinical_continuous)
    print("clinical data: {}".format(hyperpm.len_clinical))
    os.environ['CUDA_VISIBLE_DEVICES'] = hyperpm.gpu_num

    save_location = './covid_mortality_experiments/'
    save_logdir = save_location + hyperpm.expname + '/logdir/'  # fold1/2/3/4/5
    save_model = save_location + hyperpm.expname + '/model/'  # fold1/2/3/4/5, best_auc, best_loss
    save_param = save_location + hyperpm.expname + '/para.txt'
    save_result_ob = save_location + hyperpm.expname + '/result_ob.csv'
    save_result_temp_ob = save_location + hyperpm.expname + '/result_temp_ob.xlsx'
    save_result_re = save_location + hyperpm.expname + '/result_re.csv'
    save_result_temp_re = save_location + hyperpm.expname + '/result_temp_re.xlsx'


    os.makedirs(save_logdir, exist_ok=True)
    os.makedirs(save_model, exist_ok=True)


    write_parameter(hyperpm, save_param)
    patient_died_ct = pd.read_csv(hyperpm.patient_died_ct_csv)
    patient_survived_ct = pd.read_csv(hyperpm.patient_survived_ct_csv)
    patient_info = pd.read_csv(hyperpm.patients_info_csv)

    patient_died_ct['ID'] = patient_died_ct['ID'].astype(str)
    patient_survived_ct['ID'] = patient_survived_ct['ID'].astype(str)
    patient_info['ID'] = patient_info['ID'].astype(str)
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
    X_train = x_train_died + x_train_survived
    Y_train = np.ones(len(x_train_died)).tolist() + np.zeros(len(x_train_survived)).tolist()
    x_test = x_test_died + x_test_survived
    y_test = np.ones(len(x_test_died)).tolist() + np.zeros(len(x_test_survived)).tolist()
    skf = StratifiedKFold(n_splits=5, random_state=20, shuffle=True)
    Y_train = np.array(Y_train)
    X_train = np.array(X_train)

    num_fold = 0
    start_num_fold = 0

    for Train_index, Val_index in skf.split(X_train, Y_train):
    # for train_index, test_index in skf.split(input_data, label):

        '''
        from one year
        '''
        if num_fold < start_num_fold:
            num_fold = num_fold + 1
            continue

        save_logdir_fold = save_logdir + str(num_fold) + '/'
        save_model_fold = save_model + str(num_fold) + '/'
        # writer = SummaryWriter(log_dir=save_logdir_fold)

        os.makedirs(save_logdir_fold, exist_ok=True)
        os.makedirs(save_model_fold, exist_ok=True)

        x_train, x_val = X_train[Train_index], X_train[Val_index]
        y_train, y_val = Y_train[Train_index], Y_train[Val_index]
        print(y_val.sum())
        
        trainset = select_dataloader(x_train, y_train, 'ID', patient_died_ct, patient_survived_ct,
                                     hyperpm.datapath_train, hyperpm.datapath_mask_train, [1,1], hyperpm.use_clinical,
                                     hyperpm.data_clinical, hyperpm.clinical_category,
                                     hyperpm.clinical_continuous, '2D_montage_ITAC_fusion', hyperpm, '')
        valset = select_dataloader(x_val, y_val, 'ID', patient_died_ct, patient_survived_ct,
                                   hyperpm.datapath_train, hyperpm.datapath_mask_train, hyperpm.aug_val, hyperpm.use_clinical,
                                   hyperpm.data_clinical, hyperpm.clinical_category,
                                   hyperpm.clinical_continuous, '2D_montage_ITAC_fusion', hyperpm, '')
        testset = select_dataloader(x_test, y_test, 'ID', patient_died_ct, patient_survived_ct, hyperpm.datapath_test,
                                    hyperpm.datapath_test, hyperpm.aug_test, hyperpm.use_clinical, hyperpm.data_clinical,
                                    hyperpm.clinical_category,
                                    hyperpm.clinical_continuous, '2D_montage_ITAC_fusion', hyperpm, '')

        input_table = pd.read_csv('./covid/graph_feature_fold{}.csv'.format(str(clk)))

        label = np.array(input_table['label'].tolist())

        num_feature = len(hyperpm.clinical_continuous) + len(hyperpm.clinical_category)
        input_data_dims = [num_feature, 1024]
        input_data = input_table[hyperpm.clinical_continuous + hyperpm.clinical_category + ['img_' + str(i) for i in range((1024))]].values.astype('float64')


        # input_data_dims = [1, 256]
        train_index_select = [patient.split('/')[-1] for patient in trainset.data_list]
        train_index = np.array(input_table[input_table['ID'].isin(train_index_select)].index)
        test_index_select = [patient.split('/')[-1] for patient in valset.data_list]
        test_index = np.array(input_table[input_table['ID'].isin(test_index_select)].index)
        real_test_index_select = [patient.split('/')[-1] for patient in testset.data_list]
        real_test_index = np.array(input_table[input_table['ID'].isin(real_test_index_select)].index)
        '''
        from graph
        '''
        clk += 1
        agent = EvalHelper(input_data_dims, input_data, label, hyperpm, train_index, test_index, real_test_index)
        # agent = EvalHelper(input_data_dims, input_data, label, hyperpm, train_index, test_index, test_index)
        tm = time.time()
        best_val_acc, wait_cnt = 0.0, 0
        model_sav = tempfile.TemporaryFile()
        for t in range(hyperpm.nepoch):
            print('%3d/%d' % (t, hyperpm.nepoch), end=' ')
            agent.run_epoch(mode=hyperpm.mode, end=' ')
            _, cur_val_acc = agent.print_trn_acc(hyperpm.mode)  # validation
            if cur_val_acc > best_val_acc:
                wait_cnt = 0
                best_val_acc = cur_val_acc
                _, best_test_acc = agent.print_trn_acc(hyperpm.mode)
                model_sav.close()
                model_sav = tempfile.TemporaryFile()
                dict_list = [agent.ModalFusion.state_dict(),
                             agent.GraphConstruct.state_dict(),
                             agent.MessagePassing.state_dict()]
                torch.save(dict_list, model_sav)
            else:
                wait_cnt += 1
                if wait_cnt > hyperpm.early:
                    break
        print("time: %.4f sec." % (time.time() - tm))
        model_sav.seek(0)
        dict_list = torch.load(model_sav)
        agent.ModalFusion.load_state_dict(dict_list[0])
        agent.GraphConstruct.load_state_dict(dict_list[1])
        agent.MessagePassing.load_state_dict(dict_list[2])

        val_acc.append(best_val_acc)
        cur_tst_acc, cur_tst_auc = agent.print_tst_acc(hyperpm.mode)

        tst_acc.append(cur_tst_acc)
        tst_auc.append(cur_tst_auc)
        if np.array(tst_acc).mean() < 0.6 and clk == 5:
            break
    return np.array(val_acc).mean(), np.array(tst_acc).mean(), np.array(tst_acc).std(), np.array(
        tst_auc).mean(), np.array(tst_auc).std()

def main(args_str=None):
    assert float(torch.__version__[:3]) + 1e-3 >= 0.4
    parser = argparse.ArgumentParser()
    parser.add_argument('--datadir', type=str, default='./data/')
    parser.add_argument('--datname', type=str, default='TADPOLE')
    parser.add_argument('--cpu', action='store_true', default=False,
                        help='Insist on using CPU instead of CUDA.')
    parser.add_argument('--nepoch', type=int, default=1000,
                        help='Max number of epochs to train.')
    parser.add_argument('--early', type=int, default=150,
                        help='Extra iterations before early-stopping.')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Initial learning rate.')
    parser.add_argument('--reg', type=float, default=0.0036,
                        help='Weight decay (L2 loss on parameters).')
    parser.add_argument('--dropout', type=float, default=0.65,
                        help='Dropout rate (1 - keep probability).')
    parser.add_argument('--nlayer', type=int, default=3,
                        help='Number of conv layers.')
    parser.add_argument('--n_hidden', type=int, default=16,
                        help='Number of attention head.')
    parser.add_argument('--n_head', type=int, default=8,
                        help='Number of hidden units per modal.')
    parser.add_argument('--n_iter', type=int, default=10,
                        help='Number of alternate iteration.')
    parser.add_argument('--nmodal', type=int, default=6,
                        help='Size of the sampled neighborhood.')
    parser.add_argument('--th', type=float, default=0.9,
                        help='threshold of weighted cosine')
    parser.add_argument('--GC_mode', type=str, default='adaptive-learning',
                        help='graph constrcution mode')
    parser.add_argument('--MP_mode', type=str, default='GCN',
                        help='Massage Passing mode')
    parser.add_argument('--MF_mode', type=str, default=' ',
                        help='Massage Passing mode')
    parser.add_argument('--alpha', type=float, default='0.5',
                        help='alpha for GAT')
    parser.add_argument('--theta_smooth', type=float, default='1',
                        help='graph_loss_smooth')
    parser.add_argument('--theta_degree', type=float, default='0.5',
                        help='graph_loss_degree')
    parser.add_argument('--theta_sparsity', type=float, default='0.0',
                        help='graph_loss_sparsity')
    parser.add_argument('--nclass', type=int, default=3,
                        help='class number')
    parser.add_argument('--mode', type=str, default='pre-train',
                        help='training mode')
    parser.add_argument('--seed', type=int, default=0,
                        help='random seed setting')

    '''
    addparameter
    '''
    parser.add_argument('--local_rank', type=int, help='GPU id for logging results')

    parser.add_argument('--perform_statistics', action='store_true')
    parser.add_argument('--use_clinical', action='store_true')
    parser.add_argument('--use_two_benchmark', action='store_true')
    parser.add_argument('--only_clinical', action='store_true')
    parser.add_argument('--pretrained', action='store_true')
    parser.add_argument('--continue_train', action='store_true')
    parser.add_argument('--tuning_lr', action='store_true')
    parser.add_argument('--preprocess_clinical', dest="preprocess_clinical", default=0, type=int)

    parser.add_argument('--pretrain_clinical', dest="pretrain_clinical", default='fuse_crossmodal_only_clinical_1e-6', type=str)
    parser.add_argument('--pretrain_ct_axial', dest="pretrain_ct_axial", default='fuse_crossmodal_only_axial_1e-6', type=str)
    parser.add_argument('--pretrain_ct_coronal', dest="pretrain_ct_coronal", default='fuse_crossmodal_only_coronal_1e-6',type=str)

    parser.add_argument('--use_only_clinical', action='store_true')
    parser.add_argument('--use_only_ct_axial', action='store_true')
    parser.add_argument('--use_only_ct_coronal', action='store_true')
    parser.add_argument('--use_fuse_clinical_axial', action='store_true')
    parser.add_argument('--use_fuse_axial_coronal', action='store_true')
    parser.add_argument('--use_IB', action='store_true')
    parser.add_argument('--use_fix', action='store_true')
    parser.add_argument('--use_MI', action='store_true')
    parser.add_argument('--use_CONF', action='store_true')
    parser.add_argument('--use_ATTEN', action='store_true')
    parser.add_argument('--use_SPARSE', action='store_true')
    parser.add_argument('--use_similarity', action='store_true')
    parser.add_argument('--use_selfpretrain', action='store_true')
    parser.add_argument('--test_xai', action='store_true')
    parser.add_argument('--fuseadd', action='store_true')
    parser.add_argument('--fuseend', action='store_true')

    parser.add_argument('--patient_died_ct_csv', dest="patient_died_ct_csv", default='../../dataset/ITAC/patients_enrol_list/observation_enrolled_died_10.csv')
    parser.add_argument('--patient_survived_ct_csv', dest="patient_survived_ct_csv", default='../../dataset/ITAC/patients_enrol_list/observation_enrolled_survived_10.csv')
    parser.add_argument('--patients_info_csv', dest="patients_info_csv", default='../../dataset/ITAC/patients_enrol_list/observation_enrolled_all_10.csv')
    parser.add_argument('--expname', dest="expname", default="ITAC_mortality")
    parser.add_argument('--gpu_num', dest="gpu_num", default="0", type=str)

    # parameters for dataset
    # parser.add_argument('--dataloader', dest="dataloader", default='2D_montage_CAM')
    parser.add_argument('--datapath_train', dest="datapath_train", default='../../dataset/ITAC/CT_img_3D_cropped_montage_masked/')
    parser.add_argument('--datapath_coronal', dest="datapath_coronal", default='../../dataset/ITAC/CT_img_3D_resize350_350_350_montage_Coronal/')
    parser.add_argument('--datapath_mask_train', dest="datapath_mask_train", default='../../dataset/ITAC/CT_img_3D_cropped_montage_masked/')
    parser.add_argument('--datapath_test', dest="datapath_test", default='../../dataset/ITAC/CT_img_3D_cropped_montage_masked/')
    parser.add_argument('--datapath_mask_test', dest="datapath_mask_test", default='../../dataset/ITAC/CT_img_3D_cropped_montage_masked/')
    parser.add_argument('--data_clinical', dest="data_clinical", default='../../dataset/ITAC/patients_enrol_list/impute_mean_observation_enrolled_all_10.csv')
    parser.add_argument('--aug_train', dest="aug_train", nargs="+", default=[10, 12], type=int)
    parser.add_argument('--aug_val', dest="aug_val", nargs="+", default=[1, 1], type=int)
    parser.add_argument('--aug_test', dest="aug_test", nargs="+", default=[1, 1], type=int)
    parser.add_argument('--clinical_category', help='delimited list input', type=str, default="Cough,Dyspnea,Diabetes,Other CV disease,Neurological disease,label_icu")
    parser.add_argument('--clinical_continuous', help='delimited list input', type=str, default="Age,A&E_Respiratory rate,A&E_Oxygen saturation,Platelets,D-Dimer,Glucose,Urea,eGFR,GOT,PCR,ABG_pO2,ABG_measured saturation O2 ")

    # training
    parser.add_argument('--model_name', dest="model_name", default='densenet121')
    parser.add_argument('--bs', dest="bs", default=8, type=int)
    parser.add_argument('--drop_rate', dest="drop_rate", default=0, type=float)
    parser.add_argument('--device', dest="device", default="cuda", type=str)

    # model
    parser.add_argument('--in_channel', dest="in_channel", default=3, type=int)
    parser.add_argument('--len_clinical', dest="len_clinical", default=1, type=int)

    # loss
    parser.add_argument('--loss_name', dest="loss_name", default='Focal') # Focal, MSE
    parser.add_argument('--CE_loss', dest="CE_loss", default=1, type=float)

    # IB loss
    parser.add_argument('--VSD_loss', dest="VSD_loss", default=2, type=float)
    parser.add_argument('--focal_loss_representation', dest="focal_loss_representation", default=1, type=float)
    parser.add_argument('--focal_loss_observation', dest="focal_loss_observation", default=1, type=float)
    parser.add_argument('--temperature', dest="temperature", default=1, type=float)

    # loss weights
    parser.add_argument('--weight_ob', dest="weight_ob", default=1, type=float)
    parser.add_argument('--weight_re', dest="weight_re", default=1, type=float)
    parser.add_argument('--weight_axial', dest="weight_axial", default=1, type=float)
    parser.add_argument('--weight_cli_cro', dest="weight_cli_cro", default=1, type=float)
    parser.add_argument('--weight_MI', dest="weight_MI", default=1, type=float)
    parser.add_argument('--weight_IB', dest="weight_IB", default=10, type=float)
    parser.add_argument('--weight_CONF', dest="weight_CONF", default=1, type=float)
    parser.add_argument('--weight_SPARSE', dest="weight_SPARSE", default=0.5, type=float)
    parser.add_argument('--weight_similarity', dest="weight_similarity", default=1, type=float)

    # optimizer
    parser.add_argument('--opt_name', dest="opt_name", default='Adam')
    parser.add_argument('--jw_ratio', dest="jw_ratio", default=0.6, type=float)

    # IB
    parser.add_argument('--epoch', dest="epoch", default=70, type=int)
    parser.add_argument('--epoch_IB', dest="epoch_IB", default=5, type=int)
    parser.add_argument('--IB_method', dest="IB_method", default=1, type=int)
    parser.add_argument('--dim_bottleneck', dest="dim_bottleneck", default=256, type=int)

    if args_str is None:
        args = parser.parse_args()
    else:
        args = parser.parse_args(args_str.split())
    with RedirectStdStreams(stdout=sys.stderr):
        print('GC_mode:', args.GC_mode, 'MF_mode:', args.MF_mode)
        val_acc, tst_acc, tst_acc_std, tst_auc, tst_auc_std = train_and_eval(args.datadir, args.datname, args)
        print('val=%.2f%% tst_acc=%.2f%% tst_auc=%.2f%%' % (val_acc * 100, tst_acc * 100, tst_auc * 100))
        print('tst_acc_std=%.4f tst_auc_std=%.4f' % (tst_acc_std, tst_auc_std))
    return val_acc, tst_acc


if __name__ == '__main__':
    print(str(main()))
    for _ in range(5):
        gc.collect()
        torch.cuda.empty_cache()




