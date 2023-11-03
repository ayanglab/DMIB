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

setup_seed(20)

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
save_logdir = save_location + opt.expname + '/logdir/' # fold1/2/3/4/5

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
x_train_died, x_test_died, y_train_died, y_test_died = train_test_split(patient_died, np.ones(len(patient_died)).tolist(), test_size=0.25, random_state=20)
x_train_survived, x_test_survived, y_train_survived, y_test_survived = train_test_split(patient_survived, np.ones(len(patient_survived)).tolist(), test_size=0.25, random_state=20)
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

    plot_statistic_continuous(x_train, x_test, pd.concat([patient_survived_ct, patient_died_ct]), ['DiffFromLast'], plot_hist=True)
    plot_statistic_continuous(x_train_unique, x_test_unique, patient_info_unique, ['Age'])
    plot_statistic_category(x_train_unique, x_test_unique, patient_info_unique, ['Sex'])

'''
Split the train dataset by 10-fold: select the model with largest AUC on the validation for the test set .
'''

skf = StratifiedKFold(n_splits=5, random_state=20, shuffle=True)
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
                                opt.datapath_mask_test, opt.aug_test, opt.use_clinical, opt.data_clinical,
                                opt.clinical_category,
                                opt.clinical_continuous, '2D_montage_ITAC_fusion', opt, '')

    trainset_loader = DataLoader(trainset, batch_size=opt.bs, shuffle=True)
    valset_loader = DataLoader(valset, batch_size=opt.bs, shuffle=False)
    testset_loader = DataLoader(testset, batch_size=opt.bs, shuffle=False)
    
    model = select_model(opt, num_fold)
    model = model.to(opt.device)
    # model = torch.nn.DataParallel(model)
    
    # logging setup
    # wandb.watch(model)
    
    optimizer = select_optimizer(opt, model)
    # train_scheduler = CosineAnnealingLR(optimizer, T_max = 500, eta_min = 2e-8)
    # test_scheduler = None
    # train_scheduler = None
    # test_scheduler = ReduceLROnPlateau(optimizer, patience=5)
    train_scheduler = None
    test_scheduler = None
    
    loss_val_best_ob = 9999
    auc_val_best_ob = -9999
    jw5_val_best_ob = -9999
    jw6_val_best_ob = -9999
    loss_val_best_re = 9999
    auc_val_best_re = -9999
    jw5_val_best_re = -9999
    jw6_val_best_re = -9999

    if opt.use_fuse_clinical_axial:
        # state_dict_clinical = torch.load(opt.path_pretrain + opt.pretrain_clinical + '/model/' + str(num_fold) + '/best_auc.pt')['model_state_dict']
        # state_dict_ct_axial = torch.load(opt.path_pretrain + opt.pretrain_ct_axial + '/model/' + str(num_fold) + '/best_auc.pt')['model_state_dict']
        # for key in list(state_dict_clinical.keys()):
        #     if 'clinical_backbone' not in key:
        #         # print("del: {}".format(key))
        #         del state_dict_clinical[key]
        # for key in list(state_dict_ct_axial.keys()):
        #     if 'image_axial' not in key:
        #         # print("del: {}".format(key))
        #         del state_dict_ct_axial[key]
        # model.load_state_dict(state_dict_clinical, strict=False)
        # model.load_state_dict(state_dict_ct_axial, strict=False)

        # tuning fused backbone only
        if opt.use_fix:
            for name, param in model.named_parameters():
                # if "fuse_image_clinical" in name:
                #     param.requires_grad = True
                #     print("fuse {}".format(name))
                # else:
                #     param.requires_grad = False
                #     print("fix {}".format(name))
                if "clinical_backbone" in name or "image_axial" in name or 'image_coronal' in name:
                    param.requires_grad = False
                    # print("fix {}".format(name))
                else:
                    param.requires_grad = True
                    print("tuning {}".format(name))

    if opt.use_fuse_axial_coronal:
        path_pretrain = './result_exp/'

        state_dict_ct_axial = \
        torch.load(path_pretrain + opt.pretrain_ct_axial + '/model/' + str(num_fold) + '/best_auc.pt')[
            'model_state_dict']

        state_dict_ct_coronal = \
            torch.load(path_pretrain + opt.pretrain_ct_coronal + '/model/' + str(num_fold) + '/best_auc.pt')[
                'model_state_dict']

        for key in list(state_dict_ct_coronal.keys()):
            if 'image_coronal' not in key:
                # print("del: {}".format(key))
                del state_dict_ct_coronal[key]

        for key in list(state_dict_ct_axial.keys()):
            if 'image_axial' not in key:
                del state_dict_ct_axial[key]
        model.load_state_dict(state_dict_ct_coronal, strict=False)
        model.load_state_dict(state_dict_ct_axial, strict=False)

        # tuning fused backbone only
        if opt.use_fix:
            for name, param in model.named_parameters():
                # if "fuse_image_image" in name:
                #     param.requires_grad = True
                #     print("fuse {}".format(name))
                # else:
                #     param.requires_grad = False
                #     print("fix {}".format(name))

                if "clinical_backbone" in name or "image_axial" in name or 'image_coronal' in name:
                    param.requires_grad = False
                    # print("fix {}".format(name))
                else:
                    param.requires_grad = False
                    print("tuning {}".format(name))

    for epoch in range(0, opt.epoch):
        if 'wandb' in str(type(writer)):
            write_epoch = opt.epoch*num_fold + epoch
        else:
            write_epoch = epoch
        
        # Train
        model.train()
        train_result_dict = trainer_fuse(opt, trainset_loader, model, optimizer, train_scheduler, 0, epoch)
        write_result(writer, 'train', write_epoch, train_result_dict)
        for key in ['loss', 'acc', 'auc', 'sens', 'spec', 'jw5', 'jw6']:
            for k in ['ob', 're', 'axial', 'cli_cro']:
                locals()[f'{key}_train_{k}'] = train_result_dict[f'{key}_{k}']
        
        # Val
        model.eval()
        with torch.no_grad():
            val_result_dict = trainer_fuse(opt, valset_loader, model, optimizer, None, 0, epoch, False)
        write_result(writer, 'val', write_epoch, val_result_dict)
        for key in ['loss', 'acc', 'auc', 'sens', 'spec', 'jw5', 'jw6']:
            for k in ['ob', 're', 'axial', 'cli_cro']:
                locals()[f'{key}_val_{k}'] = val_result_dict[f'{key}_{k}']
        print("End check: {}, {}".format(auc_val_axial, auc_val_cli_cro))
        
        # Test
        model.eval()
        with torch.no_grad():
            test_result_dict = trainer_fuse(opt, testset_loader, model, optimizer, test_scheduler, 0, epoch, False)
        write_result(writer, 'test', write_epoch, test_result_dict)
        for key in ['loss', 'acc', 'auc', 'sens', 'spec', 'jw5', 'jw6']:
            for k in ['ob', 're', 'axial', 'cli_cro']:
                locals()[f'{key}_test_{k}'] = test_result_dict[f'{key}_{k}']
        
        # Write Results to CSV
        if opt.use_only_clinical or opt.use_only_ct_coronal:
            for _type in ['train', 'val', 'test']:
                for key in ['loss', 'acc', 'auc', 'sens', 'spec', 'jw5', 'jw6']:
                    locals()[f'{key}_{_type}_ob'] = eval(f'{_type}_result_dict')[f'{key}_{_type}_cli_cro']
                    locals()[f'{key}_{_type}_re'] = eval(f'{_type}_result_dict')[f'{key}_{_type}_cli_cro']
            
        elif opt.use_only_ct_axial:
            for _type in ['train', 'val', 'test']:
                for key in ['loss', 'acc', 'auc', 'sens', 'spec', 'jw5', 'jw6']:
                    locals()[f'{key}_{_type}_ob'] = eval(f'{_type}_result_dict')[f'{key}_{_type}_axial']
                    locals()[f'{key}_{_type}_re'] = eval(f'{_type}_result_dict')[f'{key}_{_type}_axial']

        for key in ['loss', 'auc', 'jw5', 'jw6']:
            if key == 'loss':
                compare = '<'
            else:
                compare = '>'
            if eval(f'{key}_val_ob{compare}{key}_val_best_ob'):
                torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()},
                        save_model_fold + 'best_{key}_ob.pt')
                locals()[f'result_best{key}_train_ob'] = [loss_train_ob, auc_train_ob, acc_train_ob, sens_train_ob, spec_train_ob, jw5_train_ob, jw6_train_ob]
                locals()[f'result_best{key}_val_ob'] = [loss_val_ob, auc_val_ob, acc_val_ob, sens_val_ob, spec_val_ob, jw5_val_ob, jw6_val_ob]
                locals()[f'result_best{key}_test_ob'] = [loss_test_ob, auc_test_ob, acc_test_ob, sens_test_ob, spec_test_ob, jw5_test_ob, jw6_test_ob]
            
        if loss_val_ob < loss_val_best_ob:
            loss_val_best_ob = loss_val_ob
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()},
                       # 'optimizer_state_dict': optimizer.state_dict()},
                       save_model_fold + 'best_loss_ob.pt')
            result_bestloss_train_ob = [loss_train_ob, auc_train_ob, acc_train_ob, sens_train_ob, spec_train_ob, jw5_train_ob, jw6_train_ob]
            result_bestloss_val_ob   = [loss_val_ob, auc_val_ob, acc_val_ob, sens_val_ob, spec_val_ob, jw5_val_ob, jw6_val_ob]
            result_bestloss_test_ob  = [loss_test_ob, auc_test_ob, acc_test_ob, sens_test_ob, spec_test_ob, jw5_test_ob, jw6_test_ob]
        if auc_val_ob > auc_val_best_ob:
            auc_val_best_ob = auc_val_ob
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()},
                       # 'optimizer_state_dict': optimizer.state_dict()},
                       save_model_fold + 'best_auc_ob.pt')
            result_bestauc_train_ob = [auc_train_ob, auc_train_ob, acc_train_ob, sens_train_ob, spec_train_ob, jw5_train_ob, jw6_train_ob]
            result_bestauc_val_ob =   [loss_val_ob, auc_val_ob, acc_val_ob, sens_val_ob, spec_val_ob, jw5_val_ob, jw6_val_ob]
            result_bestauc_test_ob =  [loss_test_ob, auc_test_ob, acc_test_ob, sens_test_ob, spec_test_ob, jw5_test_ob, jw6_test_ob]
        if jw5_val_ob > jw5_val_best_ob:
            jw5_val_best_ob = jw5_val_ob
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()},
                       # 'optimizer_state_dict': optimizer.state_dict()},
                       save_model_fold + 'best_jw5_ob.pt')
            result_bestjw5_train_ob = [auc_train_ob, auc_train_ob, acc_train_ob, sens_train_ob, spec_train_ob, jw5_train_ob, jw6_train_ob]
            result_bestjw5_val_ob =   [loss_val_ob, auc_val_ob, acc_val_ob, sens_val_ob, spec_val_ob, jw5_val_ob, jw6_val_ob]
            result_bestjw5_test_ob =  [loss_test_ob, auc_test_ob, acc_test_ob, sens_test_ob, spec_test_ob, jw5_test_ob, jw6_test_ob]
        if jw6_val_ob > jw6_val_best_ob:
            jw6_val_best_ob = jw6_val_ob
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()},
                       # 'optimizer_state_dict': optimizer.state_dict()},
                       save_model_fold + 'best_jw6_ob.pt')
            result_bestjw6_train_ob = [auc_train_ob, auc_train_ob, acc_train_ob, sens_train_ob, spec_train_ob,jw5_train_ob, jw6_train_ob]
            result_bestjw6_val_ob = [loss_val_ob, auc_val_ob, acc_val_ob, sens_val_ob, spec_val_ob, jw5_val_ob,jw6_val_ob]
            result_bestjw6_test_ob = [loss_test_ob, auc_test_ob, acc_test_ob, sens_test_ob, spec_test_ob, jw5_test_ob,jw6_test_ob]

        if loss_val_re < loss_val_best_re:
            loss_val_best_re = loss_val_re
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()},
                       save_model_fold + 'best_loss_re.pt')
            result_bestloss_train_re = [loss_train_re, auc_train_re, acc_train_re, sens_train_re, spec_train_re, jw5_train_re, jw6_train_re]
            result_bestloss_val_re   = [loss_val_re,   auc_val_re,   acc_val_re,   sens_val_re,   spec_val_re,   jw5_val_re,   jw6_val_re]
            result_bestloss_test_re  = [loss_test_re,  auc_test_re,  acc_test_re,  sens_test_re,  spec_test_re,  jw5_test_re,  jw6_test_re]
        if auc_val_re > auc_val_best_re:
            auc_val_best_re = auc_val_re
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()},
                       save_model_fold + 'best_auc_re.pt')
            result_bestauc_train_re = [loss_train_re, auc_train_re, acc_train_re, sens_train_re, spec_train_re,jw5_train_re, jw6_train_re]
            result_bestauc_val_re =   [loss_val_re,   auc_val_re,   acc_val_re,   sens_val_re,   spec_val_re,  jw5_val_re,   jw6_val_re]
            result_bestauc_test_re =  [loss_test_re,  auc_test_re,  acc_test_re,  sens_test_re,  spec_test_re, jw5_test_re,  jw6_test_re]
        if jw5_val_re > jw5_val_best_re:
            jw5_val_best_re = jw5_val_re
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()},
                       save_model_fold + 'best_jw5_re.pt')
            result_bestjw5_train_re = [loss_train_re, auc_train_re, acc_train_re, sens_train_re, spec_train_re, jw5_train_re, jw6_train_re]
            result_bestjw5_val_re = [loss_val_re, auc_val_re, acc_val_re, sens_val_re, spec_val_re, jw5_val_re, jw6_val_re]
            result_bestjw5_test_re = [loss_test_re, auc_test_re, acc_test_re, sens_test_re, spec_test_re, jw5_test_re, jw6_test_re]
        if jw6_val_re > jw6_val_best_re:
            jw6_val_best_re = jw6_val_re
            torch.save({'epoch': epoch,
                        'model_state_dict': model.state_dict()},
                       # 'optimizer_state_dict': optimizer.state_dict()},
                       save_model_fold + 'best_jw6_re.pt')
            result_bestjw6_train_re = [loss_train_re, auc_train_re, acc_train_re, sens_train_re, spec_train_re, jw5_train_re, jw6_train_re]
            result_bestjw6_val_re = [loss_val_re, auc_val_re, acc_val_re, sens_val_re, spec_val_re, jw5_val_re, jw6_val_re]
            result_bestjw6_test_re = [loss_test_re, auc_test_re, acc_test_re, sens_test_re, spec_test_re, jw5_test_re, jw6_test_re]

        if epoch == opt.epoch-1:
            '''
            Result of one fold
            '''
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
