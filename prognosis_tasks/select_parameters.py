import datetime
import argparse
import os

def write_parameter(paras, savepath):

    open_type = 'a' if os.path.exists(savepath) else 'w'
    now = datetime.datetime.now().strftime('%Y-%m-%d-%H:%M:%S')
    with open(savepath, open_type) as f:
        f.write(now + '\n\n')
        for arg in vars(paras):
            f.write('{}: {}\n'.format(arg, getattr(paras, arg)))
        f.write('\n')
        f.close()


def covid_mortality_parameter():
    parser = argparse.ArgumentParser(description='Process arguments')
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

    parser.add_argument('--patient_died_ct_csv', dest="patient_died_ct_csv", default='../dataset/ITAC/patients_enrol_list/observation_enrolled_died_10.csv')
    parser.add_argument('--patient_survived_ct_csv', dest="patient_survived_ct_csv", default='../dataset/ITAC/patients_enrol_list/observation_enrolled_survived_10.csv')
    parser.add_argument('--patients_info_csv', dest="patients_info_csv", default='../dataset/ITAC/patients_enrol_list/observation_enrolled_all_10.csv')
    parser.add_argument('--expname', dest="expname", default="ITAC_mortality")
    parser.add_argument('--gpu_num', dest="gpu_num", default="0", type=str)

    # parameters for dataset
    # parser.add_argument('--dataloader', dest="dataloader", default='2D_montage_CAM')
    parser.add_argument('--datapath_train', dest="datapath_train", default='../dataset/ITAC/CT_img_3D_cropped_montage_masked/')
    parser.add_argument('--datapath_coronal', dest="datapath_coronal", default='../dataset/ITAC/CT_img_3D_resize350_350_350_montage_Coronal/')
    parser.add_argument('--datapath_mask_train', dest="datapath_mask_train", default='../dataset/ITAC/CT_img_3D_cropped_montage_masked/')
    parser.add_argument('--datapath_test', dest="datapath_test", default='../dataset/ITAC/CT_img_3D_cropped_montage_masked/')
    parser.add_argument('--datapath_mask_test', dest="datapath_mask_test", default='../dataset/ITAC/CT_img_3D_cropped_montage_masked/')
    parser.add_argument('--data_clinical', dest="data_clinical", default='../dataset/ITAC/patients_enrol_list/impute_mean_observation_enrolled_all_10.csv')
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
    parser.add_argument('--alpha', dest="alpha", default=1, type=float)
    parser.add_argument('--gamma', dest="gamma", default=1, type=float)
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
    parser.add_argument('--lr', dest="lr", default=1e-6, type=float)
    parser.add_argument('--lr2', dest="lr2", default=1e-6, type=float)
    parser.add_argument('--jw_ratio', dest="jw_ratio", default=0.6, type=float)

    # IB
    parser.add_argument('--epoch', dest="epoch", default=70, type=int)
    parser.add_argument('--epoch_IB', dest="epoch_IB", default=5, type=int)
    parser.add_argument('--IB_method', dest="IB_method", default=1, type=int)
    parser.add_argument('--dim_bottleneck', dest="dim_bottleneck", default=256, type=int)

    args = parser.parse_args()
    return args