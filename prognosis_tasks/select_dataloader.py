from torch.utils.data import Dataset
import numpy as np
# import sys
# sys.path.append("..")
from PIL import Image
import torch
import pandas as pd


def select_dataloader(x, y, key_ct, died_table, survived_table, datapath, datapath_mask, aug_num, use_clinical, clinical_data, clinical_category, clinical_continuous, load_method, opt, datapath_coronal):

    if load_method == '2D_montage_ITAC':
        dataset = Dataset_2Dmontage_ITAC(x, y, key_ct, died_patient_ct = died_table, survived_patient_ct=survived_table,
                                     imagepath=datapath, maskpath=datapath_mask, use_aug=aug_num, use_clinical = use_clinical,
                                     clinical_data=clinical_data, clinical_category=clinical_category, clinical_continuous=clinical_continuous, opt=opt)

    elif load_method == '2D_montage_ITAC_fusion':
        dataset = Dataset_2Dmontage_ITAC_fusion(x, y, key_ct, died_patient_ct = died_table, survived_patient_ct=survived_table,
                                     imagepath=datapath, imagepath_coronal=opt.datapath_coronal, maskpath=datapath_mask, use_aug=aug_num, use_clinical = use_clinical,
                                     clinical_data=clinical_data, clinical_category=clinical_category, clinical_continuous=clinical_continuous, opt=opt)

    return dataset


# Cam
class Dataset_2Dmontage(Dataset):

    """Blur image dataset"""
    def __init__(self, x, y, key_ct, died_patient_ct, survived_patient_ct, imagepath, maskpath, use_aug = [1,1], use_clinical=False, clinical_data = "", clinical_continuous = [], clinical_category=[]):

        self.use_clinical = use_clinical
        if self.use_clinical:
            # self.clinical_data = process_clinical(clinical_data, clinical_continuous, clinical_category)
            df = pd.read_csv(clinical_data)
            features = clinical_continuous + clinical_category #['ID', 'mortality_status', 'progression12m', 'DOB', 'mortality_date', 'mortality_asofDate', 'HRCT_date', 'age']
            # features = ['Age', 'CT_B2']
            self.clinical_data = pd.DataFrame(df)
            self.clinical_list = []

        self.data_list_positive = []
        self.data_list_negative = []

        for i in range(len(x)):
            if y[i] == 1:
                self.data_list_positive.append(x[i])
            elif y[i] == 0:
                self.data_list_negative.append(x[i])
            else:
                print(x[i], y[i])

        self.ct_list_positive = died_patient_ct[died_patient_ct['Pseudonym'].isin(self.data_list_positive)]['CT_name']
        self.ct_list_negative = survived_patient_ct[survived_patient_ct['Pseudonym'].isin(self.data_list_negative)]['CT_name']


        self.imagepath = imagepath
        self.maskpath  = maskpath

        # self.image_resize = image_resize
        self.data_list = []
        self.mask_list = []
        self.label_list = []

        for i in range(use_aug[1]):
            ct_name = [self.imagepath + patient.replace('.nii.gz', '_s' + str(i) + '.png') for patient in self.ct_list_positive]
            ct_mask_name = [self.maskpath + patient.replace('.nii.gz', '_s' + str(i) + '.png') for patient in self.ct_list_positive]
            self.data_list.extend(ct_name)
            self.mask_list.extend(ct_mask_name)
            self.label_list.extend(np.ones(len(ct_name)).tolist())
            if self.use_clinical:
                ct_list = [df.loc[df['Pseudonym'] == int(patient.split('_')[0][5:]),features].values[0] for patient in self.ct_list_positive]
                self.clinical_list.extend(ct_list)

        for i in range(use_aug[0]):
            ct_name = [self.imagepath + patient.replace('.nii.gz', '_s' + str(i) + '.png') for patient in self.ct_list_negative]
            ct_mask_name = [self.maskpath + patient.replace('.nii.gz', '_s' + str(i) + '.png') for patient in self.ct_list_negative]
            self.data_list.extend(ct_name)
            self.mask_list.extend(ct_mask_name)
            self.label_list.extend(np.zeros(len(ct_name)).tolist())
            if self.use_clinical:
                ct_list = [df.loc[df['Pseudonym'] == int(patient.split('_')[0][5:]), features].values[0] for patient in self.ct_list_negative]
                self.clinical_list.extend(ct_list)

        print("hello")

    def __len__(self):
        return len(self.data_list)


    def __getitem__(self, idx):
        """get .mat file"""

        label = int(self.label_list[idx])

        img = Image.open(self.data_list[idx])
        img = np.asarray(img, dtype='float32')
        img = img/255.0 # include norm
        if img.ndim == 2:
            img = np.expand_dims(img, 0)
            img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img).float()

        img_mask = Image.open(self.mask_list[idx])
        img_mask = np.asarray(img_mask, dtype='float32')
        img_mask = img_mask/255.0  # include norm
        img_mask = np.expand_dims(img_mask, 0)
        img_mask = torch.from_numpy(img_mask).float()

        img_name = self.data_list[idx].split('/')[-1]
        # patient_name = img_name.split('_')[0] + '_' + img_name.split('_')[1]

        # clinical data
        if self.use_clinical:
            try:
                vectors = self.clinical_list[idx]
                vectors = vectors.astype("float32")
            except:
                print(patient_name)
        else:
            vectors = 0


        # img_path = self.data_list[idx]
        img_path = self.data_list[idx]

        return img, img_mask, vectors, label, img_name, img_path


class Dataset_2Dmontage_ITAC(Dataset):

    """Blur image dataset"""
    def __init__(self, x, y, key_ct, died_patient_ct, survived_patient_ct, imagepath, maskpath, use_aug = [1,1], use_clinical=False, clinical_data = "", clinical_continuous = [], clinical_category=[], opt=[]):

        # self.use_clinical = use_clinical
        # if self.use_clinical:
        #     self.clinical_data = process_clinical(clinical_data, clinical_continuous, clinical_category)
        self.use_clinical = use_clinical

        if self.use_clinical:
            # self.clinical_data = process_clinical(clinical_data, clinical_continuous, clinical_category)
            df = pd.read_csv(clinical_data)

            if opt.preprocess_clinical == 1:    # x/x_average-1
                for clinical_class in clinical_continuous:
                    df[clinical_class] = df[clinical_class] / df[clinical_class].mean() - 1
            elif opt.preprocess_clinical == 2: # (x-mean)/std
                for clinical_class in clinical_continuous:
                    df[clinical_class] = (df[clinical_class] - df[clinical_class].mean())/df[clinical_class].std()
            elif opt.preprocess_clinical == 3: # (x-min)/(max-min)
                for clinical_class in clinical_continuous:
                    df[clinical_class] = (df[clinical_class] - df[clinical_class].min())/(df[clinical_class].max() - df[clinical_class].min())

            features = clinical_continuous + clinical_category  # ['ID', 'mortality_status', 'progression12m', 'DOB', 'mortality_date', 'mortality_asofDate', 'HRCT_date', 'age']
            # features = ['Age', 'CT_B2']
            self.clinical_data = pd.DataFrame(df)
            self.clinical_list = []

        self.data_list_positive = []
        self.data_list_negative = []

        for i in range(len(x)):#Dataset_cam_2D_bootstrap_all
            if y[i] == 1:
                self.data_list_positive.append(x[i])
            elif y[i] == 0:
                self.data_list_negative.append(x[i])
            else:
                print(x[i], y[i])

        self.ct_list_positive = died_patient_ct[died_patient_ct[key_ct].isin(self.data_list_positive)][key_ct]
        self.ct_list_negative = survived_patient_ct[survived_patient_ct[key_ct].isin(self.data_list_negative)][key_ct]

        self.imagepath = imagepath
        self.maskpath  = maskpath

        # self.image_resize = image_resize
        self.data_list = []
        self.mask_list = []
        self.label_list = []

        for i in range(use_aug[1]):
            ct_name = [self.imagepath + patient.zfill(4)+'_s' + str(i) + '.png' for patient in self.ct_list_positive]
            ct_mask_name = [self.maskpath + patient.zfill(4)+'_s' + str(i) + '.png' for patient in self.ct_list_positive]
            self.data_list.extend(ct_name)
            self.mask_list.extend(ct_mask_name)
            self.label_list.extend(np.ones(len(ct_name)).tolist())
            if self.use_clinical:
                ct_list = [df.loc[df['ID'] == int(patient),features].values[0] for patient in self.ct_list_positive]
                self.clinical_list.extend(ct_list)

        for i in range(use_aug[0]):
            ct_name = [self.imagepath + patient.zfill(4) +'_s' + str(i) + '.png'for patient in self.ct_list_negative]
            ct_mask_name = [self.maskpath + patient.zfill(4) +'_s' + str(i) + '.png'for patient in self.ct_list_negative]
            self.data_list.extend(ct_name)
            self.mask_list.extend(ct_mask_name)
            self.label_list.extend(np.zeros(len(ct_name)).tolist())
            if self.use_clinical:
                ct_list = [df.loc[df['ID'] == int(patient), features].values[0] for patient in self.ct_list_negative]
                self.clinical_list.extend(ct_list)


    def __len__(self):
        return len(self.data_list)
        # return 20

    def __getitem__(self, idx):
        """get .mat file"""

        label = int(self.label_list[idx])
        try:
            img = Image.open(self.data_list[idx])
        except:
            print("stop here1: {}".format(self.data_list[idx]))
        try:
            img = np.asarray(img, dtype='float32')
        except:
            print("stop here2: {}".format(self.data_list[idx]))
        try:
            img = img/255.0 # include norm
        except:
            print("stop here3: {}".format(self.data_list[idx]))
        if img.ndim == 2:
            img = np.expand_dims(img, 0)
            img = np.repeat(img, 3, axis=0)
        img = torch.from_numpy(img).float()

        # img_mask = Image.open(self.mask_list[idx])
        # img_mask = np.asarray(img_mask, dtype='float32')
        # img_mask = img_mask/255.0  # include norm
        # img_mask = np.expand_dims(img_mask, 0)
        # img_mask = torch.from_numpy(img_mask).float()
        img_mask = img

        img_name = self.data_list[idx].split('/')[-1]
        # patient_name = img_name.split('_')[0] + '_' + img_name.split('_')[1]

        # clinical data
        if self.use_clinical:
            try:
                vectors = self.clinical_list[idx]
                vectors = vectors.astype("float32")
            except:
                print(patient_name)
        else:
            vectors = 0


        # img_path = self.data_list[idx]
        img_path = self.data_list[idx]

        return img, img_mask, vectors, label, img_name, img_path,


# to dos
class Dataset_2Dmontage_ITAC_fusion(Dataset):

    """Blur image dataset"""
    def __init__(self, x, y, key_ct, died_patient_ct, survived_patient_ct, imagepath, imagepath_coronal, maskpath, use_aug = [1,1], use_clinical=False, clinical_data = "", clinical_continuous = [], clinical_category=[], opt=[]):

        # self.use_clinical = use_clinical
        # if self.use_clinical:
        #     self.clinical_data = process_clinical(clinical_data, clinical_continuous, clinical_category)
        self.use_clinical = use_clinical

        self.use_only_clinical = opt.use_only_clinical
        self.use_only_ct_axial = opt.use_only_ct_axial
        self.use_only_ct_coronal = opt.use_only_ct_coronal
        self.use_fuse_clinical_axial = opt.use_fuse_clinical_axial
        self.use_fuse_axial_coronal = opt.use_fuse_axial_coronal


        if self.use_clinical:

            # self.clinical_data = process_clinical(clinical_data, clinical_continuous, clinical_category)
            df = pd.read_csv(clinical_data)

            # df['noise_age'] = ''
            # df['noise_float'] = ''
            # for i in range(len(df)):
            #     df['noise_age'][i] = np.random.randint(50, 100)
            #
            # for i in range(len(df)):
            #     df['noise_float'][i] = np.random.rand()
            # df.to_csv(clinical_data)


            if opt.preprocess_clinical == 1:    # x/x_average-1
                for clinical_class in clinical_continuous:
                    df[clinical_class] = df[clinical_class] / df[clinical_class].mean() - 1
            elif opt.preprocess_clinical == 2: # (x-mean)/std
                for clinical_class in clinical_continuous:
                    df[clinical_class] = (df[clinical_class] - df[clinical_class].mean())/df[clinical_class].std()
            elif opt.preprocess_clinical == 3: # (x-min)/(max-min)
                for clinical_class in clinical_continuous:
                    df[clinical_class] = (df[clinical_class] - df[clinical_class].min())/(df[clinical_class].max() - df[clinical_class].min())

            features = clinical_continuous + clinical_category  # ['ID', 'mortality_status', 'progression12m', 'DOB', 'mortality_date', 'mortality_asofDate', 'HRCT_date', 'age']
            # features = ['Age', 'CT_B2']
            self.clinical_data = pd.DataFrame(df)
            self.clinical_list = []

        self.data_list_positive = []
        self.data_list_negative = []

        for i in range(len(x)):#Dataset_cam_2D_bootstrap_all
            if y[i] == 1:
                self.data_list_positive.append(x[i])
            elif y[i] == 0:
                self.data_list_negative.append(x[i])
            else:
                print(x[i], y[i])

        self.ct_list_positive = died_patient_ct[died_patient_ct[key_ct].isin(self.data_list_positive)][key_ct]
        self.ct_list_negative = survived_patient_ct[survived_patient_ct[key_ct].isin(self.data_list_negative)][key_ct]

        self.imagepath = imagepath
        self.imagepath_coronal = imagepath_coronal
        self.maskpath  = maskpath

        # self.image_resize = image_resize
        self.data_list = []
        self.data_list2 = []
        self.mask_list = []
        self.label_list = []

        for i in range(use_aug[1]):

            ct_name = [self.imagepath + patient + '_s' + str(i) + '.png' for patient in self.ct_list_positive]
            ct_name_coronal = [self.maskpath + patient + '_s' + str(i) + '.png' for patient in self.ct_list_positive]
            ct_mask_name = [self.maskpath + patient + '_s' + str(i) + '.png' for patient in self.ct_list_positive]

            ct_name = [self.imagepath + patient.zfill(4)+'_s' + str(i) + '.png' for patient in self.ct_list_positive]
            ct_name_coronal = [self.imagepath_coronal + patient.zfill(4)+'_s' + str(i) + '.png' for patient in self.ct_list_positive]
            ct_mask_name = [self.maskpath + patient.zfill(4)+'_s' + str(i) + '.png' for patient in self.ct_list_positive]

            self.data_list.extend(ct_name)
            self.data_list2.extend(ct_name_coronal)
            self.mask_list.extend(ct_mask_name)
            self.label_list.extend(np.ones(len(ct_name)).tolist())
            if self.use_clinical:
                ct_list = [df.loc[df['ID'] == int(patient),features].values[0] for patient in self.ct_list_positive]
                self.clinical_list.extend(ct_list)

        for i in range(use_aug[0]):
            ct_name = [self.imagepath + patient.zfill(4) +'_s' + str(i) + '.png'for patient in self.ct_list_negative]
            ct_name_coronal = [self.imagepath_coronal + patient.zfill(4) + '_s' + str(i) + '.png' for patient in self.ct_list_negative]
            ct_mask_name = [self.maskpath + patient.zfill(4) +'_s' + str(i) + '.png'for patient in self.ct_list_negative]
            self.data_list.extend(ct_name)
            self.data_list2.extend(ct_name_coronal)
            self.mask_list.extend(ct_mask_name)
            self.label_list.extend(np.zeros(len(ct_name)).tolist())
            if self.use_clinical:
                ct_list = [df.loc[df['ID'] == int(patient), features].values[0] for patient in self.ct_list_negative]
                self.clinical_list.extend(ct_list)


    def __len__(self):
        return len(self.data_list)
        # return 50

    def __getitem__(self, idx):
        """get .mat file"""

        label = int(self.label_list[idx])


        img = Image.open(self.data_list[idx])
        img = np.asarray(img, dtype='float32')
        img = img / 255.0
        img = np.expand_dims(img, 0)
        img = np.repeat(img, 3, axis=0)
        image_axial = torch.from_numpy(img).float()

        if self.use_only_ct_coronal or self.use_fuse_axial_coronal:
            try:
                img = Image.open(self.data_list2[idx])
            except:
                print("stop here1: {}".format(self.data_list2[idx]))
            try:
                img = np.asarray(img, dtype='float32')
            except:
                print("stop here2: {}".format(self.data_list2[idx]))
            try:
                img = img / 255.0  # include norm
            except:
                print("stop here3: {}".format(self.data_list2[idx]))
            if img.ndim == 2:
                img = np.expand_dims(img, 0)
                img = np.repeat(img, 3, axis=0)
            image_coronal = torch.from_numpy(img).float()
        else:
            image_coronal = 0

        img_name = self.data_list[idx].split('/')[-1]
        # patient_name = img_name.split('_')[0] + '_' + img_name.split('_')[1]

        # clinical data
        if self.use_clinical:
            try:
                vectors = self.clinical_list[idx]
                vectors = vectors.astype("float32")
            except:
                print(patient_name)
        else:
            vectors = 0


        # img_path = self.data_list[idx]
        img_path = self.data_list[idx]

        # return img, img_mask, vectors, label, img_name, img_path,
        return image_axial, image_coronal, vectors, label, img_name, img_path
