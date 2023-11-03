import numpy as np
import pandas as pd
import scipy.stats as stats
import datetime
import os
import matplotlib.pyplot as plt


dataset = pd.read_csv('./All_Patients.csv')
patient_info_all = pd.DataFrame(dataset)

patient_info_enrolled = patient_info_all[patient_info_all['SARS-CoV-2 nucleic acids']!='Negative']
# patient_info_all_enrolled = patient_info_all[(patient_info_all['SARS-CoV-2 nucleic acids']!='Negative') | (patient_info_all['Computed tomography (CT)']!='Positive')]
# patient_info_enrolled = patient_info_enrolled[(patient_info_enrolled['Mortality outcome']=='Cured') | (patient_info_enrolled['Mortality outcome']=='Deceased')]
# patient_info_enrolled = patient_info_enrolled[(patient_info_enrolled['Mortality outcome']=='Cured') | (patient_info_enrolled['Mortality outcome']=='Deceased')]


# preprocessing
features = dataset.keys().to_list()
for key in features:
    a = patient_info_all.loc[pd.isna(patient_info_all[key])]
    rate1 = len(a)/len(patient_info_all)
    # print("{} loss: {:.2f}%".format(key, rate1*100))
    b = patient_info_enrolled.loc[pd.isna(patient_info_enrolled[key])]
    rate2 = len(b) / len(patient_info_enrolled)
    print("{} loss: {:.2f}%/{:.2f}%".format(key, rate1*100, rate2*100))

# transform category to numerics and impute by average
features_category = ['Gender','Underlying diseases']
features_continuous = ['Age','Body temperature'] + features[10:]

for index in features_category + features_continuous:
    # temp_id = str(df_all['ID'][index]).zfill(4)
    # df_all['ID'][index] = temp_id
    # impute A&E admission:
    print(index)
    if index == 'Gender':
        patient_info_enrolled.loc[patient_info_enrolled['Gender'] == 'Male', 'Gender'] = 1
        patient_info_enrolled.loc[patient_info_enrolled['Gender'] == 'Female', 'Gender'] = 0
        patient_info_enrolled.loc[~patient_info_enrolled['Gender'].isin(['Female', 'Gender']), 'Gender'] = 0.5

    if index == 'Underlying diseases':
        patient_info_enrolled.loc[patient_info_enrolled[index] == 'No', index] = -1
        patient_info_enrolled.loc[pd.isna(patient_info_enrolled[index]), index] = 0
        patient_info_enrolled.loc[~patient_info_enrolled['Gender'].isin([-1, 0]), index] = 1

    else:
        try:
            # patient_info_enrolled_ch = patient_info_enrolled.loc[(~pd.isna(patient_info_enrolled[index])) & (patient_info_enrolled_ch[index].str.contains("<"))，index]
            patient_info_enrolled.loc[pd.isna(patient_info_enrolled[index]), index] = patient_info_enrolled[index].mean()
        except:
            print("here")


print("here")

patient_info_enrolled.to_csv('./patients_enrol_list/positive_all.csv')

patient_info_enrolled_died = patient_info_enrolled[patient_info_enrolled['Mortality outcome']=='Deceased']
patient_info_enrolled_survived = patient_info_enrolled[patient_info_enrolled['Mortality outcome']=='Cured']
patient_info_enrolled_mild = patient_info_enrolled[patient_info_enrolled['Morbidity outcome']=='Mild']
patient_info_enrolled_regular = patient_info_enrolled[patient_info_enrolled['Morbidity outcome']=='Regular']
patient_info_enrolled_severe = patient_info_enrolled[patient_info_enrolled['Morbidity outcome']=='Severe']
patient_info_enrolled_criticallyill = patient_info_enrolled[patient_info_enrolled['Morbidity outcome']=='Critically ill']

# print("Death1: {}, Survived0: {}".format(len(patient_info_enrolled_died),len(patient_info_enrolled_survived)))
#
patient_info_enrolled_died.to_csv('./patients_enrol_list/enrolled_died.csv')
patient_info_enrolled_survived.to_csv('./patients_enrol_list/enrolled_survived.csv')
patient_info_enrolled_mild.to_csv('./patients_enrol_list/enrolled_mild.csv')
patient_info_enrolled_regular.to_csv('./patients_enrol_list/enrolled_regular.csv')
patient_info_enrolled_severe.to_csv('./patients_enrol_list/enrolled_severe.csv')
patient_info_enrolled_criticallyill.to_csv('./patients_enrol_list/enrolled_criticallyill.csv')



