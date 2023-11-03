from torch import true_divide
from trainer import train
import numpy as np
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process arguments')
    parser.add_argument('--dataset', default='BRCA', choices=['BRCA', 'ROSMAP'], type=str, help='Choice of dataset')
    args = parser.parse_args()
    
    data_folder = f'../datasets/{args.dataset}/'

    testonly = False
    modelpath = f'./{args.dataset}_experiments/'
    num_perform = 20

    results = {}
    for testmodel in ['dynamic', 'DMIB']: # 'dynamic'
        # testmodel = 'dynamic'
        # testmodel = 'DMIB'
        # testmodel = 'DMIB_ablation_IBsided'
        # testmodel = 'DMIB_ablation_IB'
        # testmodel = 'DMIB_ablation_directfuse'
        # testmodel = 'DMIB_ablation_sided'
        print(testmodel)
        
        list_m1 = []
        list_m2 = []
        list_m3 = []
        for i in range(num_perform):
            m1, m2, m3 = train(data_folder, modelpath, testonly, testmodel)
            list_m1.append(m1)
            list_m2.append(m2)
            list_m3.append(m3)
            print(m1, m2, m3)

        m1_mean = np.mean(list_m1)
        m1_std  = np.std(list_m1)
        m1_var  = np.var(list_m1)

        m2_mean = np.mean(list_m2)
        m2_std = np.std(list_m2)
        m2_var = np.var(list_m2)

        m3_mean = np.mean(list_m3)
        m3_std = np.std(list_m3)
        m3_var = np.var(list_m3)

        if 'BRCA' in data_folder:
            print("Method {}: ACC={:.1f}+{:.1f}, WeightedF1={:.1f}+{:.1f}, MacroF1={:.1f}+{:.1f},".format(testmodel, m1_mean * 100, m1_std * 100,
                                                                                   m2_mean * 100, m2_std * 100,
                                                                                   m3_mean * 100, m3_std * 100))
        elif 'ROSMAP' in data_folder:
            print("Method {}: ACC={:.1f}+{:.1f}, F1={:.1f}+{:.1f}, AUC={:.1f}+{:.1f},".format(testmodel, m1_mean * 100, m1_std * 100,
                                                                                   m2_mean * 100, m2_std * 100,
                                                                                   m3_mean * 100, m3_std * 100))

        results[testmodel + '_m1'] = list_m1
        results[testmodel + '_m2'] = list_m2
        results[testmodel + '_m3'] = list_m3

print("here")
# np.save('BRCA.npy', dict)
# dict_load = np.load('loaddict.npy', allow_pickle=True)
# print("dict =", dict_load.item())
# print("dict['a'] =", dict_load.item()['a'])

import scipy.stats as stats
stat_val_1, p_val_1 = stats.ttest_ind(results['dynamic_m1'], results['DMIB_m1'], equal_var=False)
stat_val_2, p_val_2 = stats.ttest_ind(results['dynamic_m2'], results['DMIB_m2'], equal_var=False)
stat_val_3, p_val_3 = stats.ttest_ind(results['dynamic_m3'], results['DMIB_m3'], equal_var=False)

# significant test
print(p_val_1, p_val_2, p_val_3)
for i in ['dynamic','DMIB_onlyIB']:
    testmodel = i
    m1_mean = np.mean(results[testmodel+'_m1'])
    m1_std  = np.std(results[testmodel+'_m1'])
    m1_var  = np.var(results[testmodel+'_m1'])

    m2_mean = np.mean(results[testmodel+'_m2'])
    m2_std  = np.std(results[testmodel+'_m2'])
    m2_var  = np.var(results[testmodel+'_m2'])

    m3_mean = np.mean(results[testmodel+'_m3'])
    m3_std  = np.std(results[testmodel+'_m3'])
    m3_var  = np.var(results[testmodel+'_m3'])
    if 'BRCA' in data_folder:
        print("Method {}: ACC={:.1f}+{:.1f}, WeightedF1={:.1f}+{:.1f}, MacroF1={:.1f}+{:.1f},".format(testmodel, m1_mean * 100, m1_std * 100,
                                                                               m2_mean * 100, m2_std * 100,
                                                                               m3_mean * 100, m3_std * 100))
    elif 'ROSMAP' in data_folder:
        print("Method {}: ACC={:.1f}+{:.1f}, F1={:.1f}+{:.1f}, AUC={:.1f}+{:.1f},".format(testmodel, m1_mean * 100, m1_std * 100,
                                                                               m2_mean * 100, m2_std * 100,
                                                                               m3_mean * 100, m3_std * 100))