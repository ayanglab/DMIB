#f_18:
python main_covid.py --GC_mode weighted-cosine --MF_mode concat --MP_mode GCN --dropout 0.35 --lr 0.0017 --mode simple-2 --n_head 4 --n_hidden 28 --nlayer 1 --reg 0.22 --datname ITAC

#f_12:
python main_covid.py --GC_mode weighted-cosine --MF_mode concat --MP_mode GCN --dropout 0.35 --lr 0.0017 --mode simple-2 --n_head 4 --n_hidden 28 --nlayer 1 --reg 0.22 \
--datname ITAC --clinical_continuous="Age,A&E_Respiratory rate,A&E_Oxygen saturation,Platelets,D-Dimer,Glucose,Urea,eGFR,GOT,PCR,ABG_pO2,ABG_measured saturation O2 " --clinical_category=""

#f_7:
python main_covid.py --GC_mode weighted-cosine --MF_mode concat --MP_mode GCN --dropout 0.35 --lr 0.0017 --mode simple-2 --n_head 4 --n_hidden 28 --nlayer 1 --reg 0.22 \
--datname ITAC --clinical_continuous="Age,A&E_Oxygen saturation,Platelets,ABG_measured saturation O2 ,A&E_Respiratory rate,ABG_pO2,D-Dimer" --clinical_category=""

#f_1:
python main_covid.py --GC_mode weighted-cosine --MF_mode concat --MP_mode GCN --dropout 0.35 --lr 0.0017 --mode simple-2 --n_head 4 --n_hidden 28 --nlayer 1 --reg 0.22 \
--datname ITAC --clinical_continuous="Age" --clinical_category=""
