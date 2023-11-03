# Overview
We include our Dynamic Multimodal Information Bottlneck (DMIB) codebase for the following:
- [prognosis tasks](./prognosis_tasks)
    + on our inhouse ITAC dataset: we cannot open-sourced the ITAC dataset due to data compliance and privacy concerns
    + on the [iCTCF](https://ngdc.cncb.ac.cn/ictcf/) public Covid19 dataset
- [diagnosis tasks](./diagnosis_tasks)
    + on the [BRCA](https://github.com/txWang/MOGONET/tree/main/BRCA) and [ROSMAP](https://github.com/txWang/MOGONET/tree/main/ROSMAP) datasets

# Quick start
## Prognosis Task for ITAC
```
cd prognosis_tasks
python train_fuse_covid.py --use_fuse_clinical_axial --use_clinical --preprocess_clinical=2 --model_name="proposed_crossmodal" --use_IB --expname="fuse_crossmodal"
```
Rundown of the arguments:
- `--use_fuse_clinical_axial`: denotes fusion model
- `--use_fuse_clinical_axial` `--preprocess_clinical=2`: load + preprocess clinical data
- `model_name="proposed_crossmodal"`: select our proposed DMIB fusion model
    + other choice of fusion models include `concat`, `attention`, `transformer`, `dynamic`
- `--use_IB`: include the information bottleneck module
- `--expname="fuse_crossmodal"`: experiment name

## Prognosis Task for iCTCF
First run the following scripts to download and preprocess the iCTCF dataset (2D montage generation)
```
cd dataset/iCTCF
python no1_patient_enrollment.py
python no2_slice_to_3D.py
python no3_generate_montage.py
```

After the dataset preprocessing has been done, train the DMIB model with
```
cd prognosis_tasks
python train_fuse_crossmodal.py --use_fuse_clinical_axial --use_clinical --preprocess_clinical=2 --model_name="proposed_crossmodal" --clinical_continuous="Age,Body temperature,MCHC,MCH,MCV,HCT,HGB,RBC,PDW,PLCT,MPV,PLT,BA,EO,MO,LY,NE,BAP,EOP,MOP,LYP,NEP,WBC,PLCR,RDWSD,RDWCV,ESR,CRP,PCT,ALG,ALB,ALP,ALT,AST,BUN,CA,CL,CO2,CREA,GGT,GLB,K,MG,Na,PHOS,TBIL,TP,URIC,CHOL,CK,HDLC,LDH,TG,AnG,DBIL,GLU,LDLC,OSM,PA,TBA,HBDH,CysC,LAP,5NT,HC,SAA,SdLDL,CD3+,CD4+,CD8+,BC,NKC,CD4/CD8,IL-2,IL-4,IL-6,IL-10,TNF,IFN" --clinical_category="Gender,Underlying diseases" --patient_died_ct_csv="../dataset/iCTCF/patients_enrol_list/enrolled_1.csv" --patient_survived_ct_csv="../dataset/iCTCF/patients_enrol_list/enrolled_0.csv" --patiens_info_csv="../dataset/iCTCF/patients_enrol_list/enrolled_all.csv" --data_clinical="../dataset/iCTCF/patients_enrol_list/enrolled_all.csv" --datapath_train="../dataset/iCTCF/2D_montage/" --datapath_test="../dataset/iCTCF/2D_montage/" --use_IB --expname="iCTCF_all"
```

## Prognosis Task Comparisons
- Concatentation `--model_name="concat"`
- Attention `--model_name="attention"`
- Dynamic weighting `--model_name="dynamic"`
- Subspace projection (with cosine similarity) `--use_similarity`
- Transformers `--model_name="transformer"`
- Graph-based ([MMGL](https://github.com/SsGood/MMGL)
```
cd prognosis_tasks/MMGL
python main_covid.py --GC_mode weighted-cosine --MF_mode concat --MP_mode GCN --dropout 0.35 --lr 0.0017 --mode simple-2 --n_head 4 --n_hidden 28 --nlayer 1 --reg 0.22 --datname ITAC
```

## Diagnosis Task for BRCA, ROSMAP
```
cd diagnosis_tasks
python train_fuse.py --dataset BRCA
python train_fuse.py --dataset ROSMAP
```

# Codebase Structure
```
└── dataset
    └── BRCA
    └── ROSMAP
    └── ITAC (not included at the moment)
    └── iCTCF
└── diagnosis_tasks
    ├── model.py (includes DMIB + ablation models)
    ├── train_fuse.py (script for DMIB training and testing)
    ├── trainer.py (training utils)
└── prognosis_tasks
    └── loss
        ├── loss_auc.py (differentiable ROC AUC loss)
        ├── loss_conf.py (dynamic confidence loss)
        ├── loss_focal.py (focal loss)
        ├── loss_IB.py (our DMIB loss)
        ├── loss_sinkhorn.py (entropy-regularized optimal transport aka sinkhorn loss)
    └── model
        ├── clinical_only.py (clinical only model)
        ├── fusion_attention.py (attention-based fusion model)
        ├── fusion_concat.py (focal loss)
        ├── fusion_dynamic.py (dynamic fusion model)
        ├── fusion_crossmodal.py (our proposed DMIB fusion model)
        ├── fusion_transformer.py (transformer-based fusion model)
    └── utils
        ├── data.py (utils for preprocessing CT images)
        ├── img_process.py (utils for processing CT images)
        ├── trainer.py (training utils)
    ├── select_dataloader.py (dataloader for CT image only / clinical data only / fused)
    ├── select_model.py (initialize models according to experimental configurations)
    ├── select_optimizer.py (select optimizer according to experimental configurations)
    ├── select_parameters.py (experimental configurations)
    ├── train_fuse_covid.py (script for DMIB training and testing)
    ├── train_fuse_covid_ML.py (script for DMIB training and testing with ML methods, e.g. XGBoost, Random Forests, SVMs)
```
