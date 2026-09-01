import os
import numpy as np
import pandas as pd
import torch
import timm

import importlib

import datasets, config
importlib.reload(datasets)
importlib.reload(config)

from config import *
from torch.utils.data import DataLoader
from datasets import ResizePad, BRSETDataset
from config import DEVICE, BINARY_METRICS, ORDINAL_METRICS
from evaluation import run_subgroup_bootstrap

#from torch.utils.data import DataLoader
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, cohen_kappa_score, mean_absolute_error
)

from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import roc_curve

RFG_PATH=f"/project/{USERID}/HowRU/retinal/assets/retfoundgreen_statedict.pth"

def encoder( checkpoint_path ):
    model = timm.create_model( "vit_small_patch14_reg4_dinov2", img_size=(392, 392), num_classes=0, checkpoint_path=checkpoint_path, dynamic_img_size=True).to(DEVICE)
    model.global_pool = "avg"
    model.eval()
    return model

def split_development():        
    gss = GroupShuffleSplit(n_splits=1, test_size=0.3, random_state=42)
    trn_idx, val_idx = next(gss.split(canon_df, groups=canon_df['patient_id']))    
    
    trn_df = canon_df.iloc[trn_idx].copy()
    val_df0 = canon_df.iloc[val_idx].copy()
    
    tst_idx, val_idx = next(gss.split(val_df0, groups=val_df0['patient_id']))
    tst_df = val_df0.iloc[tst_idx].copy()
    val_df = val_df0.iloc[val_idx].copy()
    return trn_df,val_df,tst_df

if ( 'rfg_model' in globals())==False:
    DFs={}
    print( 'Splitting dataframes')    
    DFs['trn'],DFs['val'],DFs['tst'] = split_development()
    for  k in DFs.keys():
        print( DFs[k].shape )

    DFs['external'] = external_df
    DFs['secondary'] = secondary_df
    
    print( 'loading model')
    rfg_model  = encoder( RFG_PATH )

# 1890
# ResizePad(784)   # 56 × 56 x 14px patches
# ResizePad(1120)  # 80 × 80 x 14px patches    
if ( 'feats' in globals())==False:          
    feats = {}  
    for res in [R]:
        my_transform = ResizePad(res)
        out_dir = "../features"
        os.makedirs(out_dir, exist_ok=True)    
        for k in DFs:
            path = os.path.join(out_dir, f"{res}_{k}_features.npz")    
            try:
                feats[res, k] = np.load(path)["features"]
            except FileNotFoundError:
                ds = BRSETDataset(
                    DFs[k], CFP_IMAGE_DIR, file_format="jpg", transform=my_transform
                )
                loader = DataLoader( ds, batch_size=32, shuffle=False, num_workers=8, pin_memory=True)

                with timer(f"Getting features for {k} at res={res}"):
                    with torch.inference_mode():                       
                        feats[res, k] = np.concatenate([
                            rfg_model(x.to(DEVICE, non_blocking=True)).cpu().numpy()
                            for x, _ in loader
                        ])
                with timer(f"Saving features for {k} at res={res}"):                        
                    np.savez_compressed(path, features=feats[res, k])


'''
Target	Multiple of 14	Relative to native
2,660	190	Very close to 2672
2,688	192	Very close to 2672
2,912	208	Close to 2984
2,982	213	Essentially exact to 2984

1890 = 14 x 135
2688, 2982
'''





'''
    diabetic_retinopathy- 1 present and 0 absent.
    macular_edema- 1 present and 0 absent.
    scar - 1 present and 0 absent
    nevus - 1 present and 0 absent.
    amd - 1 present and 0 absent.
    vascular_occlusion- 1 present and 0 absent.
    hypertensive_retinopathy - 1 present and 0 absent.
    drusens - 1 present and 0 absent.
    hemorrhage - 1 present and 0 absent.
    retinal_detachment - 1 present and 0 absent.
    myopic_fundus - 1 present and 0 absent.
    increased_cup_disc - 1 present and 0 absent.
    other - 1 present and 0 absent.
'''

tasks = ['scar', 'nevus', 'amd', 'vascular_occlusion',
       'hypertensive_retinopathy', 'drusen', 'hemorrhage',
       'retinal_detachment', 'myopic_fundus', 'increased_cup_disc', 'other']

def fit(name,X,y):   
    if name=="LR":return LogisticRegression(max_iter=2000,class_weight="balanced").fit(X,y)
    if name=="RF":return RandomForestClassifier(n_estimators=500,max_depth=None, min_samples_leaf=1, class_weight="balanced_subsample",random_state=SEED,n_jobs=-1).fit(X,y)    
    if name=="SVM":return CalibratedClassifierCV(LinearSVC(class_weight="balanced",random_state= SEED,max_iter=5000)).fit(X,y)
    if name=="kNN":return KNeighborsClassifier(n_neighbors=15,weights="distance",n_jobs=-1).fit(X,y)                             
    if name=="HGB":return HistGradientBoostingClassifier(class_weight="balanced",random_state=SEED).fit(X,y)                         
    return MLPClassifier(MLP_PARAMS).fit(X,y)

preds = []
for alg in [ 'HGB','RF','SVM','kNN']:
    for task in tasks:                
        for res in [R]:            
            y = DFs["trn"][task].to_numpy()
            ok = pd.notna(y)
            with timer(f"Fitting {alg} at res={res}"):      
                model = fit( alg, feats[res,"trn"][ok], y[ok].astype(int))
    
            pv = model.predict_proba(feats[res,"val"])[:,1]
            yv = DFs["val"][task].to_numpy()
            fpr,tpr,t = roc_curve(yv[~pd.isna(yv)], pv[~pd.isna(yv)])
            threshold = t[np.argmax(tpr-fpr)]
    
            for k in ["external","secondary","tst"]:
                p = model.predict_proba( feats[res,k])[:,1]
                preds.append( pd.DataFrame({
                    "image_id": DFs[k].image_id, "dataset": k, "task": task,
                    "classifier": alg, "resolution": res,
                    "y": DFs[ k ][task], "prob": p, "threshold": threshold
                }))
preds_df = pd.concat(preds, ignore_index=True)
preds_df.to_csv( f"../features/{res}_{alg}_preds.csv.gz", index=False, compression="gzip")
preds_df = preds_df.merge(
    full_df[["image_id", "patient_sex", "age_group"]].drop_duplicates("image_id"),
    on="image_id",
    how="left",
    validate="many_to_one",
)

res_df = run_subgroup_bootstrap(  preds_df=preds_df,  resolutions=[R], n_bootstrap=100 , ci_level=.95) #cfg.ci_level

printl()
print( alg )
print(res_df[
    (res_df["metric"] == "AUROC") &
    (res_df["subgroup"] == "overall")].sort_values(by=['task','dataset'])
)
res_df.to_csv( f'../features/{res}_res.csv' )    

np.savez_compressed( f'../features/{res}_TIMINGS.npz', TIMINGS )    


# Encoder no longer needed: all downstream models use fixed features
if 0:
    del rfg_model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    
