import os,sys

from pathlib import Path
import pandas as pd
import numpy as np

from config import *#SEED,set_all_seeds
set_all_seeds( SEED )

from scipy.stats import kruskal, chi2_contingency, fisher_exact, ttest_ind
from sklearn.model_selection import GroupShuffleSplit

USERID = os.environ['LOGNAME']
USERID =  f'def-{USERID}-ab' 

CFP_DATA_ROOT = f'/project/{USERID}/physionet.org/files/brazilian-ophthalmological/1.0.2'
CFP_CSV_PATH = os.path.join(CFP_DATA_ROOT, "label_brset.csv")
CFP_IMAGE_DIR = os.path.join(CFP_DATA_ROOT, "fundus_photos")

def image_exists( filename):    
    filename =  f'{CFP_DATA_ROOT}/fundus_photos/{filename}.jpg'      
    if not isinstance(filename, str):
        return False
    return Path(filename).is_file()       
    
def get_cfp():    
    cfp_df = pd.read_csv(CFP_CSV_PATH)
    
    df =  pd.read_csv( CFP_DATA_ROOT + '/fundus_photos/' + '/resolutions.txt', names=['image_path', 'resolution'], delimiter=' ')        
    df['image_path'] = (
        df['image_path']
        .str.removeprefix('./')
        .str.replace(r'\.(jpg|jpeg)$', '', regex=True, case=False)
    ) 
    df = cfp_df.merge(
        df,
        left_on='image_id',
        right_on='image_path',
        how='left'
    )    
    df['file_exists'] = df['image_path'].apply( lambda x: image_exists( x))      
    df.drop(columns="image_path",inplace=True,)    
    df["age_group"] = pd.cut(df['patient_age'],bins=[-np.inf,39,49,59,69,79,np.inf],labels=["<40","40-49","50-59","60-69","70-79","80+"])                
    df2= df[df.file_exists].copy()
    df2.reset_index(inplace=True)
    return df2, df

if os.path.exists('cfp_exist.csv'):
    cfp_df, full_df = pd.read_csv('cfp_exist.csv'), pd.read_csv('cfp_full.csv')
else:    
    cfp_df, full_df = get_cfp()
    cfp_df.to_csv('cfp_exist.csv')
    full_df.to_csv('cfp_full.csv') 


def define_splits( cfp_df ):
    quality_vars = ["focus", "illumination", "image_field", "artifacts"]
    
    # --- Data ---
    rng = np.random.default_rng(SEED)
    canon_patients = set(cfp_df.loc[cfp_df.camera == "Canon CR", "patient_id"])
    nikon = cfp_df[(cfp_df.camera == "NIKON NF5050") & ~cfp_df.patient_id.isin(canon_patients)].copy()
    
    secondary_patients, external_patients = set(), set()
    for r in nikon.resolution.dropna().unique():
        p = nikon.loc[nikon.resolution == r, "patient_id"].unique()
        rng.shuffle(p); n = len(p) // 2
        secondary_patients.update(p[:n]); external_patients.update(p[n:])
    
    secondary_df = nikon[nikon.patient_id.isin(secondary_patients)]
    external_df = nikon[nikon.patient_id.isin(external_patients)]
    canon_df = cfp_df[cfp_df.patient_id.isin(canon_patients)]
    
    df = pd.concat([ canon_df.assign(dataset="Canon"),    external_df.assign(dataset="Nikon")], ignore_index=True)
    
    # --- Descriptive: camera ---
    print(df.groupby("camera")[quality_vars].agg(["mean", "std", "median"]).round(3))
    
    # --- Descriptive: Nikon resolution ---
    print(external_df.groupby("resolution")[quality_vars].agg(["count", "mean", "std", "median"]).round(3))
    
    # --- Statistical test: resolution within Nikon ---
    for v in quality_vars:
        g = [x[v].dropna() for _, x in nikon.groupby("resolution")]
        try:
            print(v, kruskal(*g))   
        except:
            pass
     
    
    # --- Camera × resolution ---
    print(df.groupby(["camera", "resolution"])[quality_vars].mean().round(3))
 
    abnormal_rate = (
        df[quality_vars]
        .eq(2)
        .groupby(df["camera"])
        .mean()
        .mul(100)
        .round(2)
    )    
    print(abnormal_rate)     
    
    rows = []    
    for v in quality_vars:
        t = pd.crosstab(df.camera, df[v]).reindex(columns=[1, 2], fill_value=0)
        a, b, c, d = t.loc["NIKON NF5050", 2], t.loc["NIKON NF5050", 1], t.loc["Canon CR", 2], t.loc["Canon CR", 1]
        or_, p_fisher = fisher_exact([[a, b], [c, d]])
        chi2, p, _, _ = chi2_contingency(t)
        V = np.sqrt(chi2 / t.to_numpy().sum())
        rows.append([v.replace("_", " ").title(), f"{100*a/(a+b):.2f}", f"{100*c/(c+d):.2f}", f"{or_:.2f}", f"{p_fisher:.4f}", f"{V:.3f}"])
    
    results = pd.DataFrame(rows, columns=["Quality", "Nikon abnormal (%)", "Canon abnormal (%)", "OR", "p", "Cramer's V"])
    printl()
    print('Results in latex:',results.to_latex(index=False, escape=False))

    return canon_df, secondary_df, external_df, df


#
#  Step 1
#
canon_df, secondary_df, external_df, cfg_df = define_splits( cfp_df )
