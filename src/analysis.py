import os
from reproducibility import  set_all_seeds

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from scipy.stats import kruskal, chi2_contingency, fisher_exact, ttest_ind
from sklearn.model_selection import GroupShuffleSplit
import statsmodels.formula.api as smf

from config import *


def left_right_table():           
    datasets = {
        "Canon": canon_df,
        "Nikon secondary": secondary_df,
        "Nikon external": external_df
    }
    
    # Readable labels
    sex_map = {1: "Male", 2: "Female"}
    eye_map = {1: "Right", 2: "Left"}
    
    rows = []
    
    # Continuous variables
    for v, label in [
        ("patient_age", "Age, years"),
        ("diabetes_time_y", "Diabetes duration, years"),
    ]:
        rows.append(
            [label] +
            [f"{pd.to_numeric(d[v], errors='coerce').mean():.1f} ± "
             f"{pd.to_numeric(d[v], errors='coerce').std():.1f}"
             for d in datasets.values()]
        )
    
    # Sex
    for value, label in sex_map.items():
        rows.append(
            [f"Sex: {label}"] +
            [f"{(d['patient_sex'] == value).sum()} "
             f"({(d['patient_sex'] == value).mean()*100:.1f}\\%)"
             for d in datasets.values()]
        )
    
    # Insulin
    for value, label in [(1, "No"), (2, "Yes")]:
        rows.append(
            [f"Insulin: {label}"] +
            [f"{(d['insulin'] == value).sum()} "
             f"({(d['insulin'] == value).mean()*100:.1f}\\%)"
             for d in datasets.values()]
        )
    
    # Comorbidity status: simplify to presence/absence
    rows.append(
        ["Comorbidities: None"] +
        [f"{(d['comorbidities'].fillna('') == '0').sum()} "
         f"({(d['comorbidities'].fillna('') == '0').mean()*100:.1f}\\%)"
         for d in datasets.values()]
    )
    
    # Sample size
    rows.insert(
        0,
        ["Images, n"] + [f"{len(d)}" for d in datasets.values()]
    )
    
    table = pd.DataFrame(
        rows,
        columns=["Variable"] + list(datasets.keys())
    )
    
    latex = table.to_latex(
        index=False,
        escape=False,
        column_format="lccc",
        caption="Patient and cohort characteristics.",
        label="tab:cohort"
    )    
    printl()
    print(latex)
left_right_table()

def camera_resolution(df_, resolutions=None):
    df = df_.copy()
    d = df if resolutions is None else df[df.resolution.isin(resolutions)]
    rows = []
    for y in ["focus","illumination","image_field","artifacts"]:
        #d[y] = (d[y] == 2).astype(int)
        d.loc[:, y] = (d[y] == 2).astype(int)

        m = smf.logit(f"{y} ~ C(camera) + C(resolution)", data=d).fit(disp=False)
        rows.append([y.replace("_"," ").title(), f"{m.pvalues['C(camera)[T.NIKON NF5050]']:.3g}",
                  f"{m.pvalues.get('C(resolution)[T.2984x2304]', float('nan')):.3g}"])
    table=pd.DataFrame(rows, columns=["Image quality","Camera p","Resolution p"])
    print(table.to_latex(index=False,escape=False,column_format="lcccc"))
    return table

primary = camera_resolution(cfg_df)
secondary = camera_resolution(cfg_df, ["2672x2056","2984x2304"])

 
def characteristic_table():      
    datasets = {"Canon": canon_df, "Nikon secondary": secondary_df, "Nikon external": external_df}
    p = {k: v.drop_duplicates("patient_id") for k, v in datasets.items()}
    canon, ext = p["Canon"], p["Nikon external"]
    
    def fp(a,b): return fisher_exact([[a.sum(),(~a).sum()],[b.sum(),(~b).sum()]])[1]
    def tp(v): return ttest_ind(pd.to_numeric(canon[v],errors="coerce").dropna(), pd.to_numeric(ext[v],errors="coerce").dropna(), equal_var=False).pvalue
    
    rows = [
        ["Images, n"] + [len(v) for v in datasets.values()] + [""],
        ["Patients, n"] + [len(v) for v in p.values()] + [""],
    ]
    
    for v,label in [("patient_age","Age, years"),("diabetes_time_y","Diabetes duration, years")]:
        rows.append([label]+[f"{pd.to_numeric(d[v],errors='coerce').mean():.1f} ± {pd.to_numeric(d[v],errors='coerce').std():.1f}" for d in p.values()]+[f"{tp(v):.3g}"])
    
    for x,label in {1:"Male",2:"Female"}.items():
        masks=[d.patient_sex.eq(x) for d in p.values()]
        rows.append([f"Sex: {label}"]+[f"{m.sum()} ({m.mean()*100:.1f}\\%)" for m in masks]+[f"{fp(masks[0],masks[2]):.3g}"])
    
    for x in ["no","yes"]:
        masks=[d.insulin.astype(str).str.lower().eq(x) for d in p.values()]
        rows.append([f"Insulin use: {x.title()}"]+[f"{m.sum()} ({m.mean()*100:.1f}\\%)" for m in masks]+[f"{fp(masks[0],masks[2]):.3g}"])
    
    conditions=["diabetes","hypertension","dyslipidemia","hypothyroidism"]
    for c in conditions:
        masks=[d.comorbidities.fillna("").str.lower().str.contains(c,regex=False) for d in p.values()]
        rows.append([f"Comorbidities: {c.title()}"]+[f"{m.sum()} ({m.mean()*100:.1f}\\%)" for m in masks]+[f"{fp(masks[0],masks[2]):.3g}"])
    
    none=[d.comorbidities.fillna("").str.strip().eq("0") for d in p.values()]
    rows.append(["Comorbidities: None"]+[f"{m.sum()} ({m.mean()*100:.1f}\\%)" for m in none]+[f"{fp(none[0],none[2]):.3g}"])
    
    other=[d.comorbidities.fillna("").str.lower().apply(lambda x:x not in ("","0") and not any(c in x for c in conditions)) for d in p.values()]
    rows.append(["Comorbidities: Other"]+[f"{m.sum()} ({m.mean()*100:.1f}\\%)" for m in other]+[f"{fp(other[0],other[2]):.3g}"])
    
    table=pd.DataFrame(rows,columns=["Variable"]+list(datasets)+["Canon vs. external p"])
    printl()
    print(table.to_latex(index=False,escape=False,column_format="lcccc"))

characteristic_table()

