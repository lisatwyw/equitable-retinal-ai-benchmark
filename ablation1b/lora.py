import os,copy,random,argparse,re
import numpy as np,pandas as pd
from PIL import Image
import torch,torch.nn as nn,torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import functional as TF
from sklearn.model_selection import GroupKFold     
import timm
from torch.optim.lr_scheduler import CosineAnnealingLR

from sklearn.metrics import roc_auc_score,average_precision_score,accuracy_score,balanced_accuracy_score,precision_score,recall_score,f1_score,confusion_matrix

os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Forces the environment to only see 1 T4

num_workers = max(1, os.cpu_count() - 1) 
print( num_workers ,'workers available for batching')

DEVICE=torch.device("cuda" if torch.cuda.is_available() else "cpu")
pin_mem = DEVICE.type == "cuda"
_aff = getattr(os, "sched_getaffinity", None)  # not available on macOS / Windows

KAGGLE = os.path.exists("/kaggle") # ================== important !! ==================
USERID = f"def-{os.environ.get('USER', '').lower()}-ab"

TK='A'
folder_path = '/kaggle/input/datasets/andrewmvd/ocular-disease-recognition-odir5k/ODIR-5K/ODIR-5K/Training Images/' if KAGGLE else '/project/6088123/ODIR-5K/ODIR-5K/trn/'

odir_file = '/kaggle/input/datasets/andrewmvd/ocular-disease-recognition-odir5k/full_df.csv' if KAGGLE else f'/project/def-{USERID}-ab/ODIR-5K/ODIR-5K/data.xlsx';
odir_df = pd.read_csv( odir_file )

def set_seed(s=42):
    random.seed(s);np.random.seed(s);torch.manual_seed(s);torch.cuda.manual_seed_all(s)
    torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False

def preprocess(img,res=392,threshold=5,pad_percent=.025):
    a=np.array(img.convert("RGB")); m=args.mean(-1); mx=m.max()
    if mx<=0: raise ValueError("Empty image")
    m=m/mx*255.;m-=np.quantile(m,.05)
    xs=np.where(m.mean(0)>threshold)[0];ys=np.where(m.mean(1)>threshold)[0]
    if len(xs)==0 or len(ys)==0: raise ValueError("Could not detect image boundary")
    l,r,t,b=xs.min(),xs.max(),ys.min(),ys.max()
    buf=int((args.shape[0]+args.shape[1])/2*pad_percent)
    l=max(0,l-buf);r=min(args.shape[1],r+buf);t=max(0,t-buf);b=min(args.shape[0],b+buf)
    img=img.crop((l,t,r,b));w,h=img.size
    if w>h: p=(0,(w-h)//2,0,(w-h+1)//2)
    else: p=((h-w)//2,0,(h-w+1)//2,0)
    return img if w==h and res is None else img.resize((res,res),Image.Resampling.LANCZOS) if w==h else TF.pad(img,p,fill=0).resize((res,res),Image.Resampling.LANCZOS)

def metrics(y,p,t=.5):
    pred=(p>=t).astype(int);tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {"AUROC":roc_auc_score(y,p),"AUPRC":average_precision_score(y,p),"accuracy":accuracy_score(y,pred),
            "balanced_accuracy":balanced_accuracy_score(y,pred),"sensitivity":recall_score(y,pred,zero_division=0),
            "specificity":tn/(tn+fp) if tn+fp else np.nan,"precision":precision_score(y,pred,zero_division=0),
            "F1":f1_score(y,pred,zero_division=0)}
    
class QVLoRA(nn.Module):
    def __init__(self,qkv,rank=8,alpha=16,drop=.05):
        super().__init__();self.original=qkv;self.scale=alpha/rank;self.drop=nn.Dropout(drop)
        d=qkv.out_features//3;self.d=d
        self.qA=nn.Parameter(torch.empty(rank,qkv.in_features));self.qB=nn.Parameter(torch.zeros(d,rank))
        self.vA=nn.Parameter(torch.empty(rank,qkv.in_features));self.vB=nn.Parameter(torch.zeros(d,rank))
        nn.init.kaiming_uniform_(self.qA,a=np.sqrt(5));nn.init.kaiming_uniform_(self.vA,a=np.sqrt(5))
        for p in qkv.parameters(): p.requires_grad=False
    def forward(self,x):
        z=self.original(x);u=self.drop(x)
        dq=(u@self.qargs.T@self.qB.T)*self.scale;dv=(u@self.vargs.T@self.vB.T)*self.scale
        q,k,v=z[...,:self.d],z[...,self.d:2*self.d],z[...,2*self.d:]
        return torch.cat((q+dq,k,v+dv),-1)        
    
class Model(nn.Module):
    def __init__(self,ckpt,rank=8,alpha=16,drop=.05,RES=392):
        super().__init__()
        self.backbone=timm.create_model("vit_small_patch14_reg4_dinov2",img_size=(RES,RES),dynamic_img_size=True,num_classes=0)    
        
        checkpoint = torch.load(ckpt, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
    
        if (RES != 392)& ("pos_embed" in state_dict):
            pos_embed = state_dict["pos_embed"]
            B, N, C = pos_embed.shape;  old_grid = int(N ** 0.5)
            new_grid = self.backbone.patch_embed.grid_size[0]
    
            pos_embed = pos_embed.reshape(B, old_grid, old_grid, C).permute(0, 3, 1, 2)
            pos_embed = F.interpolate(pos_embed, size=(new_grid, new_grid), mode="bicubic", align_corners=False)
            pos_embed = pos_embed.permute(0, 2, 3, 1).reshape(B, new_grid * new_grid, C)    
            state_dict["pos_embed"] = pos_embed    
            print( 'Bicubic interpolation of weights')
        self.backbone.load_state_dict(state_dict, strict=False)
        
        self.backbone.global_pool="avg"
        for p in self.backbone.parameters(): p.requires_grad=False 
        for b in self.backbone.blocks: b.attn.qkv=QVLoRA(b.attn.qkv,rank,alpha,drop)
        self.head=nn.Linear(self.backbone.num_features,1)
        
    def forward(self,x): return self.head(self.backbone(x)).squeeze(1)

ap=argparse.ArgumentParser()
checkpoint = 'retfoundgreen_statedict.pth' if KAGGLE else '/project/6088123/retfoundgreen_statedict.pth'
ap.add_argument("--EP",type=int,default=1);
ap.add_argument("--BS",type=int,default=8);
ap.add_argument("--LR",type=float,default=3e-4)
ap.add_argument("--WD",type=float,default=1e-4);
ap.add_argument("--lo_rank",type=int,default=8);
ap.add_argument("--lo_alpha",type=int,default=16)
ap.add_argument("--lo_DO",type=float,default=.05);
ap.add_argument("--seed",type=int,default=42)
ap.add_argument("--RES",type=int,default=392);
a=ap.parse_known_args()[0] if KAGGLE else ap.parse_args(); 

set_seed(args.seed)

# ==================
# Initialize model
# ===================
model=Model(checkpoint,args.lo_rank,args.lo_alpha,args.lo_DO,args.RES)


prefix = "_".join(f"{k}{v}" for k, v in vars(a).items())
pref = re.sub(r"[^A-Za-z0-9_.=-]", "", prefix)
print( 'Trial parameters:')
vals = list(vars(a).values())
[ print(f'{a}={b}') for a,b in zip(vars(a),vals)] 

def split_dataset_by_patient(df, F=0, seed=42):
    df = df.sample(frac=1, random_state=seed).reset_index(drop=True)
    
    gkf = GroupKFold(n_splits=6)
    tv_idx, te_idx = next(gkf.split(df, groups=df["patient_id"]))

    for fold, (tv_idx, te_idx) in enumerate( gkf.split(df, groups=df["patient_id"])):
        if fold == F:  # second fold, since indexing starts at 0
            break    
    tv = df.iloc[tv_idx].reset_index(drop=True)
    te = df.iloc[te_idx].reset_index(drop=True)

    gkf = GroupKFold(n_splits=5)    
    for fold, (tr_idx, va_idx) in enumerate( gkf.split(tv, groups=tv["patient_id"])):
        if fold == F:
            break         
    tr = tv.iloc[tr_idx].reset_index(drop=True)
    va = tv.iloc[va_idx].reset_index(drop=True)
    n = len(df)
    
    print(f"\n[SPLIT seed={seed}] train={len(tr)} ({len(tr) / n:.3f}) val={len(va)} ({len(va) / n:.3f}) "
          f"test={len(te)} ({len(te) / n:.3f})")

    tr_patients = set(tr["patient_id"])
    va_patients = set(va["patient_id"])
    te_patients = set(te["patient_id"])    
    assert tr_patients.isdisjoint(va_patients), "Patient overlap: train/val"
    assert tr_patients.isdisjoint(te_patients), "Patient overlap: train/test"
    assert va_patients.isdisjoint(te_patients), "Patient overlap: val/test"    
    print("No patient overlap ✓")
    return tr, va, te

# ==================
# Split DEV dataset
# ===================
tr_df,va_df,te_df=split_dataset_by_patient( odir_df.rename( { 'ID':'patient_id'} ,axis=1), F, args.seed )

class DS(Dataset):
    def __init__(self,df, folder_path, img_key,label_key,train=False):
        self.df=df;self.train=train
        self.folder_path = folder_path
        self.label_key=label_key;self.img_key=img_key
        assert {img_key,label_key}<=set(self.df.columns)
    def __len__(self): return len(self.df)
    def __getitem__(self,i):
        r=self.df.iloc[i];img=preprocess(Image.open( self.folder_path + r[self.img_key]).convert("RGB"))
        if self.train and random.random()<.5: img=TF.hflip(img)
        # x=torch.from_numpy(np.asarray(img)).permute(2,0,1).float()/255.  # PIL will make it read-only
        #x=torch.from_numpy(np.asarray(img).copy()).permute(2,0,1).float()/255. # less compact
        x=torch.tensor(np.asarray(img)).permute(2,0,1).float()/255.
        x=(x-.5)/.5        
        return x,torch.tensor(float(r[self.label_key]))
        
tr=DataLoader(DS(tr_df,folder_path,img_key='Left-Fundus',label_key=TK,train=True), args.BS, shuffle=True, num_workers=num_workers, pin_memory=DEVICE.type=="cuda")
va=DataLoader(DS(va_df,folder_path,img_key='Left-Fundus',label_key=TK,train=False),args.BS, shuffle=True, num_workers=num_workers, pin_memory=DEVICE.type=="cuda")
te=DataLoader(DS(te_df,folder_path,img_key='Left-Fundus',label_key=TK,train=False),args.BS, shuffle=True, num_workers=num_workers, pin_memory=DEVICE.type=="cuda")

n=sum(p.numel() for p in model.parameters());nt=sum(p.numel() for p in model.parameters() if p.requires_grad)

print(f"Device={DEVICE} | params={n:,} | trainable={nt:,} ({100*nt/n:.3f}%)")
trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
total = sum(p.numel() for p in model.parameters())

print(f"Trainable: {trainable:,}")
print(f"Total:     {total:,}")
print(f"Percent:   {100 * trainable / total:.2f}%")

lossfn=nn.BCEWithLogitsLoss()
opt=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),lr=args.LR,weight_decay=args.WD)

#sch=torch.optim.lr_scheduler.CosineAnnealingLR(opt,args.epochs);
# Total epochs (e.g., 30-40 is standard for RETFound fine-tuning)
total_epochs = args.EP
warmup_epochs = args.EP*.2//1
print(total_epochs, 'total epochs |',  warmup_epochs, 'warmup epochs' )
sch = CosineAnnealingLR(opt, T_max=total_epochs - warmup_epochs, eta_min=1e-6)

best=-1
patience = 7
epochs_without_improvement = 0

output=f"./results/"
os.makedirs(os.path.dirname(output), exist_ok=True)

def run_epoch(model,loader,opt,lossfn,DEVICE,train=True):
    model.train(train);losses=[];ys=[];ps=[]
    for x,y in loader:
        x,y=x.to(DEVICE),y.to(DEVICE); 
        if train: opt.zero_grad(set_to_none=True)
        z=model(x);loss=lossfn(z,y)
        if train: loss.backward();opt.step()
        losses.append(loss.item()*len(y));ys.append(y.detach().cpu().numpy());ps.append(torch.sigmoid(z).detach().cpu().numpy())
    y=np.concatenate(ys);p=np.concatenate(ps);return sum(losses)/len(loader.dataset),metrics(y,p)

if torch.cudargs.device_count() > 1:
    print(f"Using {torch.cudargs.device_count()} GPUs")
    model = nn.DataParallel(model)

model = model.to(DEV)

ts0, tst_m0 = run_epoch(model,te,opt,lossfn,DEVICE,train=True)
print( 'Performance before fine-tuning with LoRA', tst_m0 )

# ====================================
# Perform fine-tuning with LORA
# =====================================
for e in range(1,args.EP+1):
    tl,_=run_epoch(model,tr,opt,lossfn, DEVICE,True);
    vl,vm=run_epoch(model,va,opt,lossfn,DEVICE,False);
    sch.step()
    print(f"{e:02d} train={tl:.4f} val={vl:.4f} AUROC={vm['AUROC']:.4f} AUPRC={vm['AUPRC']:.4f}")
    if vm["AUROC"]>best:
        epochs_without_improvement = 0
        best=vm["AUROC"]

        torch.save({
            "model_state_dict": copy.deepcopy(model.state_dict()),
            "optimizer_state_dict": opt.state_dict(),
            "scheduler_state_dict": sch.state_dict(),
            "val_AUROC": best,
            "EP": e,
            "input_size": args.RES,
            "preprocessing": ("RETFound-Green border removal + 2.5% buffer + square padding + Lanczos resize"),
            "lo_rank": args.lo_rank,
            "lo_alpha": args.lo_alpha,
            "lo_DO": args.lo_DO,
        }, output + '/' + pref + '.pth' ); print('Saved model to disc')        
    else:
        epochs_without_improvement += 1
    if epochs_without_improvement >= patience:
        print('Early stopping...')
        break
        
print(f"Best ODIR validation AUROC={best:.4f}; saved={pref}") 

# ====================================
# Perform fine-tuning with LORA
# =====================================
ts, tst_m = run_epoch(model,te,opt,lossfn,DEVICE,train=True)
print( 'Performance after fine-tuning with LoRA', tst_m)
