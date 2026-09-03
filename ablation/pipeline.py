import os, math, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch, timm
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.autograd import Function
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageDraw
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, 
    balanced_accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, cohen_kappa_score, mean_absolute_error
)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

SEED=41
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# ----------------------------------------------------
# PARAMETERS
# ----------------------------------------------------
PATCH_SIZE = 14
RES1=RES2= np.uint16(14*28*2.5)
 
NUM_CLASSES = 2
LR = 1e-4
MAX_EPOCHS = 2
tasks = ['A']  
BS = 8
centroid_crop = True 
DATA_DIR = '/kaggle/input/datasets/andrewmvd/ocular-disease-recognition-odir5k/ODIR-5K/ODIR-5K/'
odir_df = pd.read_excel( DATA_DIR + 'data.xlsx')
IM_DIR = DATA_DIR + '/Training Images'

# ----------------------------------------------------
# LOAD DATA AND SPLIT
# ----------------------------------------------------
def reformat_and_prepare_odir(df):
    """
    Reformats the ODIR dataframe to cleanly separate image sources 
    and parse the 8-digit multi-label disease vector.
    """
    # 1. Target column names given in your schema
    target_cols = ['N', 'D', 'G', 'C', 'A', 'H', 'M', 'O']
    
    # 2. Extract columns relevant to our Siamese data loader
    # We rename 'ID' to 'patient_id' to explicitly track it during splitting
    reformatted_df = df[[
        'ID', 'Left-Fundus', 'Right-Fundus'
    ] + target_cols].copy()
    
    reformatted_df = reformatted_df.rename(columns={'ID': 'patient_id'})
    
    # Convert target labels to integer format explicitly
    for col in target_cols:
        reformatted_df[col] = reformatted_df[col].astype(int)
        
    return reformatted_df, target_cols

def split_dataset_by_patient(df):
    """
    Splits the dataframe into Train, Val, and Test subsets 
    guaranteeing zero patient_id overlap.
    """
    # Clean/shuffle the dataset deterministically
    df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    
    # Use GroupKFold on patient_id to split off a 15% Test Set first
    # 6 folds means each fold is roughly 16.6% of the data
    gkf_test = GroupKFold(n_splits=6)
    for train_val_idx, test_idx in gkf_test.split(df, groups=df['patient_id']):
        train_val_df = df.iloc[train_val_idx].reset_index(drop=True)
        test_df = df.iloc[test_idx].reset_index(drop=True)
        break # Take the first split layout
        
    # Split the remaining 85% into Train (70% total) and Val (15% total)
    # 15/85 is roughly 17.6%, so a 5-fold split works perfectly
    gkf_val = GroupKFold(n_splits=5)
    for train_idx, val_idx in gkf_val.split(train_val_df, groups=train_val_df['patient_id']):
        train_df = train_val_df.iloc[train_idx].reset_index(drop=True)
        val_df = train_val_df.iloc[val_idx].reset_index(drop=True)
        break
        
    print(f"Split complete! Train patients: {train_df['patient_id'].nunique()} | Val patients: {val_df['patient_id'].nunique()} | Test patients: {test_df['patient_id'].nunique()}")
    return train_df, val_df, test_df

def filter_existing_images(df, image_dir):
    """
    Verifies that the file names listed in your DataFrame 
    actually exist in your Kaggle image directory path.
    """
    initial_count = len(df)
    
    # Check both fields to guarantee both eye assets are physically present
    left_exists = df['Left-Fundus'].apply(lambda x: os.path.exists(os.path.join(image_dir, str(x))))
    right_exists = df['Right-Fundus'].apply(lambda x: os.path.exists(os.path.join(image_dir, str(x))))
    
    # Keep only rows where both assets are valid
    clean_df = df[left_exists & right_exists].reset_index(drop=True)
    
    dropped_count = initial_count - len(clean_df)
    if dropped_count > 0:
        print(f"🧹 Cleaned Dataset: Dropped {dropped_count} rows out of {initial_count} due to missing image files.")
    else:
        print("✅ All dataset images verified and present on disk.")
        
    return clean_df

class ODIRDataset(Dataset):
    def __init__(self, dataframe, target_cols, image_dir, mode="dual_view", centroid_crop=True, target_size=(392, 392)):
        """
        Args:
            dataframe (pd.DataFrame): Input split metadata dataframe
            target_cols (list or str): List of multi-label targets like ['N', 'D', 'G']
            image_dir (str): Base path to your folder containing the JPEG images
            mode (str): "dual_view" (Siamese) or "single_view" (Flattened)
            target_size (tuple): Output image shape matching your RETFound-Green grid (e.g., 392x392)
        """
        self.mode = mode
        self.apply_dropout = apply_dropout=(self.mode == "dual_view") 
        self.centroid_crop = centroid_crop 
        if isinstance(target_cols, str):
            self.target_cols = [target_cols]
        else:
            self.target_cols = list(target_cols)
            
        
        self.image_dir = image_dir
        self.target_size = target_size
        self.dataframe = dataframe.copy()
        
        # Enforce clean numerical array targets
        for col in self.target_cols:
            self.dataframe[col] = pd.to_numeric(self.dataframe[col], errors='coerce').fillna(0).astype(np.float32)
        
        if mode == "single_view":
            left_eyes = self.dataframe[['patient_id', 'Left-Fundus'] + self.target_cols].copy().rename(columns={'Left-Fundus': 'img_path'})
            right_eyes = self.dataframe[['patient_id', 'Right-Fundus'] + self.target_cols].copy().rename(columns={'Right-Fundus': 'img_path'})
            self.samples = pd.concat([left_eyes, right_eyes], axis=0).reset_index(drop=True)
        else:
            self.samples = self.dataframe.reset_index(drop=True)

    def _apply_centroid_crop(self, tensor_img):        
        # Expects a pre-converted PyTorch tensor: [3, H, W]
        c, h, w = tensor_img.shape
        
        # 1. Isolate the bright retinal circle mask
        gray = tensor_img.mean(dim=0)
        mask = gray > 0.05 
        
        nonzero_indices = torch.nonzero(mask)
        if nonzero_indices.numel() == 0:
            return tensor_img            
        # 2. Find the true boundaries of the eye circle
        mins, _ = nonzero_indices.min(dim=0)
        maxs, _ = nonzero_indices.max(dim=0)        
        # Extract row (Y) and column (X) bounds directly as Python integers
        ymin, xmin = mins[0].item(), mins[1].item()
        ymax, xmax = maxs[0].item(), maxs[1].item()
        
        # actual physical diameter of the circle (the shorter bounding side)
        circle_h = ymax - ymin
        circle_w = xmax - xmin
        true_diameter = min(circle_h, circle_w)
        half_d = true_diameter // 2
        
        # 3. Pinpoint the true anatomical centroid
        centroid_y = (ymin + ymax) // 2
        centroid_x = (xmin + xmax) // 2
        
        # 4. Compute crop boundaries defensively relative to the image borders
        y_start = centroid_y - half_d
        y_end = centroid_y + half_d
        x_start = centroid_x - half_d
        x_end = centroid_x + half_d
        
        # 5. THE ASPECT RATIO FIX: Handle Out-of-Bounds collisions gracefully
        pad_top = max(0, -y_start)
        pad_bottom = max(0, y_end - h)
        pad_left = max(0, -x_start)
        pad_right = max(0, x_end - w)
        
        y_start_clamped = max(0, y_start)
        y_end_clamped = min(h, y_end)
        x_start_clamped = max(0, x_start)
        x_end_clamped = min(w, x_end)
        
        # Extract the unwarped patch snippet
        cropped_patch = tensor_img[:, y_start_clamped:y_end_clamped, x_start_clamped:x_end_clamped]        
        # Symmetrically pad the patch to make it a perfect square if it hit an image boundary edge
        if pad_top > 0 or pad_bottom > 0 or pad_left > 0 or pad_right > 0:
            cropped_patch = TF.pad(cropped_patch, [pad_left, pad_top, pad_right, pad_bottom], fill=0)        
        return cropped_patch 
        
    def _load_image(self, file_name):
        """Helper method to load, normalize, and resize images lazily from disk."""
        img_path = os.path.join(self.image_dir, str(file_name))        
        #try:
        if 1:
            with Image.open(img_path) as img:
                img = img.convert('RGB')                                     
            im = TF.to_tensor(img)            
            if self.centroid_crop:
                im = self._apply_centroid_crop(im)            
        #except Exception as e:
        if 0:
            print(f"Warning: Failed to load target image asset {img_path}. Error: {e}")
            # Correct structural fallback matching target size tuple dimensions
            im = torch.zeros(3, self.target_size[0], self.target_size[1])                            
        return TF.resize(im, self.target_size)         
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]
        raw_labels = row[self.target_cols].values.astype(np.float32)
        labels = torch.tensor(raw_labels, dtype=torch.float32)
        
        if self.mode == "single_view":
            view1 = self._load_image(row['img_path'])
            if (self.apply_dropout) and (random.random() < 0.30):
                view1 = torch.zeros_like(view1)
            return view1, labels, idx
        else:
            view1 = self._load_image(row['Left-Fundus'])
            view2 = self._load_image(row['Right-Fundus'])
            if (self.apply_dropout) and (random.random() < 0.30):
                if random.random() < 0.5:
                    view1 = torch.zeros_like(view1)
                else:
                    view2 = torch.zeros_like(view2)                    
            return view1, view2, labels, idx


# ----------------------------------------------------
# UTILS
# ----------------------------------------------------
def show_batch( sample_batch_imgs, n=8 ):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    
    # Map task indices back to human-readable names for titles
    task_names = tasks if 'tasks' in locals() else ['Pathology Target']
    
    for i in range( n ):
        # Grab an individual image tensor: Shape [3, 392, 392]
        img_tensor = sample_batch_imgs[i]

        max_pixel_value = img_tensor.max().item()
    
        # --- CRITICAL FORMAT CONVERSIONS FOR MATPLOTLIB ---
        # Change dimension layout from [3, 392, 392] to [392, 392, 3]
        img_numpy = img_tensor.permute(1, 2, 0).cpu().numpy()
        
        # Clip pixel values defensively to [0.0, 1.0] range to avoid floating-point display warning artifacts
        img_numpy = np.clip(img_numpy, 0.0, 1.0)
        
        # 3. Plot the matrix onto the subplot array
        axes[i].imshow(img_numpy)
        
        # Parse target labels matrix to display current true state values
        label_value = int(sample_batch_labels[i].item())
        axes[i].set_title(f"Sample {i} | Label {task_names[0]}: {label_value}", fontsize=10)
        axes[i].axis('off') # Hide coordinate grid lines for clean presentation    
    plt.tight_layout()
    plt.show()

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0, checkpoint_path="best_model.pt", mode="min"):
        """
        Args:
            patience (int): How many epochs to wait after last time validation improved.
            min_delta (float): Minimum change in monitored value to qualify as an improvement.
            checkpoint_path (str): File path to save the optimal weights checkpoint model.
            mode (str): "min" if tracking a loss function, "max" if tracking a metric like AUROC.
        """
        self.patience = patience
        self.min_delta = min_delta
        self.checkpoint_path = checkpoint_path
        self.mode = mode
        self.counter = 0
        self.best_score = None
        self.early_stop = False        
    def __call__(self, current_score, model):
        # Convert performance metrics so that higher scores are always better
        score = -current_score if self.mode == "min" else current_score

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            print(f"EarlyStopping Counter: {self.counter} out of {self.patience}")
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0
    def save_checkpoint(self, model):
        """Saves model weights when the validation performance hits a historic high."""
        torch.save(model.state_dict(), self.checkpoint_path)
        print(f"Validation improved! Saving optimal model weights state to: {self.checkpoint_path}")

def specificity_score(y, q):
    tn, fp, _, _ = confusion_matrix(y, q, labels=[0, 1]).ravel()
    return np.nan if (tn + fp) == 0 else tn / (tn + fp)
def calc_bin(y, p, t, m):
    y, p = np.asarray(y), np.asarray(p)
    ok = pd.notna(y) & np.isfinite(p)
    y, p = y[ok].astype(int), p[ok]
    if len(y) < 2 or len(np.unique(y)) < 2: return np.nan
    q = p >= t
    if m == "AUROC": return roc_auc_score(y, p)
    if m == "AUPRC": return average_precision_score(y, p)
    if m == "Accuracy": return accuracy_score(y, q)
    if m == "Balanced_Accuracy": return balanced_accuracy_score(y, q)
    if m == "Precision": return precision_score(y, q, zero_division=0)
    if m == "Recall": return recall_score(y, q, zero_division=0)
    if m == "Specificity": return specificity_score(y, q)
    return f1_score(y, q, zero_division=0)
# ----------------------------------------------------
# SIAMESE SYSTEM ARCHITECTURE
# ----------------------------------------------------
class SiameseRETFoundGreen(nn.Module):
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.backbone.global_pool = 'avg'
        self.classifier = nn.Linear(self.backbone.embed_dim, num_classes)

    def forward(self, view1, view2=None):
        feat1 = self.backbone(view1)  
        if view2 is not None:
            feat2 = self.backbone(view2)
            combined_features = torch.max(feat1, feat2)
            logits = self.classifier(combined_features)
            # Return embeddings alongside logits for training alignment losses
            return logits, feat1, feat2, combined_features
        else:            
            logits = self.classifier(feat1)        
            return logits, feat1, None, feat1
 
def get_backbone( RES1, RES2):    
    model = timm.create_model(
        "vit_small_patch14_reg4_dinov2",
        img_size=(RES1,RES2),
        num_classes=0,
        dynamic_img_size=True,
        )
    checkpoint_path='rfg_statedict.pth'
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
    )
    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint
    
    print("Checkpoint pos_embed:", state_dict["pos_embed"].shape)
    
    # --------------------------------------------------
    # Interpolate positional embeddings
    # --------------------------------------------------
    pos_embed = state_dict["pos_embed"]    
    B, N, C = pos_embed.shape    
    old_grid = int(N ** 0.5)
    new_grid = model.patch_embed.grid_size[0]    
    print("Old grid:", old_grid)
    print("New grid:", new_grid)    
    assert old_grid * old_grid == N
    assert new_grid * new_grid == model.pos_embed.shape[1]    
    # [1, 784, 384] → [1, 384, 28, 28]
    pos_embed = pos_embed.reshape(B, old_grid, old_grid, C ).permute(0, 3, 1, 2)    
    # 28×28 → 56×56
    pos_embed = F.interpolate( pos_embed, size=(new_grid, new_grid), mode="bicubic", align_corners=False )    
    # [1, 384, 56, 56] → [1, 3136, 384]
    pos_embed = pos_embed.permute( 0, 2, 3, 1).reshape( B, new_grid * new_grid, C )    
    state_dict["pos_embed"] = pos_embed    
    print("Interpolated pos_embed:", state_dict["pos_embed"].shape)
    print("Image size:", model.patch_embed.img_size)
    print("Grid size:", model.patch_embed.grid_size)
    print("Patch size:", model.patch_embed.patch_size)
    print("Pos embed:", model.pos_embed.shape)
    return model
# ----------------------------------------------------
# TRAINING ENGINE 
# ----------------------------------------------------
def train_with_early_stopping(model, trn_loader, val_loader, target_cols, mode, device, patience=5, alpha=0.1):
    print(f"\n>>> Launching Development Loop [Configuration Mode: {mode.upper()}]")
        
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)    
    early_stopper = EarlyStopping(patience=patience, checkpoint_path=f"best_{mode}_model.pt", mode="min")
    
    n_positive = trn_df[target_cols].sum()
    n_negative = len(trn_df) - n_positive    
    pos_weight = torch.tensor(
        [n_negative / n_positive],
        dtype=torch.float32,
        device=DEVICE
    ); criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train(); train_loss = 0.0        
        for data in trn_loader:
            optimizer.zero_grad()
            
            if mode == "dual_view":
                v1, v2, targets, _ = data
                v1, v2, targets = v1.to(device), v2.to(device), targets.to(device)
                outputs, z_L, z_R, z_P = model(v1, v2)
                
                # Primary Task Loss
                loss_cls = criterion(outputs, targets)
                
                # --- MATHEMATICAL LATENT ALIGNMENT MSE COST ---
                # Detach the anchor to prevent representations from collapsing into constants
                anchor = z_P.detach()
                loss_mse_L = F.mse_loss(z_L, anchor)
                loss_mse_R = F.mse_loss(z_R, anchor)
                loss_alignment = 0.5 * (loss_mse_L + loss_mse_R)
                
                # Combine losses scaled by regularization hyperparameter alpha
                loss = loss_cls + (alpha * loss_alignment)
            else:
                v1, targets, _ = data
                v1, targets = v1.to(device), targets.to(device)
                outputs, _, _, _ = model(v1, view2=None)
                loss = criterion(outputs, targets)
                
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(trn_loader)
        
        # --- B. VALIDATION PASS WITH TASK-WISE AUROC TRACKING ---
        model.eval()
        val_loss = 0.0        
        all_true_labels = []
        all_pred_probs = []        
        with torch.no_grad():
            for v1_val, targets_val, _ in val_loader:
                v1_val, targets_val = v1_val.to(device), targets_val.to(device)
                outputs_val, _, _, _ = model(v1_val, view2=None) # Single-view capabilities check
                
                loss_val = criterion(outputs_val, targets_val)
                val_loss += loss_val.item()
                
                # Collect probabilities for multi-label tasks evaluation
                probs_val = torch.sigmoid(outputs_val)
                all_true_labels.append(targets_val.cpu().numpy())
                all_pred_probs.append(probs_val.cpu().numpy())
                
        avg_val_loss = val_loss / len(val_loader)
        
        # Format stacked arrays across dimensions [Samples, Classes]
        y_true_all = np.vstack(all_true_labels)
        y_pred_all = np.vstack(all_pred_probs)
        
        # Compute Macro-AUROC using your custom calc_bin suite class-by-class
        auroc_scores = []
        for class_idx, class_name in enumerate(target_cols):
            task_auroc = calc_bin(y_true_all[:, class_idx], y_pred_all[:, class_idx], t=0.5, m="AUROC")
            if not np.isnan(task_auroc):
                auroc_scores.append(task_auroc)
                
        # Handle structural edge cases if class diversity drops during a tight partition split
        avg_val_auroc = np.mean(auroc_scores) if len(auroc_scores) > 0 else np.nan
        
        print(f"Epoch {epoch:02d} | Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f} | Val Macro-AUROC: {avg_val_auroc:.4f}")
        
        # --- C. EARLY STOPPING TRIGGER ---
        early_stopper(avg_val_loss, model)
        if early_stopper.early_stop:
            print(f">>> Early stopping triggered! Training stopped at epoch {epoch}.")
            break            
    # Load the best model weights before returning to ensure optimal test evaluations
    model.load_state_dict(torch.load(f"best_{mode}_model.pt"))
    print(f"Loaded best weights from checkpoint file successfully.")
    return model
# ----------------------------------------------------
# RUN EXPERIMENT 
# ----------------------------------------------------
def run_ablation_study():   
    # Initialize the architecture matching your targets mapping metrics dynamically
    siamese_net = SiameseRETFoundGreen(model, len(tasks))
    
    # Execute the updated optimization loop with Alpha alignment parameter active
    model_single_view = train_with_early_stopping(
        model=siamese_net, 
        trn_loader=loaders['trn'], 
        val_loader=loaders['val'], 
        target_cols=tasks, 
        mode=MODE, 
        device=DEVICE,
        patience=5,
        alpha=0.1 # Adjust this value between 0.0 and 0.1 to evaluate performance shifts
    )    
    return model_single_view          
# ----------------------------------------------------
# EVALUATION 
# ----------------------------------------------------
def evaluate_on_test_set( mode, model, tst_loader, target_cols, device):
    print("\n" + "="*60)
    print("LAUNCHING REUSED CUSTOM EVALUATION CODES ON TST_DF")
    print("="*60)    
    model.eval()    
    all_true_labels = []
    all_single_probs = []
    
    with torch.no_grad():        
        for data in tst_loader:
            if 'dual' in mode:
                test_v1, test_v2, test_targets, _ = data
                test_v2 = test_v2.to(device)
            else:
                test_v1, test_targets, _ = data
                test_v2 = None            
            test_v1 = test_v1.to(device)
            prob_single = torch.sigmoid(model(test_v1,test_v2)).cpu().numpy()
            
            all_true_labels.append(test_targets.numpy().squeeze())
            all_single_probs.append(prob_single.squeeze())
            
    y_true_matrix = np.atleast_2d(np.array(all_true_labels))
    y_single_matrix = np.atleast_2d(np.array(all_single_probs))
    
    metrics_to_track = ["AUROC", "AUPRC", "Accuracy", "Balanced_Accuracy", "Specificity"]
    performance_records = []    
    for class_idx, class_name in enumerate(target_cols):
        y_true_task = y_true_matrix[:, class_idx]
        y_single_task = y_single_matrix[:, class_idx]        
        for metric in metrics_to_track:
            score_single = calc_bin(y_true_task, y_single_task, t=0.5, m=metric)
            performance_records.append({
                "Pathology_Task": class_name,
                "Metric": metric,
                "Single_View_Only": score_single
            })            
    metrics_summary_df = pd.DataFrame(performance_records)
    print("\n>>> MACRO SUMMARY BY METRIC STRATEGIES:")
    print(metrics_summary_df.groupby(["Metric"])[["Single_View_Only"]].mean())
    return metrics_summary_df    

if __name__ == "__main__":    
    df2, tc = reformat_and_prepare_odir( odir_df )
    trn_df, val_df, tst_df = split_dataset_by_patient( df2 )    
    trn_df = filter_existing_images(trn_df, IM_DIR)
    val_df = filter_existing_images(val_df, IM_DIR)
    tst_df = filter_existing_images(tst_df, IM_DIR)    
    
    res_dfs,models,loaders={},{},{}
    for MODE in ['single_view','dual_view']:
        loaders['trn'] = DataLoader(ODIRDataset(trn_df, tasks, image_dir=IM_DIR, mode=MODE, centroid_crop=centroid_crop), batch_size=BS, shuffle=True)
        loaders['val'] = DataLoader(ODIRDataset(val_df, tasks, image_dir=IM_DIR, mode=MODE, centroid_crop=centroid_crop), batch_size=BS, shuffle=True)
        loaders['tst'] = DataLoader(ODIRDataset(tst_df, tasks, image_dir=IM_DIR, mode=MODE, centroid_crop=centroid_crop), batch_size=BS, shuffle=True)

        if 'dual' in MODE:
            sample_batch_v1, sample_batch_v2, sample_batch_labels, sample_batch_indices = next(iter(loaders2['trn']))
            show_batch(sample_batch_v2,8)            

        sample_batch_v1, sample_batch_labels, sample_batch_indices = next(iter(loaders['tst']))
        print("Returned Batch Image Shape:", sample_batch_v1.shape)   # Expected: [8, 3, 224, 224]
        print("Returned Batch Label Shape:", sample_batch_labels.shape) # Expected: [8, 1]
        print("Returned Label Values:\n", sample_batch_labels)       
        
        show_batch(sample_batch_v1)        
        model = get_backbone( RES1, RES2) 
        models[MODE] = train_with_early_stopping( SiameseRETFoundGreen(model, len(tasks)), loaders['trn'], loaders['val'], tasks, mode=MODE, device=DEVICE )    
        res_dfs[MODE] = evaluate_on_test_set( models[MODE], MODE, loaders['tst'], tasks, DEVICE)
        print(res_dfs[MODE])  
