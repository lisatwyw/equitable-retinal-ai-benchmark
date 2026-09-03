

import torch.nn.functional as F

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

# ----------------------------------------------------
# TRAINING ENGINE 
# ----------------------------------------------------
def train_with_early_stopping(model, trn_loader, val_loader, target_cols, mode, device, patience=5, alpha=0.1):
    print(f"\n>>> Launching Development Loop [Configuration Mode: {mode.upper()}]")
        
    model.to(device)
    optimizer = optim.AdamW(model.parameters(), lr=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    
    # Early Stopping monitors validation loss
    early_stopper = EarlyStopping(patience=patience, checkpoint_path=f"best_{mode}_model.pt", mode="min")
        
    for epoch in range(1, MAX_EPOCHS + 1):
        # --- A. TRAINING PASS ---
        model.train()
        train_loss = 0.0
        
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
