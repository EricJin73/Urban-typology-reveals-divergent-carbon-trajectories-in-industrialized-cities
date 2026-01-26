"""
Model Training and Evaluation Tools
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import r2_score, mean_squared_error


class ModelTrainer:
    """Model Trainer"""
    
    def __init__(self, model, device, lr=1e-3, weight_decay=1e-5, clip_norm=1.0):
        self.model = model
        self.device = device
        self.criterion = nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            model.parameters(), 
            lr=lr, 
            weight_decay=weight_decay
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            factor=0.5, 
            patience=5
        )
        self.clip_norm = clip_norm
        self.train_history = []
        self.val_history = []
        self.best_state = None
        self.best_val_loss = float('inf')
    
    def train_epoch(self, train_loader):
        """Train for one epoch."""
        self.model.train()
        loss_sum = 0.0
        n_sum = 0
        
        for xb_enc, _, xb_dec, _, cb, yb in train_loader:
            xb_enc = xb_enc.to(self.device)
            xb_dec = xb_dec.to(self.device)
            cb = cb.to(self.device)
            yb = yb.to(self.device)
            
            pred = self.model(xb_enc, cb, xb_dec)
            loss = self.criterion(pred, yb)
            
            self.optimizer.zero_grad()
            loss.backward()
            
            if self.clip_norm is not None:
                nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_norm)
            
            self.optimizer.step()
            
            bs = xb_enc.size(0)
            loss_sum += loss.item() * bs
            n_sum += bs
        
        return loss_sum / max(1, n_sum)
    
    def validate(self, val_loader):
        """Validate."""
        self.model.eval()
        val_sum = 0.0
        vn = 0
        
        with torch.no_grad():
            for xb_enc, _, xb_dec, _, cb, yb in val_loader:
                xb_enc = xb_enc.to(self.device)
                xb_dec = xb_dec.to(self.device)
                cb = cb.to(self.device)
                yb = yb.to(self.device)
                
                pred = self.model(xb_enc, cb, xb_dec)
                vloss = self.criterion(pred, yb)
                
                bs = xb_enc.size(0)
                val_sum += vloss.item() * bs
                vn += bs
        
        return val_sum / max(1, vn)
    
    def fit(self, train_loader, val_loader, epochs, verbose=True, country_code=""):
        """Complete training process."""
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.train_history.append(train_loss)
            self.val_history.append(val_loss)
            self.scheduler.step(val_loss)
            
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.best_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
                
                if verbose:
                    print(f"[{country_code}] Ep{epoch:03d} tr{train_loss:.4f} va{val_loss:.4f} *")
            elif verbose and epoch % 10 == 0:
                print(f"[{country_code}] Ep{epoch:03d} tr{train_loss:.4f} va{val_loss:.4f}")
        
        # Load best weights
        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
    
    def get_histories(self):
        """Get training histories."""
        return self.train_history, self.val_history


def calculate_metrics(model, loader, device, sy_list=None):
    """
    Calculate R² and RMSE metrics.
    
    Args:
        sy_list: If provided, calculate metrics on the original scale.
    """
    model.eval()
    y_true, y_pred = [], []
    
    with torch.no_grad():
        for xb_enc, _, xb_dec, _, city_idx, yb in loader:
            xb_enc = xb_enc.to(device)
            xb_dec = xb_dec.to(device)
            city_idx = city_idx.to(device)
            yb = yb.to(device)
            
            pred = model(xb_enc, city_idx, xb_dec)
            
            if sy_list is None:
                # Normalized space
                y_true.extend(pred.cpu().numpy().reshape(-1))
                y_pred.extend(yb.cpu().numpy().reshape(-1))
            else:
                # Original scale
                pred_np = pred.cpu().numpy()
                yb_np = yb.cpu().numpy()
                city_np = city_idx.cpu().numpy()
                B, T = pred_np.shape
                
                for b in range(B):
                    c = int(city_np[b])
                    scaler = sy_list[c]
                    if scaler is None:
                        continue
                    y_pred.extend(scaler.inverse_transform(pred_np[b].reshape(-1, 1)).reshape(-1))
                    y_true.extend(scaler.inverse_transform(yb_np[b].reshape(-1, 1)).reshape(-1))
    
    if len(y_true) == 0:
        return np.nan, np.nan
    
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    return r2_score(y_true, y_pred), np.sqrt(mean_squared_error(y_true, y_pred))


def predict_batch(model, X_enc, X_dec, city_idx, device, batch_size=512):
    """Batch prediction (deterministic)."""
    model.eval()
    outputs = []
    
    with torch.no_grad():
        n = len(X_enc)
        for i in range(0, n, batch_size):
            xb_enc = torch.from_numpy(X_enc[i:i+batch_size].astype(np.float32)).to(device)
            xb_dec = torch.from_numpy(X_dec[i:i+batch_size].astype(np.float32)).to(device)
            cb = torch.from_numpy(city_idx[i:i+batch_size].astype(np.int64)).to(device)
            
            pred = model(xb_enc, cb, xb_dec)
            outputs.append(pred.cpu().numpy())
    
    if not outputs:
        return np.zeros((0, 0), dtype=np.float32)
    
    return np.concatenate(outputs, axis=0)


def mc_dropout_predict(model, x_enc, x_dec, city_idx, n_samples):
    """
    Monte Carlo Dropout prediction (uncertainty estimation).
    
    Returns:
        np.ndarray of shape [n_samples, T]
    """
    predictions = []
    model.train()  # Enable dropout
    
    with torch.no_grad():
        for _ in range(n_samples):
            y_norm = model(x_enc, city_idx, x_dec)
            predictions.append(y_norm.squeeze(0).cpu().numpy())
    
    model.eval()
    return np.stack(predictions, axis=0)
