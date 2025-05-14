import torch
import yaml
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Union, Tuple 
import numpy as np 

def seed_everything(seed: int = 42):
    """Cố định seed cho tất cả thư viện"""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)

class TrainingTracker:
    """Theo dõi quá trình training"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        log_dir = Path(config['training']['log_dir']) / datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(log_dir)
        self._save_config()

    def _save_config(self):
        """Lưu config vào thư mục log"""
        with open(Path(self.writer.log_dir) / "config.yaml", 'w') as f:
            yaml.dump(self.config, f)

    def log(self, tag: str, value: float, step: int):
        """Ghi log metrics"""
        self.writer.add_scalar(tag, value, step)

    def close(self):
        """Đóng SummaryWriter"""
        self.writer.close()

class EarlyStopper:
    def __init__(self, config: Dict[str, Any]):  
        self.patience = config['training']['patience']
        self.min_delta = config['training'].get('min_delta', 0.01)
        self.counter = 0
        self.best_loss = float('inf')

    def check(self, current_loss: float) -> bool:
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
    
# label smoothing
def label_smooth_loss(pred, target, smoothing=0.1):
    n_class = pred.size(-1)
    one_hot = torch.full_like(pred, fill_value=smoothing/(n_class-1))
    one_hot.scatter_(-1, target.unsqueeze(-1), 1.0-smoothing)
    return F.kl_div(F.log_softmax(pred, dim=-1), one_hot, reduction='batchmean')

# Mixup augmentation
def mixup_data(x, y, alpha=0.4):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    
    batch_size = x.size()[0]
    index = torch.randperm(batch_size)
    
    mixed_x = lam * x + (1 - lam) * x[index, :]
    mixed_y = lam * y + (1 - lam) * y[index]
    return mixed_x, mixed_y

class CompositeLoss(nn.Module):
    """Kết hợp nhiều loss function với trọng số"""
    def __init__(self, losses: list, weights: list):
        super().__init__()
        self.losses = losses
        self.weights = weights
        
    def forward(self, pred, target):
        total_loss = 0
        for loss, weight in zip(self.losses, self.weights):
            total_loss += weight * loss(pred, target)
        return total_loss

class QuantileLoss(nn.Module):
    """Loss function for quantile regression"""
    def __init__(self, quantiles: list):
        super().__init__()
        self.quantiles = quantiles
        
    def forward(self, pred, target):
        losses = []
        for i, q in enumerate(self.quantiles):
            errors = target - pred[:, i]
            losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(1))
        return torch.mean(torch.sum(torch.cat(losses, dim=1), dim=1))

class ModelEMA:
    """Exponential Moving Average for model weights"""
    def __init__(self, model, decay=0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
    def update(self, model):
        for name, param in model.named_parameters():
            if param.requires_grad:
                if name not in self.shadow:
                    self.shadow[name] = param.data.clone()
                else:
                    self.shadow[name] = (self.decay * self.shadow[name] 
                                      + (1 - self.decay) * param.data)
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.backup[name] = param.data
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                param.data = self.backup[name]
        self.backup = {}
    
    @property
    def module(self):
        return self.model
