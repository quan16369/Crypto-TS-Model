import torch
import yaml
from datetime import datetime
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, Any, Optional, Union, Tuple 

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
        self.min_delta = config['training']['min_delta']
        self.counter = 0
        self.best_loss = float('inf')

    def check(self, current_loss: float) -> bool:
        if current_loss < self.best_loss - self.min_delta:
            self.best_loss = current_loss
            self.counter = 0
        else:
            self.counter += 1
        return self.counter >= self.patience
