# Trong Notebook Kaggle

!mkdir -p /kaggle/working/{checkpoints,logs}

# Cài đặt thư viện

!pip install -r /kaggle/input/crypto-rwkv-ts/requirements.txt

# Chạy training

%run /kaggle/input/crypto-rwkv-ts/src/train.py \
 --data_path /kaggle/input/crypto-data/btc_usdt_1min.csv \
 --checkpoint_dir /kaggle/working/checkpoints \
 --log_dir /kaggle/working/logs
