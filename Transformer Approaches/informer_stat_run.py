#!/usr/bin/env python3
"""
informer_stat_run.py

Multi-seed, resumable training + evaluation wrapper for Informer model.
- Saves per-epoch CSV logs (includes GPU memory stats)
- Saves per-run metrics JSON and checkpoint
- Resumes automatically from checkpoint if requested
- Uses AMP when requested
- Early-stopping by validation accuracy (patience configurable)
- DataLoader uses deterministic shuffling per-seed via torch.Generator
"""

import os
import sys
import argparse
import json
import csv
import time
import traceback
from datetime import datetime
from tqdm import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, precision_recall_curve, average_precision_score
)
from scipy import signal
import warnings
warnings.filterwarnings("ignore")

# ---------------- GPU memory helpers ----------------
def _gpu_mem_stats(device=None):
    """Return (allocated_mb, reserved_mb, peak_allocated_mb, peak_reserved_mb)."""
    try:
        if not torch.cuda.is_available():
            return 0.0, 0.0, 0.0, 0.0
        dev = torch.device(device) if device is not None else torch.device('cuda')
        torch.cuda.synchronize(dev)
        alloc = float(torch.cuda.memory_allocated(dev)) / (1024**2)
        reserved = float(torch.cuda.memory_reserved(dev)) / (1024**2)
        peak_alloc = float(torch.cuda.max_memory_allocated(dev)) / (1024**2)
        peak_reserved = float(torch.cuda.max_memory_reserved(dev)) / (1024**2)
        return alloc, reserved, peak_alloc, peak_reserved
    except Exception:
        return 0.0, 0.0, 0.0, 0.0

def _reset_gpu_peak(device=None):
    try:
        if torch.cuda.is_available():
            dev = torch.device(device) if device is not None else torch.device('cuda')
            torch.cuda.reset_peak_memory_stats(dev)
    except Exception:
        pass

def _fmt_mb(x):
    return float(round(x, 2))

# ---------------- utility IO helpers ----------------
def safe_json_write(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(obj, f, indent=2)
    os.replace(tmp, path)

def read_json(path):
    if os.path.exists(path):
        try:
            with open(path, "r") as f:
                return json.load(f)
        except Exception:
            return None
    return None

def append_csv_row(path, row, header=None):
    exists = os.path.exists(path)
    with open(path, "a", newline='') as f:
        writer = csv.writer(f)
        if (not exists) and header:
            writer.writerow(header)
        writer.writerow(row)

def plot_loss_acc_from_csv(epoch_csv_path, out_dir):
    if os.path.exists(epoch_csv_path):
        df = pd.read_csv(epoch_csv_path)
        if "epoch" in df.columns:
            if "train_loss" in df.columns and "val_loss" in df.columns:
                plt.figure(); plt.plot(df["epoch"], df["train_loss"], label="train_loss")
                plt.plot(df["epoch"], df["val_loss"], label="val_loss")
                plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss")
                plt.savefig(os.path.join(out_dir, "loss_curve.png")); plt.close()
            if "train_acc" in df.columns and "val_acc" in df.columns:
                plt.figure(); plt.plot(df["epoch"], df["train_acc"], label="train_acc")
                plt.plot(df["epoch"], df["val_acc"], label="val_acc")
                plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Accuracy")
                plt.savefig(os.path.join(out_dir, "acc_curve.png")); plt.close()

def plot_confusion(y_true, y_pred, labels, out_path):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(6,5))
    try:
        sns.heatmap(cm, annot=True, fmt='d', xticklabels=labels, yticklabels=labels)
    except Exception:
        plt.imshow(cm); plt.colorbar()
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
    plt.savefig(out_path); plt.close()

def plot_roc_multi(y_true, y_scores, class_names, out_path):
    n_classes = len(class_names)
    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        try:
            y_true_i = (np.array(y_true) == i).astype(int)
            fpr, tpr, _ = roc_curve(y_true_i, np.array(y_scores)[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f"{class_names[i]} (AUC={roc_auc:.2f})")
        except Exception:
            pass
    plt.plot([0,1],[0,1],'k--')
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title("ROC Curve"); plt.legend(loc='best')
    plt.savefig(out_path); plt.close()

def plot_pr_multi(y_true, y_scores, class_names, out_path):
    n_classes = len(class_names)
    plt.figure(figsize=(8,6))
    for i in range(n_classes):
        try:
            y_true_i = (np.array(y_true) == i).astype(int)
            prec, rec, _ = precision_recall_curve(y_true_i, np.array(y_scores)[:, i])
            ap = average_precision_score(y_true_i, np.array(y_scores)[:, i])
            plt.plot(rec, prec, label=f"{class_names[i]} (AP={ap:.2f})")
        except Exception:
            pass
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title("Precision-Recall Curve"); plt.legend(loc='best')
    plt.savefig(out_path); plt.close()

# ---------------- dataset & preprocessing (your Informer dataset) ----------------
SFREQ = 200
filter_range = [0.5, 40]
b, a = signal.butter(3, np.float32(filter_range)*2/SFREQ, 'bandpass')

# Update BASE_DIR for your dataset location
BASE_DIR = "/raid/dsamantaai/scholars/deepak/data/hms-harmful-brain-activity-classification/"
META_CSV = os.path.join(BASE_DIR, "train.csv")
if not os.path.exists(META_CSV):
    raise FileNotFoundError(f"Meta CSV not found: {META_CSV}")

metaDF_all = pd.read_csv(META_CSV)

brain_activities = ['Seizure', 'GPD', 'LRDA', 'Other', 'GRDA', 'LPD']
activity_mapping = {activity: idx for idx, activity in enumerate(brain_activities)}

def load_splits_for_seed(out_dir,seed):
    split_dir = os.path.join(out_dir, "splits", f"seed{seed}")
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(f"Splits for seed {seed} not found at {split_dir}. Generate them first.")
    trainDF = pd.read_csv(os.path.join(split_dir, "train_split.csv"))
    valDF   = pd.read_csv(os.path.join(split_dir, "val_split.csv"))
    testDF  = pd.read_csv(os.path.join(split_dir, "test_split.csv"))
    assert "eeg_id" in testDF.columns, f"Missing eeg_id in split for seed {seed}"
    return trainDF, valDF, testDF

class HMS_Dataset(Dataset):
    def __init__(self, metaDF, base_dir, activity_mapping, EEG_SECS=10):
        self.metaDF = metaDF.reset_index(drop=True)
        self.base_dir = base_dir
        self.activity_mapping = activity_mapping
        self.EEG_SECS = EEG_SECS
        # RAW_FEATURES not used for time series extraction here but kept for compatibility
        self.RAW_FEATURES = {'LL': ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1'],
                             'RL': ['Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2'],
                             'LP': ['Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1'],
                             'RP': ['Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2']}

    def __len__(self):
        return len(self.metaDF)

    def getTimeSeriesEEG(self, parquet_path, eeg_label_offset_seconds, EEG_SECS=10):
        # read parquet and return channels x time window; matches your earlier helper
        temp_df = pd.read_parquet(parquet_path)
        C1 = ['Fp1', 'Fp2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fp1', 'Fp2','F3', 'F4', 'C3', 'C4', 'P3', 'P4']
        C2 = ['F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'O1', 'O2', 'F3', 'F4', 'C3','C4', 'P3', 'P4', 'O1', 'O2']
        temp_arr1 = temp_df[C1].to_numpy().T
        temp_arr1[np.isnan(temp_arr1)] = 1e-4
        temp_arr2 = temp_df[C2].to_numpy().T
        temp_arr2[np.isnan(temp_arr2)] = 1e-4
        temp_arr = temp_arr1 - temp_arr2
        time_temp = 200*int(eeg_label_offset_seconds)
        start_seq =  round(time_temp + (50 - EEG_SECS) / 2 * 200)
        end_seq =  round(time_temp + (50 + EEG_SECS) / 2 * 200)
        return temp_arr[:, start_seq:end_seq]

    def __getitem__(self, idx):
        row = self.metaDF.iloc[idx]
        eeg_id = row["eeg_id"]
        label = row["expert_consensus"]
        offset = row["eeg_label_offset_seconds"]
        ppath = os.path.join(self.base_dir, "train_eegs", f"{eeg_id}.parquet")
        X = self.getTimeSeriesEEG(ppath, offset, EEG_SECS=self.EEG_SECS)  # shape: channels x time
        # Model expects sequence-length x input_dim (we'll keep input_dim=channels)
        X_tensor = torch.tensor(X.T, dtype=torch.float32)  # shape: T x C
        y = self.activity_mapping[label]
        y_tensor = torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.long), num_classes=len(self.activity_mapping)).float()
        return X_tensor, y_tensor

# ---------------- Informer model (simplified version from your code) ----------------
class ProbAttention(nn.Module):
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        # queries/keys/values: (B, L, d_model) - simplified single-head projection omitted
        scale = self.d_model ** -0.5
        scores = torch.matmul(queries, keys.transpose(-2, -1)) * scale
        attn = torch.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        context = torch.matmul(attn, values)
        return context

class InformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1):
        super().__init__()
        self.attn = ProbAttention(d_model, n_heads, dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model),
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        attn_output = self.attn(x, x, x)
        x = self.norm1(x + self.dropout(attn_output))
        ffn_output = self.ffn(x)
        x = self.norm2(x + self.dropout(ffn_output))
        return x

class InformerEncoder(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, num_layers, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([InformerEncoderLayer(d_model, n_heads, d_ff, dropout) for _ in range(num_layers)])

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

class Informer(nn.Module):
    def __init__(self, input_dim, seq_len, d_model, n_heads, d_ff, num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, d_model)
        self.positional_encoding = nn.Parameter(torch.zeros(1, seq_len, d_model))
        nn.init.normal_(self.positional_encoding, mean=0, std=0.02)
        self.encoder = InformerEncoder(d_model, n_heads, d_ff, num_layers, dropout)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, x):
        # x: (B, L, input_dim)  <-- note dataset returns (L, C) per-sample; DataLoader -> batch (B, L, C)
        x = self.input_proj(x) + self.positional_encoding[:, :x.size(1), :]
        x = self.encoder(x)
        x = x.mean(dim=1)
        x = self.fc(x)
        return x

# ---------------- training / evaluation functions ----------------
def train_model_informer(model, train_loader, val_loader, criterion, optimizer, scheduler,
                         device, epochs, checkpoint_path, epoch_log_csv, patience=5,
                         use_amp=False, resume_from=None):
    """
    Train model with checkpointing, early stopping, AMP optional.
    - train_loader yields (B, L, C), y_onehot shape (B, num_classes)
    """
    if device is None or (isinstance(device, str) and device == "cuda"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    start_epoch = 0
    best_val_acc = -1.0
    best_epoch = -1
    no_improve = 0

    if resume_from and os.path.exists(resume_from):
        ck = torch.load(resume_from, map_location=device)
        if isinstance(ck, dict) and 'model_state_dict' in ck:
            model.load_state_dict(ck['model_state_dict'])
            try:
                optimizer.load_state_dict(ck.get('optimizer_state_dict', {}))
            except Exception:
                print("Warning: optimizer state could not be fully loaded from checkpoint.")
            try:
                sched_state = ck.get('scheduler_state_dict', None)
                if sched_state is not None and hasattr(scheduler, 'load_state_dict'):
                    scheduler.load_state_dict(sched_state)
            except Exception:
                pass
            start_epoch = int(ck.get('epoch', 0)) + 1
            best_val_acc = float(ck.get('best_val_accuracy', best_val_acc))
            best_epoch = int(ck.get('best_epoch', best_epoch))
            print(f"Resuming from {resume_from} at epoch {start_epoch}, best_val_acc={best_val_acc}")

    # prepare CSV header
    if epoch_log_csv and (not os.path.exists(epoch_log_csv)):
        append_csv_row(epoch_log_csv, [], header=[
            "epoch","train_loss","train_acc","val_loss","val_acc","time_taken_sec","lr","timestamp",
            "gpu_alloc_mb","gpu_reserved_mb","gpu_peak_alloc_mb","gpu_peak_reserved_mb"
        ])

    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    for epoch in range(start_epoch, epochs):
        t0 = time.time()
        _reset_gpu_peak(device)
        model.train()
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for X, y in train_loader:
            # X: (B, L, C) ; y: onehot (B, num_classes)
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            optimizer.zero_grad()
            if use_amp:
                with torch.cuda.amp.autocast():
                    outputs = model(X)
                    loss = criterion(outputs, torch.argmax(y, dim=1))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(X)
                loss = criterion(outputs, torch.argmax(y, dim=1))
                loss.backward()
                optimizer.step()

            running_loss += float(loss.item()) * X.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += int((preds == torch.argmax(y, dim=1)).sum().item())
            running_total += X.size(0)

        train_loss = running_loss / max(1, running_total)
        train_acc = 100.0 * running_correct / max(1, running_total)

        # validation
        model.eval()
        val_running_loss = 0.0
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for Xv, yv in val_loader:
                Xv = Xv.to(device, non_blocking=True)
                yv = yv.to(device, non_blocking=True)
                vouts = model(Xv)
                vloss = criterion(vouts, torch.argmax(yv, dim=1))
                val_running_loss += float(vloss.item()) * Xv.size(0)
                vpreds = vouts.argmax(dim=1)
                val_correct += int((vpreds == torch.argmax(yv, dim=1)).sum().item())
                val_total += Xv.size(0)

        val_loss = val_running_loss / max(1, val_total)
        val_acc = 100.0 * val_correct / max(1, val_total)

        # scheduler step (ReduceLROnPlateau expects validation loss scalar)
        try:
            scheduler.step(val_loss)
        except Exception:
            try:
                scheduler.step()
            except Exception:
                pass

        # GPU stats after epoch
        alloc_mb, reserved_mb, peak_alloc_mb, peak_reserved_mb = _gpu_mem_stats(device)
        print(f"[Epoch {epoch}] GPU alloc={_fmt_mb(alloc_mb)}MB reserved={_fmt_mb(reserved_mb)}MB peak_alloc={_fmt_mb(peak_alloc_mb)}MB peak_reserved={_fmt_mb(peak_reserved_mb)}MB")
        lr = optimizer.param_groups[0]['lr']
        append_csv_row(epoch_log_csv, [
            int(epoch),
            float(train_loss),
            float(train_acc),
            float(val_loss),
            float(val_acc),
            round(time.time()-t0, 2),
            float(lr),
            datetime.utcnow().isoformat(),
            _fmt_mb(alloc_mb),
            _fmt_mb(reserved_mb),
            _fmt_mb(peak_alloc_mb),
            _fmt_mb(peak_reserved_mb)
        ])

        print(f"[train] epoch {epoch+1}/{epochs} train_loss={train_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f} lr={lr:.6g}")

        # early stopping & checkpointing
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            no_improve = 0
            if checkpoint_path:
                torch.save({
                    'epoch': int(epoch),
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': getattr(scheduler, 'state_dict', lambda: None)(),
                    'best_val_accuracy': float(best_val_acc),
                    'best_epoch': int(best_epoch)
                }, checkpoint_path)
                print(f"[train] Best model saved (val_acc={best_val_acc:.2f}%)")
        else:
            no_improve += 1
            print(f"[train] no improvement count: {no_improve}/{patience}")

        if no_improve >= patience:
            print(f"[train] Early stopping triggered after {patience} epochs without improvement.")
            break

    print(f"[train] Finished. Best val acc {best_val_acc:.2f} at epoch {best_epoch}")
    return model

def evaluate_model(model, test_loader, checkpoint_path=None, device=None, out_dir=None):
    if device is None or (isinstance(device, str) and device == "cuda"):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path and os.path.exists(checkpoint_path):
        ck = torch.load(checkpoint_path, map_location=device)
        if isinstance(ck, dict) and 'model_state_dict' in ck:
            model.load_state_dict(ck['model_state_dict'])
        else:
            model.load_state_dict(ck)
    model.to(device)
    model.eval()

    all_preds, all_labels, all_probs = [], [], []
    total_time = 0.0
    total_samples = 0

    _reset_gpu_peak(device)
    with torch.no_grad():
        for X, y in test_loader:
            X = X.to(device, non_blocking=True)
            y = y.to(device, non_blocking=True)
            t0 = time.time()
            outputs = model(X)
            t1 = time.time()
            total_time += (t1 - t0)
            total_samples += X.size(0)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, preds = outputs.max(1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(torch.argmax(y, dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)

    cm = confusion_matrix(all_labels, all_preds)
    if out_dir:
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=brain_activities, yticklabels=brain_activities)
        plt.title("Confusion Matrix")
        plt.savefig(os.path.join(out_dir, "confusion_matrix.png"))
        plt.close()

    # per-class AUC if probs available
    perclass_auc = []
    try:
        for c in range(all_probs.shape[1]):
            y_true_c = (all_labels == c).astype(int)
            fpr, tpr, _ = roc_curve(y_true_c, all_probs[:, c])
            perclass_auc.append(auc(fpr, tpr))
    except Exception:
        perclass_auc = None

    alloc_mb, reserved_mb, peak_alloc_mb, peak_reserved_mb = _gpu_mem_stats(device)

    result = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "perclass_auc": perclass_auc,
        "y_true": all_labels.tolist(),
        "y_pred": all_preds.tolist(),
        "y_scores": [list(map(float, s)) for s in all_probs],
        "gpu_alloc_mb": _fmt_mb(alloc_mb),
        "gpu_reserved_mb": _fmt_mb(reserved_mb),
        "gpu_peak_alloc_mb": _fmt_mb(peak_alloc_mb),
        "gpu_peak_reserved_mb": _fmt_mb(peak_reserved_mb),
        "avg_inference_time_sec": total_time / total_samples if total_samples>0 else None
    }

    if out_dir and all_probs.size:
        try:
            plot_roc_multi(all_labels, all_probs, brain_activities, os.path.join(out_dir, "roc_curve.png"))
        except Exception:
            pass
        try:
            plot_pr_multi(all_labels, all_probs, brain_activities, os.path.join(out_dir, "pr_curve.png"))
        except Exception:
            pass

    return result

# ---------------- run wrapper for one seed ----------------
def run_one_seed_informer(seed, out_dir="./multi_run_results", batch_size=64, epochs=50, num_workers=8,
                          pin_memory=True, prefetch_factor=2, persistent_workers=False,
                          resume_if_checkpoint=True, patience=5, lr=1e-4, use_amp=False,
                          model_kwargs=None, eeg_secs=10):
    seed = int(seed)
    run_name = f"Informer_seed{seed}"
    run_dir = os.path.join(out_dir, run_name)
    os.makedirs(run_dir, exist_ok=True)

    metrics_json = os.path.join(run_dir, f"metrics_{run_name}.json")
    epoch_log_csv = os.path.join(run_dir, f"epoch_log_{run_name}.csv")
    checkpoint_path = os.path.join(run_dir, f"best_{run_name}.pth")

    # Skip if completed
    existing = read_json(metrics_json)
    if existing and existing.get("status") == "completed":
        print(f"[SKIP] {run_name} already completed.")
        return existing

    # mark running
    safe_json_write(metrics_json, {"status":"running","seed":seed,"start_time":datetime.utcnow().isoformat()})
    np.random.seed(seed); import random; random.seed(seed); torch.manual_seed(seed)
    # speed settings
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    # ---- load BiVISTA splits from the shared location ----
    # 1) create splits
    trainDF, valDF, testDF =  load_splits_for_seed(out_dir,seed)
    print(f"[INFO] Loaded BiVISTA splits for seed {seed} from {os.path.join(out_dir, 'splits', f'seed{seed}')}")

    
    # create datasets + loaders
    train_ds = HMS_Dataset(metaDF=trainDF, base_dir=BASE_DIR, activity_mapping=activity_mapping, EEG_SECS=eeg_secs)
    val_ds = HMS_Dataset(metaDF=valDF, base_dir=BASE_DIR, activity_mapping=activity_mapping, EEG_SECS=eeg_secs)
    test_ds = HMS_Dataset(metaDF=testDF, base_dir=BASE_DIR, activity_mapping=activity_mapping, EEG_SECS=eeg_secs)

    g = torch.Generator()
    g.manual_seed(seed)
    def _worker_init_fn(worker_id):
        worker_seed = seed + worker_id
        np.random.seed(worker_seed); random.seed(worker_seed)
        try: torch.manual_seed(worker_seed)
        except Exception: pass

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin_memory,
                              generator=g, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, worker_init_fn=_worker_init_fn)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
                            prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, worker_init_fn=_worker_init_fn)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
                             prefetch_factor=prefetch_factor, persistent_workers=persistent_workers, worker_init_fn=_worker_init_fn)

    # model setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = train_ds[0][0].shape[1]  # C
    seq_len = train_ds[0][0].shape[0]
    # default model kwargs if None
    if model_kwargs is None:
        model_kwargs = dict(d_model=256, n_heads=8, d_ff=1024, num_layers=4, dropout=0.1)
    model = Informer(input_dim=input_dim, seq_len=seq_len, num_classes=len(activity_mapping), **model_kwargs).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

    resume_ckpt = checkpoint_path if (resume_if_checkpoint and os.path.exists(checkpoint_path)) else None

    try:
        t0 = time.time()
        model = train_model_informer(model, train_loader, val_loader, criterion, optimizer, scheduler,
                                     device=device, epochs=epochs, checkpoint_path=checkpoint_path, epoch_log_csv=epoch_log_csv,
                                     patience=patience, use_amp=use_amp, resume_from=resume_ckpt)
        train_time = time.time() - t0
        print(f"[{run_name}] training done in {train_time/3600:.2f} hrs. checkpoint: {checkpoint_path}")

        metrics = evaluate_model(model, test_loader, checkpoint_path=checkpoint_path, device=device, out_dir=run_dir)
        metrics.update({
            "seed": seed,
            "model_type": "Informer",
            "train_time_sec": train_time,
            "checkpoint_path": checkpoint_path,
            "epoch_log_csv": epoch_log_csv,
            "run_dir": run_dir,
            "status": "completed",
            "end_time": datetime.utcnow().isoformat()
        })
        safe_json_write(metrics_json, metrics)
        print(f"[{run_name}] TEST done : acc={metrics.get('accuracy')}, f1={metrics.get('f1_score')}")

        # plots from CSV
        try:
            plot_loss_acc_from_csv(epoch_log_csv, run_dir)
        except Exception as e:
            print("plot loss/acc failed:", e)

        return metrics

    except Exception as e:
        tb = traceback.format_exc()
        fail_obj = {"status":"failed","seed":seed,"error":str(e),"traceback":tb,"time":datetime.utcnow().isoformat()}
        safe_json_write(metrics_json, fail_obj)
        print(f"[{run_name}] FAILED. See {metrics_json}")
        raise

# ---------------- run multiple seeds ----------------
def run_repeats_informer(n_runs=10, start_seed=1000, out_dir="./multi_run_results", batch_size=64, epochs=50,
                         num_workers=8, pin_memory=True, prefetch_factor=2, persistent_workers=True,
                         resume_if_checkpoint=True, patience=5, lr=1e-4, use_amp=False, model_kwargs=None, eeg_secs=10):
    os.makedirs(out_dir, exist_ok=True)
    master_csv = os.path.join(out_dir, f"Informer_master.csv")
    if not os.path.exists(master_csv):
        append_csv_row(master_csv, [], header=["seed","status","accuracy","f1_score","train_time_sec","json_path","timestamp"])

    # ensure splits exist for all seeds
    missing = []
    for i in range(n_runs):
        seed = start_seed + i
        if not os.path.isdir(os.path.join(out_dir, "splits", f"seed{seed}")):
            missing.append(seed)
    if missing:
        raise FileNotFoundError(f"Missing split dirs for seeds: {missing}. Generate splits as in BiVISTA first.")


    records = []
    for i in range(n_runs):
        seed = start_seed + i
        run_dir = os.path.join(out_dir, f"Informer_seed{seed}")
        metrics_json = os.path.join(run_dir, f"metrics_Informer_seed{seed}.json")
        existing = read_json(metrics_json)
        if existing and existing.get("status") == "completed":
            rec = [seed, "completed", existing.get("accuracy"), existing.get("f1_score"), existing.get("train_time_sec"), metrics_json, existing.get("end_time")]
            append_csv_row(master_csv, rec)
            records.append(existing)
            print(f"[MASTER] Skip seed {seed} (already done)")
            continue

        print(f"[MASTER] Running seed {seed} ...")
        try:
            metrics = run_one_seed_informer(seed=seed, out_dir=out_dir, batch_size=batch_size, epochs=epochs, num_workers=num_workers,
                                           pin_memory=pin_memory, prefetch_factor=prefetch_factor, persistent_workers=persistent_workers,
                                           resume_if_checkpoint=resume_if_checkpoint, patience=patience, lr=lr, use_amp=use_amp,
                                           model_kwargs=model_kwargs, eeg_secs=eeg_secs)
            rec = [seed, metrics.get("status"), metrics.get("accuracy"), metrics.get("f1_score"), metrics.get("train_time_sec"), os.path.join(out_dir, f"Informer_seed{seed}", f"metrics_Informer_seed{seed}.json"), metrics.get("end_time")]
            append_csv_row(master_csv, rec)
            records.append(metrics)
        except Exception as e:
            print(f"[MASTER] seed {seed} failed; check {metrics_json}")
            append_csv_row(master_csv, [seed, "failed", None, None, None, metrics_json, datetime.utcnow().isoformat()])
            continue

    # aggregate summary
    df = pd.read_csv(master_csv)
    numeric_cols = ["accuracy","f1_score","train_time_sec"]
    summary = {}
    for c in numeric_cols:
        if c in df.columns:
            colvals = pd.to_numeric(df[c], errors='coerce').dropna()
            if len(colvals):
                summary[c] = {"mean": float(colvals.mean()), "std": float(colvals.std()), "n": int(len(colvals))}
    safe_json_write(os.path.join(out_dir, f"informer_aggregate_summary.json"), summary)
    print("=== AGGREGATE SUMMARY ===")
    print(pd.DataFrame(summary).T)
    return df

out_dir="./multi_run_results"
df=run_repeats_informer(n_runs=5, start_seed=1008, out_dir="./multi_run_results", batch_size=16, epochs=100,
                         num_workers=20, pin_memory=True, prefetch_factor=2, persistent_workers=True,
                         resume_if_checkpoint=True, patience=15, lr=1e-4, use_amp=False, model_kwargs=None, eeg_secs=50)
print("Done. master csv written to:", os.path.join(out_dir, "Informer_master.csv"),"\n df=",df)
