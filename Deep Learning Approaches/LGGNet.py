#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import warnings
import cv2
import time
from scipy import signal
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, cohen_kappa_score,
    confusion_matrix, roc_auc_score, roc_curve, auc
)
# Added for model configs and math
from argparse import Namespace
import math

# --- Imports for Progress Bars and Model Copying ---
from tqdm.auto import tqdm
import copy

# --- Imports for Deformer (can be removed if not needed elsewhere, but harmless) ---
try:
    from einops import rearrange
    from einops.layers.torch import Rearrange
except ImportError:
    print("einops not found (needed for Deformer, not LGGNet).")

# --- Import for LGGNet ---
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


warnings.filterwarnings('ignore')

print("All libraries imported.")


# In[ ]:


# --- Global Setup ---
SFREQ = 200
filter_range = [0.5, 40]
b, a = signal.butter(3, np.float32(filter_range)*2/SFREQ, 'bandpass')

# !!! IMPORTANT: Update this path to your BASE_DIR !!!
BASE_DIR = "/home/deepak/Documents/Deepak/Students/Saurabh_23CS60R76/My_Experiments/Brain_Harmful_Activity_Detection/Harmful_Brain_Activity/"

# --- Load and Split Metadata ---
try:
    metaDF = pd.read_csv(f"{BASE_DIR}train.csv")
    trainDF, tempDF = train_test_split(metaDF, test_size=0.2, random_state=42)
    valDF, testDF = train_test_split(tempDF, test_size=0.5, random_state=42)
    brain_activities = ['Seizure', 'GPD', 'LRDA', 'Other', 'GRDA', 'LPD']
    activity_mapping = {activity: idx for idx, activity in enumerate(brain_activities)}
    print(f"Data loaded. Train: {len(trainDF)}, Val: {len(valDF)}, Test: {len(testDF)}")
except FileNotFoundError:
    print(f"Error: train.csv not found at {BASE_DIR}. Please update 'BASE_DIR'.")
    raise SystemExit("Data loading failed.")


# In[ ]:


class HMS_Dataset(Dataset):
    def __init__(self, metaDF, base_dir, activity_mapping):
        self.metaDF = metaDF; self.base_dir = base_dir; self.activity_mapping = activity_mapping
        self.RAW_FEATURES = {'LL': ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1'], 'RL': ['Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2'], 'LP': ['Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1'], 'RP': ['Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2']}
        self.FEATS = [['Fp1','F7','T3','T5','O1'], ['Fp1','F3','C3','P3','O1'], ['Fp2','F8','T4','T6','O2'], ['Fp2','F4','C4','P4','O2']]
        self.Epartition = np.zeros((200, 8), dtype='float32'); self.Spartition = np.zeros((512, 16), dtype='float32'); self.Partition = np.zeros((16, 1296), dtype='float32')
        print(f"Dataset Initialized. Output shape (C, T) = (728, 1296).")

    def __len__(self): return len(self.metaDF)

    # --- Feature extraction functions (condensed for brevity, same logic) ---
    def _extract_raw(self, parquet_path, offset, length):
        try: raw_eeg = pd.read_parquet(parquet_path)
        except Exception as e: return None # Handle error later
        time_temp=offset*200; time_start=round(time_temp+(50-length)/2*200); time_stop=round(time_temp+(50+length)/2*200)
        eeg_default = raw_eeg.loc[time_start:(time_stop-1),:].reset_index(drop=True); list_eeg = list()
        for region in self.RAW_FEATURES.keys():
            eeg = np.zeros((4, eeg_default.shape[0]), dtype=np.float32)
            for chan_i, chan in enumerate(self.RAW_FEATURES[region]):
                eeg_1 = eeg_default.loc[:, chan.split('-')[0]]; mean_1 = eeg_1.mean(); eeg_1.fillna(value=mean_1, inplace=True); eeg_1 = eeg_1.values
                eeg_2 = eeg_default.loc[:, chan.split('-')[1]]; mean_2 = eeg_2.mean(); eeg_2.fillna(value=mean_2, inplace=True); eeg_2 = eeg_2.values
                new_eeg = eeg_1 - eeg_2; del eeg_1, eeg_2
                new_eeg = signal.filtfilt(b, a, new_eeg, axis=0); new_eeg = np.clip(new_eeg, -1024, 1024).astype("float32"); eeg[chan_i, :] = new_eeg
            eeg = np.reshape(eeg, (4, 200, length)); eeg = np.concatenate(tuple(eeg[i,:,:] for i in range(4)), 1); list_eeg.append(eeg)
        final_eeg = np.concatenate(list_eeg, 1); final_eeg /= 104; return final_eeg

    def raweeg_50s_from_eeg_v1(self, p, o): extracted = self._extract_raw(p, o, 50); return extracted if extracted is not None else np.zeros((200, 800), dtype=np.float32)
    def raweeg_20s_from_eeg_v1(self, p, o): extracted = self._extract_raw(p, o, 20); return extracted if extracted is not None else np.zeros((200, 320), dtype=np.float32)
    def raweeg_10s_from_eeg_v1(self, p, o): extracted = self._extract_raw(p, o, 10); return extracted if extracted is not None else np.zeros((200, 160), dtype=np.float32)

    def _extract_stft_v1(self, parquet_path, offset):
        EEG_LENGTH = 50
        try: eeg = pd.read_parquet(parquet_path)
        except Exception as e: return np.zeros((512, 1024), dtype=np.float32)
        time_temp=offset*200; time_start=round(time_temp+(50-EEG_LENGTH)/2*200); time_stop=round(time_temp+(50+EEG_LENGTH)/2*200)
        eeg = eeg.iloc[time_start:time_stop]; list_eeg = list()
        for k in range(4):
            COLS = self.FEATS[k]; spect = np.zeros((128,256,4),dtype='float32')
            for kk in range(4):
                eeg_1 = eeg[COLS[kk]]; mean_1 = eeg_1.mean(); eeg_1.fillna(value=mean_1, inplace=True); eeg_1 = eeg_1.values
                eeg_2 = eeg[COLS[kk+1]]; mean_2 = eeg_2.mean(); eeg_2.fillna(value=mean_2, inplace=True); eeg_2 = eeg_2.values
                new_eeg = eeg_1 - eeg_2; del eeg_1, eeg_2
                fs = 200; nperseg = 39; noverlap = 0; f, t, spec = signal.spectrogram(new_eeg, fs, nperseg=nperseg, noverlap=noverlap, nfft=256)
                spec = np.abs(spec); spec = np.log1p(spec).astype("float32"); spect[:,:,kk] += spec[:128, :]
            spect = np.concatenate(tuple(spect[:,:,i] for i in range(4)), 1); list_eeg.append(spect)
        spect = np.concatenate(list_eeg, 0); spect /= 2; return spect

    def _extract_stft_v2(self, parquet_path, offset):
        EEG_LENGTH = 50
        try: eeg = pd.read_parquet(parquet_path)
        except Exception as e: return np.zeros((512, 256), dtype=np.float32)
        time_temp=offset*200; time_start=round(time_temp+(50-EEG_LENGTH)/2*200); time_stop=round(time_temp+(50+EEG_LENGTH)/2*200)
        eeg = eeg.iloc[time_start:time_stop]; spect = np.zeros((128,256,4),dtype='float32')
        for k in range(4):
            COLS = self.FEATS[k]
            for kk in range(4):
                eeg_1 = eeg[COLS[kk]]; mean_1 = eeg_1.mean(); eeg_1.fillna(value=mean_1, inplace=True); eeg_1 = eeg_1.values
                eeg_2 = eeg[COLS[kk+1]]; mean_2 = eeg_2.mean(); eeg_2.fillna(value=mean_2, inplace=True); eeg_2 = eeg_2.values
                new_eeg = eeg_1 - eeg_2; del eeg_1, eeg_2
                fs = 200; nperseg = len(new_eeg)//256 if len(new_eeg)//256 > 0 else 1 ; noverlap = 0
                f, t, spec = signal.stft(new_eeg, fs, nperseg=nperseg, noverlap=noverlap, nfft=256)
                spec = np.abs(spec); spec = np.log1p(spec).astype("float32"); spect[:,:,k] += spec[:128, 1:257] if spec.shape[1] > 1 else np.zeros((128, 256))
            spect[:,:,k] /= 4
        spect = np.concatenate(tuple(spect[:,:,i] for i in range(4)), 0); return spect

    def stft_spec_from_50s_eeg_v1(self, p, o): return self._extract_stft_v1(p,o)
    def stft_spec_from_50s_eeg_v2(self, p, o): return self._extract_stft_v2(p,o)
    # --- End Feature extraction functions ---

    def __getitem__(self, idx):
        eeg_id, label, offset = self.metaDF.iloc[idx][["eeg_id", "expert_consensus", "eeg_label_offset_seconds"]]
        ppath = f'{self.base_dir}train_eegs/{eeg_id}.parquet'
        Xe50=self.raweeg_50s_from_eeg_v1(ppath, offset); Xe20=self.raweeg_20s_from_eeg_v1(ppath, offset); Xe10=self.raweeg_10s_from_eeg_v1(ppath, offset)
        Xe = np.concatenate((Xe50, self.Epartition, Xe20, self.Epartition, Xe10), 1); del Xe50, Xe20, Xe10
        Xs50_1=self.stft_spec_from_50s_eeg_v1(ppath, offset); Xs50_2=self.stft_spec_from_50s_eeg_v2(ppath, offset)
        Xs = np.concatenate((Xs50_1, self.Spartition, Xs50_2), 1); del Xs50_1, Xs50_2
        # Final X shape = (728, 1296) -> (C, T)
        X = np.concatenate((Xe, self.Partition, Xs), 0)
        # --- NO TRANSPOSE ---
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y = self.activity_mapping[label]
        y_tensor = torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.long), num_classes=6).float()
        return X_tensor, y_tensor


# In[ ]:


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20, device="cuda", checkpoint_path="best_model.pth", log_file_path="training_log.csv"):
    model.to(device)
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Created dir: {checkpoint_dir}")

    if os.path.exists(checkpoint_path):
        try:
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            print(f"Loaded checkpoint: {checkpoint_path}")
        except Exception as e:
            print(f"Error loading checkpoint: {e}. Starting fresh.")
    else:
        print(f"No checkpoint found at {checkpoint_path}. Starting fresh.")

    best_val_accuracy = 0.0
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        print(f"Created dir: {log_dir}")

    if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0:
        with open(log_file_path, "w") as f:
            f.write("epoch#,train_loss,train_accuracy,val_loss,val_accuracy,time_taken\n")

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss, correct, total = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for i, (X, y) in enumerate(train_pbar):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, torch.argmax(y, dim=1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(torch.argmax(y, dim=1)).sum().item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100. * correct / total:.2f}%")

        train_accuracy = 100. * correct / total if total > 0 else 0.0
        avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")

        model.eval()
        val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            for X, y in val_pbar:
                X, y = X.to(device), y.to(device)
                outputs = model(X)
                loss = criterion(outputs, torch.argmax(y, dim=1))
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(torch.argmax(y, dim=1)).sum().item()
                val_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100. * correct / total:.2f}%")

        val_accuracy = 100. * correct / total if total > 0 else 0.0
        avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        try:
            scheduler.step(avg_val_loss)
        except Exception:
            pass

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved: {checkpoint_path} @ Val Acc: {val_accuracy:.2f}%")

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} Time: {epoch_time:.2f}s")

        with open(log_file_path, "a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{train_accuracy:.2f},{avg_val_loss:.4f},{val_accuracy:.2f},{epoch_time:.2f}\n")

    print(f"Training complete. Best Val Acc: {best_val_accuracy:.2f}%")

def test_model(model, test_loader, checkpoint_path="best_model.pth", device="cuda"):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return {}

    print(f"Loading best model: {checkpoint_path}")
    try:
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    except Exception as e:
        print(f"Error loading state_dict: {e}")
        return {}

    model.to(device)
    model.eval()
    mname = os.path.splitext(checkpoint_path)[0]
    plot_dir = os.path.dirname(mname)
    if plot_dir and not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Created plot dir: {plot_dir}")

    all_preds, all_labels, all_probs = [], [], []
    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Testing"):
            X, y = X.to(device), y.to(device)
            start_time = time.time()
            outputs = model(X)
            end_time = time.time()
            batch_time = end_time - start_time
            total_time += batch_time
            total_samples += X.size(0)

            probs = torch.nn.functional.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(torch.argmax(y, dim=1).cpu().numpy())
            all_probs.extend(probs.cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=brain_activities, yticklabels=brain_activities)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {mname}")
    plt.savefig(f"{mname}_confusion_matrix.png")
    plt.close()
    print(f"Confusion matrix saved: {mname}_confusion_matrix.png")

    seizure_class_idx = activity_mapping.get('Seizure', -1)
    if seizure_class_idx != -1 and seizure_class_idx < cm.shape[0]:
        TP = cm[seizure_class_idx][seizure_class_idx]
        FN = sum(cm[seizure_class_idx]) - TP
        seizure_sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    else:
        seizure_sensitivity = None

    num_classes_cm = cm.shape[0]
    specificity_list = []
    for i in range(num_classes_cm):
        TP = cm[i, i]
        FP = cm[:, i].sum() - TP
        FN = cm[i, :].sum() - TP
        TN = cm.sum() - (TP + FP + FN)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
        specificity_list.append(specificity)
    avg_specificity = np.mean(specificity_list) if specificity_list else 0.0

    num_classes = all_probs.shape[1] if all_probs.size else 0
    auc_scores = []
    plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        if i in all_labels:
            fpr, tpr, _ = roc_curve((all_labels == i).astype(int), all_probs[:, i])
            roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc)
            plt.plot(fpr, tpr, label=f'{brain_activities[i]} (AUC = {roc_auc:.2f})')
        else:
            print(f"Skipping ROC for {brain_activities[i]}")
            auc_scores.append(np.nan)

    macro_auc = np.nanmean(auc_scores) if auc_scores else np.nan
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curve - {mname}")
    plt.legend()
    plt.savefig(f"{mname}_roc_curve.png")
    plt.close()
    print(f"ROC curve saved: {mname}_roc_curve.png")

    avg_inference_time = total_time / total_samples if total_samples > 0 else 0.0

    print("--- Test Results ---")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    print(f"Cohen's Kappa: {kappa:.2f}")
    print(f"Macro AUC: {macro_auc:.2f}" if not np.isnan(macro_auc) else "Macro AUC: nan")
    print(f"Specificity: {avg_specificity:.2%}")
    if seizure_sensitivity is not None:
        print(f"Seizure Sensitivity: {seizure_sensitivity:.2%}")
    else:
        print("Seizure Sensitivity: Class 'Seizure' not in test set.")
    print(f"Avg Inference Time: {avg_inference_time * 1000:.2f} ms")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "kappa": kappa,
        "macro_auc": macro_auc,
        "specificity": avg_specificity,
        "seizure_sensitivity": seizure_sensitivity,
        "avg_inference_time_sec": avg_inference_time
    }

print("Train and Test functions defined.")


# In[ ]:


# Instantiate Datasets
try:
    train_dataset = HMS_Dataset(metaDF=trainDF, base_dir=BASE_DIR, activity_mapping=activity_mapping)
    val_dataset = HMS_Dataset(metaDF=valDF, base_dir=BASE_DIR, activity_mapping=activity_mapping)
    test_dataset = HMS_Dataset(metaDF=testDF, base_dir=BASE_DIR, activity_mapping=activity_mapping)

    # --- Batch Size set to 1 ---
    BatchSize = 1

    num_workers = min(os.cpu_count(), 8); print(f"Using {num_workers} workers.")
    train_loader = DataLoader(train_dataset, batch_size=BatchSize, shuffle=True, num_workers=num_workers, pin_memory=True, prefetch_factor=2 if num_workers > 0 else None)
    val_loader = DataLoader(val_dataset, batch_size=BatchSize, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2 if num_workers > 0 else None)
    test_loader = DataLoader(test_dataset, batch_size=BatchSize, shuffle=False, num_workers=num_workers, pin_memory=True, prefetch_factor=2 if num_workers > 0 else None)

    print(f"DataLoaders created with BatchSize = {BatchSize}.")

    # Check a batch shape
    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch X shape: {X_batch.shape}") # Should be [1, 728, 1296] -> (B, C, T)
    print(f"Batch y shape: {y_batch.shape}") # Should be [1, 6]
except Exception as e:
    print(f"Error creating DataLoaders: {e}")


# In[ ]:


DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

class GraphConvolution(Module):
    """ simple GCN layer """
    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features; self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        torch.nn.init.xavier_uniform_(self.weight, gain=1.414)
        if bias: self.bias = Parameter(torch.zeros((1, 1, out_features), dtype=torch.float32))
        else: self.register_parameter('bias', None)
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1)); self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None: self.bias.data.uniform_(-stdv, stdv)
    def forward(self, x, adj):
        # x shape: [B, N_Nodes, N_Features_In]
        # adj shape: [B, N_Nodes, N_Nodes]
        # weight shape: [N_Features_In, N_Features_Out]
        # bias shape: [1, 1, N_Features_Out]
        support = torch.matmul(x, self.weight) # [B, N_Nodes, N_Features_Out]
        output = torch.matmul(adj, support) # [B, N_Nodes, N_Features_Out]
        if self.bias is not None: output = output + self.bias
        # Original code had Relu here and bias subtract - let's match paper's intent more closely
        # output = F.relu(torch.matmul(adj, output)) # Original had bias subtract before matmul
        return F.relu(output)

class PowerLayer(nn.Module):
    ''' The power layer: calculates the log-transformed power '''
    def __init__(self, dim, length, step):
        super(PowerLayer, self).__init__()
        self.dim = dim
        # Use AvgPool2d for compatibility with Conv2d output (B, C, H, W)
        self.pooling = nn.AvgPool2d(kernel_size=(1, length), stride=(1, step))
    def forward(self, x):
        # x shape: [B, C, H=1, W=Time]
        # Apply power first, then pool, then log
        x_pooled = self.pooling(x.pow(2))
        # Add epsilon for numerical stability before log
        return torch.log(x_pooled + 1e-6)

class Aggregator():
    ''' Aggregates channels within predefined brain areas using mean pooling. '''
    def __init__(self, idx_area_channel_counts):
        # idx_area_channel_counts: a list of channel counts per area, e.g., [91, 91, ...]
        self.area_channel_counts = idx_area_channel_counts
        # Calculate cumulative indices for slicing
        self.slice_indices = self.get_slice_indices(idx_area_channel_counts)
        self.num_areas = len(idx_area_channel_counts)
        print(f"Aggregator initialized for {self.num_areas} areas.")
        print(f"Slice indices: {self.slice_indices}")

    def forward(self, x):
        # x: batch x channel x data_dim (e.g., [B, 728, Feature])
        batch_size = x.shape[0]
        feature_dim = x.shape[2]
        aggregated_data = torch.zeros(batch_size, self.num_areas, feature_dim, device=x.device)

        for i in range(self.num_areas):
            start_idx = self.slice_indices[i]
            end_idx = self.slice_indices[i+1]
            # Aggregate channels for the current area
            aggregated_data[:, i, :] = torch.mean(x[:, start_idx:end_idx, :], dim=1)

        return aggregated_data # shape: [B, num_areas, Feature]

    def get_slice_indices(self, channel_counts):
        indices = [0] * (len(channel_counts) + 1)
        current_idx = 0
        for i, count in enumerate(channel_counts):
            current_idx += count
            indices[i+1] = current_idx
        return indices

    # Removed aggr_fun as it's integrated into forward

print("LGGNet dependencies defined.")


# In[ ]:


class LGGNet(nn.Module):
    def temporal_learner(self, in_chan, out_chan, kernel, pool, pool_step_rate):
        # Use nn.Conv2d with H=1 for temporal convolution
        # Input: (B, Freq=1, C, T) -> Output: (B, num_T, C, T_pooled)
        # Added padding='same' for simplicity, check if original intent was different
        return nn.Sequential(
            nn.Conv2d(in_chan, out_chan, kernel_size=kernel, stride=(1, 1), padding='same'), # Use 'same' padding
            PowerLayer(dim=-1, length=pool, step=int(pool_step_rate*pool))
        )

    def __init__(self, num_classes, input_size, sampling_rate, num_T,
                 out_graph, dropout_rate, pool, pool_step_rate, idx_graph):
        # input_size: (Freq=1, Channels=C, Time=T)
        super(LGGNet, self).__init__()
        self.idx = idx_graph # List of channel counts per brain area
        self.window = [0.5, 0.25, 0.125]
        self.pool = pool
        self.channel = input_size[1] # C
        self.brain_area = len(self.idx) # Number of brain areas
        self.input_size_for_calc = input_size # Store for get_size method

        # Temporal Convolution Layers (input_size[0] is Freq=1)
        self.Tception1 = self.temporal_learner(1, num_T, (1, int(self.window[0] * sampling_rate)), self.pool, pool_step_rate)
        self.Tception2 = self.temporal_learner(1, num_T, (1, int(self.window[1] * sampling_rate)), self.pool, pool_step_rate)
        self.Tception3 = self.temporal_learner(1, num_T, (1, int(self.window[2] * sampling_rate)), self.pool, pool_step_rate)

        self.BN_t = nn.BatchNorm2d(num_T) # Operates on num_T channels

        # Corrected 1x1 Conv Path (Based on reinterpretation)
        self.one_by_one_conv = nn.Conv2d(num_T, num_T, kernel_size=(1, 1))
        self.leaky_relu = nn.LeakyReLU()
        self.avg_pool_1x1 = nn.AvgPool2d((1, 2)) # Pool time dim after 1x1
        self.BN_t_ = nn.BatchNorm2d(num_T) # Operates on num_T channels

        # Calculate output size *after* temporal blocks AND corrected 1x1 conv path
        # Need to calculate this *before* defining layers that depend on it
        size, calculated_feature_dim = self.get_size_after_temporal_and_feature_dim()
        print(f"Calculated feature dimension after temporal blocks: {calculated_feature_dim}")


        # Local Graph Filter (applied after reshaping)
        # Weight shape needs to match feature dim per channel
        self.local_filter_weight = nn.Parameter(torch.FloatTensor(self.channel, calculated_feature_dim), requires_grad=True)
        nn.init.xavier_uniform_(self.local_filter_weight)
        self.local_filter_bias = nn.Parameter(torch.zeros((1, self.channel, 1), dtype=torch.float32), requires_grad=True)

        # Aggregate channels into brain areas
        self.aggregate = Aggregator(self.idx)

        # Global Graph Network
        self.global_adj = nn.Parameter(torch.FloatTensor(self.brain_area, self.brain_area), requires_grad=True)
        nn.init.xavier_uniform_(self.global_adj)
        self.bn1 = nn.BatchNorm1d(self.brain_area) # BN before GCN
        # GCN input features = calculated_feature_dim
        self.gcn = GraphConvolution(calculated_feature_dim, out_graph)
        self.bn2 = nn.BatchNorm1d(self.brain_area) # BN after GCN

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(int(self.brain_area * out_graph), num_classes))

    def get_size_after_temporal_and_feature_dim(self):
        # input_size: (Freq, C, T) - Use stored value
        input_size = self.input_size_for_calc
        # Create dummy data on the correct device (assuming model is already on a device or will be)
        device = next(self.parameters()).device if len(list(self.parameters())) > 0 else torch.device(DEVICE) # Handle case before model moved
        data = torch.ones((1, input_size[0], input_size[1], int(input_size[2])), device=device) # <-- MOVE TO DEVICE

        with torch.no_grad():
            y1 = self.Tception1(data) # (B, num_T, C, T1)
            y2 = self.Tception2(data) # (B, num_T, C, T2)
            y3 = self.Tception3(data) # (B, num_T, C, T3)
            out = torch.cat((y1, y2, y3), dim=-1) # (B, num_T, C, T_cat)
            out = self.BN_t(out)
            # Apply corrected 1x1 path
            out = self.one_by_one_conv(out)
            out = self.leaky_relu(out)
            out = self.avg_pool_1x1(out) # (B, num_T, C, F=T_cat/2)
            # BN_t_ is not needed for size calculation

            # Feature dim calculation:
            # Reshape will be (B, C, num_T * F)
            feature_dim = out.shape[1] * out.shape[3] # num_T * F

        return out.size(), feature_dim # Return [1, num_T, C, F], feature_dim

    def local_filter_fun(self, x, w):
        # x: [B, C, Feature]
        # w: [C, Feature] -> unsqueeze to [1, C, Feature] -> repeat to [B, C, Feature]
        w_repeated = w.unsqueeze(0).expand(x.size(0), -1, -1)
        filtered_x = F.relu(torch.mul(x, w_repeated) + self.local_filter_bias)
        return filtered_x

    def get_adj(self, x, self_loop=True):
        # x: [B, N_Nodes=brain_area, Feature]
        adj = self.self_similarity(x); num_nodes = adj.shape[-1]
        learned_adj = self.global_adj + self.global_adj.transpose(0, 1)
        adj = F.relu(adj * learned_adj.unsqueeze(0))
        if self_loop: adj = adj + torch.eye(num_nodes, device=x.device).unsqueeze(0)
        rowsum = torch.sum(adj, dim=-1)
        mask = torch.zeros_like(rowsum); mask[rowsum == 0] = 1; rowsum += mask
        d_inv_sqrt = torch.pow(rowsum, -0.5); d_mat_inv_sqrt = torch.diag_embed(d_inv_sqrt)
        adj_normalized = torch.bmm(torch.bmm(d_mat_inv_sqrt, adj), d_mat_inv_sqrt)
        return adj_normalized

    def self_similarity(self, x):
        x_ = x.permute(0, 2, 1); s = torch.bmm(x, x_); return s

    def forward(self, x):
        # x: (B, C, T) -> (1, 728, 1296)
        x = torch.unsqueeze(x, dim=1)  # (B, 1, C, T)

        # Temporal Feature Extraction
        y1 = self.Tception1(x); y2 = self.Tception2(x); y3 = self.Tception3(x)
        out = torch.cat((y1, y2, y3), dim=-1) # (B, num_T, C, T_cat)

        # Corrected 1x1 Conv Path
        out = self.BN_t(out)
        out = self.one_by_one_conv(out)
        out = self.leaky_relu(out)
        out = self.avg_pool_1x1(out) # (B, num_T, C, F=T_cat/2)
        out = self.BN_t_(out) # (B, num_T, C, F)

        # Reshape for Local/Global Graph
        # Permute: (B, C, num_T, F)
        out = out.permute(0, 2, 1, 3)
        # Reshape: (B, C, num_T * F = FeatureDim)
        out = torch.reshape(out, (out.size(0), self.channel, -1))

        # Apply Local Filter
        # Ensure weight device matches input device if weights were initialized before model.to(device)
        out = self.local_filter_fun(out, self.local_filter_weight.to(out.device))

        # Aggregate Channels -> Brain Areas
        out = self.aggregate.forward(out) # (B, BrainArea, FeatureDim)

        # Global Graph Network
        adj = self.get_adj(out) # Calculate adjacency
        out = self.bn1(out)
        out = self.gcn(out, adj) # Apply GCN -> (B, BrainArea, out_graph)
        out = self.bn2(out)

        # Flatten and Classify
        out = out.view(out.size()[0], -1) # (B, BrainArea * out_graph)
        out = self.fc(out)
        return out

print("New Model (LGGNet) class defined with device fix in size calculation.")


# In[ ]:


# --- Data shapes ---
N_CHAN = 728
N_TIME = 1296
N_CLASS = 6
N_FREQ = 1 # Input has 1 frequency dimension added by model

# --- LGGNet Hyperparameters ---
SAMPLING_RATE = SFREQ # From global config (200 Hz)
NUM_T = 16          # Number of temporal filters (kernels) in Tception blocks
OUT_GRAPH = 32      # Output feature dimension from GCN
DROPOUT_RATE = 0.3
POOL = 16           # Average pooling kernel size in PowerLayer
POOL_STEP_RATE = 0.25 # Step rate for pooling in PowerLayer

# --- Brain Area Definition (Crucial Assumption) ---
# Divide 728 channels into 8 roughly equal areas
num_areas = 8
channels_per_area = N_CHAN // num_areas
remainder = N_CHAN % num_areas
# Distribute remainder
IDX_GRAPH = [channels_per_area + 1] * remainder + [channels_per_area] * (num_areas - remainder)
if sum(IDX_GRAPH) != N_CHAN: # Sanity check
     raise ValueError(f"Channel counts in IDX_GRAPH ({sum(IDX_GRAPH)}) don't match N_CHAN ({N_CHAN})")
print(f"Assuming {num_areas} brain areas with channel counts: {IDX_GRAPH}")


# Create configs Namespace (optional, but keeps pattern)
configs = Namespace()
configs.num_classes = N_CLASS
configs.input_size = (N_FREQ, N_CHAN, N_TIME)
configs.sampling_rate = SAMPLING_RATE
configs.num_T = NUM_T
configs.out_graph = OUT_GRAPH
configs.dropout_rate = DROPOUT_RATE
configs.pool = POOL
configs.pool_step_rate = POOL_STEP_RATE
configs.idx_graph = IDX_GRAPH


print("\nModel Configurations for LGGNet:")
print({k: v for k, v in vars(configs).items()})


# In[ ]:


# Cell 9: Model Setup

device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using Device: {device}")

# Instantiate the LGGNet model
try:
    # Need to pass unpacked args from configs
    model = LGGNet(
        num_classes=configs.num_classes,
        input_size=configs.input_size,
        sampling_rate=configs.sampling_rate,
        num_T=configs.num_T,
        out_graph=configs.out_graph,
        dropout_rate=configs.dropout_rate,
        pool=configs.pool,
        pool_step_rate=configs.pool_step_rate,
        idx_graph=configs.idx_graph
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"New model (LGGNet) instantiated. Trainable params: {total_params/1e6:.2f}M")

    # --- Add dynamic adjustment for local_filter_weight ---
    # Recalculate the expected feature dimension after temporal block and 1x1 conv path
    print("Adjusting local_filter_weight size...")

    # === FIX: Call the correct function name ===
    # The function now returns (size_tuple, feature_dim_int)
    _, calculated_feature_dim = model.get_size_after_temporal_and_feature_dim() 
    # === END FIX ===

    current_weight_dim = model.local_filter_weight.shape[1]

    if current_weight_dim != calculated_feature_dim:
        print(f"Warning: local_filter_weight dim mismatch. Expected {calculated_feature_dim}, got {current_weight_dim}. Re-initializing.")
        # Reinitialize the parameter with the correct size on the correct device
        model.local_filter_weight = nn.Parameter(torch.FloatTensor(N_CHAN, calculated_feature_dim), requires_grad=True).to(device)
        nn.init.xavier_uniform_(model.local_filter_weight)
        # Bias shape is independent of feature dim, should be okay.
    else:
        print("local_filter_weight size is correct.")
    # --- End adjustment ---


except ValueError as ve: # Catch specific ValueErrors
    print("="*50); print(f"ERROR initializing model: {ve}"); print("="*50); raise ve
except Exception as e: print("="*50); print(f"ERROR initializing model: {e}"); print("="*50); raise e

# Setup loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

print("New model, criterion, optimizer, and scheduler set up.")


# In[ ]:


CHECKPOINT_PATH = "LGGNet_v1.pth"; LOG_FILE_PATH = "LGGNet_v1_log.csv"
print(f"Starting training for new model: {CHECKPOINT_PATH}")
try:
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                epochs=12, device=device, checkpoint_path=CHECKPOINT_PATH, log_file_path=LOG_FILE_PATH)
except RuntimeError as rte:
     print(f"\nAn error occurred during training: {rte}")
     if "CUDA out of memory" in str(rte): print(">>> CUDA OOM. Try reducing 'num_T' or 'out_graph' in Cell 8. <<<")
     elif "size mismatch" in str(rte): print(">>> Size Mismatch Error. Check model layer dimensions carefully, especially around GCN. <<<")
     else: raise rte
except ValueError as ve:
     print(f"\nAn error occurred during training: {ve}")
     if "Channel counts in IDX_GRAPH" in str(ve): print(">>> Config Error: Check IDX_GRAPH sums correctly in Cell 8. <<<")
     else: raise ve
except Exception as e: print(f"\nAn unexpected error: {e}"); raise e


# In[ ]:


print(f"\nTesting model from {CHECKPOINT_PATH}...")
try:
    # Instantiate a new model instance for testing
    test_model_instance = LGGNet(
        num_classes=configs.num_classes, input_size=configs.input_size, sampling_rate=configs.sampling_rate,
        num_T=configs.num_T, out_graph=configs.out_graph, dropout_rate=configs.dropout_rate,
        pool=configs.pool, pool_step_rate=configs.pool_step_rate, idx_graph=configs.idx_graph
    ).to(device)

    # Adjust weight size in the test instance too, just in case calculation differs slightly (though it shouldn't)
    with torch.no_grad():
        test_calc_feat_dim = test_model_instance.get_size_after_temporal(configs.input_size)[-1]
        if test_model_instance.local_filter_weight.shape[1] != test_calc_feat_dim:
             print("Adjusting test model local_filter_weight size...")
             test_model_instance.local_filter_weight = nn.Parameter(torch.FloatTensor(N_CHAN, test_calc_feat_dim), requires_grad=False).to(device)


    test_results = test_model(test_model_instance, test_loader, checkpoint_path=CHECKPOINT_PATH, device=device)
    print("\n--- Final Test Results Summary (LGGNet) ---"); print(test_results)
except FileNotFoundError: print(f"Checkpoint {CHECKPOINT_PATH} not found. Skipping test.")
except ValueError as ve: print(f"A configuration error occurred during testing setup: {ve}")
except RuntimeError as rte: print(f"A runtime error occurred during testing: {rte}")
except Exception as e: print(f"An unexpected error during testing: {e}")


# In[ ]:




