#!/usr/bin/env python
# coding: utf-8

# In[1]:


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

from argparse import Namespace
import math

from tqdm.auto import tqdm
import copy

try:
    from einops import rearrange
    from einops.layers.torch import Rearrange
except ImportError:
    print("einops not found (needed for Deformer, not TSception).")
from torch.nn.parameter import Parameter
from torch.nn.modules.module import Module


warnings.filterwarnings('ignore')

print("All libraries imported.")


# In[2]:


# %%
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


# In[3]:


# %%
class HMS_Dataset(Dataset):
    def __init__(self, metaDF, base_dir, activity_mapping):
        self.metaDF = metaDF; self.base_dir = base_dir; self.activity_mapping = activity_mapping
        self.RAW_FEATURES = {'LL': ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1'], 'RL': ['Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2'], 'LP': ['Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1'], 'RP': ['Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2']}
        self.FEATS = [['Fp1','F7','T3','T5','O1'], ['Fp1','F3','C3','P3','O1'], ['Fp2','F8','T4','T6','O2'], ['Fp2','F4','C4','P4','O2']]
        self.Epartition = np.zeros((200, 8), dtype='float32'); self.Spartition = np.zeros((512, 16), dtype='float32'); self.Partition = np.zeros((16, 1296), dtype='float32')
        print(f"Dataset Initialized. Output shape (C, T) = (728, 1296).")

    def __len__(self): return len(self.metaDF)

    # --- Feature extraction functions ---
    def _extract_raw(self, parquet_path, offset, length):
        try: raw_eeg = pd.read_parquet(parquet_path)
        except Exception as e: return None
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
        EEG_LENGTH = 50; 
        try: 
            eeg = pd.read_parquet(parquet_path); 
        except Exception as e: 
            return np.zeros((512, 1024), dtype=np.float32)
        time_temp=offset*200; time_start=round(time_temp+(50-EEG_LENGTH)/2*200); time_stop=round(time_temp+(50+EEG_LENGTH)/2*200); eeg = eeg.iloc[time_start:time_stop]; list_eeg = list()
        for k in range(4):
            COLS = self.FEATS[k]; spect = np.zeros((128,256,4),dtype='float32')
            for kk in range(4):
                eeg_1 = eeg[COLS[kk]]; mean_1 = eeg_1.mean(); eeg_1.fillna(value=mean_1, inplace=True); eeg_1 = eeg_1.values; eeg_2 = eeg[COLS[kk+1]]; mean_2 = eeg_2.mean(); eeg_2.fillna(value=mean_2, inplace=True); eeg_2 = eeg_2.values
                new_eeg = eeg_1 - eeg_2; del eeg_1, eeg_2; fs = 200; nperseg = 39; noverlap = 0; f, t, spec = signal.spectrogram(new_eeg, fs, nperseg=nperseg, noverlap=noverlap, nfft=256)
                spec = np.abs(spec); spec = np.log1p(spec).astype("float32"); spect[:,:,kk] += spec[:128, :]
            spect = np.concatenate(tuple(spect[:,:,i] for i in range(4)), 1); list_eeg.append(spect)
        spect = np.concatenate(list_eeg, 0); spect /= 2; return spect

    def _extract_stft_v2(self, parquet_path, offset):
        EEG_LENGTH = 50; 
        try: 
            eeg = pd.read_parquet(parquet_path); 
        except Exception as e: 
            return np.zeros((512, 256), dtype=np.float32)

        time_temp=offset*200; time_start=round(time_temp+(50-EEG_LENGTH)/2*200); time_stop=round(time_temp+(50+EEG_LENGTH)/2*200); eeg = eeg.iloc[time_start:time_stop]; spect = np.zeros((128,256,4),dtype='float32')
        for k in range(4):
            COLS = self.FEATS[k]
            for kk in range(4):
                eeg_1 = eeg[COLS[kk]]; mean_1 = eeg_1.mean(); eeg_1.fillna(value=mean_1, inplace=True); eeg_1 = eeg_1.values; eeg_2 = eeg[COLS[kk+1]]; mean_2 = eeg_2.mean(); eeg_2.fillna(value=mean_2, inplace=True); eeg_2 = eeg_2.values
                new_eeg = eeg_1 - eeg_2; del eeg_1, eeg_2; fs = 200; nperseg = len(new_eeg)//256 if len(new_eeg)//256 > 0 else 1 ; noverlap = 0
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


# In[4]:


# %%
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20, device="cuda", checkpoint_path="best_model.pth", log_file_path="training_log.csv"):
    model.to(device) # Ensure model is on the correct device at the start
    checkpoint_dir = os.path.dirname(checkpoint_path);
    if checkpoint_dir and not os.path.exists(checkpoint_dir): os.makedirs(checkpoint_dir, exist_ok=True); print(f"Created dir: {checkpoint_dir}")
    if os.path.exists(checkpoint_path): print(f"Loading checkpoint: {checkpoint_path}"); 
    try: 
        model.load_state_dict(torch.load(checkpoint_path, map_location=device)); 
    except Exception as e: 
        print(f"Error loading: {e}. Starting fresh.")
    else: print(f"Starting fresh. Checkpoint: {checkpoint_path}")

    best_val_accuracy = 0.0; log_dir = os.path.dirname(log_file_path);
    if log_dir and not os.path.exists(log_dir): os.makedirs(log_dir, exist_ok=True); print(f"Created dir: {log_dir}")
    if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0:
        with open(log_file_path,"w") as f: f.write("epoch#,train_loss,train_accuracy,val_loss,val_accuracy,time_taken\n")
    for epoch in range(epochs):
        start_time = time.time(); model.train(); train_loss, correct, total = 0.0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for i, (X, y) in enumerate(train_pbar):
            X, y = X.to(device), y.to(device); optimizer.zero_grad()
            outputs = model(X) # <-- Simple model call
            loss = criterion(outputs, torch.argmax(y, dim=1)); loss.backward(); optimizer.step()
            train_loss += loss.item(); _, predicted = outputs.max(1); total += y.size(0); correct += predicted.eq(torch.argmax(y, dim=1)).sum().item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%" if total > 0 else "0.00%")
        train_accuracy = 100. * correct / total if total > 0 else 0.0; avg_train_loss = train_loss / len(train_loader) if len(train_loader) > 0 else 0.0
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")
        model.eval(); val_loss, correct, total = 0.0, 0, 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            for X, y in val_pbar:
                X, y = X.to(device), y.to(device)
                outputs = model(X) # <-- Simple model call
                loss = criterion(outputs, torch.argmax(y, dim=1)); val_loss += loss.item()
                _, predicted = outputs.max(1); total += y.size(0); correct += predicted.eq(torch.argmax(y, dim=1)).sum().item()
                val_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%" if total > 0 else "0.00%")
        val_accuracy = 100. * correct / total if total > 0 else 0.0; avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%");
        # Add safeguard for scheduler step
        try: scheduler.step(avg_val_loss)
        except Exception as e: print(f"Scheduler step error: {e}")
        if val_accuracy > best_val_accuracy: best_val_accuracy = val_accuracy; torch.save(model.state_dict(), checkpoint_path); print(f"Best model saved: {checkpoint_path} @ Val Acc: {val_accuracy:.2f}%")
        epoch_time = time.time() - start_time; print(f"Epoch {epoch+1} Time: {epoch_time:.2f}s")
        with open(log_file_path,"a") as f: f.write(f"{epoch+1},{avg_train_loss:.4f},{train_accuracy:.2f},{avg_val_loss:.4f},{val_accuracy:.2f},{epoch_time:.2f}\n")
    print(f"Training complete. Best Val Acc: {best_val_accuracy:.2f}%")

def test_model(model, test_loader, checkpoint_path="best_model.pth", device="cuda"):
    if not os.path.exists(checkpoint_path): print(f"Error: Checkpoint not found: {checkpoint_path}"); return {}
    print(f"Loading best model: {checkpoint_path}");
    try: 
        model.load_state_dict(torch.load(checkpoint_path, map_location=device));
    except Exception as e: print(f"Error loading state_dict: {e}"); return {}

    model.to(device); model.eval(); mname = os.path.splitext(checkpoint_path)[0]; plot_dir = os.path.dirname(mname);
    if plot_dir and not os.path.exists(plot_dir): os.makedirs(plot_dir, exist_ok=True); print(f"Created plot dir: {plot_dir}")
    all_preds, all_labels, all_probs = [], [], []; total_time = 0.0; total_samples = 0
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Testing"):
            X, y = X.to(device), y.to(device); start_time = time.time()
            outputs = model(X) # <-- Simple model call
            end_time = time.time(); batch_time = end_time - start_time; total_time += batch_time; total_samples += X.size(0)
            probs = torch.nn.functional.softmax(outputs, dim=1); _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy()); all_labels.extend(torch.argmax(y, dim=1).cpu().numpy()); all_probs.extend(probs.cpu().numpy())
    all_preds = np.array(all_preds); all_labels = np.array(all_labels); all_probs = np.array(all_probs)
    accuracy=accuracy_score(all_labels, all_preds); precision=precision_score(all_labels, all_preds, average='weighted', zero_division=0); recall=recall_score(all_labels, all_preds, average='weighted', zero_division=0); f1=f1_score(all_labels, all_preds, average='weighted', zero_division=0); kappa=cohen_kappa_score(all_labels, all_preds); cm=confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=brain_activities, yticklabels=brain_activities); plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion Matrix - {mname}"); plt.savefig(f"{mname}_confusion_matrix.png"); plt.close(); print(f"Confusion matrix saved: {mname}_confusion_matrix.png")
    seizure_class_idx = activity_mapping.get('Seizure', -1)
    if seizure_class_idx != -1 and seizure_class_idx < cm.shape[0]: TP = cm[seizure_class_idx][seizure_class_idx]; FN = sum(cm[seizure_class_idx]) - TP; seizure_sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    else: seizure_sensitivity = None
    num_classes_cm = cm.shape[0]; specificity_list = []
    for i in range(num_classes_cm): TP = cm[i, i]; FP = cm[:, i].sum() - TP; FN = cm[i, :].sum() - TP; TN = cm.sum() - (TP + FP + FN); specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0; specificity_list.append(specificity)
    avg_specificity = np.mean(specificity_list) if specificity_list else 0.0; num_classes = all_probs.shape[1] if all_probs.size > 0 else 0; auc_scores = []; plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        if i in all_labels: fpr, tpr, _ = roc_curve((all_labels == i).astype(int), all_probs[:, i]); roc_auc = auc(fpr, tpr); auc_scores.append(roc_auc); plt.plot(fpr, tpr, label=f'{brain_activities[i]} (AUC = {roc_auc:.2f})')
        else: print(f"Skipping ROC for {brain_activities[i]}"); auc_scores.append(np.nan)
    macro_auc = np.nanmean(auc_scores) if auc_scores else np.nan
    plt.plot([0, 1], [0, 1], 'k--'); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC Curve - {mname}"); plt.legend(); plt.savefig(f"{mname}_roc_curve.png"); plt.close(); print(f"ROC curve saved: {mname}_roc_curve.png")
    avg_inference_time = total_time / total_samples if total_samples > 0 else 0.0
    print("--- Test Results ---"); print(f"Accuracy: {accuracy:.2%}"); print(f"Precision: {precision:.2%}"); print(f"Recall: {recall:.2%}"); print(f"F1 Score: {f1:.2%}"); print(f"Cohen's Kappa: {kappa:.2f}"); print(f"Macro AUC: {macro_auc:.2f}" if not np.isnan(macro_auc) else "Macro AUC: nan"); print(f"Specificity: {avg_specificity:.2%}")
    if seizure_sensitivity is not None: print(f"Seizure Sensitivity: {seizure_sensitivity:.2%}")
    else: print("Seizure Sensitivity: Class 'Seizure' not in test set.")
    print(f"Avg Inference Time: {avg_inference_time * 1000:.2f} ms")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "kappa": kappa, "macro_auc": macro_auc, "specificity": avg_specificity, "seizure_sensitivity": seizure_sensitivity, "avg_inference_time_sec": avg_inference_time}

print("Train and Test functions defined.")


# In[5]:


# %%
# Instantiate Datasets
try:
    train_dataset = HMS_Dataset(metaDF=trainDF, base_dir=BASE_DIR, activity_mapping=activity_mapping)
    val_dataset = HMS_Dataset(metaDF=valDF, base_dir=BASE_DIR, activity_mapping=activity_mapping)
    test_dataset = HMS_Dataset(metaDF=testDF, base_dir=BASE_DIR, activity_mapping=activity_mapping)

    # --- Batch Size set to 1 ---
    BatchSize = 2

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


# In[6]:


# %%
# TSception is self-contained, no external dependencies needed in this cell.
print("TSception dependencies (none external) confirmed.")


# In[7]:


# %%
class TSception(nn.Module):
    def conv_block(self, in_chan, out_chan, kernel, step, pool):
            # Ensure kernel and step are tuples
            kernel_h, kernel_w = kernel if isinstance(kernel, tuple) else (kernel, kernel)
            step_h, step_w = step if isinstance(step, tuple) else (step, step)

            # --- MANUAL PADDING CALCULATION ---
            # Calculate padding for height (spatial dimension) - standard centered padding
            padding_h = (kernel_h - 1) // 2

            # Calculate padding for width (temporal dimension) - standard centered padding
            padding_w = (kernel_w - 1) // 2

            # Combine padding values
            padding = (padding_h, padding_w)
            # --- END FIX ---

            return nn.Sequential(
                nn.Conv2d(in_channels=in_chan, out_channels=out_chan,
                        kernel_size=kernel, stride=step, padding=padding), # Use calculated padding
                nn.LeakyReLU(),
                nn.AvgPool2d(kernel_size=(1, pool), stride=(1, pool)))

    def __init__(self, num_classes, input_size, sampling_rate, num_T, num_S, hidden, dropout_rate):
        # input_size: (Freq=1, Channels=C, Time=T)
        super(TSception, self).__init__()
        self.inception_window = [0.5, 0.25, 0.125]
        self.pool = 8 # Pool size for Tception
        self.channel_dim = input_size[1] # C

        # --- Temporal Learning Block (Tception) ---
        # Input: (B, 1, C, T)
        self.Tception1 = self.conv_block(1, num_T, (1, int(self.inception_window[0] * sampling_rate)), 1, self.pool)
        self.Tception2 = self.conv_block(1, num_T, (1, int(self.inception_window[1] * sampling_rate)), 1, self.pool)
        self.Tception3 = self.conv_block(1, num_T, (1, int(self.inception_window[2] * sampling_rate)), 1, self.pool)
        # Output of each Tception block: (B, num_T, C, T_pooled)
        # After concat: (B, num_T, C, 3 * T_pooled)

        self.BN_t = nn.BatchNorm2d(num_T) # Applied to num_T features

        # --- Spatial Learning Block (Sception) ---
        # Input: (B, num_T, C, T_cat_pooled)
        # Sception1: Conv across all channels C
        self.Sception1 = self.conv_block(num_T, num_S, (self.channel_dim, 1), 1, int(self.pool * 0.25))
        # Sception2: Conv across half the channels C/2 (approx)
        scep2_kernel_h = math.ceil(self.channel_dim * 0.5) # Use ceil for potentially odd C
        # Stride should match kernel H to avoid overlap/gaps if intended as separate groups
        self.Sception2 = self.conv_block(num_T, num_S, (scep2_kernel_h, 1), (scep2_kernel_h, 1), int(self.pool * 0.25))
        # Output of Sception1: (B, num_S, 1, T_s_pooled)
        # Output of Sception2: (B, num_S, approx 2, T_s_pooled) -- Need check here!
        # The paper likely intended Sception2 to be depthwise or grouped.
        # Let's assume standard conv as written. Output H dim will depend on padding.
        # If padding='same' in conv_block, Sception1 H=1, Sception2 H=2.
        # Concatenating on dim=2 (H) -> (B, num_S, 3, T_s_pooled)

        self.BN_s = nn.BatchNorm2d(num_S) # Applied to num_S features

        # --- Fusion Block ---
        # Input: (B, num_S, H_cat, T_s_pooled) -> approx (B, num_S, 3, T_s_pooled)
        # Kernel (3, 1) operates across the concatenated height dimension
        self.fusion_layer = self.conv_block(num_S, num_S, (3, 1), 1, 4) # Pool time dim again
        # Output: (B, num_S, 1, T_f_pooled)

        self.BN_fusion = nn.BatchNorm2d(num_S)

        # --- Fully Connected Layer ---
        # Input after pooling and squeeze: (B, num_S)
        self.fc = nn.Sequential(
            nn.Linear(num_S, hidden),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden, num_classes)
        )

        # --- Debug: Calculate final size dynamically ---
        self._calculate_fc_in_features(input_size)


    def _calculate_fc_in_features(self, input_size):
        # Helper to determine the input size to the first FC layer dynamically
        # input_size: (Freq, C, T)
        print("Calculating FC input features dynamically...")
        dummy_input = torch.randn(1, input_size[0], input_size[1], input_size[2]) # B=1, F=1, C, T

        # Pass through Tception
        y1 = self.Tception1(dummy_input)
        y2 = self.Tception2(dummy_input)
        y3 = self.Tception3(dummy_input)
        out_t = torch.cat((y1, y2, y3), dim=-1) # (1, num_T, C, T_cat_pool)
        out_t = self.BN_t(out_t)

        # Pass through Sception
        z1 = self.Sception1(out_t) # (1, num_S, 1, T_s_pool)
        z2 = self.Sception2(out_t) # (1, num_S, H_s2, T_s_pool)
        out_s = torch.cat((z1, z2), dim=2) # (1, num_S, H_cat, T_s_pool)
        out_s = self.BN_s(out_s)

        # Pass through Fusion
        out_f = self.fusion_layer(out_s) # (1, num_S, H_f, T_f_pool)
        out_f = self.BN_fusion(out_f)

        # Global Average Pooling (mimics the mean and squeeze)
        gap = nn.AdaptiveAvgPool2d((1, 1))
        out_pooled = gap(out_f) # (1, num_S, 1, 1)

        fc_in_features = out_pooled.shape[1] # Should be num_S
        print(f"Calculated FC input features: {fc_in_features}")

        # Reinitialize FC layer if calculated size differs (unlikely here but good practice)
        if self.fc[0].in_features != fc_in_features:
            print(f"Reinitializing FC layer input from {self.fc[0].in_features} to {fc_in_features}")
            old_hidden = self.fc[0].out_features
            old_dropout = self.fc[2].p
            old_num_classes = self.fc[3].out_features
            self.fc = nn.Sequential(
                nn.Linear(fc_in_features, old_hidden),
                nn.ReLU(),
                nn.Dropout(old_dropout),
                nn.Linear(old_hidden, old_num_classes)
            )


    def forward(self, x):
        # x: (B, C, T) -> (1, 728, 1296)
        x = torch.unsqueeze(x, dim=1)  # (B, 1, C, T)

        # Tception
        y1 = self.Tception1(x); y2 = self.Tception2(x); y3 = self.Tception3(x)
        out = torch.cat((y1, y2, y3), dim=-1) # (B, num_T, C, T_cat_pool)
        out = self.BN_t(out)

        # Sception
        z1 = self.Sception1(out) # (B, num_S, 1, T_s_pool)
        z2 = self.Sception2(out) # (B, num_S, H_s2, T_s_pool)
        out_ = torch.cat((z1, z2), dim=2) # (B, num_S, H_cat, T_s_pool)
        out = self.BN_s(out_)

        # Fusion
        out = self.fusion_layer(out) # (B, num_S, H_f, T_f_pool)
        out = self.BN_fusion(out)

        # Global Average Pooling Equivalent
        # torch.mean(out, dim=-1) averages over T_f_pool -> (B, num_S, H_f)
        # torch.squeeze(..., dim=-1) is redundant if H_f is 1 after pooling
        # torch.mean(out, dim=(-1, -2)) averages over both H_f and T_f_pool -> (B, num_S)
        out = torch.mean(out, dim=(-1, -2)) # More robust than squeeze

        # Classification
        out = self.fc(out) # (B, num_classes)
        return out

print("New Model (TSception) class defined.")


# In[8]:


# %%
# --- Data shapes ---
N_CHAN = 728
N_TIME = 1296
N_CLASS = 6
N_FREQ = 1 # Input dimension for Conv2d

# --- TSception Hyperparameters ---
SAMPLING_RATE = SFREQ # From global config (200 Hz)
NUM_T = 15          # Number of temporal filters
NUM_S = 15          # Number of spatial filters
HIDDEN = 100        # Hidden units in FC layer
DROPOUT_RATE = 0.3

# Create configs Namespace (optional, keeps pattern)
configs = Namespace()
configs.num_classes = N_CLASS
configs.input_size = (N_FREQ, N_CHAN, N_TIME)
configs.sampling_rate = SAMPLING_RATE
configs.num_T = NUM_T
configs.num_S = NUM_S
configs.hidden = HIDDEN
configs.dropout_rate = DROPOUT_RATE

print("Model Configurations for TSception:")
print({k: v for k, v in vars(configs).items()})


# In[9]:


# %%
# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using Device: {device}")

# Instantiate the TSception model
try:
    model = TSception(
        num_classes=configs.num_classes,
        input_size=configs.input_size,
        sampling_rate=configs.sampling_rate,
        num_T=configs.num_T,
        num_S=configs.num_S,
        hidden=configs.hidden,
        dropout_rate=configs.dropout_rate
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"New model (TSception) instantiated. Trainable params: {total_params/1e6:.2f}M")

except ValueError as ve: # Catch specific ValueErrors
    print("="*50); print(f"ERROR initializing model: {ve}"); print("="*50); raise ve
except Exception as e: print("="*50); print(f"ERROR initializing model: {e}"); print("="*50); raise e

# Setup loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

print("New model, criterion, optimizer, and scheduler set up.")


# In[10]:


# %%
CHECKPOINT_PATH = "TSception_v1.pth"; LOG_FILE_PATH = "TSception_v1_log.csv"
print(f"Starting training for new model: {CHECKPOINT_PATH}")
try:
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                epochs=8, device=device, checkpoint_path=CHECKPOINT_PATH, log_file_path=LOG_FILE_PATH)
except RuntimeError as rte:
     print(f"\nAn error occurred during training: {rte}")
     if "CUDA out of memory" in str(rte): print(">>> CUDA OOM. Try reducing 'num_T', 'num_S', or 'hidden' in Cell 8. <<<")
     elif "size mismatch" in str(rte): print(">>> Size Mismatch Error. Check kernel sizes, strides, and padding in TSception's conv_block. <<<")
     else: raise rte
except ValueError as ve:
     print(f"\nAn error occurred during training: {ve}")
     raise ve
except Exception as e: print(f"\nAn unexpected error: {e}"); raise e


# In[ ]:


# %%
print(f"\nTesting model from {CHECKPOINT_PATH}...")
try:
    test_model_instance = TSception(
        num_classes=configs.num_classes, input_size=configs.input_size, sampling_rate=configs.sampling_rate,
        num_T=configs.num_T, num_S=configs.num_S, hidden=configs.hidden, dropout_rate=configs.dropout_rate
    ).to(device)

    test_results = test_model(test_model_instance, test_loader, checkpoint_path=CHECKPOINT_PATH, device=device)
    print("\n--- Final Test Results Summary (TSception) ---"); print(test_results)
except FileNotFoundError: print(f"Checkpoint {CHECKPOINT_PATH} not found. Skipping test.")
except ValueError as ve: print(f"A config error during testing setup: {ve}")
except RuntimeError as rte: print(f"A runtime error during testing: {rte}")
except Exception as e: print(f"An unexpected error during testing: {e}")

