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

from tqdm.auto import tqdm
import copy

try:
    from einops import rearrange
    from einops.layers.torch import Rearrange

except ImportError:
    print("einops not found (needed for Deformer, not EEGNet).")


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
                spec = np.abs(spec); spec = np.log1p(spec).astype("float32"); spect[:,:,k] += spec[:128, 1:257] if spec.shape[1] > 1 else np.zeros((128, 256)) # Handle short spec
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


import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dWithConstraint(nn.Conv2d):
    """
    A Conv2d layer that applies a maximum norm constraint on the weights
    along a specified dimension (typically the output channel dimension).
    This is often used in EEG models like EEGNet to control filter norms.
    """
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs):
        """
        Initializes the constrained convolutional layer.

        Args:
            *args: Standard arguments for nn.Conv2d (in_channels, out_channels, kernel_size, ...).
            doWeightNorm (bool): Whether to apply the weight normalization. Defaults to True.
            max_norm (float): The maximum norm value for the weights. Defaults to 1.
            **kwargs: Standard keyword arguments for nn.Conv2d (stride, padding, dilation, groups, bias, ...).
        """
        self.max_norm = max_norm
        self.doWeightNorm = doWeightNorm
        super(Conv2dWithConstraint, self).__init__(*args, **kwargs)

    def forward(self, x):
        """
        Applies the weight normalization constraint before the convolution.
        """
        if self.doWeightNorm:
            # torch.renorm applies norm constraint along a specific dimension.
            # For Conv2d weights (out_channels, in_channels/groups, kH, kW),
            # dim=0 constrains the norm across the filter dimension for each output channel.
            self.weight.data = torch.renorm(
                self.weight.data, p=2, dim=0, maxnorm=self.max_norm
            )
        # Call the original Conv2d forward pass
        return super(Conv2dWithConstraint, self).forward(x)

print("Conv2dWithConstraint class defined.")


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


class eegNet(nn.Module):
    def initialBlocks(self, dropoutP, *args, **kwargs):
        block1 = nn.Sequential(
                # Input is (B, 1, C, T)
                nn.Conv2d(1, self.F1, (1, self.C1),
                          padding=(0, self.C1 // 2), bias=False), # Temporal Conv
                nn.BatchNorm2d(self.F1),
                # --- FIX: Removed nn. prefix ---
                Conv2dWithConstraint(self.F1, self.F1 * self.D, (self.nChan, 1), # Use custom class directly
                                     padding=0, bias=False, max_norm=1,
                                     groups=self.F1),
                # --- END FIX ---
                nn.BatchNorm2d(self.F1 * self.D),
                nn.ELU(),
                nn.AvgPool2d((1, 4), stride=4), # Pool time dim
                nn.Dropout(p=dropoutP))
        block2 = nn.Sequential(
                # Separable Conv (Temporal Depthwise + Pointwise)
                nn.Conv2d(self.F1 * self.D, self.F1 * self.D,  (1, 22), # Depthwise temporal
                                     padding=(0, 22//2), bias=False,
                                     groups=self.F1 * self.D),
                nn.Conv2d(self.F1 * self.D, self.F2, (1, 1), # Pointwise
                                     stride=1, bias=False, padding=0),
                nn.BatchNorm2d(self.F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8), stride=8), # Pool time dim
                nn.Dropout(p=dropoutP)
                )
        return nn.Sequential(block1, block2)

    def lastBlock(self, inF, outF, kernalSize, *args, **kwargs):
        # Final classification layer as a Conv2d
        return nn.Sequential(
                nn.Conv2d(inF, outF, kernalSize, *args, **kwargs))

    def calculateOutSize(self, model, nChan, nTime):
        ''' Calculate the output size (spatial dims) after passing through model blocks '''
        data = torch.rand(1, 1, nChan, nTime)
        model.eval()
        with torch.no_grad():
             out = model(data).shape
        return out[2:] # Returns (H, W) = (1, T_final)

    def __init__(self, nChan, nTime, nClass=6,
                 dropoutP=0.25, F1=8, D=2,
                 C1=64, *args, **kwargs):
        super(eegNet, self).__init__()
        self.F2 = D*F1; self.F1 = F1; self.D = D
        self.nTime = nTime; self.nClass = nClass; self.nChan = nChan; self.C1 = C1

        self.firstBlocks = self.initialBlocks(dropoutP)
        self.fSize = self.calculateOutSize(self.firstBlocks, nChan, nTime)
        self.lastLayer = self.lastBlock(self.F2, nClass, (1, self.fSize[1]))

    def forward(self, x):
        x = torch.unsqueeze(x, dim=1)  # (B, 1, C, T)
        x = self.firstBlocks(x)        # (B, F2, 1, T_final)
        x = self.lastLayer(x)          # (B, nClass, 1, 1)
        x = torch.squeeze(x, 3)        # (B, nClass, 1)
        x = torch.squeeze(x, 2)        # (B, nClass)
        return x

print("New Model (eegNet) class defined with correct Conv2dWithConstraint usage.")


# In[ ]:


# --- Data shapes ---
# Data is (B, 728, 1296) -> (B, C, T)
N_CHAN = 728
N_TIME = 1296
N_CLASS = 6

# --- EEGNet Hyperparameters ---
DROPOUT_P = 0.3
F1 = 16 # Increased F1
D = 2
C1 = 64 # Temporal kernel size for first conv

print("Configurations for EEGNet:")
print(f"nChan={N_CHAN}, nTime={N_TIME}, nClass={N_CLASS}")
print(f"dropoutP={DROPOUT_P}, F1={F1}, D={D}, C1={C1}")


# In[ ]:


# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using Device: {device}")

# Instantiate the EEGNet model directly
try:
    model = eegNet(
        nChan=N_CHAN,
        nTime=N_TIME,
        nClass=N_CLASS,
        dropoutP=DROPOUT_P,
        F1=F1,
        D=D,
        C1=C1
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"New model (eegNet) instantiated. Trainable params: {total_params/1e6:.2f}M")

except Exception as e:
    print("="*50); print(f"ERROR initializing model: {e}"); print("="*50)
    raise e

# Setup loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

print("New model, criterion, optimizer, and scheduler are set up.")


# In[ ]:


CHECKPOINT_PATH = "EEGNet_v1.pth"; LOG_FILE_PATH = "EEGNet_v1_log.csv"
print(f"Starting training for new model: {CHECKPOINT_PATH}")
try:
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                epochs=12, device=device, checkpoint_path=CHECKPOINT_PATH, log_file_path=LOG_FILE_PATH)
except RuntimeError as rte: # Catch RuntimeErrors like shape mismatches or CUDA OOM
     print(f"\nAn error occurred during training: {rte}")
     if "CUDA out of memory" in str(rte): print(">>> CUDA Out of Memory. 'BatchSize' is 1. Try reducing 'F1' or 'D' in Cell 8. <<<")
     else: raise rte # Re-raise other RuntimeErrors
except Exception as e: print(f"\nAn unexpected error: {e}"); raise e


# In[ ]:


print(f"\nTesting model from {CHECKPOINT_PATH}...")
try:
    test_model_instance = eegNet(
        nChan=N_CHAN, nTime=N_TIME, nClass=N_CLASS,
        dropoutP=DROPOUT_P, F1=F1, D=D, C1=C1
    ).to(device)
    test_results = test_model(test_model_instance, test_loader, checkpoint_path=CHECKPOINT_PATH, device=device)
    print("\n--- Final Test Results Summary (eegNet) ---"); print(test_results)
except FileNotFoundError: print(f"Checkpoint {CHECKPOINT_PATH} not found. Skipping test.")
except Exception as e: print(f"An error during testing: {e}")

