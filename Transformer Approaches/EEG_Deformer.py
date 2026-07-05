#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install einops')
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

# --- Imports for Deformer ---
# You might need to run: pip install einops
try:
    from einops import rearrange
    from einops.layers.torch import Rearrange
except ImportError:
    print("einops not found. Please install it: pip install einops")
    raise

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
    # Stop execution if data isn't found
    raise SystemExit("Data loading failed.")


# In[ ]:


class HMS_Dataset(Dataset):
    def __init__(self, metaDF, base_dir, activity_mapping):
        self.metaDF = metaDF
        self.base_dir = base_dir
        self.activity_mapping = activity_mapping
        self.RAW_FEATURES = {'LL': ['Fp1-F7', 'F7-T3', 'T3-T5', 'T5-O1'],
                             'RL': ['Fp2-F8', 'F8-T4', 'T4-T6', 'T6-O2'],
                             'LP': ['Fp1-F3', 'F3-C3', 'C3-P3', 'P3-O1'],
                             'RP': ['Fp2-F4', 'F4-C4', 'C4-P4', 'P4-O2']}
        self.FEATS = [['Fp1','F7','T3','T5','O1'], ['Fp1','F3','C3','P3','O1'],
                      ['Fp2','F8','T4','T6','O2'], ['Fp2','F4','C4','P4','O2']]
        self.Epartition = np.zeros((200, 8), dtype='float32')
        self.Spartition = np.zeros((512, 16), dtype='float32')
        self.Partition = np.zeros((16, 1296), dtype='float32')

        print(f"Dataset Initialized. Output shape (C, T) = (728, 1296).")

    def __len__(self): return len(self.metaDF)

    # --- Feature extraction functions (unchanged) ---
    def raweeg_50s_from_eeg_v1(self, parquet_path, eeg_label_offset_seconds):
        EEG_LENGTH = 50
        try: raw_eeg = pd.read_parquet(parquet_path)
        except Exception as e: return np.zeros((200, 800), dtype=np.float32)
        time_temp=eeg_label_offset_seconds*200; time_start=round(time_temp+(50-EEG_LENGTH)/2*200); time_stop=round(time_temp+(50+EEG_LENGTH)/2*200)
        eeg_default = raw_eeg.loc[time_start:(time_stop-1),:].reset_index(drop=True); list_eeg = list()
        for region in self.RAW_FEATURES.keys():
            eeg = np.zeros((len(self.RAW_FEATURES[region]), eeg_default.shape[0]), dtype=np.float32)
            for chan_i, chan in enumerate(self.RAW_FEATURES[region]):
                eeg_1 = eeg_default.loc[:, chan.split('-')[0]]; mean_value = eeg_1.mean(); eeg_1.fillna(value=mean_value, inplace=True); eeg_1 = eeg_1.values
                eeg_2 = eeg_default.loc[:, chan.split('-')[1]]; mean_value = eeg_2.mean(); eeg_2.fillna(value=mean_value, inplace=True); eeg_2 = eeg_2.values
                new_eeg = eeg_1 - eeg_2; del eeg_1, eeg_2
                new_eeg = signal.filtfilt(b, a, new_eeg, axis=0); new_eeg = np.clip(new_eeg, -1024, 1024).astype("float32"); eeg[chan_i, :] = new_eeg
            eeg = np.reshape(eeg, (4, 200, EEG_LENGTH)); eeg = np.concatenate((eeg[0,:,:], eeg[1,:,:], eeg[2,:,:], eeg[3,:,:]), 1); list_eeg.append(eeg)
        eeg = np.concatenate(list_eeg, 1); eeg /= 104; return eeg

    def raweeg_20s_from_eeg_v1(self, parquet_path, eeg_label_offset_seconds):
        EEG_LENGTH = 20
        try: raw_eeg = pd.read_parquet(parquet_path)
        except Exception as e: return np.zeros((200, 320), dtype=np.float32)
        time_temp=eeg_label_offset_seconds*200; time_start=round(time_temp+(50-EEG_LENGTH)/2*200); time_stop=round(time_temp+(50+EEG_LENGTH)/2*200)
        eeg_default = raw_eeg.loc[time_start:(time_stop-1),:].reset_index(drop=True); list_eeg = list()
        for region in self.RAW_FEATURES.keys():
            eeg = np.zeros((len(self.RAW_FEATURES[region]), eeg_default.shape[0]), dtype=np.float32)
            for chan_i, chan in enumerate(self.RAW_FEATURES[region]):
                eeg_1 = eeg_default.loc[:, chan.split('-')[0]]; mean_value = eeg_1.mean(); eeg_1.fillna(value=mean_value, inplace=True); eeg_1 = eeg_1.values
                eeg_2 = eeg_default.loc[:, chan.split('-')[1]]; mean_value = eeg_2.mean(); eeg_2.fillna(value=mean_value, inplace=True); eeg_2 = eeg_2.values
                new_eeg = eeg_1 - eeg_2; del eeg_1, eeg_2
                new_eeg = signal.filtfilt(b, a, new_eeg, axis=0); new_eeg = np.clip(new_eeg, -1024, 1024).astype("float32"); eeg[chan_i, :] = new_eeg
            eeg = np.reshape(eeg, (4, 200, EEG_LENGTH)); eeg = np.concatenate((eeg[0,:,:], eeg[1,:,:], eeg[2,:,:], eeg[3,:,:]), 1); list_eeg.append(eeg)
        eeg = np.concatenate(list_eeg, 1); eeg /= 104; return eeg

    def raweeg_10s_from_eeg_v1(self, parquet_path, eeg_label_offset_seconds):
        EEG_LENGTH = 10
        try: raw_eeg = pd.read_parquet(parquet_path)
        except Exception as e: return np.zeros((200, 160), dtype=np.float32)
        time_temp=eeg_label_offset_seconds*200; time_start=round(time_temp+(50-EEG_LENGTH)/2*200); time_stop=round(time_temp+(50+EEG_LENGTH)/2*200)
        eeg_default = raw_eeg.loc[time_start:(time_stop-1),:].reset_index(drop=True); list_eeg = list()
        for region in self.RAW_FEATURES.keys():
            eeg = np.zeros((len(self.RAW_FEATURES[region]), eeg_default.shape[0]), dtype=np.float32)
            for chan_i, chan in enumerate(self.RAW_FEATURES[region]):
                eeg_1 = eeg_default.loc[:, chan.split('-')[0]]; mean_value = eeg_1.mean(); eeg_1.fillna(value=mean_value, inplace=True); eeg_1 = eeg_1.values
                eeg_2 = eeg_default.loc[:, chan.split('-')[1]]; mean_value = eeg_2.mean(); eeg_2.fillna(value=mean_value, inplace=True); eeg_2 = eeg_2.values
                new_eeg = eeg_1 - eeg_2; del eeg_1, eeg_2
                new_eeg = signal.filtfilt(b, a, new_eeg, axis=0); new_eeg = np.clip(new_eeg, -1024, 1024).astype("float32"); eeg[chan_i, :] = new_eeg
            eeg = np.reshape(eeg, (4, 200, EEG_LENGTH)); eeg = np.concatenate((eeg[0,:,:], eeg[1,:,:], eeg[2,:,:], eeg[3,:,:]), 1); list_eeg.append(eeg)
        eeg = np.concatenate(list_eeg, 1); eeg /= 104; return eeg

    def stft_spec_from_50s_eeg_v1(self, parquet_path, eeg_label_offset_seconds):
        EEG_LENGTH = 50
        try: eeg = pd.read_parquet(parquet_path)
        except Exception as e: return np.zeros((512, 1024), dtype=np.float32)
        time_temp=eeg_label_offset_seconds*200; time_start=round(time_temp+(50-EEG_LENGTH)/2*200); time_stop=round(time_temp+(50+EEG_LENGTH)/2*200)
        eeg = eeg.iloc[time_start:time_stop]; list_eeg = list()
        for k in range(4):
            COLS = self.FEATS[k]; spect = np.zeros((128,256,4),dtype='float32')
            for kk in range(4):
                eeg_1 = eeg[COLS[kk]]; mean_value = eeg_1.mean(); eeg_1.fillna(value=mean_value, inplace=True); eeg_1 = eeg_1.values
                eeg_2 = eeg[COLS[kk+1]]; mean_value = eeg_2.mean(); eeg_2.fillna(value=mean_value, inplace=True); eeg_2 = eeg_2.values
                new_eeg = eeg_1 - eeg_2; del eeg_1, eeg_2
                fs = 200; nperseg = 39; noverlap = 0; f, t, spec = signal.spectrogram(new_eeg, fs, nperseg=nperseg, noverlap=noverlap, nfft=256)
                spec = np.abs(spec); spec = np.log1p(spec).astype("float32"); spect[:,:,kk] += spec[:128, :]
            spect = np.concatenate((spect[:,:,0], spect[:,:,1], spect[:,:,2], spect[:,:,3]), 1); list_eeg.append(spect)
        spect = np.concatenate(list_eeg, 0); spect /= 2; return spect

    def stft_spec_from_50s_eeg_v2(self, parquet_path, eeg_label_offset_seconds):
        EEG_LENGTH = 50
        try: eeg = pd.read_parquet(parquet_path)
        except Exception as e: return np.zeros((512, 256), dtype=np.float32)
        time_temp=eeg_label_offset_seconds*200; time_start=round(time_temp+(50-EEG_LENGTH)/2*200); time_stop=round(time_temp+(50+EEG_LENGTH)/2*200)
        eeg = eeg.iloc[time_start:time_stop]; spect = np.zeros((128,256,4),dtype='float32')
        for k in range(4):
            COLS = self.FEATS[k]
            for kk in range(4):
                eeg_1 = eeg[COLS[kk]]; mean_value = eeg_1.mean(); eeg_1.fillna(value=mean_value, inplace=True); eeg_1 = eeg_1.values
                eeg_2 = eeg[COLS[kk+1]]; mean_value = eeg_2.mean(); eeg_2.fillna(value=mean_value, inplace=True); eeg_2 = eeg_2.values
                new_eeg = eeg_1 - eeg_2; del eeg_1, eeg_2
                fs = 200; nperseg = len(new_eeg)//256; noverlap = 0; f, t, spec = signal.stft(new_eeg, fs, nperseg=nperseg, noverlap=noverlap, nfft=256)
                spec = np.abs(spec); spec = np.log1p(spec).astype("float32"); spect[:,:,k] += spec[:128, 1:257]
            spect[:,:,k] /= 4
        spect = np.concatenate((spect[:,:,0], spect[:,:,1], spect[:,:,2], spect[:,:,3]), 0); return spect
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
        # --- NO TRANSPOSE for Deformer ---
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y = self.activity_mapping[label]
        y_tensor = torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.long), num_classes=6).float()
        return X_tensor, y_tensor


# In[ ]:


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20, device="cuda", checkpoint_path="best_model.pth", log_file_path="training_log.csv"):
    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Created dir: {checkpoint_dir}")

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint: {checkpoint_path}")
        try:
            model.load_state_dict(torch.load(checkpoint_path))
        except Exception as e:
            print(f"Error loading: {e}. Starting fresh.")
    else:
        print(f"Starting fresh. Checkpoint: {checkpoint_path}")

    best_val_accuracy = 0.0
    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        print(f"Created dir: {log_dir}")

    if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0:
        with open(log_file_path,"w") as f:
            f.write("epoch#,train_loss,train_accuracy,val_loss,val_accuracy,time_taken\n")

    for epoch in range(epochs):
        start_time = time.time()
        model.train()
        train_loss, correct, total = 0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for i, (X, y) in enumerate(train_pbar):
            X, y = X.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(X) # <-- MODIFIED CALL for Deformer
            loss = criterion(outputs, torch.argmax(y, dim=1))
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            total += y.size(0)
            correct += predicted.eq(torch.argmax(y, dim=1)).sum().item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

        train_accuracy = 100. * correct / total
        avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            for X, y in val_pbar:
                X, y = X.to(device), y.to(device)
                outputs = model(X) # <-- MODIFIED CALL for Deformer
                loss = criterion(outputs, torch.argmax(y, dim=1))
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(torch.argmax(y, dim=1)).sum().item()
                val_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

        val_accuracy = 100. * correct / total
        avg_val_loss = val_loss / len(val_loader)
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        scheduler.step(avg_val_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved: {checkpoint_path} @ Val Acc: {val_accuracy:.2f}%")

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} Time: {epoch_time:.2f}s")
        with open(log_file_path,"a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{train_accuracy:.2f},{avg_val_loss:.4f},{val_accuracy:.2f},{epoch_time:.2f}\n")

    print(f"Training complete. Best Val Acc: {best_val_accuracy:.2f}%")


def test_model(model, test_loader, checkpoint_path="best_model.pth", device="cuda"):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found: {checkpoint_path}")
        return {}

    print(f"Loading best model: {checkpoint_path}")
    try:
        model.load_state_dict(torch.load(checkpoint_path))
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
            outputs = model(X) # <-- MODIFIED CALL for Deformer
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

    # --- Metrics (split onto separate lines) ---
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)

    # --- Plotting (split onto separate lines) ---
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=brain_activities, yticklabels=brain_activities)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title(f"Confusion Matrix - {mname}")
    plt.savefig(f"{mname}_confusion_matrix.png")
    plt.close()
    print(f"Confusion matrix saved: {mname}_confusion_matrix.png")

    # --- Sensitivity ---
    seizure_class_idx = activity_mapping.get('Seizure', -1)
    if seizure_class_idx != -1 and seizure_class_idx < cm.shape[0]:
        TP = cm[seizure_class_idx][seizure_class_idx]
        FN = sum(cm[seizure_class_idx]) - TP
        seizure_sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    else:
        seizure_sensitivity = None

    # --- Specificity ---
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

    # --- ROC/AUC (split onto separate lines) ---
    num_classes = all_probs.shape[1]
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
    macro_auc = np.nanmean(auc_scores)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("FPR")
    plt.ylabel("TPR")
    plt.title(f"ROC Curve - {mname}")
    plt.legend()
    plt.savefig(f"{mname}_roc_curve.png")
    plt.close()
    print(f"ROC curve saved: {mname}_roc_curve.png")

    # --- Inference Time ---
    avg_inference_time = total_time / total_samples if total_samples > 0 else 0.0

    # --- Print Results (split onto separate lines) ---
    print("--- Test Results ---")
    print(f"Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    print(f"Cohen's Kappa: {kappa:.2f}")
    print(f"Macro AUC: {macro_auc:.2f}")
    print(f"Specificity: {avg_specificity:.2%}")
    if seizure_sensitivity is not None:
        print(f"Seizure Sensitivity: {seizure_sensitivity:.2%}")
    else:
        print("Seizure Sensitivity: Class 'Seizure' not in test set.")
    print(f"Avg Inference Time: {avg_inference_time * 1000:.2f} ms")

    # --- Return Dictionary ---
    return {
        "accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1,
        "kappa": kappa, "macro_auc": macro_auc, "specificity": avg_specificity,
        "seizure_sensitivity": seizure_sensitivity, "avg_inference_time_sec": avg_inference_time
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

    # Adjust num_workers based on your system's capabilities
    num_workers = min(os.cpu_count(), 8) 
    print(f"Using {num_workers} workers for DataLoaders.")

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


# --- Define Deformer Helper Modules ---
def pair(t): return t if isinstance(t, tuple) else (t, t)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__(); self.net = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, dim), nn.Dropout(dropout))
    def forward(self, x): return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__(); inner_dim = dim_head * heads; project_out = not (heads == 1 and dim_head == dim); self.heads = heads; self.scale = dim_head ** -0.5; self.attend = nn.Softmax(dim=-1); self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False); self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()
    def forward(self, x): qkv = self.to_qkv(x).chunk(3, dim=-1); q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv); dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale; attn = self.attend(dots); out = torch.matmul(attn, v); out = rearrange(out, 'b h n d -> b n (h d)'); return self.to_out(out)

class Transformer(nn.Module):
    def cnn_block(self, in_chan, kernel_size, dp): return nn.Sequential(nn.Dropout(p=dp), nn.Conv1d(in_channels=in_chan, out_channels=in_chan, kernel_size=kernel_size, padding=self.get_padding_1D(kernel=kernel_size)), nn.BatchNorm1d(in_chan), nn.ELU(), nn.MaxPool1d(kernel_size=2, stride=2))
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, in_chan, fine_grained_kernel=11, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        current_dim = dim
        for i in range(depth):
            layer_input_dim = current_dim # Dim entering the layer
            current_dim = int(current_dim * 0.5) # Dim after pooling

            # --- DIM FIX ---
            # Attention takes the dimension AFTER pooling (current_dim)
            # FeedForward also takes the dimension AFTER pooling (current_dim)
            self.layers.append(nn.ModuleList([
                Attention(current_dim, heads=heads, dim_head=dim_head, dropout=dropout),
                FeedForward(current_dim, mlp_dim, dropout=dropout),
                self.cnn_block(in_chan=in_chan, kernel_size=fine_grained_kernel, dp=dropout)
            ]))
            # --- END FIX ---
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)

    def forward(self, x):
        dense_feature = []
        for i, (attn, ff, cnn) in enumerate(self.layers):
            try: x_cg = self.pool(x) # Pool along dim D -> [B, K, D/2]
            except RuntimeError as e: print(f"Error pooling layer {i}. Input: {x.shape}"); raise e
            if x_cg.shape[-1] == 0: raise ValueError(f"Dim zero after pool layer {i}. Input: {x.shape}.")

            # Attention operates on pooled data x_cg (dim D/2)
            attn_out = attn(x_cg); x_cg = attn_out + x_cg # [B, K, D/2]

            try: x_fg = cnn(x) # CNN operates on original x (dim D), outputs D/2
            except Exception as e: print(f"Error CNN layer {i}. Input: {x.shape}"); raise e

            x_info = self.get_info(x_fg); dense_feature.append(x_info) # [B, K]

            # FeedForward operates on pooled data x_cg (dim D/2)
            ff_out = ff(x_cg) # Input [B, K, D/2] -> Output [B, K, D/2]

            x = ff_out + x_fg # Combine features -> [B, K, D/2] (Update x for next loop)

        x_dense = torch.cat(dense_feature, dim=-1); x_flat = x.reshape(x.size(0), -1); emd = torch.cat((x_flat, x_dense), dim=-1); return emd
    def get_info(self, x): return torch.log(torch.mean(x.pow(2), dim=-1) + 1e-6)
    def get_padding_1D(self, kernel): return int(0.5 * (kernel - 1))

class Conv2dWithConstraint(nn.Conv2d):
    def __init__(self, *args, doWeightNorm=True, max_norm=1, **kwargs): self.max_norm = max_norm; self.doWeightNorm = doWeightNorm; super(Conv2dWithConstraint, self).__init__(*args, **kwargs)
    def forward(self, x):
        if self.doWeightNorm: self.weight.data = torch.renorm(self.weight.data, p=2, dim=0, maxnorm=self.max_norm)
        return super(Conv2dWithConstraint, self).forward(x)

print("All model dependencies (Deformer) defined with FeedForward dim fix.")


# In[ ]:


class Deformer(nn.Module):
    def cnn_block(self, out_chan, kernel_size, num_chan):
        padding = self.get_padding(kernel_size[-1]); return nn.Sequential(Conv2dWithConstraint(1, out_chan, kernel_size, padding=padding, max_norm=2), Conv2dWithConstraint(out_chan, out_chan, (num_chan, 1), padding=0, max_norm=2), nn.BatchNorm2d(out_chan), nn.ELU(), nn.MaxPool2d((1, 2), stride=(1, 2)))
    def __init__(self, *, num_chan, num_time, temporal_kernel, num_kernel=64, num_classes, depth=4, heads=16, mlp_dim=16, dim_head=16, dropout=0.):
        super().__init__(); self.cnn_encoder = self.cnn_block(out_chan=num_kernel, kernel_size=(1, temporal_kernel), num_chan=num_chan)
        dim = num_time // 2; self.to_patch_embedding = Rearrange('b k c f -> b k (c f)')
        # Adjust positional embedding size check
        if dim <= 0: raise ValueError(f"Calculated dimension 'dim' ({dim}) after initial CNN pooling is not positive. Check 'num_time'.")
        self.pos_embedding = nn.Parameter(torch.randn(1, num_kernel, dim))
        self.transformer = Transformer(dim=dim, depth=depth, heads=heads, dim_head=dim_head, mlp_dim=mlp_dim, dropout=dropout, in_chan=num_kernel, fine_grained_kernel=temporal_kernel)
        L = self.get_hidden_size(input_size=dim, num_layer=depth); final_seq_dim = L[-1]
        if final_seq_dim <= 0: raise ValueError(f"Calculated final sequence dimension ({final_seq_dim}) after Transformer is not positive. Check depth ({depth}) vs initial dim ({dim}).")
        out_size = int(num_kernel * final_seq_dim) + int(num_kernel * depth)
        self.mlp_head = nn.Sequential(nn.LayerNorm(out_size), nn.Linear(out_size, num_classes))
    def forward(self, eeg):
        eeg = torch.unsqueeze(eeg, dim=1); x = self.cnn_encoder(eeg); x = self.to_patch_embedding(x)
        if x.shape[-1] != self.pos_embedding.shape[-1]: # Add check for dimension mismatch before adding pos_embedding
             raise RuntimeError(f"Shape mismatch: x shape {x.shape} vs pos_embedding shape {self.pos_embedding.shape}")
        x += self.pos_embedding; x = self.transformer(x); return self.mlp_head(x)
    def get_padding(self, kernel): return (0, int(0.5 * (kernel - 1)))
    def get_hidden_size(self, input_size, num_layer):
        dims = [input_size]; current_size = input_size
        for _ in range(num_layer): current_size = current_size // 2; dims.append(current_size)
        return dims

print("New Model (Deformer) class defined.")


# In[ ]:


configs = Namespace()
configs.task_name = 'supervised'; configs.num_class = 6
configs.num_chan = 728; configs.num_time = 1296 # Data shape (B, C, T)
configs.temporal_kernel = 11; configs.num_kernel = 64
configs.depth = 4; configs.heads = 8; configs.mlp_dim = 128; configs.dim_head = 64
configs.dropout = 0.3
print("Model Configurations for Deformer:"); print({k: v for k, v in vars(configs).items()})


# In[ ]:


device = torch.device("cuda" if torch.cuda.is_available() else "cpu"); print(f"Using Device: {device}")
try:
    model = Deformer(
        num_chan=configs.num_chan, num_time=configs.num_time, temporal_kernel=configs.temporal_kernel,
        num_kernel=configs.num_kernel, num_classes=configs.num_class, depth=configs.depth,
        heads=configs.heads, mlp_dim=configs.mlp_dim, dim_head=configs.dim_head, dropout=configs.dropout
    ).to(device)
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"New model (Deformer) instantiated. Trainable params: {total_params/1e6:.2f}M")
except ValueError as ve: # Catch specific ValueError from __init__
    print("="*50); print(f"ERROR initializing model: {ve}"); print("="*50)
    raise
except Exception as e: print("="*50); print(f"ERROR initializing model: {e}"); print("="*50); raise e
criterion = nn.CrossEntropyLoss(); optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=6, verbose=True)
print("New model, criterion, optimizer, and scheduler set up.")


# In[ ]:


CHECKPOINT_PATH = "Deformer_v1.pth"; LOG_FILE_PATH = "Deformer_v1_log.csv"
print(f"Starting training for new model: {CHECKPOINT_PATH}")
try:
    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler,
                epochs=12, device=device, checkpoint_path=CHECKPOINT_PATH, log_file_path=LOG_FILE_PATH)
except ValueError as ve: # Catch specific ValueError from forward pass
     print(f"\nAn error occurred during training: {ve}")
     if "Dimension became zero after pooling" in str(ve): print(">>> Pooling Error: Input dimension became zero. Try reducing 'depth' in Cell 8. <<<")
     else: raise ve # Re-raise other ValueErrors
except RuntimeError as rte: # Catch RuntimeErrors like shape mismatches
     print(f"\nAn error occurred during training: {rte}")
     if "CUDA out of memory" in str(rte): print(">>> CUDA Out of Memory. Try reducing 'num_kernel', 'depth', 'mlp_dim', or 'dim_head' in Cell 8. <<<")
     else: raise rte # Re-raise other RuntimeErrors
except Exception as e:
    print(f"\nAn unexpected error occurred during training: {e}")
    raise e


# In[ ]:


# Activate environment and run all notebooks in separate tmux sessions
conda activate sleep_env && \
tmux new-session -d -s LGGNet 'jupyter nbconvert --to script LGGNet.ipynb && python3 LGGNet.py > LGGNet.log 2>&1' && \
tmux new-session -d -s EEG_Deformer 'jupyter nbconvert --to script EEG_Deformer.ipynb && python3 EEG_Deformer.py > EEG_Deformer.log 2>&1' && \
tmux ls


# In[ ]:


print(f"\nTesting model from {CHECKPOINT_PATH}...")
try:
    test_model_instance = Deformer(
        num_chan=configs.num_chan, num_time=configs.num_time, temporal_kernel=configs.temporal_kernel,
        num_kernel=configs.num_kernel, num_classes=configs.num_class, depth=configs.depth,
        heads=configs.heads, mlp_dim=configs.mlp_dim, dim_head=configs.dim_head, dropout=configs.dropout
    ).to(device)
    test_results = test_model(test_model_instance, test_loader, checkpoint_path=CHECKPOINT_PATH, device=device)
    print("\n--- Final Test Results Summary (Deformer) ---"); print(test_results)
except FileNotFoundError:
     print(f"Checkpoint file {CHECKPOINT_PATH} not found. Skipping testing.")
except ValueError as ve: # Catch specific ValueErrors from model init/forward
     print(f"A configuration error occurred during testing setup: {ve}")
except RuntimeError as rte: # Catch RuntimeErrors
     print(f"A runtime error occurred during testing: {rte}")
except Exception as e:
     print(f"An unexpected error occurred during testing: {e}")


# In[ ]:




