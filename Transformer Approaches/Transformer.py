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
    print(f"Error: train.csv not found at {BASE_DIR}")
    print("Please update the 'BASE_DIR' variable in this cell.")


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
        self.FEATS = [['Fp1','F7','T3','T5','O1'],
                      ['Fp1','F3','C3','P3','O1'],
                      ['Fp2','F8','T4','T6','O2'],
                      ['Fp2','F4','C4','P4','O2']]

        self.Epartition = np.zeros((200, 8), dtype='float32')
        self.Spartition = np.zeros((512, 16), dtype='float32')
        self.Partition = np.zeros((16, 1296), dtype='float32')

        print(f"Dataset Initialized. Output feature shape (T, C) will be (1296, 728).")


    def __len__(self):
        return len(self.metaDF)

    # --- Feature extraction functions (raweeg_..., stft_spec_...) ---
    # (These remain unchanged from previous versions)
    def raweeg_50s_from_eeg_v1(self, parquet_path, eeg_label_offset_seconds):
        EEG_LENGTH = 50
        try:
            raw_eeg = pd.read_parquet(parquet_path)
        except Exception as e:
            return np.zeros((200, 800), dtype=np.float32)
        time_temp = eeg_label_offset_seconds*200
        time_start =  round(time_temp + (50 - EEG_LENGTH) / 2 * 200) 
        time_stop =  round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
        eeg_default = raw_eeg.loc[time_start: (time_stop - 1), :].reset_index(drop=True)
        list_eeg = list()
        for region in self.RAW_FEATURES.keys():
            eeg = np.zeros((len(self.RAW_FEATURES[region]), eeg_default.shape[0]), dtype=np.float32)
            for chan_i, chan in enumerate(self.RAW_FEATURES[region]):
                eeg_1 = eeg_default.loc[:, chan.split('-')[0]]
                mean_value = eeg_1.mean(); eeg_1.fillna(value=mean_value, inplace=True); eeg_1 = eeg_1.values
                eeg_2 = eeg_default.loc[:, chan.split('-')[1]]
                mean_value = eeg_2.mean(); eeg_2.fillna(value=mean_value, inplace=True); eeg_2 = eeg_2.values
                new_eeg = eeg_1 - eeg_2
                del eeg_1, eeg_2
                new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
                new_eeg = np.clip(new_eeg, -1024, 1024).astype("float32")
                eeg[chan_i, :] = new_eeg
            eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
            eeg = np.concatenate((eeg[0,:,:], eeg[1,:,:], eeg[2,:,:], eeg[3,:,:]), 1)
            list_eeg.append(eeg)
        eeg = np.concatenate(list_eeg, 1) 
        eeg /= 104
        return eeg

    def raweeg_20s_from_eeg_v1(self, parquet_path, eeg_label_offset_seconds):
        EEG_LENGTH = 20
        try:
            raw_eeg = pd.read_parquet(parquet_path)
        except Exception as e:
            return np.zeros((200, 320), dtype=np.float32)
        time_temp = eeg_label_offset_seconds*200
        time_start =  round(time_temp + (50 - EEG_LENGTH) / 2 * 200) 
        time_stop =  round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
        eeg_default = raw_eeg.loc[time_start: (time_stop - 1), :].reset_index(drop=True)
        list_eeg = list()
        for region in self.RAW_FEATURES.keys():
            eeg = np.zeros((len(self.RAW_FEATURES[region]), eeg_default.shape[0]), dtype=np.float32)
            for chan_i, chan in enumerate(self.RAW_FEATURES[region]):
                eeg_1 = eeg_default.loc[:, chan.split('-')[0]]
                mean_value = eeg_1.mean(); eeg_1.fillna(value=mean_value, inplace=True); eeg_1 = eeg_1.values
                eeg_2 = eeg_default.loc[:, chan.split('-')[1]]
                mean_value = eeg_2.mean(); eeg_2.fillna(value=mean_value, inplace=True); eeg_2 = eeg_2.values
                new_eeg = eeg_1 - eeg_2
                del eeg_1, eeg_2
                new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
                new_eeg = np.clip(new_eeg, -1024, 1024).astype("float32")
                eeg[chan_i, :] = new_eeg
            eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
            eeg = np.concatenate((eeg[0,:,:], eeg[1,:,:], eeg[2,:,:], eeg[3,:,:]), 1)
            list_eeg.append(eeg)
        eeg = np.concatenate(list_eeg, 1) 
        eeg /= 104
        return eeg

    def raweeg_10s_from_eeg_v1(self, parquet_path, eeg_label_offset_seconds):
        EEG_LENGTH = 10
        try:
            raw_eeg = pd.read_parquet(parquet_path)
        except Exception as e:
            return np.zeros((200, 160), dtype=np.float32)
        time_temp = eeg_label_offset_seconds*200
        time_start =  round(time_temp + (50 - EEG_LENGTH) / 2 * 200) 
        time_stop =  round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
        eeg_default = raw_eeg.loc[time_start: (time_stop - 1), :].reset_index(drop=True)
        list_eeg = list()
        for region in self.RAW_FEATURES.keys():
            eeg = np.zeros((len(self.RAW_FEATURES[region]), eeg_default.shape[0]), dtype=np.float32)
            for chan_i, chan in enumerate(self.RAW_FEATURES[region]):
                eeg_1 = eeg_default.loc[:, chan.split('-')[0]]
                mean_value = eeg_1.mean(); eeg_1.fillna(value=mean_value, inplace=True); eeg_1 = eeg_1.values
                eeg_2 = eeg_default.loc[:, chan.split('-')[1]]
                mean_value = eeg_2.mean(); eeg_2.fillna(value=mean_value, inplace=True); eeg_2 = eeg_2.values
                new_eeg = eeg_1 - eeg_2
                del eeg_1, eeg_2
                new_eeg = signal.filtfilt(b, a, new_eeg, axis=0)
                new_eeg = np.clip(new_eeg, -1024, 1024).astype("float32")
                eeg[chan_i, :] = new_eeg
            eeg = np.reshape(eeg, (4, 200, EEG_LENGTH))
            eeg = np.concatenate((eeg[0,:,:], eeg[1,:,:], eeg[2,:,:], eeg[3,:,:]), 1)
            list_eeg.append(eeg)
        eeg = np.concatenate(list_eeg, 1) 
        eeg /= 104
        return eeg

    def stft_spec_from_50s_eeg_v1(self, parquet_path, eeg_label_offset_seconds):
        EEG_LENGTH = 50
        try:
            eeg = pd.read_parquet(parquet_path)
        except Exception as e:
            return np.zeros((512, 1024), dtype=np.float32)
        time_temp = eeg_label_offset_seconds*200
        time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200) 
        time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
        eeg = eeg.iloc[time_start: time_stop]
        list_eeg = list()
        for k in range(4):
            COLS = self.FEATS[k]
            spect = np.zeros((128,256,4),dtype='float32')
            for kk in range(4):
                eeg_1 = eeg[COLS[kk]]
                mean_value = eeg_1.mean(); eeg_1.fillna(value=mean_value, inplace=True); eeg_1 = eeg_1.values
                eeg_2 = eeg[COLS[kk+1]]
                mean_value = eeg_2.mean(); eeg_2.fillna(value=mean_value, inplace=True); eeg_2 = eeg_2.values
                new_eeg = eeg_1 - eeg_2
                del eeg_1, eeg_2
                fs = 200; nperseg = 39; noverlap = 0
                f, t, spec = signal.spectrogram(new_eeg, fs, nperseg=nperseg, noverlap=noverlap, nfft=256)
                spec = np.abs(spec); spec = np.log1p(spec).astype("float32")
                spect[:,:,kk] += spec[:128, :]
            spect = np.concatenate((spect[:,:,0], spect[:,:,1], spect[:,:,2], spect[:,:,3]), 1)
            list_eeg.append(spect)
        spect = np.concatenate(list_eeg, 0)
        spect /= 2
        return spect

    def stft_spec_from_50s_eeg_v2(self, parquet_path, eeg_label_offset_seconds):
        EEG_LENGTH = 50
        try:
            eeg = pd.read_parquet(parquet_path)
        except Exception as e:
            return np.zeros((512, 256), dtype=np.float32)
        time_temp = eeg_label_offset_seconds*200
        time_start = round(time_temp + (50 - EEG_LENGTH) / 2 * 200) 
        time_stop = round(time_temp + (50 + EEG_LENGTH) / 2 * 200)
        eeg = eeg.iloc[time_start: time_stop]
        spect = np.zeros((128,256,4),dtype='float32')
        for k in range(4):
            COLS = self.FEATS[k]
            for kk in range(4):
                eeg_1 = eeg[COLS[kk]]
                mean_value = eeg_1.mean(); eeg_1.fillna(value=mean_value, inplace=True); eeg_1 = eeg_1.values
                eeg_2 = eeg[COLS[kk+1]]
                mean_value = eeg_2.mean(); eeg_2.fillna(value=mean_value, inplace=True); eeg_2 = eeg_2.values
                new_eeg = eeg_1 - eeg_2
                del eeg_1, eeg_2
                fs = 200; nperseg = len(new_eeg)//256; noverlap = 0
                f, t, spec = signal.stft(new_eeg, fs, nperseg=nperseg, noverlap=noverlap, nfft=256)
                spec = np.abs(spec); spec = np.log1p(spec).astype("float32")
                spect[:,:,k] += spec[:128, 1:257]
            spect[:,:,k] /= 4
        spect = np.concatenate((spect[:,:,0], spect[:,:,1], spect[:,:,2], spect[:,:,3]), 0)
        return spect
    # --- End Feature extraction functions ---

    def __getitem__(self, idx):
        eeg_id, label, offset = self.metaDF.iloc[idx][["eeg_id", "expert_consensus", "eeg_label_offset_seconds"]]
        ppath = f'{self.base_dir}train_eegs/{eeg_id}.parquet'

        Xe50 = self.raweeg_50s_from_eeg_v1(ppath, offset) # (200, 800)
        Xe20 = self.raweeg_20s_from_eeg_v1(ppath, offset) # (200, 320)
        Xe10 = self.raweeg_10s_from_eeg_v1(ppath, offset) # (200, 160)
        Xe = np.concatenate((Xe50, self.Epartition, Xe20, self.Epartition, Xe10), 1) # (200, 1296)
        del Xe50, Xe20, Xe10

        Xs50_1 = self.stft_spec_from_50s_eeg_v1(ppath, offset) # (512, 1024)
        Xs50_2 = self.stft_spec_from_50s_eeg_v2(ppath, offset) # (512, 256)
        Xs = np.concatenate((Xs50_1, self.Spartition, Xs50_2), 1) # (512, 1296)
        del Xs50_1, Xs50_2

        # Final X shape = (728, 1296) -> (C, T)
        X = np.concatenate((Xe, self.Partition, Xs), 0)

        # Transpose X to (T, C) for Transformer/TimesNet style models
        X = X.transpose() # Shape: (1296, 728)

        X_tensor = torch.tensor(X, dtype=torch.float32) 

        y = self.activity_mapping[label]
        y_tensor = torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.long), num_classes=6).float()

        return X_tensor, y_tensor


# In[ ]:


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20, device="cuda", checkpoint_path="best_model.pth", log_file_path="training_log.csv"):

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True); print(f"Created directory: {checkpoint_dir}")

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        try: model.load_state_dict(torch.load(checkpoint_path))
        except Exception as e: print(f"Error loading state_dict: {e}. Starting from scratch.")
    else: print(f"Starting training from scratch. Checkpoint: {checkpoint_path}")

    best_val_accuracy = 0.0

    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True); print(f"Created directory: {log_dir}")

    if not os.path.exists(log_file_path) or os.path.getsize(log_file_path) == 0:
        with open(log_file_path,"w") as f: f.write("epoch#,train_loss,train_accuracy,val_loss,val_accuracy,time_taken\n")

    for epoch in range(epochs):
        start_time = time.time(); model.train(); train_loss, correct, total = 0, 0, 0
        train_pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs} [Train]", leave=False)
        for i, (X, y) in enumerate(train_pbar):
            X, y = X.to(device), y.to(device); optimizer.zero_grad()
            x_mark_enc = torch.ones(X.shape[0], X.shape[1]).to(device) # Dummy mark
            outputs = model(X, x_mark_enc, None, None) 
            loss = criterion(outputs, torch.argmax(y, dim=1)); loss.backward(); optimizer.step()
            train_loss += loss.item(); _, predicted = outputs.max(1); total += y.size(0)
            correct += predicted.eq(torch.argmax(y, dim=1)).sum().item()
            train_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

        train_accuracy = 100. * correct / total; avg_train_loss = train_loss / len(train_loader)
        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%")

        model.eval(); val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            for X, y in val_pbar:
                X, y = X.to(device), y.to(device)
                x_mark_enc = torch.ones(X.shape[0], X.shape[1]).to(device) # Dummy mark
                outputs = model(X, x_mark_enc, None, None)
                loss = criterion(outputs, torch.argmax(y, dim=1)); val_loss += loss.item()
                _, predicted = outputs.max(1); total += y.size(0)
                correct += predicted.eq(torch.argmax(y, dim=1)).sum().item()
                val_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

        val_accuracy = 100. * correct / total; avg_val_loss = val_loss / len(val_loader)
        print(f"Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%")
        scheduler.step(avg_val_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy; torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved: {checkpoint_path} @ Val Acc: {val_accuracy:.2f}%")

        epoch_time = time.time() - start_time; print(f"Epoch {epoch+1} Time: {epoch_time:.2f}s")
        with open(log_file_path,"a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{train_accuracy:.2f},{avg_val_loss:.4f},{val_accuracy:.2f},{epoch_time:.2f}\n")

    print(f"Training complete. Best Val Acc: {best_val_accuracy:.2f}%")


def test_model(model, test_loader, checkpoint_path="best_model.pth", device="cuda"):
    if not os.path.exists(checkpoint_path): print(f"Error: Checkpoint not found: {checkpoint_path}"); return {}
    print(f"Loading best model: {checkpoint_path}"); 
    try: model.load_state_dict(torch.load(checkpoint_path))
    except Exception as e: print(f"Error loading state_dict: {e}"); return {}
    model.to(device); model.eval(); mname = os.path.splitext(checkpoint_path)[0]
    plot_dir = os.path.dirname(mname)
    if plot_dir and not os.path.exists(plot_dir): os.makedirs(plot_dir, exist_ok=True); print(f"Created plot dir: {plot_dir}")
    all_preds, all_labels, all_probs = [], [], []; total_time = 0.0; total_samples = 0
    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Testing"):
            X, y = X.to(device), y.to(device); start_time = time.time()
            x_mark_enc = torch.ones(X.shape[0], X.shape[1]).to(device) # Dummy mark
            outputs = model(X, x_mark_enc, None, None); end_time = time.time()
            batch_time = end_time - start_time; total_time += batch_time; total_samples += X.size(0)
            probs = torch.nn.functional.softmax(outputs, dim=1); _, predicted = outputs.max(1)
            all_preds.extend(predicted.cpu().numpy()); all_labels.extend(torch.argmax(y, dim=1).cpu().numpy()); all_probs.extend(probs.cpu().numpy())
    all_preds = np.array(all_preds); all_labels = np.array(all_labels); all_probs = np.array(all_probs)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6)); sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=brain_activities, yticklabels=brain_activities)
    plt.xlabel("Predicted"); plt.ylabel("True"); plt.title(f"Confusion Matrix - {mname}"); plt.savefig(f"{mname}_confusion_matrix.png"); plt.close()
    print(f"Confusion matrix saved: {mname}_confusion_matrix.png")
    seizure_class_idx = activity_mapping.get('Seizure', -1)
    if seizure_class_idx != -1 and seizure_class_idx < cm.shape[0]:
        TP = cm[seizure_class_idx][seizure_class_idx]; FN = sum(cm[seizure_class_idx]) - TP
        seizure_sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    else: seizure_sensitivity = None
    num_classes_cm = cm.shape[0]; specificity_list = []
    for i in range(num_classes_cm):
        TP = cm[i, i]; FP = cm[:, i].sum() - TP; FN = cm[i, :].sum() - TP; TN = cm.sum() - (TP + FP + FN)
        specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0; specificity_list.append(specificity)
    avg_specificity = np.mean(specificity_list) if specificity_list else 0.0
    num_classes = all_probs.shape[1]; auc_scores = []; plt.figure(figsize=(8, 6))
    for i in range(num_classes):
        if i in all_labels:
            fpr, tpr, _ = roc_curve((all_labels == i).astype(int), all_probs[:, i]); roc_auc = auc(fpr, tpr)
            auc_scores.append(roc_auc); plt.plot(fpr, tpr, label=f'{brain_activities[i]} (AUC = {roc_auc:.2f})')
        else: print(f"Skipping ROC for {brain_activities[i]} (not in test labels)"); auc_scores.append(np.nan)
    macro_auc = np.nanmean(auc_scores) 
    plt.plot([0, 1], [0, 1], 'k--'); plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate"); plt.title(f"ROC Curve - {mname}"); plt.legend(); plt.savefig(f"{mname}_roc_curve.png"); plt.close()
    print(f"ROC curve saved: {mname}_roc_curve.png")
    avg_inference_time = total_time / total_samples if total_samples > 0 else 0.0
    print("--- Test Results ---"); print(f"Accuracy: {accuracy:.2%}"); print(f"Precision: {precision:.2%}"); print(f"Recall: {recall:.2%}"); print(f"F1 Score: {f1:.2%}"); print(f"Cohen's Kappa: {kappa:.2f}"); print(f"Macro AUC: {macro_auc:.2f}"); print(f"Specificity: {avg_specificity:.2%}")
    if seizure_sensitivity is not None: print(f"Seizure Sensitivity: {seizure_sensitivity:.2%}")
    else: print("Seizure Sensitivity: Class 'Seizure' not in test set.")
    print(f"Avg Inference Time: {avg_inference_time * 1000:.2f} ms")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "kappa": kappa, "macro_auc": macro_auc, "specificity": avg_specificity, "seizure_sensitivity": seizure_sensitivity, "avg_inference_time_sec": avg_inference_time}

print("Train and Test functions defined with TQDM, directory fix, and x_mark_enc compatibility.")


# In[ ]:


# Instantiate Datasets
try:
    train_dataset = HMS_Dataset(metaDF=trainDF, base_dir=BASE_DIR, activity_mapping=activity_mapping)
    val_dataset = HMS_Dataset(metaDF=valDF, base_dir=BASE_DIR, activity_mapping=activity_mapping)
    test_dataset = HMS_Dataset(metaDF=testDF, base_dir=BASE_DIR, activity_mapping=activity_mapping)

    # --- Batch Size set to 1 ---
    BatchSize = 2 

    train_loader = DataLoader(train_dataset, batch_size=BatchSize, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=BatchSize, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=BatchSize, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)

    print(f"DataLoaders created with BatchSize = {BatchSize}.")

    # Check a batch shape 
    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch X shape: {X_batch.shape}") # Should be [1, 1296, 728] -> (B, T, C)
    print(f"Batch y shape: {y_batch.shape}") # Should be [1, 6]
except Exception as e:
    print(f"Error creating DataLoaders: {e}")


# In[ ]:


# --- Define namespaces ---
class layers:
    class Transformer_EncDec: pass
    class SelfAttention_Family: pass
    class Embed: pass
    class Augmentation: pass # New namespace

# --- Basic Attention Blocks ---
class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale; self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)
    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape; _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)
        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None: attn_mask = torch.triu(torch.ones(L, S, dtype=torch.bool), diagonal=1).to(queries.device)
            scores.masked_fill_(attn_mask, -np.inf)
        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshe->blhe", A, values)
        if self.output_attention: return (V.contiguous(), A)
        else: return (V.contiguous(), None)

class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads); d_values = d_values or (d_model // n_heads)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads
    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape; _, S, _ = keys.shape; H = self.n_heads
        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)
        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)
        return self.out_projection(out), attn

# --- Transformer Encoder Blocks ---
class EncoderLayer(nn.Module):
    def __init__(self, attention, d_model, d_ff=None, dropout=0.1, activation="relu"):
        super(EncoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.attention = attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ReLU() if activation == "relu" else nn.GELU()

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn

class Encoder(nn.Module):
    def __init__(self, attn_layers, norm_layer=None):
        super(Encoder, self).__init__()
        self.attn_layers = nn.ModuleList(attn_layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        attns = []
        for attn_layer in self.attn_layers:
            x, attn = attn_layer(x, attn_mask=attn_mask)
            attns.append(attn)
        if self.norm is not None:
            x = self.norm(x)
        return x, attns

# --- Embeddings ---
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float(); pe.require_grad = False
        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0); self.register_buffer('pe', pe)
    def forward(self, x): return self.pe[:, :x.size(1)]

class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model):
        super(TokenEmbedding, self).__init__()
        self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, 
                                   kernel_size=3, padding=1, padding_mode='circular', bias=False)
        for m in self.modules():
            if isinstance(m, nn.Conv1d): nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu')
    def forward(self, x):
        x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2)
        return x 

class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1):
        super(DataEmbedding, self).__init__()
        self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model)
        self.position_embedding = PositionalEmbedding(d_model=d_model)
        self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, x_mark):
        x = self.value_embedding(x) + self.position_embedding(x)
        return self.dropout(x) 

# --- Augmentations (Placeholders for supervised mode) ---
class AugmentationBase(nn.Module):
    """Base class for augmentations - does nothing."""
    def forward(self, x): return x

class AugFlip(AugmentationBase): pass
class AugFrequency(AugmentationBase): pass
class AugJitter(AugmentationBase): pass
class AugMask(AugmentationBase): pass
class AugChannel(AugmentationBase): pass
class AugDrop(AugmentationBase): pass
class AugNone(AugmentationBase): pass

def get_augmentation(aug_name):
    """Returns a placeholder augmentation module."""
    if aug_name == "flip": return AugFlip()
    if aug_name == "frequency": return AugFrequency()
    if aug_name == "jitter": return AugJitter()
    if aug_name == "mask": return AugMask()
    if aug_name == "channel": return AugChannel()
    if aug_name == "drop": return AugDrop()
    if aug_name == "none": return AugNone()
    print(f"Warning: Unknown augmentation '{aug_name}'. Returning identity.")
    return AugmentationBase()

# --- Assign classes to the namespace ---
layers.SelfAttention_Family.FullAttention = FullAttention
layers.SelfAttention_Family.AttentionLayer = AttentionLayer
layers.Transformer_EncDec.EncoderLayer = EncoderLayer
layers.Transformer_EncDec.Encoder = Encoder
layers.Embed.DataEmbedding = DataEmbedding
layers.Augmentation.get_augmentation = get_augmentation # Add augmentation function

print("All model dependencies (Transformer + Placeholders) are defined.")


# In[ ]:


class Model(nn.Module):
    """
    Vanilla Transformer
    """
    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.output_attention = configs.output_attention
        augmentations = configs.augmentations.split(",")
        # Augmentations are needed for pretraining, force if needed
        if augmentations == ["none"] and "pretrain" in self.task_name:
            augmentations = ["flip", "frequency", "jitter", "mask", "channel", "drop"]

        # This will now use the placeholder augmentations from Cell 6
        self.augmentation = nn.ModuleList(
            [layers.Augmentation.get_augmentation(aug) for aug in augmentations]
        )
        # Embedding
        self.enc_embedding = layers.Embed.DataEmbedding(
            configs.enc_in,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        # Encoder
        self.encoder = layers.Transformer_EncDec.Encoder(
            [
                layers.Transformer_EncDec.EncoderLayer(
                    layers.SelfAttention_Family.AttentionLayer(
                        layers.SelfAttention_Family.FullAttention(
                            False, # No masking for classification
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )

        # Decoder / Projection Head
        self.act = F.gelu
        self.dropout = nn.Dropout(configs.dropout)
        if self.task_name == "supervised" or self.task_name == "finetune":
            self.projection = nn.Linear(
                configs.d_model * configs.seq_len, configs.num_class
            )

    def supervised(self, x_enc, x_mark_enc):
        # x_enc shape: [B, T, C_in] -> [1, 1296, 728]
        # x_mark_enc is ignored (passed as None below)

        # Embedding -> [B, T, d_model]
        enc_out = self.enc_embedding(x_enc, None) 
        # Encoder -> [B, T, d_model]
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # Output
        output = self.act(enc_out)
        output = self.dropout(output)
        # Reshape -> [B, T * d_model]
        output = output.reshape(output.shape[0], -1) 
        # Projection -> [B, num_classes]
        output = self.projection(output) 
        return output

    def pretrain(self, x_enc, x_mark_enc): 
        # Data augmentation (Input: [B, T, C] -> Transpose -> Augment -> Transpose back)
        x_enc = x_enc.permute(0, 2, 1)  # [B, C, T]
        aug_idx = random.randint(0, len(self.augmentation) - 1)
        x_enc = self.augmentation[aug_idx](x_enc) # Apply placeholder augmentation
        x_enc = x_enc.permute(0, 2, 1)  # [B, T, C]
        # Embedding
        enc_out = self.enc_embedding(x_enc, None)
        enc_out, attns = self.encoder(enc_out, attn_mask=None)
        # Pooling
        repr_out = enc_out.mean(dim=1).reshape(enc_out.shape[0], -1)
        return enc_out, repr_out 

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "supervised" or self.task_name == "finetune":
            dec_out = self.supervised(x_enc, x_mark_enc) # Pass x_mark_enc, though supervised ignores it
            return dec_out 
        elif self.task_name == "pretrain_ts2vec": # Example pretrain task
            repr_h, repr_z = self.pretrain(x_enc, x_mark_enc)
            return repr_h, repr_z
        else:
            raise ValueError(f"Task name '{self.task_name}' not recognized.")

print("New Model (Transformer) class defined.")


# In[ ]:


# Use argparse.Namespace to create a simple config object
configs = Namespace()

# --- Task ---
configs.task_name = 'supervised' 
configs.num_class = 6

# --- Data shapes (from our fixed HMS_Dataset) ---
# Data is (B, 1296, 728) -> (B, T, C)
configs.seq_len = 1296   # Timestamps (T)
configs.enc_in = 728    # Channels (C)

# --- Model Architecture (Tune these as needed) ---
configs.d_model = 128   # Embedding dimension
configs.n_heads = 8     # Number of attention heads
configs.e_layers = 3    # Number of encoder layers
configs.d_ff = 512    # Feedforward dimension
configs.factor = 3      # Factor for FullAttention
configs.activation = 'gelu'

# --- Embedding ---
configs.embed = 'timeF' # Not really used by our simple embedding, but set
configs.freq = 's'      # Not really used

# --- Augmentation (Not used in supervised) ---
configs.augmentations = "none" 

# --- Regularization & Other ---
configs.dropout = 0.2
configs.output_attention = False 

# Print configs to verify
print("Model Configurations for Transformer:")
print(configs)


# In[ ]:


# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# Instantiate the new Transformer model
try:
    model = Model(configs).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"New model (Transformer) instantiated. Trainable params: {total_params/1e6:.2f}M")

except Exception as e:
    print("="*50); print(f"ERROR initializing model: {e}"); print("="*50)
    raise e

# Setup loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

print("New model, criterion, optimizer, and scheduler are set up.")


# In[ ]:


CHECKPOINT_PATH = "Transformer_v1.pth"
LOG_FILE_PATH = "Transformer_v1_log.csv"

print(f"Starting training for new model: {CHECKPOINT_PATH}")

try:
    train_model(model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                scheduler, 
                epochs=12, # Set to desired number of epochs
                device=device, 
                checkpoint_path=CHECKPOINT_PATH, 
                log_file_path=LOG_FILE_PATH)
except Exception as e:
    print(f"\nAn error occurred during training: {e}")
    if "CUDA out of memory" in str(e):
        print(">>> CUDA Out of Memory. 'BatchSize' is already 1. Try reducing 'd_model', 'e_layers', or 'd_ff' in Cell 8. <<<")
    raise e


# In[ ]:


print(f"\nTesting model from {CHECKPOINT_PATH}...")

# Re-create a model instance for testing
test_model_instance = Model(configs).to(device)

test_results = test_model(test_model_instance, 
                         test_loader, 
                         checkpoint_path=CHECKPOINT_PATH, 
                         device=device)

print("\n--- Final Test Results Summary (Transformer) ---")
print(test_results)

