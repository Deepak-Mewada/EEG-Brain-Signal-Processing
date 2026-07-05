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
# Added for model configs and math
from argparse import Namespace
import math

# --- NEW IMPORTS ---
from tqdm.auto import tqdm # For progress bars
import copy # For the new model

warnings.filterwarnings('ignore')

print("All libraries imported.")


# In[2]:


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


# In[3]:


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

        # These are the correct partition shapes from your original code
        self.Epartition = np.zeros((200, 8), dtype='float32')
        self.Spartition = np.zeros((512, 16), dtype='float32')
        self.Partition = np.zeros((16, 1296), dtype='float32')

        print(f"Dataset Initialized. Final feature shape will be (728, 1296).")


    def __len__(self):
        return len(self.metaDF)

    def raweeg_50s_from_eeg_v1(self, parquet_path, eeg_label_offset_seconds):
        EEG_LENGTH = 50
        try:
            raw_eeg = pd.read_parquet(parquet_path)
        except Exception as e:
            # print(f"Error reading parquet file {parquet_path}: {e}")
            return np.zeros((200, 800), dtype=np.float32) # Return correct empty shape

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

        # This concatenates 4 regions, each (200, 200) wide -> (200, 800)
        eeg = np.concatenate(list_eeg, 1) 
        eeg /= 104
        return eeg

    def raweeg_20s_from_eeg_v1(self, parquet_path, eeg_label_offset_seconds):
        EEG_LENGTH = 20
        try:
            raw_eeg = pd.read_parquet(parquet_path)
        except Exception as e:
            # print(f"Error reading parquet file {parquet_path}: {e}")
            return np.zeros((200, 320), dtype=np.float32) # Return correct empty shape

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

        # This concatenates 4 regions, each (200, 80) wide -> (200, 320)
        eeg = np.concatenate(list_eeg, 1) 
        eeg /= 104
        return eeg

    def raweeg_10s_from_eeg_v1(self, parquet_path, eeg_label_offset_seconds):
        EEG_LENGTH = 10
        try:
            raw_eeg = pd.read_parquet(parquet_path)
        except Exception as e:
            # print(f"Error reading parquet file {parquet_path}: {e}")
            return np.zeros((200, 160), dtype=np.float32) # Return correct empty shape

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

        # This concatenates 4 regions, each (200, 40) wide -> (200, 160)
        eeg = np.concatenate(list_eeg, 1) 
        eeg /= 104
        return eeg

    def stft_spec_from_50s_eeg_v1(self, parquet_path, eeg_label_offset_seconds):
        EEG_LENGTH = 50
        try:
            eeg = pd.read_parquet(parquet_path)
        except Exception as e:
            # print(f"Error reading parquet file {parquet_path}: {e}")
            return np.zeros((512, 1024), dtype=np.float32) # Return correct empty shape

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
            # print(f"Error reading parquet file {parquet_path}: {e}")
            return np.zeros((512, 256), dtype=np.float32) # Return correct empty shape

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

    def __getitem__(self, idx):
        eeg_id, label, offset = self.metaDF.iloc[idx][["eeg_id", "expert_consensus", "eeg_label_offset_seconds"]]
        ppath = f'{self.base_dir}train_eegs/{eeg_id}.parquet'

        Xe50 = self.raweeg_50s_from_eeg_v1(ppath, offset) # (200, 800)
        Xe20 = self.raweeg_20s_from_eeg_v1(ppath, offset) # (200, 320)
        Xe10 = self.raweeg_10s_from_eeg_v1(ppath, offset) # (200, 160)

        # Xe shape = (200, 800 + 8 + 320 + 8 + 160) = (200, 1296)
        Xe = np.concatenate((Xe50, self.Epartition, Xe20, self.Epartition, Xe10), 1)

        del Xe50, Xe20, Xe10

        Xs50_1 = self.stft_spec_from_50s_eeg_v1(ppath, offset) # (512, 1024)
        Xs50_2 = self.stft_spec_from_50s_eeg_v2(ppath, offset) # (512, 256)

        # Xs shape = (512, 1024 + 16 + 256) = (512, 1296)
        Xs = np.concatenate((Xs50_1, self.Spartition, Xs50_2), 1)

        del Xs50_1, Xs50_2

        # Xe=(200, 1296), Partition=(16, 1296), Xs=(512, 1296)
        # Total X shape = (200+16+512, 1296) = (728, 1296). This is correct.
        X = np.concatenate((Xe, self.Partition, Xs), 0)

        X_tensor = torch.tensor(X, dtype=torch.float32) 

        y = self.activity_mapping[label]
        y_tensor = torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.long), num_classes=6).float()

        return X_tensor, y_tensor


# In[4]:


def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, epochs=20, device="cuda", checkpoint_path="best_model.pth", log_file_path="training_log.csv"):

    checkpoint_dir = os.path.dirname(checkpoint_path)
    if checkpoint_dir and not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir, exist_ok=True)
        print(f"Created directory for checkpoint: {checkpoint_dir}")

    if os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        try:
            model.load_state_dict(torch.load(checkpoint_path))
        except Exception as e:
            print(f"Error loading state_dict: {e}. Starting from scratch.")
    else:
        print(f"Starting training from scratch. Checkpoint will be saved to {checkpoint_path}")

    best_val_accuracy = 0.0

    log_dir = os.path.dirname(log_file_path)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)
        print(f"Created directory for log file: {log_dir}")

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

            # The model's forward() correctly handles this call
            # We pass X as x_enc, other args are None
            outputs = model(X, None, None, None) 

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

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%")

        model.eval()
        val_loss, correct, total = 0, 0, 0
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Epoch {epoch+1}/{epochs} [Val]", leave=False)
            for X, y in val_pbar:
                X, y = X.to(device), y.to(device)
                outputs = model(X, None, None, None)
                loss = criterion(outputs, torch.argmax(y, dim=1))
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += y.size(0)
                correct += predicted.eq(torch.argmax(y, dim=1)).sum().item()

                val_pbar.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.2f}%")

        val_accuracy = 100. * correct / total
        avg_val_loss = val_loss / len(val_loader)
        print(f"Validation Loss: {avg_val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%")

        scheduler.step(avg_val_loss)

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Best model saved to {checkpoint_path} with Validation Accuracy: {val_accuracy:.2f}%")

        epoch_time = time.time() - start_time
        print(f"Epoch {epoch+1} completed in {epoch_time:.2f} seconds")
        with open(log_file_path,"a") as f:
            f.write(f"{epoch+1},{avg_train_loss:.4f},{train_accuracy:.2f},{avg_val_loss:.4f},{val_accuracy:.2f},{epoch_time:.2f}\n")

    print(f"Training complete. Best Validation Accuracy: {best_val_accuracy:.2f}%")


def test_model(model, test_loader, checkpoint_path="best_model.pth", device="cuda"):
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint file not found at {checkpoint_path}")
        return {}

    print(f"Loading best model from {checkpoint_path} for testing...")
    try:
        model.load_state_dict(torch.load(checkpoint_path))
    except Exception as e:
        print(f"Error loading state_dict for testing: {e}")
        return {}

    model.to(device)
    model.eval()

    mname = os.path.splitext(checkpoint_path)[0]

    plot_dir = os.path.dirname(mname)
    if plot_dir and not os.path.exists(plot_dir):
        os.makedirs(plot_dir, exist_ok=True)
        print(f"Created directory for plots: {plot_dir}")

    all_preds, all_labels, all_probs = [], [], []
    total_time = 0.0
    total_samples = 0

    with torch.no_grad():
        for X, y in tqdm(test_loader, desc="Testing"):
            X, y = X.to(device), y.to(device)
            start_time = time.time()
            outputs = model(X, None, None, None)
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

    # ... (rest of the function is the same) ...

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    kappa = cohen_kappa_score(all_labels, all_preds)

    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=brain_activities, yticklabels=brain_activities)
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title(f"Confusion Matrix - {mname}")
    plt.savefig(f"{mname}_confusion_matrix.png")
    plt.close()
    print(f"Confusion matrix saved to {mname}_confusion_matrix.png")

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
            print(f"Skipping ROC for class {brain_activities[i]} as it's not in the test labels.")
            auc_scores.append(np.nan)

    macro_auc = np.nanmean(auc_scores) 
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {mname}")
    plt.legend()
    plt.savefig(f"{mname}_roc_curve.png")
    plt.close()
    print(f"ROC curve saved to {mname}_roc_curve.png")

    avg_inference_time = total_time / total_samples if total_samples > 0 else 0.0

    print("--- Test Results ---")
    print(f"Test Accuracy: {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall: {recall:.2%}")
    print(f"F1 Score: {f1:.2%}")
    print(f"Cohen's Kappa: {kappa:.2f}")
    print(f"Macro AUC: {macro_auc:.2f}")
    print(f"Specificity: {avg_specificity:.2%}")
    if seizure_sensitivity is not None:
        print(f"Seizure Sensitivity: {seizure_sensitivity:.2%}")
    else:
        print(f"Seizure Sensitivity: Class 'Seizure' not present in test set.")
    print(f"Average inference time per sample: {avg_inference_time * 1000:.2f} ms")

    return {
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1_score": f1, "kappa": kappa, "macro_auc": macro_auc,
        "specificity": avg_specificity, "seizure_sensitivity": seizure_sensitivity,
        "avg_inference_time_sec": avg_inference_time
    }

print("Train and Test functions defined with TQDM and directory fix.")


# In[5]:


# Instantiate Datasets
try:
    train_dataset = HMS_Dataset(metaDF=trainDF, base_dir=BASE_DIR, activity_mapping=activity_mapping)
    val_dataset = HMS_Dataset(metaDF=valDF, base_dir=BASE_DIR, activity_mapping=activity_mapping)
    test_dataset = HMS_Dataset(metaDF=testDF, base_dir=BASE_DIR, activity_mapping=activity_mapping)

    # --- Batch Size set to 1 ---
    BatchSize = 1 

    train_loader = DataLoader(train_dataset, batch_size=BatchSize, shuffle=True, num_workers=8, pin_memory=True, prefetch_factor=2)
    val_loader = DataLoader(val_dataset, batch_size=BatchSize, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)
    test_loader = DataLoader(test_dataset, batch_size=BatchSize, shuffle=False, num_workers=8, pin_memory=True, prefetch_factor=2)

    print(f"DataLoaders created with BatchSize = {BatchSize}.")

    # Check a batch shape to confirm our fix
    X_batch, y_batch = next(iter(train_loader))
    print(f"Batch X shape: {X_batch.shape}") # Should be [1, 728, 1296]
    print(f"Batch y shape: {y_batch.shape}") # Should be [1, 6]
except Exception as e:
    print(f"Error creating DataLoaders: {e}")
    print("This might be due to the BASE_DIR path being incorrect or file read errors.")


# In[6]:


# --- Define a namespace for our custom layers ---
# This avoids cluttering the global namespace
class layers:
    pass

# --- Define a namespace for utils ---
class utils:
    class tools:
        pass

# --- `layers.SelfAttention_Family` ---

class FullAttention(nn.Module):
    def __init__(self, mask_flag=True, factor=5, scale=None, attention_dropout=0.1, output_attention=False):
        super(FullAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag
        self.output_attention = output_attention
        self.dropout = nn.Dropout(attention_dropout)

    def forward(self, queries, keys, values, attn_mask):
        B, L, H, E = queries.shape
        _, S, _, D = values.shape
        scale = self.scale or 1. / math.sqrt(E)

        scores = torch.einsum("blhe,bshe->bhls", queries, keys)
        if self.mask_flag:
            if attn_mask is None:
                attn_mask = torch.triu(torch.ones(L, S, dtype=torch.bool), diagonal=1).to(queries.device)
            scores.masked_fill_(attn_mask, -np.inf)

        A = self.dropout(torch.softmax(scale * scores, dim=-1))
        V = torch.einsum("bhls,bshe->blhe", A, values)

        if self.output_attention:
            return (V.contiguous(), A)
        else:
            return (V.contiguous(), None)
layers.FullAttention = FullAttention


class AttentionLayer(nn.Module):
    def __init__(self, attention, d_model, n_heads, d_keys=None, d_values=None):
        super(AttentionLayer, self).__init__()
        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)

        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model, d_keys * n_heads)
        self.key_projection = nn.Linear(d_model, d_keys * n_heads)
        self.value_projection = nn.Linear(d_model, d_values * n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)
        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads

        queries = self.query_projection(queries).view(B, L, H, -1)
        keys = self.key_projection(keys).view(B, S, H, -1)
        values = self.value_projection(values).view(B, S, H, -1)

        out, attn = self.inner_attention(queries, keys, values, attn_mask)
        out = out.view(B, L, -1)

        return self.out_projection(out), attn
layers.AttentionLayer = AttentionLayer


# --- `layers.Transformer_EncDec` ---

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
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        new_x, attn = self.attention(x, x, x, attn_mask=attn_mask)
        x = x + self.dropout(new_x)
        y = x = self.norm1(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y), attn
layers.EncoderLayer = EncoderLayer


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
layers.Encoder = Encoder


class DecoderLayer(nn.Module):
    def __init__(self, self_attention, cross_attention, d_model, d_ff=None,
                 dropout=0.1, activation="relu"):
        super(DecoderLayer, self).__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        x = x + self.dropout(self.self_attention(x, x, x, attn_mask=x_mask)[0])
        x = self.norm1(x)
        x = x + self.dropout(self.cross_attention(x, cross, cross, attn_mask=cross_mask)[0])
        y = x = self.norm2(x)
        y = self.dropout(self.activation(self.conv1(y.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm3(x + y)
layers.DecoderLayer = DecoderLayer


class Decoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, cross, x_mask=None, cross_mask=None):
        for layer in self.layers:
            x = layer(x, cross, x_mask=x_mask, cross_mask=cross_mask)
        if self.norm is not None:
            x = self.norm(x)
        return x
layers.Decoder = Decoder


# --- `layers.Embed` ---

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return self.pe[:, :x.size(1)]
layers.PositionalEmbedding = PositionalEmbedding


class EEG2RepEmbedding(nn.Module):
    def __init__(self, c_in, d_model, pooling_size, dropout=0.1):
        super(EEG2RepEmbedding, self).__init__()
        self.c_in = c_in
        self.d_model = d_model

        # 1D conv to embed channels
        self.tokenConv = nn.Conv1d(in_channels=c_in, 
                                   out_channels=d_model, 
                                   kernel_size=3, 
                                   padding=1,
                                   bias=False)
        # Max pooling to downsample
        self.pooling = nn.MaxPool1d(kernel_size=pooling_size, stride=pooling_size)

        self.dropout = nn.Dropout(dropout)
        self.activation = nn.ELU()

    def forward(self, x):
        # x shape: [B, C, T] -> [B, 728, 1296]
        x = self.tokenConv(x) # Shape: [B, d_model, 1296]
        x = self.activation(x)
        x = self.pooling(x) # Shape: [B, d_model, 1296 / pooling_size]

        # Transformer expects [B, T, d_model]
        x = x.permute(0, 2, 1) 
        return self.dropout(x)
layers.EEG2RepEmbedding = EEG2RepEmbedding

# (DataEmbedding is not used by this model)
layers.DataEmbedding = None # Placeholder


# --- `utils.tools` ---
def semantic_subsequence_preserving(index, sub_len, mask_ratio):
    # This is a placeholder to make the import work.
    # The real function is complex and NOT needed for 'supervised' mode.
    print("Warning: Using placeholder 'semantic_subsequence_preserving'. This is OK for supervised mode.")
    total_len = len(index)
    mask_len = int(total_len * mask_ratio)
    visible_len = total_len - mask_len

    # Simple random selection (not semantic)
    visible_indices = np.random.choice(index, visible_len, replace=False)
    return [visible_indices] # Return a list containing an array of indices
utils.tools.semantic_subsequence_preserving = semantic_subsequence_preserving


print("All model dependencies (layers, utils) are defined.")


# In[8]:


# This cell uses the custom classes defined in Cell 6.
# The 'from layers...' imports have been removed as they caused the error.

class Model(nn.Module):
    """
        EEG2Rep: https://github.com/Navidfoumani/EEG2Rep
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.output_attention = configs.output_attention
        self.mask_ratio = configs.mask_ratio
        pooling_size = 2
        # Embedding
        self.PositionalEncoding = layers.PositionalEmbedding(configs.d_model)
        self.enc_embedding = layers.EEG2RepEmbedding(
            configs.enc_in,
            configs.d_model,
            pooling_size,
        )
        # Encoder
        self.encoder = layers.Encoder(
            [
                layers.EncoderLayer(
                    layers.AttentionLayer(
                        layers.FullAttention(
                            False,
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

        self.norm = nn.LayerNorm(configs.d_model)
        self.norm2 = nn.LayerNorm(configs.d_model)
        # Decoder
        if self.task_name == "supervised" or self.task_name == "finetune":
            self.projection = nn.Linear(
                configs.d_model,
                configs.num_class
            )
        elif self.task_name == "pretrain_eeg2rep":
            self.mask_token = nn.Parameter(torch.randn(configs.d_model, ))
            # for cross attention
            self.Predictor = layers.Decoder(
                [
                    layers.DecoderLayer(
                        layers.AttentionLayer(
                            layers.FullAttention(True, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        layers.AttentionLayer(
                            layers.FullAttention(False, configs.factor, attention_dropout=configs.dropout,
                                            output_attention=False),
                            configs.d_model, configs.n_heads),
                        configs.d_model,
                        configs.d_ff,
                        dropout=configs.dropout,
                        activation=configs.activation,
                    )
                    for l in range(configs.d_layers)
                ],
                norm_layer=torch.nn.LayerNorm(configs.d_model),
            )
            self.m_encoder = copy.deepcopy(self.encoder)  # same initialization as model
            for param in self.m_encoder.parameters():
                param.requires_grad = False

    def pretrain(self, x_enc, x_mark_enc):  # x_enc (batch_size, seq_length, enc_in)
        enc_out = self.enc_embedding(x_enc)
        output = self.norm(enc_out)
        output = output + self.PositionalEncoding(output)
        output = self.norm2(output)

        rep_mask_token = self.mask_token.repeat(output.shape[0], output.shape[1], 1)
        rep_mask_token = rep_mask_token + self.PositionalEncoding(rep_mask_token)

        index = np.arange(output.shape[1])
        index_chunk = utils.tools.semantic_subsequence_preserving(index, 2, self.mask_ratio)
        v_index = np.ravel(index_chunk)  # visible patches index (may have repeated index)
        m_index = np.setdiff1d(index, v_index)  # mask patches index

        visible = output[:, v_index, :]
        rep_mask_token = rep_mask_token[:, m_index, :]
        rep_contex, _ = self.encoder(visible, attn_mask=None)
        with torch.no_grad():
            rep_target, _ = self.m_encoder(output, attn_mask=None)
            rep_mask = rep_target[:, m_index, :]
        rep_mask_prediction = self.Predictor(rep_mask_token, rep_contex)  # cross attention
        return rep_mask, rep_mask_prediction

    def supervised(self, x_enc, x_mark_enc):
        # Embedding
        # x_enc shape: [B, 728, 1296]
        enc_out = self.enc_embedding(x_enc) # Shape: [B, 1296/2, d_model] -> [B, 648, d_model]
        output = self.norm(enc_out)
        output = output + self.PositionalEncoding(output) # Add positional encoding
        output = self.norm2(output)
        output, attns = self.encoder(output, attn_mask=None) # Shape: [B, 648, d_model]

        # Global Average Pooling
        output_pooled = torch.mean(output, dim=1) # Shape: [B, d_model]

        # Final projection
        return self.projection(output_pooled) # Shape: [B, num_class]

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == "supervised" or self.task_name == "finetune":
            dec_out = self.supervised(x_enc, x_mark_enc)
            return dec_out  # [B, N]
        elif self.task_name == "pretrain_eeg2rep":
            repr_m, repr_m_p = self.pretrain(x_enc, x_mark_enc)
            return repr_m, repr_m_p
        return None

print("New Model (EEG2Rep) class defined.")


# In[9]:


# Use argparse.Namespace to create a simple config object
configs = Namespace()

# --- Task ---
# CRITICAL: Set to 'supervised' to match the train_model function
configs.task_name = 'supervised' 
configs.num_class = 6

# --- Data shapes (from our fixed HMS_Dataset) ---
configs.enc_in = 728    # Input channels (C)
configs.seq_len = 1296   # Input sequence length (T)

# --- Model Architecture (Tune these as needed) ---
configs.d_model = 128   # Embedding dimension
configs.n_heads = 8     # Number of attention heads
configs.e_layers = 3    # Number of encoder layers
configs.d_layers = 1    # Number of decoder layers (for pretrain, but must be defined)
configs.d_ff = 512    # Feedforward dimension
configs.factor = 3      # Factor for FullAttention
configs.activation = 'gelu'

# --- Regularization & Pretrain Params ---
configs.dropout = 0.2
configs.mask_ratio = 0.0 # Not used in supervised mode
configs.output_attention = False 

# Print configs to verify
print("Model Configurations for EEG2Rep:")
print(configs)


# In[10]:


# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# Instantiate the new EEG2Rep model
try:
    model = Model(configs).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"New model (EEG2Rep) instantiated. Total trainable parameters: {total_params/1e6:.2f}M")

except Exception as e:
    print("="*50)
    print(f"ERROR: Failed to initialize model: {e}")
    print("="*50)
    raise e

# Setup loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

print("New model, criterion, optimizer, and scheduler are set up.")


# In[12]:


# Define paths for the new model
CHECKPOINT_PATH = "EEG2Rep_v1.pth"
LOG_FILE_PATH = "EEG2Rep_v1_log.csv"

print(f"Starting training for new model: {CHECKPOINT_PATH}")

try:
    train_model(model, 
                train_loader, 
                val_loader, 
                criterion, 
                optimizer, 
                scheduler, 
                epochs=1, # Set to desired number of epochs
                device=device, 
                checkpoint_path=CHECKPOINT_PATH, 
                log_file_path=LOG_FILE_PATH)
except Exception as e:
    print(f"\nAn error occurred during training: {e}")
    if "CUDA out of memory" in str(e):
        print(">>> CUDA Out of Memory. 'BatchSize' is already 1. Try reducing 'd_model', 'e_layers', or 'd_ff' in Cell 8. <<<")
    raise e


# In[13]:


print(f"\nTesting model from {CHECKPOINT_PATH}...")

# Re-create a model instance for testing
test_model_instance = Model(configs).to(device)

test_results = test_model(test_model_instance, 
                         test_loader, 
                         checkpoint_path=CHECKPOINT_PATH, 
                         device=device)

print("\n--- Final Test Results Summary (EEG2Rep) ---")
print(test_results)


# In[ ]:




