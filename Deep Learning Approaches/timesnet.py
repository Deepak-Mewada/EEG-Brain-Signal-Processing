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

        # Final X shape = (728, 1296)
        X = np.concatenate((Xe, self.Partition, Xs), 0)

        # --- INPUT SHAPE CHANGE ---
        # The TCN and TimesNet models expect (B, T, C)
        # Our X is (C, T) = (728, 1296)
        # We will keep it this way (C, T) and let the Dataloader batch it to (B, C, T)
        # The model's `forward` pass will then get (B, C, T) and must transpose it
        #
        # Let's re-check the model.
        # TCN model: `self.encoder(x_enc.transpose(1, 2))` -> Expects (B, T, C)
        # TimesNet model: `self.enc_embedding(x_enc, None)` -> Expects (B, T, C)
        #
        # This means our Dataset MUST return (T, C) not (C, T).
        # We must transpose X here.

        X = X.transpose() # Shape from (728, 1296) to (1296, 728) -> (T, C)

        X_tensor = torch.tensor(X, dtype=torch.float32) 

        y = self.activity_mapping[label]
        y_tensor = torch.nn.functional.one_hot(torch.tensor(y, dtype=torch.long), num_classes=6).float()

        return X_tensor, y_tensor


# In[ ]:


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

            # --- TIMESNET FIX ---
            # Create a dummy x_mark_enc tensor of shape (B, T) with all ones.
            # X shape is (B, T, C) -> e.g., (1, 1296, 728)
            x_mark_enc = torch.ones(X.shape[0], X.shape[1]).to(device)

            # Pass it to the model
            outputs = model(X, x_mark_enc, None, None) 
            # --- END FIX ---

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

                # --- TIMESNET FIX ---
                x_mark_enc = torch.ones(X.shape[0], X.shape[1]).to(device)
                outputs = model(X, x_mark_enc, None, None)
                # --- END FIX ---

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

            # --- TIMESNET FIX ---
            x_mark_enc = torch.ones(X.shape[0], X.shape[1]).to(device)
            outputs = model(X, x_mark_enc, None, None)
            # --- END FIX ---

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

print("Train and Test functions defined with TQDM, directory fix, and x_mark_enc fix.")


# In[ ]:


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
    print(f"Batch X shape: {X_batch.shape}") # Should be [1, 1296, 728] -> (B, T, C)
    print(f"Batch y shape: {y_batch.shape}") # Should be [1, 6]
except Exception as e:
    print(f"Error creating DataLoaders: {e}")
    print("This might be due to the BASE_DIR path being incorrect or file read errors.")


# In[ ]:


# %%
# --- Define namespaces for our custom layers ---
class layers:
    class Conv_Blocks: pass
    class Embed: pass

# --- `layers.Conv_Blocks` ---
class Inception_Block_V1(nn.Module):
    """
    A 2D Inception block as used in TimesNet.
    Splits the output channels among a list of kernels.
    Uses 'same' padding.
    Includes check to skip kernels larger than input size.
    """
    def __init__(self, in_channels, out_channels, num_kernels=6, kernel_size=1):
        super(Inception_Block_V1, self).__init__()
        self.num_kernels = num_kernels
        self.in_channels = in_channels
        self.out_channels = out_channels

        if out_channels % num_kernels != 0:
             # Adjust num_kernels if out_channels is not divisible
             self.num_kernels = 1 # Fallback to a single kernel
             print(f"Warning: out_channels ({out_channels}) not divisible by num_kernels ({num_kernels}). Falling back to num_kernels=1.")
             out_per_kernel = out_channels # Assign all channels to the single kernel
        else:
             out_per_kernel = out_channels // num_kernels

        self.kernels = nn.ModuleList()
        self.kernel_sizes = [] # Store kernel sizes for checking
        for i in range(self.num_kernels):
            eff_kernel_size = kernel_size * i + 1
            self.kernel_sizes.append(eff_kernel_size)
            # Use padding='same'
            self.kernels.append(
                nn.Conv2d(in_channels, out_per_kernel,
                          kernel_size=eff_kernel_size, # Use eff_kernel_size directly
                          padding='same')
            )

    def forward(self, x):
        # x shape: [B, C_in, H, W]
        res = []
        input_h, input_w = x.shape[2], x.shape[3]

        for i in range(self.num_kernels):
            eff_kernel_size = self.kernel_sizes[i]
            # --- KERNEL SIZE CHECK ---
            # Check if kernel fits the input dimensions
            if eff_kernel_size <= input_h and eff_kernel_size <= input_w:
                res.append(self.kernels[i](x))
            # --- END CHECK ---

        if not res:
            # Fallback if no kernels could be applied (e.g., input is 1x1 and all kernels > 1)
            # Apply the first kernel (usually 1x1) if it exists, otherwise return zeros
            print(f"Warning: No kernels fit input size ({input_h}x{input_w}). Applying 1x1 fallback if possible.")
            if self.num_kernels > 0 and self.kernel_sizes[0] <= input_h and self.kernel_sizes[0] <= input_w:
                 # Ensure output has correct number of channels even if only 1x1 runs
                 first_kernel_out = self.kernels[0](x)
                 if self.num_kernels > 1:
                     # Pad with zeros if other kernels were skipped
                     zeros_to_pad = self.out_channels - first_kernel_out.shape[1]
                     if zeros_to_pad > 0:
                         padding = torch.zeros(*first_kernel_out.shape[:1], zeros_to_pad, *first_kernel_out.shape[2:], device=x.device)
                         return torch.cat((first_kernel_out, padding), dim=1)
                     else:
                          return first_kernel_out
                 else: # Only one kernel defined
                     return first_kernel_out

            else: # Cannot even apply 1x1
                 print(f"Error: Cannot apply any kernel to input size ({input_h}x{input_w}). Returning zeros.")
                 return torch.zeros(x.shape[0], self.out_channels, input_h, input_w, device=x.device)


        # Concatenate results from applicable kernels
        output = torch.cat(res, dim=1)

        # Pad output channels with zeros if some kernels were skipped
        if output.shape[1] < self.out_channels:
             zeros_to_pad = self.out_channels - output.shape[1]
             padding = torch.zeros(*output.shape[:1], zeros_to_pad, *output.shape[2:], device=x.device)
             output = torch.cat((output, padding), dim=1)

        return output # Shape: [B, C_out, H, W]

layers.Conv_Blocks.Inception_Block_V1 = Inception_Block_V1


# --- `layers.Embed` --- (No changes below)
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000): super(PositionalEmbedding, self).__init__(); pe = torch.zeros(max_len, d_model).float(); pe.require_grad = False; position = torch.arange(0, max_len).float().unsqueeze(1); div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp(); pe[:, 0::2] = torch.sin(position * div_term); pe[:, 1::2] = torch.cos(position * div_term); pe = pe.unsqueeze(0); self.register_buffer('pe', pe)
    def forward(self, x): return self.pe[:, :x.size(1)]
class TokenEmbedding(nn.Module):
    def __init__(self, c_in, d_model): super(TokenEmbedding, self).__init__(); self.tokenConv = nn.Conv1d(in_channels=c_in, out_channels=d_model, kernel_size=3, padding=1, padding_mode='circular', bias=False); [nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='leaky_relu') for m in self.modules() if isinstance(m, nn.Conv1d)]
    def forward(self, x): x = self.tokenConv(x.permute(0, 2, 1)).transpose(1, 2); return x
class DataEmbedding(nn.Module):
    def __init__(self, c_in, d_model, embed_type='fixed', freq='h', dropout=0.1): super(DataEmbedding, self).__init__(); self.value_embedding = TokenEmbedding(c_in=c_in, d_model=d_model); self.position_embedding = PositionalEmbedding(d_model=d_model); self.dropout = nn.Dropout(p=dropout)
    def forward(self, x, x_mark): x = self.value_embedding(x) + self.position_embedding(x); return self.dropout(x)
layers.Embed.DataEmbedding = DataEmbedding

print("All model dependencies (TimesNet) defined with robust Inception block.")


# In[ ]:


# %%
def FFT_for_Period(x, k=2):
    # x: [B, T, C]
    xf = torch.fft.rfft(x, dim=1) # [B, T//2+1, C]
    frequency_list = abs(xf).mean(0).mean(-1) # Average over Batch and Channel -> [T//2+1]
    frequency_list[0] = 0 # Remove DC component
    _, top_list = torch.topk(frequency_list, k) # Indices of top k frequencies
    top_list = top_list.detach().cpu().numpy()
    period = x.shape[1] // top_list # Calculate periods: T / freq_index
    # Return periods and weights (amplitudes of top frequencies)
    return period, abs(xf).mean(-1)[:, top_list] # [B, k]

class TimesBlock(nn.Module):
    def __init__(self, configs):
        super(TimesBlock, self).__init__(); self.seq_len = configs.seq_len; self.pred_len = configs.pred_len; self.k = configs.top_k
        self.conv = nn.Sequential(layers.Conv_Blocks.Inception_Block_V1(configs.d_model, configs.d_ff, num_kernels=configs.num_kernels), nn.GELU(), layers.Conv_Blocks.Inception_Block_V1(configs.d_ff, configs.d_model, num_kernels=configs.num_kernels))
    def forward(self, x):
        B, T, N = x.size() # x shape: [B, T=seq_len, N=d_model]
        period_list, period_weight = FFT_for_Period(x, self.k)
        res = []
        for i in range(self.k):
            period = period_list[i]
            # Handle potential division by zero or invalid period
            if period == 0 or T % period != 0: # Simplified padding logic, assumes T is divisible for simplicity here
                 # Basic padding if not divisible - might need refinement based on exact paper impl.
                 padding_len = period - (T % period) if period != 0 else 0
                 padding = torch.zeros([B, padding_len, N], device=x.device)
                 out = torch.cat([x, padding], dim=1)
                 length = T + padding_len
            else: length = T; out = x

            if length // period <= 0: continue # Skip if period is larger than length

            out = out.reshape(B, length // period, period, N).permute(0, 3, 1, 2).contiguous() # [B, N, L//P, P]
            out = self.conv(out) # [B, N, L//P, P]
            out = out.permute(0, 2, 3, 1).reshape(B, -1, N) # [B, L, N]
            res.append(out[:, :T, :]) # Truncate back to original seq_len T
        if not res: return x # Return input if no valid periods found
        res = torch.stack(res, dim=-1) # [B, T, N, k]
        period_weight = F.softmax(period_weight, dim=1) # [B, k]
        period_weight = period_weight.unsqueeze(1).unsqueeze(1).repeat(1, T, N, 1) # [B, T, N, k]
        res = torch.sum(res * period_weight, -1) # Weighted sum -> [B, T, N]
        res = res + x # Residual connection
        return res

class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__(); self.configs = configs; self.task_name = configs.task_name; self.seq_len = configs.seq_len; self.label_len = configs.label_len
        self.model = nn.ModuleList([TimesBlock(configs) for _ in range(configs.e_layers)])
        self.enc_embedding = layers.Embed.DataEmbedding(configs.enc_in, configs.d_model, configs.embed, configs.freq, configs.dropout)
        self.layer = configs.e_layers; self.layer_norm = nn.LayerNorm(configs.d_model)
        if self.task_name == 'supervised':
            self.act = F.gelu; self.dropout = nn.Dropout(configs.dropout)
            # Adjust projection layer input size
            projection_in_features = configs.d_model * configs.seq_len
            self.projection = nn.Linear(projection_in_features, configs.num_class)

    def supervised(self, x_enc, x_mark_enc):
        # x_enc shape: [B, T, C_in] -> [1, 1296, 728]
        enc_out = self.enc_embedding(x_enc, x_mark_enc) # [B, T, d_model] -> [1, 1296, 192]
        for i in range(self.layer): enc_out = self.layer_norm(self.model[i](enc_out))
        output = self.act(enc_out); output = self.dropout(output)
        if x_mark_enc is not None: output = output * x_mark_enc.unsqueeze(-1)
        output = output.reshape(output.shape[0], -1) # [B, T * d_model]
        output = self.projection(output) # [B, num_classes]
        return output

    def forward(self, x_enc, x_mark_enc, x_dec, x_mark_dec, mask=None):
        if self.task_name == 'supervised': return self.supervised(x_enc, x_mark_enc)
        return None

print("New Model (TimesNet) class defined.")


# In[ ]:


# Use argparse.Namespace to create a simple config object
configs = Namespace()

# --- Task ---
configs.task_name = 'supervised' 
configs.num_class = 6
configs.pred_len = 0  # Not used for classification
configs.label_len = 0 # Not used for classification

# --- Data shapes (from our fixed HMS_Dataset) ---
# Data is (B, 1296, 728) -> (B, T, C)
configs.seq_len = 1296   # Timestamps (T)
configs.enc_in = 728    # Channels (C)

# --- Model Architecture (Tune these as needed) ---
configs.e_layers = 2    # Number of TimesBlock layers
configs.top_k = 5       # Top k frequencies in TimesBlock
configs.num_kernels = 6 # Number of kernels in InceptionBlock
configs.d_model = 192   # Embedding dim (must be divisible by num_kernels)
configs.d_ff = 384    # Feedforward dim (must be divisible by num_kernels)
configs.activation = 'gelu'

# --- Embedding ---
configs.embed = 'timeF' # Not really used by our simple embedding, but set
configs.freq = 's'      # Not really used

# --- Regularization ---
configs.dropout = 0.2
configs.output_attention = False 

# Print configs to verify
print("Model Configurations for TimesNet:")
print(configs)


# In[ ]:


# Model setup
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using Device: {device}")

# Instantiate the new TimesNet model
try:
    model = Model(configs).to(device)

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"New model (TimesNet) instantiated. Total trainable parameters: {total_params/1e6:.2f}M")

except Exception as e:
    print("="*50)
    print(f"ERROR: Failed to initialize model: {e}")
    print("This may be due to 'd_model' or 'd_ff' not being divisible by 'num_kernels'.")
    print("="*50)
    raise e

# Setup loss, optimizer, and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3, verbose=True)

print("New model, criterion, optimizer, and scheduler are set up.")


# In[ ]:


# Define paths for the new model
CHECKPOINT_PATH = "TimesNet_v1.pth"
LOG_FILE_PATH = "TimesNet_v1_log.csv"

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
        print(">>> CUDA Out of Memory. 'BatchSize' is already 1. Try reducing 'd_model', 'd_ff', or 'e_layers' in Cell 8. <<<")
    raise e


# In[ ]:


print(f"\nTesting model from {CHECKPOINT_PATH}...")

# Re-create a model instance for testing
test_model_instance = Model(configs).to(device)

test_results = test_model(test_model_instance, 
                         test_loader, 
                         checkpoint_path=CHECKPOINT_PATH, 
                         device=device)

print("\n--- Final Test Results Summary (TimesNet) ---")
print(test_results)

