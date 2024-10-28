# -*- coding: utf-8 -*-
"""
@Time: Created on 2020/7/05
@author: Qichang Zhao
"""

import random
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from model import AttentionDTI
from dataset import CustomDataSet, collate_fn
from prefetch_generator import BackgroundGenerator
from tqdm import tqdm
from hyperparameter import hyperparameter
from pytorchtools import EarlyStopping
from tensorboardX import SummaryWriter
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, precision_recall_curve, auc

def show_result(DATASET, label, Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List):
    Accuracy_mean, Accuracy_var = np.mean(Accuracy_List), np.var(Accuracy_List)
    Precision_mean, Precision_var = np.mean(Precision_List), np.var(Precision_List)
    Recall_mean, Recall_var = np.mean(Recall_List), np.var(Recall_List)
    AUC_mean, AUC_var = np.mean(AUC_List), np.var(AUC_List)
    PRC_mean, PRC_var = np.mean(AUPR_List), np.var(AUPR_List)
    print(f"The {label} model's results:")
    with open(f"./{DATASET}/results.txt", 'w') as f:
        f.write(f'Accuracy(std): {Accuracy_mean:.4f} ({Accuracy_var:.4f})\n')
        f.write(f'Precision(std): {Precision_mean:.4f} ({Precision_var:.4f})\n')
        f.write(f'Recall(std): {Recall_mean:.4f} ({Recall_var:.4f})\n')
        f.write(f'AUC(std): {AUC_mean:.4f} ({AUC_var:.4f})\n')
        f.write(f'PRC(std): {PRC_mean:.4f} ({PRC_var:.4f})\n')
    print(f'Accuracy(std): {Accuracy_mean:.4f} ({Accuracy_var:.4f})')
    print(f'Precision(std): {Precision_mean:.4f} ({Precision_var:.4f})')
    print(f'Recall(std): {Recall_mean:.4f} ({Recall_var:.4f})')
    print(f'AUC(std): {AUC_mean:.4f} ({AUC_var:.4f})')
    print(f'PRC(std): {PRC_mean:.4f} ({PRC_var:.4f})')

def load_tensor(file_name, dtype):
    return [dtype(d) for d in np.load(file_name + '.npy', allow_pickle=True)]

def test_process(model, pbar, LOSS):
    model.eval()
    test_losses = []
    Y, P, S = [], [], []
    with torch.no_grad():
        for _, data in pbar:
            compounds, proteins, labels = data
            compounds, proteins, labels = compounds.to(device), proteins.to(device), labels.to(device)

            predicted_scores = model(compounds, proteins)
            loss = LOSS(predicted_scores, labels)
            correct_labels = labels.cpu().data.numpy()
            predicted_scores = F.softmax(predicted_scores, 1).cpu().data.numpy()
            predicted_labels = np.argmax(predicted_scores, axis=1)
            predicted_scores = predicted_scores[:, 1]

            Y.extend(correct_labels)
            P.extend(predicted_labels)
            S.extend(predicted_scores)
            test_losses.append(loss.item())

    # Calculate metrics
    accuracy = accuracy_score(Y, P)
    # Use zero_division parameter to avoid warning
    precision = precision_score(Y, P, zero_division=0)
    recall = recall_score(Y, P)
    auc_score = roc_auc_score(Y, S)

    # Get precision-recall values and ensure they are sorted
    precision_values, recall_values, _ = precision_recall_curve(Y, S)
    sorted_indices = np.argsort(recall_values)
    sorted_precision = precision_values[sorted_indices]
    sorted_recall = recall_values[sorted_indices]

    # Calculate AUC for the precision-recall curve
    prc_auc = auc(sorted_recall, sorted_precision)

    return {
        "Y": Y,
        "P": P,
        "loss": np.average(test_losses),
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "AUC": auc_score,
        "PRC": prc_auc
    }

def test_model(model, dataset_load, save_path, DATASET, LOSS, dataset="Train", label="best", save=True):
    test_pbar = tqdm(enumerate(BackgroundGenerator(dataset_load)), total=len(dataset_load))
    test_results = test_process(model, test_pbar, LOSS)

    if save:
        with open(f"{save_path}/{DATASET}_{dataset}_{label}_prediction.txt", 'a') as f:
            for i in range(len(test_results["Y"])):
                f.write(f"{test_results['Y'][i]} {test_results['P'][i]}\n")

    results_msg = (
        f'{label}_set--Loss: {test_results["loss"]:.5f}; Accuracy: {test_results["Accuracy"]:.5f}; '
        f'Precision: {test_results["Precision"]:.5f}; Recall: {test_results["Recall"]:.5f}; '
        f'AUC: {test_results["AUC"]:.5f}; PRC: {test_results["PRC"]:.5f}.'
    )
    print(results_msg)
    return results_msg, test_results

def get_kfold_data(i, datasets, k=2):
    fold_size = len(datasets) // k
    val_start = i * fold_size
    if i == 0:
        return datasets[fold_size:], datasets[:fold_size]
    elif i == k - 1:
        return datasets[:val_start], datasets[val_start:]
    return datasets[:val_start] + datasets[val_start + fold_size:], datasets[val_start:val_start + fold_size]

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

if __name__ == "__main__":
    SEED = 1234
    random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    hp = hyperparameter()
    DATASET = "DrugBank"
    current_dir_path = os.path.dirname(os.path.abspath(__file__))
    dir_input = os.path.join(current_dir_path, "data", f"{DATASET}.txt")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print("Loading data")
    with open(dir_input, "r") as f:
        train_data_list = f.read().strip().split('\n')
    print("Data loaded and shuffled")
    dataset = shuffle_dataset(train_data_list, SEED)
    K_Fold = 2

    Accuracy_List, AUC_List, AUPR_List, Recall_List, Precision_List = [], [], [], [], []

    for i_fold in range(K_Fold):
        print(f'********** Fold {i_fold + 1}/{K_Fold} **********')
        train_dataset, test_dataset = get_kfold_data(i_fold, dataset)
        TVdataset = CustomDataSet(train_dataset)
        test_dataset = CustomDataSet(test_dataset)

        train_size = int(0.8 * len(TVdataset))
        valid_size = len(TVdataset) - train_size
        train_dataset, valid_dataset = torch.utils.data.random_split(TVdataset, [train_size, valid_size])

        train_loader = DataLoader(train_dataset, batch_size=hp.Batch_size, shuffle=True, collate_fn=collate_fn)
        valid_loader = DataLoader(valid_dataset, batch_size=hp.Batch_size, shuffle=False, collate_fn=collate_fn)
        test_loader = DataLoader(test_dataset, batch_size=hp.Batch_size, shuffle=False, collate_fn=collate_fn)

        model = AttentionDTI(hp).to(device)
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

        optimizer = optim.AdamW(model.parameters(), lr=hp.Learning_rate, weight_decay=hp.weight_decay)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer, base_lr=hp.Learning_rate, max_lr=hp.Learning_rate * 10, cycle_momentum=False,
                                                step_size_up=train_size // hp.Batch_size)
        Loss = nn.CrossEntropyLoss()
        save_path = os.path.join(current_dir_path, DATASET, str(i_fold))
        writer = SummaryWriter(log_dir=save_path)

        os.makedirs(save_path, exist_ok=True)
        early_stopping = EarlyStopping(save_path, patience=hp.Patience)

        for epoch in range(1, hp.Epoch + 1):
            model.train()
            train_losses = []
            for _, data in tqdm(enumerate(BackgroundGenerator(train_loader)), total=len(train_loader)):
                compounds, proteins, labels = [x.to(device) for x in data]
                optimizer.zero_grad()
                predictions = model(compounds, proteins)
                loss = Loss(predictions, labels)
                loss.backward()
                optimizer.step()
                scheduler.step()
                train_losses.append(loss.item())

            valid_results = test_process(model, tqdm(enumerate(BackgroundGenerator(valid_loader)), total=len(valid_loader)), Loss)
            writer.add_scalar('Train Loss', np.mean(train_losses), epoch)
            writer.add_scalar('Valid Loss', valid_results["loss"], epoch)

            early_stopping(valid_results["loss"], model, epoch)

            if early_stopping.early_stop:
                break

        test_results = test_model(model, test_loader, save_path, DATASET, Loss, dataset="Test", label="stable")[1]
        Accuracy_List.append(test_results["Accuracy"])
        Precision_List.append(test_results["Precision"])
        Recall_List.append(test_results["Recall"])
        AUC_List.append(test_results["AUC"])
        AUPR_List.append(test_results["PRC"])

    show_result(DATASET, "stable", Accuracy_List, Precision_List, Recall_List, AUC_List, AUPR_List)
