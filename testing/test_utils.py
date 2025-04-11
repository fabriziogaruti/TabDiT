import os
from os.path import join
import numpy as np
import torch
from loguru import logger
from tqdm import tqdm
import json
import pandas as pd
from datetime import datetime
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import precision_recall_curve, average_precision_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path



def binary_results_from_predictions(preds, labels):
    confusion_mat = confusion_matrix(labels, preds)
    tp, fp = int(confusion_mat[1,1]), int(confusion_mat[0,1])
    tn, fn = int(confusion_mat[0,0]), int(confusion_mat[1,0])
    accuracy = accuracy_score(labels, preds)
    accuracy_mean = balanced_accuracy_score(labels, preds)
    f1_class1 = f1_score(labels, preds, average='binary')
    f1_macro = f1_score(labels, preds, average='macro')
    f1_weighted = f1_score(labels, preds, average='weighted')
    precision = precision_score(labels, preds)
    recall = recall_score(labels, preds)
    return {
        'accuracy': accuracy,
        'accuracy_mean': accuracy_mean,
        'confusion_matrix': {'tp':tp, 'fp': fp, 'fn': fn, 'tn': tn},
        'f1_target': f1_class1,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'precision': precision,
        'recall': recall,
    }



def print_statistics_binary(probs, labels, output_dir, split="", log=False):
    complete_labels = labels.copy()
    labels = labels[(complete_labels==0) | (complete_labels==1)]
    probs = probs[(complete_labels==0) | (complete_labels==1)]

    ### FIRST COMPUTE RESULTS NOT DEPENDENT FROM THRESHOLD 
    ### THEN TRY DIFFERENT THRESHOLDING POLICIES
    total_true = int(np.sum(labels==1))
    total_false = int(np.sum(labels==0))
    total_doubt = int(np.sum((complete_labels!=0) & (complete_labels!=1)))

    # COMPUTE ROC Area-Under-Curve
    roc_auc = roc_auc_score(labels, probs)
    # COMPUTE Precision-Recall Area-Under-Curve
    PR_auc_classes = average_precision_score(labels, probs, average=None)
    PR_auc_macro = average_precision_score(labels, probs, average='macro')

    # Consider banal threshold of 0.5
    preds_05 = np.array(probs>0.5).astype(np.int32)
    results_05 = binary_results_from_predictions(preds_05, labels)

    # estimate threshold from precision recall curve maximizing F1-Score
    precision_list, recall_list, thresholds_list = precision_recall_curve(labels, probs)
    assert min(thresholds_list)>=0 and max(thresholds_list)<=1, f'the thresholds are not in [0,1] but in [{min(thresholds_list)},{max(thresholds_list)}], \n->{thresholds_list}'
    f1_list = (2*precision_list*recall_list)/(precision_list+recall_list)
    index1 = np.nanargmax(f1_list)
    threshold_f1 = thresholds_list[index1]
    preds_f1 = np.array(probs>threshold_f1).astype(np.int32)
    results_f1 = binary_results_from_predictions(preds_f1, labels)

    # estimate threshold from roc curve maximizing G-mean
    fpr, tpr, thresholds_roc = roc_curve(labels, probs)
    gmeans = np.sqrt(tpr * (1-fpr))
    index2 = np.nanargmax(gmeans)
    threshold_roc = thresholds_roc[index2]
    preds_roc = np.array(probs>threshold_roc).astype(np.int32)
    results_roc = binary_results_from_predictions(preds_roc, labels)

    # PLOT PRECISION RECALL CURVE WITH AVERAGE PRECISION AND F1 SCORE
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig_img_rocauc = ax
    fig_img_rocauc.plot(recall_list, precision_list, label='precision recall curve')
    fig_img_rocauc.plot(recall_list, f1_list, label='precision recall f1 score')
    fig_img_rocauc.scatter(recall_list[index1], precision_list[index1], c='r', label='optimal threshold')
    fig_img_rocauc.title.set_text('Image precision recall, AP-score: {0:.2f} F1-score: {1:.2f}%, Optimal Threshold: {2:.2f}'.format(PR_auc_macro, f1_list[index1], threshold_f1))
    fig_img_rocauc.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'prec-rec_curve{str(split)}.png'), dpi=100)

    # PLOT ROC CURVE WITH ROC-AUC AND G-MEAN SCORE
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    fig_img_rocauc = ax
    fig_img_rocauc.plot(fpr, tpr, label='img_ROCAUC:{0:.3f}'.format(roc_auc))
    fig_img_rocauc.scatter(fpr[index2], tpr[index2], c='r', label='optimal threshold')
    fig_img_rocauc.title.set_text('Image ROCAUC:{0:.3f}, G-MEAN: {1:.1f}%, Optimal Threshold: {2:.2f}'.format(roc_auc, gmeans[index2], threshold_roc))
    fig_img_rocauc.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'roc_curve{str(split)}.png'), dpi=100)

    stats_dict = {
        'total_true_samples':total_true,
        'total_false_sample':total_false,
        'total_doubt_samples':total_doubt,
        'AUC': float(roc_auc),
        'Average-Precisions': float(PR_auc_classes),
        'Average_precision_macro': float(PR_auc_macro),
        'Results thr=0,5': results_05,
        'Optimal thr F1': float(threshold_f1),
        'Results optimal F1': results_f1,
        'Optimal thr Gmean': float(threshold_roc),
        'Results optimal Gmean': results_roc,
    }
    if log:
        logger.info(f'COMPUTED RESULTS OF TESTING : \n{json.dumps(stats_dict, indent=4)}')
    with open(join(output_dir, f'main_statistics{str(split)}.json'), 'w+') as fw:
        json.dump(stats_dict, fw, indent=4)


