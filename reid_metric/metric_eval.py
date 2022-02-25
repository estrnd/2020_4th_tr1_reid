import pandas as pd
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score
from tqdm import tqdm
import numpy as np
import torch

def calc_acc_(df, th):
    pred_pos = th < df['sim_pos']
    pred_neg = df['sim_neg'] < th 
    pred = pred_pos & pred_neg
    acc = sum(pred)/len(pred)
    return acc

def calc_acc(df):
    accs = [(calc_acc_(df, th), th) for th in np.arange(-1, 1, 0.005)]
    best_acc, best_th = sorted(accs, reverse=True)[0]
    return best_acc


def calc_f1(total_outs, total_labels, th = 0.7):
    t1 = total_outs
    t2 = total_outs.transpose(0,1)
    sims = torch.matmul(t1, t2)
    
    def calc_f1_(idx, th):
        pivot_out = total_outs[idx]
        pivot_label = total_labels[idx]
    
        sims_i = sims[idx].clone()
        pred_i = (sims_i > th)*1
        gt   = [1 if e == pivot_label else 0 for e in total_labels]
        return f1_score(gt, pred_i), precision_score(gt, pred_i), recall_score(gt, pred_i), accuracy_score(gt, pred_i)
        return f1_score(gt, pred_i)
    
    f1s = [calc_f1_(idx,th) for idx in tqdm(range(len(total_outs)))] 
    return np.mean(f1s)#, f1s
    


