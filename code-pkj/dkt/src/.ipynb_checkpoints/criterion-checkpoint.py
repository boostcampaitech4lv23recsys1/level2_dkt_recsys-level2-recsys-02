import torch.nn as nn
import torch

def get_criterion(pred, target):
    loss = nn.BCEWithLogitsLoss(reduction="none")
    #loss = nn.BCEWithLogitsLoss(reduction="none", 
    #                            pos_weight=torch.tensor((len(target-sum(target)))/sum(target)))
    return loss(pred, target)
