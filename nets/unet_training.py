# coding=utf-8
import math
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

def computer_precision_recall_f1_miou_acc(pred, gt):
    temp_gt = gt
    temp_pred = pred

    temp_pred = torch.max(temp_pred, 1)[1].to('cuda')

    tp = torch.logical_and(temp_pred == 1, temp_gt == 1)
    fp = torch.logical_and(temp_pred == 1, temp_gt == 0)
    fn = torch.logical_and(temp_pred == 0, temp_gt == 1)
    tn = torch.logical_and(temp_pred == 0, temp_gt == 0)
    
    union_0 = torch.logical_or(temp_pred == 0, temp_gt == 0).sum()
    union_1 = torch.logical_or(temp_pred == 1, temp_gt == 1).sum()

    flag = 1

    if (tp.sum()+fp.sum()+fn.sum()).item() == 0:
        
        flag = 0
        precison = torch.tensor(0.0).to('cuda')
        recall = torch.tensor(0.0).to('cuda')
        f1 = torch.tensor(0.0).to('cuda')
        
    else:
        
        f1 = (2*(tp.sum()))/(fn.sum()+2*tp.sum()+fp.sum())
        
        if (tp.sum()+fp.sum()).item()==0:
            
            precison = torch.tensor(0.0).to('cuda')
        else:
            precison = tp.sum()/(tp.sum()+fp.sum())
    
        if (tp.sum()+fn.sum()).item() == 0:
            
            recall = torch.tensor(0.0).to('cuda')
            
        else:
            
            recall = tp.sum()/(tp.sum()+fn.sum())

    if union_0 == 0:
        iou_0 = 0
    else:
        iou_0 = tn.sum().float() / union_0.float()
    if union_1 == 0:
        iou_1 = 0
    else:
        iou_1 = tp.sum().float() / union_1.float()
    mIoU = (iou_0 + iou_1) / 2
    
    total_pixels = temp_gt.numel()
    correct_pixels = (temp_pred == temp_gt).sum()
    acc = (correct_pixels.float() / total_pixels)
    
    return precison,recall,f1,flag,mIoU,acc

def TP_FP_Loss(pred, gt,flag_train):

    temp_gt = gt
    temp_pred = pred

    temp_pred = torch.max(temp_pred, 1)[1].to('cuda')

    tp = torch.logical_and(temp_pred == 1, temp_gt == 1)
    fp = torch.logical_and(temp_pred == 1, temp_gt == 0)
    fn = torch.logical_and(temp_pred == 0, temp_gt == 1)

    
    if (tp.sum()+fn.sum()).item()==0:
        if flag_train:
            fn_loss = torch.tensor(0.0, requires_grad=True, device='cuda')
        else:
            fn_loss = torch.tensor(0.0, requires_grad=False, device='cuda')
    else:
        if flag_train:
            fn_loss = torch.tensor((fn.sum()/(tp.sum()+fn.sum())).item(), requires_grad=True, device='cuda')
        else:
            fn_loss = torch.tensor((fn.sum()/(tp.sum()+fn.sum())).item(), requires_grad=False, device='cuda')

    if (tp.sum()+fp.sum()).item()==0:
        if flag_train:
            fp_loss = torch.tensor(0.0, requires_grad=True, device='cuda')
        else:
            fp_loss = torch.tensor(0.0, requires_grad=False, device='cuda')
    else:
        if flag_train:
            fp_loss = torch.tensor((fp.sum() / (tp.sum() + fp.sum())).item(), requires_grad=True, device='cuda')
        else:
            fp_loss = torch.tensor((fp.sum() / (tp.sum() + fp.sum())).item(), requires_grad=False, device='cuda')

    loss = fn_loss + fp_loss
    
    if flag_train:
        loss.requires_grad_(True)

    return loss


def CE_Loss(inputs, target, num_classes=2):
    n, c, h, w = inputs.size()
    nt, ht, wt = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    temp_target = target.view(-1)

    CE_loss  = nn.CrossEntropyLoss(ignore_index=num_classes)(temp_inputs, temp_target)
    return CE_loss

def Focal_Loss(inputs, target, num_classes=2, alpha=0.5, gamma=2):
    n, c, h, w = inputs.size()

    temp_inputs = inputs.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)

    temp_target = target.view(-1)

    logpt  = -nn.CrossEntropyLoss(ignore_index=num_classes, reduction='none')(temp_inputs, temp_target)
    pt = torch.exp(logpt)
    if alpha is not None:
        logpt *= alpha
    loss = -((1 - pt) ** gamma) * logpt
    loss = loss.mean()
    return loss

def Dice_loss(inputs, target, beta=1, smooth=1e-5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)

    # --------------------------------------------#
    #   计算dice loss
    # --------------------------------------------#
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    dice_loss = 1 - torch.mean(score)
    return dice_loss

def weights_init(net, init_type='normal', init_gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and classname.find('Conv') != -1:
            if init_type == 'normal':
                torch.nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                torch.nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                torch.nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                torch.nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
        elif classname.find('BatchNorm2d') != -1:
            torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
            torch.nn.init.constant_(m.bias.data, 0.0)
    print('initialize network with %s type' % init_type)
    net.apply(init_func)