# coding=utf-8
import os
import datetime
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from torch.utils.data import DataLoader
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss, TP_FP_Loss, computer_precision_recall_f1_miou_acc, weights_init
from nets.unet import Unet
from utils.dataloader import UnetDataset, unet_dataset_collate
from utils.utils import download_weights, show_config
from utils.callbacks import LossHistory, EvalCallback
from tqdm import tqdm

if __name__ == "__main__":

    Cuda = True

    num_classes = 2  # your class numbers

    pretrained = False

    model_path = ''  # pretrained model path

    backbone = "resnet50"  # choose backbone : resnet50 or vgg

    Epoch = 100

    batch_size = 4

    lr = 1e-4

    optimizer_type = "adam"

    weight_decay = 0

    save_period = 1  # the period of save training model path

    save_dir = 'logs'  # the path of save model

    input_shape = [800, 800]  # your input shape

    dataset_path = ''  # your dataset path

    num_workers = 4


    eval_flag = True # evaluate the cycle of the model
    eval_period = 1

    model = Unet(num_classes=num_classes, pretrained=pretrained, backbone=backbone).to(device)

    model.unfreeze_backbone()
    model = torch.nn.DataParallel(model)  # multi-GPU

    if not pretrained:
        weights_init(model)

    if model_path != '':

        print('Load weights {}.'.format(model_path))

        model_dict = model.state_dict()
        pretrained_dict = torch.load(model_path)
        load_key, no_load_key, temp_dict = [], [], {}

        for k, v in pretrained_dict.items():
            if k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v):
                temp_dict[k] = v
                load_key.append(k)
            else:
                no_load_key.append(k)
        model_dict.update(temp_dict)
        model.load_state_dict(model_dict)
        # ------------------------------------------------------#
        #   显示没有匹配上的Key
        # ------------------------------------------------------#
        print("\nSuccessful Load Key:", str(load_key)[:500], "……\nSuccessful Load Key Num:", len(load_key))
        print("\nFail To Load Key:", str(no_load_key)[:500], "……\nFail To Load Key num:", len(no_load_key))
        print("\n\033[1;33;44m温馨提示，head部分没有载入是正常现象，Backbone部分没有载入是错误的。\033[0m")

    time_str = datetime.datetime.strftime(datetime.datetime.now(), '%Y_%m_%d_%H_%M_%S')
    log_dir = os.path.join(save_dir, "loss_" + str(time_str))
    loss_history = LossHistory(log_dir, model, input_shape=input_shape)

    with open(os.path.join(VOCdevkit_path, "2d_txt/train.txt"), "r") as f:
        train_lines = f.readlines()
    with open(os.path.join(VOCdevkit_path, "2d_txt/val.txt"), "r") as f:
        val_lines = f.readlines()
    num_train = len(train_lines)
    num_val = len(val_lines)

    show_config(
        num_classes=num_classes, backbone=backbone, input_shape=input_shape,
        Epoch=Epoch, batch_size=batch_size, optimizer_type=optimizer_type,
        save_period=save_period, save_dir=save_dir, num_workers=num_workers, num_train=num_train, num_val=num_val
    )

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)


    epoch_step = num_train // batch_size
    epoch_step_val = num_val // batch_size

    train_dataset = UnetDataset(train_lines, input_shape, num_classes, VOCdevkit_path)
    val_dataset = UnetDataset(val_lines, input_shape, num_classes, VOCdevkit_path)

    shuffle = True

    train_dl = DataLoader(train_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                          drop_last=True, collate_fn=unet_dataset_collate)
    val_dl = DataLoader(val_dataset, shuffle=shuffle, batch_size=batch_size, num_workers=num_workers,
                        drop_last=True, collate_fn=unet_dataset_collate)

    eval_callback = EvalCallback(model, input_shape, num_classes, val_lines, VOCdevkit_path, log_dir, Cuda, \
                                 eval_flag=eval_flag, period=eval_period) # record the eval map curve


    best = [0, 0]  # initializes the epoch and performance of the optimal model
    for epoch in range(0, Epoch):

        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.5)

        model.train()

        train_loss = 0.0
        train_Precision = 0.0
        train_Recall = 0.0
        train_f1 = 0.0
        train_miou = 0.0
        train_acc = 0.0

        val_loss = 0.0
        val_Precision = 0.0
        val_Recall = 0.0
        val_f1 = 0.0
        val_miou = 0.0
        val_acc = 0.0

        train_effect_cnt = 0

        val_effect_cnt = 0

        flag_train = True

        for iteration, batch in enumerate(train_dl):

            imgs, masks, labels = batch
            with torch.no_grad():
                if Cuda:
                    imgs = imgs.to(device)
                    masks = masks.to(device)
                    labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(imgs)

            loss = TP_FP_Loss(outputs, masks, flag_train) + Focal_Loss(outputs, masks)

            with torch.no_grad():
                _Precision, _Recall, _f1, flag, _miou, _acc = computer_precision_recall_f1_miou_acc(outputs, masks)
                if flag:
                    train_effect_cnt += 1

            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_Precision += _Precision.item()
            train_Recall += _Recall.item()
            train_f1 += _f1.item()
            train_miou += _miou.item()
            train_acc += _acc.item()

            pbar.set_postfix(**{'train_loss': train_loss / (iteration + 1),
                                'Precision': train_Precision / (train_effect_cnt + 1),
                                'Recall': train_Recall / (train_effect_cnt + 1),
                                'f1': train_f1 / (train_effect_cnt + 1),
                                'mIoU': train_miou / (iteration + 1),
                                'acc': train_acc / (iteration + 1),
                                })
            pbar.update(1)

        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.5)

        model.eval()
        flag_train = False
        for iteration, batch in enumerate(val_dl):
            if iteration >= epoch_step_val:
                break
            imgs, masks, labels = batch

            with torch.no_grad():
                if Cuda:
                    imgs = imgs.to(device)
                    masks = masks.to(device)
                    labels = labels.to(device)

                outputs = model(imgs)

                loss = TP_FP_Loss(outputs, masks, flag_train) + Focal_Loss(outputs, masks)

                _Precision, _Recall, _f1, flag, _miou, _acc = computer_precision_recall_f1_miou_acc(outputs, masks)

                if flag:
                    val_effect_cnt += 1

                val_loss += loss.item()
                val_Precision += _Precision.item()
                val_Recall += _Recall.item()
                val_f1 += _f1.item()
                val_miou += _miou.item()
                val_acc += _acc.item()

            pbar.set_postfix(**{'val_loss': val_loss / (iteration + 1),
                                'Precision': val_Precision / (val_effect_cnt + 1),
                                'Recall': val_Recall / (val_effect_cnt + 1),
                                'f1': val_f1 / (val_effect_cnt + 1),
                                'mIoU': val_miou / (iteration + 1),
                                'acc': val_acc / (iteration + 1),
                                })
            pbar.update(1)

        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, train_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model, val_f1 / (val_effect_cnt + 1), val_miou / (iteration + 1),
                                   val_acc / (iteration + 1))

        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Train Loss: %.3f || Val Loss: %.3f ' % (train_loss / epoch_step, val_loss / epoch_step_val))

        state = {'net': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
        torch.save(state, os.path.join(save_dir, 'latest_model.pth'))
        if val_f1 / (val_effect_cnt + 1) > best[1]:
            print('Saving best model')
            torch.save(state, os.path.join(save_dir, 'best_model.pth'))
            best[0] = epoch
            best[1] = val_f1 / (val_effect_cnt + 1)
        print('Best performance at Epoch: {} | {}'.format(best[0], best[1]))

    loss_history.writer.close()

# CUDA_VISIBLE_DEVICES=0,1 python3 train_crab_easy.py


