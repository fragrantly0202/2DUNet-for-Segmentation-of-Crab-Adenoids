import os

import matplotlib
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
from matplotlib import pyplot as plt
import scipy.signal

import cv2
import shutil
import numpy as np

from PIL import Image
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from .utils import cvtColor, preprocess_input, resize_image
from .utils_metrics import compute_mIoU


class LossHistory():
    def __init__(self, log_dir, model, input_shape, val_loss_flag=True):
        self.log_dir        = log_dir
        self.val_loss_flag  = val_loss_flag

        self.losses         = []
        if self.val_loss_flag:
            self.val_loss   = []
        
        os.makedirs(self.log_dir)
        self.writer     = SummaryWriter(self.log_dir)
        try:
            dummy_input     = torch.randn(2, 3, input_shape[0], input_shape[1])
            self.writer.add_graph(model, dummy_input)
        except:
            pass

    def append_loss(self, epoch, loss, val_loss = None):
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        self.losses.append(loss)
        if self.val_loss_flag:
            self.val_loss.append(val_loss)
        
        with open(os.path.join(self.log_dir, "epoch_loss.txt"), 'a') as f:
            f.write(str(loss))
            f.write("\n")
        if self.val_loss_flag:
            with open(os.path.join(self.log_dir, "epoch_val_loss.txt"), 'a') as f:
                f.write(str(val_loss))
                f.write("\n")
            
        self.writer.add_scalar('loss', loss, epoch)
        if self.val_loss_flag:
            self.writer.add_scalar('val_loss', val_loss, epoch)
            
        self.loss_plot()

    def loss_plot(self):
        iters = range(len(self.losses))

        plt.figure()
        plt.plot(iters, self.losses, 'red', linewidth = 2, label='train loss')
        if self.val_loss_flag:
            plt.plot(iters, self.val_loss, 'coral', linewidth = 2, label='val loss')
            
        try:
            if len(self.losses) < 25:
                num = 5
            else:
                num = 15
            
            plt.plot(iters, scipy.signal.savgol_filter(self.losses, num, 3), 'green', linestyle = '--', linewidth = 2, label='smooth train loss')
            if self.val_loss_flag:
                plt.plot(iters, scipy.signal.savgol_filter(self.val_loss, num, 3), '#8B4513', linestyle = '--', linewidth = 2, label='smooth val loss')
        except:
            pass

        plt.grid(True)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend(loc="upper right")

        plt.savefig(os.path.join(self.log_dir, "epoch_loss.png"))

        plt.cla()
        plt.close("all")

class EvalCallback():
    def __init__(self, net, input_shape, num_classes, image_ids, dataset_path, log_dir, cuda, \
            f1_out_path=".temp_f1_out", miou_out_path=".temp_miou_out", acc_out_path=".temp_acc_out", eval_flag=True, period=1):
        super(EvalCallback, self).__init__()
        
        self.net                = net
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.image_ids          = image_ids
        self.dataset_path       = dataset_path
        self.log_dir            = log_dir
        self.cuda               = cuda
        self.f1_out_path        = f1_out_path
        self.miou_out_path      = miou_out_path
        self.acc_out_path       = acc_out_path
        self.eval_flag          = eval_flag
        self.period             = period
        
        self.image_ids          = [image_id.split()[0] for image_id in image_ids]
        self.f1s      = [0]
        self.mious      = [0]
        self.accs      = [0]
        self.epoches_f1s    = [0]
        self.epoches_mious    = [0]
        self.epoches_accs    = [0]
        if self.eval_flag:
            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(0))
                f.write("\n")
    
    def on_epoch_end(self, epoch, model_eval, f1, miou, acc):
        if epoch % self.period == 0 and self.eval_flag:
            self.net    = model_eval
            # gt_dir      = os.path.join(self.dataset_path, "mask_5std_filter_crop800/")
            pred_dir    = os.path.join(self.miou_out_path, 'detection-results')
            if not os.path.exists(self.miou_out_path):
                os.makedirs(self.miou_out_path)
            if not os.path.exists(self.f1_out_path):
                os.makedirs(self.f1_out_path)
            if not os.path.exists(self.acc_out_path):
                os.makedirs(self.acc_out_path)
            
            if not os.path.exists(pred_dir):
                os.makedirs(pred_dir)
            print("Get miou.")
                        
            print("Calculate f1,miou,acc.")
            IoUs = miou
            f1_score = f1
            accuracy = acc
            
            # plot f1
            temp_f1 = np.nanmean(f1_score) * 100

            self.f1s.append(temp_f1)
            self.epoches_f1s.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_f1.txt"), 'a') as f:
                f.write(str(temp_f1))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches_f1s, self.f1s, 'red', linewidth = 2, label='val f1')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('F1')
            plt.title('A F1 Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_f1.png"))
            plt.cla()
            plt.close("all")
            
            #plot miou
            temp_miou = np.nanmean(IoUs) * 100

            self.mious.append(temp_miou)
            self.epoches_mious.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_miou.txt"), 'a') as f:
                f.write(str(temp_miou))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches_mious, self.mious, 'red', linewidth = 2, label='val miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Miou')
            plt.title('A Miou Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_miou.png"))
            plt.cla()
            plt.close("all")
            
            #plot acc
            temp_acc = np.nanmean(accuracy) * 100

            self.accs.append(temp_acc)
            self.epoches_accs.append(epoch)

            with open(os.path.join(self.log_dir, "epoch_acc.txt"), 'a') as f:
                f.write(str(temp_acc))
                f.write("\n")
            
            plt.figure()
            plt.plot(self.epoches_accs, self.accs, 'red', linewidth = 2, label='val miou')

            plt.grid(True)
            plt.xlabel('Epoch')
            plt.ylabel('Acc')
            plt.title('A Acc Curve')
            plt.legend(loc="upper right")

            plt.savefig(os.path.join(self.log_dir, "epoch_acc.png"))
            plt.cla()
            plt.close("all")

            print("Get f1,miou,acc done.")
            
            
            
            shutil.rmtree(self.miou_out_path)
