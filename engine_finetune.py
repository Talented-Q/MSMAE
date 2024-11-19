# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# BEiT: https://github.com/microsoft/unilm/tree/master/beit
# MAE: https://github.com/facebookresearch/mae
# --------------------------------------------------------

import math
import sys
from typing import Iterable, Optional

import torch

from timm.data import Mixup
from timm.utils import accuracy

import util.misc as misc
import util.lr_sched as lr_sched
from sklearn.metrics import f1_score

import torch.nn.functional as F
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import os

def train_one_epoch(model: torch.nn.Module, criterion: torch.nn.Module,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                    mixup_fn: Optional[Mixup] = None, log_writer=None,
                    args=None):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 20

    accum_iter = args.accum_iter

    optimizer.zero_grad()

    if log_writer is not None:
        print('log_dir: {}'.format(log_writer.log_dir))


    for data_iter_step, (samples, targets, mask_weights) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        # we use a per iteration (instead of per epoch) lr scheduler
        if data_iter_step % accum_iter == 0:
            lr_sched.adjust_learning_rate(optimizer, data_iter_step / len(data_loader) + epoch, args)

        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        mask_weights = mask_weights.to(device, non_blocking=True)

        if mixup_fn is not None:
            samples, targets = mixup_fn(samples, targets)

        with torch.cuda.amp.autocast():
            outputs = model(samples, mask_weights, mask_ratio=args.mask_ratio, throw_ratio=args.throw_ratio)
            loss = criterion(outputs, targets)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        loss /= accum_iter
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=False,
                    update_grad=(data_iter_step + 1) % accum_iter == 0)
        if (data_iter_step + 1) % accum_iter == 0:
            optimizer.zero_grad()

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)

        loss_value_reduce = misc.all_reduce_mean(loss_value)
        if log_writer is not None and (data_iter_step + 1) % accum_iter == 0:
            """ We use epoch_1000x as the x-axis in tensorboard.
            This calibrates different curves when batch size changes.
            """
            epoch_1000x = int((data_iter_step / len(data_loader) + epoch) * 1000)
            log_writer.add_scalar('loss', loss_value_reduce, epoch_1000x)
            log_writer.add_scalar('lr', max_lr, epoch_1000x)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model.forward_test(images)
            loss = criterion(output, target)

        acc1, acc5 = accuracy(output, target, topk=(1, 2))

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, top5=metric_logger.acc5, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


#-------------ROC
@torch.no_grad()
def evaluate_with_roc(data_loader, model, device, output_dir=None, epoch=None):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    model.eval()

    y_true = []
    y_score = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            output = model.forward_test(images)
            loss = criterion(output, target)

        probabilities = torch.softmax(output, dim=1)

        y_true.extend(target.cpu().numpy())
        y_score.extend(probabilities.cpu().numpy())

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())

    y_true = np.array(y_true)
    y_score = np.array(y_score)
    num_classes = y_score.shape[1]

    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    plt.figure(figsize=(10, 8))
    for i in range(num_classes):
        fpr[i], tpr[i], _ = roc_curve(y_true, y_score[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])
        plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.2f})')

    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) - Epoch {epoch}' if epoch is not None else 'Receiver Operating Characteristic (ROC)')
    plt.legend(loc='lower right')

    if output_dir:
        # Define the file name based on the epoch
        roc_curve_path = os.path.join(output_dir, f'roc_curve_epoch_{epoch}.png' if epoch is not None else 'roc_curve.png')
        plt.savefig(roc_curve_path)
        print(f"ROC curve saved to {roc_curve_path}")

    plt.close()

    avg_auc = np.mean(list(roc_auc.values()))
    metric_logger.meters['auc'] = misc.SmoothedValue(window_size=1, fmt='{value:.3f}')
    metric_logger.meters['auc'].update(avg_auc)

    return {'roc_auc': avg_auc, 'roc_curve': roc_curve_path if output_dir else None}

@torch.no_grad()
def evaluate_1(data_loader, model, device):
    criterion = torch.nn.CrossEntropyLoss()

    metric_logger = misc.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()

    y_true = []
    y_pred = []

    for batch in metric_logger.log_every(data_loader, 10, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        # compute output
        with torch.cuda.amp.autocast():
            output = model.forward_test(images)
            loss = criterion(output, target)

        _, predicted = torch.max(output, 1)

        y_true.extend(target.cpu().numpy())
        y_pred.extend(predicted.cpu().numpy())

        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())

    f1 = f1_score(y_true, y_pred, average='macro')

    metric_logger.meters['f1_score'] = misc.SmoothedValue(window_size=1, fmt='{value:.3f}')
    metric_logger.meters['f1_score'].update(f1)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print('F1 Score {f1_score.global_avg:.3f}'.format(f1_score=metric_logger.f1_score))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
