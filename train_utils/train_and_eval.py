import numpy as np
import torch
from torch import nn
import train_utils.distributed_utils as utils
from .dice_coefficient_loss import dice_loss, build_target
from typing import Iterable, Set, Tuple
from torch import Tensor
from scipy.ndimage import distance_transform_edt as distance


def criterion(device, inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    for name, x in inputs.items():
        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        if dice is True:
            dice_target = build_target(target, num_classes, ignore_index)
            loss = loss + dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        new_gt_list, logits_list, ln_dis_list = RBS_dis(x, target)
        wm_iou = weighted_miou(new_gt_list, logits_list, ln_dis_list, device)
        wm_iou = 1 - wm_iou
        wm_iou = torch.sqrt(wm_iou)
        loss = loss + 1 * wm_iou

        losses[name] = loss
    if len(losses) == 1:
        return losses['out'], wm_iou
    return losses['out'] + 0.5 * losses['aux'], wm_iou

def evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)
    wm_iou_values = []

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Val:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)

            output = model(image)
            output = output['out']

            confmat.update(target.flatten(), output.argmax(1).flatten())
            dice.update(output, target)

            new_gt_list, logits_list, ln_dis_list = RBS_dis(output, target)
            wm_iou = weighted_miou(new_gt_list, logits_list, ln_dis_list, device)
            wm_iou_values.append(wm_iou.item())  # 存储当前迭代步骤的 wm_iou 值

        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    return confmat, dice.value.item(), wm_iou_values

def pre_evaluate(model, data_loader, device, num_classes):
    model.eval()
    confmat = utils.ConfusionMatrix(num_classes)
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Predict:'
    with torch.no_grad():
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)
            output = output['out']
            confmat.update(target.flatten(), output.argmax(1).flatten())

        confmat.reduce_from_all_processes()
    return confmat


def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        with torch.cuda.amp.autocast(enabled=scaler is not None):
            output = model(image)
            loss, wm_iou = criterion(device, output, target, loss_weight, num_classes=num_classes, ignore_index=255)

        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        lr_scheduler.step()
        lr = optimizer.param_groups[0]["lr"]
        metric_logger.update(loss=loss.item(), lr=lr, wm_iou=wm_iou.item())
    return metric_logger.meters["loss"].global_avg, lr, metric_logger.meters["wm_iou"].global_avg


def create_lr_scheduler(optimizer,num_step: int,epochs: int,warmup=True,warmup_epochs=1,warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)


def get_non_padding_regions(images):
    b, h, w = images.shape
    regions = []
    for i in range(b):
        img = images[i]

        coords_0 = np.argwhere(img == 0)
        coords_1 = np.argwhere(img == 1)

        if coords_0.size == 0:
            min_x_0, min_y_0, max_x_0, max_y_0 = 0, 0, 0, 0
        else:
            min_x_0, min_y_0 = np.min(coords_0, axis=0)
            max_x_0, max_y_0 = np.max(coords_0, axis=0)
        if coords_1.size == 0:
            min_x_1, min_y_1, max_x_1, max_y_1 = 0, 0, 0, 0
        else:
            min_x_1, min_y_1 = np.min(coords_1, axis=0)
            max_x_1, max_y_1 = np.max(coords_1, axis=0)

        min_x = min(min_x_0, min_x_1)
        min_y = min(min_y_0, min_y_1)
        max_x = max(max_x_0, max_x_1)
        max_y = max(max_y_0, max_y_1)
        regions.append((min_x, min_y, max_x, max_y))

    return regions


def RBS_dis(x, target):
    regions = get_non_padding_regions(target.cpu().numpy())

    ln_dis_list = []
    logits_list = []
    new_gt_list = []
    i = 0
    for region in regions:

        min_x, min_y, max_x, max_y = region
        new_gt = target[i][min_x:max_x + 1, min_y:max_y + 1].clone()
        new_gt = new_gt.unsqueeze(0)
        new_gt_list.append(new_gt)

        data2 = class2one_hot_ori(new_gt, 2)
        data2 = data2.cpu().numpy()
        dis_posmask, dis_negmask = one_hot2dist(data2)
        dis1 = dis_posmask + dis_negmask

        d = np.mean(dis1)
        dis_ = np.where(dis1 > d, d, dis1)
        dis2 = (d + np.e - 1) - dis_
        ln_dis = np.log(dis2)
        ln_dis = torch.from_numpy(ln_dis).cuda().float().requires_grad_()
        ln_dis_list.append(ln_dis)

        logi = x[i]
        logit = torch.sigmoid(logi)
        logits = torch.argmax(logit, dim=0)
        logits_values = logits[min_x:max_x + 1, min_y:max_y + 1].clone()
        logits_list.append(logits_values.unsqueeze(0))
        i = i+1

    return new_gt_list, logits_list, ln_dis_list

def weighted_miou(new_gt_list, logits_list, ln_dis_list, device):
    class_num = 2
    batch_size = len(new_gt_list)
    confusion_matrix = torch.zeros(batch_size, class_num, class_num, device=device, dtype=torch.float32)
    for i in range(class_num):
        for j in range(class_num):
            for k in range(batch_size):
                confusion_matrix[k, i, j] = torch.sum(ln_dis_list[k] *
                            (logits_list[k] == i) * (new_gt_list[k] == j), dim=(1, 2))
    confusion_matrix_total = confusion_matrix.sum(0)
    true_positive = torch.diag(confusion_matrix_total)
    mother = (confusion_matrix_total.sum(1) + confusion_matrix_total.sum(0) - torch.diag(confusion_matrix_total))
    iou_per_class = true_positive / mother
    wm_iou = torch.mean(iou_per_class)
    return wm_iou

def class2one_hot_ori(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    if len(seg.shape) == 3:  # Only w, h, d, used by the dataloader
        b, w, h = seg.shape  # type: Tuple[int, int, int]
        res = torch.stack([seg == c for c in range(K)], dim=1).type(torch.int32)
        assert res.shape == (b, K, w, h)
        assert one_hot(res)
        return res
    assert sset(seg, list(range(K))), (uniq(seg), K)
    b, *img_shape = seg.shape  # type: Tuple[int, ...]
    device = seg.device
    res = torch.zeros((b, K, *img_shape), dtype=torch.int32, device=device).scatter_(1, seg[:, None, ...], 1)
    assert res.shape == (b, K, *img_shape)
    assert one_hot(res)
    return res

def one_hot2dist(seg: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    assert one_hot(torch.tensor(seg))
    b, num_classes, h, w = seg.shape
    dis_posmask = np.zeros((b, h, w), dtype=float)
    dis_negmask = np.zeros((b, h, w), dtype=float)
    for i in range(b):
        posmask = seg[i, 0].astype(np.bool)
        dis_posmask[i] = distance(posmask)
        negmask = ~posmask
        dis_negmask[i] = distance(negmask)
    return dis_posmask, dis_negmask

def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)

def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])

def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)
