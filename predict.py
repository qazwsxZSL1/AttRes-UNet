import os
import time
import torch
import cv2
from torchvision import transforms
import numpy as np
from PIL import Image
from src import AttResUNet
import sys
import re

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from torch import Tensor
from torch import einsum
from scipy.ndimage import distance_transform_edt as distance
from typing import Any, Callable, Iterable, List, Set, Tuple, TypeVar, Union, cast
from torchvision import transforms as T
smooth = 100

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

def weigth_dis(x, target):
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
        mean_value = 8.0
        dis_ = np.where(dis1 > mean_value, mean_value, dis1)
        dis2 = (mean_value + np.e - 1) - dis_
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

def weighted_iou(new_gt_list, logits_list, ln_dis_list, device):
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

def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))
    b, w, h = seg.shape  # type: Tuple[int, int, int]
    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res
def class2one_hot_ori(seg: Tensor, K: int) -> Tensor:
    # Breaking change but otherwise can't deal with both 2d and 3d
    if len(seg.shape) == 3:
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
        # posmask = seg[i, 0].astype(np.bool)
        posmask = seg[i, 0].astype(bool)
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
    # Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())

def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)

def load_images(img_path, file_names, mean, std):
    images = []
    for file in file_names:
        original_img = Image.open(os.path.join(img_path, file + ".jpg")).convert('RGB')
        h = np.array(original_img).shape[0]
        w = np.array(original_img).shape[1]
        original_img = original_img.convert("RGB")
        # from pil image to tensor and normalize
        data_transform = transforms.Compose([transforms.Resize(565),
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=mean, std=std)])
        img = data_transform(original_img)
        # expand batch dimension
        img = torch.unsqueeze(img, dim=0)
        images.append((file, img, h, w))

    return images

def dice_equation(mask1, mask2):
    intersection = (mask1 * mask2).sum()
    union = (mask1 + mask2).sum()
    if union != 0:
        dices = float((2 * intersection) / union)
    else:
        dices = 0
    return dices

def f_score(inputs, target, beta=1, smooth=1e-5, threhold=0.5):
    n, c, h, w = inputs.size()
    nt, ht, wt, ct = target.size()
    if h != ht and w != wt:
        inputs = F.interpolate(inputs, size=(ht, wt), mode="bilinear", align_corners=True)

    temp_inputs = torch.softmax(inputs.transpose(1, 2).transpose(2, 3).contiguous().view(n, -1, c), -1)
    temp_target = target.view(n, -1, ct)
    # dice
    temp_inputs = torch.gt(temp_inputs, threhold).float()
    tp = torch.sum(temp_target[..., :-1] * temp_inputs, axis=[0, 1])
    fp = torch.sum(temp_inputs, axis=[0, 1]) - tp
    fn = torch.sum(temp_target[..., :-1], axis=[0, 1]) - tp

    score = ((1 + beta ** 2) * tp + smooth) / ((1 + beta ** 2) * tp + beta ** 2 * fn + fp + smooth)
    score = torch.mean(score)
    return score

def fast_hist(a, b, n):
    k = (a >= 0) & (a < n)
    return np.bincount(n * a[k].astype(int) + b[k].astype(int), minlength=n ** 2).reshape(n, n)

def per_class_iu(hist):
    return np.diag(hist) / np.maximum((hist.sum(1) + hist.sum(0) - np.diag(hist)), 1)

def per_class_PA_Recall(hist):
    return np.diag(hist) / np.maximum(hist.sum(1), 1)

def per_class_Precision(hist):
    return np.diag(hist) / np.maximum(hist.sum(0), 1)

def per_Accuracy(hist):
    return np.sum(np.diag(hist)) / np.maximum(np.sum(hist), 1)

def compute_mIoU(metric_output, gt_dir, pred_dir, png_name_list, png_name_list_2, num_classes=2, name_classes=["_background_", "Tactile_paving"]):
    hist = np.zeros((num_classes, num_classes))
    gt_imgs = [join(gt_dir, x) for x in png_name_list]
    pred_imgs = [join(pred_dir, x) for x in png_name_list_2]
    dice = 0
    for ind in range(len(gt_imgs)):
        pred = np.array(Image.open(pred_imgs[ind])) / 255
        label = np.array(Image.open(gt_imgs[ind])) / 255
        xasx = label.flatten()
        if len(label.flatten()) != len(pred.flatten()):
            continue

        hist += fast_hist(label.flatten(), pred.flatten(), num_classes)

    IoUs = per_class_iu(hist)
    PA_Recall = per_class_PA_Recall(hist)
    Precision = per_class_Precision(hist)

    if name_classes is not None:
        for ind_class in range(num_classes):
            print('===>' + name_classes[ind_class] + ':\tIou-' + str(round(IoUs[ind_class] * 100, 2)) \
                  + '; Recall (equal to the PA)-' + str(round(PA_Recall[ind_class] * 100, 2)) + '; Precision-' + str(
                round(Precision[ind_class] * 100, 2)))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wm_iou_values = []
    for i in range(len(metric_output)):
        label = Image.fromarray(np.array(Image.open(gt_imgs[i])) / 255)
        data_transform = transforms.Compose([
            transforms.Resize(565, interpolation=T.InterpolationMode.NEAREST)  # 使用最近邻插值
        ])
        target = data_transform(label)
        target = np.expand_dims(np.array(target), axis=0)
        target = torch.from_numpy(target).cuda()
        input_x = metric_output[i]
        new_gt_list, logits_list, ln_dis_list = weigth_dis(input_x, target)
        wm_iou = weighted_iou(new_gt_list, logits_list, ln_dis_list, device)
        wm_iou_values.append(wm_iou.item())
    metric_WMIoU = sum(wm_iou_values) / len(wm_iou_values)
    print('===> mIoU: ' + str(round(np.nanmean(IoUs) * 100, 4)) + '; mPA: ' + str(
        round(np.nanmean(PA_Recall) * 100, 4)) + '; Accuracy: ' + str(round(per_Accuracy(hist) * 100, 4))
          + '; wm_iou: ' + str(round(metric_WMIoU * 100, 4)))
    return np.array(hist, int), IoUs, PA_Recall, Precision, dice, metric_WMIoU

def adjust_axes(r, t, fig, axes):
    bb = t.get_window_extent(renderer=r)
    text_width_inches = bb.width / fig.dpi
    current_fig_width = fig.get_figwidth()
    new_fig_width = current_fig_width + text_width_inches
    propotion = new_fig_width / current_fig_width
    x_lim = axes.get_xlim()
    axes.set_xlim([x_lim[0], x_lim[1] * propotion])

def draw_plot_func(values, name_classes, plot_title, x_label, output_path, tick_font_size=12, plt_show=True):
    fig = plt.gcf()
    axes = plt.gca()
    plt.barh(range(len(values)), values, color='royalblue')
    plt.title(plot_title, fontsize=tick_font_size + 2)
    plt.xlabel(x_label, fontsize=tick_font_size)
    plt.yticks(range(len(values)), name_classes, fontsize=tick_font_size)
    r = fig.canvas.get_renderer()
    for i, val in enumerate(values):
        str_val = " " + str(val)
        if val < 1.0:
            str_val = " {0:.4f}".format(val)
        t = plt.text(val, i, str_val, color='royalblue', va='center', fontweight='bold')
        if i == (len(values) - 1):
            adjust_axes(r, t, fig, axes)

    fig.tight_layout()
    fig.savefig(output_path)
    if plt_show:
        plt.show()
    plt.close()
def extract_number_from_filename(filename):
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    else:
        return float('inf')
def main11():
    classes = 1  # exclude background
    model_folder = "E:/Studydata/unet/save_weights/20"
    img_path = "E:/Studydata/unet/data/DRIVE/JPEGImages"
    txt_path = "E:/Studydata/unet/data/DRIVE/ImageSets/Segmentation/predict.txt"
    save_results = "./results/"
    os.makedirs(save_results, exist_ok=True)

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))
    print(torch.cuda.is_available())
    model = AttResUNet(in_channels=3, num_classes=classes + 1, base_c=64)
    # Load the list of model files
    model_files = [f for f in os.listdir(model_folder) if f.endswith(".pth")]
    # Load the list of model files and sort them based on the numeric part of the filename
    model_files.sort(key=extract_number_from_filename)
    print(model_files)
    for model_file in model_files:
        print("begin:")
        print(model_file)
        # 新建一个文件用于保存 model_file 的名称
        with open('save_weights/predict_results.txt', 'a') as file:
            file.write(model_file + '\n')
        # Create model
        weights_path = os.path.join(model_folder, model_file)

        # Load weights
        model.load_state_dict(torch.load(weights_path, map_location='cuda:0')['model'])
        model.to(device)

        total_time = 0
        with open(os.path.join(txt_path), 'r') as f:
            file_name1 = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        metric_output = []
        images = load_images(img_path, file_name1, mean, std)
        for file, img, h, w in images:
            model.eval()
            with torch.no_grad():
                # init model
                img_height, img_width = img.shape[-2:]
                init_img = torch.zeros((1, 3, img_height, img_width), device=device)
                model(init_img)
                t_start = time_synchronized()
                output = model(img.to(device))
                t_end = time_synchronized()
                total_time = total_time + (t_end - t_start)
                metric_output.append(output['out'])
                prediction = output['out'].argmax(1).squeeze(0)
                prediction = prediction.to("cpu").numpy().astype(np.uint8)
                prediction = cv2.resize(prediction, (w, h), interpolation=cv2.INTER_LINEAR)
                prediction[prediction == 1] = 255
                prediction[prediction == 0] = 0
                mask = Image.fromarray(prediction)
                mask = mask.convert("L")
                name = file[0:4]
                # mask = mask.convert("L")
                mask.save(os.path.join(save_results, f'{name}.png'))
        fps = 1 / (total_time / 286)
        print("FPS: {}".format(fps))

        # metric
        g_path = "E:/Studydata/unet/data/DRIVE/ground truth"
        pre_path = "./results/"
        txt_path1 = "E:/Studydata/unet/data/DRIVE/ImageSets/Segmentation/predict.txt"
        count = 0
        png_name_list = []
        png_name_list_2 = []
        with open(os.path.join(txt_path1), 'r') as f:
            file_name2 = [x.strip() for x in f.readlines() if len(x.strip()) > 0]
        for file in file_name2:
            count = count + 1
            png_name_list.append(file + ".png")
        count2 = 0
        for file in file_name2:
            count2 = count2 + 1
            png_name_list_2.append(file + ".png")

        with open('save_weights/predict_results.txt', 'a') as file:
            sys.stdout = file
            compute_mIoU(metric_output, g_path, pre_path, png_name_list, png_name_list_2, num_classes=2,
                     name_classes=["_background_", "Tactile_paving"])
            sys.stdout = sys.__stdout__
main11()
