# AttRes-UNet: Visual segmentation network with attention residual structure and weighted mIoU loss
### 
#### 

* Structure of the proposed AttRes-UNet network:
![](assets/img_1.png)


## Experimental results
****

### Comparison of experimental prediction results of advanced network models:

| Method            | IoU    | Precison | Recall |Accuracy | mPA    | mIoU   | wmIoU  |
|-------------------|--------|----------|--------|---------|--------|--------|--------|
| FCN               | 52.57  | 85.21    | 57.84  | 93.51   | 78.21  | 72.79  |    /   |
| Simt              | 74.28  | 97.32    | 75.83  | 96.74   | 87.77  | 85.34  |    /   |
| PSPNet            | 88.49  | 95.77    | 92.08  | 98.51   | 95.75  | 93.40  |    /   |
| SegNet            | 70.69  | 91.12    | 84.62  | 95.64   | 90.91  | 82.91  |    /   |
| DeepLabV3         | 87.09  | 95.18    | 91.11  | 98.32   | 95.23  | 92.60  |    /   |
| DeepLabV3+        | 85.26  | 93.55    | 90.59  | 98.85   | 94.85  | 91.53  |    /   |
| Mask R-CNN        | 77.31  | 90.38    | 77.17  | 96.14   | 88.00  | 83.52  |    /   |
| AttRes-UNet       | 89.91  | 97.16    | 92.33  | 98.71   | 95.98  | 94.22  | 88.31  |
| GRFB-UNet         | 91.21  | 96.65    | 94.20  | 98.85   | 96.76  | 84.85  | 89.43  |
| AttRes-GRFB-UNet  | 92.97  | 96.84    | 95.88  | 99.09   | 97.72  | 95.97  | 90.94  |

Notation: The results in the table are all percentage data (%).
## Installation
****

* Install dependencies. CUDA, Python and PyTorch:

&ensp;1. [Install CUDA](https://developer.nvidia.com/cuda-downloads);

&ensp;2. Install anaconda, and create conda environment;

&ensp;3. [Install PyTorch 1.13.0 or later](https://pytorch.org/get-started/locally/);

&ensp;4. Install other dependencies.

The environment can be installed with following orders:
```bash
conda create -n py python=3.7
conda activate py
conda install pytorch==1.13.0 torchvision==0.14.1 cudatoolkit=11.7 -c pytorch
pip install -r requirements.txt
```
or
```bash
conda create -n py python=3.7
conda activate py
conda install pytorch==1.13.0 torchvision==0.14.1 cudatoolkit=11.7 -c pytorch
conda install --yes --file requirements.txt
```

## Data preparation
***
**The TP-dataset can be downloaded from the following links:**

&ensp;1. [Baidu Net-disk](https://pan.baidu.com/s/1YgutfaiVE2KkqcKnWfGLSQ) (password: 9ope)

&ensp;2. [Google Drive](https://drive.google.com/drive/folders/1jByE5f_oUKpYdoR829wLqFSqlBYtOZM6?usp=sharing)

* After downloading, you should prepare the data in the following structure

```
TP-Dataset
   |——————JPEGImages
   |        └——————Part01
   |        └——————… …
   |——————GroundTruth
   |        └——————Part01
   |        └——————… …
   └——————SegmentationClassPNG
   |        └——————Part01
   |        └——————… …
   └——————Index
            └——————train.txt
            └——————val.txt
            └——————predict.txt
```
For each part in TP-Dataset, the images are stored in the file ./JPEGImages, and ground truths are stored in file ./ground truth. The masked images are stored in the file ./ SegmentationClassPNG and only used for visualization, which can be removed without any influence.
The indexes of train / validate / test samples are stored in flie ./Index.
Then, you can replace the file ./data/TP-dataset in ${GRFBNet_ROOT} with the downloaded dataset for training phase. 

Note that the images in this project are just some instances, they must be replaced by the aforemetioned dataset.

### Training

```console
> python train.py
usage: train.py [--data-path DP] [--num-classes NC] [--device D]
                [--batch-size BZ] [--epochs E] [--lr LR] [--momentum M]
		[--weight-decay WD][--print-freq PF][--resume R]
		[--start-epoch SE][--save-best SB][--amp A]

Train the UNet on images and ground truth (target masks)

optional arguments:
  '--data-path', default='./data/', help='dataset file path including ./TP-Dataset'
  '-num-classes', default=1, type=int
  '--device', default='cuda', help='training device'
  '-b', '--batch-size', default=8, type=int
  '--epochs', default=1800, type=int, metavar='N', help='number of total epochs to train'
  '--lr', default=0.02, type=float, help='initial learning rate'
  '--momentum', default=0.9, type=float, metavar='M', help='momentum'
  '--wd', '--weight-decay', default=1e-4, type=float, metavar='W', help='weight decay (default: 1e-4)', dest='weight_decay'
  '--print-freq', default=1, type=int, help='print frequency'
  '--resume', default='', help='resume from checkpoint'
  '--start-epoch', default=0, type=int, metavar='N', help='start epoch'
  '--save-best', default=True, type=bool, help='only save best dice weights'
  '--amp', default=False, type=bool, help='Use torch.cuda.amp for mixed precision training'
```
The pre-trained model can be downloaded from (/code/AttRes-main/save_weights/model_best.pth).
## Prediction
***
* After training phase, the models are saved in the file ./save_weights. At last, the testing images with the labels in predict.txt are predicted with

```console
python predict.py
usage: predict.py [--weights_path WP] [--img_path IP]
                [--txt_path TP] [--save_result SR]

optional arguments:
'--weights_path', default='./save_weights/model_best.pth', help='The root of TP-Dataset ground truth list file')
'--img_path', default='./data/TP-Dataset/JPEGImages', help='The path of testing sample images')
'--txt_path', default='./data/TP-Dataset/Index/predict.txt', help='The path of testing sample list')
'--save_result', default='./predict', help='The path of saved predicted results in images')
```

## Citation
```

```
## Acknowledgement
Our code is constructed based on the basic structure of [UNet](https://arxiv.org/abs/1505.04597). Thanks for their wonderful works.
```
@inproceedings{ronneberger2015u,
  title={U-net: Convolutional networks for biomedical image segmentation},
  author={Ronneberger, Olaf and Fischer, Philipp and Brox, Thomas},
  booktitle={Medical Image Computing and Computer-Assisted Intervention--MICCAI 2015: 18th International Conference, Munich, Germany, October 5-9, 2015, Proceedings, Part III 18},
  pages={234--241},
  year={2015},
  organization={Springer}
}
```
