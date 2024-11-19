# MSMAE

Pytorch Code for MSMAE Segmentation

# Preparation 
## Requirements
* mmcv-full==1.2.7
* opencv-python==4.5.1.48

## Datasets

If you would like to reproduce the results of our paper, please download our randomly split dataset below.
1) BUSI for Segmentation - [Link](https://drive.google.com/file/d/1eSNF0iOAa_szXFzntxCs7sj9Vxcio91k/view?usp=sharing)

## Data Format
If you want to train yourself, please split the dataset according to the settings in our appendix to reproduce our experimental results.
Make sure to put the files as the following structure (e.g. the number of classes is 2):

```
inputs
└── <dataset name>
    ├── images
    |   ├── 001.png
    │   ├── 002.png
    │   ├── 003.png
    │   ├── ...
    |
    └── masks
        ├── 0
        |   ├── 001.png
        |   ├── 002.png
        |   ├── 003.png
        |   ├── ...
        |
```

For binary segmentation problems, just use folder 0.

**Trained Model:**

| Dataset   | Backbone     | Method             | F1-Score            | IoU              | Checkpoint  |
| :-------- | :----------: |:-----------------: |:-------------------:|:----------------:| :----------------:|
| BUSI      | SETR         | MSMAE             | 60.66               | 45.53            | [Baidu Net (SMAE)](https://pan.baidu.com/s/1gK-IJ6CrfCCB-rKWQSnZ5A) |

## Training and Validation

1. Train the model.
```
python train.py \
    --name setr \
    --batch_size 8 \
    --epochs 200 \
    --arch setr \
    --input_w 224 \
    --input_h 224 \
    --dataset busi \
    --img_ext .png \
    --mask_ext .png \
    --root ${Dataset_DIR} \
    --checkpoint ${Pretrained Model Path}
```

2. Resume from checkpoint.
```
python train.py \
    --name setr \
    --batch_size 8 \
    --epochs 200 \
    --arch setr \
    --input_w 224 \
    --input_h 224 \
    --dataset busi \
    --img_ext .png \
    --mask_ext .png \
    --root ${Dataset_DIR} \
    --resume ${Previous Checkpoint Path}
```

3. Evaluate.
```
python train.py \
    --name setr \
    --batch_size 8 \
    --arch setr \
    --input_w 224 \
    --input_h 224 \
    --dataset busi \
    --img_ext .png \
    --mask_ext .png \
    --root ${Dataset_DIR} \
    --evaluate ${Trained Checkpoint Path}
```

