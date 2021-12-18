# S7-Assignment
## 85% on CIFAR10 dataset with less than 200k parameters
### Submitted by Sumit Sarkar and Sadhana S

### Network Architecture 
1. Four convolutional layers
2. Third layer is a dilated convolutional layer
3. Fourth layer is a Depth Wise Seperable layer
4. First two convolutional layers are followed by a "Maxpool" layer. These "Maxpool" layers are implemented using 3x3 kernels with stride of 2
5. Output layer consists of a single fully connected layer
6. Total number of parameters in the network is < 200k

### Model Summary
```
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 36, 32, 32]             972
              ReLU-2           [-1, 36, 32, 32]               0
       BatchNorm2d-3           [-1, 36, 32, 32]              72
            Conv2d-4           [-1, 36, 16, 16]          11,664
              ReLU-5           [-1, 36, 16, 16]               0
       BatchNorm2d-6           [-1, 36, 16, 16]              72
            Conv2d-7           [-1, 70, 14, 14]          22,680
              ReLU-8           [-1, 70, 14, 14]               0
       BatchNorm2d-9           [-1, 70, 14, 14]             140
           Conv2d-10             [-1, 70, 7, 7]          44,100
             ReLU-11             [-1, 70, 7, 7]               0
      BatchNorm2d-12             [-1, 70, 7, 7]             140
           Conv2d-13            [-1, 128, 5, 5]          80,640
             ReLU-14            [-1, 128, 5, 5]               0
      BatchNorm2d-15            [-1, 128, 5, 5]             256
           Conv2d-16            [-1, 128, 3, 3]           1,152
           Conv2d-17            [-1, 256, 3, 3]          32,768
 DepthWiseSepConv-18            [-1, 256, 3, 3]               0
             ReLU-19            [-1, 256, 3, 3]               0
      BatchNorm2d-20            [-1, 256, 3, 3]             512
        AvgPool2d-21            [-1, 256, 1, 1]               0
           Linear-22                   [-1, 10]           2,560
================================================================
Total params: 197,728
Trainable params: 197,728
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 1.60
Params size (MB): 0.75
Estimated Total Size (MB): 2.37
----------------------------------------------------------------
```
### Data Augmentation 

Albumentation is a Python library for fast and flexible image augmenation
1. Albumentation library is used for image transformations
2. Since CIFAR10 dataset comes from Torch library a seperate class is written combining CIFAR10 dataset and Albumentation library transforms

transformation performed on the data are

1. HorizontalFlip - Flip the input image.
2. ShiftScaleRotate - Randomly apply affine transforms: translate, scale and rotate the input.
3. CoarseDropout - CoarseDropout of the rectangular regions in the image.
4. RandomBrightnessContrast-Randomly change brightness and contrast of the input image.
5. Normalizee - Normalization is applied by the formula: img = (img - mean * max_pixel_value) / (std * max_pixel_value) to noramlize the image.
6. ToTensorV2 - Converts image and mask to torch.Tensor. 

```
def getTrainTransforms():
    train_transforms = A.Compose([
         A.HorizontalFlip(p=0.5),
         A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=10, p=0.5),
         A.CoarseDropout(max_holes = 1, max_height=16, max_width=16, min_holes = 1, min_height=16, min_width=16,
                        fill_value=(0.47)),
         A.RandomBrightnessContrast (brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, 
                                     always_apply=False, p=0.4),
         A.Normalize(mean=(0.49139968, 0.48215827 ,0.44653124), std=(0.24703233,0.24348505,0.26158768)),
         ToTensorV2(),
         ])
    return train_transforms
```

### Results 
```
EPOCH: 79
Loss=0.4280511438846588 Batch_id=195 Accuracy=86.38: 100%|█| 196/196 [00:07
Accuracy of the network on the 10000 test images: 85 %
```

85% accuracy on test dataset is achieved in 80 epochs
