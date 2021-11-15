# S7-Assignment
## 85% on CIFAR10 dataset with less than 200k parameters
### Submitted by Sumit Sarkar and Sadhana S
-----------------------------------------------------------------
### Network Architecture ###
1. Four convolutional layers
2. Third layer is a dilated convolutional layer
3. Fourth layer is a Depth Wise Seperable layer
4. First two convolutional layers are followed by a "Maxpool" layer. These "Maxpool" layers are implemented using 3x3 kernels with stride of 2
5. Output layer consists of a single fully connected layer

-----------------------------------------------------------------
### Data Augmentation ###
1. Albumentation library is used for image transformations
2. Since CIFAR10 dataset comes from Torch library a seperate class is written combining CIFAR10 dataset and Albumentation library transforms

-----------------------------------------------------------------
### Results ###
85% accuracy on test dataset is achieved in 80 epochs
