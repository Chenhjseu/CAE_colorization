# CAE_colorization

## 数据来源
  The CIFAR-10 dataset consists of 60 000 thousand color images of dimensions 32 × 32. Images can be of any of 10 categories, and each image may only be of one such category, although for this lab the category of the images is largely irrelevant. 
  Links are provided to download the dataset in a format already prepared for python or Matlab. However, since this dataset is so common, there is likely a binding in your preferred machine learning toolkit that will download the data for you in a format ready to be used. For instance:  
• Keras – https://keras.io/api/datasets/cifar10/.  
• Matlab – Look up the helperCIFAR10Data function.  
• Pytorch – Look up the torchvision.datasets.CIFAR10 class.

## Libraries
```
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import keras
```
