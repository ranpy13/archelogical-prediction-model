# Maya Contest Machine Learning Pipeline

Hint: for installation & run instructions go to section "Installation and Deployment"

## Overview of the solution pipeline
We applied these seven steps to obtain the results:

1. _Synthetic data generation_. As with many archaeological datasets, the original dataset turned out to have insufficient samples for the classes to be detected properly. So we use image processing techniques to create additional images and masks out of the available samples. → see section (3) Synthetic data generation below.
2. _Mask re-formatting (“normalization”)_. The library fastai (www.fast.ai) from Univ. San Francisco, which we use, requires binary segmentation masks to be “normalized” in the [0, 1] value range, so we re-formatted the provided masks from images with values [255, 0] to [0,1]. This is super easy, so we only provide the normalized masks. But in case you want the source code, just let us know.
3. Deep Learning model training. For each object class in {aguada, platform, building} a total of 2 x 5 neural networks are trained as follows:
    1. the LiDAR training dataset including the synthetic data (from step A above) is randomly shuffled and then split into 5 “folds” of equal size ( “K-Fold cross validation”, here K=5) which iteratively serve as validation dataset (and the remaining 4 folds are used as the training dataset), resulting into a 80% / 20% training / validation data split.
    2. Each of these 5 datasets then serves as input for training two different neural network architectures, namely:
        1. a DeepLabV3+ with ResNet-101 backbone
        2. an HRNet with HRNet_W30 backbone 
           
    For details see section (4) below.



