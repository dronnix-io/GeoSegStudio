# **Deep Learning QGIS plugin.**

This is an **open-source** python-based plugin for QGIS software to provide the user with code-free implementation of DL-basec Semantic Segmentation operation for Geo-referenced raster files

**Note:** This plugin makes QGIS users able of taking advantage of applying the Deep Learning methods on their raster data either using either system GPU(s) or CPU(s)

The GUI uses *Pytorch* on the backend as its deep learning module for conducting its deep learning operations

**Author:** ___Salar Ghaffarian___   
    
**Email:** salarghaffarian1363@gmail.com


## The main ***workflow*** contains the following steps:

1. **R**asterizing the vector file of the ROIs (regions of interest)

    * This needs to consider the raster file's:
        - spatial resolution 
        - size (height, width)
   
1. **C**lipping Raster and Rasterized Shapefile/Vectors and save the clipped images.
    * This operation gets two parameters from the user:
        - Window size needed for clipping
        - Stride size which needs to move the clipping window along the image.
    * The final results is saved temporarily or permanently in a folder as ***All/Total*** 
   
1. **D**ataset Division is done based on the user-defined pecentages and the clipped images are divided into following categories and saved in their own directories:
    - Training Dataset
    - Validation Dataset
    - Testing Dataset
   
1. **A**ugmentation is applied to the Traing, Validation and Testing Dataset based on the user-selected augemnting techniques. 
    - **Note**: it is worth saving the augmentation results in a different folder than they were read!
   
1. **R**eading the Training and Validation Dataset using Dataset and DataLoader functions as pytorch tensors for both:
    - Images
    - Masks
    
1. **I**t gets the required hyper-parameters from the user for training the DL model from the user and train the model.
    - During the model training, Validation data is used for recording the model loss and its accuracy.
    - Checkpoints can be defined. to check the performance of the model during its trianing.
     
1. **B**est Trained model is defined and selected using the model accuracy against the Testing dataset.
    
1. **O**nce a desired trained model is acquired, the trained model can be used for conducting the semantic segmentation on a larger area.
    - This needs to use the following specification on the sliding window.
        - Window size which is normally equal to the size the clipped data in the beginning which is the input size for the DL model.
        - Stride/ moving window step size can be simply set as the window size. Otherwise, it needs to take care of the overlapped areas too.
        - Padding techniques needed to be applied for the edges of the large raster image in case it does not fit the sliding window at the very edges. 
        - StepMap normalization method can be used for overlapping cases.  
    
1. **S**ince the final result of the semantic segmentation is a probablity of belongining of a pixels to different classes, this results needs to be converted into solid classified binary masks in case only two classes is considered for classification.
    





 


