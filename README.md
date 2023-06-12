# 2018 Data Science Bowl using U-Net for Image Segmentation

Creating a image segmentation for 2018 Data Science Bowl using deep learning approach with U-Net

### Description
Objective: Create a deep learning model using U-Net
network to create a sentiment analysis for Ecommerce dataset

* Model training - Deep learning
* Method: Sequential, MobileNetV2, Concatenate, Conv2DTranspose
* Module: IPython, datetime, cv2, numpy, matplotlib & Tensorflow

In this image segmentation, dataset used from https://www.kaggle.com/competitions/data-science-bowl-2018/overview

### About The Dataset:
The dataset used in this analysis is 2018-Data-Science-Bowl dataset that contain 2 folder of train and test which have 67 images for both inputs and masks

The dataset will use os.listdir() method to list down all the image file, then use a for loop to read the images.

### Deep learning model with U-Net
A sequential model was created with 3 Sequential layer, 3 Concatenate layer, 1 Conv2DTranspose layer:
<p align="center">
  <img src="https://github.com/Ghost0705/Image_Segmentation_Data_Science_Bowl_2018/blob/main/Image/Architecture_flow.png">
</p>
<p align="center">
  <img src="https://github.com/Ghost0705/Image_Segmentation_Data_Science_Bowl_2018/blob/main/Image/model_architecture.png">
</p>

Data were trained with 10 epoch:
<p align="center">
  <img src="https://github.com/Ghost0705/Image_Segmentation_Data_Science_Bowl_2018/blob/main/Image/Model_performance.png">
</p>

Model deployment:
<p align="center">
  <img src="https://github.com/Ghost0705/Image_Segmentation_Data_Science_Bowl_2018/blob/main/Image/Model_deployment.png">
</p>

After the deployment, the model accuracy able to achieve 96% and val_accuracy of 95% for the dataset. The model is good enough to be used to predict the spot nuclei in the image dataset.

### How to run the pythons file:
1. Download the dataset
2. Change the filepath using your own path of the file
3. Run the image_segmentation.py 

Enjoy!

