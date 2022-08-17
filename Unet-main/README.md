# Unet and NestedUnet for Horse Database



<img src=".\predict\20_0_ori.png" alt="20_0_ori" style="zoom:200%;" />

<img src=".\predict\20_0_predict.png" alt="20_0_predict" style="zoom:200%;" />

<img src=".\predict\20_0_predict_boundary.png" alt="20_0_predict_boundary" style="zoom:200%;" />


This repository uses the Weizmann Horse Database for training and segmentation prediction of standard Unet and NestedUnet networks.

**[U-Net: Convolutional Networks for Biomedical Image Segmentation](https://arxiv.org/abs/1505.04597)**

**[UNet++: A Nested U-Net Architecture for Medical Image Segmentation](https://arxiv.org/abs/1807.10165)**

##Environment

This model can run smoothly on the cpu , needless of gpu or other computation source 
.If you are willing to save more time (actually since the database is just about 300 pictures which makes using cpu to compute very quick)
you can try to use cuda to compute.

## Installation

My code has been tested on Python 3.10 and PyTorch 1.11.0. Please follow the instructions to configure your environment. See other required packages in `requirements.txt`.

## Download the pretrained Models ##

In the project: 

### ***Please click on the links below to get the models**.*

**_Password_：nvd7**

**Please place the model files in the same level as the .py file.**

**[Unet pretrained model](https://pan.baidu.com/s/1GPckAVctH4R4Yjx245wawA)** holds the Unet structure parameters of the best model from my training process. 

**[NestedUnet pretrained model](https://pan.baidu.com/s/149W_sDxGGwShX2F_FnDrMA )** holds the NestedUnet parameters of the best model from my training process. 

## Prepare Your Data

1. Please obtain the dataset from [Weizmann Horse Database | Kaggle](https://www.kaggle.com/datasets/ztaihong/weizmann-horse-database/metadata).
2. The dataset contains 327 images of horses and masked images.
3. In the train you need to change **"root"** to an **absolute path** to the weizmann_horse_db.
4. I used the **first 85%** of the images in the dataset for training and the **second 15%** for testing.
5. The final path structure used in my code looks like this:

````
$PATH_TO_DATASET/horse/weizmann_horse_db
├──── horse(327 images)
│    ├──── horse001.png
│  
├──── mask(327images)
│    ├──── horse001.png

````

## A Quick Demo

I picked a random image from the test images and then used the pre-trained model to semantically segment it and present the results on the console.

    python demo.py

## Training

Run the following command to train Unet :

    python train.py

- I have trained a model which you can use directly by setting the parameters in train.py.
- I was able to train the model on CPU(AMD Ryzen 7 4800H with 2.9GHz). The training took approximately 2 hours in 100 Epoch conditions.
    
If you want to train the NestedUnet,you should modify the train.py tools.py to change the model name so as to train the NestedUnet.

Attention:There will be no warning if you wrongly using the NestedUnet parameter to the Unet,if you choose to run the best_model.pth and the initial training mean IoU is under 0.90,you should check the right parameter.

## Testing ##

In the test set, the NestedUnet model I trained achieved about **0.89** of the MIoU, 0.68 of the Boundary IoU in the testing set.

The Unet model I trained achieved about 0.88 of the MIoU
and 0.67 of the Boundary IoU.


The results can be found in the and "log" folders.

