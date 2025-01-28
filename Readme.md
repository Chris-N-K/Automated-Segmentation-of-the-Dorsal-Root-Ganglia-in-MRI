# Automated Segmentation of the Dorsal Root Ganglia in MRI

This repository contains two Jupyter notebooks and one python file to demonstrate the data analysis described in the paper:

Nauroth-Kreß et al. (__2024__), _Automated Segmentation of the Dorsal Root Ganglia in MRI_

In this paper, we trained a customized version of [nnU-Net](https://github.com/MIC-DKFZ/nnUNet) to predict labels for Dorsal Root Ganglia in 3D MRI scans of patients with Fabry disease and healthy controls.

----

## 1) Inter-annotator variability and segmentation performance

[AnnoVar_ModelPerformance_eval.ipynb](AnnoVar_ModelPerformance_eval.ipynb) is the Jupyter notebook used for calculation and visualisation of the inter-annotator metric scores and the evaluation of the segmentation performance of the model. For this, the variability between the annotators is performed by calculating the metric scores for each possible annotator pair on labels. To test for the performance of the model, the Dice Similarity Coefficient (DSC) and the Average Surface Distance (ASD) are calculated for the test set labels created by each model.

## 2) Calculation and visualization of DRG features

[DRG_Feature_Analysis.ipynb](DRG_Feature_Analysis.ipynb) is the Jupyter notebook used to calculate as well as visualise the DRG features of our predicted and ground truth labels for the Fabry disease cohort and the healthy controls.

## 3) Custom segmentation loss for the nnU-Net framework

[LCD_loss.py](LCD_loss.py) contains two custom loss functions. The existing nnU-Net framework developed by Isensee et al. (2021) which includes a standard loss function of the nnU-Net (DC-CE), a combination of the dice loss (DC) and the binary cross-entropy (CE), was extended by the following two custom loss functions, as described in the manuscript:

1. a compound of the default loss and a custom penalty term (DC-CE-LSP), and 
2. a compound loss of DC and TopK (DC-TopK) (see [Ma et al., 2021](https://github.com/JunMa11/SegLossOdyssey))

## 4) How to use the nnU-Net

Download the ZIP file with the model weights from Zenodo (https://zenodo.org/uploads/14753009) and use nnU-Net V2 by unzipping the file and runnung the following command:

nnUNetv2_predict_from_modelfolder -i input/ -o output/ -m DRGNet/ -f ‘all’

The input files should be named #_0000.nii.gz and are expected in the folder input. One example image file is provided ([Example_MRI_data]).
    
----
