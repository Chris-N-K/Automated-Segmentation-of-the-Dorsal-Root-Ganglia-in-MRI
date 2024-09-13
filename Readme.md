# Code underlying the paper: <br>Automated-Segmentation-of-the-Dorsal-Root-Ganglia-in-MRI
The first jupyter notebook (Annotator Variability and Model Performance Evaluation) is used for calculation and visualisation of the inter-annotator metric scores and the evaluation of the segmentation performance of the model.
For this, the variability between the annotators is performed by calculating the metric scores for each possible annotator pair on labels. To test for the performance of the model, the Dice Similarity Coefficient (DSC) and the Average Surface Distance (ASD) are calculated for the test set labels created by each model.

The existing nnU-Net framework developed by Isensee et al. (2021) which includes a standard loss function of the nnU-Net (DC-CE), a combination of the dice loss (DC) and the binary cross-entropy (CE) was extended by our custom loss functions (LCD_loss): a compound of the default loss and a custom penalty term (DC-CE-LSP), and 2) a compound loss of DC and TopK (DC-TopK) (Ma et al., 2021).

Finally, and to calculate as well as visualise the DRG features of our predicted and ground truth labels for our Fabry disease cohort and our healthy controls our jupyter notebook "DRG Feature Analysis" was used.
