# BigEarthNet v2 Multi-Label Classifier

This project demonstrates a multi-label computer vision training workflow using CVDMS dataset artifacts as the source of truth. It trains a PyTorch model on a 17-class BigEarthNet v2 dataset version exported from CVDMS, using multi-hot labels, BCE-with-logits loss, thresholded predictions, and multi-label evaluation metrics.

## Dataset

The following charts are derived from the custom CVDMS visualization tool. You can view the images
by opening them in the readme_imgs/ folder. Some of them are referenced here.
See [CVDMS CDK infrastructure](../../../AWS_Tools/Computer_Vision_DMS/cvdms_cdk/) for more.

This image indicates our dataset is a small subset of the BigEarthNet V2
dataset, consisting of 5,000 training images and 1,000 validation and testing
images.

<p align="center">
  <img src="readme_imgs/split_counts_percentages.png" alt="Split counts" width="900"><br>
  <em>CVDMS dataset visualization overview for a subset of the BigEarthNet v2 multi-label dataset.</em>
</p>

The following is a breakdown of the dataset by class and split.

<p align="center">
  <img src="readme_imgs/class_split_counts.png" alt="Class counts by split" width="900" height="300"><br>
  <em>Class breakdown by split.</em>
</p>
