# BigEarthNet v2 Multi-Label Classifier

This project demonstrates a multi-label computer vision training workflow using CVDMS dataset artifacts as the source of truth. It trains a PyTorch model on a 17-class BigEarthNet v2 dataset version exported from CVDMS, using multi-hot labels, BCE-with-logits loss, thresholded predictions, and multi-label evaluation metrics.

## Dataset

The following chart are derived from the custom CVDMS visualization tool.
See [CVDMS CDK infrastructure](../../../AWS_Tools/Computer_Vision_DMS/cvdms_cdk/) for more.

<p align="center">
  <img src="readme_imgs/split_counts_percentages.png" alt="Split counts" width="600"><br>
  <em>CVDMS dataset visualization overview for the BigEarthNet v2 multi-label dataset.</em>
</p>