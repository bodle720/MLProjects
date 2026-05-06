# BigEarthNet v2 Multi-Label Classifier

This project demonstrates a multi-label computer vision training workflow using CVDMS dataset artifacts as the source
of truth. It trains a PyTorch model on a 17-class BigEarthNet v2 dataset version
exported from CVDMS, using multi-hot labels, BCE-with-logits loss, thresholded
predictions, and multi-label evaluation metrics.

## Dataset

The following charts were generated from the custom CVDMS visualization tool. 
Additional screenshots are available in the [readme_imgs](readme_imgs/) folder.
See the [CVDMS CDK infrastructure](../../../AWS_Tools/Computer_Vision_DMS/cvdms_cdk/)
for more context on the dataset creation pipeline.

This dataset is a small curated subset of BigEarthNet v2, consisting of 5,000 training
images, 1,000 validation images, and 1,000 test images.

<p align="center">
  <img src="readme_imgs/split_counts_percentages.png" alt="Split counts" width="900"><br>
  <em>CVDMS dataset visualization overview for a subset of the BigEarthNet v2 multi-label dataset.</em>
</p>

The following chart shows the class distribution across the train, validation, and test splits.
Note the heavy class imbalance.

<p align="center">
  <img src="readme_imgs/class_split_counts.png" alt="Class counts by split" width="900"><br>
  <em>Class breakdown by split.</em>
</p>

The following chart summarizes lighting categories by split. These image-quality summaries are
produced by the CVDMS visualization tool using Streamlit. The lighting distribution
is roughly consistent across the train, validation, and test splits.

<p align="center">
  <img src="readme_imgs/lighting_by_split.png" alt="Lighting buckets by split" width="900"><br>
  <em>Lighting breakdown by split.</em>
</p>

The following histogram shows the dark fraction distribution for each split.
Dark fraction is the fraction of image pixels whose brightness, or luma, falls
below a predefined dark threshold. Higher values indicate that a larger portion
of the image is composed of dark pixels.

For satellite imagery, this can help identify shadows, dark forested regions,
water-heavy scenes, cloudy or dim imagery, and other brightness-related distribution
differences. In this dataset, the dark fraction distributions are broadly similar
across splits, with visible clustering near 0.1.

<p align="center">
  <img src="readme_imgs/dark_frac_by_split.png" alt="Dark frac histogram by split" width="900"><br>
  <em>Dark fraction distribution by split.</em>
</p>

The following histogram compares image contrast across splits. Here,
contrast is approximated using luma standard deviation. Lower luma standard
deviation means most pixels have similar brightness values, producing a flatter
or lower-contrast image. Higher luma standard deviation means brightness values
vary more strongly across the image, which corresponds to stronger contrast.

<p align="center">
  <img src="readme_imgs/contrast_luma_std.png" alt="Contrast histogram by split" width="900"><br>
  <em>Contrast distribution by split.</em>
</p>