"""
Standalone bootstrapper for Global Wheat Head Detection 2021.

This package downloads/extracts the GWHD 2021 dataset, preserves the official
train/val/test split choice, uploads selected images to the private CVDMS S3
bucket, and writes CVDMS-ready object-detection manifests.
"""