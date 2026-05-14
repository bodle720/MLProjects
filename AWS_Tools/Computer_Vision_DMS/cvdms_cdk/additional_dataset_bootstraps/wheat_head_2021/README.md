# Global Wheat Head Detection 2021 CVDMS Bootstrapper

Standalone bootstrapper for preparing the Global Wheat Head Detection 2021 dataset for CVDMS object-detection ingestion. It downloads/reuses the Zenodo `gwhd_2021.zip` archive, extracts the official split CSVs and images, uploads selected images to the private CVDMS S3 bucket, and writes CVDMS-ready `manifest.jsonl`, `manifest.csv`, `summary.json`, and `failures.json` files.

This is intentionally dataset-specific code and is not part of the reusable `dataset_bootstrap` package. It supports one task, object detection, with one class: `wheat_head`. The upstream dataset provides official `train`, `val`, and `test` splits, so later CVDMS dataset construction should use `honor_source_splits = true`.

## Location

```text
cvdms_cdk/additional_dataset_bootstraps/wheat_head_2021/
```

## Dataset

```text
Source: https://zenodo.org/records/5092309
Archive: gwhd_2021.zip
Task: object-detection
Class: wheat_head
Splits: train, val, test
Images: PNG
Boxes: [x_min,y_min,x_max,y_max]
```

Rows with `BoxesString = no_box` are skipped by this bootstrapper because the generated CVDMS object-detection manifest rows require at least one annotation.

## Example commands

Run from the `cvdms_cdk` root. Omit `--max-items` to include all images available.

```bash
python -m additional_dataset_bootstraps.wheat_head_2021.main --aws-profile your_aws_profile --bucket your-bucket-name --split train --max-items 5000
```

or, for all the images,

```bash
python -m additional_dataset_bootstraps.wheat_head_2021.main --aws-profile your_aws_profile --bucket your-bucket-name --split train
```

Similarly, for the validation and test splits, run:

```bash
python -m additional_dataset_bootstraps.wheat_head_2021.main --aws-profile your_aws_profile --bucket your-bucket-name --split val --max-items 1000 --reuse-from-run-dir "C:\cvdms_files\runs\wheat_head_2021_object_detection_YYYYMMDD_HHMMSS_train"
```

and

```bash
python -m additional_dataset_bootstraps.wheat_head_2021.main --aws-profile your_aws_profile --bucket your-bucket-name --split test --max-items 1000 --reuse-from-run-dir "C:\cvdms_files\runs\wheat_head_2021_object_detection_YYYYMMDD_HHMMSS_train"
```

## Output

Each run creates a timestamped local folder under `--output-root`, defaulting to:

```text
C:\cvdms_files\runs\
```

The run folder contains:

```text
_work/downloads/
_work/extracted/
manifest.jsonl
manifest.csv
summary.json
failures.json
```

Images are uploaded to S3 under:

```text
s3://<bucket>/<s3-prefix>/global_wheat_head_2021/object-detection/images/<split>/<image_name>.png
```

Use `--reuse-from-run-dir` to reuse a previous run’s downloaded/extracted source files while still creating fresh manifests and S3 uploads for the current run.

## Attribution

The Global Wheat Head Dataset 2021 is published on Zenodo under CC BY 4.0.
