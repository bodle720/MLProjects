# Global Wheat Head Detection 2021 CVDMS Bootstrapper

Standalone bootstrapper for preparing the Global Wheat Head Detection 2021 dataset for CVDMS object-detection ingestion.

This script downloads or reuses the Zenodo `gwhd_2021.zip` archive, extracts the official split CSVs and images, uploads selected images to a private CVDMS S3 bucket, and writes CVDMS-ready output files:

* `manifest.jsonl`
* `manifest.csv`
* `summary.json`
* `failures.json`

This is intentionally dataset-specific code and is not part of the reusable `dataset_bootstrap/` package.

The upstream dataset provides official `train`, `val`, and `test` splits, so later CVDMS dataset construction should use:

```python
honor_source_splits = True
```

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
Boxes: [x_min, y_min, x_max, y_max]
```

Rows with `BoxesString = no_box` are skipped because CVDMS object-detection manifest rows require at least one annotation.

## Example Commands

Run from the `cvdms_cdk` root.

### Train split

Use `--max-items` for a smaller sample:

```bash
python -m additional_dataset_bootstraps.wheat_head_2021.main --aws-profile your_aws_profile --bucket your-bucket-name --split train --max-items 5000
```

Omit `--max-items` to include all available images:

```bash
python -m additional_dataset_bootstraps.wheat_head_2021.main --aws-profile your_aws_profile --bucket your-bucket-name --split train
```

### Validation and test splits

Use `--reuse-from-run-dir` to reuse the train run’s downloaded and extracted source files:

```bash
python -m additional_dataset_bootstraps.wheat_head_2021.main --aws-profile your_aws_profile --bucket your-bucket-name --split val --max-items 1000 --reuse-from-run-dir "C:\cvdms_files\runs\wheat_head_2021_object_detection_YYYYMMDD_HHMMSS_train"
```

```bash
python -m additional_dataset_bootstraps.wheat_head_2021.main --aws-profile your_aws_profile --bucket your-bucket-name --split test --max-items 1000 --reuse-from-run-dir "C:\cvdms_files\runs\wheat_head_2021_object_detection_YYYYMMDD_HHMMSS_train"
```

## Output

Each run creates a timestamped local folder under `--output-root`.

Default output root:

```text
C:\cvdms_files\runs\
```

Each run folder contains:

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

Use `--reuse-from-run-dir` to reuse a previous run’s downloaded and extracted source files while still creating fresh manifests and S3 uploads for the current split.

## Attribution

The Global Wheat Head Dataset 2021 is published on Zenodo under CC BY 4.0.
