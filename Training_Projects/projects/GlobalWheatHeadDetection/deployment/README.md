# Global Wheat Head Detection Deployment

This folder contains the FastAPI serving app and Docker deployment files for the Global Wheat Head Detection YOLO model.

The app loads the selected MLflow pyfunc model from the MLflow Model Registry and serves wheat-head detections over HTTP.

Default model URI:

```text
models:/GlobalWheatHeadDetector@champion
```

The expected local serving flow is:

```text
Start MLflow server → build Docker image → run FastAPI container → send image prediction requests
```

## Folder layout

```text
deployment/
├── app/
│   ├── main.py
│   ├── app_factory.py
│   ├── config.py
│   ├── dependencies.py
│   ├── schemas.py
│   ├── routers/
│   └── services/
├── model_runtime/
│   └── ultralytics_pyfunc.py
├── requirements.txt
├── Dockerfile
├── .gitignore
└── README.md
```

## Prerequisites

Before running the deployment app:

1. The training workflow must have logged a valid MLflow pyfunc model.
2. The model-selection script must have registered `GlobalWheatHeadDetector`.
3. The registered model must have the `champion` alias.
4. Docker Desktop must be running.
5. The MLflow server must be started from the same project/location used for training and model registration.

You can verify the champion alias with:

```bash
python -c "import mlflow; from mlflow import MlflowClient; mlflow.set_tracking_uri('http://127.0.0.1:5000'); client = MlflowClient(); mv = client.get_model_version_by_alias(name='GlobalWheatHeadDetector', alias='champion'); print('version:', mv.version); print('source:', mv.source); print('run_id:', mv.run_id)"
```

## 1. Start MLflow

From the project root, start the MLflow server:

```bash
mlflow server --host 0.0.0.0 --port 5000 --allowed-hosts "localhost,localhost:*,127.0.0.1,127.0.0.1:*,host.docker.internal,host.docker.internal:*"
```

The MLflow UI should be available on the host machine at:

```text
http://127.0.0.1:5000
```

Inside Docker, the FastAPI container reaches the same MLflow server through:

```text
http://host.docker.internal:5000
```

## 2. Build the Docker image

From the `deployment/` folder:

```bash
docker build -t global-wheat-head-api:latest .
```

## 3. Run the API container

CPU mode is the default and safest Docker path:

```bash
docker run --rm -p 8000:8000 ^
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 ^
  -e MODEL_URI=models:/GlobalWheatHeadDetector@champion ^
  -e INFERENCE_DEVICE=cpu ^
  global-wheat-head-api:latest
```

On macOS/Linux shells, use backslashes:

```bash
docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MODEL_URI=models:/GlobalWheatHeadDetector@champion \
  -e INFERENCE_DEVICE=cpu \
  global-wheat-head-api:latest
```

The API should be available at:

```text
http://127.0.0.1:8000
```

## CPU vs GPU inference

The deployment app defaults to CPU inference because that is the most portable Docker demo path.

Default:

```text
INFERENCE_DEVICE=cpu
```

For local non-Docker testing, you can use a GPU device string such as:

```text
INFERENCE_DEVICE=0
```

or:

```text
INFERENCE_DEVICE=cuda:0
```

GPU inference inside Docker requires additional NVIDIA Docker setup, such as NVIDIA Container Toolkit and a Docker run command using `--gpus all`. Unless that is configured, use CPU mode for the container.

Example GPU-style Docker command, only after GPU Docker support is configured:

```bash
docker run --rm --gpus all -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MODEL_URI=models:/GlobalWheatHeadDetector@champion \
  -e INFERENCE_DEVICE=0 \
  global-wheat-head-api:latest
```

## 4. Health check

```bash
curl http://127.0.0.1:8000/health
```

PowerShell:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Expected shape:

```json
{
  "status": "ok",
  "app_name": "Global Wheat Head Detection API",
  "app_version": "0.1.0"
}
```

## 5. Check loaded model info

```bash
curl http://127.0.0.1:8000/model/info
```

PowerShell:

```powershell
curl.exe http://127.0.0.1:8000/model/info
```

Expected shape:

```json
{
  "model_loaded": true,
  "model_uri": "models:/GlobalWheatHeadDetector@champion",
  "mlflow_tracking_uri": "http://host.docker.internal:5000",
  "model_type": "PyFuncModel",
  "default_confidence_threshold": 0.25,
  "default_iou_threshold": 0.8,
  "default_image_size": 640,
  "default_max_det": 1000,
  "default_device": "cpu"
}
```

## 6. Send a prediction request

Replace the image path with a real wheat image:

```bash
curl -X POST http://127.0.0.1:8000/predict -F "file=@Absolute/path/to/image.jpg"
```

PowerShell:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict -F "file=@C:\path\to\image.jpg"
```

Expected response shape:

```json
{
  "filename": "image.jpg",
  "content_type": "image/jpeg",
  "image_width": 1024,
  "image_height": 1024,
  "detection_count": 3,
  "detections": [
    {
      "class_id": 0,
      "class_name": "wheat_head",
      "confidence": 0.87,
      "bbox_xyxy": {
        "x_min": 120.5,
        "y_min": 88.2,
        "x_max": 171.4,
        "y_max": 145.9
      },
      "bbox_xywh": null
    }
  ],
  "inference_settings": {
    "confidence_threshold": 0.25,
    "iou_threshold": 0.8,
    "image_size": 640,
    "max_det": 1000,
    "device": "cpu"
  },
  "timing_ms": {
    "upload_read_validate_save_ms": 12.3,
    "model_inference_ms": 145.8,
    "prediction_parse_ms": 1.2,
    "total_request_ms": 159.7
  },
  "latency_ms": 159.7
}
```

## 7. Debug raw model output

Use the debug endpoint if the clean parser needs troubleshooting:

```bash
curl -X POST http://127.0.0.1:8000/predict/debug -F "file=@path/to/image.jpg"
```

PowerShell:

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict/debug -F "file=@C:\path\to\image.jpg"
```

This returns the raw MLflow pyfunc output records.

## Configuration

The app reads configuration from environment variables.

| Environment variable       |                                    Default | Purpose                                     |
| -------------------------- | -----------------------------------------: | ------------------------------------------- |
| `APP_NAME`                 |          `Global Wheat Head Detection API` | FastAPI app name                            |
| `APP_VERSION`              |                                    `0.1.0` | FastAPI app version                         |
| `MLFLOW_TRACKING_URI`      |                    `http://127.0.0.1:5000` | MLflow tracking server URI                  |
| `MODEL_URI`                | `models:/GlobalWheatHeadDetector@champion` | Registered MLflow model URI                 |
| `CONFIDENCE_THRESHOLD`     |                                     `0.25` | YOLO confidence threshold                   |
| `IOU_THRESHOLD`            |                                      `0.8` | YOLO IoU/NMS threshold                      |
| `IMAGE_SIZE`               |                                      `640` | Inference image size                        |
| `MAX_DET`                  |                                     `1000` | Maximum detections per image                |
| `INFERENCE_DEVICE`         |                                      `cpu` | Inference device passed to the MLflow model |
| `TEMP_UPLOAD_DIR`          |                          `app/tmp/uploads` | Temporary upload location                   |
| `ENABLE_INFERENCE_LOGGING` |                                     `true` | Whether to write JSONL inference logs       |
| `INFERENCE_LOG_PATH`       |             `app/logs/inference_log.jsonl` | Local inference log path                    |

Example with custom settings:

```bash
docker run --rm -p 8000:8000 \
  -e MLFLOW_TRACKING_URI=http://host.docker.internal:5000 \
  -e MODEL_URI=models:/GlobalWheatHeadDetector@champion \
  -e CONFIDENCE_THRESHOLD=0.30 \
  -e IOU_THRESHOLD=0.8 \
  -e IMAGE_SIZE=640 \
  -e MAX_DET=1000 \
  -e INFERENCE_DEVICE=cpu \
  global-wheat-head-api:latest
```

## Latency logging

The API returns a timing breakdown in every `/predict` response:

```text
upload_read_validate_save_ms
model_inference_ms
prediction_parse_ms
total_request_ms
```

When `ENABLE_INFERENCE_LOGGING=true`, successful requests are also appended to:

```text
app/logs/inference_log.jsonl
```

These logs can be used later to create README performance charts, such as:

* request latency distribution
* model inference time distribution
* detection count vs. latency
* CPU vs. GPU inference comparison

The logged timings are app-level measurements, not a full production observability system.

## Why Docker uses `host.docker.internal`

From the host machine, MLflow is reachable at:

```text
http://127.0.0.1:5000
```

Inside Docker, `127.0.0.1` means the container itself. Docker Desktop provides:

```text
host.docker.internal
```

so the container can reach the MLflow server running on the host machine.

## Troubleshooting

### Container cannot connect to MLflow

Make sure MLflow is running with:

```bash
mlflow server --host 0.0.0.0 --port 5000
```

Then make sure the container uses:

```text
MLFLOW_TRACKING_URI=http://host.docker.internal:5000
```

### Model alias not found

Make sure model selection registered the model and assigned the `champion` alias:

```text
models:/GlobalWheatHeadDetector@champion
```

### Artifact loading fails

If the container reaches MLflow but cannot load the model artifact, check the registered model source in the MLflow UI.

The Docker app should be able to resolve and download the registered MLflow model through the tracking server. If the registered model source points to a raw local Windows path that is not accessible from Docker, re-register the model using an MLflow artifact source that the server can provide.

### Prediction output format is unsupported

Use:

```bash
curl -X POST http://127.0.0.1:8000/predict/debug -F "file=@path/to/image.jpg"
```

Then inspect the raw output and update:

```text
app/services/prediction_service.py
```

## Notes

The Docker image does not contain the trained YOLO weights directly. It loads the registered champion model from MLflow at startup.

Uploaded images are written to a temporary file because the MLflow pyfunc wrapper expects an `image_path` input column. The temporary file is deleted after each request.
