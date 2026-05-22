import argparse
import json
import random
import statistics
import time
from pathlib import Path

import requests


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def find_images(image_dir):
    image_dir = Path(image_dir)
    images = [p for p in image_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS]
    if not images:
        raise ValueError(f"No images found under: {image_dir}")
    return images

def percentile(values, q):
    if not values:
        return None
    sorted_values = sorted(values)
    idx = (len(sorted_values) - 1) * q
    lower = int(idx)
    upper = min(lower + 1, len(sorted_values) - 1)
    weight = idx - lower
    return sorted_values[lower] * (1 - weight) + sorted_values[upper] * weight

def summarize(values):
    values = [v for v in values if v is not None]
    if not values:
        return None

    return {
        "count": len(values),
        "mean_ms": statistics.mean(values),
        "median_ms": statistics.median(values),
        "p90_ms": percentile(values, 0.90),
        "p95_ms": percentile(values, 0.95),
        "min_ms": min(values),
        "max_ms": max(values),
        "std_ms": statistics.stdev(values) if len(values) > 1 else 0.0,
    }

def find_numeric_field(obj, candidate_names):
    if isinstance(obj, dict):
        for name in candidate_names:
            value = obj.get(name)
            if isinstance(value, (int, float)):
                return float(value)

        for value in obj.values():
            found = find_numeric_field(value, candidate_names)
            if found is not None:
                return found

    if isinstance(obj, list):
        for item in obj:
            found = find_numeric_field(item, candidate_names)
            if found is not None:
                return found

    return None

def count_detections(response_json):
    candidates = [
        "detection_count",
        "num_detections",
        "prediction_count",
    ]

    found = find_numeric_field(response_json, candidates)
    if found is not None:
        return int(found)

    detections = response_json.get("detections") if isinstance(response_json, dict) else None
    if isinstance(detections, list):
        return len(detections)

    predictions = response_json.get("predictions") if isinstance(response_json, dict) else None
    if isinstance(predictions, list):
        return len(predictions)

    return None

def post_image(session, url, image_path, params):
    start = time.perf_counter()

    with open(image_path, "rb") as f:
        files = {"file": (image_path.name, f, "application/octet-stream")}
        response = session.post(url, files=files, params=params, timeout=120)

    client_wall_ms = (time.perf_counter() - start) * 1000.0
    response.raise_for_status()
    response_json = response.json()

    return {
        "image_path": str(image_path),
        "status_code": response.status_code,
        "client_wall_ms": client_wall_ms,
        "server_total_request_ms": find_numeric_field(
            response_json,
            ["total_request_ms", "latency_ms", "request_latency_ms"],
        ),
        "server_model_inference_ms": find_numeric_field(
            response_json,
            ["model_inference_ms", "inference_ms"],
        ),
        "server_prediction_parse_ms": find_numeric_field(
            response_json,
            ["prediction_parse_ms", "parse_ms"],
        ),
        "server_upload_validation_ms": find_numeric_field(
            response_json,
            ["upload_read_validate_save_ms", "upload_validation_ms"],
        ),
        "detection_count": count_detections(response_json),
    }

def write_markdown_summary(summary, output_path):
    rows = [
        ("Client wall-clock round trip", summary.get("client_wall_ms")),
        ("Server total request", summary.get("server_total_request_ms")),
        ("Server model inference", summary.get("server_model_inference_ms")),
        ("Server prediction parsing", summary.get("server_prediction_parse_ms")),
        ("Server upload / validation", summary.get("server_upload_validation_ms")),
    ]

    lines = [
        "# FastAPI Docker Latency Benchmark",
        "",
        "| Measurement | Count | Mean | Median | P90 | P95 | Min | Max | Std. dev. |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for label, stats in rows:
        if stats is None:
            continue

        lines.append(
            f"| {label} | "
            f"{stats['count']} | "
            f"{stats['mean_ms']:.3f} ms | "
            f"{stats['median_ms']:.3f} ms | "
            f"{stats['p90_ms']:.3f} ms | "
            f"{stats['p95_ms']:.3f} ms | "
            f"{stats['min_ms']:.3f} ms | "
            f"{stats['max_ms']:.3f} ms | "
            f"{stats['std_ms']:.3f} ms |"
        )

    output_path.write_text("\n".join(lines), encoding="utf-8")

def parse_params(param_items):
    params = {}
    for item in param_items:
        if "=" not in item:
            raise ValueError(f"Expected --param key=value, got: {item}")
        key, value = item.split("=", 1)
        params[key] = value
    return params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://127.0.0.1:8000/predict")
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--output-dir", default="deployment_latency_outputs")
    parser.add_argument("--requests", type=int, default=100)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--sleep-ms", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--param", action="append", default=[])
    args = parser.parse_args()

    random.seed(args.seed)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    images = find_images(args.image_dir)
    params = parse_params(args.param)

    records_path = output_dir / "latency_records.jsonl"
    summary_path = output_dir / "latency_summary.json"
    markdown_path = output_dir / "latency_summary.md"

    session = requests.Session()

    print(f"Found {len(images)} images")
    print(f"Running {args.warmup} warmup requests...")
    for _ in range(args.warmup):
        image_path = random.choice(images)
        post_image(session, args.url, image_path, params)

    print(f"Running {args.requests} measured requests...")
    records = []

    with records_path.open("w", encoding="utf-8") as f:
        for i in range(args.requests):
            image_path = random.choice(images)
            record = post_image(session, args.url, image_path, params)
            record["request_index"] = i
            records.append(record)
            f.write(json.dumps(record) + "\n")

            print(
                f"{i + 1}/{args.requests} "
                f"client={record['client_wall_ms']:.1f} ms "
                f"server={record['server_total_request_ms']} ms "
                f"detections={record['detection_count']}"
            )

            if args.sleep_ms > 0:
                time.sleep(args.sleep_ms / 1000.0)

    summary = {
        "url": args.url,
        "image_dir": str(Path(args.image_dir)),
        "requests": args.requests,
        "warmup": args.warmup,
        "sleep_ms": args.sleep_ms,
        "params": params,
        "client_wall_ms": summarize([r["client_wall_ms"] for r in records]),
        "server_total_request_ms": summarize([r["server_total_request_ms"] for r in records]),
        "server_model_inference_ms": summarize([r["server_model_inference_ms"] for r in records]),
        "server_prediction_parse_ms": summarize([r["server_prediction_parse_ms"] for r in records]),
        "server_upload_validation_ms": summarize([r["server_upload_validation_ms"] for r in records]),
        "detection_count": summarize([r["detection_count"] for r in records]),
    }

    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    write_markdown_summary(summary, markdown_path)

    print(f"\nSaved records to: {records_path}")
    print(f"Saved summary to: {summary_path}")
    print(f"Saved markdown table to: {markdown_path}")


if __name__ == "__main__":
    main()