"""
Image loading helpers for CVDMS training projects.

This module provides small, swappable image-loader classes used by PyTorch
Dataset implementations.

The default loader reads images directly from S3 using the `source_ref` values
in CVDMS manifests. Later projects can swap in a local-cache loader or a
SageMaker input-channel loader without changing the Dataset class itself.
"""

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from PIL import Image, UnidentifiedImageError
from botocore.client import BaseClient

from cvdms_training_common.s3_io import parse_s3_uri, read_s3_bytes

class ImageLoader(Protocol):
    """
    Protocol for image loaders.

    Any callable with this shape can be used by CVDMS Dataset classes.
    """

    def __call__(self, uri: str) -> Image.Image:
        ...

@dataclass(frozen=True)
class S3ImageLoader:
    """
    Load PIL images directly from S3.

    Args:
        s3_client:
            Optional boto3 S3 client. If omitted, read_s3_bytes() creates one
            through the normal boto3 credential chain.
        mode:
            PIL image mode to convert into. For most image classification
            workflows, "RGB" is the correct default.
    """

    s3_client: BaseClient | None = None
    mode: str | None = "RGB"

    def __call__(self, uri: str) -> Image.Image:
        try:
            data = read_s3_bytes(uri, s3_client=self.s3_client)
        except Exception as exc:
            raise RuntimeError(f"Failed to read image bytes from {uri!r}") from exc

        try:
            with Image.open(io.BytesIO(data)) as img:
                if self.mode is None:
                    return img.copy()
                return img.convert(self.mode)
        except UnidentifiedImageError as exc:
            raise ValueError(f"S3 object is not a readable image: {uri!r}") from exc
        except OSError as exc:
            raise ValueError(f"Failed to decode image from S3 object: {uri!r}") from exc

@dataclass(frozen=True)
class LocalImageLoader:
    """
    Load PIL images from local filesystem paths.

    This is useful for debugging or for future cloud-training modes where the
    S3 data has already been materialized locally.

    The input URI may be:
        - a plain local path
        - file:///path/to/image.jpg

    Args:
        mode:
            PIL image mode to convert into.
    """

    mode: str | None = "RGB"

    def __call__(self, uri: str) -> Image.Image:
        path = _local_path_from_uri(uri)

        try:
            with Image.open(path) as img:
                if self.mode is None:
                    return img.copy()
                return img.convert(self.mode)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Local image does not exist: {path}") from exc
        except UnidentifiedImageError as exc:
            raise ValueError(f"Local file is not a readable image: {path}") from exc
        except OSError as exc:
            raise ValueError(f"Failed to decode local image: {path}") from exc

@dataclass(frozen=True)
class LocalMirrorImageLoader:
    """
    Load images from a local mirror of S3 object keys.

    This supports a useful future pattern:

        source_ref:
            s3://my-bucket/canonical/images/eurosat/img_001.jpg

        local_root:
            /opt/ml/input/data/images

        resolved local path:
            /opt/ml/input/data/images/canonical/images/eurosat/img_001.jpg

    This is useful when SageMaker or a preprocessing step has downloaded image
    objects locally but the manifests still contain S3 URIs.

    Args:
        local_root:
            Root directory containing files mirrored by S3 key.
        mode:
            PIL image mode to convert into.
    """

    local_root: str | Path
    mode: str | None = "RGB"

    def __call__(self, uri: str) -> Image.Image:
        parsed = parse_s3_uri(uri)
        path = Path(self.local_root) / parsed.key

        try:
            with Image.open(path) as img:
                if self.mode is None:
                    return img.copy()
                return img.convert(self.mode)
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                f"Mirrored local image does not exist for {uri!r}: {path}"
            ) from exc
        except UnidentifiedImageError as exc:
            raise ValueError(
                f"Mirrored local file is not a readable image for {uri!r}: {path}"
            ) from exc
        except OSError as exc:
            raise ValueError(
                f"Failed to decode mirrored local image for {uri!r}: {path}"
            ) from exc

def load_image_from_s3(
    uri: str,
    *,
    s3_client: BaseClient | None = None,
    mode: str = "RGB",
) -> Image.Image:
    """
    Convenience function for loading one image from S3.
    """
    return S3ImageLoader(s3_client=s3_client, mode=mode)(uri)

def load_image_from_local_path(
    uri: str,
    *,
    mode: str = "RGB",
) -> Image.Image:
    """
    Convenience function for loading one image from a local path or file URI.
    """
    return LocalImageLoader(mode=mode)(uri)

def _local_path_from_uri(uri: str) -> Path:
    if not isinstance(uri, str):
        raise TypeError(f"Expected image path string, got {type(uri).__name__}")

    text = uri.strip()
    if not text:
        raise ValueError("Image path cannot be empty")

    if text.startswith("file://"):
        text = text[7:]

    return Path(text)