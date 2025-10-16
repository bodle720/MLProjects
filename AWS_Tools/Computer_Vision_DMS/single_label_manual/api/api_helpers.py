# -*- coding: utf-8 -*-
"""
Utility functions for the API.
"""

import os
import xml.etree.ElementTree as ET

import tifffile
import imagehash
from PIL import Image

VALID_BANDS = {"r", "g", "b", "l", "nir", "swir1", "swir2"}  # "l" = grayscale

BAND_ALIASES = {
    # visible
    "r": "r", "red": "r",
    "g": "g", "green": "g",
    "b": "b", "blue": "b",
    # grayscale
    "l": "l", "gray": "l", "grey": "l", "grayscale": "l", "lum": "l", "luma": "l",
    # multispectral
    "nir": "nir", "nearinfrared": "nir", "near-infrared": "nir",
    "swir1": "swir1",
    "swir2": "swir2"
}

def band_name_to_valid_name(band_name: str) -> str:
    key = band_name.strip().lower()
    if key in BAND_ALIASES:
        return BAND_ALIASES[key]
    raise ValueError(f"Invalid band name: {band_name} (expected one of {sorted(VALID_BANDS)})")

def extension_to_mime(ext: str) -> str:
    ext = ext.lower().lstrip(".")
    if ext in ("jpg", "jpeg"):
        return "image/jpeg"
    elif ext == "png":
        return "image/png"
    elif ext in ("tif", "tiff"):
        return "image/tiff"
    else:
        raise ValueError(f"Unsupported extension: {ext}")

def validate_band_info(band_info: dict[str, str]):
    if not isinstance(band_info, dict):
        raise ValueError("band_info must be a dict[str,str].")

    # Require consecutive "0..N-1" as strings
    keys = sorted(band_info.keys(), key=lambda k: int(k))
    expected = [str(i) for i in range(len(keys))]
    if keys != expected:
        raise ValueError(f"band_info keys must be consecutive strings starting at '0'. Got {keys}")

    # Require that the *raw* values are already valid canonical band names
    values = list(band_info.values())
    for v in values:
        if v not in VALID_BANDS:
            raise ValueError(f"Invalid band name '{v}'. Must be one of {sorted(VALID_BANDS)}.")

    # No duplicates allowed
    if len(set(values)) != len(values):
        raise ValueError(f"Duplicate band names found in {values}.")

def bands_appear_valid(path: str, desired_bands_order: list[str]) -> tuple[bool, str]:
    """
    Lightweight sanity check that the image at `path` appears consistent with desired_bands_order.
    Returns (True, "") if valid, or (False, reason) if not.
    """
    ext = os.path.splitext(path)[1].lower()

    try:
        if ext in (".tif", ".tiff"):
            with tifffile.TiffFile(path) as tif:
                arr = tif.asarray()
                bands_count = arr.shape[2] if arr.ndim == 3 else 1

                if bands_count != len(desired_bands_order):
                    return False, f"Band count {bands_count} does not match expected {len(desired_bands_order)}"

                # Try GDAL metadata
                tags = {}
                for page in tif.pages:
                    for tag in page.tags.values():
                        tags[tag.name] = tag.value
                if "GDAL_METADATA" in tags:
                    try:
                        root = ET.fromstring(tags["GDAL_METADATA"])
                        names = [item.text for item in root.findall(".//Item[@name='BandName']") if item.text]
                        if names and len(names) == bands_count:
                            norm_names = [band_name_to_valid_name(n) for n in names]
                            if norm_names != desired_bands_order:
                                return False, f"GDAL metadata bands {norm_names} do not match expected {desired_bands_order}"
                    except Exception as parse_err:
                        return False, f"Failed to parse GDAL metadata: {parse_err}"

        elif ext in (".jpeg", ".jpg", ".png"):
            with Image.open(path) as img:
                bands_count = len(img.getbands())
                if bands_count != len(desired_bands_order):
                    return False, f"Image has {bands_count} channels but expected {len(desired_bands_order)}"

        else:
            return False, f"Unsupported extension {ext}"

    except Exception as e:
        return False, f"Error inspecting {path}: {e}"

    return True, ""


def compute_phash(path: str) -> str:
    """Compute perceptual hash (phash) of an image file."""
    with Image.open(path) as img:
        # Ensure consistent conversion
        img = img.convert("L")
        return str(imagehash.phash(img))
    
def load_config_from_ssm(ssm_client, infrastructure_name: str) -> dict:
    """
    Loads all parameters for a given infrastructure_name from SSM Parameter Store
    and returns them as a dict.

    Raises:
        ValueError: if no parameters are found for the given infrastructure_name.
    """
    prefix = f"/cv-datasets/single-label/{infrastructure_name}/infrastructure/"
    config = {}

    paginator = ssm_client.get_paginator("get_parameters_by_path")
    for page in paginator.paginate(Path=prefix, Recursive=True, WithDecryption=True):
        for param in page.get("Parameters", []):
            key = param["Name"].split("/")[-1]  # last segment is the config key
            config[key] = param["Value"]

    if not config:
        raise ValueError(
            f"No parameters found under {prefix}. "
            f"Did you run part 2 to register this infrastructure?"
        )

    return config
