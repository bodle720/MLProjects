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
    


def load_config_from_cf(cf_client, stack_name: str) -> dict:
    """
    Loads all CloudFormation outputs for a given stack into a dict,
    enforcing uniqueness (exactly one stack with that name must exist).

    Args:
        cf_client: boto3 CloudFormation client
        stack_name: the exact name of the deployed CDK stack

    Returns:
        dict mapping OutputKey -> OutputValue

    Raises:
        ValueError: if no stack or more than one stack is found,
                    or if the stack has no outputs.
    """
    resp = cf_client.describe_stacks(StackName=stack_name)

    stacks = resp.get("Stacks", [])
    if len(stacks) == 0:
        raise ValueError(f"No stack found with name '{stack_name}'")
    if len(stacks) > 1:
        raise ValueError(f"Multiple stacks found with name '{stack_name}', expected exactly one")

    outputs = stacks[0].get("Outputs", [])
    if not outputs:
        raise ValueError(f"Stack '{stack_name}' has no outputs")

    config = {o["OutputKey"]: o["OutputValue"] for o in outputs}
    return config


