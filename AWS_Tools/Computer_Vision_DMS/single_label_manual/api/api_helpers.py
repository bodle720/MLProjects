# -*- coding: utf-8 -*-
"""
Utility functions for the API.
"""

import os
from PIL import Image
import imagehash
import tifffile
import xml.etree.ElementTree as ET
import numpy as np

VALID_BANDS = {"Red","Green","Blue","Gray","NIR","SWIR1","SWIR2"}  # extendable

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

def validate_band_info(band_info: dict[str,str]):
    if not isinstance(band_info, dict):
        raise ValueError("band_info must be a dict[str,str].")
        
    keys = list(band_info.keys())
    expected = [str(i) for i in range(len(keys))]
    
    if sorted(keys, key=int) != expected:
        raise ValueError(f"band_info keys must be consecutive strings starting at '0'. Got {keys}")
        
    for v in band_info.values():
        if v not in VALID_BANDS:
            raise ValueError(f"Invalid band name '{v}'. Must be one of {VALID_BANDS}.")
    
def extract_bands(path: str, bands: list[str]):
    """
    Inspect image and return band metadata. No conversion, just validation.
    """
    ext = os.path.splitext(path)[1].lower()
    bands_count = None
    bands_map = {}
    source = "api_arg"

    if ext in (".tif",".tiff"):
        with tifffile.TiffFile(path) as tif:
            arr = tif.asarray()
            tags = {tag.name: tag.value for page in tif.pages for tag in page.tags.values()}
        bands_count = arr.shape[2] if arr.ndim == 3 else 1

        if "GDAL_METADATA" in tags:
            try:
                xml_str = tags["GDAL_METADATA"]
                root = ET.fromstring(xml_str)
                names = []
                for band_meta in root.findall(".//BandMetadata"):
                    for item in band_meta.findall("Item"):
                        if item.attrib.get("name") == "BandName":
                            names.append(item.text)
                if len(names) == bands_count:
                    bands_map = {str(i): n for i,n in enumerate(names)}
                    source = "gdal_metadata"
            except Exception:
                pass
        else:
            bands_map = {str(i): b for i,b in enumerate(bands)}

    else:  # PNG/JPEG
        with Image.open(path) as img:
            arr = np.array(img)
            bands_count = len(img.getbands())
        bands_map = {str(i): b for i,b in enumerate(bands)}

    if len(bands) != bands_count:
        raise ValueError(f"Provided bands {bands} do not match image band count {bands_count}")

    return {"bands_count": bands_count, "bands_map": bands_map, "bands_source": source}

def compute_phash(path: str) -> str:
    """Compute perceptual hash (phash) of an image file."""
    with Image.open(path) as img:
        return str(imagehash.phash(img))