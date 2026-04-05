from .bigearthnetv2 import BigEarthNetV2Bootstrapper
from .eurosat import EuroSATBootstrapper
from .spacenet2 import SpaceNet2Bootstrapper

DATASET_HELPERS = {
    "eurosat": EuroSATBootstrapper(),
    "bigearthnet-v2": BigEarthNetV2Bootstrapper(),
    "spacenet2": SpaceNet2Bootstrapper(),
}