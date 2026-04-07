from dataset_bootstrap.dataset_scripts.bigearthnetv2 import BigEarthNetV2Bootstrapper
from dataset_bootstrap.dataset_scripts.eurosat import EuroSATBootstrapper
from dataset_bootstrap.dataset_scripts.spacenet2 import SpaceNet2Bootstrapper

DATASET_HELPERS = {
    "eurosat": EuroSATBootstrapper(),
    "bigearthnet-v2": BigEarthNetV2Bootstrapper(),
    "spacenet2": SpaceNet2Bootstrapper(),
}