from dataset_bootstrap.dataset_scripts.bigearthnetv2 import BigEarthNetV2Bootstrapper
from dataset_bootstrap.dataset_scripts.eurosat import EuroSATBootstrapper
from dataset_bootstrap.dataset_scripts.coco2017 import CocoBootstrapper

DATASET_HELPERS = {
    "eurosat": EuroSATBootstrapper(),
    "bigearthnet-v2": BigEarthNetV2Bootstrapper(),
    "coco": CocoBootstrapper(),
}