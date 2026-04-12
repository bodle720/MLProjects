from dataset_bootstrap.dataset_helpers.common import (
    BootstrapConfig,
    BootstrapResult,
    DatasetBootstrapper,
)
from dataset_bootstrap.dataset_helpers.coco2017.object_detection import coco_object_detection
from dataset_bootstrap.dataset_helpers.coco2017.semantic_segmentation import coco_semantic_segmentation
from dataset_bootstrap.dataset_helpers.coco2017.instance_segmentation import coco_instance_segmentation

class CocoBootstrapper(DatasetBootstrapper):
    dataset_name = "coco"
    supported_exts = {".jpg", ".jpeg", ".png"}
    supported_tasks = {
        "object-detection",
        "semantic-segmentation",
        "instance-segmentation",
    }

    def bootstrap(self, config: BootstrapConfig, s3_client) -> BootstrapResult:
        self.validate_task(config.task)

        if config.task == "object-detection":
            return coco_object_detection(config=config, s3_client=s3_client)

        if config.task == "semantic-segmentation":
            return coco_semantic_segmentation(config=config, s3_client=s3_client)

        if config.task == "instance-segmentation":
            return coco_instance_segmentation(config=config, s3_client=s3_client)

        raise NotImplementedError(f"Unsupported COCO task: {config.task}")