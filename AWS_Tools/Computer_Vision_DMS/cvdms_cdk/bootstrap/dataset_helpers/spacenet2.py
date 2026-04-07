from .common import BootstrapConfig, BootstrapResult, DatasetBootstrapper

class SpaceNet2Bootstrapper(DatasetBootstrapper):
    dataset_name = "spacenet2"
    supported_exts = [".jpg", ".jpeg", ".png"]
    supported_tasks = {
        "object-detection",
        "semantic-segmentation",
        "instance-segmentation",
    }

    SPACENET_PUBLIC_BUCKET = "spacenet-dataset"
    SPACENET_ROOT_PREFIX = "spacenet/SN2_buildings/tarballs"

    AOI_TARBALL_KEYS = {
        "AOI_2_Vegas": "spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_2_Vegas.tar.gz",
        "AOI_3_Paris": "spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_3_Paris.tar.gz",
        "AOI_4_Shanghai": "spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_4_Shanghai.tar.gz",
        "AOI_5_Khartoum": "spacenet/SN2_buildings/tarballs/SN2_buildings_train_AOI_5_Khartoum.tar.gz",
    }

    def bootstrap(self, config: BootstrapConfig, s3_client) -> BootstrapResult:
        self.validate_task(config.task)

        raise NotImplementedError(
            "SpaceNet 2 scaffold is wired, but dataset-specific conversion is still pending. "
            "Next step: download one or more official AOI tarballs from the public SpaceNet bucket, "
            "parse the building polygons, then convert them by task: "
            "(1) object-detection -> bbox annotations, "
            "(2) semantic-segmentation -> merged building-vs-background PNG mask + color_map, "
            "(3) instance-segmentation -> synthetic worker_response_ref JSON + encoded PNG instance mask."
        )