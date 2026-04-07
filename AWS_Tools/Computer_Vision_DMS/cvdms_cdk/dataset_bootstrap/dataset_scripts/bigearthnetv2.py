from dataset_bootstrap.dataset_helpers.common import BootstrapConfig, BootstrapResult, DatasetBootstrapper


class BigEarthNetV2Bootstrapper(DatasetBootstrapper):
    dataset_name = "bigearthnet-v2"
    supported_exts = [".jpg", ".jpeg", ".png"]
    supported_tasks = {"multi-label"}

    BIGEARTHNET_S2_URL = "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1"
    BIGEARTHNET_METADATA_URL = "https://zenodo.org/records/10891137/files/metadata.parquet?download=1"
    BIGEARTHNET_REFERENCE_MAPS_URL = "https://zenodo.org/records/10891137/files/Reference_Maps.tar.zst?download=1"

    def bootstrap(self, config: BootstrapConfig, s3_client) -> BootstrapResult:
        self.validate_task(config.task)

        raise NotImplementedError(
            "BigEarthNet v2.0 scaffold is wired, but dataset-specific conversion is still pending. "
            "Next step: download metadata.parquet and BigEarthNet-S2.tar.zst, extract patch folders, "
            "render RGB composites from B04/B03/B02 into uploadable image files, upload those images "
            "to s3://<bucket>/<prefix>/bigearthnet-v2/multi-label/images/, and emit multi-label manifest rows."
        )