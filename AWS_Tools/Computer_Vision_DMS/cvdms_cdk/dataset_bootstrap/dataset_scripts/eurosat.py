from pathlib import Path

from dataset_bootstrap.dataset_helpers.common import (
    BootstrapConfig,
    BootstrapFailure,
    BootstrapResult,
    DatasetBootstrapper,
    deterministic_sample,
    download_http_file,
    ensure_dir,
    extract_zip,
    make_single_label_row,
    s3_key_join,
    upload_file_to_s3
)

from common.general_utils.class_normalizer import canonicalize_class_name

class EuroSATBootstrapper(DatasetBootstrapper):
    dataset_name = "eurosat"
    supported_tasks = {"single-label"}
    supported_exts = {".jpg", ".jpeg", ".png"}

    EUROSAT_RGB_ZIP_URL = "https://zenodo.org/records/7711810/files/EuroSAT_RGB.zip?download=1"

    def bootstrap(self, config: BootstrapConfig, s3_client) -> BootstrapResult:
        self.validate_task(config.task)

        download_dir = config.work_dir / "downloads"
        extract_dir = config.work_dir / "extracted"

        ensure_dir(download_dir)
        ensure_dir(extract_dir)

        url = self.EUROSAT_RGB_ZIP_URL
        destination = download_dir / "EuroSAT_RGB.zip"
        zip_path = download_http_file(
            url,
            destination,
        )
        extract_zip(zip_path, extract_dir)

        class_root = self._find_class_root(extract_dir)
        all_items = self._collect_items(class_root)
        selected_items = deterministic_sample(all_items, config.max_items, config.sample_seed)

        manifest_rows = []
        failures: list[BootstrapFailure] = []

        total = len(selected_items)
        idx = 0
        for image_path, class_name in selected_items:
            idx += 1
            if idx % 100 == 0 or idx == 1:
                print(f"On {idx} out of {total}. class_name = {class_name}")
            try:
                s3_key = s3_key_join(
                    config.s3_prefix,
                    self.dataset_name,
                    "single-label",
                    "images",
                    class_name,
                    image_path.name,
                )
                source_ref = upload_file_to_s3(
                    s3_client=s3_client,
                    local_path=image_path,
                    bucket=config.bucket,
                    key=s3_key,
                )
                manifest_rows.append(make_single_label_row(source_ref=source_ref, label=class_name))
            except Exception as exc:  # noqa: BLE001
                failures.append(
                    BootstrapFailure(
                        dataset_item_id=str(image_path),
                        reason=str(exc),
                        context={"class_name": class_name},
                    )
                )

        stats = {
            "upstream_dataset": "EuroSAT RGB",
            "download_url": self.EUROSAT_RGB_ZIP_URL,
            "class_root": str(class_root),
            "discovered_count": len(all_items),
            "selected_count": len(selected_items),
        }

        return BootstrapResult(
            manifest_rows=manifest_rows,
            failures=failures,
            stats=stats,
        )

    def _find_class_root(self, extracted_root: Path) -> Path:
        """
        Finds the first directory whose direct children look like class folders
        containing image files.
        """
        for candidate in [extracted_root] + [p for p in extracted_root.rglob("*") if p.is_dir()]:
            child_dirs = [p for p in candidate.iterdir() if p.is_dir()]
            if not child_dirs:
                continue

            valid_class_dirs = 0
            for child in child_dirs:
                has_images = any(
                    grandchild.is_file() and grandchild.suffix.lower() in self.supported_exts
                    for grandchild in child.iterdir()
                )
                if has_images:
                    valid_class_dirs += 1

            if valid_class_dirs >= 2:
                return candidate

        raise RuntimeError(
            f"Could not locate EuroSAT class-root under extracted directory: {extracted_root}"
        )

    def _collect_items(self, class_root: Path) -> list[tuple[Path, str]]:
        items: list[tuple[Path, str]] = []

        for class_dir in sorted(p for p in class_root.iterdir() if p.is_dir()):
            class_name = canonicalize_class_name(class_dir.name, field_name="class_dir.name")
            for image_path in sorted(class_dir.iterdir()):
                if image_path.is_file() and image_path.suffix.lower() in self.supported_exts:
                    items.append((image_path, class_name))

        if not items:
            raise RuntimeError(f"No EuroSAT images found under: {class_root}")

        return items