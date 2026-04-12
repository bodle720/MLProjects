import tarfile
from pathlib import Path
from typing import Any
import time
import numpy as np
import pyarrow.parquet as pq
import tifffile
import zstandard as zstd
from PIL import Image
import random

from dataset_bootstrap.dataset_helpers.common import (
    BootstrapConfig,
    BootstrapFailure,
    BootstrapResult,
    DatasetBootstrapper,
    deterministic_sample,
    download_http_file,
    ensure_dir,
    make_multi_label_row,
    s3_key_join,
    upload_file_to_s3,
    format_bytes
)

class BigEarthNetV2Bootstrapper(DatasetBootstrapper):
    dataset_name = "bigearthnet-v2"
    supported_tasks = {"multi-label"}
    # Produces PNGs from extracted tiffs

    BIGEARTHNET_S2_URL = "https://zenodo.org/records/10891137/files/BigEarthNet-S2.tar.zst?download=1"
    BIGEARTHNET_METADATA_URL = "https://zenodo.org/records/10891137/files/metadata.parquet?download=1"

    def bootstrap(self, config: BootstrapConfig, s3_client) -> BootstrapResult:
        self.validate_task(config.task)

        download_dir = config.work_dir / "downloads"
        extract_dir = config.work_dir / "extracted"
        rendered_dir = config.work_dir / "rendered_rgb"

        ensure_dir(download_dir)
        ensure_dir(extract_dir)
        ensure_dir(rendered_dir)

        reuse_stats = {
            "reuse_from_run_dir": str(config.reuse_from_run_dir) if config.reuse_from_run_dir else None,
            "reused_metadata": False,
            "reused_archive": False,
            "reused_extracted_tree": False,
        }

        metadata_path, s2_archive_path, s2_root = self._resolve_bigearthnet_inputs(
            config=config,
            download_dir=download_dir,
            extract_dir=extract_dir,
            reuse_stats=reuse_stats,
        )

        all_items = self._load_metadata_rows(metadata_path)
        split_items = self._filter_rows_for_split(all_items, config.split)
        selected_items = deterministic_sample(split_items, config.max_items, config.sample_seed)

        manifest_rows = []
        failures: list[BootstrapFailure] = []

        total = len(selected_items)
        for idx, item in enumerate(selected_items, start=1):
            patch_id = item["patch_id"]
            labels = item["labels"]
            split = item.get("split", "unknown")

            if idx % 100 == 0 or idx == 1:
                print(f"On {idx} out of {total}. patch_id = {patch_id}, split = {split}")

            try:
                patch_dir = self._resolve_patch_dir(s2_root=s2_root, patch_id=patch_id)
                rgb_png_path = rendered_dir / split / f"{patch_id}.png"
                ensure_dir(rgb_png_path.parent)

                if not rgb_png_path.exists():
                    self._render_rgb_png(
                        patch_dir=patch_dir,
                        patch_id=patch_id,
                        out_path=rgb_png_path,
                    )

                s3_key = s3_key_join(
                    config.s3_prefix,
                    self.dataset_name,
                    "multi-label",
                    "images",
                    split,
                    rgb_png_path.name,
                )
                source_ref = upload_file_to_s3(
                    s3_client=s3_client,
                    local_path=rgb_png_path,
                    bucket=config.bucket,
                    key=s3_key,
                )

                manifest_rows.append(
                    make_multi_label_row(
                        source_ref=source_ref,
                        labels=labels,
                    )
                )

            except Exception as exc:  # noqa: BLE001
                failures.append(
                    BootstrapFailure(
                        dataset_item_id=patch_id,
                        reason=str(exc),
                        context={
                            "split": split,
                            "labels": labels,
                        },
                    )
                )

        stats = {
            "upstream_dataset": "BigEarthNet v2.0 Sentinel-2",
            "requested_split": config.split,
            "metadata_path": str(metadata_path),
            "s2_archive_path": str(s2_archive_path) if s2_archive_path is not None else None,
            "s2_root": str(s2_root),
            "discovered_count": len(all_items),
            "split_count": len(split_items),
            "selected_count": len(selected_items),
            **reuse_stats,
        }

        return BootstrapResult(
            manifest_rows=manifest_rows,
            failures=failures,
            stats=stats,
        )

    def _resolve_bigearthnet_inputs(
            self,
            *,
            config: BootstrapConfig,
            download_dir: Path,
            extract_dir: Path,
            reuse_stats: dict[str, Any],
    ) -> tuple[Path, Path | None, Path]:
        old_metadata_path = None
        old_archive_path = None
        old_s2_root = None

        if config.reuse_from_run_dir:
            old_work_dir = config.reuse_from_run_dir / "_work"
            old_metadata_path = old_work_dir / "downloads" / "metadata.parquet"
            old_archive_path = old_work_dir / "downloads" / "BigEarthNet-S2.tar.zst"
            old_s2_root = old_work_dir / "extracted" / "BigEarthNet-S2"

        if old_metadata_path and old_metadata_path.is_file():
            metadata_path = old_metadata_path
            reuse_stats["reused_metadata"] = True
        else:
            metadata_path = download_http_file(
                self.BIGEARTHNET_METADATA_URL,
                download_dir / "metadata.parquet",
            )

        if old_s2_root and old_s2_root.is_dir():
            if self._reused_s2_root_looks_usable(
                    s2_root=old_s2_root,
                    metadata_path=metadata_path,
                    probe_count=5,
                    seed=config.sample_seed,
            ):
                s2_root = old_s2_root
                s2_archive_path = old_archive_path if old_archive_path and old_archive_path.is_file() else None
                reuse_stats["reused_extracted_tree"] = True
                if s2_archive_path is not None:
                    reuse_stats["reused_archive"] = True
                return metadata_path, s2_archive_path, s2_root

            print(f"[reuse] ignoring bad extracted tree from prior run: {old_s2_root}")

        if old_archive_path and old_archive_path.is_file():
            s2_archive_path = old_archive_path
            reuse_stats["reused_archive"] = True
        else:
            s2_archive_path = download_http_file(
                self.BIGEARTHNET_S2_URL,
                download_dir / "BigEarthNet-S2.tar.zst",
            )

        s2_root = extract_dir / "BigEarthNet-S2"
        if not s2_root.exists():
            self._extract_tar_zst(
                archive_path=s2_archive_path,
                destination_dir=extract_dir,
            )

        return metadata_path, s2_archive_path, s2_root

    def _load_metadata_rows(self, metadata_path: Path) -> list[dict[str, Any]]:
        table = pq.read_table(metadata_path, columns=["patch_id", "labels", "split"])
        rows = table.to_pylist()

        out: list[dict[str, Any]] = []
        for idx, row in enumerate(rows):
            patch_id = row.get("patch_id")
            labels = row.get("labels")
            split = row.get("split")

            if not isinstance(patch_id, str) or not patch_id.strip():
                raise ValueError(f"metadata row {idx}: missing or invalid patch_id")

            if not isinstance(labels, list) or not labels:
                raise ValueError(f"metadata row {idx}: missing or invalid labels for patch_id={patch_id}")

            normalized_labels = []
            for lab in labels:
                if not isinstance(lab, str):
                    raise ValueError(f"metadata row {idx}: non-string label for patch_id={patch_id}: {lab!r}")
                lab_norm = lab.strip()
                if not lab_norm:
                    raise ValueError(f"metadata row {idx}: empty label after stripping for patch_id={patch_id}")
                normalized_labels.append(lab_norm)

            split_norm = self._normalize_bigearthnet_split(
                split,
                row_idx=idx,
                patch_id=patch_id,
            )

            out.append(
                {
                    "patch_id": patch_id.strip(),
                    "labels": normalized_labels,
                    "split": split_norm,
                }
            )

        return out

    def _resolve_patch_dir(self, s2_root: Path, patch_id: str) -> Path:
        # BigEarthNet-S2/<source-tile>/<patch-id>/
        # source tile dir is patch_id without the final _<H-Order>_<V-Order>
        tile_dir_name = patch_id.rsplit("_", 2)[0]
        patch_dir = s2_root / tile_dir_name / patch_id

        if not patch_dir.exists() or not patch_dir.is_dir():
            raise FileNotFoundError(f"Patch directory not found for patch_id={patch_id}: {patch_dir}")

        return patch_dir

    def _render_rgb_png(self, patch_dir: Path, patch_id: str, out_path: Path) -> None:
        # BigEarthNet Sentinel-2 RGB composite uses B04 (R), B03 (G), B02 (B).
        b04 = tifffile.imread(patch_dir / f"{patch_id}_B04.tif")
        b03 = tifffile.imread(patch_dir / f"{patch_id}_B03.tif")
        b02 = tifffile.imread(patch_dir / f"{patch_id}_B02.tif")

        rgb = np.stack([b04, b03, b02], axis=-1).astype(np.float32)
        rgb_uint8 = self._stretch_to_uint8(rgb)

        image = Image.fromarray(rgb_uint8, mode="RGB")
        image.save(out_path, format="PNG")

    def _stretch_to_uint8(self, rgb: np.ndarray) -> np.ndarray:
        # Engineering choice for human-viewable composite:
        # percentile stretch each channel independently, then clip to 0..255.
        out = np.zeros_like(rgb, dtype=np.uint8)

        for c in range(3):
            channel = rgb[..., c]
            lo = np.percentile(channel, 2)
            hi = np.percentile(channel, 98)

            if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
                scaled = np.zeros_like(channel, dtype=np.uint8)
            else:
                scaled_f = (channel - lo) / (hi - lo)
                scaled_f = np.clip(scaled_f, 0.0, 1.0)
                scaled = (scaled_f * 255.0).round().astype(np.uint8)

            out[..., c] = scaled

        return out

    def _extract_tar_zst(self, archive_path: Path, destination_dir: Path) -> None:
        ensure_dir(destination_dir)

        total_compressed_size = archive_path.stat().st_size
        extracted_members = 0
        extracted_logical_bytes = 0
        last_print_time = time.time()

        with archive_path.open("rb") as fh:
            dctx = zstd.ZstdDecompressor()
            with dctx.stream_reader(fh) as reader:
                with tarfile.open(fileobj=reader, mode="r|") as tf:
                    for member in tf:
                        tf.extract(member, destination_dir, filter="data")
                        extracted_members += 1

                        if member.isfile():
                            extracted_logical_bytes += member.size

                        now = time.time()
                        if now - last_print_time >= 5.0:
                            compressed_done = fh.tell()
                            pct = (
                                (compressed_done / total_compressed_size) * 100.0
                                if total_compressed_size > 0
                                else 0.0
                            )
                            print(
                                f"[extract] {format_bytes(compressed_done)} / "
                                f"{format_bytes(total_compressed_size)} "
                                f"({pct:.1f}%) | members={extracted_members:,} | "
                                f"logical_extracted={format_bytes(extracted_logical_bytes)}"
                            )
                            last_print_time = now

        print(
            f"[extract] completed | members={extracted_members:,} | "
            f"logical_extracted={format_bytes(extracted_logical_bytes)}"
        )

    def _reused_s2_root_looks_usable(
            self,
            *,
            s2_root: Path,
            metadata_path: Path,
            probe_count: int = 5,
            seed: int = 42,
    ) -> bool:
        if not s2_root.is_dir():
            print(f"[reuse] extracted tree is not a directory: {s2_root}")
            return False

        try:
            has_any_child_dir = any(child.is_dir() for child in s2_root.iterdir())
        except OSError as exc:
            print(f"[reuse] failed listing extracted tree {s2_root}: {exc}")
            return False

        if not has_any_child_dir:
            print(f"[reuse] extracted tree looks empty: {s2_root}")
            return False

        try:
            probe_patch_ids = self._sample_patch_ids_for_sanity_check(
                metadata_path=metadata_path,
                probe_count=probe_count,
                seed=seed,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[reuse] failed reading metadata for extracted-tree sanity check: {exc}")
            return False

        if not probe_patch_ids:
            print("[reuse] no valid patch_ids available for extracted-tree sanity check")
            return False

        for patch_id in probe_patch_ids:
            try:
                patch_dir = self._resolve_patch_dir(s2_root=s2_root, patch_id=patch_id)
            except Exception as exc:  # noqa: BLE001
                print(f"[reuse] patch dir probe failed for patch_id={patch_id}: {exc}")
                return False

            required_files = [
                patch_dir / f"{patch_id}_B04.tif",
                patch_dir / f"{patch_id}_B03.tif",
                patch_dir / f"{patch_id}_B02.tif",
            ]

            missing = [str(path.name) for path in required_files if not path.is_file()]
            if missing:
                print(
                    f"[reuse] extracted-tree probe failed for patch_id={patch_id}; "
                    f"missing files: {missing}"
                )
                return False

        print(
            f"[reuse] extracted tree passed sanity check at {s2_root} "
            f"using {len(probe_patch_ids)} probe patch(es)"
        )
        return True

    def _sample_patch_ids_for_sanity_check(
            self,
            *,
            metadata_path: Path,
            probe_count: int = 5,
            seed: int = 42,
    ) -> list[str]:
        """
        Deterministically sample a small set of patch_ids from metadata.parquet
        without loading the entire table into memory just for the sanity check.

        Uses reservoir sampling so the probes are spread through the dataset
        instead of always coming from the beginning.
        """
        rng = random.Random(seed)
        sample: list[str] = []
        seen = 0

        parquet_file = pq.ParquetFile(metadata_path)
        for batch in parquet_file.iter_batches(batch_size=2048, columns=["patch_id"]):
            for patch_id in batch.column(0).to_pylist():
                if not isinstance(patch_id, str):
                    continue
                patch_id = patch_id.strip()
                if not patch_id:
                    continue

                seen += 1
                if len(sample) < probe_count:
                    sample.append(patch_id)
                else:
                    j = rng.randrange(seen)
                    if j < probe_count:
                        sample[j] = patch_id

        return sample

    def _filter_rows_for_split(
            self,
            rows: list[dict[str, Any]],
            requested_split: str | None) -> list[dict[str, Any]]:

        if requested_split is None:
            raise ValueError("bigearthnet-v2 requires config.split to be set")

        filtered = [row for row in rows if row.get("split") == requested_split]

        if not filtered:
            raise ValueError(
                f"No BigEarthNet rows found for split={requested_split}"
            )

        return filtered

    def _normalize_bigearthnet_split(self, value: Any, *, row_idx: int, patch_id: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"metadata row {row_idx}: missing split for patch_id={patch_id}")

        raw = value.strip().lower()

        alias_map = {
            "train": "train",
            "training": "train",
            "val": "val",
            "valid": "val",
            "validation": "val",
            "test": "test",
            "testing": "test",
        }

        normalized = alias_map.get(raw)
        if normalized is None:
            raise ValueError(
                f"metadata row {row_idx}: unsupported split value for patch_id={patch_id}: {value!r}"
            )

        return normalized