"""
Common mosaic-generation utilities for CVDMS training projects.

This module is intentionally task-agnostic. It does not know about single-label,
multi-label, object-detection, or segmentation semantics. It only handles the
shared mechanics:

    MosaicItem(source_ref=...)
    -> image_loader(source_ref)
    -> resize/pad into fixed-size tiles
    -> arrange tiles into an R x C grid
    -> save one or more mosaic sheets

Task-specific modules such as mosaic_generators.multi_label should decide how
to order, group, and name mosaic sets.
"""

from dataclasses import dataclass, field
from math import ceil
from pathlib import Path
from typing import Any, Callable, Sequence, TypeVar, cast

from PIL import Image, ImageOps

ImageLoaderFn = Callable[[str], Image.Image]
Color = tuple[int, int, int]
T = TypeVar("T")

@dataclass(frozen=True)
class MosaicItem:
    """
    One image reference to render into a mosaic.

    Args:
        source_ref:
            Image URI/path to pass into the configured image loader. In CVDMS
            projects this is usually the manifest row's source_ref.
        metadata:
            Optional task-specific metadata. Common rendering code ignores this,
            but task-specific modules may keep labels, split, image_id, etc. here.
    """

    source_ref: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.source_ref, str):
            raise TypeError(f"source_ref must be a string, got {type(self.source_ref).__name__}")

        if not self.source_ref.strip():
            raise ValueError("source_ref cannot be empty")

        if not isinstance(self.metadata, dict):
            raise TypeError(f"metadata must be a dictionary, got {type(self.metadata).__name__}")

@dataclass(frozen=True)
class MosaicConfig:
    """
    Rendering configuration for mosaic sheets.

    Args:
        rows:
            Number of tile rows per mosaic sheet.
        cols:
            Number of tile columns per mosaic sheet.
        tile_width:
            Width of each tile in pixels.
        tile_height:
            Height of each tile in pixels.
        background_color:
            RGB color used for padding around aspect-ratio-preserved images.
        image_mode:
            PIL mode to convert images into before resizing. "RGB" is usually
            correct for classification mosaics.
        output_format:
            PIL output format, usually "PNG".
        filename_extension:
            File extension used when saving sheets.
    """

    rows: int = 10
    cols: int = 10
    tile_width: int = 128
    tile_height: int = 128
    background_color: Color = (0, 0, 0)
    image_mode: str | None = "RGB"
    output_format: str = "PNG"
    filename_extension: str = "png"

    def __post_init__(self) -> None:
        _validate_positive_int(self.rows, "rows")
        _validate_positive_int(self.cols, "cols")
        _validate_positive_int(self.tile_width, "tile_width")
        _validate_positive_int(self.tile_height, "tile_height")
        _validate_color(self.background_color, "background_color")

        if self.image_mode is not None and not str(self.image_mode).strip():
            raise ValueError("image_mode cannot be empty when provided")

        if not str(self.output_format).strip():
            raise ValueError("output_format cannot be empty")

        if not str(self.filename_extension).strip():
            raise ValueError("filename_extension cannot be empty")

    @property
    def tiles_per_sheet(self) -> int:
        return self.rows * self.cols

    @property
    def canvas_width(self) -> int:
        return self.cols * self.tile_width

    @property
    def canvas_height(self) -> int:
        return self.rows * self.tile_height

    @property
    def canvas_size(self) -> tuple[int, int]:
        return self.canvas_width, self.canvas_height

    @property
    def tile_size(self) -> tuple[int, int]:
        return self.tile_width, self.tile_height

@dataclass(frozen=True)
class MosaicSheetResult:
    """
    Result metadata for one saved mosaic sheet.
    """

    path: Path
    sheet_index: int
    item_count: int
    first_item_index: int
    last_item_index: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": str(self.path),
            "sheet_index": self.sheet_index,
            "item_count": self.item_count,
            "first_item_index": self.first_item_index,
            "last_item_index": self.last_item_index,
        }

def save_mosaic_sheets(
    *,
    items: Sequence[MosaicItem],
    image_loader: ImageLoaderFn,
    output_dir: str | Path,
    filename_prefix: str,
    config: MosaicConfig,
    start_sheet_index: int = 1,
) -> list[MosaicSheetResult]:
    """
    Render and save one or more mosaic sheets.

    Args:
        items:
            Ordered image items to render.
        image_loader:
            Callable that loads a PIL image from item.source_ref.
        output_dir:
            Directory where mosaic sheets are written.
        filename_prefix:
            Filename prefix such as "train__order-cardinality-signature".
            The grid size and sheet number are appended automatically.
        config:
            Mosaic render configuration.
        start_sheet_index:
            First sheet index used in filenames.

    Returns:
        List of saved sheet results.
    """
    if not items:
        raise ValueError("items cannot be empty")

    _validate_positive_int(start_sheet_index, "start_sheet_index")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    safe_prefix = safe_filename_part(filename_prefix)
    results: list[MosaicSheetResult] = []

    for offset, chunk in enumerate(chunk_sequence(items, config.tiles_per_sheet)):
        sheet_index = start_sheet_index + offset
        first_item_index = offset * config.tiles_per_sheet
        last_item_index = first_item_index + len(chunk) - 1

        mosaic = render_mosaic_sheet(
            items=chunk,
            image_loader=image_loader,
            config=config,
        )

        filename = build_mosaic_filename(
            filename_prefix=safe_prefix,
            config=config,
            sheet_index=sheet_index,
        )
        destination = output_path / filename

        mosaic.save(destination, format=config.output_format)

        results.append(
            MosaicSheetResult(
                path=destination,
                sheet_index=sheet_index,
                item_count=len(chunk),
                first_item_index=first_item_index,
                last_item_index=last_item_index,
            )
        )

    return results

def render_mosaic_sheet(
    *,
    items: Sequence[MosaicItem],
    image_loader: ImageLoaderFn,
    config: MosaicConfig,
) -> Image.Image:
    """
    Render one mosaic sheet from up to config.tiles_per_sheet items.

    No text or labels are drawn on the image. Each input image is resized to fit
    within its tile while preserving aspect ratio, then centered on a padded
    background.
    """
    if not items:
        raise ValueError("items cannot be empty")

    if len(items) > config.tiles_per_sheet:
        raise ValueError(
            f"items has {len(items)} entries, but config only allows "
            f"{config.tiles_per_sheet} tiles per sheet"
        )

    canvas = Image.new(
        "RGB",
        config.canvas_size,
        color=config.background_color,
    )

    for item_index, item in enumerate(items):
        row = item_index // config.cols
        col = item_index % config.cols
        x = col * config.tile_width
        y = row * config.tile_height

        tile = load_item_as_tile(
            item=item,
            image_loader=image_loader,
            config=config,
        )
        canvas.paste(tile, (x, y))

    return canvas

def load_item_as_tile(
    *,
    item: MosaicItem,
    image_loader: ImageLoaderFn,
    config: MosaicConfig,
) -> Image.Image:
    """
    Load a MosaicItem and convert it into a padded tile.
    """
    try:
        image = image_loader(item.source_ref)
    except Exception as exc:
        raise RuntimeError(f"Failed to load image for mosaic item: {item.source_ref!r}") from exc

    try:
        return fit_image_to_tile(
            image=image,
            tile_width=config.tile_width,
            tile_height=config.tile_height,
            background_color=config.background_color,
            image_mode=config.image_mode,
        )
    except Exception as exc:
        raise RuntimeError(f"Failed to render image tile for: {item.source_ref!r}") from exc

def fit_image_to_tile(
    *,
    image: Image.Image,
    tile_width: int,
    tile_height: int,
    background_color: Color = (0, 0, 0),
    image_mode: str | None = "RGB",
) -> Image.Image:
    """
    Resize an image to fit within a fixed tile while preserving aspect ratio.

    The resized image is centered on a background-color canvas. This avoids
    cropping and keeps the mosaic faithful to the source imagery.
    """
    _validate_positive_int(tile_width, "tile_width")
    _validate_positive_int(tile_height, "tile_height")
    _validate_color(background_color, "background_color")

    if image.width < 1 or image.height < 1:
        raise ValueError(f"Image dimensions must be positive, got {image.size}")

    working = ImageOps.exif_transpose(image)

    if image_mode is not None:
        working = working.convert(image_mode)

    scale = min(tile_width / working.width, tile_height / working.height)
    resized_width = max(1, int(round(working.width * scale)))
    resized_height = max(1, int(round(working.height * scale)))

    resized = working.resize(
        (resized_width, resized_height),
        resample=_resample_filter(),
    )

    tile = Image.new(
        "RGB",
        (tile_width, tile_height),
        color=background_color,
    )

    if resized.mode != "RGB":
        resized = resized.convert("RGB")

    paste_x = (tile_width - resized_width) // 2
    paste_y = (tile_height - resized_height) // 2
    tile.paste(resized, (paste_x, paste_y))

    return tile

def build_mosaic_filename(
    *,
    filename_prefix: str,
    config: MosaicConfig,
    sheet_index: int,
) -> str:
    """
    Build a standard mosaic filename.

    Example:
        train__order-cardinality-signature__grid-10x10__sheet-001.png
    """
    _validate_positive_int(sheet_index, "sheet_index")

    safe_prefix = safe_filename_part(filename_prefix)
    extension = safe_filename_part(config.filename_extension).lower().lstrip(".")

    return (
        f"{safe_prefix}"
        f"__grid-{config.rows}x{config.cols}"
        f"__sheet-{sheet_index:03d}"
        f".{extension}"
    )

def chunk_sequence(items: Sequence[T], chunk_size: int) -> list[list[T]]:
    """
    Chunk a sequence into list chunks.
    """
    _validate_positive_int(chunk_size, "chunk_size")
    return [
        list(items[start:start + chunk_size])
        for start in range(0, len(items), chunk_size)
    ]

def count_mosaic_sheets(item_count: int, config: MosaicConfig) -> int:
    """
    Return how many mosaic sheets are needed for item_count images.
    """
    if isinstance(item_count, bool) or not isinstance(item_count, int):
        raise TypeError(f"item_count must be an int, got {item_count!r}")

    if item_count < 0:
        raise ValueError(f"item_count must be >= 0, got {item_count}")

    if item_count == 0:
        return 0

    return ceil(item_count / config.tiles_per_sheet)

def make_mosaic_items(
    source_refs: Sequence[str],
    *,
    metadata_by_source_ref: dict[str, dict[str, Any]] | None = None,
) -> list[MosaicItem]:
    """
    Build MosaicItem objects from source_ref strings.
    """
    items: list[MosaicItem] = []

    for source_ref in source_refs:
        metadata = {}
        if metadata_by_source_ref is not None:
            metadata = dict(metadata_by_source_ref.get(source_ref, {}))

        items.append(
            MosaicItem(
                source_ref=source_ref,
                metadata=metadata,
            )
        )

    return items

def safe_filename_part(value: Any) -> str:
    """
    Convert arbitrary text into a conservative filename component.
    """
    text = str(value).strip().lower()
    safe_chars: list[str] = []

    for char in text:
        if char.isalnum():
            safe_chars.append(char)
        elif char in {"-", "_"}:
            safe_chars.append(char)
        elif char.isspace():
            safe_chars.append("_")
        else:
            safe_chars.append("_")

    safe = "".join(safe_chars)

    while "__" in safe:
        safe = safe.replace("__", "_")

    safe = safe.strip("._-")
    return safe or "mosaic"

def _resample_filter() -> int:
    try:
        return cast(int, Image.Resampling.LANCZOS)
    except AttributeError:  # pragma: no cover - older Pillow fallback
        return cast(int, Image.LANCZOS)

def _validate_positive_int(value: Any, field_name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{field_name} must be an int, got {value!r}")

    if value < 1:
        raise ValueError(f"{field_name} must be >= 1, got {value}")

def _validate_color(value: Any, field_name: str) -> None:
    if not isinstance(value, tuple) or len(value) != 3:
        raise TypeError(f"{field_name} must be an RGB tuple of length 3, got {value!r}")

    for idx, channel in enumerate(value):
        if isinstance(channel, bool) or not isinstance(channel, int):
            raise TypeError(
                f"{field_name}[{idx}] must be an int in [0, 255], got {channel!r}"
            )

        if not 0 <= channel <= 255:
            raise ValueError(
                f"{field_name}[{idx}] must be in [0, 255], got {channel}"
            )