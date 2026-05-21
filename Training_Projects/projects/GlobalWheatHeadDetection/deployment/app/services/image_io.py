from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import HTTPException, UploadFile
from PIL import Image, UnidentifiedImageError


@dataclass(frozen=True)
class SavedUpload:
    original_filename: str
    content_type: str | None
    path: Path
    image_width: int
    image_height: int
    image_format: str | None


async def save_upload_to_temp_file(
    upload_file: UploadFile,
    temp_upload_dir: str,
) -> SavedUpload:
    raw_bytes = await upload_file.read()

    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    image_width, image_height, image_format = inspect_image(raw_bytes)

    upload_dir = Path(temp_upload_dir)
    upload_dir.mkdir(parents=True, exist_ok=True)

    suffix = resolve_image_suffix(
        filename=upload_file.filename,
        image_format=image_format,
    )

    with NamedTemporaryFile(
        mode="wb",
        suffix=suffix,
        prefix="upload_",
        dir=upload_dir,
        delete=False,
    ) as temp_file:
        temp_file.write(raw_bytes)
        temp_path = Path(temp_file.name)

    return SavedUpload(
        original_filename=upload_file.filename or "uploaded_image",
        content_type=upload_file.content_type,
        path=temp_path,
        image_width=image_width,
        image_height=image_height,
        image_format=image_format,
    )


def inspect_image(raw_bytes: bytes) -> tuple[int, int, str | None]:
    try:
        with Image.open(BytesIO(raw_bytes)) as image:
            image.load()
            width, height = image.size
            image_format = image.format
    except UnidentifiedImageError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded file is not a valid image.",
        ) from exc
    except OSError as exc:
        raise HTTPException(
            status_code=400,
            detail="Uploaded image could not be read.",
        ) from exc

    return width, height, image_format


def resolve_image_suffix(filename: str | None, image_format: str | None) -> str:
    suffix_from_name = Path(filename or "").suffix.lower()

    if suffix_from_name in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        return suffix_from_name

    suffix_by_format = {
        "JPEG": ".jpg",
        "PNG": ".png",
        "BMP": ".bmp",
        "WEBP": ".webp",
    }

    return suffix_by_format.get((image_format or "").upper(), ".jpg")


def delete_file_safely(path: Path | str | None) -> None:
    if path is None:
        return

    try:
        Path(path).unlink(missing_ok=True)
    except OSError:
        pass