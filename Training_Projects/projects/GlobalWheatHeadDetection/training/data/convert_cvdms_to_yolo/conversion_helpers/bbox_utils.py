from dataclasses import dataclass
from decimal import Decimal, InvalidOperation, ROUND_HALF_UP
from typing import Any


YOLO_COORD_QUANT = Decimal("0.000001")


@dataclass(frozen=True)
class CvdmsBox:
    top: Decimal
    left: Decimal
    height: Decimal
    width: Decimal


@dataclass(frozen=True)
class YoloBox:
    x_center: Decimal
    y_center: Decimal
    width: Decimal
    height: Decimal


def parse_finite_decimal(value: Any, field_name: str) -> Decimal:
    """
    Parse a CVDMS bbox numeric field into a finite Decimal.

    CVDMS bbox JSON stores values as strings such as "92.0000",
    but this also accepts ints/floats for robustness.
    """
    if value is None:
        raise ValueError(f"Missing required bbox field: {field_name}")

    try:
        decimal_value = Decimal(str(value))
    except (InvalidOperation, ValueError) as exc:
        raise ValueError(f"Invalid decimal value for {field_name}: {value!r}") from exc

    if not decimal_value.is_finite():
        raise ValueError(f"Non-finite decimal value for {field_name}: {value!r}")

    return decimal_value


def parse_positive_image_dimension(value: Any, field_name: str) -> Decimal:
    """
    Parse image width/height and ensure it is positive.
    """
    decimal_value = parse_finite_decimal(value, field_name)

    if decimal_value <= 0:
        raise ValueError(f"{field_name} must be positive, got: {value!r}")

    return decimal_value


def cvdms_annotation_to_box(annotation: dict[str, Any]) -> CvdmsBox:
    """
    Convert one CVDMS annotation dictionary into a CvdmsBox.

    Expected annotation shape:

        {
            "class_name": "wheat_head",
            "top": "0.0000",
            "left": "0.0000",
            "height": "92.0000",
            "width": "40.0000"
        }
    """
    top = parse_finite_decimal(annotation.get("top"), "top")
    left = parse_finite_decimal(annotation.get("left"), "left")
    height = parse_finite_decimal(annotation.get("height"), "height")
    width = parse_finite_decimal(annotation.get("width"), "width")

    if height <= 0:
        raise ValueError(f"Bounding box height must be positive, got: {height}")

    if width <= 0:
        raise ValueError(f"Bounding box width must be positive, got: {width}")

    return CvdmsBox(
        top=top,
        left=left,
        height=height,
        width=width,
    )


def clip_cvdms_box_to_image(
    box: CvdmsBox,
    image_width: int | float | Decimal,
    image_height: int | float | Decimal,
) -> CvdmsBox | None:
    """
    Clip a CVDMS top/left/height/width box to image boundaries.

    Returns None if the box has no remaining positive area after clipping.
    """
    img_w = parse_positive_image_dimension(image_width, "image_width")
    img_h = parse_positive_image_dimension(image_height, "image_height")

    x1 = box.left
    y1 = box.top
    x2 = box.left + box.width
    y2 = box.top + box.height

    clipped_x1 = max(Decimal("0"), min(x1, img_w))
    clipped_y1 = max(Decimal("0"), min(y1, img_h))
    clipped_x2 = max(Decimal("0"), min(x2, img_w))
    clipped_y2 = max(Decimal("0"), min(y2, img_h))

    clipped_width = clipped_x2 - clipped_x1
    clipped_height = clipped_y2 - clipped_y1

    if clipped_width <= 0 or clipped_height <= 0:
        return None

    return CvdmsBox(
        top=clipped_y1,
        left=clipped_x1,
        height=clipped_height,
        width=clipped_width,
    )


def cvdms_box_to_yolo_box(
    box: CvdmsBox,
    image_width: int | float | Decimal,
    image_height: int | float | Decimal,
    clip_to_image: bool = True,
) -> YoloBox | None:
    """
    Convert a CVDMS pixel-space box into YOLO normalized xywh format.

    CVDMS format:
        top, left, height, width

    YOLO format:
        x_center, y_center, width, height

    All YOLO coordinates are normalized to [0, 1].
    """
    img_w = parse_positive_image_dimension(image_width, "image_width")
    img_h = parse_positive_image_dimension(image_height, "image_height")

    working_box = box

    if clip_to_image:
        clipped = clip_cvdms_box_to_image(box, img_w, img_h)
        if clipped is None:
            return None
        working_box = clipped

    x_center = (working_box.left + (working_box.width / Decimal("2"))) / img_w
    y_center = (working_box.top + (working_box.height / Decimal("2"))) / img_h
    norm_width = working_box.width / img_w
    norm_height = working_box.height / img_h

    yolo_box = YoloBox(
        x_center=x_center,
        y_center=y_center,
        width=norm_width,
        height=norm_height,
    )

    if not is_valid_yolo_box(yolo_box):
        return None

    return yolo_box


def is_valid_yolo_box(box: YoloBox) -> bool:
    """
    Validate normalized YOLO box coordinates.
    """
    values = [box.x_center, box.y_center, box.width, box.height]

    if any(not value.is_finite() for value in values):
        return False

    if box.width <= 0 or box.height <= 0:
        return False

    if box.x_center < 0 or box.x_center > 1:
        return False

    if box.y_center < 0 or box.y_center > 1:
        return False

    if box.width > 1 or box.height > 1:
        return False

    return True


def quantize_yolo_value(value: Decimal) -> Decimal:
    """
    Quantize YOLO coordinate values to 6 decimal places.
    """
    return value.quantize(YOLO_COORD_QUANT, rounding=ROUND_HALF_UP)


def format_yolo_value(value: Decimal) -> str:
    """
    Format a YOLO coordinate as a stable decimal string.
    """
    quantized = quantize_yolo_value(value)

    if quantized == Decimal("-0.000000"):
        quantized = Decimal("0.000000")

    return f"{quantized:.6f}"


def format_yolo_label_row(class_id: int, box: YoloBox) -> str:
    """
    Format one YOLO label row:

        class_id x_center y_center width height
    """
    if class_id < 0:
        raise ValueError(f"class_id must be non-negative, got: {class_id}")

    if not is_valid_yolo_box(box):
        raise ValueError(f"Invalid YOLO box: {box}")

    return (
        f"{class_id} "
        f"{format_yolo_value(box.x_center)} "
        f"{format_yolo_value(box.y_center)} "
        f"{format_yolo_value(box.width)} "
        f"{format_yolo_value(box.height)}"
    )


def convert_cvdms_annotation_to_yolo_row(
    annotation: dict[str, Any],
    class_to_id: dict[str, int],
    image_width: int | float | Decimal,
    image_height: int | float | Decimal,
    clip_to_image: bool = True,
) -> str | None:
    """
    Convert one raw CVDMS annotation dictionary into one YOLO label row.

    Returns None when:
      - the annotation class is outside the selected dataset class set
      - the bbox becomes invalid after clipping

    Raises ValueError when required bbox fields are malformed.
    """
    class_name = annotation.get("class_name")

    if not isinstance(class_name, str) or not class_name.strip():
        raise ValueError(f"Invalid annotation class_name: {class_name!r}")

    normalized_class_name = class_name.strip()

    if normalized_class_name not in class_to_id:
        return None

    cvdms_box = cvdms_annotation_to_box(annotation)
    yolo_box = cvdms_box_to_yolo_box(
        box=cvdms_box,
        image_width=image_width,
        image_height=image_height,
        clip_to_image=clip_to_image,
    )

    if yolo_box is None:
        return None

    class_id = class_to_id[normalized_class_name]
    return format_yolo_label_row(class_id, yolo_box)