from typing import Any


def box_iou(box_a: list[float], box_b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)

    inter_width = max(0.0, inter_x2 - inter_x1)
    inter_height = max(0.0, inter_y2 - inter_y1)
    inter_area = inter_width * inter_height

    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)

    union = area_a + area_b - inter_area

    if union <= 0:
        return 0.0

    return inter_area / union


def match_predictions_to_ground_truth(
    predictions: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    match_iou_threshold: float,
) -> dict[str, Any]:
    matched_gt_indices: set[int] = set()
    matched_predictions: list[dict[str, Any]] = []
    unmatched_predictions: list[dict[str, Any]] = []

    sorted_predictions = sorted(
        predictions,
        key=lambda item: item.get("confidence", 0.0),
        reverse=True,
    )

    for prediction in sorted_predictions:
        best_iou = 0.0
        best_gt_idx: int | None = None

        for gt_idx, gt_box in enumerate(ground_truth):
            if gt_idx in matched_gt_indices:
                continue

            if prediction["class_id"] != gt_box["class_id"]:
                continue

            iou = box_iou(prediction["bbox_xyxy"], gt_box["bbox_xyxy"])

            if iou > best_iou:
                best_iou = iou
                best_gt_idx = gt_idx

        prediction_with_match = dict(prediction)
        prediction_with_match["best_iou"] = best_iou

        if best_gt_idx is not None and best_iou >= match_iou_threshold:
            prediction_with_match["matched"] = True
            prediction_with_match["matched_gt_index"] = best_gt_idx
            matched_gt_indices.add(best_gt_idx)
            matched_predictions.append(prediction_with_match)
        else:
            prediction_with_match["matched"] = False
            prediction_with_match["matched_gt_index"] = None
            unmatched_predictions.append(prediction_with_match)

    missed_ground_truth = [
        item
        for idx, item in enumerate(ground_truth)
        if idx not in matched_gt_indices
    ]

    return {
        "matched_predictions": matched_predictions,
        "unmatched_predictions": unmatched_predictions,
        "missed_ground_truth": missed_ground_truth,
        "matched_count": len(matched_predictions),
        "unmatched_prediction_count": len(unmatched_predictions),
        "missed_ground_truth_count": len(missed_ground_truth),
    }