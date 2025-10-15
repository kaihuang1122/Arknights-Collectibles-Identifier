import cv2
import numpy as np
from pathlib import Path
from multiprocessing import Pool
import json
import argparse

FALLBACK_MIN_MATCH_COUNT = 1
RELAXED_INLIER_MARGIN = 2
TEMPLATE_FEATURE_CACHE = {}
LOOSE_RATIO_THRESHOLD = 0.9
SLOT_MARGIN_RATIO = 0.25


def start_match():
    global img, kp_img, des_img, flann, sift
    img = cv2.imread(args.img_path)
    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = img
    sift = cv2.SIFT_create()
    kp_img, des_img = sift.detectAndCompute(gray, None)
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=4)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

def match_one(template_path: str, min_match_count: int):
    template_path = str(template_path)
    tpl = cv2.imread(template_path)
    if tpl is None:
        return {
            "success": False,
            "match_count": 0,
            "good_count": 0,
            "points": [],
            "bbox": None,
            "centroid": None,
            "template_path": template_path,
            "min_match_count": min_match_count,
        }
    kp_tpl, des_tpl = sift.detectAndCompute(tpl, None)
    if des_tpl is None:
        return {
            "success": False,
            "match_count": 0,
            "good_count": 0,
            "points": [],
            "bbox": None,
            "centroid": None,
            "template_path": template_path,
            "min_match_count": min_match_count,
        }

    matches = flann.knnMatch(des_tpl, des_img, k=2)
    good_matches = []
    for match_pair in matches:
        if len(match_pair) < 2:
            continue
        m, n = match_pair
        if m.distance < 0.6 * n.distance:
            good_matches.append(m)

    if len(good_matches):
        pts_tpl = np.float32([kp_tpl[m.queryIdx].pt for m in good_matches]).reshape(-1, 2)
        pts_img = np.float32([kp_img[m.trainIdx].pt for m in good_matches]).reshape(-1, 2)
        M, mask = cv2.estimateAffinePartial2D(
            pts_tpl, pts_img,
            method=cv2.RANSAC,
            ransacReprojThreshold=3,
            maxIters=2000
        )
        if M is None:
            inlier_matches = []
            inlier_pts_img = np.empty((0, 2), dtype=np.float32)
        else:
            a, b = M[0, 0], M[0, 1]
            theta = np.degrees(np.arctan2(b, a))
            if abs(theta) < 1.0:
                inlier_matches = [good for good, m in zip(good_matches, mask.ravel()) if m]
                inlier_pts_img = np.float32([kp_img[m.trainIdx].pt for m in inlier_matches]).reshape(-1, 2)
            else:
                inlier_matches = []
                inlier_pts_img = np.empty((0, 2), dtype=np.float32)
    else:
        inlier_matches = []
        inlier_pts_img = np.empty((0, 2), dtype=np.float32)

    if inlier_pts_img.size:
        centroid_x, centroid_y = np.mean(inlier_pts_img, axis=0)
        min_x, min_y = np.min(inlier_pts_img, axis=0)
        max_x, max_y = np.max(inlier_pts_img, axis=0)
        bbox = (float(min_x), float(min_y), float(max_x), float(max_y))
        centroid = (float(centroid_x), float(centroid_y))
        matched_points = [(float(x), float(y)) for x, y in inlier_pts_img]
    else:
        bbox = None
        centroid = None
        matched_points = []

    success = len(inlier_matches) >= min_match_count
    return {
        "success": success,
        "match_count": len(inlier_matches),
        "good_count": len(good_matches),
        "points": matched_points,
        "bbox": bbox,
        "centroid": centroid,
        "template_path": template_path,
        "min_match_count": min_match_count,
    }


def relax_near_threshold(results, margin=RELAXED_INLIER_MARGIN):
    relaxed = []
    for result in results:
        if result["success"]:
            continue
        min_required = result["min_match_count"]
        if min_required <= FALLBACK_MIN_MATCH_COUNT:
            continue
        relaxed_threshold = max(FALLBACK_MIN_MATCH_COUNT, min_required - margin)
        if result["match_count"] >= relaxed_threshold and result["match_count"] > 0:
            result["success"] = True
            result["promoted"] = True
            result["promotion_reason"] = "relaxed"
            result["effective_min_match_count"] = relaxed_threshold
            relaxed.append(result)
    return relaxed


def load_template_features(template_path: str):
    template_path = str(template_path)
    cached = TEMPLATE_FEATURE_CACHE.get(template_path)
    if cached is not None:
        return cached
    tpl = cv2.imread(template_path)
    if tpl is None:
        TEMPLATE_FEATURE_CACHE[template_path] = (None, None)
        return TEMPLATE_FEATURE_CACHE[template_path]
    _, des_tpl = sift.detectAndCompute(tpl, None)
    TEMPLATE_FEATURE_CACHE[template_path] = (tpl, des_tpl)
    return TEMPLATE_FEATURE_CACHE[template_path]


def find_missing_slots(answers, layout):
    present = {}
    for obj in answers:
        present.setdefault(obj["row"], set()).add(obj["col"])
    missing = []
    for row_idx, count in layout.items():
        for col_idx in range(count):
            if col_idx not in present.get(row_idx, set()):
                missing.append((row_idx, col_idx))
    return missing


def estimate_slot_region(row_objects, missing_col, image_shape):
    if not row_objects:
        return None, None
    valid_objects = [obj for obj in row_objects if obj.get("bbox")]
    if not valid_objects:
        return None, None
    sorted_objs = sorted(valid_objects, key=lambda o: o["col"])
    widths = [obj["bbox"][2] - obj["bbox"][0] for obj in sorted_objs]
    default_width = float(np.median(widths)) if widths else image_shape[1] / max(len(sorted_objs), 1)

    left = None
    right = None
    for obj in reversed(sorted_objs):
        if obj["col"] < missing_col:
            left = obj
            break
    for obj in sorted_objs:
        if obj["col"] > missing_col:
            right = obj
            break

    def center_and_width(target):
        bbox = target["bbox"]
        width = bbox[2] - bbox[0]
        center_x = (bbox[0] + bbox[2]) / 2.0
        return center_x, width

    if left and right:
        left_center, left_width = center_and_width(left)
        right_center, right_width = center_and_width(right)
        center_x = (left_center + right_center) / 2.0
        width = max((left_width + right_width) / 2.0, default_width)
    elif left:
        left_center, left_width = center_and_width(left)
        center_x = left_center + left_width
        width = max(left_width, default_width)
    elif right:
        right_center, right_width = center_and_width(right)
        center_x = right_center - right_width
        width = max(right_width, default_width)
    else:
        width = default_width
        center_x = width / 2.0

    half_width = width * (0.5 + SLOT_MARGIN_RATIO)
    h, w = image_shape[:2]
    x1 = max(0.0, center_x - half_width)
    x2 = min(float(w), center_x + half_width)

    row_min_y = min(obj["bbox"][1] for obj in valid_objects)
    row_max_y = max(obj["bbox"][3] for obj in valid_objects)
    row_height = row_max_y - row_min_y if row_max_y > row_min_y else width
    y_margin = row_height * SLOT_MARGIN_RATIO
    y1 = max(0.0, row_min_y - y_margin)
    y2 = min(float(h), row_max_y + y_margin)

    x1_int, x2_int = int(round(x1)), int(round(x2))
    y1_int, y2_int = int(round(y1)), int(round(y2))

    if x2_int <= x1_int:
        x2_int = min(w, x1_int + int(round(default_width)))
    if y2_int <= y1_int:
        y2_int = min(h, y1_int + int(round(row_height if row_height > 0 else default_width)))

    x2_int = min(w, x2_int)
    y2_int = min(h, y2_int)

    if x2_int <= x1_int or y2_int <= y1_int:
        return None, None

    return (x1_int, y1_int, x2_int, y2_int), (x2_int - x1_int) * (y2_int - y1_int)


def crop_image_region(image, bbox):
    if bbox is None:
        return None
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]


def loose_match_region(region_img, templates, ratio_threshold=LOOSE_RATIO_THRESHOLD, excluded_names=None):
    if region_img is None or region_img.size == 0:
        return None
    kp_region, des_region = sift.detectAndCompute(region_img, None)
    if des_region is None:
        return None
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=6)
    search_params = dict(checks=50)
    matcher = cv2.FlannBasedMatcher(index_params, search_params)
    best = None
    for template in templates:
        if excluded_names and template["name"] in excluded_names:
            continue
        tpl_img, des_tpl = load_template_features(template["path"])
        if des_tpl is None:
            continue
        matches = matcher.knnMatch(des_tpl, des_region, k=2)
        good = []
        for pair in matches:
            if len(pair) < 2:
                continue
            m, n = pair
            if m.distance < ratio_threshold * n.distance:
                good.append(m)
        if not good:
            continue
        match_count = len(good)
        avg_distance = float(np.mean([m.distance for m in good]))
        if best is None or match_count > best["match_count"] or (match_count == best["match_count"] and avg_distance < best["avg_distance"]):
            best = {
                "name": template["name"],
                "path": template["path"],
                "match_count": match_count,
                "avg_distance": avg_distance,
                "good_total": len(matches),
            }
    return best


def assign_column_indices(columns_sorted):
    if not columns_sorted:
        return []

    positions = [col["mean_x"] for col in columns_sorted]
    widths = [col["max_x"] - col["min_x"] for col in columns_sorted]
    default_spacing = float(np.median(widths)) if widths else 40.0

    diffs = [positions[i + 1] - positions[i] for i in range(len(positions) - 1)]
    if diffs:
        spacing = float(np.median(diffs))
        if spacing < default_spacing * 0.5:
            spacing = default_spacing
    else:
        spacing = default_spacing
    spacing = max(spacing, 1.0)

    assignments = []
    current_col = 0
    assignments.append((columns_sorted[0], current_col))
    for idx in range(1, len(columns_sorted)):
        gap = positions[idx] - positions[idx - 1]
        estimated_slots = int(round(gap / spacing))
        if estimated_slots < 1:
            estimated_slots = 1
        current_col += estimated_slots
        assignments.append((columns_sorted[idx], current_col))
    return assignments


def cluster_collectible_locations(results):
    candidates = []
    for result in results:
        if not result.get("points"):
            continue
        candidates.append({
            "name": result["name"],
            "path": result["path"],
            "count": result["match_count"],
            "good": result["good_count"],
            "success": result["success"],
            "centroid": result["centroid"],
            "bbox": result["bbox"],
            "min_match_count": result["min_match_count"],
            "effective_min_match_count": result.get("effective_min_match_count", result["min_match_count"]),
            "promoted": result.get("promoted", False),
            "promotion_reason": result.get("promotion_reason"),
            "result": result,
        })

    if not candidates:
        return []

    heights = [cand["bbox"][3] - cand["bbox"][1] for cand in candidates]
    widths = [cand["bbox"][2] - cand["bbox"][0] for cand in candidates]
    avg_height = float(np.median(heights)) if heights else 40.0
    avg_width = float(np.median(widths)) if widths else 40.0
    row_tol = max(20.0, avg_height * 0.6)
    col_tol = max(20.0, avg_width * 0.6)

    row_clusters = []
    for candidate in sorted(candidates, key=lambda c: c["centroid"][1]):
        y = candidate["centroid"][1]
        target_row = None
        for row in row_clusters:
            if abs(y - row["mean_y"]) <= row_tol:
                target_row = row
                break
        if target_row is None:
            target_row = {
                "items": [],
                "sum_y": 0.0,
                "mean_y": y,
                "min_y": candidate["bbox"][1],
                "max_y": candidate["bbox"][3],
                "columns": [],
            }
            row_clusters.append(target_row)
        target_row["items"].append(candidate)
        target_row["sum_y"] += y
        target_row["mean_y"] = target_row["sum_y"] / len(target_row["items"])
        target_row["min_y"] = min(target_row["min_y"], candidate["bbox"][1])
        target_row["max_y"] = max(target_row["max_y"], candidate["bbox"][3])

    grouped_objects = []

    for row_index, row in enumerate(sorted(row_clusters, key=lambda r: r["mean_y"])):
        columns = []
        for candidate in sorted(row["items"], key=lambda c: c["centroid"][0]):
            x = candidate["centroid"][0]
            target_col = None
            for col in columns:
                if abs(x - col["mean_x"]) <= col_tol:
                    target_col = col
                    break
            if target_col is None:
                target_col = {
                    "items": [],
                    "sum_x": 0.0,
                    "mean_x": x,
                    "min_x": candidate["bbox"][0],
                    "max_x": candidate["bbox"][2],
                    "min_y": candidate["bbox"][1],
                    "max_y": candidate["bbox"][3],
                }
                columns.append(target_col)
            target_col["items"].append(candidate)
            target_col["sum_x"] += x
            target_col["mean_x"] = target_col["sum_x"] / len(target_col["items"])
            target_col["min_x"] = min(target_col["min_x"], candidate["bbox"][0])
            target_col["max_x"] = max(target_col["max_x"], candidate["bbox"][2])
            target_col["min_y"] = min(target_col["min_y"], candidate["bbox"][1])
            target_col["max_y"] = max(target_col["max_y"], candidate["bbox"][3])
        sorted_columns = sorted(columns, key=lambda c: c["mean_x"])
        assignments = assign_column_indices(sorted_columns)
        for col, col_index in assignments:
            for candidate in col["items"]:
                candidate["col"] = col_index
                candidate["row"] = row_index
            best_candidate = max(
                col["items"],
                key=lambda c: (
                    1 if c["success"] else 0,
                    c["count"],
                    c["good"],
                )
            )
            grouped_objects.append({
                "row": row_index,
                "col": col_index,
                "bbox": (
                    float(col["min_x"]),
                    float(col["min_y"]),
                    float(col["max_x"]),
                    float(col["max_y"]),
                ),
                "best": best_candidate,
                "candidates": col["items"],
                "promoted": best_candidate.get("promoted", False),
                "promotion_reason": best_candidate.get("promotion_reason"),
                "effective_min_match_count": best_candidate.get("effective_min_match_count", best_candidate["min_match_count"]),
            })
    return grouped_objects


def promote_gap_collectibles(grouped_objects, fallback_threshold=FALLBACK_MIN_MATCH_COUNT):
    if not grouped_objects:
        return grouped_objects
    fallback_threshold = max(1, fallback_threshold)
    rows = {}
    for obj in grouped_objects:
        rows.setdefault(obj["row"], []).append(obj)

    for row_objects in rows.values():
        ordered = sorted(row_objects, key=lambda o: o["col"])
        success_flags = [item["best"]["success"] for item in ordered]
        for idx, obj in enumerate(ordered):
            best = obj["best"]
            if best["success"]:
                continue
            if best["count"] <= 0:
                continue
            has_left = any(success_flags[:idx])
            has_right = any(success_flags[idx + 1:])
            if not (has_left and has_right):
                continue
            target_threshold = max(fallback_threshold, min(best["min_match_count"], best["count"]))
            if best["count"] >= target_threshold:
                best["success"] = True
                best["promoted"] = True
                best["promotion_reason"] = best.get("promotion_reason") or "gap"
                best["effective_min_match_count"] = target_threshold
                if "result" in best:
                    best["result"]["success"] = True
                    best["result"]["promoted"] = True
                    best["result"]["promotion_reason"] = best["promotion_reason"]
                    best["result"]["effective_min_match_count"] = target_threshold
                obj["promoted"] = True
                obj["promotion_reason"] = best["promotion_reason"]
                obj["effective_min_match_count"] = target_threshold
                success_flags[idx] = True
    return grouped_objects


def truncate_rows_with_empty(grouped_objects):
    rows = {}
    for obj in grouped_objects:
        rows.setdefault(obj["row"], []).append(obj)

    filtered = []
    for row_index, row_objects in sorted(rows.items()):
        ordered = sorted(row_objects, key=lambda o: o["col"])
        cutoff = len(ordered)
        for idx, obj in enumerate(ordered):
            best = obj["best"]
            is_empty_slot = not best["success"]
            if not is_empty_slot:
                continue
            trailing = ordered[idx + 1:]
            if trailing and all(candidate["best"]["count"] < 5 for candidate in trailing):
                cutoff = idx
                break
        filtered.extend(ordered[:cutoff])
    return filtered


def show_selected_collectibles(selected_objects):
    if not selected_objects:
        return
    display = img.copy()
    for obj in selected_objects:
        bbox = obj.get("bbox")
        if not bbox:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        reason = obj.get("promotion_reason")
        color = (0, 255, 0)
        if reason == "gap":
            color = (0, 165, 255)
        elif reason == "relaxed":
            color = (255, 255, 0)
        cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
        if reason == "gap":
            label_suffix = "*"
        elif reason == "relaxed":
            label_suffix = "~"
        else:
            label_suffix = ""
        label = f"R{obj['row'] + 1}C{obj['col'] + 1}:{obj['best']['count']}{label_suffix}"
        text_origin = (x1, max(0, y1 - 8))
        cv2.putText(display, label, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    cv2.imshow('Collectibles', display)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# print(match_one(Path(template_dir)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Match collectibles in a screenshot.')
    parser.add_argument('--img_path', type=str, required=True, help='Directory containing collectible templates.')
    parser.add_argument('--IS', type=int, default=6, help='IS version (default: Sui\'s Garden of Grotesqueries)')
    parser.add_argument('--show', action='store_true', help='Show matching results.')
    parser.add_argument('--show-metrics', action='store_true', help='Display detailed matching metrics.')
    args = parser.parse_args()
    if args.IS > 6 or args.IS < 1:
        print("IS must be 1 to 6")
        exit(1)
    path = f"./assets/IS{args.IS}_Collectibles.json"
    SHOW = args.show
    SHOW_METRICS = args.show_metrics
    start_match()
    with open(path, 'r', encoding='utf-8') as f:
        collectibles = json.load(f)
    match_results = []
    for item in collectibles:
        item["path"] = Path(item['img_url'][item['img_url'].find("assets/"):])
        template_path = str(item["path"])
        min_match_count = 5
        match_result = match_one(template_path, min_match_count)
        match_result["name"] = item["name"]
        match_result["path"] = template_path
        match_results.append(match_result)

    relax_near_threshold(match_results)
    grouped_objects = cluster_collectible_locations(match_results)
    promote_gap_collectibles(grouped_objects)
    filtered_objects = truncate_rows_with_empty(grouped_objects)
    answers = [obj for obj in filtered_objects if obj["best"]["success"]]
    if not answers:
        print("No collectibles identified via grouping.")
    else:
        for obj in answers:
            row_index = obj["row"] + 1
            col_index = obj["col"] + 1
            best = obj["best"]
            reason = obj.get("promotion_reason")
            if reason == "gap":
                tag = " (gap fallback)"
            elif reason == "relaxed":
                tag = " (relaxed threshold)"
            else:
                tag = ""
            effective_min = obj.get(
                "effective_min_match_count",
                best.get("effective_min_match_count", best["min_match_count"])
            )
            metrics_str = ""
            if SHOW_METRICS:
                metrics_str = (
                    f" (matches: {best['count']}, good: {best['good']}, "
                    f"threshold: {effective_min})"
                )
            print(f"Row {row_index} Col {col_index}: {best['name']}{tag}{metrics_str}")
        print(f"{len(answers)} collectibles selected after grouping.")
        if SHOW:
            show_selected_collectibles(answers)
    if SHOW and not answers:
        print("No answers to display.")

    expected_layout = {0: 10, 1: 10, 2: 9}
    row_map = {}
    for obj in answers:
        row_map.setdefault(obj["row"], []).append(obj)
    missing_slots = find_missing_slots(answers, expected_layout)
    if missing_slots:
        suggestions = []
        existing_names = {obj["best"]["name"] for obj in answers}
        for row_idx, col_idx in missing_slots:
            bbox, area = estimate_slot_region(row_map.get(row_idx, []), col_idx, img.shape)
            if bbox is None:
                continue
            region_img = crop_image_region(img, bbox)
            prediction = loose_match_region(region_img, match_results, excluded_names=existing_names)
            if prediction:
                prediction.update({
                    "row": row_idx,
                    "col": col_idx,
                    "bbox": bbox,
                    "area": area,
                })
                suggestions.append(prediction)
                existing_names.add(prediction["name"])
        if suggestions:
            print("\nLoose-match suggestions for missing slots:")
            for suggestion in sorted(suggestions, key=lambda s: (s["row"], s["col"])):
                row_idx = suggestion["row"] + 1
                col_idx = suggestion["col"] + 1
                print(
                    f"Row {row_idx} Col {col_idx}: {suggestion['name']} "
                    f"(loose matches: {suggestion['match_count']}, avg distance: {suggestion['avg_distance']:.2f})"
                )
        else:
            print("\nNo loose-match suggestions could be produced for missing slots.")
