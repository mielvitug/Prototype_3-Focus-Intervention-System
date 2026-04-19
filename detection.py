try:
    import cv2
except ImportError:
    cv2 = None

from config import (
    FACE_CASCADE_PATH,
    FACE_FALLBACK_MIN_NEIGHBORS,
    FACE_FALLBACK_MIN_SIZE,
    FACE_MIN_NEIGHBORS,
    FACE_SCALE_FACTOR,
    MIN_FACE_SIZE,
)


MAX_TRACKED_FACES = 2
MERGE_IOU_THRESHOLD = 0.30


def _profile_cascade_path():
    if cv2 is None:
        return None
    return cv2.data.haarcascades + "haarcascade_profileface.xml"


def _frontal_alt_cascade_path():
    if cv2 is None:
        return None
    return cv2.data.haarcascades + "haarcascade_frontalface_alt2.xml"


def _load_classifier(path):
    if not path:
        return None

    classifier = cv2.CascadeClassifier(path)
    if classifier.empty():
        return None
    return classifier


def load_face_detector():
    if cv2 is None:
        return None, "OpenCV is not installed."

    if FACE_CASCADE_PATH is None:
        return None, "Face cascade path is not available."

    frontal = _load_classifier(FACE_CASCADE_PATH)
    if frontal is None:
        return None, "Could not load the face detector."

    detector = {
        "frontal": frontal,
        "frontal_alt": _load_classifier(_frontal_alt_cascade_path()),
        "profile": _load_classifier(_profile_cascade_path()),
    }
    return detector, None


def _box_area(box):
    _, _, width, height = box
    return max(0, width) * max(0, height)


def _to_corners(box):
    x, y, width, height = box
    return x, y, x + width, y + height


def _iou(box_a, box_b):
    left_a, top_a, right_a, bottom_a = _to_corners(box_a)
    left_b, top_b, right_b, bottom_b = _to_corners(box_b)

    inter_left = max(left_a, left_b)
    inter_top = max(top_a, top_b)
    inter_right = min(right_a, right_b)
    inter_bottom = min(bottom_a, bottom_b)

    inter_width = max(0, inter_right - inter_left)
    inter_height = max(0, inter_bottom - inter_top)
    intersection = inter_width * inter_height
    if intersection == 0:
        return 0.0

    union = _box_area(box_a) + _box_area(box_b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def _merge_box_pair(box_a, box_b):
    x1, y1, w1, h1 = box_a
    x2, y2, w2, h2 = box_b
    left = min(x1, x2)
    top = min(y1, y2)
    right = max(x1 + w1, x2 + w2)
    bottom = max(y1 + h1, y2 + h2)
    return (left, top, right - left, bottom - top)


def _merge_face_boxes(boxes):
    merged = []

    for box in boxes:
        current = tuple(int(v) for v in box)
        merged_into_existing = False

        for index, existing in enumerate(merged):
            if _iou(current, existing) >= MERGE_IOU_THRESHOLD:
                merged[index] = _merge_box_pair(existing, current)
                merged_into_existing = True
                break

        if not merged_into_existing:
            merged.append(current)

    return merged


def _rank_face_boxes(boxes, frame_shape):
    frame_height, frame_width = frame_shape[:2]
    center_x = frame_width / 2.0
    center_y = frame_height / 2.0
    frame_area = max(frame_width * frame_height, 1)

    ranked = []
    for box in boxes:
        x, y, width, height = box
        face_center_x = x + (width / 2.0)
        face_center_y = y + (height / 2.0)
        area_score = _box_area(box) / frame_area
        dx = abs(face_center_x - center_x) / max(center_x, 1.0)
        dy = abs(face_center_y - center_y) / max(center_y, 1.0)
        center_score = 1.0 - min(1.0, (dx * 0.65) + (dy * 0.35))
        edge_penalty = 0.0
        if x <= 5 or y <= 5 or (x + width) >= (frame_width - 5) or (y + height) >= (frame_height - 5):
            edge_penalty = 0.08

        score = (area_score * 2.8) + center_score - edge_penalty
        ranked.append((score, box))

    ranked.sort(key=lambda item: item[0], reverse=True)
    return [box for _, box in ranked[:MAX_TRACKED_FACES]]


def _detect_with_classifier(classifier, frame, scale_factor, min_neighbors, min_size):
    if classifier is None:
        return []

    faces = classifier.detectMultiScale(
        frame,
        scaleFactor=scale_factor,
        minNeighbors=min_neighbors,
        minSize=min_size,
    )
    return [tuple(int(v) for v in face) for face in faces]


def detect_faces(frame, detector):
    if cv2 is None:
        return None, "OpenCV is not installed."

    if frame is None:
        return None, "Frame is missing."

    if detector is None:
        return None, "Face detector is not loaded."

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    equalized_frame = cv2.equalizeHist(gray_frame)

    frontal_faces = _detect_with_classifier(
        detector.get("frontal"),
        equalized_frame,
        FACE_SCALE_FACTOR,
        FACE_MIN_NEIGHBORS,
        MIN_FACE_SIZE,
    )

    alt_faces = _detect_with_classifier(
        detector.get("frontal_alt"),
        equalized_frame,
        1.05,
        max(FACE_MIN_NEIGHBORS - 1, 3),
        FACE_FALLBACK_MIN_SIZE,
    )

    profile_faces = _detect_with_classifier(
        detector.get("profile"),
        equalized_frame,
        1.08,
        max(FACE_FALLBACK_MIN_NEIGHBORS, 3),
        FACE_FALLBACK_MIN_SIZE,
    )

    flipped_equalized = cv2.flip(equalized_frame, 1)
    flipped_profile_faces = _detect_with_classifier(
        detector.get("profile"),
        flipped_equalized,
        1.08,
        max(FACE_FALLBACK_MIN_NEIGHBORS, 3),
        FACE_FALLBACK_MIN_SIZE,
    )
    frame_width = frame.shape[1]
    restored_flipped_profiles = [
        (frame_width - x - width, y, width, height)
        for (x, y, width, height) in flipped_profile_faces
    ]

    fallback_faces = []
    if not frontal_faces and not alt_faces and not profile_faces and not restored_flipped_profiles:
        fallback_faces = _detect_with_classifier(
            detector.get("frontal"),
            gray_frame,
            1.05,
            FACE_FALLBACK_MIN_NEIGHBORS,
            FACE_FALLBACK_MIN_SIZE,
        )

    all_faces = frontal_faces + alt_faces + profile_faces + restored_flipped_profiles + fallback_faces
    merged_faces = _merge_face_boxes(all_faces)
    ranked_faces = _rank_face_boxes(merged_faces, frame.shape)
    return ranked_faces, None


def has_face(frame, detector):
    faces, error = detect_faces(frame, detector)

    if error is not None:
        return False, error

    return len(faces) > 0, None


def annotate_faces(frame, faces):
    if cv2 is None or frame is None:
        return frame

    annotated_frame = frame.copy()

    for index, (x, y, width, height) in enumerate(faces):
        color = (0, 255, 255) if index == 0 else (0, 255, 0)
        thickness = 3 if index == 0 else 2
        cv2.rectangle(
            annotated_frame,
            (x, y),
            (x + width, y + height),
            color,
            thickness,
        )

    return annotated_frame
