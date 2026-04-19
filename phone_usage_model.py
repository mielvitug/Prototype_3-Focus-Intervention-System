import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


SUPPORTED_IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


def _resample_mode():
    if hasattr(Image, "Resampling"):
        return Image.Resampling.LANCZOS
    return Image.LANCZOS


def load_image_vector(image_path, image_size):
    image = Image.open(image_path).convert("L")
    return image_to_vector(image, image_size)


def image_to_vector(image, image_size):
    image = image.convert("L")
    image = ImageOps.autocontrast(image)
    image = ImageOps.fit(image, image_size, method=_resample_mode())
    image_array = np.asarray(image, dtype=np.float32) / 255.0
    return image_array.reshape(-1)


def clamp_box(left, top, right, bottom, width, height):
    left = max(0, min(int(left), width - 1))
    top = max(0, min(int(top), height - 1))
    right = max(left + 1, min(int(right), width))
    bottom = max(top + 1, min(int(bottom), height))
    return left, top, right, bottom


def crop_to_vector(image, box, image_size):
    width, height = image.size
    left, top, right, bottom = clamp_box(*box, width, height)
    crop = image.crop((left, top, right, bottom))
    return image_to_vector(crop, image_size)


def box_area(box):
    left, top, right, bottom = box
    return max(0, right - left) * max(0, bottom - top)


def box_iou(box_a, box_b):
    left = max(box_a[0], box_b[0])
    top = max(box_a[1], box_b[1])
    right = min(box_a[2], box_b[2])
    bottom = min(box_a[3], box_b[3])

    intersection = box_area((left, top, right, bottom))
    if intersection <= 0:
        return 0.0

    union = box_area(box_a) + box_area(box_b) - intersection
    if union <= 0:
        return 0.0
    return intersection / union


def frame_to_vector(frame, image_size):
    rgb_frame = frame[:, :, ::-1]
    image = Image.fromarray(rgb_frame)
    return image_to_vector(image, image_size).reshape(1, -1)


def load_annotations(dataset_dir):
    labels_path = Path(dataset_dir) / "labels.csv"
    if not labels_path.exists():
        return {}

    annotations = {}

    with labels_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                box = (
                    int(float(row["xmin"])),
                    int(float(row["ymin"])),
                    int(float(row["xmax"])),
                    int(float(row["ymax"])),
                )
                annotations.setdefault(row["filename"], []).append(box)
            except Exception:
                continue

    return annotations


def expand_box(box, image_size, scale):
    width, height = image_size
    left, top, right, bottom = box
    center_x = (left + right) / 2.0
    center_y = (top + bottom) / 2.0
    box_width = (right - left) * scale
    box_height = (bottom - top) * scale
    return clamp_box(
        center_x - (box_width / 2.0),
        center_y - (box_height / 2.0),
        center_x + (box_width / 2.0),
        center_y + (box_height / 2.0),
        width,
        height,
    )


def _append_candidate_box(candidates, frame_width, frame_height, left, top, right, bottom, weight):
    box = clamp_box(left, top, right, bottom, frame_width, frame_height)
    if box_area(box) <= 0:
        return
    candidates.append((box, float(weight)))


def load_labeled_phone_crops(dataset_dir, image_size):
    positive_dir = Path(dataset_dir) / "positive"
    if not positive_dir.exists():
        return []

    annotations = load_annotations(dataset_dir)
    samples = []

    for filename, boxes in annotations.items():
        image_path = positive_dir / filename
        if not image_path.exists():
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        for box in boxes:
            samples.append(crop_to_vector(image, box, image_size))
            samples.append(crop_to_vector(image, expand_box(box, image.size, 0.92), image_size))
            samples.append(crop_to_vector(image, expand_box(box, image.size, 1.15), image_size))
            samples.append(crop_to_vector(image, expand_box(box, image.size, 1.35), image_size))
            samples.append(crop_to_vector(image, expand_box(box, image.size, 1.55), image_size))

    return samples


def sample_negative_crops(dataset_dir, image_size, per_image=2):
    negative_dir = Path(dataset_dir) / "negative"
    if not negative_dir.exists():
        return []

    rng = np.random.default_rng(42)
    samples = []

    for image_path in sorted(negative_dir.iterdir()):
        if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        width, height = image.size
        min_dim = min(width, height)
        if min_dim < 24:
            continue

        for _ in range(per_image):
            crop_size = int(rng.integers(max(24, min_dim // 4), max(25, min_dim + 1)))
            left = int(rng.integers(0, max(1, width - crop_size + 1)))
            top = int(rng.integers(0, max(1, height - crop_size + 1)))
            box = (left, top, left + crop_size, top + crop_size)
            samples.append(crop_to_vector(image, box, image_size))

    return samples


def sample_hard_negative_crops(dataset_dir, image_size, per_image=4):
    positive_dir = Path(dataset_dir) / "positive"
    if not positive_dir.exists():
        return []

    annotations = load_annotations(dataset_dir)
    if not annotations:
        return []

    rng = np.random.default_rng(123)
    samples = []

    for filename, boxes in annotations.items():
        image_path = positive_dir / filename
        if not image_path.exists():
            continue

        try:
            image = Image.open(image_path).convert("RGB")
        except Exception:
            continue

        width, height = image.size
        max_attempts = per_image * 20
        accepted = 0
        attempts = 0

        while accepted < per_image and attempts < max_attempts:
            attempts += 1
            ref_box = boxes[int(rng.integers(0, len(boxes)))]
            ref_width = max(24, ref_box[2] - ref_box[0])
            ref_height = max(24, ref_box[3] - ref_box[1])
            crop_scale = float(rng.uniform(0.9, 1.8))
            crop_width = int(ref_width * crop_scale)
            crop_height = int(ref_height * crop_scale)
            aspect = float(rng.uniform(0.75, 1.25))
            crop_width = max(24, int(crop_width * aspect))
            crop_height = max(24, int(crop_height / aspect))

            left = int(rng.integers(0, max(1, width - crop_width + 1)))
            top = int(rng.integers(0, max(1, height - crop_height + 1)))
            candidate = clamp_box(left, top, left + crop_width, top + crop_height, width, height)

            if any(box_iou(candidate, phone_box) > 0.08 for phone_box in boxes):
                continue

            samples.append(crop_to_vector(image, candidate, image_size))
            accepted += 1

    return samples


def load_dataset(dataset_dir, image_size):
    dataset_dir = Path(dataset_dir)
    samples = []
    labels = []

    positive_samples = load_labeled_phone_crops(dataset_dir, image_size)
    negative_samples = sample_negative_crops(dataset_dir, image_size, per_image=3)
    hard_negative_samples = sample_hard_negative_crops(dataset_dir, image_size, per_image=5)

    if positive_samples and (negative_samples or hard_negative_samples):
        samples.extend(positive_samples)
        labels.extend([1] * len(positive_samples))
        samples.extend(negative_samples)
        labels.extend([0] * len(negative_samples))
        samples.extend(hard_negative_samples)
        labels.extend([0] * len(hard_negative_samples))
    else:
        for folder_name, label in (("negative", 0), ("positive", 1)):
            class_dir = dataset_dir / folder_name
            if not class_dir.exists():
                continue

            for image_path in sorted(class_dir.iterdir()):
                if image_path.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
                    continue

                try:
                    samples.append(load_image_vector(image_path, image_size))
                    labels.append(label)
                except Exception:
                    continue

    if not samples:
        raise ValueError("No training images were loaded from the dataset folders.")

    features = np.vstack(samples)
    targets = np.asarray(labels, dtype=np.float32)
    return features, targets


def train_validation_split(features, targets, validation_fraction=0.2, seed=42):
    rng = np.random.default_rng(seed)
    train_indices = []
    validation_indices = []

    for label in (0.0, 1.0):
        label_indices = np.where(targets == label)[0]
        rng.shuffle(label_indices)
        split_index = max(1, int(len(label_indices) * (1 - validation_fraction)))
        train_indices.extend(label_indices[:split_index])
        validation_indices.extend(label_indices[split_index:])

    train_indices = np.asarray(train_indices, dtype=np.int32)
    validation_indices = np.asarray(validation_indices, dtype=np.int32)
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)

    return (
        features[train_indices],
        targets[train_indices],
        features[validation_indices],
        targets[validation_indices],
    )


def generate_phone_candidate_boxes(frame_shape, faces):
    frame_height, frame_width = frame_shape[:2]
    candidates = []

    for (x, y, width, height) in faces:
        body_width = max(width, int(width * 1.1))
        candidate_specs = [
            (x - 0.90 * body_width, y + 0.40 * height, x + 0.15 * width, y + 1.70 * height, 1.00),
            (x + 0.85 * width, y + 0.40 * height, x + 1.90 * body_width, y + 1.70 * height, 1.00),
            (x - 0.78 * body_width, y + 0.95 * height, x + 0.22 * width, y + 2.35 * height, 0.96),
            (x + 0.78 * width, y + 0.95 * height, x + 1.78 * body_width, y + 2.35 * height, 0.96),
            (x - 0.25 * width, y + 0.88 * height, x + 1.25 * width, y + 2.15 * height, 0.88),
            (x + 0.10 * width, y + 1.00 * height, x + 0.92 * width, y + 2.18 * height, 0.92),
            (x - 0.45 * width, y - 0.12 * height, x + 0.28 * width, y + 1.10 * height, 0.82),
            (x + 0.72 * width, y - 0.12 * height, x + 1.45 * width, y + 1.10 * height, 0.82),
        ]

        for left, top, right, bottom, weight in candidate_specs:
            _append_candidate_box(candidates, frame_width, frame_height, left, top, right, bottom, weight)

        torso_left = x - 0.68 * width
        torso_right = x + 1.68 * width
        torso_top = y + 0.36 * height
        torso_bottom = y + 2.45 * height
        torso_width = torso_right - torso_left
        torso_height = torso_bottom - torso_top

        grid_cols = 3
        grid_rows = 2
        window_specs = (
            (0.34, 0.48, 0.92),
            (0.44, 0.60, 0.84),
        )

        for width_scale, height_scale, weight in window_specs:
            window_width = torso_width * width_scale
            window_height = torso_height * height_scale

            for row in range(grid_rows):
                for col in range(grid_cols):
                    center_x = torso_left + torso_width * ((col + 0.5) / grid_cols)
                    center_y = torso_top + torso_height * ((row + 0.68) / (grid_rows + 0.45))
                    _append_candidate_box(
                        candidates,
                        frame_width,
                        frame_height,
                        center_x - (window_width / 2.0),
                        center_y - (window_height / 2.0),
                        center_x + (window_width / 2.0),
                        center_y + (window_height / 2.0),
                        weight,
                    )

    if not candidates:
        fallback_specs = (
            (frame_width * 0.04, frame_height * 0.46, frame_width * 0.30, frame_height * 0.92, 0.55),
            (frame_width * 0.35, frame_height * 0.44, frame_width * 0.65, frame_height * 0.92, 0.52),
            (frame_width * 0.70, frame_height * 0.46, frame_width * 0.96, frame_height * 0.92, 0.55),
        )
        for left, top, right, bottom, weight in fallback_specs:
            _append_candidate_box(candidates, frame_width, frame_height, left, top, right, bottom, weight)

    unique_boxes = {}
    for box, weight in candidates:
        key = tuple(int(value) for value in box)
        unique_boxes[key] = max(weight, unique_boxes.get(key, 0.0))

    return [(box, weight) for box, weight in unique_boxes.items()]


class PhoneUsageModel:
    def __init__(
        self,
        image_size=(24, 24),
        learning_rate=0.1,
        epochs=400,
        regularization=1e-2,
        pca_components=128,
        decision_threshold=0.5,
    ):
        self.image_size = tuple(image_size)
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.regularization = regularization
        self.pca_components = pca_components
        self.decision_threshold = decision_threshold
        self.weights = None
        self.bias = 0.0
        self.mean = None
        self.std = None
        self.projection = None

    @staticmethod
    def _sigmoid(values):
        clipped = np.clip(values, -40.0, 40.0)
        return 1.0 / (1.0 + np.exp(-clipped))

    def _normalize(self, features):
        return (features - self.mean) / self.std

    def _transform(self, features):
        normalized = self._normalize(features)
        if self.projection is not None:
            return normalized @ self.projection
        return normalized

    def fit(self, features, targets):
        self.mean = features.mean(axis=0)
        self.std = features.std(axis=0) + 1e-6

        normalized = self._normalize(features)

        if self.pca_components:
            max_components = min(normalized.shape[0], normalized.shape[1], self.pca_components)
            _, _, vt = np.linalg.svd(normalized, full_matrices=False)
            self.projection = vt[:max_components].T.astype(np.float32)
            training_features = normalized @ self.projection
        else:
            self.projection = None
            training_features = normalized

        feature_count = training_features.shape[1]
        self.weights = np.zeros(feature_count, dtype=np.float32)
        self.bias = 0.0
        positive_count = max(float(targets.sum()), 1.0)
        negative_count = max(float(len(targets) - targets.sum()), 1.0)
        positive_weight = len(targets) / (2.0 * positive_count)
        negative_weight = len(targets) / (2.0 * negative_count)
        sample_weights = np.where(targets > 0.5, positive_weight, negative_weight).astype(np.float32)

        history = []

        for epoch in range(self.epochs):
            logits = training_features @ self.weights + self.bias
            probabilities = self._sigmoid(logits)

            errors = (probabilities - targets) * sample_weights
            gradient_w = (training_features.T @ errors) / len(training_features)
            gradient_w += self.regularization * self.weights
            gradient_b = float(errors.mean())

            self.weights -= self.learning_rate * gradient_w
            self.bias -= self.learning_rate * gradient_b

            if epoch % 25 == 0 or epoch == self.epochs - 1:
                epsilon = 1e-7
                loss = -np.mean(
                    targets * np.log(probabilities + epsilon)
                    + (1 - targets) * np.log(1 - probabilities + epsilon)
                )
                history.append((epoch, float(loss)))

        return history

    def predict_proba(self, features):
        transformed = self._transform(features)
        logits = transformed @ self.weights + self.bias
        return self._sigmoid(logits)

    def predict(self, features, threshold=None):
        if threshold is None:
            threshold = self.decision_threshold
        return (self.predict_proba(features) >= threshold).astype(np.int32)

    def predict_frame_probability(self, frame, faces=None):
        rgb_frame = frame[:, :, ::-1]
        image = Image.fromarray(rgb_frame)
        candidate_boxes = generate_phone_candidate_boxes(frame.shape, faces or [])
        frame_area = max(frame.shape[0] * frame.shape[1], 1)

        candidate_scores = []
        for box, candidate_weight in candidate_boxes:
            box_variants = [
                box,
                expand_box(box, image.size, 1.18),
                expand_box(box, image.size, 0.86),
            ]

            box_width = max(1, box[2] - box[0])
            box_height = max(1, box[3] - box[1])
            aspect_ratio = box_height / box_width
            area_ratio = box_area(box) / frame_area
            geometry_weight = candidate_weight

            if aspect_ratio > 2.85:
                geometry_weight *= 0.72
            elif aspect_ratio > 2.35:
                geometry_weight *= 0.82
            elif aspect_ratio < 0.78:
                geometry_weight *= 0.86

            if area_ratio > 0.22:
                geometry_weight *= 0.78
            elif area_ratio > 0.16:
                geometry_weight *= 0.88
            elif area_ratio < 0.012:
                geometry_weight *= 0.80

            best_variant_probability = 0.0
            for variant in box_variants:
                features = crop_to_vector(image, variant, self.image_size).reshape(1, -1)
                probability = float(self.predict_proba(features)[0])
                if probability > best_variant_probability:
                    best_variant_probability = probability

            adjusted_probability = min(best_variant_probability * geometry_weight, 1.0)
            candidate_scores.append((adjusted_probability, best_variant_probability))

        if not candidate_scores:
            return 0.0

        candidate_scores.sort(key=lambda item: item[0], reverse=True)
        top_scores = [score for score, _ in candidate_scores[:3]]
        top_raw_score = max(raw_score for _, raw_score in candidate_scores)
        supporting_hits = sum(
            1 for adjusted_score, raw_score in candidate_scores
            if adjusted_score >= 0.58 or raw_score >= 0.72
        )

        if len(top_scores) == 1:
            return top_scores[0] * 0.88

        if top_scores[0] >= 0.82 and supporting_hits >= 2:
            return min(1.0, (top_scores[0] * 0.72) + (top_scores[1] * 0.28) + 0.04)

        weights = (0.60, 0.25, 0.15)
        blended_score = 0.0
        for score, weight in zip(top_scores, weights):
            blended_score += score * weight

        support_bonus = min(0.12, max(0, supporting_hits - 1) * 0.04)
        isolated_penalty = 0.80 if supporting_hits <= 1 and top_scores[0] < 0.90 else 1.0
        raw_bonus = max(0.0, top_raw_score - top_scores[0]) * 0.08
        return min(1.0, max(0.0, (blended_score + support_bonus + raw_bonus) * isolated_penalty))

    def save(self, output_path):
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            output_path,
            weights=self.weights,
            bias=self.bias,
            mean=self.mean,
            std=self.std,
            projection=np.array([]) if self.projection is None else self.projection,
            image_width=self.image_size[0],
            image_height=self.image_size[1],
            pca_components=-1 if self.pca_components is None else self.pca_components,
            decision_threshold=self.decision_threshold,
        )

    @classmethod
    def load(cls, model_path):
        model_data = np.load(model_path)
        image_size = (int(model_data["image_width"]), int(model_data["image_height"]))
        model = cls(image_size=image_size)
        model.weights = model_data["weights"]
        model.bias = float(model_data["bias"])
        model.mean = model_data["mean"]
        model.std = model_data["std"]
        projection = model_data["projection"]
        if projection.size == 0:
            projection = None
        model.projection = projection
        saved_pca_components = int(model_data["pca_components"])
        model.pca_components = None if saved_pca_components == -1 else saved_pca_components
        model.decision_threshold = float(model_data["decision_threshold"])
        return model


def accuracy_score(model, features, targets, threshold=None):
    predictions = model.predict(features, threshold=threshold)
    return float((predictions == targets).mean())


def find_best_threshold(model, features, targets, thresholds=None):
    if thresholds is None:
        thresholds = np.linspace(0.35, 0.9, 23)

    probabilities = model.predict_proba(features)
    best_accuracy = -1.0
    best_threshold = model.decision_threshold

    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(np.int32)
        true_positive = float(((predictions == 1) & (targets == 1)).sum())
        true_negative = float(((predictions == 0) & (targets == 0)).sum())
        false_positive = float(((predictions == 1) & (targets == 0)).sum())
        false_negative = float(((predictions == 0) & (targets == 1)).sum())
        positive_recall = true_positive / max(true_positive + false_negative, 1.0)
        negative_recall = true_negative / max(true_negative + false_positive, 1.0)
        balanced_accuracy = (positive_recall + negative_recall) / 2.0
        precision = true_positive / max(true_positive + false_positive, 1.0)
        score = balanced_accuracy + (precision * 0.15)
        if score > best_accuracy:
            best_accuracy = score
            best_threshold = float(threshold)

    return best_threshold, best_accuracy
