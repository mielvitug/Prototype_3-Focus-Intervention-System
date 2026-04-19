from pathlib import Path

from config import DATASET_DIR, PHONE_USAGE_IMAGE_SIZE, PHONE_USAGE_MODEL_PATH
from phone_usage_model import (
    PhoneUsageModel,
    accuracy_score,
    find_best_threshold,
    load_dataset,
    train_validation_split,
)


def main():
    print("Loading dataset from:", DATASET_DIR)
    features, targets = load_dataset(DATASET_DIR, PHONE_USAGE_IMAGE_SIZE)

    print(f"Loaded {len(features)} images.")
    print(f"Positive samples: {int(targets.sum())}")
    print(f"Negative samples: {int((targets == 0).sum())}")

    train_x, train_y, val_x, val_y = train_validation_split(features, targets)

    model = PhoneUsageModel(
        image_size=PHONE_USAGE_IMAGE_SIZE,
        learning_rate=0.01,
        epochs=700,
        regularization=2e-2,
        pca_components=128,
    )
    history = model.fit(train_x, train_y)

    train_accuracy = accuracy_score(model, train_x, train_y)
    validation_accuracy = train_accuracy

    if len(val_x):
        best_threshold, validation_accuracy = find_best_threshold(model, val_x, val_y)
        model.decision_threshold = best_threshold
        train_accuracy = accuracy_score(model, train_x, train_y)
        validation_accuracy = accuracy_score(model, val_x, val_y)

    model.save(PHONE_USAGE_MODEL_PATH)

    print("Training history:")
    for epoch, loss in history:
        print(f"  epoch {epoch:03d} | loss={loss:.4f}")

    print(f"Train accuracy: {train_accuracy:.3f}")
    print(f"Validation accuracy: {validation_accuracy:.3f}")
    print(f"Decision threshold: {model.decision_threshold:.3f}")
    print("Saved model to:", PHONE_USAGE_MODEL_PATH)


if __name__ == "__main__":
    main()
