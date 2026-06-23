from pathlib import Path

from ultralytics import YOLO


BASE_DIR = Path(__file__).resolve().parent

MODEL_PATH = (
    BASE_DIR
    / ".."
    / "Model-Training"
    / "Outputs"
    / "runs"
    / "detect"
    / "yolov8s_ppe_css_200_epochs"
    / "weights"
    / "best.pt"
).resolve()

IMAGE_PATH = (
    BASE_DIR
    / "soloun.png"
).resolve()

OUTPUT_DIR = (
    BASE_DIR
    / "runs"
).resolve()


if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )

if not IMAGE_PATH.exists():
    raise FileNotFoundError(
        f"Image not found:\n{IMAGE_PATH}"
    )


model = YOLO(str(MODEL_PATH))

results = model.predict(
    source=str(IMAGE_PATH),
    conf=0.25,
    save=True,
    show=True,
    project=str(OUTPUT_DIR),
    name="ppe_test",
)

print("\nCompleted successfully.")
print(f"Model:\n{MODEL_PATH}")
print(f"Image:\n{IMAGE_PATH}")
print(f"Result folder:\n{OUTPUT_DIR / 'ppe_test'}")