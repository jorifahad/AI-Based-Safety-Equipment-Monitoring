from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

import kmeans_auto_denoising as kd


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

# ضعي الصورة المشوشة بهذا الاسم داخل Model-Deployment
INPUT_IMAGE = BASE_DIR / "factory_workers_salt_pepper.png"

DENOISED_IMAGE = BASE_DIR / "denoised_image.png"
FINAL_RESULT_IMAGE = BASE_DIR / "final_ppe_result.png"

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


# ==========================================================
# YOLO settings
# ==========================================================

CONFIDENCE_THRESHOLD = 0.25


# ==========================================================
# Colors
# OpenCV uses BGR
# ==========================================================

GREEN = (0, 255, 0)
RED = (0, 0, 255)
YELLOW = (0, 255, 255)


COMPLIANT_CLASSES = {
    "hardhat",
    "mask",
    "safety vest",
}

NON_COMPLIANT_CLASSES = {
    "no-hardhat",
    "no-mask",
    "no-safety vest",
}


# ==========================================================
# K-Means denoising
# ==========================================================

def run_kmeans_denoising(
    input_path: Path,
) -> tuple[np.ndarray, str]:
    image = kd.read_color_image(input_path)

    print("=" * 60)
    print("STEP 1: Automatic noise detection")
    print("=" * 60)

    (
        detected_noise_type,
        impulse_ratio,
        estimated_sigma,
    ) = kd.detect_noise_type(image)

    print(f"Detected noise type: {detected_noise_type}")
    print(f"Impulse ratio: {impulse_ratio:.4%}")
    print(f"Estimated Gaussian sigma: {estimated_sigma:.2f}")

    result = image.copy()

    # Salt-and-pepper or mixed noise
    if detected_noise_type in {
        "salt_pepper",
        "mixed",
    }:
        print("\nRemoving Salt-and-Pepper noise using K-Means...")

        impulse_mask = kd.detect_impulse_pixels_kmeans(
            result
        )

        result = kd.remove_impulse_noise(
            result,
            impulse_mask,
        )

    # Gaussian or mixed noise
    if detected_noise_type in {
        "gaussian",
        "mixed",
    }:
        print("\nRemoving Gaussian noise using K-Means patches...")

        result = kd.remove_gaussian_noise_kmeans(
            result,
            estimated_sigma,
        )

    if detected_noise_type == "clean":
        print("\nNo meaningful noise detected.")

    kd.save_png(
        DENOISED_IMAGE,
        result,
    )

    print(f"\nDenoised image saved:\n{DENOISED_IMAGE}")

    return result, detected_noise_type


# ==========================================================
# YOLO detection
# ==========================================================

def run_yolo_detection(
    image: np.ndarray,
) -> None:
    print("\n" + "=" * 60)
    print("STEP 2: YOLO PPE detection")
    print("=" * 60)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"YOLO model was not found:\n{MODEL_PATH}"
        )

    model = YOLO(
        str(MODEL_PATH)
    )

    results = model.predict(
        source=image,
        conf=CONFIDENCE_THRESHOLD,
        save=False,
        verbose=False,
    )

    result = results[0]
    annotated_image = image.copy()

    print("\nDetected objects:")

    if result.boxes is None or len(result.boxes) == 0:
        print("No objects detected.")

    else:
        for box in result.boxes:
            x1, y1, x2, y2 = map(
                int,
                box.xyxy[0].tolist(),
            )

            class_id = int(
                box.cls[0]
            )

            confidence = float(
                box.conf[0]
            )

            class_name = str(
                model.names[class_id]
            )

            normalized_name = (
                class_name
                .strip()
                .lower()
            )

            if normalized_name in NON_COMPLIANT_CLASSES:
                color = RED
                status = "NON-COMPLIANT"

            elif normalized_name in COMPLIANT_CLASSES:
                color = GREEN
                status = "COMPLIANT"

            else:
                color = YELLOW
                status = "OBJECT"

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )

            cv2.rectangle(
                annotated_image,
                (x1, y1),
                (x2, y2),
                color,
                3,
            )

            text_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                2,
            )

            text_width, text_height = text_size

            cv2.rectangle(
                annotated_image,
                (x1, max(0, y1 - text_height - 12)),
                (x1 + text_width + 8, y1),
                color,
                -1,
            )

            cv2.putText(
                annotated_image,
                label,
                (x1 + 4, max(15, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2,
            )

            print(
                f"- {class_name}: "
                f"{confidence:.2f} "
                f"[{status}]"
            )

    kd.save_png(
        FINAL_RESULT_IMAGE,
        annotated_image,
    )

    print(
        f"\nFinal YOLO result saved:\n"
        f"{FINAL_RESULT_IMAGE}"
    )

    cv2.imshow(
        "K-Means + YOLO PPE Detection",
        annotated_image,
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ==========================================================
# Main pipeline
# ==========================================================

def main() -> None:
    if not INPUT_IMAGE.exists():
        raise FileNotFoundError(
            "Input image was not found.\n"
            f"Place the noisy image here:\n{INPUT_IMAGE}"
        )

    denoised_image, noise_type = run_kmeans_denoising(
        INPUT_IMAGE
    )

    run_yolo_detection(
        denoised_image
    )

    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)
    print(f"Detected noise: {noise_type}")
    print(f"Denoised image: {DENOISED_IMAGE}")
    print(f"Final PPE result: {FINAL_RESULT_IMAGE}")


if __name__ == "__main__":
    main()
