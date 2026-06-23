from flask import Flask, render_template, Response, jsonify, request, send_file
from flask_socketio import SocketIO
import cv2
from ultralytics import YOLO
from datetime import datetime
import os
import threading
import time
from detection_logic import InstanceDetector, ComplianceChecker, SnapshotManager
from database import Database
import json
from pathlib import Path
from werkzeug.utils import secure_filename
import uuid

import kmeans_auto_denoising as kd


app = Flask(__name__)
app.config["SECRET_KEY"] = "N/A"

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode="threading",
)


# ==========================================================
# Paths and upload settings
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
RESULT_FOLDER = BASE_DIR / "static" / "results"

UPLOAD_FOLDER.mkdir(parents=True, exist_ok=True)
RESULT_FOLDER.mkdir(parents=True, exist_ok=True)

app.config["UPLOAD_FOLDER"] = str(UPLOAD_FOLDER)
app.config["RESULT_FOLDER"] = str(RESULT_FOLDER)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    "png",
    "jpg",
    "jpeg",
    "webp",
}


# ==========================================================
# Global objects
# ==========================================================

model = None
camera = None
streaming = False
stream_lock = threading.Lock()
dev_mode = False

db = Database()
instance_detector = InstanceDetector()
compliance_checker = ComplianceChecker()
snapshot_manager = SnapshotManager()


# ==========================================================
# Settings
# ==========================================================

SETTINGS_FILE = "settings.json"

DEFAULT_SETTINGS = {
    "required_ppe": {
        "helmet": True,
        "vest": True,
        "mask": False,
    },
    "non_compliance_delay": 3,
    "instance_reset_timeout": 5,
    "detection_mode": "single",
}


def load_settings():
    try:
        if os.path.exists(SETTINGS_FILE):
            with open(
                SETTINGS_FILE,
                "r",
                encoding="utf-8",
            ) as file:
                return json.load(file)
    except Exception as error:
        print(f"Error loading settings: {error}")

    return DEFAULT_SETTINGS.copy()


def save_settings_to_file(settings):
    try:
        with open(
            SETTINGS_FILE,
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                settings,
                file,
                indent=2,
            )

        return True

    except Exception as error:
        print(f"Error saving settings: {error}")
        return False


current_settings = load_settings()


# ==========================================================
# YOLO model
# ==========================================================

def load_model():
    global model

    preferred_model = (
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

    fallback_model = (
        BASE_DIR
        / ".."
        / "Model-Training"
        / "Outputs"
        / "runs"
        / "detect"
        / "yolov8s_ppe_css_80_epochs"
        / "weights"
        / "best.pt"
    ).resolve()

    if preferred_model.exists():
        model_path = preferred_model

    elif fallback_model.exists():
        model_path = fallback_model

    else:
        raise FileNotFoundError(
            "PPE model was not found.\n"
            f"Checked:\n{preferred_model}\n{fallback_model}"
        )

    print(f"Loading PPE model:\n{model_path}")
    model = YOLO(str(model_path))


# ==========================================================
# Helpers
# ==========================================================

def allowed_image(filename):
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower()
        in ALLOWED_EXTENSIONS
    )


def normalize_class_name(name):
    return (
        str(name)
        .strip()
        .lower()
        .replace("_", " ")
    )


def process_uploaded_image(
    input_path,
    denoised_path,
    result_path,
):
    global model

    if model is None:
        load_model()

    image = kd.read_color_image(
        Path(input_path)
    )

    (
        noise_type,
        impulse_ratio,
        estimated_sigma,
    ) = kd.detect_noise_type(image)

    print("=" * 60)
    print("Uploaded image analysis")
    print("=" * 60)
    print(f"Noise type: {noise_type}")
    print(f"Impulse ratio: {impulse_ratio:.4%}")
    print(f"Gaussian sigma: {estimated_sigma:.2f}")

    denoised = image.copy()

    if noise_type in {
        "salt_pepper",
        "mixed",
    }:
        impulse_mask = (
            kd.detect_impulse_pixels_kmeans(
                denoised
            )
        )

        denoised = kd.remove_impulse_noise(
            denoised,
            impulse_mask,
        )

    if noise_type in {
        "gaussian",
        "mixed",
    }:
        denoised = (
            kd.remove_gaussian_noise_kmeans(
                denoised,
                estimated_sigma,
            )
        )

    kd.save_png(
        Path(denoised_path),
        denoised,
    )

    yolo_results = model.predict(
        source=denoised,
        conf=0.25,
        save=False,
        verbose=False,
    )

    detections = []
    annotated = denoised.copy()

    compliant_classes = {
        "hardhat",
        "mask",
        "safety vest",
    }

    non_compliant_classes = {
        "no-hardhat",
        "no-mask",
        "no-safety vest",
    }

    boxes = yolo_results[0].boxes

    if boxes is not None:
        for box in boxes:
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

            class_name = model.names[
                class_id
            ]

            normalized = normalize_class_name(
                class_name
            )

            if normalized in non_compliant_classes:
                color = (0, 0, 255)
                status = "non_compliant"

            elif normalized in compliant_classes:
                color = (0, 255, 0)
                status = "compliant"

            else:
                color = (0, 255, 255)
                status = "object"

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )

            cv2.rectangle(
                annotated,
                (x1, y1),
                (x2, y2),
                color,
                3,
            )

            text_size, _ = cv2.getTextSize(
                label,
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                2,
            )

            text_width, text_height = (
                text_size
            )

            label_top = max(
                0,
                y1 - text_height - 12,
            )

            cv2.rectangle(
                annotated,
                (x1, label_top),
                (x1 + text_width + 8, y1),
                color,
                -1,
            )

            cv2.putText(
                annotated,
                label,
                (
                    x1 + 4,
                    max(15, y1 - 6),
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
            )

            detections.append(
                {
                    "class": class_name,
                    "confidence": round(
                        confidence,
                        3,
                    ),
                    "status": status,
                    "bbox": [
                        x1,
                        y1,
                        x2,
                        y2,
                    ],
                }
            )

    kd.save_png(
        Path(result_path),
        annotated,
    )

    non_compliant_count = sum(
        detection["status"]
        == "non_compliant"
        for detection in detections
    )

    compliant_count = sum(
        detection["status"]
        == "compliant"
        for detection in detections
    )

    return {
        "noise_type": noise_type,
        "impulse_ratio": round(
            impulse_ratio,
            6,
        ),
        "estimated_sigma": round(
            estimated_sigma,
            2,
        ),
        "detections": detections,
        "total_detections": len(
            detections
        ),
        "compliant_count": compliant_count,
        "non_compliant_count": (
            non_compliant_count
        ),
        "is_compliant": (
            non_compliant_count == 0
        ),
    }


# ==========================================================
# Camera stream
# ==========================================================

def generate_frames():
    global camera
    global streaming
    global model
    global current_settings

    last_alert_time = 0
    alert_cooldown = current_settings[
        "non_compliance_delay"
    ]

    last_snapshot_time = 0
    snapshot_interval = current_settings[
        "instance_reset_timeout"
    ]

    while streaming:
        try:
            with stream_lock:
                if (
                    camera is None
                    or not camera.isOpened()
                ):
                    break

                success, frame = camera.read()

            if not success or frame is None:
                time.sleep(0.1)
                continue

            results = model(frame)

            all_detections = []

            for result in results:
                boxes = result.boxes

                for box in boxes:
                    x1, y1, x2, y2 = (
                        box.xyxy[0]
                        .cpu()
                        .numpy()
                    )

                    confidence = float(
                        box.conf[0]
                    )

                    class_id = int(
                        box.cls[0]
                    )

                    class_name = (
                        model.names[class_id]
                    )

                    all_detections.append(
                        {
                            "class": class_name,
                            "confidence": confidence,
                            "bbox": [
                                int(x1),
                                int(y1),
                                int(x2),
                                int(y2),
                            ],
                        }
                    )

            instance_result = (
                instance_detector
                .process_detection(
                    all_detections,
                    dev_mode,
                    current_settings,
                )
            )

            is_compliant = (
                compliance_checker
                .check_compliance(
                    instance_result,
                    dev_mode,
                )
            )

            annotated_frame = (
                results[0].plot()
            )

            current_time = time.time()

            if (
                instance_result[
                    "should_capture"
                ]
                and (
                    current_time
                    - last_snapshot_time
                )
                >= snapshot_interval
            ):
                snapshot_filename = (
                    instance_detector
                    .get_next_snapshot_filename()
                )

                if snapshot_filename:
                    snapshot_path = (
                        snapshot_manager
                        .save_snapshot(
                            frame,
                            snapshot_filename,
                        )
                    )

                    if snapshot_path:
                        db.log_instance_snapshot(
                            instance_id=(
                                instance_result[
                                    "instance_id"
                                ]
                            ),
                            missing_ppe=(
                                instance_result[
                                    "missing_ppe"
                                ]
                            ),
                            detected_ppe=(
                                instance_result[
                                    "detected_ppe"
                                ]
                            ),
                            snapshot_path=(
                                snapshot_path
                            ),
                        )

                        last_snapshot_time = (
                            current_time
                        )

            if (
                not is_compliant
                and instance_result[
                    "has_person"
                ]
            ):
                overlay = (
                    annotated_frame.copy()
                )

                cv2.rectangle(
                    overlay,
                    (0, 0),
                    (
                        annotated_frame.shape[1],
                        annotated_frame.shape[0],
                    ),
                    (0, 0, 255),
                    20,
                )

                annotated_frame = (
                    cv2.addWeighted(
                        annotated_frame,
                        0.8,
                        overlay,
                        0.2,
                        0,
                    )
                )

                alert_text = (
                    "DEV MODE - TESTING"
                    if dev_mode
                    else (
                        "NON-COMPLIANT "
                        "DETECTED"
                    )
                )

                cv2.putText(
                    annotated_frame,
                    alert_text,
                    (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    3,
                )

                if (
                    current_time
                    - last_alert_time
                    > alert_cooldown
                ):
                    db.log_alert(
                        "NON_COMPLIANCE",
                        (
                            "PPE non-compliance "
                            "detected"
                        ),
                        None,
                    )

                    socketio.emit(
                        "alert",
                        {
                            "timestamp": (
                                datetime.now()
                                .isoformat()
                            ),
                            "type": (
                                "NON_COMPLIANCE"
                            ),
                            "description": (
                                "PPE non-compliance "
                                "detected"
                            ),
                            "dev_mode": dev_mode,
                        },
                    )

                    last_alert_time = (
                        current_time
                    )

            socketio.emit(
                "detection_update",
                {
                    "timestamp": (
                        datetime.now()
                        .isoformat()
                    ),
                    "is_compliant": (
                        is_compliant
                    ),
                    "detection_details": (
                        instance_result
                    ),
                    "dev_mode": dev_mode,
                },
            )

            success, buffer = cv2.imencode(
                ".jpg",
                annotated_frame,
                [
                    cv2.IMWRITE_JPEG_QUALITY,
                    85,
                ],
            )

            if not success:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: "
                b"image/jpeg\r\n\r\n"
                + frame_bytes
                + b"\r\n"
            )

        except GeneratorExit:
            break

        except Exception as error:
            print(
                f"Error in generate_frames: "
                f"{error}"
            )
            break


# ==========================================================
# Pages
# ==========================================================

@app.route("/")
def index():
    return render_template(
        "index.html"
    )


@app.route("/review")
def review():
    return render_template(
        "review.html"
    )


@app.route("/settings")
def settings():
    return render_template(
        "settings.html"
    )


# ==========================================================
# Image upload API
# ==========================================================

@app.route(
    "/api/process-image",
    methods=["POST"],
)
def process_image():
    try:
        if "image" not in request.files:
            return jsonify(
                {
                    "status": "error",
                    "message": (
                        "No image was uploaded."
                    ),
                }
            ), 400

        uploaded_file = request.files[
            "image"
        ]

        if uploaded_file.filename == "":
            return jsonify(
                {
                    "status": "error",
                    "message": (
                        "No image was selected."
                    ),
                }
            ), 400

        if not allowed_image(
            uploaded_file.filename
        ):
            return jsonify(
                {
                    "status": "error",
                    "message": (
                        "Supported formats: "
                        "PNG, JPG, JPEG, WEBP."
                    ),
                }
            ), 400

        extension = (
            uploaded_file.filename
            .rsplit(".", 1)[1]
            .lower()
        )

        unique_id = uuid.uuid4().hex

        input_filename = (
            f"{unique_id}_original."
            f"{extension}"
        )

        denoised_filename = (
            f"{unique_id}_denoised.png"
        )

        result_filename = (
            f"{unique_id}_result.png"
        )

        input_path = (
            UPLOAD_FOLDER
            / secure_filename(
                input_filename
            )
        )

        denoised_path = (
            RESULT_FOLDER
            / denoised_filename
        )

        result_path = (
            RESULT_FOLDER
            / result_filename
        )

        uploaded_file.save(
            str(input_path)
        )

        analysis = (
            process_uploaded_image(
                input_path,
                denoised_path,
                result_path,
            )
        )

        return jsonify(
            {
                "status": "success",
                "original_url": (
                    "/static/uploads/"
                    f"{input_filename}"
                ),
                "denoised_url": (
                    "/static/results/"
                    f"{denoised_filename}"
                ),
                "result_url": (
                    "/static/results/"
                    f"{result_filename}"
                ),
                **analysis,
            }
        )

    except Exception as error:
        print(
            f"Image processing error: "
            f"{error}"
        )

        return jsonify(
            {
                "status": "error",
                "message": str(error),
            }
        ), 500


# ==========================================================
# Settings API
# ==========================================================

@app.route(
    "/api/settings",
    methods=["GET"],
)
def get_settings():
    return jsonify(
        current_settings
    )


@app.route(
    "/api/settings",
    methods=["POST"],
)
def update_settings():
    global current_settings

    try:
        new_settings = request.json
        current_settings = new_settings

        instance_detector.update_settings(
            new_settings
        )

        if save_settings_to_file(
            new_settings
        ):
            return jsonify(
                {
                    "status": "success",
                    "message": (
                        "Settings saved"
                    ),
                }
            )

        return jsonify(
            {
                "status": "error",
                "message": (
                    "Failed to save settings "
                    "to file"
                ),
            }
        ), 500

    except Exception as error:
        return jsonify(
            {
                "status": "error",
                "message": str(error),
            }
        ), 500


@app.route(
    "/api/settings/reset",
    methods=["POST"],
)
def reset_settings():
    global current_settings

    try:
        current_settings = (
            DEFAULT_SETTINGS.copy()
        )

        instance_detector.update_settings(
            current_settings
        )

        if save_settings_to_file(
            current_settings
        ):
            return jsonify(
                {
                    "status": "success",
                    "message": (
                        "Settings reset "
                        "to defaults"
                    ),
                    "settings": (
                        current_settings
                    ),
                }
            )

        return jsonify(
            {
                "status": "error",
                "message": (
                    "Failed to save settings "
                    "to file"
                ),
            }
        ), 500

    except Exception as error:
        return jsonify(
            {
                "status": "error",
                "message": str(error),
            }
        ), 500


# ==========================================================
# Stream API
# ==========================================================

@app.route("/video_feed")
def video_feed():
    try:
        return Response(
            generate_frames(),
            mimetype=(
                "multipart/x-mixed-replace; "
                "boundary=frame"
            ),
        )

    except Exception as error:
        print(
            f"Error in video_feed: "
            f"{error}"
        )

        return "", 500


@app.route(
    "/start_stream",
    methods=["POST"],
)
def start_stream():
    global camera
    global streaming
    global model

    try:
        if model is None:
            load_model()

        with stream_lock:
            if (
                camera is None
                or not camera.isOpened()
            ):
                camera = cv2.VideoCapture(0)

                camera.set(
                    cv2.CAP_PROP_BUFFERSIZE,
                    1,
                )

        streaming = True

        return jsonify(
            {
                "status": "success",
                "message": (
                    "Stream started"
                ),
            }
        )

    except Exception as error:
        return jsonify(
            {
                "status": "error",
                "message": str(error),
            }
        ), 500


@app.route(
    "/stop_stream",
    methods=["POST"],
)
def stop_stream():
    global camera
    global streaming

    try:
        streaming = False
        time.sleep(0.3)

        with stream_lock:
            if camera is not None:
                camera.release()
                camera = None

        return jsonify(
            {
                "status": "success",
                "message": (
                    "Stream stopped"
                ),
            }
        )

    except Exception as error:
        return jsonify(
            {
                "status": "error",
                "message": str(error),
            }
        ), 500


@app.route(
    "/toggle_dev_mode",
    methods=["POST"],
)
def toggle_dev_mode():
    global dev_mode

    dev_mode = not dev_mode

    return jsonify(
        {
            "status": "success",
            "dev_mode": dev_mode,
            "message": (
                "Dev mode enabled"
                if dev_mode
                else "Dev mode disabled"
            ),
        }
    )


# ==========================================================
# Statistics and review API
# ==========================================================

@app.route("/stats")
def get_stats():
    stats = db.get_statistics()
    stats["dev_mode"] = dev_mode
    return jsonify(stats)


@app.route("/api/instances")
def get_instances():
    try:
        sort_by = request.args.get(
            "sort",
            "first_detected",
        )

        sort_order = request.args.get(
            "order",
            "desc",
        )

        instances = db.get_all_instances(
            sort_by,
            sort_order,
        )

        return jsonify(instances)

    except Exception as error:
        return jsonify(
            {
                "error": str(error),
            }
        ), 500


@app.route(
    "/api/instance/"
    "<instance_id>/snapshots"
)
def get_instance_snapshots(instance_id):
    try:
        data = db.get_instance_snapshots(
            instance_id
        )

        if data:
            return jsonify(data)

        return jsonify(
            {
                "error": (
                    "Instance not found"
                ),
            }
        ), 404

    except Exception as error:
        return jsonify(
            {
                "error": str(error),
            }
        ), 500


@app.route(
    "/download_snapshot/"
    "<path:filename>"
)
def download_snapshot(filename):
    try:
        filepath = (
            filename
            if os.path.isabs(filename)
            else os.path.join(
                "snapshots",
                filename,
            )
        )

        if os.path.exists(filepath):
            return send_file(
                filepath,
                as_attachment=True,
            )

        return jsonify(
            {
                "error": "File not found",
            }
        ), 404

    except Exception as error:
        return jsonify(
            {
                "error": str(error),
            }
        ), 500


@app.route(
    "/snapshots/<path:filename>"
)
def serve_snapshot(filename):
    try:
        filepath = os.path.join(
            "snapshots",
            filename,
        )

        if os.path.exists(filepath):
            return send_file(
                filepath,
                mimetype="image/jpeg",
            )

        return jsonify(
            {
                "error": "File not found",
            }
        ), 404

    except Exception as error:
        return jsonify(
            {
                "error": str(error),
            }
        ), 500


@app.route(
    "/api/delete_instance/"
    "<instance_id>",
    methods=["DELETE"],
)
def delete_instance(instance_id):
    try:
        success = db.delete_instance(
            instance_id
        )

        if success:
            return jsonify(
                {
                    "status": "success",
                    "message": (
                        "Instance deleted"
                    ),
                }
            )

        return jsonify(
            {
                "error": (
                    "Failed to delete "
                    "instance"
                ),
            }
        ), 500

    except Exception as error:
        return jsonify(
            {
                "error": str(error),
            }
        ), 500


# ==========================================================
# Start application
# ==========================================================

if __name__ == "__main__":
    try:
        db.init_db()
        load_model()

        instance_detector.update_settings(
            current_settings
        )

        socketio.run(
            app,
            debug=True,
            host="localhost",
            port=3333,
        )

    except Exception as error:
        print(
            f"Fatal error starting app: "
            f"{error}"
        )
