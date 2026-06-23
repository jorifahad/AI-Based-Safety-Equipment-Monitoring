
from pathlib import Path

import cv2
import numpy as np
from sklearn.cluster import MiniBatchKMeans
from sklearn.preprocessing import StandardScaler
from skimage.restoration import estimate_sigma


# ==========================================================
# Paths
# ==========================================================

BASE_DIR = Path(__file__).resolve().parent

INPUT_IMAGE = BASE_DIR / "scenario_mixed.png"
OUTPUT_IMAGE = BASE_DIR / "denoised_image.png"
IMPULSE_MASK_IMAGE = BASE_DIR / "detected_impulse_mask.png"
AFTER_IMPULSE_IMAGE = BASE_DIR / "after_impulse_removal.png"


# ==========================================================
# General settings
# ==========================================================

NOISE_TYPE = "auto"  # "auto", "gaussian", "salt_pepper", or "mixed"

PATCH_SIZE = 7
PATCH_STEP = 3
N_CLUSTERS = 64
MAX_KMEANS_SAMPLES = 60_000
RANDOM_STATE = 42

# None = automatic estimation.
GAUSSIAN_SIGMA = None

# Automatic detection thresholds.
AUTO_IMPULSE_RATIO_THRESHOLD = 0.001
AUTO_GAUSSIAN_SIGMA_THRESHOLD = 4.0

# Salt-and-pepper detection sensitivity.
IMPULSE_MIN_DIFFERENCE = 18.0
IMPULSE_MAX_RATIO = 0.20


# ==========================================================
# Read and save
# ==========================================================

def read_color_image(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(
            f"Image not found:\n{path.resolve()}"
        )

    data = np.fromfile(str(path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError(
            f"Could not read image:\n{path.resolve()}"
        )

    return image


def save_png(path: Path, image: np.ndarray) -> None:
    ok, encoded = cv2.imencode(".png", image)

    if not ok:
        raise IOError(
            f"Could not save image:\n{path.resolve()}"
        )

    encoded.tofile(str(path))


# ==========================================================
# Automatic noise analysis
# ==========================================================

def estimate_gaussian_sigma(image: np.ndarray) -> float:
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB,
    ).astype(np.float32) / 255.0

    sigma = estimate_sigma(
        rgb,
        channel_axis=-1,
        average_sigmas=True,
    )

    return float(sigma * 255.0)


def estimate_impulse_ratio(image: np.ndarray) -> float:
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)

    gray_u8 = gray.astype(np.uint8)

    local_median = cv2.medianBlur(
        gray_u8,
        3,
    ).astype(np.float32)

    difference = np.abs(
        gray - local_median
    )

    near_extreme = (
        (gray <= 20)
        | (gray >= 235)
    )

    strong_outlier = (
        difference >= 20
    )

    impulse_pixels = (
        near_extreme
        & strong_outlier
    )

    return float(
        np.mean(impulse_pixels)
    )


def detect_noise_type(
    image: np.ndarray,
) -> tuple[str, float, float]:
    impulse_ratio = estimate_impulse_ratio(
        image
    )

    gaussian_sigma = estimate_gaussian_sigma(
        image
    )

    has_impulse = (
        impulse_ratio
        >= AUTO_IMPULSE_RATIO_THRESHOLD
    )

    has_gaussian = (
        gaussian_sigma
        >= AUTO_GAUSSIAN_SIGMA_THRESHOLD
    )

    if has_impulse and has_gaussian:
        detected_type = "mixed"
    elif has_impulse:
        detected_type = "salt_pepper"
    elif has_gaussian:
        detected_type = "gaussian"
    else:
        detected_type = "clean"

    return (
        detected_type,
        impulse_ratio,
        gaussian_sigma,
    )


# ==========================================================
# Salt-and-pepper detection using K-Means
# ==========================================================

def detect_impulse_pixels_kmeans(
    image: np.ndarray,
    n_clusters: int = 3,
) -> np.ndarray:
    gray = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2GRAY,
    ).astype(np.float32)

    gray_u8 = gray.astype(np.uint8)

    med3 = cv2.medianBlur(
        gray_u8,
        3,
    ).astype(np.float32)

    med5 = cv2.medianBlur(
        gray_u8,
        5,
    ).astype(np.float32)

    diff3 = np.abs(
        gray - med3
    )

    diff5 = np.abs(
        gray - med5
    )

    local_mean = cv2.blur(
        gray,
        (5, 5),
    )

    local_sq_mean = cv2.blur(
        gray * gray,
        (5, 5),
    )

    local_std = np.sqrt(
        np.maximum(
            local_sq_mean
            - local_mean * local_mean,
            0.0,
        )
    )

    extreme_distance = np.minimum(
        gray,
        255.0 - gray,
    )

    features = np.column_stack(
        [
            diff3.reshape(-1),
            diff5.reshape(-1),
            local_std.reshape(-1),
            extreme_distance.reshape(-1),
        ]
    ).astype(np.float32)

    scaler = StandardScaler()
    scaled = scaler.fit_transform(
        features
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    if len(scaled) > MAX_KMEANS_SAMPLES:
        idx = rng.choice(
            len(scaled),
            MAX_KMEANS_SAMPLES,
            replace=False,
        )
        train = scaled[idx]
    else:
        train = scaled

    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        init="k-means++",
        n_init=10,
        batch_size=4096,
        max_iter=200,
        random_state=RANDOM_STATE,
    )

    model.fit(train)
    labels = model.predict(scaled)

    stats = []

    for cluster_id in range(n_clusters):
        mask = labels == cluster_id

        if not np.any(mask):
            continue

        cluster = features[mask]

        median_diff = float(
            np.median(cluster[:, 0])
        )

        larger_diff = float(
            np.median(cluster[:, 1])
        )

        extreme = float(
            np.median(cluster[:, 3])
        )

        ratio = float(
            np.mean(mask)
        )

        score = (
            1.5 * median_diff
            + larger_diff
            - 0.45 * extreme
        )

        stats.append(
            (
                score,
                cluster_id,
                median_diff,
                larger_diff,
                extreme,
                ratio,
            )
        )

    if not stats:
        return np.zeros_like(
            gray_u8,
            dtype=np.uint8,
        )

    stats.sort(reverse=True)
    impulse_cluster = stats[0][1]

    height, width = gray.shape

    cluster_mask = (
        labels.reshape(height, width)
        == impulse_cluster
    )

    candidate = (
        cluster_mask
        & (
            diff3
            >= IMPULSE_MIN_DIFFERENCE
        )
        & (
            (gray <= 30)
            | (gray >= 225)
            | (diff5 >= 35)
        )
    )

    if float(np.mean(candidate)) > IMPULSE_MAX_RATIO:
        candidate &= diff3 >= 35

    return (
        candidate.astype(np.uint8)
        * 255
    )


def remove_impulse_noise(
    image: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    ratio = float(
        np.mean(mask > 0)
    )

    print(
        f"Detected impulse ratio: "
        f"{ratio:.3%}"
    )

    if ratio == 0:
        return image.copy()

    kernel_size = (
        3 if ratio < 0.05 else 5
    )

    median_image = cv2.medianBlur(
        image,
        kernel_size,
    )

    result = image.copy()
    result[mask > 0] = median_image[mask > 0]

    return result


# ==========================================================
# Patch extraction
# ==========================================================

def make_positions(
    height: int,
    width: int,
    patch_size: int,
    step: int,
) -> list[tuple[int, int]]:
    rows = list(
        range(
            0,
            max(
                height - patch_size + 1,
                1,
            ),
            step,
        )
    )

    cols = list(
        range(
            0,
            max(
                width - patch_size + 1,
                1,
            ),
            step,
        )
    )

    last_row = max(
        height - patch_size,
        0,
    )

    last_col = max(
        width - patch_size,
        0,
    )

    if last_row not in rows:
        rows.append(last_row)

    if last_col not in cols:
        cols.append(last_col)

    return [
        (row, col)
        for row in rows
        for col in cols
    ]


def extract_patches(
    channel: np.ndarray,
    positions: list[tuple[int, int]],
    patch_size: int,
) -> np.ndarray:
    patches = np.empty(
        (
            len(positions),
            patch_size * patch_size,
        ),
        dtype=np.float32,
    )

    for index, (row, col) in enumerate(
        positions
    ):
        patch = channel[
            row:row + patch_size,
            col:col + patch_size,
        ]

        patches[index] = patch.reshape(-1)

    return patches


# ==========================================================
# K-Means patch grouping
# ==========================================================

def cluster_patches_kmeans(
    y_patches: np.ndarray,
) -> np.ndarray:
    patch_means = y_patches.mean(
        axis=1,
        keepdims=True,
    )

    centered = (
        y_patches - patch_means
    )

    norms = np.linalg.norm(
        centered,
        axis=1,
        keepdims=True,
    )

    features = (
        centered
        / np.maximum(norms, 1e-6)
    )

    rng = np.random.default_rng(
        RANDOM_STATE
    )

    if len(features) > MAX_KMEANS_SAMPLES:
        idx = rng.choice(
            len(features),
            MAX_KMEANS_SAMPLES,
            replace=False,
        )
        train = features[idx]
    else:
        train = features

    actual_clusters = min(
        N_CLUSTERS,
        len(train),
    )

    model = MiniBatchKMeans(
        n_clusters=actual_clusters,
        init="k-means++",
        n_init=5,
        batch_size=4096,
        max_iter=200,
        random_state=RANDOM_STATE,
    )

    model.fit(train)

    return model.predict(
        features
    )


# ==========================================================
# PCA-Wiener denoising inside each K-Means cluster
# ==========================================================

def pca_wiener_cluster_denoise(
    patches: np.ndarray,
    labels: np.ndarray,
    sigma: float,
) -> np.ndarray:
    output = np.empty_like(
        patches,
        dtype=np.float32,
    )

    noise_variance = float(
        sigma * sigma
    )

    for cluster_id in np.unique(labels):
        idx = np.flatnonzero(
            labels == cluster_id
        )

        group = patches[idx]

        if len(group) < 8:
            output[idx] = group
            continue

        patch_means = group.mean(
            axis=1,
            keepdims=True,
        )

        group_centered = (
            group - patch_means
        )

        cluster_mean = group_centered.mean(
            axis=0,
            keepdims=True,
        )

        x = (
            group_centered
            - cluster_mean
        )

        covariance = (
            x.T @ x
        ) / max(
            len(group) - 1,
            1,
        )

        eigenvalues, eigenvectors = np.linalg.eigh(
            covariance
        )

        order = np.argsort(
            eigenvalues
        )[::-1]

        eigenvalues = eigenvalues[order]
        eigenvectors = eigenvectors[:, order]

        coefficients = (
            x @ eigenvectors
        )

        signal_variance = np.maximum(
            eigenvalues
            - noise_variance,
            0.0,
        )

        gain = (
            signal_variance
            / (
                signal_variance
                + noise_variance
                + 1e-8
            )
        )

        reconstructed = (
            (
                coefficients
                * gain[np.newaxis, :]
            )
            @ eigenvectors.T
            + cluster_mean
            + patch_means
        )

        output[idx] = reconstructed

    return output


# ==========================================================
# Reconstruct channel
# ==========================================================

def reconstruct_channel(
    patches: np.ndarray,
    positions: list[tuple[int, int]],
    height: int,
    width: int,
    patch_size: int,
) -> np.ndarray:
    image_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    weight_sum = np.zeros(
        (height, width),
        dtype=np.float32,
    )

    one_dimensional_window = np.hanning(
        patch_size + 2
    )[1:-1].astype(np.float32)

    weight = np.maximum(
        np.outer(
            one_dimensional_window,
            one_dimensional_window,
        ),
        0.05,
    )

    for patch_vector, (row, col) in zip(
        patches,
        positions,
    ):
        patch = patch_vector.reshape(
            patch_size,
            patch_size,
        )

        image_sum[
            row:row + patch_size,
            col:col + patch_size,
        ] += patch * weight

        weight_sum[
            row:row + patch_size,
            col:col + patch_size,
        ] += weight

    result = (
        image_sum
        / np.maximum(
            weight_sum,
            1e-8,
        )
    )

    return np.clip(
        result,
        0,
        255,
    ).astype(np.uint8)


# ==========================================================
# Gaussian denoising using K-Means
# ==========================================================

def remove_gaussian_noise_kmeans(
    image: np.ndarray,
    sigma: float | None,
) -> np.ndarray:
    if sigma is None:
        sigma = estimate_gaussian_sigma(
            image
        )

    sigma = float(
        np.clip(
            sigma,
            1.0,
            40.0,
        )
    )

    print(
        f"Gaussian sigma used: "
        f"{sigma:.2f}"
    )

    ycrcb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2YCrCb,
    ).astype(np.float32)

    height, width = image.shape[:2]

    positions = make_positions(
        height,
        width,
        PATCH_SIZE,
        PATCH_STEP,
    )

    print(
        f"Number of patches: "
        f"{len(positions)}"
    )

    y_patches = extract_patches(
        ycrcb[:, :, 0],
        positions,
        PATCH_SIZE,
    )

    labels = cluster_patches_kmeans(
        y_patches
    )

    output_channels = []

    channel_sigmas = [
        sigma,
        sigma * 0.75,
        sigma * 0.75,
    ]

    for channel_index, channel_sigma in enumerate(
        channel_sigmas
    ):
        print(
            f"Processing channel "
            f"{channel_index + 1}/3"
        )

        patches = extract_patches(
            ycrcb[:, :, channel_index],
            positions,
            PATCH_SIZE,
        )

        denoised_patches = (
            pca_wiener_cluster_denoise(
                patches,
                labels,
                channel_sigma,
            )
        )

        channel = reconstruct_channel(
            denoised_patches,
            positions,
            height,
            width,
            PATCH_SIZE,
        )

        output_channels.append(
            channel
        )

    denoised_ycrcb = cv2.merge(
        output_channels
    )

    return cv2.cvtColor(
        denoised_ycrcb,
        cv2.COLOR_YCrCb2BGR,
    )


# ==========================================================
# Main
# ==========================================================

def main() -> None:
    valid_noise_types = {
        "auto",
        "gaussian",
        "salt_pepper",
        "mixed",
    }

    if NOISE_TYPE not in valid_noise_types:
        raise ValueError(
            'NOISE_TYPE must be "auto", '
            '"gaussian", "salt_pepper", '
            'or "mixed".'
        )

    image = read_color_image(
        INPUT_IMAGE
    )

    result = image.copy()

    print("=" * 65)
    print("K-Means Automatic Image Denoising")
    print("=" * 65)

    print(
        f"Input image:\n"
        f"{INPUT_IMAGE.resolve()}"
    )

    print(
        f"Input shape: "
        f"{image.shape}"
    )

    if NOISE_TYPE == "auto":
        (
            selected_noise_type,
            impulse_ratio,
            estimated_sigma,
        ) = detect_noise_type(
            image
        )

        print("\nAutomatic noise analysis:")

        print(
            f"Impulse-noise ratio: "
            f"{impulse_ratio:.4%}"
        )

        print(
            f"Estimated Gaussian sigma: "
            f"{estimated_sigma:.2f}"
        )

        print(
            f"Detected noise type: "
            f"{selected_noise_type}"
        )

    else:
        selected_noise_type = NOISE_TYPE
        estimated_sigma = GAUSSIAN_SIGMA

        print(
            f"Selected noise type: "
            f"{selected_noise_type}"
        )

    if selected_noise_type in {
        "salt_pepper",
        "mixed",
    }:
        print(
            "\nDetecting impulse noise "
            "using K-Means..."
        )

        impulse_mask = (
            detect_impulse_pixels_kmeans(
                result
            )
        )

        result = remove_impulse_noise(
            result,
            impulse_mask,
        )

        save_png(
            IMPULSE_MASK_IMAGE,
            impulse_mask,
        )

        save_png(
            AFTER_IMPULSE_IMAGE,
            result,
        )

    if selected_noise_type in {
        "gaussian",
        "mixed",
    }:
        if NOISE_TYPE == "auto":
            gaussian_sigma = estimated_sigma
        else:
            gaussian_sigma = GAUSSIAN_SIGMA

        result = remove_gaussian_noise_kmeans(
            result,
            gaussian_sigma,
        )

    save_png(
        OUTPUT_IMAGE,
        result,
    )

    print("\n" + "=" * 65)
    print("Processing completed.")

    print(
        f"Final noise type: "
        f"{selected_noise_type}"
    )

    print(
        f"Saved image:\n"
        f"{OUTPUT_IMAGE.resolve()}"
    )

    print(
        f"Output shape: "
        f"{result.shape}"
    )

    cv2.imshow(
        "Denoised Image",
        result,
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
