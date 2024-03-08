import cv2
import numpy as np
from numpy.typing import NDArray


def convert_to_bw(image: NDArray) -> NDArray:
    color_threshold = np.mean(image)
    binary_mask = image > color_threshold
    return np.where(binary_mask, 255, 0).astype(np.uint8)


def apply_blur(image: NDArray, kernel_size: int = 5) -> NDArray:
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def create_noise_mask_from_bw_image(bw_image: NDArray, noise_fraction: float = .1) -> NDArray:
    black_pixels = np.sum(bw_image == 0)
    noise_pixels = int(black_pixels * noise_fraction)
    noise_positions = np.random.choice(bw_image.size, noise_pixels, replace=False)
    noise_mask = np.zeros_like(bw_image)
    noise_mask.flat[noise_positions] = 1
    return noise_mask


def apply_noise_mask(image: NDArray, noise_mask: NDArray) -> NDArray:
    return np.where(noise_mask, 255, image)
