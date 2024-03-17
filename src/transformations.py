import cv2
import numpy as np
from numpy.typing import NDArray


def convert_to_bw(image: NDArray) -> NDArray:
    grayscale_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    color_threshold = np.mean(grayscale_image)
    binary_mask = grayscale_image > color_threshold
    return np.where(binary_mask, 255, 0).astype(np.uint8)


def apply_blur(image: NDArray, kernel_size: int = 11) -> NDArray:
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def apply_noise(image: NDArray, noise_fraction: float = .1) -> NDArray:
    bw_image = convert_to_bw(image)
    black_pixels = np.sum(bw_image == 0)
    noise_pixels = int(black_pixels * noise_fraction)
    noise_mask = np.zeros_like(bw_image, dtype=bool)
    noise_mask.flat[np.random.choice(bw_image.size, noise_pixels, replace=False)] = True
    image_with_noise = image.copy()
    image_with_noise[noise_mask] = np.random.randint(0, 256, size=(noise_pixels, 3))
    return image_with_noise
