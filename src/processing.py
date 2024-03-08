from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from os import listdir
from pathlib import Path

from src.transformations import (
    convert_to_bw,
    create_noise_mask_from_bw_image,
    apply_blur,
    apply_noise_mask
)
from src.utils import timeit, load_grayscale_image


def _transform_image(image_filepath: Path) -> None:
    image = load_grayscale_image(image_filepath)
    bw_image = convert_to_bw(image)
    apply_noise_mask(
        image=apply_blur(bw_image),
        noise_mask=create_noise_mask_from_bw_image(bw_image, noise_fraction=0.1)
    )


@timeit
def process_images_sequential(image_dir: Path) -> None:
    for filename in listdir(image_dir):
        _transform_image(image_filepath=image_dir / filename)


@timeit
def process_images_parallel_1(image_dir: Path) -> None:
    with ThreadPoolExecutor() as executor:
        executor.map(
            _transform_image,
            [image_dir / filename for filename in listdir(image_dir)]
        )


@timeit
def process_images_parallel_2(image_dir: Path) -> None:
    with ProcessPoolExecutor() as executor:
        executor.map(
            _transform_image,
            [image_dir / filename for filename in listdir(image_dir)]
        )
