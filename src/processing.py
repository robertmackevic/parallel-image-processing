from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from multiprocessing import Process
from os import listdir
from pathlib import Path
from typing import Callable, List, Type

from numpy.typing import NDArray

from src.transformations import (
    convert_to_bw,
    apply_blur,
    apply_noise
)
from src.utils import timeit, load_image


def _apply_transformations(image_filepath: Path) -> None:
    image = load_image(image_filepath)
    convert_to_bw(image)
    apply_blur(image)
    apply_noise(image)


def _load_and_transform(image_filepath: Path, transform: Callable[[NDArray], NDArray]) -> NDArray:
    return transform(load_image(image_filepath))


def _apply_transformation_with_pooling(
        image_filepaths: List[Path],
        transform: Callable[[NDArray], NDArray],
        pool_executor: Type
) -> None:
    with pool_executor() as executor:
        executor.map(partial(_load_and_transform, transform=transform), image_filepaths)


@timeit
def process_images_sequential(image_dir: Path) -> None:
    for filename in listdir(image_dir):
        _apply_transformations(image_filepath=image_dir / filename)


@timeit
def process_images_parallel_1_multithread(image_dir: Path) -> None:
    with ThreadPoolExecutor() as executor:
        executor.map(
            _apply_transformations,
            [image_dir / filename for filename in listdir(image_dir)]
        )


@timeit
def process_images_parallel_1_multiprocess(image_dir: Path) -> None:
    with ProcessPoolExecutor() as executor:
        executor.map(
            _apply_transformations,
            [image_dir / filename for filename in listdir(image_dir)]
        )


@timeit
def process_images_parallel_2_multithread(image_dir: Path) -> None:
    filepaths = [image_dir / filename for filename in listdir(image_dir)]

    _apply_transformation_with_pooling(filepaths, convert_to_bw, ThreadPoolExecutor)
    _apply_transformation_with_pooling(filepaths, apply_blur, ThreadPoolExecutor)
    _apply_transformation_with_pooling(filepaths, apply_noise, ThreadPoolExecutor)


@timeit
def process_images_parallel_2_multiprocess(image_dir: Path) -> None:
    filepaths = [image_dir / filename for filename in listdir(image_dir)]

    _apply_transformation_with_pooling(filepaths, convert_to_bw, ProcessPoolExecutor)
    _apply_transformation_with_pooling(filepaths, apply_blur, ProcessPoolExecutor)
    _apply_transformation_with_pooling(filepaths, apply_noise, ProcessPoolExecutor)


@timeit
def process_images_parallel_3(image_dir: Path) -> None:
    filepaths = [image_dir / filename for filename in listdir(image_dir)]

    processes = [
        Process(target=_apply_transformation_with_pooling, args=(filepaths, transform, ThreadPoolExecutor))
        for transform in (convert_to_bw, apply_blur, apply_noise)
    ]

    for process in processes:
        process.start()

    for process in processes:
        process.join()
