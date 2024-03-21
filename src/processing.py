from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from functools import partial
from os import listdir, makedirs
from pathlib import Path
from typing import Callable, Type, Optional

from numpy.typing import NDArray

from src.paths import OUTPUT_DIR
from src.utils import timeit, load_image, save_image


def _load_transform_and_save(
        image_filepath: Path,
        transform: Callable[[NDArray], NDArray],
        output_dir: Path
) -> None:
    image = load_image(image_filepath)
    image = transform(image)
    save_image(filepath=output_dir / image_filepath.name, image=image)
    return


@timeit
def process_images_sequential(image_dir: Path, transform: Callable[[NDArray], NDArray]) -> None:
    output_dir = OUTPUT_DIR / transform.__name__
    makedirs(output_dir, exist_ok=True)

    for filename in listdir(image_dir):
        _load_transform_and_save(image_dir / filename, transform, output_dir)


@timeit
def process_images_parallel_1(
        image_dir: Path,
        transform: Callable[[NDArray], NDArray],
        pool_executor: Type[ProcessPoolExecutor | ThreadPoolExecutor],
        num_workers: Optional[int] = None
) -> None:
    output_dir = OUTPUT_DIR / transform.__name__
    makedirs(output_dir, exist_ok=True)

    with pool_executor(num_workers) as executor:
        executor.map(
            partial(_load_transform_and_save, transform=transform, output_dir=output_dir),
            [image_dir / filename for filename in listdir(image_dir)]
        )
