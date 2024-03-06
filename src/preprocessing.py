import multiprocessing
from pathlib import Path
from typing import Any, Tuple

import cv2
from tqdm import tqdm


def resize_images(input_dir: Path, output_dir: Path, img_size: Tuple[int, int]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    image_paths = (
        list(
            input_dir.glob('*.jpg'),
        )
        + list(input_dir.glob('*.jpeg'))
        + list(input_dir.glob('*.png'))
    )
    with multiprocessing.Pool() as pool, tqdm(total=len(image_paths), desc='Resizing images') as pbar:  # noqa: WPS316
        resize_args = [(image_path, output_dir, img_size) for image_path in image_paths]
        for _ in pool.imap_unordered(_resize_and_save_image, resize_args):
            pbar.update(1)


def _resize_and_save_image(resize_args: Any) -> None:
    image_path, output_dir, img_size = resize_args
    image = cv2.imread(str(image_path))
    resized_image = cv2.resize(image, img_size)
    filename = image_path.name
    cv2.imwrite(str(output_dir / filename), resized_image)
