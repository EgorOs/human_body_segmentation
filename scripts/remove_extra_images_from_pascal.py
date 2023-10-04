import os
from pathlib import Path

DATASET_PATH = Path(__file__).resolve().parents[1] / '.data_temp'
ANNOTATIONS_LIST = DATASET_PATH / 'pascal_person_part' / 'pascal_person_part_trainval_list' / 'trainval.txt'
PASCAL_VOC_DEV = DATASET_PATH / 'VOCdevkit' / 'VOC2010'
IMAGES_FOLDER = PASCAL_VOC_DEV / 'JPEGImages'


def main() -> None:
    files_to_keep = set()
    with open(ANNOTATIONS_LIST, 'r') as ann_f:
        for record in ann_f.readlines():
            img, gt = [Path(f_path.strip()).name for f_path in record.split(' ')]  # noqa: WPS221
            files_to_keep.update({img, gt})
    for img_path in IMAGES_FOLDER.glob('*'):
        if img_path.name not in files_to_keep:
            os.remove(img_path)


if __name__ == '__main__':
    main()
