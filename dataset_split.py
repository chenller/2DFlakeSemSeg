from __future__ import annotations

import os
import shutil
import random
from pathlib import Path
from mmengine import Config

random.seed(12345678)


def replice(sample_image_filepath: list[str]) -> list[str]:
    new_paths = []
    fp = list(Path('/home/share/annlab_2dmat_2024/coco_old/MoS2/').glob('**/*.jpg'))
    fp_dict = {i.name: str(i) for i in fp}
    new_paths = [fp_dict[i.split('/')[-1]] for i in sample_image_filepath]
    return new_paths


def split(i=2, mat_name: str = 'MoS2'):
    # 原始图片路径列表
    new_root = f'/home/share/annlab_2dmat_2024/batchs/{mat_name}/{i}'
    cfg_root = f'./sample_image/sample_image_filepath{i}.py'
    cfg = Config.fromfile(cfg_root)

    sample_image_filepath = cfg['sample_image_filepath']
    # sample_image_filepath = [
    #     '/home/share/annlab_2dmat_2024/coco/graphene/val2024/002179.jpg',
    #     '/home/share/annlab_2dmat_2024/coco/graphene/val2024/005698.jpg',
    #     '/home/share/annlab_2dmat_2024/coco/graphene/val2024/007847.jpg',
    #     '/home/share/annlab_2dmat_2024/coco/graphene/val2024/003443.jpg',
    #     '/home/share/annlab_2dmat_2024/coco/graphene/val2024/003342.jpg',
    # ]
    # sample_image_filepath=replice(sample_image_filepath)
    # 打乱原始数据集
    sample_image_filepath.sort()
    random.shuffle(sample_image_filepath)
    random.shuffle(sample_image_filepath)

    # 定义目标目录
    target_dir_train = Path(new_root) / 'train2024'
    target_dir_val = Path(new_root) / 'val2024'
    target_dir_annotations_train = Path(new_root) / 'annotations_semseg/train2024'
    target_dir_annotations_val = Path(new_root) / 'annotations_semseg/val2024'

    # 创建目标目录
    os.makedirs(target_dir_train, exist_ok=True)
    os.makedirs(target_dir_val, exist_ok=True)
    os.makedirs(target_dir_annotations_train, exist_ok=True)
    os.makedirs(target_dir_annotations_val, exist_ok=True)

    # 切分数据集
    split_point = int(len(sample_image_filepath) * 0.8)  # 80% for train, 20% for validation (to match 4:1 ratio)

    train_files = sample_image_filepath[:split_point]
    val_files = sample_image_filepath[split_point:]

    # 复制文件到相应目录
    for img_path in train_files:
        if 'val' in img_path:
            annotation_path = img_path.replace('val2024', 'annotations_semseg/val2024').replace('.jpg', '.png')
        else:
            annotation_path = img_path.replace('train2024', 'annotations_semseg/train2024').replace('.jpg', '.png')
        new_img_path = os.path.join(target_dir_train, os.path.basename(img_path))
        new_annotation_path = os.path.join(target_dir_annotations_train, os.path.basename(annotation_path))
        print(img_path, new_img_path)
        print(annotation_path, new_annotation_path)
        shutil.copyfile(img_path, new_img_path)
        shutil.copyfile(annotation_path, new_annotation_path)

    for img_path in val_files:
        if 'val' in img_path:
            annotation_path = img_path.replace('val2024', 'annotations_semseg/val2024').replace('.jpg', '.png')
        else:
            annotation_path = img_path.replace('train2024', 'annotations_semseg/train2024').replace('.jpg', '.png')
        new_img_path = os.path.join(target_dir_val, os.path.basename(img_path))
        new_annotation_path = os.path.join(target_dir_annotations_val, os.path.basename(annotation_path))

        print(img_path, new_img_path)
        print(annotation_path, new_annotation_path)
        shutil.copyfile(img_path, new_img_path)
        shutil.copyfile(annotation_path, new_annotation_path)


def merge_dataset():
    srcs = [f'/home/share/annlab_2dmat_2024/batchs/MoS2/{i}' for i in range(1, 6)]
    dst = '/home/share/annlab_2dmat_2024/coco/MoS2'

    for src in srcs:
        fps = list(Path(src).glob('**/*.*'))
        fps = [i.absolute() for i in fps]
        for fp in fps:
            if fp.is_file():
                fp_new = str(fp).replace(src, dst)
                _fp_new_p = Path(fp_new).parent
                if not _fp_new_p.exists():
                    _fp_new_p.mkdir(parents=True, exist_ok=True)
                print((fp, fp_new))
                shutil.copyfile(fp, fp_new)


if __name__ == '__main__':
    split(5)
    # merge_dataset()
