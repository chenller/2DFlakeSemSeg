from __future__ import annotations

from pathlib import Path
from pprint import pprint
from typing import List

import torch
from mmengine import Config
from mmseg.datasets.basesegdataset import Compose as PipelineCompose
from mmseg.models import BaseSegmentor
from mmseg.registry import MODELS
import mmsegext
import mmseg2dmat
from tqdm import tqdm

from mmsegalext.strategy import ALSTRATEGY, Strategy
from mmsegalext.alselect import ALSample


def get_image_path(root, root_rm):
    src = Path(root).glob('**/*.jpg')
    src = {i.name: i for i in src}
    src_rm = {}
    for _root_rm in root_rm:
        _src_rm = Path(_root_rm).glob('**/*.jpg')
        _src_rm = {i.name: i for i in _src_rm}
        src_rm.update(_src_rm)

    src_use = {k: v for k, v in src.items() if k not in src_rm}

    print(f'{len(src)=} {len(src_rm)=} {len(src_use)=}')
    return list(src_use.values())


# Execute the following code block when the script is run as the main program
if __name__ == '__main__':
    # Set the sampling quantity to 2, indicating that 2 images will be selected from the image data for processing
    select_num = 2

    # Specify the file path to save the sampling results
    save_filepath = './sample_image_filepath.py'

    # Image directory path, pointing to the folder containing .jpg format images
    image_dir: str = '/path/to/image/.jpg/format'

    # Define a string list used to specify image directories that need to be excluded (currently empty)
    image_dir_remove: List[str] = []

    # Pre-trained model configuration file path
    cfg_fp = '/path/to/pretrained_2dflake/graphene/upernet_flash_internimage_b_in1k_768.py'

    # Pre-trained model weight file path
    weight_fp = '/path/to/pretrained_2dflake/graphene/best_mIoU_iter.pth'

    # Print configuration file and weight file paths for confirmation
    print(cfg_fp)
    print(weight_fp)

    # Get the image path list and filter out items specified in image_dir_remove
    image_path = get_image_path(image_dir, image_dir_remove)

    # Convert all image paths to string type
    image_path = [str(p) for p in image_path]

    # Output the total number of images
    print(len(image_path))

    # Create ALSample object instance, using MarginSampling strategy for active learning sample selection
    # The commented version shows the usage of additional parameter cfg_merge (currently not enabled)
    # als = ALSample(cfg_fp=cfg_fp, weight_fp=weight_fp, strategy=dict(type='MarginSampling'), cfg_merge=cfg_merge)
    als = ALSample(cfg_fp=cfg_fp, weight_fp=weight_fp, strategy=dict(type='MarginSampling'), )

    # Call the sample method to sample images, select select_num images and save to the specified file
    als.sample(image_path, num=select_num, save_filepath=save_filepath)
