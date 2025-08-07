from pathlib import Path
import math

import cv2
import numpy as np


def concat_images(img_filepath_list, save_path='output.png', *, value_scale: float = None,
                  img_width=120, img_height=80, cols: int = None, rows: int = None):
    n = len(img_filepath_list)

    # 计算行数和列数
    if cols != None and rows != None:
        assert cols * rows >= n
    elif cols != None:
        rows = n // cols + 1
    elif rows != None:
        cols = n // rows + 1
    else:
        cols = rows = int(math.ceil(math.sqrt(n)))

    # 创建一个空白的图像矩阵
    total_width = cols * img_width
    total_height = rows * img_height
    new_img = np.ones((total_height, total_width, 3), dtype=np.uint8) * 255  # 白色背景

    # 拼接图像
    for i, img_path in enumerate(img_filepath_list):
        img = cv2.imread(str(img_path))
        img = cv2.resize(img, (img_width, img_height))  # 调整图像大小
        x = (i % cols) * img_width
        y = (i // cols) * img_height
        new_img[y:y + img_height, x:x + img_width] = img
    if value_scale:
        new_img = new_img * value_scale
        new_img = new_img.astype(np.uint8)
    # 保存或显示拼接后的图像
    cv2.imwrite(save_path, new_img)
    # cv2.imshow('Merged Image', new_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    img_filename_list = ['003732.jpg', '006302.jpg', '003679.jpg', '006301.jpg', '006830.jpg', '005899.jpg',
                         '003347.jpg', '005028.jpg', '006423.jpg', '006128.jpg', '005288.jpg', '005334.jpg',
                         '006735.jpg', '006081.jpg', '007289.jpg', '002212.jpg', '002234.jpg', '003622.jpg',
                         '006129.jpg', '003373.jpg', '003619.jpg', '003756.jpg', '006053.jpg', '006592.jpg',
                         '007034.jpg', '003755.jpg', '007259.jpg', '008187.jpg', '003099.jpg', '003603.jpg',
                         '004507.jpg', '004786.jpg', '005403.jpg', '003430.jpg', '005598.jpg', '003773.jpg',
                         '003910.jpg', '006722.jpg', '002208.jpg', '002832.jpg', '005139.jpg', '005872.jpg',
                         '006765.jpg', '002893.jpg', '003292.jpg', '004683.jpg', '004880.jpg', '005774.jpg',
                         '006568.jpg', '002214.jpg', '005053.jpg', '006259.jpg', '006685.jpg', '003865.jpg',
                         '005177.jpg', '006443.jpg', '003453.jpg', '003652.jpg', '005704.jpg', '008078.jpg',
                         '004985.jpg', '002125.jpg', '002177.jpg', '002235.jpg', '003826.jpg', '005190.jpg',
                         '006363.jpg', '007068.jpg', '003336.jpg', '005119.jpg', '005245.jpg', '006317.jpg',
                         '005631.jpg', '006913.jpg', '005996.jpg', '003493.jpg', '003848.jpg', '005017.jpg',
                         '004835.jpg', '005131.jpg', '005718.jpg', '005790.jpg', '005799.jpg', '005895.jpg',
                         '006098.jpg', '006314.jpg', '006365.jpg', '008241.jpg', '003446.jpg', '006276.jpg',
                         '006670.jpg', '007135.jpg', '007230.jpg', '002570.jpg', '003733.jpg', '007116.jpg',
                         '003088.jpg', '003558.jpg', '003841.jpg', '004496.jpg', '004921.jpg', '005276.jpg',
                         '006458.jpg', '007019.jpg', '008200.jpg', '005446.jpg', '007107.jpg', '008108.jpg',
                         '002935.jpg', '002951.jpg', '005658.jpg', '003270.jpg', '004264.jpg', '005101.jpg',
                         '006566.jpg', '007137.jpg', '006538.jpg', '004621.jpg', '003600.jpg', '004417.jpg',
                         '004971.jpg', '005675.jpg', '006143.jpg', '003212.jpg', '004005.jpg', '005480.jpg',
                         '006819.jpg', '006905.jpg', '007293.jpg', '008228.jpg', '003103.jpg', '003422.jpg',
                         '003550.jpg', '004511.jpg', '006190.jpg', '006454.jpg', '006717.jpg', '006972.jpg',
                         '007024.jpg', '003166.jpg', '003857.jpg', '004557.jpg', '005112.jpg', '006506.jpg',
                         '008225.jpg', '003580.jpg', '004250.jpg', '005927.jpg', '003581.jpg', '005348.jpg',
                         '005429.jpg', '006841.jpg', '007003.jpg', '003286.jpg', '003344.jpg', '004233.jpg',
                         '005663.jpg', '005685.jpg', '006686.jpg', '003847.jpg', '006398.jpg', '002853.jpg',
                         '004422.jpg', '005968.jpg', '006018.jpg', '006892.jpg', '006902.jpg']
    img_root = '/home/share/annlab_2dmat_2024/coco/graphene/train2024/'
    mask_root = '/home/share/annlab_2dmat_2024/coco/graphene/annotations_semseg/train2024/'

    img_filepath_list = [Path(img_root) / i for i in img_filename_list]
    concat_images(img_filepath_list, save_path='graphene_overview_img.jpg')

    img_filepath_list = [(Path(mask_root) / i).with_suffix('.png') for i in img_filename_list]
    concat_images(img_filepath_list, save_path='graphene_overview_mask.jpg', value_scale=255)
