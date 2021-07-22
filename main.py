#  Copyright (c) 2021 by Yingdong Chen <chen.yingdong.cs9@is.naist.jp>,
#  Imaging-based Computational Biomedicine Laboratory, Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yingdong Chen.

import os
from util.MHDHelper import MHDHelper
from util.ImageHelper import ImageHelper
from util.OSHelper import OSHelper
import cv2
import ants
import numpy as np
from tqdm import tqdm


def main():
    data_root = "/mnt/d/NARA_ENHANCE"
    output_dir = "temp"

    vol, _ = MHDHelper.read(os.path.join(data_root, "NaraMed_MHD_paired_Enhanced", "N0001_Enhanced.mhd"))
    vol = ImageHelper.normalize_hu(vol).astype(np.uint8)  # [0, 255]

    print(vol.shape)

    OSHelper.mkdirs(output_dir)
    for i in tqdm(range(vol.shape[0]), desc="Saving Image"):
        cv2.imwrite(os.path.join(output_dir, "{0:04d}.png".format(i)), vol[i])
    pass


if __name__ == '__main__':
    main()
