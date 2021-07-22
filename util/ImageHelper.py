#  Copyright (c) 2021 by Yingdong Chen <chen.yingdong.cs9@is.naist.jp>,
#  Imaging-based Computational Biomedicine Laboratory, Nara Institution of Science and Technology.
#  All rights reserved.
#  This file can not be copied and/or distributed without the express permission of Yingdong Chen.

import cv2
import numpy as np
import io
import matplotlib.pyplot as plt


class ImageHelper:
    @staticmethod
    def label_images(
            images: np.ndarray,
            labels: np.ndarray,
            colors: list,
            thickness: int,
            condit_labels: np.ndarray = None,
            condict_label_color=None,
    ):
        # images (N, H, W, 3) or (N, H, W, 1)
        # labels (N, H, W)
        images = images.copy()
        if images.shape[-1] == 1:
            images = ImageHelper.convert_images(images, cv2.COLOR_GRAY2BGR)
        assert images.shape[-1] == 3
        image_count = images.shape[0]
        for i in range(image_count):
            if condit_labels is not None and condict_label_color is not None:
                images[i] = ImageHelper.label_image(images[i], condit_labels[i], condict_label_color, 2)
            images[i] = ImageHelper.label_image(images[i], labels[i], colors, thickness)
        return images

    @staticmethod
    def label_image(image, label, colors: list, thickness: int) -> np.ndarray:
        # image (H, W, 3) (np.uint8)
        # label (H, W)
        for i in range(1, len(colors) + 1):
            label_mask = label == i
            label_mask = label_mask.astype(np.uint8)
            cont, _ = cv2.findContours(label_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
            image = cv2.drawContours(image, cont, -1, colors[i-1], thickness)

        return image
        pass

    @staticmethod
    def convert_images(images, cv_code) -> np.ndarray or [np.ndarray, ...]:
        re = []
        if isinstance(images, list):
            num_images = len(images)
        elif isinstance(images, np.ndarray):
            num_images = images.shape[0]
        else:
            raise NotImplementedError("Unknown type for images: {}.".format(type(images)))

        for i in range(num_images):
            image = images[i]
            converted_image = cv2.cvtColor(image, cv_code)
            re.append(converted_image)
        if isinstance(images, list):
            return re
        else:
            return np.asarray(re)

    @staticmethod
    def plt_figure_to_opencv_array(fig, format="png", dpi=180):
        buf = io.BytesIO()
        fig.savefig(buf, format=format, dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        return cv2.imdecode(img_arr, 1)

    @staticmethod
    def wrap_imgs_with_plt(imgs: np.ndarray, title=None, subtitles=None, if_color_bar=False, clim=None, ticks=None, dpi=180):
        figure = plt.figure()
        if title is not None:
            figure.suptitle(title, fontsize=16)
        figure.tight_layout()
        assert imgs.ndim == 4
        N, H, W, C = imgs.shape
        if C != 1:
            if_color_bar = False
            clim = None
            ticks = None

        for i in range(N):
            plot = figure.add_subplot(1, N, i+1)
            if subtitles is not None:
                plot.title.set_text(subtitles[i])
                plot.axis("off")
                plot_img = plot.imshow(imgs[i])
                if clim is not None and (isinstance(clim, tuple) or isinstance(clim, list)):
                    plot_img.set_clim(*clim)
                if if_color_bar:
                    plot.colorbar(plot_img, ticks=ticks)
        img = ImageHelper.plt_figure_to_opencv_array(figure, dpi=dpi)
        plt.close(figure)
        return img

    @staticmethod
    def wrap_img_with_plt(img: np.ndarray, title=None, if_color_bar=False, ticks=None, clim=None, dpi=180):
        figure = plt.figure()
        figure.tight_layout()

        plot = figure.add_subplot(1, 1, 1)
        plot.axis("off")
        if title is not None:
            figure.suptitle(title, fontsize=16)

        plt_img = plot.imshow(img)
        if clim is not None:
            plt_img.set_clim(*clim)
        if if_color_bar:
            figure.colorbar(plt_img, ticks=ticks)

        ret = ImageHelper.plt_figure_to_opencv_array(figure, dpi=dpi)
        plt.close(figure)
        return ret

    @staticmethod
    def resize(image: np.ndarray, resize: int):
        _, _, channel = image.shape
        if resize is not None:
            image = cv2.resize(image, (resize, resize))
            if channel == 1:
                image = np.expand_dims(image, axis=2)
        return image

    @staticmethod
    def standardize(image: np.ndarray, mean=None, std=None):
        if mean is None:
            mean = np.mean(image)
        if std is None:
            std = np.std(image)
        return (image - mean) / std

    @staticmethod
    def normalize_hu(image):
        image_clim = (-150, 350.)

        image = np.clip(image, image_clim[0], image_clim[1])

        image = (image - image_clim[0]) / (image_clim[1] - image_clim[0])  # [0, 1]
        image *= 255.
        image = image.astype("float32")

        return image  # [0, 255]


if __name__ == '__main__':
    path = r'D:\scallop-chen\crop_CT_label\K9086\crop_original_image.mhd'
    from MHDHelper import MHDHelper

    img, _ = MHDHelper.read(path)
    image = img[250, :, :]
    nor_image = ImageHelper.normalize_hu(image)
    import matplotlib.pyplot as plt
    plt.subplot(121)
    plt.imshow(image)
    plt.colorbar()
    plt.subplot(122)
    plt.imshow(nor_image)
    plt.colorbar()
    plt.show()

