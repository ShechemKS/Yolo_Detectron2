import detectron2.data.transforms as T
import cv2
import numpy as np


def build_yolo_aug(cfg):
    augs = [T.Resize(cfg.INPUT.SIZE, cfg.INPUT.SIZE)]
    if cfg.INPUT.DEGREES > 0.0:
        augs.append(T.RandomRotation(cfg.INPUT.DEGREES))
    if cfg.INPUT.TRANSLATE > 0.0 or cfg.INPUT.SCALE > 0.0:
        scale = (cfg.INPUT.SCALE, cfg.INPUT.SCALE)
        shift = (cfg.INPUT.TRANSLATE, cfg.INPUT.TRANSLATE)
        augs.append(T.RandomExtent(scale, shift))
    if cfg.INPUT.FLIPUD > 0.0:
        augs.append(T.RandomFlip(cfg.INPUT.FLIPUD, horizontal=False, vertical=True))
    if cfg.INPUT.FLIPLR > 0.0:
        augs.append(T.RandomFlip(cfg.INPUT.FLIPLR, horizontal=True, vertical=False))
    if cfg.INPUT.HSV_H or cfg.INPUT.HSV_S or cfg.INPUT.HSV_V:
        augs.append(ColorAugmentation(cfg.INPUT))


class ColorAugmentation(T.Augmentation):

    def __init__(self, cfg):
        super().__init__()
        self.img_format = cfg.FORMAT
        if self.img_format == "BGR":
            self.cvt = cv2.COLOR_BGR2HSV
            self.back_cvt = cv2.COLOR_HSV2BGR
        elif self.img_format == "RGB":
            self.cvt = cv2.COLOR_RGB2HSV
            self.back_cvt = cv2.COLOR_HSV2RGB
        else:
            raise NotImplementedError
        self.h = cfg.HSV_H
        self.s = cfg.HSV_S
        self.v = cfg.HSV_V
        self.r = [self.h, self.s, self.v]

    def get_transform(self, image):
        r = np.random.uniform(-1, 1, 3) * self.r + 1
        return HSVTransform(r, self.cvt, self.back_cvt)


class HSVTransform(T.Transform):
    def __init__(
        self,
        r,
        cvt,
        back_cvt
    ):
        super().__init__()
        self._set_attributes(locals())

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_image(self, img, interp=None):
        dtype = img.dtype
        h, s, v = cv2.split(cv2.cvtColor(img, self.cvt))
        img_hsv = cv2.merge(
            (h * self.r[0]) % 180,
            np.clip(s * self.r[1], 0, 255),
            np.clip(v * self.r[2], 0, 255)
        ).astype(dtype)     # Apply the HSV tranformation
        img = cv2.cvtColor(img_hsv, self.back_cvt)
        return img
