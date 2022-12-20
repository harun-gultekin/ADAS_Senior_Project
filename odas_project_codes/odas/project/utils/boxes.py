"""boxes.py

drawing bounding boxes based on object detection results.
"""

import numpy as np
import cv2


# Constants
ALPHA = 0.5
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)


def gen_colors(num_colors):

    import colorsys

    hsvs = [[float(x) / num_colors, 1., 0.7] for x in range(num_colors)]
    random.seed(1234)
    random.shuffle(hsvs)
    rgbs = list(map(lambda x: list(colorsys.hsv_to_rgb(*x)), hsvs))
    bgrs = [(int(rgb[2] * 255), int(rgb[1] * 255),  int(rgb[0] * 255))
            for rgb in rgbs]
    return bgrs


class BoxVisualization():


    def __init__(self, cls_dict):
        self.cls_dict = cls_dict
        self.colors = gen_colors(len(cls_dict))

    def draw_boxes(self, img, boxes, confs, clss):
        for bb, cf, cl in zip(boxes, confs, clss):
            cl = int(cl)
            x_min, y_min, x_max, y_max = bb[0], bb[1], bb[2], bb[3]
            color = self.colors[cl]
            cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
            txt_loc = (max(x_min+2, 0), max(y_min+2, 0))
            cls_name = self.cls_dict.get(cl, 'CLS{}'.format(cl))
            txt = '{} {:.2f}'.format(cls_name, cf)
        return img
