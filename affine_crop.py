import numpy as np
# import torch

from PIL import Image
# from torch.utils.data import Dataset
import os
from collections import defaultdict
from skimage import transform as trans
import cv2
import xmltodict
import json

def crop_transform_with_box(img, box, image_size, **kwargs):
    det = box
    margin = kwargs.get('margin', 44)
    bb = np.zeros(4, dtype=np.int32)
    bb[0] = np.maximum(det[0] - det[2] / 2 - margin/2, 0)
    bb[1] = np.maximum(det[1] - det[3] / 2 - margin/2, 0)
    bb[2] = np.minimum(det[0] + det[2] / 2 + margin/2, img.shape[1])
    bb[3] = np.minimum(det[1] + det[3] / 2 + margin/2, img.shape[0])
    ret = img[bb[1]:bb[3],bb[0]:bb[2],:]
    if len(image_size)>0:
        ret = cv2.resize(ret, (image_size[1], image_size[0]))
    return ret

def crop_transform_with_landmark(rimg, landmark, image_size, **kwargs):
    assert landmark.shape[0] == 68 or landmark.shape[0] == 5
    assert landmark.shape[1] == 2
    if landmark.shape[0] == 68:
        landmark5 = np.zeros((5, 2), dtype=np.float32)
        landmark5[0] = (landmark[36]+landmark[39])/2
        landmark5[1] = (landmark[42]+landmark[45])/2
        landmark5[2] = landmark[30]
        landmark5[3] = landmark[48]
        landmark5[4] = landmark[54]
    else:
        landmark5 = landmark
    tform = trans.SimilarityTransform()
    src = np.array([
            [87, 99.0],
            [161, 99.0],
            [124.0, 138.0],
            [97.0, 182.0],
            [151.0, 182.0]], dtype=np.float32)
    tform.estimate(landmark5, src)
    M = tform.params[0:2, :]
    img = cv2.warpAffine(rimg, M, (image_size[1], image_size[0]), borderValue = 0.0)
    return img

class Dataset():
    def __init__(self, record_list, img_size=256):
        super(Dataset, self).__init__()
        self.img_size = img_size
        assert self.img_size == 256
        self.meta = []
        for line in open(record_list):

            line = line.rstrip().split()
            assert len(line) == 11 or len(line) == 5
            self.meta.append((line[0], [float(k) for k in line[1:]]))

        self.total_num = len(self.meta)


    def affine_image(self):
        for index in range(self.total_num):
            name = self.meta[index][0]
            meta = self.meta[index][1]
            assert name.endswith('.jpg')
            save_path = name[:-4] + '_cropped.jpg'
            sample = cv2.imread(name)
            if len(meta) == 4:
                sample = crop_transform_with_box(sample, meta, [self.img_size, self.img_size], margin=10)
            else:
                sample = crop_transform_with_landmark(sample, meta, [self.img_size, self.img_size])
            cv2.imwrite(save_path,sample)
            if index % 1000 == 0:
                print('success process ',index)
   
def crop_pic_main():
    dataset = Dataset('record.txt')
    dataset.affine_image()

if __name__ == '__main__':
    crop_pic_main()
