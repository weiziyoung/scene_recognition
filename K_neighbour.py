# -*- coding: utf-8 -*-
# @Time    : 07/11/2018 23:49
# @Author  : weiziyang
# @FileName: K_means.py
# @Software: PyCharm
import os
from collections import Counter

import numpy as np
import cv2

import base


class KNeighbour(base.Base):
    def __init__(self, k, pic_size=16):
        super().__init__()
        self.parameter_token = '{name}-K:{k}-Pic_size:{S}'.format(name=self.__class__.__name__, k=k, S=pic_size)
        self.k = k
        self.pic_size = pic_size

        self.image_matrix = None
        self.image_vector_matrix = None
        self.training_data_matrix_file = os.path.join(self.data_dir,
                                                      '{size}*{size}matrix.pkl'.format(size=self.pic_size))

    def train(self):
        self.image_matrix = self.compute_obj(self.training_data_matrix_file, self.convert_pic2matrix)
        self.image_vector_matrix = np.zeros((self.total_num, self.pic_size * self.pic_size))
        for i in range(self.total_num):
            image_2d = self.image_matrix[:, :, i]
            image_1d = image_2d.flatten()
            self.image_vector_matrix[i, :] = image_1d

    def convert_pic2matrix(self):
        image_matrix = np.zeros((self.pic_size, self.pic_size, self.total_num))
        index = 0
        for category in self.categories:
            for image_name in self.images_name:
                image_path = os.path.join(self.raw_training_dir, category, image_name)
                image_data = cv2.resize(cv2.imread(image_path)[:, :, 0], (self.pic_size, self.pic_size))
                image_matrix[:, :, index] = image_data
                index += 1
        return image_matrix

    def predict(self, image):
        image = image[:, :, 0]
        resized_image = cv2.resize(image, (self.pic_size, self.pic_size))
        test_image_vector = resized_image.flatten()
        # calculate the nearest image
        diff = self.image_vector_matrix - test_image_vector
        distance = np.sum(diff * diff, axis=1)
        most_similar_image = distance.argsort()[:self.k]
        most_similar_class = list(map(lambda a: a//100, most_similar_image))
        counter = Counter(most_similar_class)
        label = max(most_similar_class, key=lambda a: counter[a])
        return label


if __name__ == '__main__':
    run1 = KNeighbour(k=1, pic_size=8)
    run1.train()
    run1.test_model()
