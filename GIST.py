# -*- coding: utf-8 -*-
# @Time    : 12/12/2018 22:27
# @Author  : weiziyang
# @FileName: gist.py
# @Software: PyCharm

import os

import cv2
import gist
from tqdm import tqdm
import numpy as np
from sklearn.svm import SVC

from base import Base, count_time


class GIST(Base):
    def __init__(self, pic_size, force_generate_again=False):
        super().__init__(force_generate_again=force_generate_again)
        self.parameter_token = "{name}-pic_size:{size}".format(
            name=self.__class__.__name__, size=pic_size)
        self.pic_size = pic_size

        self.working_dir = os.path.join(self.data_dir, self.parameter_token)
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        self.svm_file = os.path.join(self.working_dir, 'svm.pkl')

        self.gist_features = None
        self.svm = None

    @count_time
    def train(self):
        self.gist_features = self.generate_gist(self.pic_size)
        self.svm = self.compute_obj(self.svm_file, self.compute_svm)

    @count_time
    def generate_gist(self, size):
        gist_feature_matrix = np.zeros((self.total_num, 960))
        n = 0
        with tqdm(total=self.total_num) as pbar:
            for category in self.categories:
                for image_name in self.images_name:
                    image_path = os.path.join(self.raw_training_dir, category, image_name)
                    data = cv2.resize(cv2.imread(image_path), (size[0], size[1]))
                    gist_feature = gist.extract(data)
                    pbar.update(1)
                    gist_feature_matrix[n, :] = gist_feature
                    n += 1
        return gist_feature_matrix

    @count_time
    def compute_svm(self):
        svm = SVC(kernel='linear')
        svm.fit(self.gist_features, self.training_y)
        return svm

    def predict(self, image):
        data = cv2.resize(image, (self.pic_size[0], self.pic_size[1]))
        feature = gist.extract(data)
        prediction = self.svm.predict(np.atleast_2d(feature))
        return prediction


if __name__ == "__main__":
    g = GIST(pic_size=(512, 512))
    g.train()
    g.test_model()