# -*- coding: utf-8 -*-
# @Time    : 06/12/2018 15:56
# @Author  : weiziyang
# @FileName: sift.py
# @Software: PyCharm
import os

import cv2
from tqdm import tqdm
import numpy as np
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.svm import SVC

from base import Base, count_time


class SIFT(Base):
    def __init__(self, cluster_num, mini_patch=True, force_generate_again=False):
        super().__init__(force_generate_again=force_generate_again)
        self.parameter_token = "{name}-Cluster_num:{cluster}Kmeans_type:{M}".format(
            name=self.__class__.__name__, cluster=cluster_num, M='MINI' if mini_patch else 'NORMAL')
        self.cluster_num = cluster_num
        self.mini_patch = mini_patch

        self.working_dir = os.path.join(self.data_dir, self.parameter_token)
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)
        self.sift_features_list_file = os.path.join(self.data_dir, 'sift_features_list.pkl')
        self.k_means_file = os.path.join(self.working_dir, 'k_means.pkl')
        self.svm_file = os.path.join(self.working_dir, 'svm.pkl')

        self.sift_features_list = None
        self.sift_features = None
        self.k_means = None
        self.svm = None

    @count_time
    def train(self):
        self.sift_features_list = self.compute_obj(self.sift_features_list_file, self.generate_sift)
        self.sift_features = np.vstack(self.sift_features_list)
        self.k_means = self.compute_obj(self.k_means_file, self.compute_k_means)
        self.svm = self.compute_obj(self.svm_file, self.compute_svm)


    @count_time
    def generate_sift(self):
        sift = cv2.xfeatures2d.SIFT_create()
        sift_features_list = []
        with tqdm(total=self.total_num) as pbar:
            for category in self.categories:
                for image_name in self.images_name:
                    image_path = os.path.join(self.raw_training_dir, category, image_name)
                    image_data = cv2.imread(image_path)
                    kp, sift_feature = sift.detectAndCompute(image_data, None)
                    if sift_feature is None:
                        sift_feature = np.zeros((1, 128))
                    pbar.update(1)
                    sift_features_list.append(sift_feature)
        return sift_features_list

    @count_time
    def compute_k_means(self):
        if self.mini_patch:
            k_means = MiniBatchKMeans(n_clusters=self.cluster_num, init_size=3 * self.cluster_num).fit(self.sift_features)
        else:
            k_means = KMeans(n_clusters=self.cluster_num).fit(self.sift_features)
        return k_means

    @count_time
    def compute_svm(self):
        feature_matrix = np.zeros((self.total_num, self.cluster_num))
        index = 0
        with tqdm(total=self.total_num) as pbar:
            for sift_features in self.sift_features_list:
                feature_histogram = self.convert2visual_words(sift_features)
                feature_matrix[index, :] = feature_histogram
                index += 1
                pbar.update(1)
        svm = SVC(kernel='linear')
        svm.fit(feature_matrix, self.training_y)
        return svm

    def convert2visual_words(self, sift_features):
        feature_histogram = np.zeros(self.cluster_num)
        locations = self.k_means.predict(sift_features)
        for loc in locations:
            feature_histogram[loc] += 1
        return feature_histogram

    def predict(self, image):
        sift = cv2.xfeatures2d.SIFT_create()
        kp, sift_features = sift.detectAndCompute(image, None)
        feature_histogram = self.convert2visual_words(sift_features)
        prediction = self.svm.predict(np.atleast_2d(feature_histogram))
        return prediction


if __name__ == "__main__":
    sift = SIFT(cluster_num=600, mini_patch=True, force_generate_again=True)
    sift.train()
    sift.test_model()






