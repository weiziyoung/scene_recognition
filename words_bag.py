# -*- coding: utf-8 -*-
# @Time    : 05/12/2018 18:11
# @Author  : weiziyang
# @FileName: run2.py
# @Software: PyCharm
import os

import cv2

from tqdm import tqdm
import numpy as np
from sklearn.svm import SVC
from sklearn.cluster import KMeans, MiniBatchKMeans

import base


class WordsBag(base.Base):
    def __init__(self, patch_gap, patch_size, cluster_num, pic_size=None, output_patch_image=False, mini_patch=True,
                 force_generate_again=False):
        super().__init__(force_generate_again=force_generate_again)
        self.parameter_token = "{name}-Patch_size:{patch_size}-Patch_gap:{gap}-Pic_size:{size}-" \
                               "Cluster_num:{n_cluster}-Kmeans_type:{m}".format(
                                name=self.__class__.__name__, patch_size=patch_size, gap=patch_gap,
                                size=pic_size, n_cluster=cluster_num, m='MINI' if mini_patch else 'NORMAL')
        self.training_dir = self.training_dir.format(pic_size=pic_size)

        # initializing var
        self.logger.info('Start initializing, parameter token is:{token}'.format(token=self.parameter_token))
        self.pic_size = pic_size
        self.patch_gap = patch_gap
        self.patch_size = patch_size
        self.cluster_num = cluster_num
        self.output_patch_image = output_patch_image
        self.mini_patch = mini_patch

        # initializing file path
        self.working_dir = os.path.join(self.data_dir, self.parameter_token)
        if not os.path.exists(self.working_dir):
            os.makedirs(self.working_dir)

        # Some data have been generated so that we don't need to compute them again
        self.training_data_matrix_file = os.path.join(self.data_dir, '{size}*{size}matrix.pkl'.format(size=self.pic_size))
        self.svm_file = os.path.join(self.working_dir, 'svm.pkl')
        self.k_means_file = os.path.join(self.working_dir, 'k_means.pkl')

        self.training_matrix = None
        self.k_means = None
        self.svm = None

    @base.count_time
    def train(self):
        self.training_matrix = self.compute_obj(self.training_data_matrix_file, self.convert_pic2matrix)
        self.k_means = self.compute_obj(self.k_means_file, self.compute_k_means)
        self.svm = self.compute_obj(self.svm_file, self.compute_svm)

    @base.count_time
    def compute_k_means(self):
        """
        For every images, we divide them into many patches, and cluster into 500 category or more...
        :return:k means object
        """
        # divide many patches
        patch_matrix_list = []
        for image_index in range(self.total_num):
            if self.pic_size:
                image = self.training_matrix[:, :, image_index]
            else:
                image = self.training_matrix[image_index]
            y_size, x_size = image.shape[0], image.shape[1]
            y = 0
            while y + self.patch_size <= y_size:
                x = 0
                while x + self.patch_size <= x_size:
                    patch = image[y:y + self.patch_size, x:x + self.patch_size]
                    patch_vector = base.normalize(patch.flatten())
                    patch_matrix_list.append(patch_vector)
                    x += self.patch_gap
                y += self.patch_gap
        patch_matrix = np.vstack(patch_matrix_list)
        if self.mini_patch:
            k_means = MiniBatchKMeans(n_clusters=self.cluster_num, init_size=3*self.cluster_num).fit(patch_matrix)
        else:
            k_means = KMeans(n_clusters=self.cluster_num).fit(patch_matrix)
        return k_means

    @base.count_time
    def compute_svm(self):
        # compute visual word vector
        image_batch_matrics = np.zeros((self.total_num, self.cluster_num))
        with tqdm(total=self.total_num) as pbar:
            for image_index in range(self.total_num):
                image = self.training_matrix[:, :, image_index] if self.pic_size else self.training_matrix[image_index]
                visual_word_vector = self.convert2visual_word(image)
                image_batch_matrics[image_index, :] = visual_word_vector
                pbar.update(1)
        svm = SVC(kernel='linear')
        svm.fit(image_batch_matrics, self.training_y)
        return svm

    def convert2visual_word(self, image):
        """
        For every image,this method can convert it into a vector that is represented by the number of patches.
        :param image:
        :return:
        """
        image_batch_vector = np.zeros(self.cluster_num)
        patch_matrix_list = []
        y_size, x_size = image.shape[0], image.shape[1]
        y = 0
        while y + self.patch_size <= y_size:
            x = 0
            while x + self.patch_size <= x_size:
                patch = image[y:y + self.patch_size, x:x + self.patch_size]
                patch_vector = base.normalize(patch.flatten())
                patch_matrix_list.append(patch_vector)
                x += self.patch_gap
            y += self.patch_gap
        patch_matrix = np.vstack(patch_matrix_list)
        locations = self.k_means.predict(patch_matrix)
        if self.output_patch_image:
            self.logger.info('Output image...')
            index = 0
            for patch_vector, location in zip(patch_matrix, locations):
                patch_img = patch_vector.reshape((self.patch_size, self.patch_size))
                patch_path = os.path.join(self.working_dir, 'patches', str(location))
                if not os.path.exists(patch_path):
                    os.makedirs(patch_path)
                cv2.imwrite(os.path.join(patch_path, '{}.jpg'.format(index)), patch_img)
                index += 1
        for loc in locations:
            image_batch_vector[loc] += 1
        return np.atleast_2d(image_batch_vector)

    def convert_pic2matrix(self):
        """
        resize and convert the origin image to matrix and save them into a file so that we don't need to read the file
        and do the same stuff again and again
        :return:
        """
        if self.pic_size:
            image_matrix = np.zeros((self.pic_size, self.pic_size, self.total_num))
        else:
            image_matrix = []
        index = 0
        for category in self.categories:
            for image_name in self.images_name:
                image_path = os.path.join(self.raw_training_dir, category, image_name)
                if self.pic_size:
                    image_data = cv2.resize(cv2.imread(image_path)[:, :, 0], (self.pic_size, self.pic_size))
                    image_matrix[:, :, index] = image_data
                else:
                    image_data = cv2.imread(image_path)[:, :, 0]
                    image_matrix.append(image_data)
                index += 1
        return image_matrix

    def predict(self, image):
        image = image[:, :, 0]
        if self.pic_size:
            image = cv2.resize(image, (self.pic_size, self.pic_size))
        batch_image = self.convert2visual_word(image)
        prediction = self.svm.predict(batch_image)
        return prediction


if __name__ == "__main__":
    pic_size = None
    patch_gap = 4
    patch_size = 8
    cluster_num = 700
    output_patch_image = True
    mini_patch = True
    run2 = WordsBag(patch_gap=patch_gap, patch_size=patch_size, cluster_num=cluster_num,
                    output_patch_image=output_patch_image, mini_patch=mini_patch, pic_size=pic_size,
                    force_generate_again=True)
    run2.train()
    run2.test_model()



