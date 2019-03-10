# -*- coding: utf-8 -*-
# @Time    : 06/12/2018 20:05
# @Author  : weiziyang
# @FileName: base.py
# @Software: PyCharm

import os
import getpass
import logging
import pickle
import datetime
from tqdm import tqdm

import yaml
import cv2
import numpy as np


def count_time(func):
    """
    :param func: Any methods or func
    :return: return how many time the function consumes
    """
    def int_time(self, *args, **kwargs):
        start_time = datetime.datetime.now()  # start time
        result = func(self, *args, **kwargs)
        over_time = datetime.datetime.now()  # end time
        total_time = (over_time - start_time).total_seconds()
        self.logger.info('The function %s costs %s seconds' % (func.__name__, total_time))
        return result
    return int_time


def normalize(matrix):
    """
    :param matrix: any shape matrix
    :return: a normalized matrix that has a the same shape with the input matrix
    """
    mean = np.mean(matrix)
    mean_matrix = matrix - mean
    matrix_range = np.max(matrix) - np.min(matrix)
    if matrix_range == 0:
        matrix = np.zeros(matrix.shape)
    else:
        matrix = mean_matrix / matrix_range
    return matrix


class Base(object):
    def __init__(self, force_generate_again=False):
        if getpass.getuser() in ['root', 'weiziyang666']:
            config_file = 'production_config.yaml'
        elif getpass.getuser() == 'weiziyang':
            config_file = 'config.yaml'
        else:
            config_file = None
            raise Exception('You should config your file')
        with open(config_file, 'r') as f:
            config = yaml.load(f)
            self.root_dir = config['root_dir']
            self.data_dir = os.path.join(self.root_dir, config['data_root_dir'])
            self.raw_training_dir = os.path.join(self.root_dir, config['raw_training_dir'])
            self.training_dir = os.path.join(self.root_dir, config['training_dir'])
            self.test_dir = os.path.join(self.root_dir, config['test_dir'])

        # initializing logging
        self.parameter_token = ''
        self.logger = logging.getLogger(datetime.datetime.now().strftime('%y-%m-%d_%H:%M:%S'))
        self.logger.setLevel(logging.INFO)
        handler = logging.FileHandler(config['logging_file'])
        handler.setLevel(logging.INFO)
        formatter = logging.Formatter(config['logging_format'])
        handler.setFormatter(formatter)
        self.logger.addHandler(handler)

        if config['logging_print']:
            console = logging.StreamHandler()
            console.setLevel(logging.INFO)
            console.setFormatter(formatter)
            self.logger.addHandler(console)

        # if it set to False, it will not read the existing model, it will generate a new one instead.
        self.force_generate = force_generate_again
        self.categories = ['Forest', 'bedroom', 'Office', 'Highway', 'Coast', 'Insidecity', 'TallBuilding',
                           'industrial', 'Street', 'livingroom', 'Suburb', 'Mountain', 'kitchen', 'OpenCountry', 'store']
        self.images_name = [str(i)+'.jpg' for i in range(100)]
        self.total_num = 1500
        self.training_y = np.zeros(self.total_num)
        for n in range(self.total_num):
            category = n // 100
            self.training_y[n] = category

    def compute_obj(self, file, method):
        """
        Like a decorator, we can use this method to save the model so as to speed up the procedure
        :param file: The file that will save the object(svm object or k_means object...
        :param method:The method that can be used to return a model
        :return:object
        """
        if os.path.exists(file) and not self.force_generate:
            with open(file, 'rb') as f:
                obj = pickle.load(f)
        else:
            obj = method()
            with open(file, 'wb') as f:
                pickle.dump(obj, f)
        return obj

    def test_model(self):
        """
        To test the accuracy of the generated model
        :return:
        """
        self.logger.info('Start test model')
        total_correct = 0
        total_sample = 0
        for n, category in enumerate(self.categories):
            category_correct = 0
            true_category = n
            test_category_path = os.path.join(self.test_dir, category)
            image_names = [each for each in os.listdir(test_category_path) if not each.startswith('.')]
            category_sample = len(image_names)
            for path in image_names:
                image_path = os.path.join(test_category_path, path)
                image = cv2.imread(image_path)
                prediction = self.predict(image)
                predict_category = self.categories[int(prediction)]
                self.logger.info('image:{image_path}, predict:{predict} fact:{true_category}'.format(
                    image_path=image_path, predict=predict_category, true_category=category))
                if prediction == true_category:
                    category_correct += 1
            correct_ratio = category_correct / category_sample
            total_correct += category_correct
            total_sample += category_sample
            self.logger.log(logging.INFO, 'category:{category} correct_ratio:{ratio}%'.format(
                category=category, ratio=correct_ratio * 100))
        self.logger.info('Parameter:{parameter}:sample num:{total} - Correct: {correct}%'.format(
            parameter=self.parameter_token, total=total_sample, correct=total_correct / total_sample * 100))
        return total_correct / total_sample * 100

    def predict_all(self):
        """
        Mark all of the test image...
        :return:
        """
        testing_catalogue = os.path.join(self.root_dir, 'testing')
        files = sorted([each for each in os.listdir(testing_catalogue) if not each.startswith('.')], key=lambda a: int(a.split('.')[0]))
        prediction_text = ''''''
        with tqdm(total=len(files)) as pbar:
            for file in files:
                image_path = os.path.join(testing_catalogue, file)
                image = cv2.imread(image_path)
                category_no = self.predict(image)
                category_name = self.categories[int(category_no)]
                temp_string = file + ' ' + category_name.lower() + '\n'
                prediction_text += temp_string
                pbar.update(1)
        return prediction_text

    def predict(self, image):
        raise Exception('Predict method has not been implemented yet')

    def train(self):
        raise Exception('Train method has not been implemented yet')
