import os
import cv2
#import dlib
import numpy as np
import shutil
import random
import tensorflow as tf
import keras
from time import time
from datetime import datetime
from collections import Counter
import scipy.io as io
from scipy.io import loadmat
from keras.preprocessing.image import img_to_array

ROOT = '/opt/sdb/workspace'
#fer2013
FACE_DATA_DIR_FER2013 = ROOT + "/data/fer2013/fer2013.csv"
OUT_PUT_DIR_FER2013 = ROOT + '/data/fer2013_preprocess'

#imdb
FACE_DATA_DIR_IMDB = ROOT + '/data/IMDB-WIKI/imdb_crop'
IMDB_MAT_PATH = ROOT + '/data/IMDB-WIKI/imdb_crop/imdb.mat'

#wiki
FACE_DATA_DIR_WIKI = ROOT + '/data/IMDB-WIKI/wiki_crop'
WIKI_MAT_PATH = ROOT + '/data/IMDB-WIKI/wiki_crop/wiki.mat'

flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean('random_flip_up_down', False, 'If uses flip')
flags.DEFINE_boolean('random_flip_left_right', True, 'If uses flip')
flags.DEFINE_boolean('random_brightness', True, 'If uses brightness')
flags.DEFINE_boolean('random_contrast', False, 'If uses contrast')
flags.DEFINE_boolean('random_saturation', False, 'If uses saturation')
flags.DEFINE_integer('image_size', 224, 'image size.')
flags.DEFINE_boolean('resize', False, 'If uses image resize')
"""
#flags examples
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_string('train_dir', 'data', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data for unit testing.')
"""
def pre_process(images):

    if FLAGS.random_flip_up_down:
        images = tf.image.random_flip_up_down(images)
    if FLAGS.random_flip_left_right:
        images = tf.image.random_flip_left_right(images)
    if FLAGS.random_brightness:
        images = tf.image.random_brightness(images, max_delta=0.15)
    if FLAGS.random_contrast:
        images = tf.image.random_contrast(images, 0.8, 1.2)
    if FLAGS.random_saturation:
        images = tf.image.random_saturation(images, 0.3, 0.5)
    # if FLAGS.resize:
    #     new_size = tf.constant([FLAGS.image_size, FLAGS.image_size], dtype=tf.int32)
    #     images = tf.image.resize_images(images, new_size)
    return images

class DataManager(object):
    """
    load the dataset fer2013 and imdb
    """
    def __init__(self, dataset_name='fer2013', dataset_path=None, num_classes = 8,image_size=(224, 224),b_gray_chanel = True):
        """

        :param dataset_name: select the dataset "fer2013" or "imdb"
        :param dataset_path: the dataset location dir
        :param num_classes: the classes number of dataset
        :param image_size: the image size output you want
        :param b_gray_chanel: if or not convert image to gray

        :return the tuple have image datas and image labels
        """
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.image_size = image_size
        self.b_gray_chanel = b_gray_chanel
        self.num_classes = num_classes
        if self.dataset_path != None:
            self.dataset_path = dataset_path
        elif self.dataset_name == 'fer2013':
            self.dataset_path = FACE_DATA_DIR_FER2013
        elif self.dataset_name == 'imdb':
            self.dataset_path = FACE_DATA_DIR_IMDB
            self.mat_path = IMDB_MAT_PATH
        elif self.dataset_name == 'wiki':
            self.dataset_path = FACE_DATA_DIR_WIKI
            self.mat_path = WIKI_MAT_PATH
        else:
            raise Exception('Incorrect dataset name, please input CK+ or fer2013')

    def get_data(self):
        if self.dataset_name == 'fer2013':
            data = self._load_fer2013()
        elif self.dataset_name == 'imdb':
            data = self._load_imdb()
        elif self.dataset_name == 'wiki':
            data = self._load_wiki()
        return data

    def _load_fer2013(self):
        """ load the dataset of fer2013 for the file fer2013.csv
        :return: a list contains the training ,private test and public test set
        :type: list
        """
        # fer2013 dataset:
        # Training       28709
        # PrivateTest     3589
        # PublicTest      3589

        # emotion labels from FER2013:
        # emotion = {'Angry': 0, 'Disgust': 1, 'Fear': 2, 'Happy': 3,
        #          'Sad': 4, 'Surprise': 5, 'Neutral': 6}
        num_classes = 7
        with open(self.dataset_path) as f:
            content = f.readlines()

        lines = np.array(content)
        num_of_instances = lines.size
        print("number of instances: ", num_of_instances)
        print("instance length: ", len(lines[1].split(",")[1].split(" ")))

        # ------------------------------
        # initialize train set, val set and test set
        x_train, y_train, x_test, y_test = [], [], [], []
        x_val, y_val = [], []
        # ------------------------------
        # transfer train, val and test set data
        for i in range(1, num_of_instances):
            emotion, img, usage = lines[i].split(",")
            val = img.split(" ")
            pixels = np.array(val, 'float32')
            emotion = keras.utils.to_categorical(emotion, num_classes)
            face = pixels.reshape((48, 48))
            face = cv2.resize(face.astype('uint8'),self.image_size)
            face.astype('float32')
            if 'Training' in usage:
                y_train.append(emotion)
                x_train.append(face)
            elif 'PublicTest' in usage:
                y_val.append(emotion)
                x_val.append(face)
            elif 'PrivateTest' in usage:
                y_test.append(emotion)
                x_test.append(face)
        # ------------------------------
        # data transformation for train ,val, and test sets
        x_train = np.expand_dims(np.asarray(x_train),-1)
        y_train = np.array(y_train, 'float32')
        x_val = np.expand_dims(np.asarray(x_val), -1)
        y_val = np.array(y_val, 'float32')
        return x_train, y_train, x_val, y_val
        
    def calculate_age(self, taken, dob):
        birth = datetime.fromordinal(max(int(dob) - 366, 1))
        # assume the photo was taken in the middle of the year
        if birth.month < 7:
            return taken - birth.year
        else:
            return taken - birth.year - 1
            
    def _load_imdb(self):
        # gender and age labels from imdb:
        # gender = {'female': 0, 'male': 1}
        # age = [0, 1, 2, 3 ..., 100]
        num_gender_classes = 2
        num_age_classes = 101
        
        dataset = loadmat(self.mat_path)
        full_path = dataset['imdb']['full_path'][0, 0][0]
        gender = dataset['imdb']['gender'][0, 0][0]

        dob = dataset['imdb'][0, 0]["dob"][0]  # Matlab serial date number
        photo_taken = dataset['imdb'][0, 0]["photo_taken"][0]  # year
        age = [self.calculate_age(photo_taken[i], dob[i]) for i in range(len(dob))]
        
        face_score = dataset['imdb']['face_score'][0, 0][0]
        second_face_score = dataset['imdb']['second_face_score'][0,0][0]
        ''' 
        print('full_path:', full_path[:5])
        print('gender:', gender[:5])
        print('face_score:', face_score[:5])
        print('second_face_score:', second_face_score[:5])
        print('age:', age[:5])
        '''
        img_datas = []
        gender_datas = []
        age_datas = []
        min_score = 1.0  #default 
        for i in range(len(full_path)):
            if face_score[i] < min_score:
                continue
            if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
                continue
            if ~(0 <= age[i] <= 100):
                continue

            if np.isnan(gender[i]):
                continue
           
            abs_path = os.path.join(self.dataset_path, str(full_path[i][0]))
            if not os.path.exists(abs_path):
                continue
            img = cv2.imread(abs_path)
            if img is None:
                continue
            img = cv2.resize(img, self.image_size)
            img = img_to_array(img)
            img_datas.append(img)
            
            gender_datas.append(keras.utils.to_categorical(int(gender[i]), num_gender_classes))
            age_datas.append(keras.utils.to_categorical(age[i], num_age_classes))
        img_datas = np.array(img_datas, dtype="float") / 255.0
        gender_datas = np.array(gender_datas)
        age_datas = np.array(age_datas)
        print('number of samples:',len(img_datas))
        print('img_datas shape:', img_datas.shape)
        print('gender_datas shape:', gender_datas.shape)
        print('age_datas shape:', age_datas.shape)
        return img_datas, gender_datas, age_datas
        
    def _load_wiki(self):
        # gender and age labels from wiki:
        # gender = {'female': 0, 'male': 1}
        # age = [0, 1, 2, 3 ..., 100]
        num_gender_classes = 2
        num_age_classes = 101
        
        dataset = loadmat(self.mat_path)
        full_path = dataset['wiki']['full_path'][0, 0][0]
        gender = dataset['wiki']['gender'][0, 0][0]

        dob = dataset['wiki'][0, 0]["dob"][0]  # Matlab serial date number
        photo_taken = dataset['wiki'][0, 0]["photo_taken"][0]  # year
        age = [self.calculate_age(photo_taken[i], dob[i]) for i in range(len(dob))]
        
        face_score = dataset['wiki']['face_score'][0, 0][0]
        second_face_score = dataset['wiki']['second_face_score'][0,0][0]
        ''' 
        print('full_path:', full_path[:5])
        print('gender:', gender[:5])
        print('face_score:', face_score[:5])
        print('second_face_score:', second_face_score[:5])
        print('age:', age[:5])
        '''
        img_datas = []
        gender_datas = []
        age_datas = []
        min_score = 1.0  #default 
        for i in range(len(full_path)):
            if face_score[i] < min_score:
                continue
            if (~np.isnan(second_face_score[i])) and second_face_score[i] > 0.0:
                continue
            if ~(0 <= age[i] <= 100):
                continue

            if np.isnan(gender[i]):
                continue
           
            abs_path = os.path.join(self.dataset_path, str(full_path[i][0]))
            if not os.path.exists(abs_path):
                continue
            img = cv2.imread(abs_path)
            if img is None:
                continue
            img = cv2.resize(img, self.image_size)
            img = img_to_array(img)
            img_datas.append(img)
            
            gender_datas.append(keras.utils.to_categorical(int(gender[i]), num_gender_classes))
            age_datas.append(keras.utils.to_categorical(age[i], num_age_classes))
        img_datas = np.array(img_datas, dtype="float") / 255.0
        gender_datas = np.array(gender_datas)
        age_datas = np.array(age_datas)
        print('number of samples:',len(img_datas))
        print('img_datas shape:', img_datas.shape)
        print('gender_datas shape:', gender_datas.shape)
        print('age_datas shape:', age_datas.shape)
        return img_datas, gender_datas, age_datas

def preprocess_input(x,v2 = True):
    """normalize the data to [0,1] and select transform it to [-0.5,0.5] or not"""
    x = x.astype('float32')
    x = x / 255.0
    if v2:
        x = x - 0.5
        x = x * 2.0
    return x

if __name__ == "__main__":
    #data = DataManager(dataset_name='fer2013',image_size=(64,64)).get_data()
    
    img_datas, gender_datas, age_datas = DataManager(dataset_name='wiki',image_size=(64,64)).get_data()
    '''
    for index in range(len(img_datas)):
        if index >5:
            break;
        print(gender_datas[index], age_datas[index])
    '''
