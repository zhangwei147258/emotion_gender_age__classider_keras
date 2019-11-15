"""
File: train_emotion_classifier.py
Github: https://github.com/xiuweihe
Description: Train emotion classification model
"""

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

from src.cnn import gender_age_XCEPTION
from utils.builddata import DataManager
from utils.builddata import preprocess_input

import os
import numpy as np
from sklearn.model_selection import train_test_split

from keras import optimizers

# parameters
batch_size = 32
num_epochs = 500
input_shape = (64, 64, 3)
validation_split = .2
verbose = 1
num_gender_classes=2
num_age_classes=101
patience = 50
base_path = './trained_models/gender_age_models/'
try:
    os.makedirs(base_path)
except OSError:
    pass

# data generator
data_generator = ImageDataGenerator(
                        featurewise_center=False,
                        featurewise_std_normalization=False,
                        rotation_range=10,
                        width_shift_range=0.1,
                        height_shift_range=0.1,
                        zoom_range=.1,
                        horizontal_flip=True)

# model parameters/compilation
model = gender_age_XCEPTION(input_shape, num_gender_classes, num_age_classes)
losses = { 'gender_output':'categorical_crossentropy', 'age_output':'categorical_crossentropy' }
loss_weights = {'gender_output':1.0, 'age_output':1.0}

sgd = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss=losses,
              loss_weights=loss_weights, metrics=['accuracy'])
model.summary()


datasets = ['imdb']
for dataset_name in datasets:
    print('Training dataset:', dataset_name)

    # callbacks
    log_file_path = base_path + dataset_name + '_gender_age_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping(monitor='val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=int(patience/4), verbose=1, min_lr=0.0005)
    model_names = dataset_name + '_gender_age_XCEPTION_{epoch:02d}-{val_gender_output_accuracy:.2f}-{val_age_output_accuracy:.2f}.h5'
    model_path = os.path.join(base_path, model_names)
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # loading dataset
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    img_datas, gender_datas, age_datas = data_loader.get_data()
    (x_train, x_val, y_train_gender, y_val_gender, y_train_age, y_val_age) = train_test_split(img_datas, gender_datas, age_datas, test_size=0.2, random_state=42)
    print('x_train shape:{}, x_val shape:{}'.format(x_train.shape, x_val.shape))
    print('y_train_gender shape:{}, y_val_gender shape:{}'.format(y_train_gender.shape, y_val_gender.shape))
    print('y_train_age shape:{}, y_val_age shape:{}'.format(y_train_age.shape, y_val_age.shape))

    #model.fit_generator(data_generator.flow(x_train, {'gender_output': y_train_gender, 'age_output':y_train_age}, batch_size),
    #                    steps_per_epoch=len(x_train) / batch_size,
    #                    epochs=num_epochs, verbose=1, callbacks=callbacks,
    #                    validation_data=(x_val,{'gender_output': y_val_gender, 'age_output':y_val_age}))
    model.fit(x_train, {'gender_output': y_train_gender, 'age_output':y_train_age},
              batch_size = batch_size,
              epochs=num_epochs, verbose=1, 
              callbacks=callbacks,
              validation_data=(x_val,{'gender_output': y_val_gender, 'age_output':y_val_age}))
