"""
File: train_emotion_classifier.py
Github: https://github.com/xiuweihe
Description: Train emotion classification model
"""

from keras.callbacks import CSVLogger, ModelCheckpoint, EarlyStopping
from keras.callbacks import ReduceLROnPlateau
from keras.preprocessing.image import ImageDataGenerator

from src.cnn import mini_XCEPTION
from utils.builddata import DataManager
from utils.builddata import preprocess_input

import os

cur_path = os.path.dirname(__file__)
parent_path = os.path.dirname(cur_path)
# parameters
batch_size = 32
num_epochs = 500
input_shape = (64, 64, 1)
validation_split = .2
verbose = 1
num_classes = 7
patience = 20
#base_path = parent_path + '/trained_models/emotion_models/'
base_path = './trained_models/emotion_models/'
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
model = mini_XCEPTION(input_shape, num_classes)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()


datasets = ['fer2013']
for dataset_name in datasets:
    print('Training dataset:', dataset_name)

    # callbacks
    log_file_path = base_path + dataset_name + '_emotion_training.log'
    csv_logger = CSVLogger(log_file_path, append=False)
    early_stop = EarlyStopping(monitor='val_loss', patience=patience)
    reduce_lr = ReduceLROnPlateau('val_loss', factor=0.1,
                                  patience=int(patience/4), verbose=1)
    model_names = dataset_name + '_mini_XCEPTION_{epoch:02d}-{val_accuracy:.2f}.h5'
    model_path = os.path.join(base_path, model_names)
    model_checkpoint = ModelCheckpoint(model_path, monitor='val_loss', verbose=1, save_best_only=True)
    callbacks = [model_checkpoint, csv_logger, early_stop, reduce_lr]

    # loading dataset
    data_loader = DataManager(dataset_name, image_size=input_shape[:2])
    # faces, emotions = data_loader.get_data()
    x_train, y_train, x_val, y_val = data_loader.get_data()
    x_train = preprocess_input(x_train)
    x_val = preprocess_input(x_val)
    # x_test = preprocess_input(x_test)
    # num_samples, num_classes = emotions.shape
    # train_data, val_data = split_data(faces, emotions, validation_split)
    # train_faces, train_emotions = train_data
    model.fit_generator(data_generator.flow(x_train, y_train, batch_size),
                        steps_per_epoch=len(x_train) / batch_size,
                        epochs=num_epochs, verbose=1, callbacks=callbacks,
                        validation_data=(x_val,y_val))
