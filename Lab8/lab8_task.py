# -*- coding: utf-8 -*-
"""
Submission Template for Lab 8
Important notice: DO NOT use any global variables in this submission file
"""

# Task 1
def data_preprocessing(data_dir, cate2Idx, img_size):
  x = []
  y = []
  # Note: OpenCV reads an image with its channel being in 'BGR' order,
  # you need to change the order back to 'RGB'.
  ###############################################################################
  # TODO: your code starts here
  
  for cate in category_list:
    for img in os.listdir('{}/{}'.format(data_dir, cate)):
      # opencv preprocess
      img = cv2.imread('{}/{}/{}'.format(data_dir, cate, img))
      img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # change to RGB
      img = cv2.resize(img,(img_size, img_size),interpolation=cv2.INTER_LINEAR)

      x.append(img)
      y.append(cate2Idx[cate])

  # TODO: your code ends here
  ###############################################################################
  x = np.asarray(x)
  y = np.asarray(y)
  return x, y


# Task 2
def get_datagen():
  datagen = None
  ###############################################################################
  # TODO: your code starts here
  datagen = ImageDataGenerator(
      rotation_range=14,
      width_shift_range=0.21,
      height_shift_range=0.21,

      horizontal_flip=True,
      vertical_flip=True,

      zoom_range=0.21,
      shear_range=0.21,

      fill_mode='nearest'
  )
  # TODO: your code ends here
  ###############################################################################
  return datagen


# Task 3
def custom_model():
  model = None
  ###############################################################################
  # TODO: your code starts here
  model = Sequential()
  model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
  model.add(Dropout(0.03))
  model.add(MaxPooling2D((2, 2)))

  model.add(Conv2D(64, (3, 3), activation='relu'))
  model.add(Dropout(0.03))
  model.add(MaxPooling2D((2, 2)))

  model.add(Conv2D(128, (3, 3), activation='relu'))
  model.add(Dropout(0.03))
  model.add(AveragePooling2D((2, 2)))

  model.add(Conv2D(256, (3, 3), activation='relu'))
  model.add(Dropout(0.03))
  model.add(AveragePooling2D((2, 2)))

  model.add(Flatten())
  model.add(Dense(512, activation='relu'))
  model.add(Dropout(0.5))
  model.add(Dense(len(category_list), activation='softmax'))
  # TODO: your code ends here
  ###############################################################################
  return model


if __name__ == '__main__':
  # Import necessary libraries
  import os, cv2
  import numpy as np
  from sklearn.model_selection import train_test_split
  import keras
  from keras.utils import np_utils
  from keras.models import Sequential
  from keras.layers import Conv2D, MaxPooling2D
  from keras.layers import Dense, Dropout, Flatten
  from keras.preprocessing.image import ImageDataGenerator
