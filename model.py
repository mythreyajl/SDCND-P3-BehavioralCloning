import numpy as np
import csv
import cv2
import os
import os.path as osp

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import sklearn

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D

measurements = []
images = []
with open('data/data_dl/driving_log_1.csv') as f:
    readCSV = csv.reader(f, delimiter=',')
    for row in readCSV:
        img_path = osp.join(os.getcwd(), 'data/data_dl/IMG/' + row[0].split('/')[-1] )
        img = cv2.imread(img_path)
        images.append(img)
        measurements.append(float(row[3]))

with open('data/flat/cw/driving_log.csv') as f:
    readCSV = csv.reader(f, delimiter=',')
    for row in readCSV:
        img_path = osp.join(os.getcwd(), 'data/flat/cw/IMG/' + row[0].split('/')[-1] )
        img = cv2.imread(img_path)
        images.append(img)
        measurements.append(float(row[3]))

with open('data/hilly/cw/driving_log.csv') as f:
    readCSV = csv.reader(f, delimiter=',')
    for row in readCSV:
        img_path = osp.join(os.getcwd(), 'data/hilly/cw/IMG/' + row[0].split('/')[-1])
        img = cv2.imread(img_path)
        images.append(img)
        measurements.append(float(row[3]))

images = np.array(images)
measurements = np.array(measurements)

X_train, X_valid, y_train, y_valid = train_test_split(images, measurements, test_size=0.2)


def generator(X, y, batch_size=32):
    num_samples = len(y)
    while 1:  # Loop forever so the generator never terminates
        X, y = shuffle(X, y)
        for offset in range(0, num_samples, batch_size):
            images = X[offset:offset+batch_size]
            angles = y[offset:offset+batch_size]

            # trim image to only see section with road
            # X_train = np.array(images)
            # y_train = np.array(angles)
            yield sklearn.utils.shuffle(images, angles)


# compile and train the model using the generator function
train_generator = generator(X_train, y_train, batch_size=32)
validation_generator = generator(X_valid, y_valid, batch_size=32)

model = Sequential()
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160, 320, 3)))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(y_train), validation_data=validation_generator, nb_val_samples=len(y_valid), nb_epoch=3)

model.save('model.h5')
exit()

