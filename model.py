import os
import csv

samples = []
with open('./data/driving_log.csv') as csvfile:
    reader = csv.reader(csvfile)
    for line in reader:
        samples.append(line)

from sklearn.model_selection import train_test_split
train_samples, validation_samples = train_test_split(samples, test_size=0.2)

import cv2
import numpy as np
import sklearn
from sklearn.utils import shuffle


def generator(samples, batch_size=6):
    num_samples = len(samples)
    correction = 0.2
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                name = './data/IMG/' + batch_sample[0].split('/')[-1]
                center_image = cv2.imread(name)
                center_angle = float(batch_sample[3])
                name = './data/IMG/' + batch_sample[1].split('/')[-1]
                left_image = cv2.imread(name)
                left_angle = center_angle + 0.2
                name = './data/IMG/' + batch_sample[2].split('/')[-1]
                right_image = cv2.imread(name)
                right_angle = center_angle - 0.2
                images.append(center_image)
                images.append(left_image)
                images.append(right_image)
                images.append(np.fliplr(center_image))
                images.append(np.fliplr(left_image))
                images.append(np.fliplr(right_image))
                angles.append(center_angle)
                angles.append(left_angle)
                angles.append(right_angle)
                angles.append(-center_angle)
                angles.append(-left_angle)
                angles.append(-right_angle)

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=6)
validation_generator = generator(validation_samples, batch_size=6)

ch, row, col = 3, 80, 320  # Trimmed image format

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D

trim = [60, 20]

model = Sequential()
model.add(Cropping2D(cropping=((trim[0], 0), (trim[1], 0)), input_shape=(row+trim[0]+trim[1], col, ch)))
model.add(Lambda(lambda x: x/127.5 - 1.))
model.add(Convolution2D(6, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Convolution2D(16, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(120))
model.add(Dense(84))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(3*train_samples), validation_data=validation_generator, nb_val_samples=len(3*validation_samples), nb_epoch=3)

model.save('model.h5')

exit()