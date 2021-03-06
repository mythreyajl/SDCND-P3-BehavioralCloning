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
    correction = 0.3
    while 1:  # Loop forever so the generator never terminates
        shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            angles = []
            for batch_sample in batch_samples:
                c_image = cv2.imread('./data/IMG/' + batch_sample[0].split('/')[-1])
                l_image = cv2.imread('./data/IMG/' + batch_sample[1].split('/')[-1])
                r_image = cv2.imread('./data/IMG/' + batch_sample[2].split('/')[-1])
                c_angle = float(batch_sample[3])
                l_angle = c_angle + correction
                r_angle = c_angle - correction
                images.extend([c_image, l_image, r_image, np.fliplr(c_image), np.fliplr(l_image), np.fliplr(r_image)])
                angles.extend([c_angle, l_angle, r_angle, -c_angle, -l_angle, -r_angle])

            # trim image to only see section with road
            X_train = np.array(images)
            y_train = np.array(angles)
            yield sklearn.utils.shuffle(X_train, y_train)


# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size=6)
validation_generator = generator(validation_samples, batch_size=6)

ch, row, col = 3, 60, 320  # Trimmed image format

from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Dropout
from keras.layers import Convolution2D, MaxPooling2D, Cropping2D
from keras.utils.visualize_util import plot

trim = [80, 20]

model = Sequential()
model.add(Cropping2D(cropping=((trim[0], 0), (trim[1], 0)), input_shape=(row+trim[0]+trim[1], col, ch)))
model.add(Lambda(lambda x: x/127.5 - 1.))
model.add(Convolution2D(20, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.5))
model.add(Convolution2D(40, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.35))
model.add(Convolution2D(80, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.25))
model.add(Convolution2D(160, 5, 5, activation='relu'))
model.add(MaxPooling2D())
model.add(Dropout(0.15))
model.add(Flatten())
model.add(Dense(400))
model.add(Dense(200))
model.add(Dense(120))
model.add(Dense(1))

plot(model, to_file='model_plot.png')

model.compile(loss='mse', optimizer='adam')
model.fit_generator(train_generator, samples_per_epoch=len(6*train_samples), validation_data=validation_generator, nb_val_samples=len(6*validation_samples), nb_epoch=10)

model.save('model.h5')

exit()
