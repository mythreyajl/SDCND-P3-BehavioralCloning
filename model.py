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

def parse_data(path):
    measurements = []
    images = []
    correction = 0.2
    for root, dirs, files in os.walk(path):
        if not files:
            continue

        if files[0].split('.')[-1] == 'csv':
            csv_path = osp.join(root, files[0])
            print(osp.join(root, dirs[0]))
            with open(csv_path) as f:
                readCSV = csv.reader(f, delimiter=',')
                for row in readCSV:
                    c_img_path = osp.join(root, dirs[0], row[0].split('/')[-1])
                    l_img_path = osp.join(root, dirs[0], row[1].split('/')[-1])
                    r_img_path = osp.join(root, dirs[0], row[2].split('/')[-1])
                    c_img = cv2.imread(c_img_path)
                    l_img = cv2.imread(l_img_path)
                    r_img = cv2.imread(r_img_path)
                    #cf_img = np.fliplr(c_img)
                    #lf_img = np.fliplr(l_img)
                    #rf_img = np.fliplr(r_img)
                    c_measurement = float(row[3])
                    l_measurement = c_measurement + correction 
                    r_measurement = c_measurement - correction
                    images.append(c_img)
                    images.append(l_img)
                    images.append(r_img)
                    # images.append(cf_img)
                    # images.append(lf_img)
                    # images.append(rf_img)
                    measurements.append(c_measurement)
                    measurements.append(l_measurement)
                    measurements.append(r_measurement)
                    # measurements.append(-c_measurement)
                    # measurements.append(-l_measurement)
                    # measurements.append(-r_measurement)
    return images, measurements

"""
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
"""
images, measurements = parse_data('data')
images = np.array(images)
measurements = np.array(measurements)

X_train, X_valid, y_train, y_valid = train_test_split(images, measurements, test_size=0.2)
print(len(y_train), len(y_valid))

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

