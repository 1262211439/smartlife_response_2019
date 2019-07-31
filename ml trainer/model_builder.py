import os
import pickle
import numpy as np

from knndtw import KnnDtw
from re import search

FALL = 1
NOT_FALL = 2

model = KnnDtw()    # change parameters here! using defaults currently

class IMUData():
    def __init__(self):
        self.kalmanX = list()
        self.kalmanY = list()
        self.x = list()
        self.y = list()
        self.z = list()

    def append(self, s):
        pattern = r"(-?[0-9]+\.[0-9]+)"
        self.kalmanX.append(float(search("kalmanX " + pattern, s)[1]))
        self.kalmanY.append(float(search("kalmanY " + pattern, s)[1]))
        self.x.append(float(search("X = " + pattern, s)[1]))
        self.y.append(float(search("Y = " + pattern, s)[1]))
        self.z.append(float(search("Z = " + pattern, s)[1]))
        

train_files = list()
for root, dirs, files in os.walk(r".\data\training\fall", topdown=False):
    for name in files:
        print(os.path.join(root, name))
        train_files.append(os.path.join(root, name))
labels = [FALL] * len(train_files)
for root, dirs, files in os.walk(r".\data\training\not fall", topdown=False):
    for name in files:
        print(os.path.join(root, name))
        train_files.append(os.path.join(root, name))
labels.extend([NOT_FALL] * (len(labels) - len(train_files)))

train_data = list()
for file in train_files:
    data = IMUData()
    with open(file, 'r') as f:
        for line in f:
            data.append(line)
    train_data.append(data)

kalmanX = list()
kalmanY = list()
x = list()
y = list()
z = list()
for data in train_data:
    kalmanX.append(data.kalmanX)
    kalmanY.append(data.kalmanY)
    x.append(data.x)
    y.append(data.y)
    z.append(data.z)
data = [np.array(kalmanX), np.array(kalmanY), np.array(x), np.array(y), np.array(z)]

model.fit(data, np.array(labels))

with open("model.p", "wb") as f:
    pickle.dump(model, f)