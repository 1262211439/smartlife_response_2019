##
##  SMARTLIFE RESPONSIVE SYSTEM
##
##  Written by Xander Will
##
##  trainer.py
##  
##  "Builds model.p using data found in .\data\training"
##

import os
import pickle

from numpy import array
from knndtw import KnnDtw
from imudata import IMUData

FALL = 1
NOT_FALL = 2

model = KnnDtw(subsample_step=5)    # change parameters here! using defaults currently        

train_files = list()
for root, dirs, files in os.walk(r".\data\training\fall", topdown=False):
    for name in files:
        train_files.append(os.path.join(root, name))
labels = [FALL] * len(train_files)
for root, dirs, files in os.walk(r".\data\training\not fall", topdown=False):
    for name in files:
        train_files.append(os.path.join(root, name))
labels.extend([NOT_FALL] * (len(train_files) - len(labels)))

train_data = list()
for file in train_files:
    print("Reading", file)
    data = IMUData()
    with open(file, 'r') as f:
        for i, line in enumerate(f):
            data.append(line)
            if i == 112:
                    break
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
data = [array(kalmanX), array(kalmanY), array(x), array(y), array(z)]

model.fit(data, array(labels))

with open("model.p", "wb") as f:
    pickle.dump(model, f)