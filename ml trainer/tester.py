import os
import pickle

from numpy import array
from imudata import IMUData

with open("model.p", "rb") as f:
    model = pickle.load(f)

test_files = list()
for root, dirs, files in os.walk(r".\data\testing", topdown=False):
    for name in files:
        test_files.append(os.path.join(root, name))

test_data = list()
for file in test_files:
    data = IMUData()
    with open(file, 'r') as f:
        for line in f:
            data.append(line)
    print(file, "=", model.predict([array(data.kalmanX), array(data.kalmanY), array(data.x), array(data.y), array(data.z)]))