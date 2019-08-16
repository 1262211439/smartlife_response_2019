import os
import time
import csv

from numpy import array
from knndtw import KnnDtw
from imudata import IMUData

def scrapeNames(l, path):
    for root, _, files in os.walk(path, topdown=False):
        for name in files:
            l.append(os.path.join(root, name))

def labelResult(orig, predict):
    if orig == 1:
        if predict == 1:
            return "truepos"
        if predict == 2:
            return "falseneg"
    if orig == 2:
        if predict == 1:
            return "falsepos"
        if predict == 2:
            return "trueneg"

def accuracy(results):
    num, acc_num, recall, precision = 0, 0, 0, 0
    for result in results:
        if result == "truepos":
            num += 1
            acc_num += 1
            recall += 1
            precision += 1
        if result == "falsepos":
            precision += 1
        if result == "trueneg":
            acc_num += 1
        if result == "falseneg":
            recall += 1
    return (acc_num / len(results), num / precision, num / recall)

FALL = 1
NOT_FALL = 2

model = KnnDtw()    # change parameters here! using defaults currently  

train_files = list()
scrapeNames(train_files, r".\data\training\fall")
labels = [FALL] * len(train_files)
scrapeNames(train_files, r".\data\training\not fall")
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

test_files = list()
scrapeNames(test_files, r".\data\testing\fall")
labels = [FALL] * len(test_files)        
scrapeNames(test_files, r".\data\testing\not fall")
labels.extend([NOT_FALL] * (len(test_files) - len(labels)))   

results = [list(), list(), list(), list(), list()]
certainty = [list(), list(), list(), list(), list()]
times = [list(), list(), list(), list(), list()]
for i, file in enumerate(test_files):
    print("Predicting for", file)
    data = IMUData()
    with open(file, 'r') as f:
        for line in f:
            data.append(line)
    a = [array(data.kalmanX), array(data.kalmanY), array(data.x), array(data.y), array(data.z)]
    for j in range(1,6):    # testing subsample_step 1 through 5
        model.subsample_step = j
        t = time.time()
        result = model.predict(a)
        times[j-1].append(time.time() - t)
        results[j-1].append(labelResult(labels[i], result.mode[0]))
        certainty[j-1].append(result.count[0][0] / 5)

csv_name = ".\\test results\\test" + str(round(time.time())) + ".csv"
headers = ["train","test","step","accuracy","precision","recall","certainty","time"]
with open(csv_name, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow(headers)
    for i in range(0,5):
        acc, pre, rec = accuracy(results[i])
        writer.writerow([
            str(len(train_files)),
            str(len(test_files)),
            str(i + 1),
            str(acc),
            str(pre),
            str(rec),
            str(sum(certainty[i]) / len(certainty[i])),
            str(sum(times[i]) / len(times[i]))
        ])
