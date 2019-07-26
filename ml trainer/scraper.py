import os

from trainer import KnnDtw

data_files = list()
for root, dirs, files in os.walk(".\data", topdown=False):
    for name in files:
        print(os.path.join(root, name))
        data_files.append(os.path.join(root, name))

