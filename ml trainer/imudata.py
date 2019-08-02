from numpy import array
from re import search

class IMUData():
    def __init__(self):
        self.kalmanX = list()
        self.kalmanY = list()
        self.x = list()
        self.y = list()
        self.z = list()

    def append(self, s):
        pattern = r"(-?[0-9]+\.[0-9]+)"
        self.kalmanX.append(float(search("kalmanX +" + pattern, s)[1]))
        self.kalmanY.append(float(search("kalmanY +" + pattern, s)[1]))
        self.x.append(float(search("X = +" + pattern, s)[1]))
        self.y.append(float(search("Y = +" + pattern, s)[1]))
        self.z.append(float(search("Z = +" + pattern, s)[1]))

    def convert(self):
        self.kalmanX = array(self.kalmanX)
