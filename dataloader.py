import os
from string import ascii_lowercase
from random import sample
import cv2
import numpy as np

class dataloader:
    def __init__(self, train_path="./alphabet_training/",test_path="./alphabet_test/",data_type="alphabets"):
        self.train_path = train_path
        self.test_path = test_path
        self.data_type = data_type
        self.y_map = self.one_hot_init()
        self.x = []
        self.y = []

    def one_hot_init(self):
        dictionary = {}
        counter = 0
        if self.data_type == "alphabets":
            for i in ascii_lowercase:
                if i != "j" and i != "z":
                    dictionary[str(i)] = counter
                    counter+=1
            return dictionary
        if self.data_type == "digits":
            for i in range(10):
                dictionary[str(i)] = i
            return dictionary
        if self.data_type == "all":
            for i in range(10):
                dictionary[str(i)] = i
            counter = 10
            for i in ascii_lowercase:
                if i != "j" and i != "z":
                    dictionary[str(i)] = counter
                    counter+=1

    def load_images(self, path):
        self.x = []
        self.y = []
        for dir,subdir,files in os.walk(path):
            for file in files:
                if file.endswith(".jpg") or file.endswith(".jpeg") or file.endswith(".png") and dir != path:
                    img = cv2.imread(os.path.join(dir,file))
                    img = cv2.resize(img,(200,200))
                    img = cv2.normalize(img, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
                    self.x.append(img)
                    self.y.append(self.y_map[dir[-1]])

    def training_set(self):
        self.load_images(self.train_path)
        return np.array(self.x, dtype=np.float32), np.array(self.y, dtype=np.int32)

    def test_set(self):
        self.load_images(self.test_path)
        return np.array(self.x, dtype=np.float32), np.array(self.y, dtype=np.int32)
