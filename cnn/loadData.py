import numpy as np

class mnist(object):
    def load_test_labels(file_path):
        label = np.fromfile(file_path,dtype = np.uint8, count=-1)
        return label[8:]
    def load_test_images(file_path):
        image = np.fromfile(file_path,dtype = np.uint8, count=-1)
        return image[16:].reshape(10000,28,28)
    def load_training_labels(file_path):
        label = np.fromfile(file_path,dtype = np.uint8, count=-1)
        return label[8:]
    def load_training_images(file_path):
        image = np.fromfile(file_path,dtype = np.uint8, count=-1)
        return np.delete(image, np.s_[0:16]).reshape(60000,28,28)



