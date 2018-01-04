import cnn
import importlib
import numpy as np
#make sure module is updated
importlib.reload(cnn)

input_array2d= np.ones((9,9), dtype = np.int16)
input_array3d = np.ones((3,9,9),dtype = np.int16)

kernel_array = np.ones((3,3),dtype = np.int16)

conv = cnn.ConvLayer(9,9,3,3,3,1,1,1,'relu',0.01)