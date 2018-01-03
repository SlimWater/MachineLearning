import cnn
import importlib

#make sure module is updated
importlib.reload(cnn)

conv = cnn.ConvLayer(2,3,3,4,5,5,2,1,'relu',0.01)