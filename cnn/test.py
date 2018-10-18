import cnn
import loadData
import importlib
from matplotlib import pyplot as plt
import numpy as np
import copy
#make sure module is updated
importlib.reload(cnn)
importlib.reload(loadData)
def init_fc_test():
    a = np.array(
        [[[0,1,1,0,2],
          [2,2,2,2,1],
          [1,0,0,2,0],
          [0,1,1,0,0],
          [1,2,0,0,2]],
         [[1,0,2,2,0],
          [0,0,0,2,0],
          [1,2,1,2,1],
          [1,0,0,0,0],
          [1,2,1,1,1]],
         [[2,1,2,0,0],
          [1,0,0,1,0],
          [0,2,1,0,1],
          [0,1,2,2,2],
          [2,1,0,0,1]]])
    b = np.array(
        [[[0,1,1],
          [2,2,2],
          [1,0,0]],
         [[1,0,2],
          [0,0,0],
          [1,2,1]]])
    relu = cnn.ReluActivator()
    fc = cnn.fc(18,3,relu,0.001)
    for filter in fc.filters:
        filter.weights = np.random.randint(2,size = 18)
        filter.bias = 0
    return a, b, fc

def init_test():
    a = np.array(
        [[[0,1,1,0,2],
          [2,2,2,2,1],
          [1,0,0,2,0],
          [0,1,1,0,0],
          [1,2,0,0,2]],
         [[1,0,2,2,0],
          [0,0,0,2,0],
          [1,2,1,2,1],
          [1,0,0,0,0],
          [1,2,1,1,1]],
         [[2,1,2,0,0],
          [1,0,0,1,0],
          [0,2,1,0,1],
          [0,1,2,2,2],
          [2,1,0,0,1]]])
    b = np.array(
        [[[0,1,1],
          [2,2,2],
          [1,0,0]],
         [[1,0,2],
          [0,0,0],
          [1,2,1]]])
    relu = cnn.ReluActivator()
    cl = cnn.ConvLayer(5,5,3,3,3,2,1,2,relu,0.001)
    cl.filters[0].weights = np.array(
        [[[-1,1,0],
          [0,1,0],
          [0,1,1]],
         [[-1,-1,0],
          [0,0,0],
          [0,-1,0]],
         [[0,0,-1],
          [0,1,0],
          [1,-1,-1]]], dtype=np.float64)
    cl.filters[0].bias=1
    cl.filters[1].weights = np.array(
        [[[1,1,-1],
          [-1,-1,1],
          [0,-1,1]],
         [[0,1,0],
         [-1,0,-1],
          [-1,1,0]],
         [[-1,0,0],
          [-1,0,1],
          [-1,0,0]]], dtype=np.float64)
    cl.filters[1].bias = 1

    return a, b, cl
def gradient_check():
    error_function = lambda o: o.sum()
    a, b, cl = init_test()
    cl.forward(a)
    sensitivity_array = np.ones(cl.output_array.shape, dtype = np.float64)
    cl.backward(a,sensitivity_array, cl.activator)
    epsilon = 10e-4

    #'''
    for d in range(cl.filters[0].weights_grad.shape[0]):
        for i in range(cl.filters[0].weights_grad.shape[1]):
            for j in range(cl.filters[0].weights_grad.shape[2]):
                cl.filters[0].weights[d,i,j] += epsilon
                cl.forward(a)
                err1 = error_function(cl.output_array)
                cl.filters[0].weights[d,i,j] -= 2*epsilon
                cl.forward(a)
                err2 = error_function(cl.output_array)

                expect_grad = (err1 - err2) / (2 * epsilon)
                cl.filters[0].weights[d,i,j] += epsilon
                print ('weights(%d,%d,%d): expected - actural %f - %f' % ( d, i, j, expect_grad, cl.filters[0].weights_grad[d,i,j]))
    #'''
def test():
    a, b, cl = init_test()
    cl.forward(a)
    index = np.where(a == a.max())
    print (index)

def test_bp():
    a, b, cl = init_test()
    sensitivity_array = np.ones(cl.output_array.shape, dtype=np.float64)
    cl.backward(a,sensitivity_array, cl.activator)
    #cl.update()

#'''
    print ('Befor filer0')
    print( cl.filters[0].get_weights())
    print ('Befor filer1')
    print( cl.filters[1].get_weights())
    print('*****************************')
    cl.update()
    print (cl.filters[0].get_weights())
    print('*****************************')
    print (cl.filters[1].get_weights())
#'''
def polling_test():
    a, b, cl = init_test()
    polling = cnn.MaxPolling(5,5,3,3,3,2)
    polling.forward(a)
    sensitivity_array = np.ones((3,2,2), dtype=np.float64)
    polling.backward(sensitivity_array)
    print(a)
    print(polling.output_array)
    print(polling.delta_array)

def fc_test():
    a,b,fc = init_fc_test()
    fc.forward(b)
    sensitivity_array = np.ones((3), dtype=np.float64)
    fc.backward(sensitivity_array)
    print(fc.input_array)
    print(fc.filters[0].get_weights())
    print(fc.filters[0].weights_grad)
    fc.update()
    print(fc.filters[0].get_weights())


def loadData_test():
    file_path_tl = '/home/xinying/MachineLearning/dataset/test_label'
    file_path_ti = '/home/xinying/MachineLearning/dataset/test_image'
    file_path_trl = '/home/xinying/MachineLearning/dataset/training_label'
    file_path_tri = '/home/xinying/MachineLearning/dataset/training_image'
    test_lables = loadData.mnist.load_test_labels(file_path_tl)
    test_images = loadData.mnist.load_test_images(file_path_ti)
    training_lables = loadData.mnist.load_training_labels(file_path_trl)
    training_images = loadData.mnist.load_training_images(file_path_tri)
    print(test_lables[0])
    plt.imshow(test_images[0])
    print(training_lables[0])
    plt.figure()
    plt.imshow(training_images[0])
    plt.show()
def softmax_test():
    softmax = cnn.softmax()
    X = np.array(range(10))
    forw = softmax.forward(X)
    backw = softmax.backward(X)
    print(forw)
    print(backw)
def cross_entropy():
    softmax = cnn.softmax()
    X = np.array(range(10))
    forw = softmax.forward(X)
    y = np.array([0,0,0,1,0,0,0,0,0,0])
    ce = cnn.cost_function.cross_entropy() 
    ceforw = ce.forward(forw,y)
    cebackw = ce.backward(forw,y)
    print(ceforw)
    print(cebackw)

#test()
#test_bp()
#test_bp()
#gradient_check()
#polling_test()
#fc_test()
#loadData_test()
#softmax_test()
#cross_entropy()
