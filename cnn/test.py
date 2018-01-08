import cnn
import importlib
import numpy as np
#make sure module is updated
importlib.reload(cnn)

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

    return a, b, cl
def gradient_check():
    error_function = lambda o: o.sum()
    a, b, cl = init_test()
    cl.forward(a)
    sensitivity_array = np.ones(cl.output_array.shape, dtpye = np.float64)
    cl.backward(a, sensitivity_array, cl.activator)

    epsilon = 10e-4
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

def test():
    a, b, cl = init_test()
    cl.forward(a)
    print (cl.output_array)

def test_bp():
    a, b, cl = init_test()
    cl.backward(a, cl.activator)
    cl.update()
    print (cl.filters[0])
    print (cl.filters[1])

test_bp()