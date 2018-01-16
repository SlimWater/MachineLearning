import cnn
import loadData
import importlib
import numpy as np
import copy
#make sure module is updated
importlib.reload(cnn)
importlib.reload(loadData)


file_path_tl = '/home/xinying/PycharmProjects/MachineLearning/dataset/mnist/test_labels'
file_path_ti = '/home/xinying/PycharmProjects/MachineLearning/dataset/mnist/test_images'
file_path_trl = '/home/xinying/PycharmProjects/MachineLearning/dataset/mnist/training_labels'
file_path_tri = '/home/xinying/PycharmProjects/MachineLearning/dataset/mnist/training_images'
test_lables = loadData.mnist.load_test_labels(file_path_tl)
test_images = loadData.mnist.load_test_images(file_path_ti)
training_lables = loadData.mnist.load_training_labels(file_path_trl)
training_images = loadData.mnist.load_training_images(file_path_tri)

epoches  = 1
learning_rate = 0.0001
num_training = 60000
success_count = 0
relu = cnn.ReluActivator()
# lenet5 INPUT(32X32)->Conv1(6X28X28)->Submap2(6X14X14)->Conv3(16X10X10)->Submap4(16X5X5)->fc5(120)->fc6(84)->output(10)

conv1 = cnn.ConvLayer(28,28,1,5,5,6,2,1,relu,learning_rate)
submap2 = cnn.MaxPolling(28,28,6,2,2,2)
conv3 = cnn.ConvLayer(14,14,6,5,5,16,0,1,relu,learning_rate)
submap4 = cnn.MaxPolling(10,10,16,2,2,2)
fc5 = cnn.fc(400,84,relu,learning_rate)
'''
for filter in fc5.filters:
    filter.weights = np.random.uniform(-0.1,0.1, (84,400))
    filter.bias = 0
'''
fc6 = cnn.fc(84,10,relu,learning_rate)
'''
for filter in fc6.filters:
    filter.weights = np.random.uniform(-0.1,0.1, (10,84))
    filter.bias = 0
'''

####Training######
for i in range(epoches):
    success_count = 0
    for j in range(num_training):
        conv1.forward(training_images[j])

        submap2.forward(conv1.output_array)

        conv3.forward(submap2.output_array)

        submap4.forward(conv3.output_array)
        #print(submap4.output_array)

        fc5.forward(submap4.output_array)
        #print(fc5.output_array)
        fc6.forward(fc5.output_array)
        print(fc6.output_array)
        print(training_lables[j], fc6.output_array.argmax())
        if training_lables[j] == fc6.output_array.argmax():
           success_count +=1

        else:
            output_sensitivity_array = np.ones(10)*0.1
            output_sensitivity_array[training_lables[j]] = -1.0
            fc6.backward(output_sensitivity_array)
            fc6.update()
            fc5.backward(fc6.delta_array)
            fc5.update()
            submap4.backward(fc5.delta_array)
            conv3.backward(submap4.delta_array)
            conv3.update()
            submap2.backward(conv3.delta_array)
            conv1.backward(submap2.delta_array)
            conv1.update()

    error_rate = success_count/60000.0*100
    print("Error rate for epoche %d is: %f" % (i,error_rate))
