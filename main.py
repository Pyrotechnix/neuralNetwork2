import neural_network
import numpy as np
import idx2numpy
from matplotlib import pyplot as plt


data_dir = "mnist_data"

#TO DO: Fix shape of batches so that the loss is actually correct
#FIXED IT WORKS NOW :)
#backpropogation - in progress 👍
#Fix softmax so that it doesn't have zeros theres an explaination just follow it
#FIXED


#if I can be fucked fix the matrixes so that it doesn't use reduntant functions
#FIXED

#CHANGES
#Added softmax instead of sigmoid for the output layer, since its better for class classification
#switched to using multi class cross entropy outputs


def main():
    np.set_printoptions(suppress=False, precision=3)
    #copied from gemini:
    train_images_path = 'mnist_data/train-images.idx3-ubyte'
    train_labels_path = 'mnist_data/train-labels.idx1-ubyte'
    trainingData = idx2numpy.convert_from_file(train_images_path)
    trainingLabels = idx2numpy.convert_from_file(train_labels_path)
    print(trainingData)
    print(len(trainingData))
    print(trainingLabels)
    #784 input layer 2x layers of 16 10 output neurons corresponding to each letter
    nn = neural_network.neuralNetwork([784, 5, 5, 10])
    #nn.setValues([[[4, 8], [1, 10]], [[1, 1]]], [[[1], [4]], [[0], [0]]])
    #
    arr = trainingData[0]
    print(arr.flatten())
    nn.setTrainingData(trainingData, trainingLabels)
    nn.backPropagate()



    nn.formalCalculatingLossAndStuff(300)


"""
    f, subpl = plt.subplots(5, 10)
    for i in range(0, 5):
        for j in range(0, 10):
            subpl[i][j].imshow(trainingData[i * 10 + j], cmap='gray', vmin=0, vmax=255)
            subpl[i][j].set_title(trainingLabels[i * 10 + j])
            subpl[i][j].axis('off')
    plt.show()
"""

main()





