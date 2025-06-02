import numpy as np


# these functions are kinda useless as I rewrote them after finding out that the built-in methods were way better
def matrixMultiply(a, b):
    if len(a[0]) != len(b):
        print("Invalid!")
        return None
    else:
        return a @ b


def matrixAdd(a, b):
    if len(a) != len(b):
        print("Invalid!")
        return None
    else:
        return a + b


def matrixBroadcast(m, broadcastLen):
    returnMatrix = np.zeros((len(m), broadcastLen))
    for i in range(0, len(m)):
        returnMatrix[i] = (np.full(broadcastLen, m[i][0]))
    return returnMatrix


def sigmoid(n):
    return 1 / (1 + np.exp(-1 * n))


def sigmoidDerivative(n):
    return sigmoid(n) * (1 - sigmoid(n))


def sigmoidDerivativeBetter(sigNum):
    return sigNum * (1 - sigNum)


def activationFunction(n):
    return np.maximum(0, n)


class neuralNetwork:
    # takes inputNum as integer, outputNum as integer, layerInformation as list of ints detailing how many neurons
    # per layer
    inputCount = None
    outputCount = None
    # defining variables to help keep track of the number of inputs and outputs, these are just defined with
    # layerInformation anyway, so it's not really that important
    _weightMatrix = None
    # shouldn't need a matrix of neurons since the vector multiplication should output the value for the neurons anyway
    _biasMatrix = None
    _trainingData = None
    _trainingLabels = None
    _currentOutput = None
    _activationValues = None
    _catagoryDict = None

    def __init__(self, layerInformation):
        if len(layerInformation) < 3:
            print("Invalid, no hidden layer")
        else:
            self.inputCount = layerInformation[0]
            self.outputCount = layerInformation[-1]
            self._weightMatrix = []
            self._biasMatrix = []
            self._activationValues = []

            # initialising random weights
            for i in range(1, len(layerInformation)):
                weightLayer = np.random.randn(layerInformation[i], layerInformation[i - 1])
                self._weightMatrix.append(weightLayer)
                print(f"weight between layer {i - 1} and layer {i}:\n{weightLayer}")

            # initialising random biases
            for i in range(1, len(layerInformation)):
                biasLayer = np.random.randn(layerInformation[i])
                self._biasMatrix.append(biasLayer.reshape(-1, 1))
                print(f"bias matrix for layer {i}:\n{biasLayer}")

            for i in range(0, len(layerInformation)):
                layer = np.zeros(layerInformation[i])
                self._activationValues.append(layer.reshape(-1, 1))

    # trainingData should be only a 2 dimensional array, as it is converted into a 2 dimensional vector later
    def setTrainingData(self, trainingData, trainingLabels):
        self._trainingData = np.zeros([len(trainingData), len(trainingData[0].flatten())])
        for i in range(0, len(self._trainingData)):
            self._trainingData[i] = trainingData[i].flatten()
        # diving all values by the max to keep them in between 0 and 1 instead of making them big, since the
        # numpy exp function doesn't like large numbers, and its considered good practice anyway
        self._trainingData /= self._trainingData.max()
        print(self._trainingData[1])
        self.interpretCatagories(trainingLabels)
        self._trainingLabels = np.zeros([len(trainingData), len(self._catagoryDict), 1])
        for i in range(0, len(trainingLabels)):
            self._trainingLabels[i] = self._catagoryDict[str(trainingLabels[i])]
        print(self._trainingLabels)
        print("$$$")

    #converts labels to a list of catagories, and assigns an output neuron for each
    #this assumes that the number of output neurons is already correct
    def interpretCatagories(self, labels):
        catagories = []
        for i in labels:
            if i not in catagories:
                catagories.append(i)
        catagoryCount = len(catagories)
        for i in range(0, len(catagories)):
            catagories[i] = str(catagories[i])
        print(catagories)
        print(catagoryCount)
        self._catagoryDict = {}
        for i in range(0, catagoryCount):
            addArr = np.array([np.zeros(catagoryCount)])
            addArr[0][i] = 1
            self._catagoryDict[catagories[i]] = addArr.T
        return self._catagoryDict

    def setValues(self, weights, biases):
        print(self._weightMatrix)
        self._weightMatrix = weights
        print(self._weightMatrix)
        print(self._biasMatrix)
        self._biasMatrix = biases
        print(self._biasMatrix)

    # inputs taken as list of arrays, since it converts the arrays to 2d matrices later anyway
    def feedForward(self, inputs):
        if len(inputs[0]) != self.inputCount:
            print('invalid operation, number of inputs does not match')
        else:
            print('feeding forward!')
            currentLayer = np.array(inputs).T
            # print(currentLayer)
            self._activationValues[0] = currentLayer
            for i in range(0, len(self._weightMatrix) - 1):
                # multiplying and adding weights
                currentLayer = matrixMultiply(self._weightMatrix[i], currentLayer)
                # adding biases
                currentLayer = matrixAdd(matrixBroadcast(self._biasMatrix[i], len(currentLayer[0])), currentLayer)
                # sigmoid results
                currentLayer = activationFunction(currentLayer)
                self._activationValues[i + 1] = currentLayer
                # print(f"layer {i + 1}: {currentLayer}\n-------")
            currentLayer = matrixMultiply(self._weightMatrix[len(self._weightMatrix) - 1], currentLayer)
            # adding biases
            currentLayer = matrixAdd(matrixBroadcast(self._biasMatrix[len(self._biasMatrix) - 1], len(currentLayer[0])), currentLayer)
            print("Current layer before softmax:")
            print(currentLayer)
            #softmax function
            maxPerSample = np.max(currentLayer, axis=0, keepdims=True)
            print(maxPerSample)
            #employing a technique to prevent overflow where you shift untreated outputs down by the max value
            #this is supposed to prevent overflow and underflow
            shiftedCurrentLayer = currentLayer - maxPerSample
            print(shiftedCurrentLayer)
            #creating the denominators for the softmax function by column
            summedDenominators = np.sum(np.exp(shiftedCurrentLayer), axis=0, keepdims=True)
            currentLayer = np.exp(shiftedCurrentLayer) / summedDenominators
            self._currentOutput = currentLayer
            self._activationValues[-1] = currentLayer
            #setting last layer of activation values too just in case
            # print(f"activation values:\n{self._activationValues}")
            print("current layer after softmax:")
            print(currentLayer)
            return currentLayer

    #calculate the loss using MULTI ClASS CROSS ENTROPY 😨😨😨
    def calculateLoss(self, expected):
        print("Output:")
        print(self._currentOutput)
        print("Expected:")
        print(expected)
        losses = np.hstack(expected) * np.log(self._currentOutput)
        print(losses)
        return -np.sum(losses) / len(losses[0])

    def calculateLossDerivatives(self, expected):
        #giving us (actual - expected) / n
        outputDerivatives = (self._currentOutput - np.hstack(expected)) / len(self._currentOutput[0])
        #the / n part is important for later as it uses matrix multiplication to calcualte averages
        #otherwise it would have to multiply by 1/n later, but since we do it here it doesn't have to
        #activation values for the second last layer
        #this part is really cool and handles the averages automatically:
        #since the matrix multiplication is the dot prod of rows and columns
        #output derivatives is a C * B matrix, where B is the batch size
        #self._activationValues[-2] is the second last layer of activation values, and is size A * B
        #self._weights[-1] is the last layer of weights, and is size C * A
        #we want a matrix that is size C * A, so we multiply the C * B matrix by a B * A matrix
        #This can be achieved by multiplying derivatives * the transposition of the activation values
        #This also works because when doing this, the resulting numbers are the dot product between
        #the weight derivatives for that class, and all activations of that value in the previous layer
        #dot producting all of these will give the average derivative per weight over each weight, since
        #derivative of weight = derivative of output class * activation value, and for each weight, you sum
        #the derivatives for the weight with all the activations and sum them together
        #then it is already divided by 1/n, since that was done before
        #this does work I understand it and if I forget it thats ok because it works
        outputWeightDerivatives = outputDerivatives @ self._activationValues[-2].T
        print("changes to weights")
        print(outputWeightDerivatives)
        #this is more or less the same, but since the biases are just adjusted by an amount, you can simply
        #sum the derivatives to get an overall value that the biases should be adjusted by
        outputBiasDerivatives = np.sum(outputDerivatives, axis=1, keepdims=True)
        #fuck bro ts genius asf
        #these are the output derivatives for the last layer.
        #next goal is to find error signals of previous activation values.
        #you can do this in the same way as the weights, just in reverse, since the derivative of the cost with respect
        #to the activation values will just be the weight, which will be averaged in the same way.
        print("derivatives of outputs: ")
        print(outputDerivatives)
        #goal is to have it such that
        #This multiplication gets the sum of each weight for the neurons in the previous layer, and multiplies them by
        #the cost for the output neuron they correspond to, showing the average direction that the neuron should change
        beforeActivationDerivatives = outputDerivatives
        for i in range(1, len(self._weightMatrix) + 1):
            #multiplying weights by derivative of current layer to get activation derivatives of prev layer
            prevActivationDerivatives = self._weightMatrix[-i].T @ beforeActivationDerivatives
            #derivative should also have relu activations
            beforeActivationDerivatives = prevActivationDerivatives * (self._activationValues[-i - 1] > 0).astype(float)
            #i think this is wrong ^
            print("previous error signals:")
            #Something is wrong here, I'm not too sure what yet but I'll figure it out tomorrow
            #fixed, it was because I wasn't applying the activation derivative in the same loop
            print(beforeActivationDerivatives)
            print(self._activationValues[-i-1])
            print("weight changes:")
            #This makes the derivatives 0 if the activation for this
            print(beforeActivationDerivatives @ self._activationValues[-i - 1].T)
            print("bias changes:")
            print(np.sum(beforeActivationDerivatives, axis=1, keepdims=True))
            print("weight matrix: ")
            print(i)
            print(self._weightMatrix[-i])






    def formalCalculatingLossAndStuff(self, sampleCount):
        self.feedForward(self._trainingData[:sampleCount])
        self.calculateLoss(self._trainingLabels[:sampleCount])
        print('loss derivatives')
        self.calculateLossDerivatives(self._trainingLabels[:sampleCount])


    def testValue(self, index):
        # input nth training data sample into the network
        self.feedForward(self._trainingData[index])
        # error = (actual - expected) ^ 2
        print(f"actual output: {self._currentOutput}")
        print(f"expected output: {self._trainingLabels[index]}")
        return np.square(self._currentOutput - self._trainingLabels[index])

    # assuming batches are made previously, using all of self._trainingData as a batch for now
    def testBatch(self, batchSize):
        print(f"testing batch of {batchSize}")
        self.feedForward(self._trainingData[:batchSize])
        # print(f"actual output:\n{self._currentOutput}")
        labels = np.hstack(self._trainingLabels[:batchSize])
        # do this to get it into the right format, as before it wasn't
        # print(f"expected output:\n{labels}")
        costs = self.calculateLoss(labels)
        return costs

    # to work out derivative in terms of weight:
    # should be the average derivative in terms of each weight over all training samples
    # final total cost = sum of all

    # INCORRECT VVVV

    def calculateWeightDerivatives(self, layer, expectedResults):
        learningRate = 0.1
        print(f"Weights:\n{self._weightMatrix[layer]}")
        print(f"Activations of previous layer:\n{self._activationValues[layer - 1]}")
        print(f"Activations of current layer:\n{self._activationValues[layer]}")
        print(f"Expected results:\n{expectedResults}")
        # derivatives = self._activationValues[layer - 1] * ("""sigmoid derivative""" * (2 * activationValues[layer]
        derivatives = (2 * (self._activationValues[layer] - expectedResults) * sigmoidDerivative(
            self._activationValues[layer]))
        derivatives = derivatives.sum(axis=1, keepdims=True)
        print(derivatives)
        # print(derivatives * self._activationValues[layer - 1].T)
        summedActivationValues = self._activationValues[layer - 1].sum(axis=1, keepdims=True)
        print(summedActivationValues)
        derivatives = derivatives * summedActivationValues.T
        print("---")
        print(derivatives * learningRate)
        self._weightMatrix[layer] -= derivatives * learningRate

    def backPropagate(self):
        expected = self._trainingLabels
        actual = self._currentOutput
        # want to find derivative of cost function with respect to weights
        # for each layer
        # cost function = (y - actualY)^2/number of outputs
        # dC/dW = dC/dt * dy/dW

        # for each neuron y = sigmoid(Sum of (weights * z + bias))


    def backOneLayer(self, l):
        learningRate = 0.1
        # print("$$$$$$$$$$$")
        # weight derivative: dC/dA*dA/dZ*dZ/dW
        # A = (a - expected)^2
        # dC/dA = 2(a - expected)
        # dA/dZ = sigmoid derivative z
        # z = w*a(-1) + b
        # dZ/dW = a(-1)
        # dC/dW = 2(a - expected) * sigmoidDerivativeBetter(a) * a(-1)
        weightDerivative = 2 * (self._activationValues[l] - self._trainingLabels)
        weightDerivative *= sigmoidDerivativeBetter(self._activationValues[l])
        weightDerivative *= np.sum(self._activationValues[l - 1])
        # print(weightDerivative)
        # print(self._weightMatrix[l])
        # print("modifying values")
        self._weightMatrix[l] -= (weightDerivative * learningRate)
        # print(self._weightMatrix[l])
