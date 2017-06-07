import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_mldata

"""
X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = 'float32') #dane wejsciowe
Y = np.array([0,1,1,0], dtype = 'int8')
A = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = 'float32') #dane wejsciowe
B = np.array([0,1,1,0], dtype = 'int8')"""
mnist = fetch_mldata('MNIST original')
X = mnist.data[0:60000,:].astype('float32')
Y = mnist.target[:60000].astype('int8')

A = mnist.data[60000:,:].astype('float32')
B = mnist.target[60000:].astype('int8')

S = 256
Xn = X / S
Yn = Y
An = A / S
Bn = B
class NN(object):  #self oznacza ze zmienna jest dostepna globalnie
    def __init__(self):
        self.I = 784  # ilosc wejsc
        self.L = 300  #ilosc neuronow w ukrytej warstwie
        self.O = 10  # ilosc wyjsc
        self.S = 60000 # ilosc probek
        self.n = 0.1 #tempo uczenia sie
        self.n1 = 0.001
        self.W1 = np.random.randn(self.I, self.L).astype('float32') / np.sqrt(self.I/1)
        self.W2 = np.random.randn(self.L, self.O).astype('float32') / np.sqrt(self.L/1)
        self.B1 = np.zeros((1, self.L))
        self.B2 = np.zeros((1, self.O))
    def forward(self, In):
        self.S = In.shape[0]
        self.hidden_layer = np.maximum(0, np.dot(In, self.W1) + self.B1)  # note, ReLU activation
        self.scores = np.dot(self.hidden_layer, self.W2) + self.B2
        exp_scores = np.exp(self.scores)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]
        return self.scores
    def lossCalc(self, y):
        corect_logprobs = -np.log(self.probs[range(self.S), y])
        data_loss = np.sum(corect_logprobs) / self.S
        reg_loss = 0.5 * self.n1 * np.sum(self.W1 * self.W1) + 0.5 * self.n1 * np.sum(self.W2 * self.W2)
        self.loss = data_loss + reg_loss
        return self.loss
    def backprop(self, In, y):
        delta3 = self.probs
        delta3[range(self.S), y] -= 1
        delta3 /= self.S
        self.dW2 = np.dot(self.hidden_layer.T, delta3) + self.n1 * self.W2
        db2 = np.sum(delta3, axis=0, keepdims=True)
        delta2 = np.dot(delta3, self.W2.T)
        delta2[self.hidden_layer <= 0] = 0
        self.dW1 = np.dot(In.T, delta2) + self.n1 * self.W1
        db = np.sum(delta2, axis=0, keepdims=True)
        self.W1 += -self.n * self.dW1
        self.B1 += -self.n * db
        self.W2 += -self.n * self.dW2
        self.B2 += -self.n * db2
    def getParams(self):
        params = np.concatenate((self.W1.ravel(), self.W2.ravel()))
        return params
    def setParams(self, params):
        W1_start = 0
        W1_end = self.L * self.I
        self.W1 = np.reshape(params[W1_start:W1_end], (self.I , self.L))
        W2_end = W1_end + self.L*self.O
        self.W2 = np.reshape(params[W1_end:W2_end], (self.L, self.O))
    def computeGradients(self, X, y):
        self.forward(X)
        self.lossCalc(y)
        self.backprop(X, y)
        return np.concatenate((self.dW1.ravel(), self.dW2.ravel()))
N = NN()

def computeNumericalGradient(N, X, Y):
    paramsInitial = N.getParams()
    numgrad = np.zeros(paramsInitial.shape)
    perturb = np.zeros(paramsInitial.shape)
    e = 1e-5
    for p in range(len(paramsInitial)):
        perturb[p] = e
        N.setParams(paramsInitial + perturb)
        N.forward(X)
        loss2 = N.lossCalc(Yn)
        N.setParams(paramsInitial - perturb)
        N.forward(X)
        loss1 = N.lossCalc(Yn)
        numgrad[p] = (loss2 - loss1) / (2 * e)
        perturb[p] = 0
    N.setParams(paramsInitial)
    grad = N.computeGradients(X, Y)
    return np.linalg.norm(grad - numgrad) / np.linalg.norm(grad + numgrad)
def classicalTrain(a, b, c = 100):
    list = []
    list1 = []
    i = 0
    while i < a:
        N.forward(Xn)
        N.lossCalc(Yn)
        list.append(N.loss)
        N.backprop(Xn, Yn)
        N.forward(An)
        N.lossCalc(Bn)
        list1.append(N.loss)
        if i % b == 0:
            print("round:", i,"   train loss:", list[i], "   test loss:", list1[i])

        i += 1
    list = np.minimum(c, list)
    list1 = np.minimum(c, list1)
    #print("Sprawdzenie algorytmu: ", computeNumericalGradient(N, Xn, Yn))

    N.forward(Xn)
    predicted_class = np.argmax(N.scores, axis=1)
    print('training accuracy: %.2f' % (np.mean(predicted_class == Yn)))
    N.forward(An)
    predicted_class = np.argmax(N.scores, axis=1)
    print('testing accuracy: %.2f' % (np.mean(predicted_class == Bn)))

    plt.plot(list, label="training")
    plt.plot(list1, label="test")
    plt.legend()
    plt.grid(1)

classicalTrain(20,1)
plt.show()


