import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
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
        self.L = 10  #ilosc neuronow w ukrytej warstwie
        self.O = 10  # ilosc wyjsc
        self.S = 60000 # ilosc probek
        self.n = 0.5 #tempo uczenia sie
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
        dscores = self.probs
        dscores[range(self.S), y] -= 1
        dscores /= self.S
        self.dW2 = np.dot(self.hidden_layer.T, dscores)
        db2 = np.sum(dscores, axis=0, keepdims=True)
        dhidden = np.dot(dscores, self.W2.T)
        dhidden[self.hidden_layer <= 0] = 0
        self.dW1 = np.dot(In.T, dhidden)
        db = np.sum(dhidden, axis=0, keepdims=True)
        self.dW2 += self.n1 * self.W2
        self.dW1 += self.n1 * self.W1
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

class trainer(object):
    def __init__(self, N):
        self.N = N
    def callbackF(self, params):
        self.N.setParams(params)
        self.N.forward(self.X)
        self.J.append(self.N.lossCalc( self.y))
        self.N.forward(self.testX)
        self.testJ.append(self.N.lossCalc( self.testY))
    def costFunctionWrapper(self, params, X, y):
        self.N.setParams(params)
        self.N.forward(X)
        cost = self.N.lossCalc(y)
        grad = self.N.computeGradients(X, y)
        return cost, grad
    def train(self, X, y, testX, testY):
        self.X = X
        self.y = y
        self.testX = testX
        self.testY = testY
        self.J = []
        self.testJ = []
        params0 = self.N.getParams()
        options = {'maxiter': 200, 'disp': True}
        _res = optimize.minimize(self.costFunctionWrapper, params0, jac=True, method='BFGS', args=(X, y), options=options, callback=self.callbackF)
        self.N.setParams(_res.x)
        self.optimizationResults = _res

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
            print("train loss:", list[i], "   test loss:", list1[i])

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
    #plt.yscale('log')
    plt.legend()
    plt.grid(1)
def newTrain():
    T = trainer(N)
    T.train(Xn,Yn,An,Bn)
    N.forward(Xn)
    #print("Sprawdzenie algorytmu: ", computeNumericalGradient(N, Xn, Yn))


    plt.plot(T.J, label="training")
    plt.plot(T.testJ, label="test")
    plt.grid(1)
    plt.xlabel("Iterations")
    plt.ylabel("Cost")
    plt.legend()
def Plot3d():
    from mpl_toolkits.mplot3d import Axes3D
    axis1 = np.linspace(0, 1, 100)
    axis2 = np.linspace(0, 1, 100)

    a, b = np.meshgrid(axis1, axis2)

    allInputs = np.zeros((a.size, 2))
    allInputs[:, 0] = a.ravel()
    allInputs[:, 1] = b.ravel()
    allOutputs = N.forward(allInputs)
    yy = np.dot(axis2.reshape(100, 1), np.ones((1, 100)))
    xx = np.dot(axis1.reshape(100, 1), np.ones((1, 100))).T
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    ax.scatter(X[:, 0], X[:, 1], Y, c='k', alpha=1, s=20)
    #ax.scatter(S * An[:, 0], S * An[:, 1], Bn * S1, c='m', alpha=1, s=10)
    surf = ax.plot_surface(xx*S, yy*S, allOutputs.reshape(100, 100)*S1, cmap=plt.cm.jet, alpha = 0.66)

    ax.set_xlabel('x1')
    ax.set_ylabel('x2')
    ax.set_zlabel('y')
classicalTrain(10,1)
#newTrain()
#print(N.forward([0.3, 0.5]))
#Plot3d()
plt.show()


