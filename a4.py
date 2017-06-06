import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from sklearn.datasets import fetch_mldata


"""X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = 'float32') #dane wejsciowe
Y = np.array([[1,0], [0,1], [0,1], [1,0]], dtype = 'float32')
A = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = 'float32') #dane wejsciowe
B = np.array([[1,0], [0,1], [0,1], [1,0]], dtype = 'float32')"""

X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = 'float32') #dane wejsciowe
Y = np.array([[0], [1], [1], [0]], dtype = 'int8')
A = np.array([[0, 0], [0, 1], [1, 0], [1, 1]], dtype = 'float32') #dane wejsciowe
B = np.array([[0], [1], [1], [0]], dtype = 'int8')

"""X = np.array(([3, 5], [5, 1], [10, 2], [6,1.5]), dtype=float)
Y = np.array(([75], [82], [93], [70]), dtype=float)
A = np.array(([4, 5.5], [4.5, 1], [9, 2.5], [6, 2]), dtype=float)
B = np.array(([[70, 89, 85, 75]]), dtype=float).T"""

"""X = np.array(([[10,2,0], [-1,1,7], [8,5,8], [0,3,-1], [-9,-4,10], [-8,6,3], [-7,-10,-2], [-4,-1,3], [0,2,-3], [6,8,7]]), dtype=float)
Y = np.array(([[31, -17.5, 19.5, 19.5, -70, 8, -63, -22.5, 20, 35]]), dtype=float).T
A = np.array(([[3,4,-8], [-2,9,8], [-3,3,5], [0,-9,-4], [-5,-5,-6], [-2,1,1], [-9,7,1], [8,-5,0], [6,3,5], [-5,10,2]]), dtype=float)
B = np.array(([[52, 21.5, -4.5, -37.5, -19.5, -1.5, 17.5, -11.5, 13.5, 39]]), dtype=float).T"""

"""X = np.array(([[9,9,0], [9,2,10], [7,0,10], [2,6,9], [6,8,1], [7,5,5], [1,9,5], [6,8,6], [1,1,6], [5,3,10]]), dtype=float)
Y = np.array(([[76, 17.5, 2.5, 27.5, 62.5, 40, 50, 52.5, 4, 15]]), dtype=float).T
A = np.array(([[9,5,6], [4,9,1], [1,2,4], [5,8,2], [8,3,2], [7,4,8], [7,6,5], [3,8,5], [2,5,1], [4,4,5]]), dtype=float)
B = np.array(([[42, 64, 13.5, 58.5, 37, 28.5, 45.5, 48.5, 38, 28.5]]), dtype=float).T"""

"""mnist = fetch_mldata('MNIST original')
X = mnist.data[0:60000,:].astype('float32')
Y = mnist.target[:60000]

a2 = mnist.data[60000:,:].astype('float32')
b2 = mnist.target[60000:]"""

S = 1
S1 = 1

Xn = X / S
Yn = Y
An = A / S
Bn = B

class NN(object):  #self oznacza ze zmienna jest dostepna globalnie
    def __init__(self):
        self.I = 2  # ilosc wejsc
        self.L = 10  #ilosc neuronow w ukrytej warstwie
        self.O = 2  # ilosc wyjsc
        self.S = 4 # ilosc probek
        self.n = 0.1 #tempo uczenia sie
        self.n1 = 0.001
        self.act = 1 # 0-sig 1-tanh 2-relu
        self.W1 = np.random.randn(self.I, self.L).astype('float32') / np.sqrt(self.I/1)
        self.W2 = np.random.randn(self.L, self.O).astype('float32') / np.sqrt(self.L/1)
        self.B1 = np.zeros((1, self.L))
        self.B2 = np.zeros((1, self.O))
        self.layer = np.zeros((self.S, self.L))
        self.output = np.zeros((self.S, self.O))
    def sigmoid(self, z):
        if self.act == 0:
            return 1/(1+np.exp(-z))
        if self.act == 1:
            za = np.exp(2*z)
            return (za - 1)/(za + 1)
        if self.act == 2:
            return  np.maximum(z, 0, z)
    def sigmoidPrime(self, z, z1 = 0):
        if self.act == 0:
            return np.exp(-z)/((1+np.exp(-z))**2)
        if self.act == 1:
            return 4/((np.exp(-z)+(np.exp(z)))**2)
        if self.act == 2:
            z[z <= 0] = 0
            z[z > 0] = 1
            return z

    def forward(self, In):
        self.hidden_layer = np.maximum(0, np.dot(In, self.W1) + self.B1)  # note, ReLU activation
        self.scores = np.dot(self.hidden_layer, self.W2) + self.B2
        exp_scores = np.exp(self.scores)
        self.probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)  # [N x K]
        return self.scores
    def lossCalc(self, y):
        corect_logprobs = -np.log(self.probs[range(self.S), y.T])
        data_loss = np.sum(corect_logprobs) / self.S
        reg_loss = 0.5 * self.n1 * np.sum(self.W1 * self.W1) + 0.5 * self.n1 * np.sum(self.W2 * self.W2)
        self.loss = data_loss + reg_loss
        return self.loss
    def backprop(self, In, y):
        dscores = self.probs
        dscores[range(self.S), y.T] -= 1
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
        if i % b == 0:
            print("iteration %d: loss %f:" % (i, N.loss))
        list.append(N.loss)
        N.backprop(Xn, Yn)

        N.forward(An)
        N.lossCalc(Bn)
        list1.append(N.loss)
        if i % b == 0:
            print("loss on test data: % f: " % (N.loss))
        i += 1
    N.forward(Xn)
    #print(N.outputS * S1)
    list = np.minimum(c, list)
    list1 = np.minimum(c, list1)
    print("Sprawdzenie algorytmu: ", computeNumericalGradient(N, Xn, Yn))
    plt.plot(list, label="training")
    plt.plot(list1, label="test")
    #plt.yscale('log')
    plt.legend()
    plt.grid(1)
def newTrain():
    T = trainer(N)
    T.train(Xn,Yn,An,Bn)
    N.forward(Xn)
    print(N.outputS)
    print("Sprawdzenie algorytmu: ", computeNumericalGradient(N, Xn, Yn))


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
classicalTrain(10000,1000)
#newTrain()
#print(N.forward([0.3, 0.5]))
#Plot3d()
plt.show()


