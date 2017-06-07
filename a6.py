import numpy as np
import matplotlib.pyplot as plt
from scipy import misc
from sklearn.datasets import fetch_mldata

mnist = fetch_mldata('MNIST original')
X = mnist.data[0:60000, :].astype('float32')
Y = mnist.target[:60000].astype('int8')
A = mnist.data[60000:, :].astype('float32')
B = mnist.target[60000:].astype('int8')

S = 255
Xn = X / S
Yn = Y
An = A / S
Bn = B


class NN(object):  # self oznacza ze zmienna jest dostepna globalnie
    def __init__(self):
        self.I = 784  # ilosc wejsc
        self.L = 200  # ilosc neuronow w ukrytej warstwie
        self.O = 10  # ilosc wyjsc
        self.S = 60000  # ilosc probek
        self.n = 0.2  # tempo uczenia sie
        self.n1 = 0.01  # temp regularyzacjiwagi na
        self.W1 = np.random.randn(self.I, self.L).astype('float32') / np.sqrt(self.I / 1)  # losowe  1 warstwie
        self.W2 = np.random.randn(self.L, self.O).astype('float32') / np.sqrt(self.L / 1)  # losowe  1 warstwie
        self.B1 = np.zeros((1, self.L))  # deklaracja pustej tablicy bias1
        self.B2 = np.zeros((1, self.O))  # deklaracja pustej tablicy bias2

    def forward(self, In):
        self.S = In.shape[0]  # odczytuje ilosc probek
        self.layer = np.maximum(0, np.dot(In, self.W1) + self.B1)  # obliczanie wartosci na ukrytej warstwie
        self.output = np.dot(self.layer, self.W2) + self.B2  # obliczanie wartosci na wyjsciu
        return self.output

    def lossCalc(self, y):
        output_exponential = np.exp(self.output)  # e^x
        self.smx_exponent = output_exponential / np.sum(output_exponential, axis=1, keepdims=True)
        smx = -np.log(self.smx_exponent[range(self.S), y])
        data_loss = np.sum(smx) / self.S
        reg_loss = 0.5 * self.n1 * np.sum(self.W1 * self.W1) + 0.5 * self.n1 * np.sum(self.W2 * self.W2)
        self.loss = data_loss + reg_loss
        return self.loss

    def backprop(self, In, target):
        deltaO = self.smx_exponent
        deltaO[range(self.S), target] -= 1
        deltaO /= self.S
        self.dW2 = np.dot(self.layer.T, deltaO) + self.n1 * self.W2
        db2 = np.sum(deltaO, axis=0, keepdims=True)
        delta2 = np.dot(deltaO, self.W2.T)
        delta2[self.layer <= 0] = 0  # pochodna relu
        self.dW1 = np.dot(In.T, delta2) + self.n1 * self.W1
        db = np.sum(delta2, axis=0, keepdims=True)
        self.W1 += -self.n * self.dW1
        self.B1 += -self.n * db
        self.W2 += -self.n * self.dW2
        self.B2 += -self.n * db2


def save(w1, w2, b1, b2):
    np.save('weights1.npy', w1)
    np.save('weights2.npy', w2)
    np.save('biases1.npy', b1)
    np.save('biases2.npy', b2)


def load():
    w1 = np.load('weights1.npy')
    w2 = np.load('weights2.npy')
    b1 = np.load('biases1.npy')
    b2 = np.load('biases2.npy')
    return w1, w2, b1, b2


def readPic():
    face = misc.face()
    face = misc.imread('Bez.jpg')
    x = (np.average(face, axis=2))
    return np.reshape(x, 28 ** 2)

def classicalTrain(N, a, b, c=100):
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
            print("round:", i, "   train loss:", list[i], "   test loss:", list1[i])
        i += 1
    list = np.minimum(c, list)
    list1 = np.minimum(c, list1)

    N.forward(Xn)
    predicted_class = np.argmax(N.output, axis=1)
    print('training accuracy: %.2f' % (np.mean(predicted_class == Yn)))
    N.forward(An)
    predicted_class = np.argmax(N.output, axis=1)
    print('testing accuracy: %.2f' % (np.mean(predicted_class == Bn)))

    plt.plot(list, label="training")
    plt.plot(list1, label="test")
    plt.legend()
    plt.grid(1)


def loop():
    N = NN()
    while True:
        print(" press 1 to learn\n press 2 to test\n press 3 to save wages\n press 4 to read wages")
        print(" press q to exit\n")
        d = input()
        if d == '1':
            print("ile rund?")
            ilosc = int(input())
            print("co ile wyswietlac wynik?")
            f = int(input())
            classicalTrain(N, ilosc, f)
            plt.show()
        if d == '2':
            Q = readPic()
            Q /= 255
            N.forward(Q)
            e = np.argsort(-N.output)
            print(e)
            print(N.output)
        if d == '3':
            save(N.W1, N.W2, N.B1, N.B2)
            print("saved")
        if d == '4':
            N.W1, N.W2, N.B1, N.B2 = load()
            print("read")
        if d == 'q':
            break

loop()
