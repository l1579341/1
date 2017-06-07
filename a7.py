from PyQt5 import QtCore, QtGui, QtWidgets
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

ilosc = 10
freq = 10

class NN(object):  # self oznacza ze zmienna jest dostepna globalnie
    def __init__(self):
        self.I = 784  # ilosc wejsc
        self.L = 100  # ilosc neuronow w ukrytej warstwie
        self.O = 10  # ilosc wyjsc
        self.S = 60000  # ilosc probek
        self.n = 0.1  # tempo uczenia sie
        self.n1 = 0.001  # temp regularyzacjiwagi na
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
            freq = int(input())
            classicalTrain(N, ilosc, freq)
            plt.show()
        if d == '2':
            Q = readPic()
            N.forward(Q)
            e = np.argsort(-N.output)
            print(e)
        if d == '3':
            save(N.W1, N.W2, N.B1, N.B2)
            print("saved")
        if d == '4':
            N.W1, N.W2, N.B1, N.B2 = load()
            print("read")
        if d == 'q':
            break

def display(N):
    print(ilosc, freq, N.L, N.n, N.n1)

def il_neur(N):
    N.L = ui.lineEdit_7.displayText()
def lern_spd(N):
    N.n = ui.lineEdit_8.displayText()
def reg_spd(N):
    N.n1 = ui.lineEdit_9.displayText()
def ilosc1():
    global ilosc
    ilosc = ui.lineEdit_2.displayText()
def freq1():
    global freq
    freq = ui.lineEdit_3.displayText()


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(768, 506)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.label = QtWidgets.QLabel(self.centralwidget)
        self.label.setGeometry(QtCore.QRect(10, 10, 231, 201))
        self.label.setText("")
        self.label.setPixmap(QtGui.QPixmap("../Desktop/Bez.jpg"))
        self.label.setScaledContents(True)
        self.label.setObjectName("label")
        self.label_2 = QtWidgets.QLabel(self.centralwidget)
        self.label_2.setGeometry(QtCore.QRect(270, 10, 171, 21))
        self.label_2.setObjectName("label_2")
        self.label_3 = QtWidgets.QLabel(self.centralwidget)
        self.label_3.setGeometry(QtCore.QRect(470, 10, 171, 21))
        self.label_3.setObjectName("label_3")
        self.label_5 = QtWidgets.QLabel(self.centralwidget)
        self.label_5.setGeometry(QtCore.QRect(560, 440, 201, 21))
        self.label_5.setObjectName("label_5")
        self.gridLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget.setGeometry(QtCore.QRect(480, 80, 279, 83))
        self.gridLayoutWidget.setObjectName("gridLayoutWidget")
        self.gridLayout = QtWidgets.QGridLayout(self.gridLayoutWidget)
        self.gridLayout.setContentsMargins(0, 0, 0, 0)
        self.gridLayout.setObjectName("gridLayout")
        self.label_6 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.gridLayout.addWidget(self.label_6, 0, 0, 1, 1)
        self.lineEdit_2 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_2.setObjectName("lineEdit_2")
        self.gridLayout.addWidget(self.lineEdit_2, 0, 1, 1, 1)
        self.label_8 = QtWidgets.QLabel(self.gridLayoutWidget)
        self.label_8.setObjectName("label_8")
        self.gridLayout.addWidget(self.label_8, 1, 0, 1, 1)
        self.lineEdit_3 = QtWidgets.QLineEdit(self.gridLayoutWidget)
        self.lineEdit_3.setObjectName("lineEdit_3")
        self.gridLayout.addWidget(self.lineEdit_3, 1, 1, 1, 1)
        self.pushButton_7 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_7.setObjectName("pushButton_7")
        self.gridLayout.addWidget(self.pushButton_7, 1, 2, 1, 1)
        self.pushButton_6 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_6.setObjectName("pushButton_6")
        self.gridLayout.addWidget(self.pushButton_6, 0, 2, 1, 1)
        self.pushButton_8 = QtWidgets.QPushButton(self.gridLayoutWidget)
        self.pushButton_8.setObjectName("pushButton_8")
        self.gridLayout.addWidget(self.pushButton_8, 2, 0, 1, 3)
        self.gridLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget_2.setGeometry(QtCore.QRect(10, 210, 231, 41))
        self.gridLayoutWidget_2.setObjectName("gridLayoutWidget_2")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.gridLayoutWidget_2)
        self.gridLayout_3.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.label_9 = QtWidgets.QLabel(self.gridLayoutWidget_2)
        self.label_9.setObjectName("label_9")
        self.gridLayout_3.addWidget(self.label_9, 0, 0, 1, 1)
        self.lineEdit_6 = QtWidgets.QLineEdit(self.gridLayoutWidget_2)
        self.lineEdit_6.setObjectName("lineEdit_6")
        self.gridLayout_3.addWidget(self.lineEdit_6, 0, 1, 1, 1)
        self.gridLayoutWidget_3 = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget_3.setGeometry(QtCore.QRect(10, 290, 261, 121))
        self.gridLayoutWidget_3.setObjectName("gridLayoutWidget_3")
        self.gridLayout_4 = QtWidgets.QGridLayout(self.gridLayoutWidget_3)
        self.gridLayout_4.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_4.setObjectName("gridLayout_4")
        self.label_10 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label_10.setObjectName("label_10")
        self.gridLayout_4.addWidget(self.label_10, 0, 0, 1, 1)
        self.lineEdit_8 = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.lineEdit_8.setObjectName("lineEdit_8")
        self.gridLayout_4.addWidget(self.lineEdit_8, 1, 1, 1, 1)
        self.lineEdit_9 = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.lineEdit_9.setObjectName("lineEdit_9")
        self.gridLayout_4.addWidget(self.lineEdit_9, 2, 1, 1, 1)
        self.lineEdit_7 = QtWidgets.QLineEdit(self.gridLayoutWidget_3)
        self.lineEdit_7.setObjectName("lineEdit_7")
        self.gridLayout_4.addWidget(self.lineEdit_7, 0, 1, 1, 1)
        self.label_11 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label_11.setObjectName("label_11")
        self.gridLayout_4.addWidget(self.label_11, 1, 0, 1, 1)
        self.label_12 = QtWidgets.QLabel(self.gridLayoutWidget_3)
        self.label_12.setObjectName("label_12")
        self.gridLayout_4.addWidget(self.label_12, 2, 0, 1, 1)
        self.pushButton_2 = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        self.pushButton_2.setObjectName("pushButton_2")
        self.gridLayout_4.addWidget(self.pushButton_2, 0, 2, 1, 1)
        self.pushButton_3 = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        self.pushButton_3.setObjectName("pushButton_3")
        self.gridLayout_4.addWidget(self.pushButton_3, 1, 2, 1, 1)
        self.pushButton_4 = QtWidgets.QPushButton(self.gridLayoutWidget_3)
        self.pushButton_4.setObjectName("pushButton_4")
        self.gridLayout_4.addWidget(self.pushButton_4, 2, 2, 1, 1)
        self.gridLayoutWidget_4 = QtWidgets.QWidget(self.centralwidget)
        self.gridLayoutWidget_4.setGeometry(QtCore.QRect(250, 80, 201, 80))
        self.gridLayoutWidget_4.setObjectName("gridLayoutWidget_4")
        self.gridLayout_5 = QtWidgets.QGridLayout(self.gridLayoutWidget_4)
        self.gridLayout_5.setContentsMargins(0, 0, 0, 0)
        self.gridLayout_5.setObjectName("gridLayout_5")
        self.pushButton = QtWidgets.QPushButton(self.gridLayoutWidget_4)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout_5.addWidget(self.pushButton, 3, 0, 1, 2)
        self.label_4 = QtWidgets.QLabel(self.gridLayoutWidget_4)
        self.label_4.setObjectName("label_4")
        self.gridLayout_5.addWidget(self.label_4, 1, 0, 1, 1)
        self.lineEdit = QtWidgets.QLineEdit(self.gridLayoutWidget_4)
        self.lineEdit.setObjectName("lineEdit")
        self.gridLayout_5.addWidget(self.lineEdit, 1, 1, 1, 1)
        self.pushButton_5 = QtWidgets.QPushButton(self.gridLayoutWidget_4)
        self.pushButton_5.setObjectName("pushButton_5")
        self.gridLayout_5.addWidget(self.pushButton_5, 2, 0, 1, 2)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 768, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "TEST"))
        self.label_3.setText(_translate("MainWindow", "TRENOWANIE"))
        self.label_5.setText(_translate("MainWindow", "Miziołek & Słowik 2017 All rights reserved"))
        self.label_6.setText(_translate("MainWindow", "ILOSĆ CYKLI UCZENIA"))
        self.lineEdit_2.setText(_translate("MainWindow", "10"))
        self.label_8.setText(_translate("MainWindow", "CZĘSTOTLIWOŚĆ RAPORTU"))
        self.lineEdit_3.setText(_translate("MainWindow", "1"))
        self.pushButton_7.setText(_translate("MainWindow", "ZATWIERDŹ"))
        self.pushButton_6.setText(_translate("MainWindow", "ZATWIERDŹ"))
        self.pushButton_8.setText(_translate("MainWindow", "START TRENING"))
        self.label_9.setText(_translate("MainWindow", "PRZEWIDYWANY WYNIK"))
        self.label_10.setText(_translate("MainWindow", "ILOŚĆ NEURONÓW"))
        self.lineEdit_8.setText(_translate("MainWindow", "0.1"))
        self.lineEdit_9.setText(_translate("MainWindow", "0.001"))
        self.lineEdit_7.setText(_translate("MainWindow", "100"))
        self.label_11.setText(_translate("MainWindow", "TEMPO UCZENIA"))
        self.label_12.setText(_translate("MainWindow", "TEMPO REGULARYZACJI"))
        self.pushButton_2.setText(_translate("MainWindow", "ZATWIERDŹ"))
        self.pushButton_3.setText(_translate("MainWindow", "ZATWIERDŹ"))
        self.pushButton_4.setText(_translate("MainWindow", "ZATWIERDŹ"))
        self.pushButton.setText(_translate("MainWindow", "START TEST"))
        self.label_4.setText(_translate("MainWindow", "ŚCIEŻKA:"))
        self.pushButton_5.setText(_translate("MainWindow", "ZATWIERDŹ"))


        self.pushButton_2.clicked.connect(lambda: il_neur(N))
        self.pushButton_3.clicked.connect(lambda: lern_spd(N))
        self.pushButton_4.clicked.connect(lambda: reg_spd(N))
        self.pushButton_6.clicked.connect(lambda: ilosc1())
        self.pushButton_7.clicked.connect(lambda: freq1())


        self.pushButton_5.clicked.connect(lambda: display(N))

if __name__ == "__main__":
    import sys
    N = NN()
    app = QtWidgets.QApplication(sys.argv)
    MainWindow = QtWidgets.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())