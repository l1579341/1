from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import numpy as np
from scipy import misc
"""
mnist = fetch_mldata('MNIST original')
a1 = mnist.data[0:60000,:].astype('float32')
b1 = mnist.target[:60000]

a2 = mnist.data[60000:,:]
b2 = mnist.target[60000:]

a3 = a1[0,:].reshape((28,28))
plt.imshow(a3, cmap='gray')
plt.show()
"""

a = np.array([[1,3,4],[4,6,7]])
b = np.array([[2,3,4],[4,6,7]])
c = np.array([[3,3,4],[4,6,7]])
d = np.array([[4,3,4],[4,6,7]])
"""
plik = open('aaa.txt', 'r+')
plik.write(a)
plik.close()
plik = open('aaa.txt', 'r+')
#print(linecache.getline('aaa.txt',2))
print(plik.open())
"""
"""
np.save('b.npy', a)
print(np.load('b.npy'))
"""
def save(w1,w2,b1,b2):
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
#save(a, b, c, d)
#e, f, g, h = load()
#print(f)

def readPic():
    face = misc.face()
    face = misc.imread(r'C:\Users\Pawel\Desktop\Bez.jpg')
    x = (np.average(face, axis = 2))
    return x
print(readPic())
plt.imshow(readPic(), cmap='gray')
plt.show()
