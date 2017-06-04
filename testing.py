from sklearn.datasets import fetch_mldata
import matplotlib.pyplot as plt
import numpy as np


mnist = fetch_mldata('MNIST original')
a1 = mnist.data[0:60000,:]
b1 = mnist.target[:60000]

a2 = mnist.data[60000:,:]
b2 = mnist.target[60000:]

a3 = a1[0,:].reshape((28,28))
plt.imshow(a3, cmap='gray')
plt.show()
