import matplotlib.pyplot as plt
import numpy as np

# test = np.arange(-8, 8, 0.01)
# plt.plot(sigmoid(test))
# plt.show()
a = np.array([[1, 2], [3, 4], [5, 6]])
b = np.array([[1], [2], [3]])
w = np.array([[0.1, -0.2, 0.3], [-0.4, 0.5, 0.6]])
#print(np.dot(a, w) + b)
print(w)
B = np.array(([[70, 89, 85, 75]]), dtype=float)

def sigmoidPrime(z):
    z[z <= 0] = 0
    z[z > 0] = 1
    return z


print(sigmoidPrime(w))



