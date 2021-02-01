import dataset as dataset
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
import numpy as np

#Create subplots and figures:

fig = plt.figure()
ax1 = plt.subplot(12, 2, 1)
ax2 = plt.subplot(12, 2, 2)
ax3 = plt.subplot(12, 2, 3)
ax4 = plt.subplot(12, 2, 4)
ax5 = plt.subplot(12, 2, 5)
ax6 = plt.subplot(12, 2, 6)
ax7 = plt.subplot(12, 2, 7)
ax8 = plt.subplot(12, 2, 8)
ax9 = plt.subplot(12, 2, 9)
ax10 = plt.subplot(12, 2, 10)
ax11 = plt.subplot(12, 2, 11)
ax12 = plt.subplot(12, 2, 12)

#load iris dataset:

iris = load_iris()
data = np.array(iris['data'])
targets = np.array(iris['target'])

#color the different classes:

cd = {0: 'r', 1: 'b', 2: 'g'}
cols = np.array([cd[target] for target in targets])

#plot graphs:

ax1.scatter(data[:, 0], data[:, 1], c=cols)
ax2.scatter(data[:, 0], data[:, 2], c=cols)
ax3.scatter(data[:, 0], data[:, 3], c=cols)
ax4.scatter(data[:, 1], data[:, 0], c=cols)
ax5.scatter(data[:, 1], data[:, 1], c=cols)
ax6.scatter(data[:, 1], data[:, 2], c=cols)
ax7.scatter(data[:, 1], data[:, 3], c=cols)
ax8.scatter(data[:, 2], data[:, 0], c=cols)
ax9.scatter(data[:, 2], data[:, 1], c=cols)
ax10.scatter(data[:, 2], data[:, 2], c=cols)
ax11.scatter(data[:, 2], data[:, 3], c=cols)
ax12.scatter(data[:, 3], data[:, 0], c=cols)

plt.show()









