import numpy as np
import matplotlib.pyplot as plt

x = [50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100]
y_CNN_repro = [83.72, 79.49, 75.3, 71.42, 69.77, 66.51, 65.5, 63.41, 60.33, 59.73, 57.32]
y_CNN = [82, 80, 76, 71.5, 68.5, 66, 64.5, 62.5, 61, 58.5, 57.5]

y_NME_repro = [83.44, 79.16, 74.77, 69.94, 67.77, 64.32, 62.65, 59.88, 56.07, 54.82, 52.64]
y_NME = [82, 80, 74.5, 72, 66.5, 64, 61.5, 60, 58, 56, 53]
'''
y_CNN_repro = [83.72, 78.13, 73.69, 68.4, 62.98, 60.42]
y_CNN = [82, 77.5, 71.5, 68.5, 64, 60]

y_NME_repro = [83.44, 77.1, 71.46, 65.32, 58.87, 55.24]
y_NME = [82, 76.5, 69.5, 65, 60.5, 56]
'''

plt.plot(x, y_CNN, '-o', label='Paper reported(CNN)')
plt.plot(x, y_CNN_repro, '-o', label='Reproduce(CNN)')
plt.xlim(48, 102)
plt.ylim(0, 100)
plt.xticks(np.arange(50, 105, 5.0))
plt.yticks(np.arange(0, 110, 10))
plt.title('UCIR-CNN, ImageNet-Subset, 50 base, 5 increments')
plt.xlabel('Number of classes')
plt.ylabel('Accuracy (%)')
plt.grid()
plt.legend()
plt.savefig('./tmp.png')
