import numpy as np
import matplotlib.pyplot as plt

x = [50, 60, 70, 80, 90, 100]
y_CNN_repro = [83.72, 78.13, 73.69, 68.4, 62.98, 60.42]
y_CNN = [82, 77.5, 71.5, 68.5, 64, 60]

y_NME_repro = [83.44, 77.1, 71.46, 65.32, 58.87, 55.24]
y_NME = [82, 76.5, 69.5, 65, 60.5, 56]

plt.plot(x, y_NME, '-o', label='Paper reported(NME)')
plt.plot(x, y_NME_repro, '-o', label='Reproduce(NME)')
plt.xlim(48, 102)
plt.ylim(0, 100)
plt.xticks(np.arange(50, 105, 5.0))
plt.yticks(np.arange(0, 110, 10))
plt.title('UCIR-NME, ImageNet-Subset, 50 base, 10 increments')
plt.xlabel('Number of classes')
plt.ylabel('Accuracy (%)')
plt.grid()
plt.legend()
plt.savefig('./tmp.png')
