import numpy as np
import matplotlib.pyplot as plt

fn = 'log/results.csv'
results = np.loadtxt(fn, delimiter=",", dtype=np.float)
# print(results)
plt.plot(results[:, 0], results[:, 2])
plt.plot(results[:, 0], results[:, 3])
plt.xlabel('Epoches')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Eval'])
plt.show()