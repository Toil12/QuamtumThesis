import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

t =np.ndarray(int([2]))
sig = 1 / (1 + np.exp(-t))

plt.axvline(color="grey")

plt.xlim(-10, 10)
plt.xlabel("t")
plt.legend(fontsize=14)

plt.plot(t)

ax = plt.figure().gca()
ax.xaxis.set_major_locator(MaxNLocator(integer=True))

plt.show()


