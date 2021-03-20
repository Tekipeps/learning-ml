import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns

sns.set()                       
plt.axis([0, 50, 0, 50])
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Reservations", fontsize=20)
plt.ylabel("Pizzas", fontsize=20)
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
plt.plot(X, Y, "bo")
plt.show()