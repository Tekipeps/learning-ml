import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def predict(X, w):
    return X * w

def loss(X, Y, w):
    return np.average((predict(X, w) - Y) ** 2)

def train(X, Y, iterations, lr):
    w = 0
    for i in range(iterations):
        current_loss = loss(X, Y, w)
        print("Iteration %4d => loss: %.6f" % (i, current_loss))

        if loss(X, Y, w + lr) < current_loss:
            w += lr
        elif loss(X, Y, w - lr) < current_loss:
            w -= lr
        else:
            return w
    raise Exception("Couldn't converge within %d iterations" % iterations)

X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)

w = train(X, Y, iterations=10000, lr=0.01)

print("\nw=%.3f" % w)

print("Prediction: x=%d => y=%.2f" % (20, predict(20, w)))

sns.set()                       
plt.axis([0, 50, 0, 50])
plt.xticks(fontsize=10)
plt.yticks(fontsize=10)
plt.xlabel("Reservations", fontsize=20)
plt.ylabel("Pizzas", fontsize=20)
X, Y = np.loadtxt("pizza.txt", skiprows=1, unpack=True)
plt.plot(X, Y, "bo")
plt.show()