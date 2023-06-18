import matplotlib.pyplot as plt
import random

x=[i for i in range(-25, 51)]
y_hat=[(5*i**2)+(2*i)+3+random.randint(-1000,1000) for i in range(-25, 51)]
y=[(5*i**2)+(2*i)+3 for i in range(-25, 51)]
s = [n*0.5 for n in range(len(x))]
plt.scatter(x, y_hat, s, color='r')
plt.plot(x, y, label='5x^2+2x+3', color='g')
plt.legend()
plt.show()
