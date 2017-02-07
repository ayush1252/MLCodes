import numpy as np
import matplotlib.pyplot as plt

def findvar(X,Y):
	a1=((X*Y).mean()-X.mean()*Y.mean())/((X**2).mean()-X.mean()**2)
	a0=Y.mean()-a1*X.mean()
	return a0 , a1

X = np.array([1,3,8,5,12,10])
Y = np.array([3.4,6.7,11.8,19.5,34.3 ,28.5])

plt.scatter(X, Y)
a0,a1 = findvar(X, Y)

X1 = 0
pred1 = a0 + a1 * X1
X2 = 15
pred2 = a0 + a1 * X2

X_Line = [X1, X2]
Y_Line = [pred1, pred2]
print(X_Line)
print(Y_Line)
plt.plot(X_Line, Y_Line)
plt.show()
