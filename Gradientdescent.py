import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
def returnderv(a0,a1,X_Train,Y_Train):
	temp0=(X_Train.mean()*a1+a0+Y_Train.mean())
	temp0=temp0*2
	temp1=(X_Train.mean()*a0+(X_Train**2).mean()*a1-(X_Train*Y_Train).mean())
	temp1=temp1*2
	#print(temp0)
	#print(temp1)
	return temp0,temp1

def func(X_Train,Y_Train,a0,a1):
	sum=0
	for i in range(0,len(X_Train)):
		sum=sum+((a0+a1*X_Train[i])-Y_Train[i])**2
	return sum


x=np.array([1,2,3,4,5,6,7,8,9,10])
y=np.array([1.2,4.2,6.7,10.5,13.6,18.9,19.6,23,25,27])

X_Train,X_Test,Y_Train,Y_Test=train_test_split(x,y,test_size=0.0)
a0=1
a1=1
alpha=0.003
oldval=func(X_Train,Y_Train,a0,a1)
der1,der2=returnderv(a0,a1,X_Train,Y_Train)
temp0=a0-(alpha*der1)
temp1=a1-(alpha*der2)
a0=temp0
a1=temp1
newval=func(X_Train,Y_Train,a0,a1)
while newval-oldval<-0.0000001:

	
	der1,der2=returnderv(a0,a1,X_Train,Y_Train)
	temp0=a0-(alpha*der1)
	temp1=a1-(alpha*der2)
	a0=temp0
	a1=temp1
	oldval=newval
	newval=func(X_Train,Y_Train,a0,a1)


ypred1=a0+a1*7.7
ypred2=a0+a1*1.4

X_plot=np.array([7.7,1.4])
Y_plot=np.array([ypred1,ypred2])
print(Y_plot)
plt.scatter(x,y)
plt.plot(X_plot,Y_plot)
plt.show()
print(a0)
print(a1)





