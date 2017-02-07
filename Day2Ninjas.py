from sklearn import datasets
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np 
from sklearn import metrics


datast=datasets.load_breast_cancer()
print datast.DESCR
X=datast.data
Y=datast.target
df=pd.DataFrame(datast.data)
X_Train,X_Test,Y_Train,Y_Test=train_test_split(X,Y,test_size=0.3)

model=LinearRegression()
model.fit(X_Train,Y_Train)
Y_Predict=model.predict(X_Test)
accuracy=model.score(X_Test,Y_Test)
Y_Predict1=np.round(Y_Predict)
print "This is through the roudning off of result"
print(metrics.classification_report(Y_Test,Y_Predict1))


print "This is through the cieling off of result"
Y_Predict2=np.ceil(Y_Predict)
print(metrics.classification_report(Y_Test,Y_Predict2))


model=KNeighborsClassifier(n_neighbors=7,weights='distance')
model.fit(X_Train,Y_Train)
Ypredict=model.predict(X_Test)
print "The Accuracy through KNN is:- "
val=metrics.accuracy_score(Y_Test,Ypredict)
print(val)










