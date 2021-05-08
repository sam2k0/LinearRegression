import sys
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("Height_weight.csv")
df=df.dropna()
x1=df["Height"].values
x=np.vstack((np.ones(len(x1)),x1)).T
y=df["Weight"].values.reshape(15,1)
def model(x,y,lr=0.25):
    m = len(y)
    min=sys.maxsize
    theta = np.zeros((2, 1))
    ypred = np.dot(x, theta)
    dtheta = np.dot(x.T, ypred - y) / m
    prev=dtheta
    theta = theta - lr * dtheta
    count=0
    while(True):
        count=count+1
        ypred=np.dot(x,theta)
        dtheta=np.dot(x.T,ypred-y)/m
        if prev[1]*dtheta[1]<0:
            break
        else:
            prev=dtheta
            theta = theta - lr * dtheta
    return theta

theta=model(x,y)
ypred=np.dot(x,theta)
print(ypred)
print(r2_score(y,ypred)*100)
plt.scatter(x1,y)
plt.plot(x1,ypred)
plt.show()