import sys
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
df=pd.read_csv("LinearRegression\Height_weight.csv")
df=df.dropna()
x=df[["Height"]].values
y=df["Weight"].values
plt.show()
print(x.shape,y.shape)
def train_model(x,y,lr=0.25):
    n=len(x)
    x_mean=sum(x)/n
    y_mean=sum(y)/n
    xy_sum=0
    x_square_sum=0
    for k in range(n):
        xy_sum=xy_sum+(x[k]*y[k])
        x_square_sum=x_square_sum+(x[k]*x[k])
    m=(xy_sum-n*x_mean*y_mean)/(x_square_sum-n*x_mean*x_mean)
    c=y_mean-(m*x_mean)
    return (m,c)

m,c=train_model(x,y)
ypred=[]
for i in range(len(x)):
    yp=m*x+c
    ypred.append(yp)
print(r2_score(y,ypred[0])*100)
plt.scatter(x,y)
plt.plot(x,ypred[0])
plt.show()