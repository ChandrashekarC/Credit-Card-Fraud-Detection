import pandas as pd
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.externals import joblib

#preparing data 
data=PrepareData()
X=data.iloc[:, :-1]
Y=data["Class"]
XTrain,XTest,YTrain,YTest=train_test_split(X,Y,test_size=0.3,random_state=1)
model=LogisticRegression()
model.fit(XTrain,YTrain)
predictedValue=model.predict(XTest)
print("The Confusion Matrix:\n", confusion_matrix(YTest,predictedValue))
print("accuracy_score:",accuracy_score(YTest,predictedValue))
filename = 'CreditCardFraudDetectionModel.sav'
joblib.dump(model, filename)


#the data has -ve values in it so i have added the least value in the colm to each row in the colm
def PrepareData():
    minval=[]
    fileDir = os.path.dirname(os.path.realpath('__file__'))
    data = pd.read_csv(os.path.join(fileDir, "Dataset\creditcard.csv"))
    for i in data:
        minval.append(data[i].min())    
    for i,j in zip(data,minval):
        data[i]-=j
    return data   
