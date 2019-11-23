from sklearn.externals import joblib
import CreditCardFraudDetetionModel as model
from sklearn.model_selection import train_test_split
import pandas as pd


def main():
    data=model.PrepareData()
    x=data.iloc[:, :-1]
    y=data["Class"]
    XTrain,XTest,YTrain,YTest=train_test_split(x,y,test_size=0.3,random_state=1)
    filename = 'finalized_model.sav'
    loaded_model = joblib.load(filename)
    result = loaded_model.score(XTest, YTest)
    print(result)

if __name__== "__main__":
    main()


