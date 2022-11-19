import pandas as pd
from sklearn import linear_model
import pickle


def train_model():
    df = pd.read_csv('app/prices.csv')

    y = df['Value'] 
    X = df[['Rooms', 'Distance']]

    lm = linear_model.LinearRegression()
    lm.fit(X, y)

    pickle.dump(lm, open('model.pkl','wb'))
    print("Model has been trained and saved!")

train_model()