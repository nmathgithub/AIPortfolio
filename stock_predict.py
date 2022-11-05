import pandas as pd 
import numpy as np
import yfinance as yf 
import datetime 
from datetime import date, timedelta 
import plotly.graph_objects as go 
from sklearn.model_selection import train_test_split 
from keras.models import Sequential 
from keras.layers import Dense, LSTM

today = date.today()

d1 = today.strftime("%Y-%m-%d")
end_date = d1 
d2 = date.today() - timedelta(days = 5000) 
d2 = d2.strftime("%Y-%m-%d")
start_date = d2 

# Test on Apple Stock
# Step I: Collect Data and store it in Dataframe 
data = yf.download('AAPL', 
                    start = start_date,
                    end = end_date, 
                    progress=False)
data["Date"] = data.index


data = data[["Date", "Open", "High",
            "Close", "Adj Close", "Volume"]]
data.reset_index(drop=True, inplace=True)
data.tail()
# print(data)


""" # Step II: Visualize
figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                        open=data["Open"], 
                                        high=data["High"],
                                        low=data["Low"], 
                                        close=data["Close"])])
figure.update_layout(title = "Apple Stock Price Analysis", 
                     xaxis_rangeslider_visible=False)
figure.show() """

# Step III: Correlation 
correlation = data.corr()
print(correlation["Close"].sort_values(ascending=False))

# Step IV: Now we Train the LSTM model 
x = data[["Open", "High", "Volume"]]
y = data["Close"]
x = x.to_numpy()
y = y.to_numpy()
y = y.reshape(-1,1)
xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size = 0.2, random_state = 42)

# Step 5: NN Architecture 
model = Sequential()
model.add(LSTM(128, return_sequences=True, input_shape=(xtrain.shape[1], 1)))
model.add(LSTM(64, return_sequences =False))
model.add(Dense(25))
model.add(Dense(1))
model.summary()

# Step 6: Now we train our neural network model
model.compile(optimizer='adam',          
                loss='mean_squared_error')
model.fit(xtrain, ytrain, batch_size=1, epochs=30)

# Step 7: Test 
features = np.array([[177.089996, 180.419998, 174919600]])
print(model.predict(features))
# [[181.37862]]

# 11/5/2022 
# Following a Tutorial 
# https://thecleverprogrammer.com/2022/01/03/stock-price-prediction-with-lstm/
# Goal: Stokc Price Prediction with LSTM 

#LSTM - Long Short Term Memory Networks 
#RNN - Use for regression & time series forecasting 
