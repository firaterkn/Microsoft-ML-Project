# Microsoft (MSFT) Stock Price Prediction using LSTM and GRU Models

# Overview 
This project aims to predict the closing prices of Microsoft (MSFT) stock using Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) models, which are types of recurrent neural networks (RNNs) widely used for time-series forecasting. The predictions are made using historical stock price data, and the models' performances are evaluated based on their accuracy and the ability to capture trends in stock prices.


# The Questions to be Answered

1. Can we accurately predict the closing prices of Microsoft stock using historical data?
2. How does the performance of the LSTM model compare with the GRU model?
3. What are the key insights that can be drawn from the prediction results?


# Tools Used

- **Python:** The primary programming language used for the analysis.
- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical computations.
- **Scikit-learn:** For data preprocessing (e.g., scaling).
- **Keras:** For building and training the deep learning models (LSTM and GRU).
- **Matplotlib:** For data visualization.
- **Pandas Datareader:** To retrieve stock data from the 'stooq' source.

# Data Preparation

The dataset used for this project consists of the historical closing prices of Microsoft stock from January 1, 2012, to May 1, 2024. Initially, the dataset was retrieved in reverse chronological order (most recent to oldest) but was subsequently reversed to follow a chronological sequence (oldest to most recent).

``` python

# Load the dataset
df = web.DataReader('MSFT', data_source='stooq', start='2012-01-01', end='2024-05-01')
df = df.iloc[::-1]  
data= df.filter(["Close"])

```

The data was split into a training set (80%) and a testing set (20%). Before feeding the data into the models, the closing prices were normalized using MinMaxScaler to scale the values between 0 and 1. A sliding window approach was employed, where the model was trained on sequences of 60 days of stock prices to predict the price on the 61st day.

``` python

dataset= data.values 
training_data_length= math.ceil(len(dataset) * .8) 

# Scale data
scaler= MinMaxScaler(feature_range=(0,1)) 
scaled_data= scaler.fit_transform(dataset) 

# Prepare training and testing data sets
train_data= scaled_data[0:training_data_length , :] 

x_train= []
y_train= []

# Create the training dataset
for i in range(60, len(train_data)):
  x_train.append(train_data[i-60:i, 0]) 
  y_train.append(train_data[i, 0])

```


# Model Architecture

 ## LSTM Model
 - **Layers**
   
    - 2 LSTM layers (50 units each)
    - 2 Dense layers (25 units and 1 unit)
      
  - **Loss Function:** Mean Squared Error (MSE)
    
  - **Optimizer:** Adam

 ## GRU Model

 - **Layers**
   
    - 2 LSTM layers (50 units each)
    - 2 Dense layers (25 units and 1 unit)
      
  - **Loss Function:** Mean Squared Error (MSE)
    
  - **Optimizer:** Adam

# Results 

![LSTM]()
   





