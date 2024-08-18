# Microsoft (MSFT) Stock Price Prediction using LSTM and GRU Models

<br>

# Overview 
This project aims to predict the closing prices of Microsoft (MSFT) stock using Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) models, which are types of recurrent neural networks (RNNs) widely used for time-series forecasting. The predictions are made using historical stock price data, and the models' performances are evaluated based on their accuracy and the ability to capture trends in stock prices.

<br>

# The Questions to be Answered

1. Can we accurately predict the closing prices of Microsoft stock using historical data?
2. How does the performance of the LSTM model compare with the GRU model?
3. What are the key insights that can be drawn from the prediction results?

<br>

# Tools Used

- **Python:** The primary programming language used for the analysis.
- **Pandas:** For data manipulation and analysis.
- **NumPy:** For numerical computations.
- **Scikit-learn:** For data preprocessing (e.g., scaling).
- **Keras:** For building and training the deep learning models (LSTM and GRU).
- **Matplotlib:** For data visualization.
- **Pandas Datareader:** To retrieve stock data from the 'stooq' source.

<br>

# Data Preparation

The dataset used for this project consists of the historical closing prices of Microsoft stock from January 1, 2012, to May 1, 2024. Initially, the dataset was retrieved in reverse chronological order (most recent to oldest) but was subsequently reversed to follow a chronological sequence (oldest to most recent).

``` python

# Load the dataset
df = web.DataReader('MSFT', data_source='stooq', start='2012-01-01', end='2024-05-01')
df = df.iloc[::-1]  
data= df.filter(["Close"])

```

<br>

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

<br>

# Model Architecture

 ## LSTM Model
 - **Layers**
   
    - 2 LSTM layers (50 units each)
    - 2 Dense layers (25 units and 1 unit)
      
  - **Loss Function:** Root Mean Squared Error (RMSE)
    
  - **Optimizer:** Adam

 ## GRU Model

 - **Layers**
   
    - 2 LSTM layers (50 units each)
    - 2 Dense layers (25 units and 1 unit)
      
  - **Loss Function:** Root Mean Squared Error (RMSE)
    
  - **Optimizer:** Adam

<br>

# Results 

The models were evaluated on the test set, with predictions visualized against actual closing prices. Below are the results:

![LSTM](https://github.com/firaterkn/Microsoft-ML-Project/blob/main/LSTM.PNG)

The LSTM model was able to capture the general trend of the stock prices with reasonable accuracy. However, there were certain points where the model's predictions deviated from actual values.

**RMSE (Root Mean Squared Error):** 0.0019
<br>
**Completion Time:** 95 seconds

<br>
<br>

![GRU](https://github.com/firaterkn/Microsoft-ML-Project/blob/main/GRU.PNG)

The GRU model showed better performance compared to the LSTM model, especially in terms of closely following the validation set's trend.

**RMSE (Root Mean Squared Error):** 0.0013
<br>
**Completion Time:** 61 seconds

<br>

# Insights About the Result

**Prediction Accuracy:** Both models demonstrate strong performance, but the GRU model showed better alignment with the actual stock price movements, suggesting it may be more effective for this type of time-series data.

**Stock Market Volatility:** The models captured the general upward trend in Microsoft's stock price but the LSTM model struggled with the high volatility seen during certain periods.

<br>

# Challenges Faced

**Data Reversal:** The initial data order (newest to oldest) was reversed, which required correction before training the models.

**Overfitting:** Training the model for a long time may result in overfitting, where the model fits the training data very well but does not perform well on the test data. How to fix?
- **Low Number of Epochs:** It reduces the risk of overfitting because the model was stopped before it overfits the training data.
- **Validation Set Usage:** The performance of the model was also tested on the validation set to see if the model was overfitting.
- **Simple Model Structure:** The structure of the model used was aimed at reducing the risk of overfitting. The model contains only two LSTM/GRU layers and two Dense layers
  <br>
  
**Model Training Time:** Due to the complexity of the models, especially LSTM, the training process was time-consuming.

**Volatility:** The high volatility in stock prices made it difficult for the LSTM model to predict sudden spikes or drops accurately.

<br>

# Conclusion

This project successfully implemented LSTM and GRU models to predict Microsoft's stock prices. The GRU model outperformed the LSTM model, indicating that it might be more suitable for this type of financial time-series data. However, both models face challenges in predicting highly volatile periods. Future work could explore more complex architectures, longer training periods, or the inclusion of additional features (e.g., trading volume, technical indicators) to improve prediction accuracy.
