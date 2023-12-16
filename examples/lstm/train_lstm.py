import sys
sys.path.append('../../')

import pandas as pd
import torch
from models import lstm


df = pd.read_csv("airline-passengers.csv")
timeseries = df[["Passengers"]].values.astype('float32')

train_size = int(len(timeseries)*0.67)
test_size = len(timeseries) - train_size
train, test = timeseries[:train_size], timeseries[train_size:]

def create_dataset(dataset, lookback):
    X,y = [], []
    for i in range(len(dataset)-lookback):
        feature = dataset[i:i+lookback]
        target = dataset[i+1:i+lookback+1]
        X.append(feature)
        y.append(target)
    return torch.tensor(X), torch.tensor(y)


lookback = 4
X_train, y_train = create_dataset(train, lookback=lookback)
X_test, y_test = create_dataset(test, lookback=lookback)
print(X_train.shape, y_train.shape)
print(X_test.shape, y_test.shape)

train_loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(X_train, y_train), shuffle=True, batch_size=8)

input_size = 1
output_size = 1
model = lstm.LSTM(input_size, output_size)


learning_rate = 0.001
epochs = 2000
model.train_model(train_loader,learning_rate, epochs)

model.save_model('lstm_passengers.pth')


