import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas import datetime
import math, time
import itertools
from sklearn import preprocessing
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from math import sqrt
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.layers.recurrent import LSTM
from keras.layers.advanced_activations import LeakyReLU, PReLU
import csv
file_name='LT.NS.csv'
col_name=['Open','High','Low','Close']
stocks=pd.read_csv(file_name,header=0,names=col_name)
df=pd.DataFrame(stocks)
df.to_csv('inter.csv')
df['High'] = df['High']/1000
df['Low'] =df['Low']/1000
df['Open'] =df['Open']/1000
df['Close'] =df['Close']/1000
def load_data(stock,seq_len):
	amount_of_features = len(stock.columns)
	data = stock.as_matrix() #pd.DataFrame(stock)
	sequence_length = seq_len + 1
	result = []
	for index in range(len(data) - sequence_length):
		result.append(data[index: index + sequence_length])
	result=np.array(result)
	print("result"+ str(result.shape))
	row= 0.95*result.shape[0]
	train = result[:int(row), :]
	x_train = train[:, :-1]
	y_train = train[:, -1][:,-1]
	x_test = result[int(row):, :-1]
	y_test = result[int(row):, -1][:,-1]
	x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
	x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))
	#print(x_train)
	#print(x_test)
	print("y_test"+str(y_test.shape))
	return [x_train,x_test,y_train, y_test]
x_train,x_test,y_train,y_test=load_data(df[::-1],5)
print('y_test'+str(y_test.shape))
print('y_train'+str(y_train.shape))
print('x_test'+str(x_test.shape))
print('x_train'+str(x_train.shape))

def build():
	d=0.2
	model=Sequential()
	model.add(LSTM(256,input_shape=(5,4),return_sequences=True))
	model.add(Dropout(0.3))
	model.add(LSTM(128,input_shape=(5,4),return_sequences=True))
	model.add(Dropout(d))
	model.add(LSTM(64,input_shape=(5,4),return_sequences=False))
	model.add(Dropout(d))
	model.add(Dense(16,kernel_initializer='uniform',activation='relu'))
	model.add(Dense(1,kernel_initializer='uniform',activation='relu'))
	model.compile(loss='mse',optimizer='adam',metrics=['accuracy'])
	return model

model=build()
model.fit(x_train,y_train,batch_size=32,epochs=50,validation_split=0.1,verbose=2)
p=model.predict(x_test)
import matplotlib.pyplot as plt2
plt2.plot(p,color='red', label='prediction')
plt2.plot(y_test,color='blue', label='y_test')
plt2.legend(loc='upper left')
plt2.show()

	

















