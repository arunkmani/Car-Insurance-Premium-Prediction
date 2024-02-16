import pandas as pd 
import tensorflow as tf
import os 
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

#from keras.layers.convolutional import Conv1D
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
print(tf.__version__)
data1= pd.read_csv('data1.csv')
data2= pd.read_csv('data2.csv')
data1['CLAIM_PAID']=data1['CLAIM_PAID'].fillna(0)
data2['CLAIM_PAID']=data2['CLAIM_PAID'].fillna(0)
data1['INSR_BEGIN']=pd.to_datetime(data1['INSR_BEGIN'])
data1['INSR_END']=pd.to_datetime(data1['INSR_END'])
data2['INSR_BEGIN']=pd.to_datetime(data2['INSR_BEGIN'])
data2['INSR_END']=pd.to_datetime(data2['INSR_END'])
data1 = pd.concat([data1,data2],ignore_index='True')
data1['INSR_BEGIN'] = (data1['INSR_BEGIN'] - pd.to_datetime('2010-01-01')).dt.days
data1['INSR_END'] = (data1['INSR_END'] - pd.to_datetime('2010-01-01')).dt.days
data2['INSR_BEGIN'] = (data2['INSR_BEGIN'] - pd.to_datetime('2010-01-01')).dt.days
data2['INSR_END'] = (data2['INSR_END'] - pd.to_datetime('2010-01-01')).dt.days
#data1= data1[data1['EFFECTIVE_YR'].notna()]
#data1= data1[data1['EFFECTIVE_YR'].notna()]
data1=data1.dropna(how='any')
data1.reset_index(drop=True, inplace=True)
data1 = 
data1=data1.head(int(len(data1) * 0.2))
tt=data1['TYPE_VEHICLE'][1200]
data1= data1[data1.EFFECTIVE_YR.str.isnumeric()]
vehicles=data1['TYPE_VEHICLE'].unique().tolist()
# vehicle_index=[]
# for i in range(len(data1['TYPE_VEHICLE'])):
#     vehicle_index.append( vehicles.index(data1['TYPE_VEHICLE'][i]))
# data1['TYPE_VEHICLE'] = vehicle_index
# print('Vehicle done')
# vehicles=data1['MAKE'].unique().tolist()
# vehicle_index=[]
# for i in range(len(data1['MAKE'])):
#     vehicle_index.append( vehicles.index(data1['MAKE'][i]))
# data1['MAKE'] = vehicle_index
# print('Make DOne')

# vehicles=data1['USAGE'].unique().tolist()
# vehicle_index=[]
# for i in range(len(data1['USAGE'])):
#     vehicle_index.append( vehicles.index(data1['USAGE'][i]))
# data1['USAGE'] = vehicle_index
i=0
for value in vehicles:
    data1['TYPE_VEHICLE'].replace(value,i,inplace=True)
    i=i+1
    print(value,' done')
vehicles=data1['MAKE'].unique().tolist()
i=0
for value in vehicles:
    data1['MAKE'].replace(value,i,inplace=True)
    i=i+1
vehicles=data1['USAGE'].unique().tolist()
for value in vehicles:
    data1['USAGE'].replace(value,i,inplace=True)
    i=i+1
    print(value,' done')
y= data1['PREMIUM']
data1=data1.drop('PREMIUM',axis=1)

X_train,X_test,y_train,y_test=train_test_split(data1,y,test_size=.3,random_state=42)
y_train=y_train.astype('int')
y_test=y_test.astype('int')
scaler=StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.fit_transform(X_test)
verbose, epochs, batch_size = 1, 10, 32
print(X_train.shape[0], X_train.shape[1], y_train.shape[0])
data1.to_csv('fuckthis.csv')
model = Sequential()
model.add(Dense(512, activation='relu', input_shape=(15,)))
model.add(Dropout(0.2))
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.2))
model.add(Dense(10, activation='relu'))
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
#model.fit(X_train, y_train, epochs=10, batch_size=32)
mlp = MLPClassifier(hidden_layer_sizes=(348,), max_iter=15, alpha=1e-4,solver='sgd', verbose=10, random_state=1,learning_rate_init=0.1)
mlp.fit(X_train, y_train)
y_pred = mlp.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)

# fit network
print('MSE:',mse)
print('MAE:',mae)




