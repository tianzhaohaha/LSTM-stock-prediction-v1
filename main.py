import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pandas_datareader import data
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,LSTM,Dropout

from tensorflow.keras.optimizers import SGD

dataset = data.DataReader('BABA', start='2015', end='2020',
data_source='yahoo')

train_set = dataset[:'2018'].iloc[:,[0]].values#获取训练集选取第一列的数据作为特征
test_set = dataset['2019':].iloc[:,[0]].values#

def plot_pridicetions(test_result,pridict_result):
    """
    test_result:测试真是值
    pridict_result：预测值
    """
    plt.plot(test_result,color='red',label='ALIBABA True Stock Price')
    plt.plot(pridict_result,color='blue',label='ALIBABA Pridicted Stock Price')
    plt.title('ALIBABA Stock Price')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()


#dataset['High'][:'2018'].plot(title="ALIBABA Stock Price",figsize=(16,4),legend="train set before 2019")
#dataset['High']['2019':].plot(figsize=(16,4),legend="test set after 2019")
#plt.legend(["train set before 2019",
        #   "test set after 2019"])

sc = MinMaxScaler(feature_range=[0,1])
train_set_scaled = sc.fit_transform(train_set)


#创建训练数据集（训练和测试）
#30个数据集为一个样本，一个输出
#这里的意思是使用前0-29个作为输入，第30个作为标签，即为需要预测的值
X_train=[]
Y_train=[]

for i in range(60,1007):
    X_train.append(train_set_scaled[i-60:i,0])#这里选的是0-29的样本,30个样本
    Y_train.append(train_set_scaled[i,0])
X_train,Y_train=np.array(X_train),np.array(Y_train)#numpy类型转换


#LSTM 输入：（samples, sequence, features)
#reshape:训练集 （977，30）--->（977，30，1）
X_train=np.reshape(X_train,(X_train.shape[0],X_train.shape[1],1))



model = Sequential()

#LSTM第一层
model.add(LSTM(200,return_sequences = True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))

#LSTM第二层
model.add(LSTM(200,return_sequences = True, input_shape=(X_train.shape[1],1)))
model.add(Dropout(0.2))

#第三层
model.add(LSTM(200))
model.add(Dropout(0.2))

#Dense层
model.add(Dense(units=1))


#模型编译
model.compile(optimizer='rmsprop',loss='mse')

#模型训练
model.fit(X_train,Y_train,epochs=20,batch_size=32)

dataset_total=pd.concat((dataset['High'][:'2019'],dataset['High']['2019':]),axis=0)
#构建输入数据
inputs = dataset_total[len(dataset_total)-len(test_set)-60:].values
dataset_total[len(dataset_total)-len(test_set)-60:]

inputs = inputs.reshape(-1,1)
inputs = sc.transform(inputs)

# 准备测试集X_test进行预测
X_test = []
for i in range(60, 312):
    X_test.append(inputs[i - 60:i, 0])

X_test = np.array(X_test)  # numpy类型转换

X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
predict_test = model.predict(X_test)#模型预测,这里的预测值也是经过归一化的

predict_stock_price = sc.inverse_transform(predict_test)

#会值测试结果和真实结果
plot_pridicetions(test_set,predict_stock_price)