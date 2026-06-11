import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import SimpleRNN
from tensorflow.keras.layers import LSTM

#load dataset 
df = pd.read_csv(
    r"E:/Project Y/AI_Spacecraft/CMaps/train_FD001.txt" , sep=" ", header=None)
df.dropna(axis=1,inplace=True)
print(df.columns)

columns = [

    "engine_id",
    "cycle",

    "op_setting_1",
    "op_setting_2",
    "op_setting_3"

]
for i in range(1,22):
    columns.append(

        f"sensor_{i}"
    )
df.columns = columns

print(df.head())
print(df.columns)

max_cycle = df.groupby(
    "engine_id"
)["cycle"].max()

print(max_cycle)

df = df.merge(
    max_cycle.to_frame(name='max_cycle'),
    on='engine_id'
)

print(df.head())


df["RUL"] = df["max_cycle"]-df["cycle"]

print(
    df[["engine_id","cycle","max_cycle","RUL"]].head()
)

X = df.drop(
   ["engine_id","cycle","max_cycle","RUL"] ,axis =1
)

Y  = df["RUL"]

from sklearn.preprocessing import StandardScaler

scaler = MinMaxScaler()
scaled_features = scaler.fit_transform(X)

sequence = 30
x =[]
y = []

for i in range(sequence,len(scaled_features)):
    x.append(scaled_features[i-sequence:i])
    y.append(Y.iloc[i])
x = np.array(x)
y = np.array(y)
print(X.shape)

print(y.shape)

split = int(
    len(x) * 0.8
)
x_train = x[:split]
x_test = x[split:]
y_train = y[:split]
y_test = y[split:]

print("\n Train Shape")
print(x_train.shape)
print("Test shape")
print(x_test.shape)

model = Sequential([
    SimpleRNN(128,return_sequences = True,input_shape=(x.shape[1],x.shape[2])),
    Dropout(0.2),
    LSTM(64),
    Dropout(0.2),
    Dense(32,activation='relu'),
    Dense(1)
])
model.summary()
model.compile(optimizer = 'Adam',loss = 'mse',metrics =['mae'])

history = model.fit(
    x_train,y_train,epochs = 20,batch_size=64,validation_split=0.2
)

loss,mae = model.evaluate(
    x_test,y_test

)

predictions = model.predict(

    x_test

)
print(predictions)

plt.figure(figsize=(12,6))
plt.plot(
    y_test[:100],
    label = 'actual RUL'
)
plt.plot(
    predictions[:100],
    label='predicted RUL'
)
plt.title("Spacecratf Failure Prediction")
plt.xlabel("Time")
plt.ylabel("Reamining UseFul Life")
plt.legend()
plt.show()

plt.figure(figsize=(10,6))
plt.plot(
    history.history['loss'],
    label = 'Training Loss'
)
plt.title(
    "RNN + LSTM Trainig Loss"
)
plt.xlabel("Epochs")
plt.ylabel("loss")
plt.legend()
plt.show()

model.save(
    "Space Craft _RNN _Lstm.h5")
print("Model Saved Succesfylluy")