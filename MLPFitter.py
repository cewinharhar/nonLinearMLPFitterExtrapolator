import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.metrics import MeanAbsoluteError
from keras.optimizers import *
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import numpy as np

optimizer = keras.optimizers.Adam()



#Red data from csv file for training and validation data
nlFun = lambda x: (1/(0.3 * x**0.2) +5 )

x = np.random.randint(low=0, high=1500, size=(10000,))+1
y= nlFun(x)  + np.random.normal(scale=0.02, size=(len(x)))

plt.scatter(x=x, y=y, s=0.1); plt.show()

# create model
model = Sequential()
model.add(Dense(units = 1, activation = 'linear', input_shape=[1]))
model.add(Dense(units = 32, activation = 'tanh'))
model.add(Dense(units = 32, activation = 'tanh'))
model.add(Dense(units = 1, activation = 'linear'))

callback = keras.callbacks.EarlyStopping(monitor='loss', patience=30)

# Compile model
model.compile(loss='mse', optimizer="adam")

# Fit the model and define history output
log = model.fit(x, y, epochs=25, validation_split=0.1, batch_size=32, callbacks=[callback])

plt.plot(log.history["loss"])
plt.plot(log.history["val_loss"]); plt.show()


# Calculate predictions
xpred = np.arange(3000)+1
PredTestSet = model.predict(xpred)

plt.scatter(x=x, y=y, s=0.1, c="r")
plt.scatter(x=xpred, y=PredTestSet, s=0.1, c="g");plt.show()

plt.scatter(x=np.arange(1000)+1, y=nlFun(np.arange(1000)+1), s=0.1)