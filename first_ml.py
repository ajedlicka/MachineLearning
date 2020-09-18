import numpy as np
from tensorflow import keras

# setting up neural network
model = keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
model.compile(optimizer='sgd', loss='mean_squared_error')

# setting up the initial databases
xs = np.array([-1.0, 0.0, 1.0, 2.0, 3.0, 4.0], dtype=float)
ys = np.array([-3.0, -1.0, 1.0, 3.0, 5.0, 7.0], dtype=float)

# running 500 epochs
model.fit(xs, ys, epochs=500)

# printing results
print(model.predict([10.0]))
