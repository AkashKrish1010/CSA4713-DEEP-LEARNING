import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

# Synthetic linear data
X = np.random.rand(200, 1)
y = 3 * X + 2 + np.random.randn(200, 1) * 0.1

# Optimizers to compare: Gradient Descent (Batch), SGD, Momentum, and Adam
optimizers = {
    'Batch GD': (keras.optimizers.SGD(0.01), len(X)),
    'SGD': (keras.optimizers.SGD(0.01), 1),
    'Momentum': (keras.optimizers.SGD(0.01, momentum=0.9), 16),
    'Adam': (keras.optimizers.Adam(0.01), 16)
}

plt.figure(figsize=(8, 5))
for name, (opt, b_size) in optimizers.items():
    model = keras.Sequential([keras.layers.Input(shape=(1,)), keras.layers.Dense(1)])
    model.compile(optimizer=opt, loss='mse')
    history = model.fit(X, y, epochs=30, batch_size=b_size, verbose=0)
    plt.plot(history.history['loss'], label=name)

plt.title("Optimizer Comparison: GD, SGD, Momentum, Adam")
plt.xlabel("Epochs")
plt.ylabel("Loss (MSE)")
plt.legend()
plt.show()
