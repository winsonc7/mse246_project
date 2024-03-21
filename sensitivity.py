import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import util
import numpy as np

x_train, y_train = util.load_csv("MS&E 246 Data Updated 3/df_train_norm_full.csv")
x_test, y_test = util.load_csv("MS&E 246 Data Updated 3/df_test_norm_full.csv")

# Define your neural network architecture using TensorFlow/Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=(x_train.shape[1],)),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model

model.fit(x_train, y_train, epochs=12, batch_size=32, validation_split=0.2)

eps = 0.01
base_prediction = model.predict(x_test)
sensitivities = []
for i in range(x_test.shape[1]):
  copy = np.copy(x_test)
  copy[:, i] += eps
  pred = model.predict(copy)
  sensitivity = np.mean((pred - base_prediction) / eps)
  sensitivities.append(sensitivity)
print(sensitivities)