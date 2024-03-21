import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import util

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

vals = []
for i in range(4, 10):
  model.fit(x_train, y_train, epochs=i, batch_size=32, validation_split=0.2)
  y_pred = model.predict(x_test)
  vals.append([i, y_pred])

# Evaluate the model on the testing dataset
# loss, accuracy = model.evaluate(x_test, y_test)

# TRAIN STATS
# y_pred = model.predict(x_train)

# TEST STATS
y_pred = model.predict(x_test)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
for i in range(len(vals)):
    fpr, tpr, thresholds = roc_curve(y_test, vals[i][1])
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'Epochs = {vals[i][0]} (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
