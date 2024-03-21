import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from ensemble_util import ensemble_neural_network
import util

x_train, y_train = util.load_csv("MS&E 246 Data Updated 3/df_train_norm_min.csv")
x_test, y_test = util.load_csv("MS&E 246 Data Updated 3/df_test_norm_min.csv")

vals = []
for i in range(1, 14, 2):
  ensemble_predictor = ensemble_neural_network(x_train, y_train, B=i)
  y_pred = ensemble_predictor(x_test)
  vals.append([i, y_pred])

# TRAIN STATS
# y_pred = ensemble_predictor(x_train)

# Calculate ROC curve and AUC
# fpr, tpr, thresholds = roc_curve(y_train, y_pred)
# roc_auc = auc(fpr, tpr)

# Plot ROC curve
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Receiver Operating Characteristic')
# plt.legend(loc="lower right")
# plt.show()

# TEST STATS
y_pred = ensemble_predictor(x_test)

# Calculate ROC curve and AUC
# fpr, tpr, thresholds = roc_curve(y_test, y_pred)
# roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
for i in range(len(vals)):
  fpr, tpr, thresholds = roc_curve(y_test, vals[i][1])
  roc_auc = auc(fpr, tpr)
  plt.plot(fpr, tpr, lw=2, label=f'B={vals[i][0]} (area = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
