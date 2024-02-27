import pandas as pd
import tensorflow as tf
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from ensemble_util import ensemble_neural_network

df_train = pd.read_csv("MS&E 246 Data Updated 3/df_train.csv")
df_test = pd.read_csv("MS&E 246 Data Updated 3/df_test.csv")

df_train['LoanStatus'] = df_train['LoanStatus'].map({'PIF': 0, 'CHGOFF': 1})
df_test['LoanStatus'] = df_test['LoanStatus'].map({'PIF': 0, 'CHGOFF': 1})

X_train_mess = df_train.drop(columns=['LoanStatus'])
y_train = df_train['LoanStatus']
X_test_mess = df_test.drop(columns=['LoanStatus'])
y_test = df_test['LoanStatus']

columns_to_include = ['ThirdPartyDollars']

"""['ThirdPartyDollars', 'GrossApproval', 'Same State' , 'In CA', 'Missing Interest', 'INDIVIDUAL', 'Unemployment YR', 'Avg Home Price', 'GDP Delta YR', 'Log S&P Open', 'Missing Borr GDP']"""

"""ThirdPartyDollars,GrossApproval,ApprovalDate,ApprovalFiscalYear,DeliveryMethod,subpgmdesc,TermInMonths,NaicsCode,ProjectState,BusinessType,LoanStatus,ChargeOffDate,GrossChargeOffAmount,Term Multiple,Same State,In CA,Is ThirdParty,Missing Interest,Refinance,Delta,Private Sector,Premier,CORPORATION,INDIVIDUAL,MISSING,PARTNERSHIP,SP500 YR,Unemployment YR,Avg Home Price,GDP Delta YR,Log S&P Open,BorrState Unemployment,ProjectState Unemployment,BorrState Income,ProjState Income,Missing Borr Income,Missing Proj Income,BorrState GDP,ProjState GDP,Missing Borr GDP,Missing Proj GDP,BorrState Vacancy,ProjectState Vacancy"""

X_train = X_train_mess[columns_to_include].copy()
X_test = X_test_mess[columns_to_include].copy()

X_train["Intercept"] = 1
X_test["Intercept"] = 1

print(X_train)

# Define your neural network architecture using TensorFlow/Keras
model = tf.keras.Sequential([
    tf.keras.layers.Dense(units=64, activation='relu', input_shape=(X_train.shape[1],)),
    tf.keras.layers.Dense(units=64, activation='relu'),
    tf.keras.layers.Dense(units=1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=100, validation_split=0.2)

# Evaluate the model on the testing dataset
loss, accuracy = model.evaluate(X_test, y_test)

# TRAIN STATS
y_pred = model.predict(X_train)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_train, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# TEST STATS
y_pred = model.predict(X_test)

# Calculate ROC curve and AUC
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()
