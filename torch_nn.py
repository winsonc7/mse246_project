import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import util

x_train, y_train = util.load_csv("Data/train_updated_norm_full.csv")
x_test, y_test = util.load_csv("Data/test_updated_norm_full.csv")


# Convert data to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Define your neural network architecture using PyTorch
class NeuralNetwork(nn.Module):
    def __init__(self, input_size):
        super(NeuralNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.sigmoid(self.fc3(x))
        return x

# Initialize the model
model = NeuralNetwork(X_train.shape[1])

# Define loss function and optimizer
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters())

# Train the model
num_epochs = 10
batch_size = 32
for epoch in range(num_epochs):
    for i in range(0, len(X_train), batch_size):
        inputs = X_train[i:i+batch_size]
        labels = y_train[i:i+batch_size]
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels.unsqueeze(1))
        loss.backward()
        optimizer.step()

# Evaluate the model
with torch.no_grad():
    outputs = model(X_test)
    loss = criterion(outputs, y_test.unsqueeze(1))
    accuracy = ((outputs >= 0.5).type(torch.float32) == y_test.unsqueeze(1)).float().mean()

print(f'Test Loss: {loss.item():.4f}, Test Accuracy: {accuracy.item():.4f}')

# Calculate ROC curve and AUC for training dataset
with torch.no_grad():
    outputs = model(X_train)
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, outputs.numpy())
    roc_auc_train = auc(fpr_train, tpr_train)

# Plot ROC curve for training dataset
plt.figure()
plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_train)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Training)')
plt.legend(loc="lower right")
plt.show()

# Calculate ROC curve and AUC for testing dataset
with torch.no_grad():
    outputs = model(X_test)
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, outputs.numpy())
    roc_auc_test = auc(fpr_test, tpr_test)

# Plot ROC curve for testing dataset
plt.figure()
plt.plot(fpr_test, tpr_test, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc_test)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (Testing)')
plt.legend(loc="lower right")
plt.show()
